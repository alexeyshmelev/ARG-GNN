"""H031 CPU-only fixed H028 observed-variant information diagnostic.

Author/source reporting opens no payloads. Executing requires explicit active
root diagnostic admission and a <=300MB exclusive reservation. Observation
predictions for ALL four parents are sealed before any target/weight/previous
prediction payload is opened. No fit, new windows, calibration or floor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time

from . import census_tsinfer_fit_v1 as fit

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/observed_variant_signal_diagnostic_v1.py"
PROTOCOL = "docs/observed_variant_signal_diagnostic_v1.md"
OUTPUT_CAP = 300000000
H028_MODULE = {"path": "src/arggnn/observed_variant_residual_v1.py",
 "sha256": "695a78d266d8dc4cf226f6e8ff54669979eaf94f30e5cc8cb3062f14ec02c904"}
H028_CONFIG = {"path": "configs/observed_variant_residual_v1.json",
 "sha256": "7af23c1612d8acbf8e7a21cfdcc8e87fbd352e5821073e7caf180da82e861f91"}


def canonical_ref_v1(reader, record):
    return {"path": reader.relative(record["path"]), "sha256": record["sha256"]}


def inputs_v1(record):
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "observed_variant_signal_diagnostic_input_v1"
            or spec.get("status") != "active" or spec.get("root_authorizer") != "root_agent"
            or spec.get("diagnostic_authorized") is not True or spec.get("payload_reads_authorized") is not True
            or spec.get("device") != "cpu" or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Active reviewed root H031 CPU diagnostic/payload admission required")
    fit._check_active_v1(spec)
    if spec.get("source_sha256", {}).get(SOURCE) != fit._hash(SOURCE):
        raise ValueError("Root input must pin final new diagnostic source")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    refs = spec["refs"]
    if (canonical_ref_v1(reader, refs["h028_module"]) != H028_MODULE
            or canonical_ref_v1(reader, refs["h028_config"]) != H028_CONFIG
            or reader.relative(refs["protocol"]["path"]) != PROTOCOL):
        raise ValueError("Immutable H028 functions/config and new protocol required")
    reader.read(refs["h028_module"]); reader.read(refs["protocol"])
    cfg = reader.decode_json(refs["h028_config"])
    if cfg["genotype"] != dict(alpha=0.5, census=350.0, window_bp=10000, ld_block_bp=1000):
        raise ValueError("No new windows, weights or pseudocounts in H031")
    for ref in cfg["helper_refs"].values():
        reader.read(ref)
    runtime = reader.decode_json(refs["runtime"])
    registration = reader.decode_json(refs["registration"])
    if (runtime.get("device") != "cpu" or runtime.get("job_id") != spec["job_id"]
            or registration.get("hypothesis_id", registration.get("id")) != "H031"):
        raise ValueError("Root CPU runtime and H031 registration required")
    storage = reader.decode_json(refs["storage"])
    from .project_budget_v8_consumer import validate_storage_v8
    validate_storage_v8(spec,dict(cfg,output_cap_bytes=OUTPUT_CAP),storage,
        source_ref={"path":SOURCE,"sha256":spec["source_sha256"][SOURCE]},
        config_ref=canonical_ref_v1(reader,refs["h028_config"]),
        runtime_ref=canonical_ref_v1(reader,refs["runtime"]))
    variants = reader.decode_json(refs["observed_variant_bindings"])
    bindings = reader.decode_json(refs["target_bindings"])
    base = reader.decode_json(cfg["base_fit_config"])
    target_meta = reader.decode_json(base["external_target_manifest"])
    by_id = {r["parent_id"]: r for r in target_meta["replicates"]}
    if set(spec["observed_parents"]) != set(fit.PARENTS) or set(variants["parents"]) != set(fit.PARENTS):
        raise ValueError("Same four original observation parents required")
    if set(bindings["parents"]) != set(fit.PARENTS):
        raise ValueError("Same four original external target metadata bindings required")
    for parent_id, binding in bindings["parents"].items():
        expected = {"path": f"data/biological_paths_execution_pilot_v1/{parent_id}/truth/full_truth_intervals.npz",
                    "sha256": by_id[parent_id]["files"]["truth/full_truth_intervals.npz"]["sha256"]}
        if canonical_ref_v1(reader, binding["truth_npz"]) != expected or binding["split"] != by_id[parent_id]["split"]:
            raise ValueError("Original external target binding changed")
    completed = reader.decode_json(refs["h028_completed"])
    baseline = reader.decode_json(refs["h028_frozen_baseline_metrics"])
    if (not completed["status"].startswith("completed_") or completed["hypothesis_id"] != "H028"
            or canonical_ref_v1(reader, completed["config_ref"]) != H028_CONFIG
            or baseline["arm"] != "frozen_genomic_smoothing"):
        raise ValueError("Completed immutable H028 genotype and baseline metadata required")
    return spec, cfg, variants, bindings, completed, baseline, runtime, reader, storage


def resources_v1(spec, storage):
    fit._check_active_v1(spec)
    if storage.get("expires_at_utc") and datetime.now(timezone.utc) >= fit._utc(storage["expires_at_utc"]):
        raise ValueError("CPU diagnostic reservation expired")
    from .project_budget_v8_consumer import budget_check_v8
    budget_check_v8(storage,boundary=True)
    import resource
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > spec["max_cpu_rss_bytes"]:
        raise ValueError("CPU diagnostic exceeds root RSS limit")


class ObservationsOnlyV1:
    def __init__(self, reader):
        self.reader, self.root = reader, reader.root

    def read(self, record):
        path = self.reader.relative(record["path"])
        if ("/truth/" in "/" + path.lower() + "/" or "oracle" in path.lower()
                or Path(path).suffix.lower() in (".trees", ".pt", ".npz", ".h5", ".hdf5", ".pkl")):
            raise ValueError("Observed-field preparation rejects truth, graph, model and previous prediction arrays")
        return self.reader.read(record)

    def decode_json(self, record):
        return json.loads(self.read(record))


def physical_summary_v1(value, widths):
    np = fit.np
    # value is [original bins, queries, channels].
    value = np.asarray(value, dtype=np.float64)
    weight = np.asarray(widths, dtype=np.float64)
    total = float(weight.sum() * value.shape[1])
    mean = np.einsum("l,lqc->c", weight, value) / total
    second = np.einsum("l,lqc->c", weight, value * value) / total
    query_mean = np.einsum("l,lqc->qc", weight, value) / weight.sum()
    spatial = second - (query_mean * query_mean).mean(0)
    covariance = np.einsum("l,lqi,lqj->ij", weight, value, value) / total - mean[:, None] * mean[None, :]
    eigen = np.linalg.eigvalsh(covariance)
    return {"mean_by_channel": mean, "RMS_by_channel": np.sqrt(np.maximum(second, 0.)),
            "global_std_by_channel": np.sqrt(np.maximum(second - mean * mean, 0.)),
            "spatial_std_RMS_by_channel": np.sqrt(np.maximum(spatial, 0.)),
            "covariance_eigenvalues_ascending": eigen,
            "numerical_rank_eigenvalue_gt_1e_minus10": int((eigen > 1e-10).sum()),
            "min_by_channel": value.min((0, 1)), "max_by_channel": value.max((0, 1))}


def effective_blocks_v1(positions, available, centers, coverage, widths, class_counts):
    np = fit.np
    inside = (positions >= coverage[0]) & (positions < coverage[1])
    x = positions[inside]
    available = available[inside]
    channels = [4 * h + int(np.flatnonzero(class_counts[h] > 0)[0]) for h in range(2)]
    called = available[:, :, channels]
    blocks, inverse, density = np.unique((x // 1000).astype(np.int64), return_inverse=True, return_counts=True)
    weights = 1. / density[inverse]
    weighted = called * weights[:, None, None]
    prefix = np.concatenate((np.zeros((1, *weighted.shape[1:])), np.cumsum(weighted, axis=0)), axis=0)
    start = np.r_[0, np.flatnonzero(np.diff(inverse)) + 1]
    end = np.r_[start[1:], len(x)]
    lo = np.searchsorted(x, np.maximum(centers - 5000., coverage[0]), side="left")
    hi = np.searchsorted(x, np.minimum(centers + 5000., coverage[1]), side="left")
    effective = np.zeros((len(centers), available.shape[1], 2), dtype=np.float64)
    represented = np.zeros_like(effective)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b <= a:
            continue
        first, last = int(inverse[a]), int(inverse[b - 1])
        mass = np.stack([prefix[min(b, end[j])] - prefix[max(a, start[j])] for j in range(first, last + 1)])
        total, square = mass.sum(0), (mass * mass).sum(0)
        np.divide(total * total, square, out=effective[i], where=square > 0)
        represented[i] = (mass > 0).sum(0)
    return {"definition": "(sum_called_block_mass)^2/sum_squared_called_block_mass;zero_if_empty",
            "original_1kb_blocks_in_coverage": len(blocks),
            "effective_blocks": physical_summary_v1(effective, widths),
            "called_blocks": physical_summary_v1(represented, widths)}


def prepare_and_seal_v1(spec, cfg, variants, reader, output, storage):
    """Observed SNPs/metadata only. All four seals precede target/weight reads."""
    from .observed_variant_residual_v1 import (
        parse_observed_vcf_v1, categorical_log_evidence_v1, composite_window_fields_v1,
        center_observed_query_fields_v1, validate_observed_roster_v1)
    np = fit.np
    observed = ObservationsOnlyV1(reader)
    parents, seals, diagnostics = {}, {}, []
    for parent_id in fit.PARENTS:
        resources_v1(spec, storage)
        refs = spec["observed_parents"][parent_id]
        prepared = observed.decode_json(refs["prepared_manifest"])
        entry = observed.decode_json(refs["observed_parent_entry"])
        original = observed.decode_json(entry["original_observed_sample_manifest"])
        inferred = observed.decode_json(refs["observed_roster"])
        joined = validate_observed_roster_v1(original)
        compare = ("sample_index", "sample_id", "haplotype", "role", "age_generations",
                   "observed_cohort", "observed_reference_class")
        if ([{k:r[k] for k in compare} for r in original] != [{k:r[k] for k in compare} for r in inferred]
                or prepared["parent_id"] != parent_id or entry["parent_id"] != parent_id
                or prepared["coordinate_contract"] != entry["coordinate_contract"]
                or prepared["files_sha256"]["observed_roster.json"] != refs["observed_roster"]["sha256"]
                or entry["no_input_variant_filtering"] is not True):
            raise ValueError("Original H028 observed identities/date/classes/support required")
        vrefs = variants["parents"][parent_id]
        folder = f"data/biological_paths_execution_pilot_v1/{parent_id}/observed"
        for key, name in {"vcf":"genotypes.vcf", "site_coordinates":"site_coordinate_manifest.json",
                          "source_roster":"sample_manifest.json"}.items():
            if canonical_ref_v1(reader, vrefs[key]) != {"path":folder+"/"+name,
                "sha256":entry["input_observed_sha256"][name]}:
                raise ValueError("Original H028 SNP lineage changed")
        if observed.decode_json(vrefs["source_roster"]) != original:
            raise ValueError("SNP and observation source rosters differ")
        coordinate = prepared["coordinate_contract"]
        coverage = np.asarray(coordinate["coverage_original_bp0"], dtype=np.float64)
        indexes = np.asarray(coordinate["retained_fixed100bp_indexes"], dtype=np.int64)
        full = 50. + 100. * np.arange(20000)
        expected = np.flatnonzero((full >= coverage[0]) & (full < coverage[1]))
        if not np.array_equal(indexes, expected) or not len(indexes) or np.any(np.diff(indexes) != 1):
            raise ValueError("Exact unchanged retained original scoring centers required")
        centers = full[indexes]
        edges = np.r_[100. * indexes, 100. * (indexes[-1] + 1)]
        edges[0], edges[-1] = max(edges[0], coverage[0]), min(edges[-1], coverage[1])
        widths = np.diff(edges)
        qi, ri = joined["query_indexes"], joined["reference_indexes"]
        labels = np.asarray([original[i]["observed_reference_class"] for i in ri], dtype=np.int64)
        ages = np.asarray([original[i]["age_generations"] for i in ri], dtype=np.float64)
        query_ages = [original[i]["age_generations"] for i in qi]
        alleles = parse_observed_vcf_v1(observed.read(vrefs["vcf"]).decode("utf-8"),
            original, observed.decode_json(vrefs["site_coordinates"]))
        scores, available, classes = categorical_log_evidence_v1(alleles.genotypes,
            alleles.allele_counts, qi, ri, labels, ages,
            alpha=cfg["genotype"]["alpha"], census=cfg["genotype"]["census"])
        if not np.array_equal(classes, [[24,16,22,22],[0,8,2,2]]):
            raise ValueError("Same observed younger84/older12 classes and absent olderANA required")
        fields, counts, window_meta = composite_window_fields_v1(alleles.positions_bp0,
            scores, available, centers, coverage, window_bp=cfg["genotype"]["window_bp"],
            ld_block_bp=cfg["genotype"]["ld_block_bp"])
        if len(alleles.positions_bp0) != entry["native_full_sites"] or np.any(fields[:,:,4] != 0):
            raise ValueError("Original full variant count and exact absent olderANA required")
        centered = center_observed_query_fields_v1(fields, counts, query_ages)
        direct = fields[:,:,:4].astype(np.float32)
        fallback = counts[:,:,:4].sum(-1) == 0
        prior = (classes[0] / classes[0].sum()).astype(np.float32)
        direct[fallback] = prior
        if (not np.isfinite(direct).all() or np.any(direct < 0)
                or not np.allclose(direct.sum(-1), 1., rtol=1e-5, atol=1e-6)):
            raise ValueError("Raw youngerC4 probabilities invalid; no floor or renormalization")
        split = "train" if "_train_" in parent_id else "val"
        parent = {"parent_id":parent_id, "split":split, "samples":original,
            "query_slots":np.asarray(qi,dtype=np.int64),
            "raw_query_nodes":np.asarray([original[i]["observed_node_id"] for i in qi],dtype=np.int64),
            "cohorts":np.asarray([original[i]["observed_cohort"] for i in qi]),
            "positions":centers, "edges":edges, "retained_indexes":indexes,
            "coverage_original":coverage.tolist(), "query_ids":np.asarray(qi,dtype=np.int64)}
        fallback_report = {}
        for name, columns in fit.strata_v1(parent).items():
            denominator = float(widths.sum() * len(columns))
            bp = float(np.sum(widths[:,None] * fallback[:,columns]))
            fallback_report[name] = {"fallback_query_bp":bp, "denominator_query_bp":denominator,
                "observed_evidence_query_bp":denominator-bp, "fallback_fraction":bp/denominator}
        effective = effective_blocks_v1(alleles.positions_bp0, available, centers, coverage, widths, classes)
        source_report = {"parent_id":parent_id, "split":split, "variant_refs":vrefs,
            **alleles.metadata, **window_meta, "younger_class_counts":classes[0],
            "older_class_counts":classes[1], "older_ANA_evidence_exact_zero":True,
            "raw_H028_unit_fields":physical_summary_v1(fields,widths),
            "centered_H028_fields":physical_summary_v1(centered.numpy(),widths),
            "direct_young_probabilities_with_declared_fallback":physical_summary_v1(direct,widths),
            "effective_blocks":effective, "fallback":fallback_report, "fallback_prior_C4":prior,
            "support_redefined_by_evidence":False, "targets_or_weights_read":False}
        seals[parent_id] = output.save(f"raw_younger_C4.{parent_id}.npz", {
            "C4_probabilities":direct, "positions_bp":centers, "bin_edges_bp":edges,
            "retained_fixed100bp_indexes":indexes, "query_sample_indexes":parent["query_slots"],
            "no_called_younger_window":fallback}, "npz")
        if split == "val":
            parent["_centered"] = centered
        parents[parent_id] = parent
        diagnostics.append(source_report)
        del alleles, scores, available, fields, counts, direct, centered
        print(json.dumps({"stage":"observed_direct_prediction_sealed","parent_id":parent_id,
                          "source_only":True,"centers":len(centers)}),flush=True)
    source_seal = output.save("observation_features_ready.json", {
        "schema":"observed_variant_signal_preparation_v1",
        "status":"all_four_predictions_sealed_before_targets_weights_previous_predictions",
        "prediction_seals":seals,"parents":diagnostics,"query_targets_opened":False,
        "weights_or_previous_prediction_payloads_opened":False})
    return parents, seals, source_seal


def read_prediction_v1(record, parent, reader, *, require_logits=False):
    np = fit.np
    with np.load(io.BytesIO(reader.read(record)), allow_pickle=False) as archive:
        p = archive["C4_probabilities"].copy()
        checks = {"positions_bp":parent["positions"], "bin_edges_bp":parent["edges"],
            "retained_fixed100bp_indexes":parent["retained_indexes"],
            "query_sample_indexes":parent["query_slots"]}
        if any(not np.array_equal(archive[k],value) for k,value in checks.items()):
            raise ValueError("Sealed prediction sample/physical support mismatch")
        logits = archive["logits"].copy() if "logits" in archive.files else None
    if (p.shape != (len(parent["positions"]),120,4) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(-1),1.,rtol=1e-5,atol=1e-6)):
        raise ValueError("Unfloored raw C4 prediction required")
    if require_logits and (logits is None or logits.shape != p.shape or not np.isfinite(logits).all()):
        raise ValueError("Original H028 baseline/selected logits required")
    return p, logits


def compact_score_v1(score):
    queries = score.pop("per_query")
    short = [tract for query in queries for tract in query["short_true_tracts"]]
    represented = sum(r["right_bp"]-r["left_bp"] for r in short)
    correct = sum((r["right_bp"]-r["left_bp"])*r["correct_overlap_fraction"] for r in short)
    score["short_tract_summary"] = {"count":len(short),"represented_query_bp":represented,
        "correct_query_bp":correct,"accuracy":correct/represented if represented else None}
    score.pop("retained_original_grid_indexes")
    return score


def coefficient_report_v1(checkpoint_ref, reader):
    torch, np = fit.torch, fit.np
    checkpoint = torch.load(io.BytesIO(reader.read(checkpoint_ref)), map_location="cpu", weights_only=True)
    if checkpoint["arm"] != "genotype_only" or checkpoint["seed"] != 101:
        raise ValueError("Immutable selected H028 genotype-only head required")
    state = checkpoint["model_state"]
    if set(state) != {"context_head.weight"}:
        raise ValueError("No alternate model or calibration head in diagnostic")
    w = state["context_head.weight"]
    if w.shape != (4,8) or w.dtype != torch.float32 or not bool(torch.isfinite(w).all()):
        raise ValueError("Actual finite selected FP32 four-by-eight head required")
    array = w.numpy()
    decision = array - array.mean(0, keepdims=True)
    return w, {"checkpoint":checkpoint_ref,"epoch":checkpoint["epoch"],
        "allocated_head_parameters":32,"potential_data_responsive_parameters":28,
        "identically_absent_older_ANA_column":4,"absent_column_max_abs":float(np.abs(array[:,4]).max()),
        "selected_coefficients":array,"abs_mean_by_channel":np.abs(array).mean(0),
        "abs_max_by_channel":np.abs(array).max(0),"decision_row_L2":np.linalg.norm(decision,axis=1),
        "decision_singular_values":np.linalg.svd(decision,compute_uv=False),
        "baseline_logit_head_input_or_independent_calibration":False}


def correction_scale_v1(centered, weight, baseline_logits, selected_logits, widths):
    torch, np = fit.torch, fit.np
    correction = torch.nn.functional.linear(centered, weight).numpy()
    actual = selected_logits - baseline_logits
    decision = correction - correction.mean(-1,keepdims=True)
    ordered = np.sort(baseline_logits, axis=-1)
    margins = (ordered[:,:,-1]-ordered[:,:,-2])[:,:,None]
    scale = physical_summary_v1(decision,widths)
    margin = physical_summary_v1(margins,widths)
    return {"reconstructed_residual_decision_logits":scale,"baseline_top_two_logit_margin":margin,
        "decision_RMS_over_baseline_margin_RMS":float(np.sqrt(np.mean(scale["RMS_by_channel"]**2)) /
            margin["RMS_by_channel"][0]) if margin["RMS_by_channel"][0] else None,
        "reconstruction_max_abs_difference_from_sealed_correction":float(np.abs(correction-actual).max()),
        "CPU_reconstruction_numeric_scope":"FP32_linear_on_unchanged_centered_fields;compare_to_original_GPU_seals_without_refit"}


def complementarity_v1(parent, truth, direct, baseline, selected, columns, helpers):
    """Exact original truth intersections; no bin-majority or hard truth proxy."""
    np = fit.np
    bins, rows, spans = helpers.intersections(parent["edges"], truth.intervals)
    table = np.zeros((2,2),dtype=np.float64)
    class_table = np.zeros((4,2,2),dtype=np.float64)
    conditional = {name:{"query_bp":0.,"direct_correct_bp":0.,"direct_truth_probability_bp":0.,
        "direct_probability_gt_baseline_for_truth_bp":0.,"direct_nll_sum":0.,"infinite_nll_query_bp":0.,
        "direct_confusion":np.zeros((4,4),dtype=np.float64)}
        for name in ("baseline_correct","baseline_wrong")}
    outcomes = {name:0. for name in ("direct_could_fix_and_residual_fixes",
        "direct_could_fix_but_residual_stays_wrong","baseline_wrong_residual_fixes",
        "baseline_correct_residual_damages","both_baseline_and_direct_wrong")}
    class_outcomes = {name:np.zeros(4,dtype=np.float64) for name in outcomes}
    for start in range(0,len(bins),1024):
        b, r, s = bins[start:start+1024], rows[start:start+1024], spans[start:start+1024]
        y = truth.c4[r][:,columns]
        pb, pg, ps = baseline[b][:,columns], direct[b][:,columns], selected[b][:,columns]
        bp = np.broadcast_to(s[:,None],y.shape)
        b_good, g_good, s_good = pb.argmax(-1)==y, pg.argmax(-1)==y, ps.argmax(-1)==y
        np.add.at(table,(b_good.astype(int).ravel(),g_good.astype(int).ravel()),bp.ravel())
        np.add.at(class_table,(y.ravel(),b_good.astype(int).ravel(),g_good.astype(int).ravel()),bp.ravel())
        chosen = np.take_along_axis(pg,y[...,None],axis=-1)[...,0]
        previous = np.take_along_axis(pb,y[...,None],axis=-1)[...,0]
        positive = chosen > 0
        for name, mask in (("baseline_correct",b_good),("baseline_wrong",~b_good)):
            cell=conditional[name]
            cell["query_bp"] += float(np.sum(bp*mask))
            cell["direct_correct_bp"] += float(np.sum(bp*mask*g_good))
            cell["direct_truth_probability_bp"] += float(np.sum(bp*mask*chosen))
            cell["direct_probability_gt_baseline_for_truth_bp"] += float(np.sum(bp*mask*(chosen>previous)))
            cell["infinite_nll_query_bp"] += float(np.sum(bp*mask*~positive))
            cell["direct_nll_sum"] += float(np.sum(bp*mask*-np.log(np.where(positive,chosen,1.))))
            np.add.at(cell["direct_confusion"],(y.ravel(),pg.argmax(-1).ravel()),(bp*mask).ravel())
        masks={
            "direct_could_fix_and_residual_fixes":~b_good&g_good&s_good,
            "direct_could_fix_but_residual_stays_wrong":~b_good&g_good&~s_good,
            "baseline_wrong_residual_fixes":~b_good&s_good,
            "baseline_correct_residual_damages":b_good&~s_good,
            "both_baseline_and_direct_wrong":~b_good&~g_good}
        for name, mask in masks.items():
            outcomes[name] += float(np.sum(bp*mask))
            np.add.at(class_outcomes[name],y.ravel(),(bp*mask).ravel())
    denominator=float(table.sum())
    for cell in conditional.values():
        bp=cell["query_bp"]
        cell.update(direct_accuracy=cell["direct_correct_bp"]/bp if bp else None,
            direct_mean_probability_for_truth=cell["direct_truth_probability_bp"]/bp if bp else None,
            direct_fraction_probability_for_truth_gt_baseline=cell["direct_probability_gt_baseline_for_truth_bp"]/bp if bp else None,
            direct_nll=None if cell["infinite_nll_query_bp"] or not bp else cell["direct_nll_sum"]/bp)
    complement=float(table[0,1])
    return {"denominator_query_bp":denominator,"joint_correctness_baseline_rows_direct_columns":table,
        "joint_correctness_by_truth_class":class_table,"conditional":conditional,
        "conditional_outcomes_query_bp":outcomes,"conditional_outcomes_by_truth_class":class_outcomes,
        "direct_could_fix_baseline_wrong_query_bp":complement,
        "direct_could_fix_fraction_total":complement/denominator,
        "fraction_direct_correctable_errors_missed_by_selected_residual":
            outcomes["direct_could_fix_but_residual_stays_wrong"]/complement if complement else None,
        "oracle_union_accuracy_diagnostic_bound_only":1.-table[0,0]/denominator}


def run_v1(record):
    spec,cfg,variants,bindings,completed,baseline_meta,runtime,reader,storage = inputs_v1(record)
    # All scientific payload operations start only after reviewed root admission.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("H031 cannot execute with an initialized CUDA context")
    fit.np,fit.torch=np,torch
    from .project_budget_v8_consumer import PilotOutputV8
    output=PilotOutputV8(spec["output_directory"],storage,reader)
    helpers=fit.metric_helpers_v1()
    started={"schema":"observed_variant_signal_diagnostic_evidence_v1","hypothesis_id":"H031",
        "status":"running","input_ref":record,"device":"cpu","runtime":runtime,
        "configuration":H028_CONFIG,"immutable_genotype_module":H028_MODULE,
        "new_windows_weights_calibration_or_fits":False,"output_cap_bytes":OUTPUT_CAP,
        "TRAIN_scoring":"loss_free_diagnostic","TEST_parents":0,"development_only":True}
    output.save("started.json",started)
    begin=time.monotonic()
    try:
        parents,seals,source_seal=prepare_and_seal_v1(spec,cfg,variants,reader,output,storage)
        # Read every new prediction seal before the first target or old array.
        for parent_id,ref in seals.items():
            read_prediction_v1(ref,parents[parent_id],reader)
        reader.read(source_seal)
        selected=completed["models"]["genotype_only"]["selected"]
        old_genotype={r["parent_id"]:r["prediction_seal"] for r in selected["cells"]}
        old_baseline={r["parent_id"]:r["prediction_seal"] for r in baseline_meta["cells"]}
        val_ids={p for p,parent in parents.items() if parent["split"]=="val"}
        if set(old_genotype)!=val_ids or set(old_baseline)!=val_ids:
            raise ValueError("Both immutable original VAL comparisons required; no invented TRAIN baseline")
        comparisons={}
        for parent_id in sorted(val_ids):
            bp,bl=read_prediction_v1(old_baseline[parent_id],parents[parent_id],reader,require_logits=True)
            rp,rl=read_prediction_v1(old_genotype[parent_id],parents[parent_id],reader,require_logits=True)
            comparisons[parent_id]=(bp,bl,rp,rl)
        weight,weights=coefficient_report_v1(selected["checkpoint"],reader)
        if weights["epoch"] != selected["epoch"]:
            raise ValueError("Selected checkpoint epoch disagrees with immutable H028 receipt")
        output.save("selected_H028_genotype_coefficients.json",weights)
        scored,conditional,scales=[],[],[]
        for parent_id,parent in parents.items():
            resources_v1(spec,storage)
            binding=bindings["parents"][parent_id]
            if (binding["query_sample_indexes"]!=parent["query_slots"].tolist()
                    or binding["query_raw_node_ids"]!=parent["raw_query_nodes"].tolist()):
                raise ValueError("External truth identities differ from observation-only source join")
            direct,_=read_prediction_v1(seals[parent_id],parent,reader)
            bp=rp=base_logits=resid_logits=None
            if parent["split"]=="val":
                bp,base_logits,rp,resid_logits=comparisons.pop(parent_id)
                scales.append({"parent_id":parent_id,**correction_scale_v1(parent["_centered"],weight,
                    base_logits,resid_logits,np.diff(parent["edges"]))})
            purpose="TRAIN_loss" if parent["split"]=="train" else "VAL_after_sealed_prediction"
            truth=fit.ExternalTargetV1(parent,binding,purpose,reader,
                None if parent["split"]=="train" else seals[parent_id])
            groups={name:compact_score_v1(helpers.full_span_score(direct,parent,truth,cols,cfg["metrics"]))
                    for name,cols in fit.strata_v1(parent).items()}
            scored.append({"parent_id":parent_id,"split":parent["split"],"prediction_seal":seals[parent_id],
                "target_provenance":truth.provenance,"target_use":"loss_free_diagnostic_not_fit",
                "groups":groups,"unchanged_original_full_span_scorer":True})
            if parent["split"]=="val":
                conditional.append({"parent_id":parent_id,
                    "baseline_prediction":old_baseline[parent_id],"selected_residual_prediction":old_genotype[parent_id],
                    "direct_prediction":seals[parent_id],
                    "groups":{name:complementarity_v1(parent,truth,direct,bp,rp,cols,helpers)
                              for name,cols in fit.strata_v1(parent).items()}})
                del parent["_centered"]
            del truth,direct,bp,rp,base_logits,resid_logits
        means={}
        for split in ("train","val"):
            cells=[r for r in scored if r["split"]==split]
            means[split]={}
            for name in cells[0]["groups"]:
                values=[r["groups"][name] for r in cells]
                infinite=any(r["infinite_nll_query_bp"] for r in values)
                means[split][name]={"accuracy":float(np.mean([r["accuracy"] for r in values])),
                    "nll":None if infinite else float(np.mean([r["nll"] for r in values])),
                    "NLL_is_infinite":infinite,"brier":float(np.mean([r["brier"] for r in values]))}
        score_ref=output.save("raw_younger_C4.metrics.json",{"cells":scored,
            "equal_parent_metrics_by_split":means,"primary":"present_day64","no_probability_floor":True,
            "TRAIN_is_loss_free_diagnostic":True,"independent_validation_claim":False})
        conditional_ref=output.save("VAL_conditional_complementarity.json",{"cells":conditional,
            "scope":"two_original_VAL_parents_only;exact_span_mass;diagnostic_no_selector_or_fit",
            "prior_H028_baseline_full_metrics_ref":spec["refs"]["h028_frozen_baseline_metrics"],
            "prior_H028_selected_full_metrics_source":spec["refs"]["h028_completed"]})
        scale_ref=output.save("VAL_selected_residual_scale.json",{"cells":scales,
            "thresholds_fitted_or_selected":False,"representation_or_optimizer_cause_established":False})
        resources_v1(spec,storage)
        reader.read(record)
        result={**started,"status":"completed_CPU_fixed_variant_signal_diagnostic",
            "observation_feature_seal":source_seal,"prediction_seals":seals,
            "direct_score":score_ref,"conditional_complementarity":conditional_ref,"residual_scale":scale_ref,
            "equal_parent_metrics_by_split":means,"elapsed_seconds":time.monotonic()-begin,
            "output_accounted_bytes":output.used,"fits_run":False,"GPU_used":False}
        output.save("completed.json",result)
        return result
    except Exception as error:
        output.save("failed.json",{**started,"status":"failed_partial_preserved",
            "error_type":type(error).__name__,"error":str(error),"elapsed_seconds":time.monotonic()-begin,
            "output_accounted_bytes":output.used})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status":"source_only_not_executed","hypothesis_id":"H031",
            "implementation":{"path":SOURCE,"sha256":fit._hash(SOURCE)},
            "device":"cpu","output_cap_bytes":OUTPUT_CAP,"immutable_H028_config":H028_CONFIG},indent=2))
        return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Pinned reviewed active root H031 CPU input required")
    run_v1({"path":args.input,"sha256":args.expected_input_sha256})
    return 0


if __name__=="__main__":
    raise SystemExit(main())

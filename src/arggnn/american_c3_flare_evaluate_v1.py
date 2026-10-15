"""Stream native FLARE calls/raw rounded ANP onto prespecified C3 focals, then score externally."""
import contextlib
import gzip
import re

from .american_c3_common_v1 import CLASSES, require, read, bound, cli
from .american_c3_flare_common_v1 import (
    PIN_COMMIT, PROJECTION, NATIVE_OPTIONS, observation_ref, stage_completion, validate_focals,
)

SOURCE = "src/arggnn/american_c3_flare_evaluate_v1.py"
METHOD = "FLARE_native_v0.6.0_tag_CURRENT_RECONSTRUCTION"
DEFAULTS = dict(
    schema="american_c3_flare_evaluation_v1", execution_enabled=False,
    mode="seal_native", native_run_ref=None, focal_registry_ref=None,
    sealed_prediction_completion_ref=None, prediction_ref=None, external_truth_ref=None,
    projection=PROJECTION, max_focal_records=10000, max_markers=1000000,
    write_derived_marker_tracts=True, max_tract_records=10000000,
    output_cap_bytes=2000000000, method=METHOD,
)


def seal_native(cfg, w):
    import numpy as np
    from .american_c3_evaluate_v1 import registry, load_npz, verify_prediction
    require(cfg["external_truth_ref"] is None and cfg["prediction_ref"] is None
            and cfg["projection"] == PROJECTION, "Native sealing cannot read private truth")
    native = stage_completion(cfg["native_run_ref"], "completed_C3_native_FLARE")
    converted = stage_completion(native["conversion_ref"], "completed_C3_FLARE_observation_conversion")
    require(native["source_commit"] == converted["native_source_commit"] == PIN_COMMIT
            and native["native_options"] == converted["native_options"] == NATIVE_OPTIONS
            and native["native_outputs_modified"] is False
            and native["focal_registry_ref"] == converted["focal_registry_ref"] == cfg["focal_registry_ref"]
            and native["projection"] == converted["projection"] == PROJECTION,
            "Exact unmodified native inputs/common focal projection required")
    r = registry(observation_ref(cfg["focal_registry_ref"]))
    n = len(r["record_ids"])
    require(n <= cfg["max_focal_records"], "Full focal-count cap exceeded")
    bridge = read(observation_ref(converted["phased_bridge_ref"]))
    m = read(observation_ref(converted["observation_ref"]))
    rows = read(observation_ref(m["roster_ref"]))["samples"]
    validate_focals(r, m, rows)
    require(bridge["parent_id"] == native["parent_id"] == r["parent_id"].item()
            and bridge["focal_registry_ref"] == cfg["focal_registry_ref"]
            and bridge["observation_ref"] == converted["observation_ref"]
            and bridge["class_order"] == list(CLASSES),
            "Exact observed phased identity bridge required")
    grid = load_npz(observation_ref(converted["expected_marker_grid_ref"]))
    positions, source_ix = grid["positions_bp0"], grid["source_site_indexes"]
    require(positions.dtype == source_ix.dtype == np.int64 and positions.ndim == 1
            and source_ix.shape == positions.shape and 0 < len(positions) <= cfg["max_markers"]
            and (np.diff(positions) > 0).all() and (np.diff(source_ix) > 0).all()
            and grid["parent_id"].item() == bridge["parent_id"],
            "Pre-truth native marker grid required")
    query_pairs = bridge["query_pairs"]
    query_ids = [p["vcf_sample_id"] for p in query_pairs]
    columns = np.asarray([i for p in query_pairs for i in p["sample_indices"]], dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(columns)}
    require(len(lookup) == len(columns) and all(int(i) in lookup for i in r["sample_indices"]),
            "Every focal must map to an exact native phased query haplotype")
    focal_hap = np.asarray([lookup[int(i)] for i in r["sample_indices"]], dtype=np.int64)
    g = np.load(bound(observation_ref(m["genotypes_ref"])), mmap_mode="r", allow_pickle=False)
    input_pos = np.load(bound(observation_ref(m["positions_ref"])), allow_pickle=False)
    require(g.dtype == np.int8 and g.shape == (len(input_pos), len(rows))
            and np.array_equal(input_pos[source_ix], positions),
            "Native output must refer to unchanged observed genotype markers")
    length = float(bridge["sequence_length_bp"])
    order = np.argsort(r["positions_bp0"], kind="stable")
    sorted_pos = r["positions_bp0"][order]
    calls = np.full(n, 255, dtype=np.uint8)
    native_codes = np.full(n, 255, dtype=np.uint8)
    raw_probs = np.full((n, 3), np.nan, dtype=np.float32)
    raw_strings = np.full(n, "", dtype="U32")
    chosen_marker = np.full(n, -1, dtype=np.int64)
    cursor, count, allele_cursor, tract_count = 0, 0, -1, 0
    ancestry_order, inverse, class_order, previous = None, None, None, None
    starts, active = np.zeros(len(columns), dtype=np.float64), None
    tract_cap = cfg["max_tract_records"]*96+4096 if cfg["write_derived_marker_tracts"] else 0
    require(tract_cap + n*512 + 16777216 <= cfg["output_cap_bytes"],
            "Complete focal/raw-posterior/tract count bound exceeds cap")
    tract_path = w.prepare("derived_native_marker_tracts.tsv", tract_cap) if tract_cap else None
    tract_context = tract_path.open("x") if tract_path else contextlib.nullcontext(None)

    def assign(stop, marker):
        nonlocal cursor
        ids = order[cursor:stop]
        h = focal_hap[ids]
        calls[ids] = marker["calls"][h]
        native_codes[ids] = marker["native_codes"][h]
        raw_probs[ids] = marker["probabilities"][h]
        raw_strings[ids] = marker["raw_strings"][h]
        chosen_marker[ids] = marker["position"]
        cursor = stop

    line_cap = 1024+128*len(query_pairs)
    raw_cap = 4194304+len(positions)*line_cap
    raw_bytes = 0
    with tract_context as tracts, bound(observation_ref(m["alleles_ref"])).open() as allele_stream:
        if tracts: tracts.write("sample_index\tleft_bp0\tright_bp0\tclass_index\n")
        def emit(hap, end):
            nonlocal tract_count
            if tracts:
                tract_count += 1
                require(tract_count <= cfg["max_tract_records"], "Derived tract-count cap exceeded")
                require(end > starts[hap], "Positive complete derived tract interval required")
                tracts.write("%d\t%.17g\t%.17g\t%d\n" % (
                    columns[hap], starts[hap], end, active[hap]))
        header_seen = False
        with gzip.open(bound(native["native_output_refs"][".anc.vcf.gz"]), "rt", encoding="ascii") as stream:
            while True:
                line = stream.readline(line_cap+1)
                if not line: break
                raw_bytes += len(line)
                require(len(line) <= line_cap and raw_bytes <= raw_cap, "Native VCF decompression/line cap exceeded")
                if line.startswith("##ANCESTRY="):
                    match = re.fullmatch(r"##ANCESTRY=<([^>]+)>\n?", line)
                    require(match and ancestry_order is None, "Exactly one native ancestry-index header")
                    labels = dict(part.split("=") for part in match.group(1).split(","))
                    require(set(labels) == set(CLASSES)
                            and {int(x) for x in labels.values()} == set(range(3)),
                            "Exact native EUR/EAS/AFR ancestry ontology")
                    ancestry_order = [next(k for k, v in labels.items() if int(v) == j) for j in range(3)]
                    inverse = np.asarray([CLASSES.index(k) for k in ancestry_order], dtype=np.uint8)
                    class_order = np.asarray([int(labels[k]) for k in CLASSES])
                elif line.startswith("##"):
                    continue
                elif line.startswith("#CHROM"):
                    require(not header_seen and line.rstrip("\n").split("\t")[9:] == query_ids,
                            "Exact target VCF sample order required")
                    header_seen = True
                else:
                    require(header_seen and ancestry_order is not None and count < len(positions),
                            "Complete native VCF header/marker sequence required")
                    fields = line.rstrip("\n").split("\t")
                    require(len(fields) == 9+len(query_ids)
                            and fields[0] == bridge["chromosome"]
                            and int(fields[1])-1 == int(positions[count])
                            and fields[2] == "c3_site_%d" % source_ix[count]
                            and fields[8] == "GT:AN1:AN2:ANP1:ANP2",
                            "No missing/reordered/relabelled native marker or output fields")
                    site = int(source_ix[count])
                    while allele_cursor < site:
                        allele_line = allele_stream.readline(129)
                        require(allele_line and len(allele_line) <= 128, "Bounded complete allele metadata")
                        allele_cursor += 1
                    import json
                    alleles = json.loads(allele_line)
                    require([fields[3]]+fields[4].split(",") == alleles,
                            "Native allele strings/order differ from complete observed VCF")
                    marker_calls, marker_codes, marker_probs, strings, gt = [], [], [], [], []
                    for value in fields[9:]:
                        tokens = value.split(":")
                        require(len(tokens) == 5 and "|" in tokens[0] and "/" not in tokens[0],
                                "Native phased complete GT/call/posterior fields required")
                        pair_gt = tokens[0].split("|")
                        require(len(pair_gt) == 2, "Diploid native phase required")
                        gt.extend(int(a) for a in pair_gt)
                        for hap in (0, 1):
                            code = int(tokens[hap+1])
                            raw = tokens[hap+3]
                            prob = np.asarray([float(a) for a in raw.split(",")], dtype=np.float32)
                            require(0 <= code < 3 and prob.shape == (3,)
                                    and np.isfinite(prob).all() and ((prob >= 0)&(prob <= 1)).all()
                                    and abs(float(prob.sum())-1.) <= .0151 and len(raw) <= 32,
                                    "Native two-decimal posterior range/rounding envelope required")
                            marker_codes.append(code)
                            marker_calls.append(inverse[code])
                            marker_probs.append(prob[class_order])
                            strings.append(raw)
                    require(np.array_equal(np.asarray(gt, dtype=np.int64), g[site, columns]),
                            "Native phased genotypes must preserve each exact original haplotype column")
                    current = dict(
                        position=int(positions[count]), calls=np.asarray(marker_calls, dtype=np.uint8),
                        native_codes=np.asarray(marker_codes, dtype=np.uint8),
                        probabilities=np.asarray(marker_probs, dtype=np.float32),
                        raw_strings=np.asarray(strings, dtype="U32"),
                    )
                    if previous is not None:
                        boundary = .5*(previous["position"]+current["position"])
                        assign(int(np.searchsorted(sorted_pos, boundary, side="left")), previous)
                        changed = np.flatnonzero(active != current["calls"])
                        for hap in changed: emit(int(hap), boundary)
                        starts[changed] = boundary
                        active[changed] = current["calls"][changed]
                    else:
                        active = current["calls"].copy()
                    previous = current
                    count += 1
                    if count % 4096 == 0: w.adapter.budget_check_v9(w.storage, boundary=True)
        require(count == len(positions) and previous is not None,
                "Every expected native-default marker must be present; no missing-call deletion")
        assign(n, previous)
        for hap in range(len(columns)): emit(hap, length)
    require((calls < 3).all() and np.isfinite(raw_probs).all()
            and (chosen_marker >= 0).all(), "All prespecified common focals must be complete")
    identity = {k: r[k] for k in (
        "record_ids", "sample_nodes", "sample_indices", "positions_bp0", "parent_id", "class_order",
    )}
    prediction = dict(identity, probabilities=np.eye(3, dtype=np.float32)[calls])
    verify_prediction(prediction, r)
    prediction_ref = w.npz("predictions.npz", **prediction)
    raw_ref = w.npz(
        "native_focal_posteriors_preserved.npz", **identity,
        native_ANP_probabilities_C3_order=raw_probs, native_ANP_decimal_strings=raw_strings,
        native_ancestry_order=np.asarray(ancestry_order), native_ancestry_indices=native_codes,
        native_hard_calls_C3=calls, native_marker_positions_bp0=chosen_marker,
    )
    tract_ref = w.finish(tract_path, tract_cap) if tract_path else None
    edge_carried = int(((r["positions_bp0"] < positions[0]) | (r["positions_bp0"] > positions[-1])).sum())
    return w.json("completed.json", dict(
        status="sealed_C3_native_FLARE_calls_and_raw_posteriors",
        method=METHOD, prediction_ref=prediction_ref, raw_focal_posterior_ref=raw_ref,
        derived_marker_tracts_ref=tract_ref, derived_tract_records=tract_count,
        native_run_ref=cfg["native_run_ref"], conversion_ref=native["conversion_ref"],
        raw_native_output_refs=native["native_output_refs"],
        focal_registry_ref=cfg["focal_registry_ref"], parent_id=bridge["parent_id"],
        class_order=list(CLASSES), records=n, native_markers=count,
        projection=PROJECTION, edge_carried_focal_records=edge_carried,
        common_focal_registry_unchanged=True, native_AN1_AN2_calls_used_unchanged=True,
        one_hot_is_native_call_transport=True, rounded_ANP_renormalized=False,
        raw_rounded_ANP_modified=False, native_VCF_modified=False,
        query_ancestry_truth_opened=False, author_Table1_exact_reproduction=False,
    ))


def execute(cfg, w):
    require(cfg["schema"] == DEFAULTS["schema"] and cfg["method"] == METHOD,
            "Exact native C3 baseline identity required")
    if cfg["mode"] == "seal_native": return seal_native(cfg, w)
    require(cfg["mode"] == "score", "Only native seal or external score modes")
    sealed = stage_completion(cfg["sealed_prediction_completion_ref"],
                              "sealed_C3_native_FLARE_calls_and_raw_posteriors")
    require(sealed["prediction_ref"] == cfg["prediction_ref"]
            and sealed["focal_registry_ref"] == cfg["focal_registry_ref"]
            and sealed["common_focal_registry_unchanged"] is True
            and sealed["native_AN1_AN2_calls_used_unchanged"] is True
            and sealed["rounded_ANP_renormalized"] is False,
            "Only already sealed native calls enter common external scoring")
    from . import american_c3_evaluate_v1 as common
    score_cfg = dict(common.DEFAULTS, mode="score", method=METHOD,
        prediction_ref=cfg["prediction_ref"], focal_registry_ref=cfg["focal_registry_ref"],
        external_truth_ref=cfg["external_truth_ref"], support=PROJECTION)
    return common.execute(score_cfg, w)


if __name__ == "__main__":
    cli(SOURCE, DEFAULTS, execute, dependencies=(
        "src/arggnn/american_c3_flare_common_v1.py", "src/arggnn/american_c3_evaluate_v1.py",
    ))

"""H028 one-seed FP32 observed-SNP information discovery.

Selected H025 smoother stays the exact output skip. Observation feature
preparation ends before TRAIN target arrays open. No calibration bypass or
new-GNN novelty claim. Same full-support scorer and per-class/tract products.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import time

from . import census_tsinfer_fit_v1 as fit
from . import train_persistence_trace_pilot_v1 as h025

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/train_observed_variant_residual_v1.py"
MODULE = "src/arggnn/observed_variant_residual_v1.py"
CONFIG = "configs/observed_variant_residual_v1.json"
ARMS = ("genotype_only", "graph_only", "genotype_graph_fusion")


def inputs_v1(record):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    lease = validate_lease(); apply_and_validate_title()
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "observed_variant_residual_input_v1"
            or spec.get("root_authorizer") != "root_agent" or spec.get("fit_authorized") is not True
            or spec.get("status") != "active" or spec.get("seed") != 101
            or spec.get("device") != "cuda" or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Active root H028 seed101 CUDA input required")
    fit._check_active_v1(spec); fit.require_logical_job_v1(lease, spec)
    refs = spec["refs"]
    cfg = reader.decode_json(refs["config"])
    if (refs["config"]["path"] != CONFIG or cfg["hypothesis_id"] != "H028"
            or cfg["arms"] != list(ARMS) or cfg["seeds"] != [101] or cfg["dtype"] != "float32"
            or spec.get("epochs", 10) != 10):
        raise ValueError("New H028 three-arm10epoch160step config required")
    for path in (SOURCE, MODULE):
        if spec["source_sha256"].get(path) != fit._hash(path):
            raise ValueError("Root input must pin both NEW H028 implementations")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    for ref in cfg["helper_refs"].values():
        reader.read(ref)
    base = reader.decode_json(cfg["base_fit_config"])
    if cfg["metrics"] != dict(base["metrics"], aggregation="unweighted_parent_mean_one_seed_no_probability_ensemble"):
        raise ValueError("Original scorer policy with explicit single-fit aggregation required")
    if (cfg["genotype"] != dict(alpha=0.5, census=350.0, window_bp=10000, ld_block_bp=1000)
            or cfg["epochs"] != 10 or cfg["optimizer_steps_per_arm"] != 160
            or cfg["TRAIN_tiles_per_epoch"] != 16 or cfg["output_cap_bytes"] != 1000000000
            or cfg["parameters_per_arm"] != 32 or cfg["tile_bp"] != 250000
            or cfg["graph_field"] != "native_reference_from_fixed_H026_metric_tree_context"):
        raise ValueError("Fixed H028 SNP/cohort/budget/graph-context recipe required")
    variants = reader.decode_json(refs["observed_variant_bindings"])
    if variants.get("schema") != "observed_variant_bindings_v1" or set(variants["parents"]) != set(fit.PARENTS):
        raise ValueError("Exact four-parent original observed SNP metadata bindings required")
    reader.read(refs["runtime"])
    reader.read(refs["frozen_baseline_source"])
    if (ROOT / refs["frozen_baseline_source"]["path"]).resolve() != Path(h025.__file__).resolve():
        raise ValueError("Frozen baseline source must bind the actual H025 factory")
    baseline_cfg = reader.decode_json(refs["frozen_baseline_config"])
    completed = reader.decode_json(refs["frozen_baseline_completed"])
    selected = completed["models"]["genomic_smoothing"]["selected"]
    if (selected["checkpoint"] != refs["frozen_baseline_checkpoint"] or selected["epoch"] != 10
            or completed["seed"] != 101 or not completed["status"].startswith("completed_")):
        raise ValueError("Immutable selected H025 genomic-smoothing epoch10 checkpoint required")
    reader.read(refs["frozen_baseline_checkpoint"])
    storage = reader.decode_json(refs["storage"])
    reservation = storage["reservation"]
    destination = (reader.root / reader.relative(spec["output_directory"])).resolve()
    if (storage["job_id"] != spec["job_id"] or storage["output_directory"] != spec["output_directory"]
            or not 0 < storage["reserved_new_bytes"] <= cfg["output_cap_bytes"]
            or reservation["reserved_bytes"] != storage["reserved_new_bytes"]
            or reservation["output"] != str(destination)
            or any(ref not in reservation["consumer_bindings"] for ref in (
                {"path": SOURCE, "sha256": spec["source_sha256"][SOURCE]}, refs["config"], refs["runtime"]))):
        raise ValueError("Separate root v4 output reservation of at most1GB required")
    fit.budget_check_v1(storage, first=True)
    emitter = reader.decode_json(refs["emitter_bindings"])
    if (set(spec["observed_parents"]) != set(fit.PARENTS) or set(emitter["parents"]) != set(fit.PARENTS)
            or emitter["emitter_condition"] != cfg["emitter_condition"]
            or emitter["training_predictions"] != "in_sample_not_out_of_fold"
            or emitter["class_order"] != ["ANA", "CHG", "WHG", "EHG"]):
        raise ValueError("Same explicit four-parent hybrid emitter/observation joins required")
    bindings = reader.decode_json(refs["target_bindings"])
    metadata = reader.decode_json(base["external_target_manifest"])
    by_id = {row["parent_id"]: row for row in metadata["replicates"]}
    if set(bindings["parents"]) != set(fit.PARENTS):
        raise ValueError("Same four original TRAIN/VAL target bindings required")
    for parent_id, binding in bindings["parents"].items():
        truth = f"data/biological_paths_execution_pilot_v1/{parent_id}/truth/full_truth_intervals.npz"
        if (binding["truth_npz"] != {"path": truth,
                "sha256": by_id[parent_id]["files"]["truth/full_truth_intervals.npz"]["sha256"]}
                or binding["split"] != by_id[parent_id]["split"]):
            raise ValueError("Original external C4 target/support binding changed")
    fit._science_v1()
    return spec, cfg, base, baseline_cfg, emitter, bindings, variants, reader, storage


class ObservationOnlyReaderV1:
    """Payload boundary for feature preparation, including inherited helpers."""
    def __init__(self, reader):
        self.reader = reader
        self.root = reader.root

    def read(self, record, **kwargs):
        path = self.reader.relative(record["path"])
        parts = path.lower().split("/")
        if "truth" in parts or any("oracle" in part for part in parts):
            raise ValueError("Feature preparation cannot read truth or oracle payloads")
        return self.reader.read(record, **kwargs)

    def decode_json(self, record):
        return json.loads(self.read(record))


def prepare_v1(spec, cfg, baseline_cfg, emitter, variants, reader, output, storage, lease, device):
    """Observation features and exact teacher skip, without any target reader."""
    import tempfile
    import tskit
    from .persistence_trace_kernel_v1 import MetricTreeHoldV1
    from .reference_census_innovation_v1 import prepare_reference_census_context_v1
    from .observed_variant_residual_v1 import (
        parse_observed_vcf_v1, prepare_observed_variant_fields_v1, ObservedVariantContextV1)
    reader = ObservationOnlyReaderV1(reader)
    np, torch = fit.np, fit.torch
    cached, parents, train_ids, val_ids, diagnostics = {}, {}, [], {}, []
    tmp = output.path / "tmp"
    tmp.mkdir(exist_ok=True)
    factory = h025.models_v1(baseline_cfg, torch.device("cpu"))
    teacher = factory["genomic_smoothing"]["model"]
    del factory
    saved = torch.load(io.BytesIO(reader.read(spec["refs"]["frozen_baseline_checkpoint"])),
                       map_location="cpu", weights_only=True)
    if saved["arm"] != "genomic_smoothing" or saved["epoch"] != 10:
        raise ValueError("Selected frozen H025 smoother checkpoint identity differs")
    teacher.load_state_dict(saved["model_state"], strict=True)
    teacher.to(device=device, dtype=torch.float32).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parent_id in fit.PARENTS:
        h025.resources_v1(spec, storage, lease)
        refs = spec["observed_parents"][parent_id]
        prepared = reader.decode_json(refs["prepared_manifest"])
        entry = reader.decode_json(refs["observed_parent_entry"])
        source_ref = entry["original_observed_sample_manifest"]
        source = reader.decode_json(source_ref)
        tree_ref = {k: entry["inferred_tree"][k] for k in ("path", "sha256")}
        inferred = reader.decode_json(refs["observed_roster"])
        frame = emitter["parents"][parent_id]
        emitted = reader.decode_json(frame["observed_roster"])
        source_map, inferred_map, emitted_map = map(h025.roster_v1, (source, inferred, emitted))
        split = "train" if "_train_" in parent_id else "val"
        if (prepared["parent_id"] != parent_id or prepared["parent_split"] != split
                or prepared["files_sha256"]["observed_roster.json"] != refs["observed_roster"]["sha256"]
                or "tsinfer" not in entry["inferred_tree"]["condition"].lower()
                or set(source_map) != set(inferred_map) or set(source_map) != set(emitted_map)
                or entry["no_input_variant_filtering"] is not True):
            raise ValueError("Fixed tsinfer and exact cross-emitter observed joins required")
        for key, row in source_map.items():
            if any(row[field] != other[field] for other in (inferred_map[key], emitted_map[key])
                   for field in ("sample_index", "role", "age_generations", "observed_cohort", "observed_reference_class")):
                raise ValueError("Observed identities/roles/cohorts/dates/classes differ across hybrid join")
        observed_folder = f"data/biological_paths_execution_pilot_v1/{parent_id}/observed"
        variant_refs = variants["parents"][parent_id]
        expected_names = {"vcf": "genotypes.vcf", "site_coordinates": "site_coordinate_manifest.json",
                          "source_roster": "sample_manifest.json"}
        for key, filename in expected_names.items():
            if variant_refs[key] != {"path": observed_folder + "/" + filename,
                                      "sha256": entry["input_observed_sha256"][filename]}:
                raise ValueError("VCF/coordinate/roster binding differs from original inference observations")
        if json.loads(reader.read(variant_refs["source_roster"])) != source:
            raise ValueError("Original observed variant and graph source rosters differ")
        query_rows = [row for row in source if row["role"] == "query"]
        query_keys = [(row["sample_id"], row["haplotype"]) for row in query_rows]
        emitter_nodes = np.asarray([emitted_map[key]["inferred_tree_node_id"] for key in query_keys], dtype=np.int64)
        reader.read(frame["grid_prediction"])
        with np.load(ROOT / frame["grid_prediction"]["path"], allow_pickle=False) as archive:
            grid = {name: archive[name].copy() for name in ("probabilities", "supported", "positions_bp",
                "local_positions_bp", "grid_indexes", "bin_edges_bp", "query_nodes")}
        columns = {int(node): column for column, node in enumerate(grid["query_nodes"])}
        if len(columns) != 120 or set(columns) != set(emitter_nodes.tolist()):
            raise ValueError("Frozen emitter Q120 columns must join original sample identities")
        probabilities = grid["probabilities"][:, [columns[int(node)] for node in emitter_nodes]]
        if (probabilities.dtype != np.float32 or probabilities.shape[1:] != (120, 4)
                or not grid["supported"].all() or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0) or not np.allclose(probabilities.sum(-1), 1, atol=1e-5, rtol=0)):
            raise ValueError("Complete positive frozen emitter grid required; no floor or support drop")
        coordinate = prepared["coordinate_contract"]
        coverage, offset = coordinate["coverage_original_bp0"], coordinate["local_to_original_offset_bp"]
        roles = [SimpleNamespace(actual_id=row["sample_index"], role=row["role"]) for row in source]
        loaded = (None, roles, 50. + 100. * np.arange(20000), offset, coverage, {"source": source_ref}, None)
        observed_query_join = {"query_sample_indexes": [r["sample_index"] for r in query_rows],
                               "query_raw_node_ids": [r["observed_node_id"] for r in query_rows]}
        parent = fit.scoring_parent_v1({"parent_id": parent_id, "split": split}, None,
                                      loaded, reader, observed_query_join)
        if (not np.array_equal(grid["positions_bp"], parent["positions"])
                or not np.array_equal(grid["grid_indexes"], parent["retained_indexes"])
                or not np.array_equal(grid["grid_indexes"], coordinate["retained_fixed100bp_indexes"])
                or not np.array_equal(grid["bin_edges_bp"], parent["edges"])
                or not np.array_equal(grid["positions_bp"], 50. + 100. * grid["grid_indexes"])
                or not np.array_equal(grid["positions_bp"] - grid["local_positions_bp"],
                                      np.full(len(grid["positions_bp"]), offset))
                or coverage != entry["coordinate_contract"]["coverage_original_bp0"]):
            raise ValueError("Original SNP/emitter/graph physical support differs")
        coordinates = reader.decode_json(variant_refs["site_coordinates"])
        alleles = parse_observed_vcf_v1(reader.read(variant_refs["vcf"]).decode("utf-8"), source, coordinates)
        genotype, genotype_meta = prepare_observed_variant_fields_v1(alleles, source,
            parent["positions"], coverage, **cfg["genotype"])
        if (genotype_meta["original_sites"] != entry["native_full_sites"]
                or genotype_meta["younger_reference_count"] != 84 or genotype_meta["older_reference_count"] != 12
                or genotype_meta["older_class_counts"][0] != 0):
            raise ValueError("Complete original variants, both donor cohorts and absent olderANA required")
        del alleles
        with tempfile.NamedTemporaryFile(dir=tmp, suffix=".observed.trees") as file:
            file.write(reader.read(tree_ref)); file.flush()
            ts = tskit.load(file.name)
        native_samples = [int(inferred_map[(row["sample_id"], row["haplotype"])]["inferred_tree_node_id"])
                          for row in source]
        actual_ids = torch.tensor([row["sample_index"] for row in source], dtype=torch.long)
        if (len(set(native_samples)) != 280 or set(native_samples) != set(map(int, ts.samples()))
                or ts.sequence_length != coverage[1] - coverage[0]
                or any(float(ts.node(node).time) != float(row["age_generations"])
                       for node, row in zip(native_samples, source))):
            raise ValueError("Complete actual/native sample/date join differs")
        if ts.num_populations or ts.num_migrations or ts.num_individuals or ts.metadata:
            raise ValueError("Observed inferred graph must have removed ancestry/population metadata")
        if np.any(ts.tables.nodes.population != -1) or any(len(table.metadata)
                for table in (ts.tables.nodes, ts.tables.edges, ts.tables.sites, ts.tables.mutations)):
            raise ValueError("No graph ancestry-label feature is allowed")
        parent["probabilities"] = probabilities
        direct = torch.from_numpy(np.log(probabilities))
        parents[parent_id] = parent
        reference_rows = [r for r in source if r["role"] == "pure_source_reference"]
        reference_ids = torch.tensor([r["sample_index"] for r in reference_rows], dtype=torch.long)
        reference_classes = torch.tensor([r["observed_reference_class"] for r in reference_rows], dtype=torch.long)
        query_ids = torch.tensor([r["sample_index"] for r in query_rows], dtype=torch.long)
        namespace = parent_id + ":" + tree_ref["sha256"]
        baseline = np.empty_like(probabilities)
        seen = np.zeros(len(parent["positions"]), dtype=bool)
        tile_meta = []
        for tile_id in range(8):
            h025.resources_v1(spec, storage, lease)
            left = max(tile_id * cfg["tile_bp"], float(parent["edges"][0]))
            right = min((tile_id + 1) * cfg["tile_bp"], float(parent["edges"][-1]))
            indexes = np.flatnonzero((parent["positions"] >= left) & (parent["positions"] < right))
            if not len(indexes):
                raise ValueError("All eight original physical tiles must retain readouts")
            with torch.no_grad():
                frozen = teacher(direct[indexes].to(device)).detach().cpu()
            baseline[indexes] = frozen.numpy(); seen[indexes] = True
            trees, intervals = [], []
            for tree in ts.trees():
                lo, hi = max(left, tree.interval.left + offset), min(right, tree.interval.right + offset)
                if hi <= lo:
                    continue
                union = set()
                for sample in native_samples:
                    node = sample
                    while node != -1 and node not in union:
                        union.add(node); node = tree.parent(node)
                packed = [int(node) for node in tree.nodes(order="postorder") if node in union]
                local = {node: i for i, node in enumerate(packed)}
                trees.append(MetricTreeHoldV1(
                    torch.tensor([-1 if tree.parent(node) == -1 else local[tree.parent(node)] for node in packed]),
                    torch.tensor([tree.time(node) for node in packed], dtype=torch.float64),
                    torch.tensor(packed), namespace,
                    torch.tensor([local[node] for node in native_samples]), actual_ids))
                intervals.append([lo, hi])
            graph = prepare_reference_census_context_v1(trees, torch.tensor(intervals, dtype=torch.float64),
                torch.from_numpy(parent["positions"][indexes]), query_ids, reference_ids, reference_classes,
                query_diploid_ids=[r["sample_id"] for r in query_rows],
                query_haplotypes=[r["haplotype"] for r in query_rows],
                parent_namespace=namespace, seed=101, **cfg["context"])
            if (graph.metadata["younger_reference_count"] != 84 or graph.metadata["older_reference_count"] != 12
                    or graph.metadata["younger_class_counts"] != genotype_meta["younger_class_counts"]
                    or graph.metadata["older_class_counts"] != genotype_meta["older_class_counts"]):
                raise ValueError("Identical observed donor classes/date cohorts in both channels required")
            graph_field = graph.fields["native_reference"].clone()
            geno_field = genotype[indexes].clone()
            if (geno_field.shape != graph_field.shape or graph_field.dtype != torch.float32
                    or not bool(torch.isfinite(geno_field).all() & torch.isfinite(graph_field).all())
                    or bool((geno_field[:, :, 4] != 0).any()) or bool((graph_field[:, :, 4] != 0).any())):
                raise ValueError("Commensurate fixedFP32 fields with exact absent olderANA required")
            identity = f"{parent_id}:{tile_id}"
            context = ObservedVariantContextV1(geno_field, graph_field, {
                "identity": identity, "genotype": "original_parent10kb_block_weighted_composite",
                "graph": "fixed_H026_native_reference", "fusion": "fixed_arithmetic_mean"})
            cached[identity] = dict(parent_id=parent_id, indexes=indexes, baseline=frozen, context=context)
            tile_meta.append({"identity": identity, "readouts": len(indexes), "graph_holds": len(trees)})
            (train_ids if split == "train" else val_ids.setdefault(parent_id, [])).append(identity)
            del graph, trees
            print(json.dumps({"stage": "observed_variant_prepare", "identity": identity,
                "readouts": len(indexes), "source_only": True}), flush=True)
        if not seen.all():
            raise ValueError("Fixed teacher or observation fields lack original support")
        parent["frozen_baseline_logits"] = baseline
        diagnostics.append({"parent_id": parent_id, "split": split, "variant_refs": variant_refs,
            "tree_ref": tree_ref, "coverage_original_bp0": coverage, "tiles": tile_meta, **genotype_meta})
        del ts, genotype, direct
    del teacher, saved
    if len(cached) != 32 or len(train_ids) != 16 or len(val_ids) != 2:
        raise ValueError("Original32 tiles,16TRAIN tiles and twoVAL parents required")
    feature_seal = output.save("observation_features_ready.json", {
        "schema": "observed_variant_feature_preparation_v1", "status": "complete_observed_features_before_targets",
        "parents": diagnostics, "frozen_baseline_checkpoint": spec["refs"]["frozen_baseline_checkpoint"],
        "query_truth_opened_during_preparation": False, "fitted_ancestry_label_feature": False,
        "field_storage": "FP32_RAM_only", "input_ref": spec["refs"]["observed_variant_bindings"],
        "fusion": cfg["fusion"], "novel_GNN_claim": False})
    return cached, parents, train_ids, val_ids, feature_seal


def train_targets_after_features_v1(parents, bindings, reader, helpers, feature_seal):
    reader.read(feature_seal)
    targets = {}
    for parent_id, parent in parents.items():
        binding = bindings["parents"][parent_id]
        if (binding["query_sample_indexes"] != parent["query_slots"].tolist()
                or binding["query_raw_node_ids"] != parent["raw_query_nodes"].tolist()):
            raise ValueError("External target identity metadata differs from observation-only preparation")
        if parent["split"] == "train":
            target = fit.ExternalTargetV1(parent, binding, "TRAIN_loss", reader)
            targets[parent_id] = target.target_mass(parent, helpers)
    return targets


def run_v1(record):
    spec, cfg, base, baseline_cfg, emitter, bindings, variants, reader, storage = inputs_v1(record)
    from .biological_candidate import configure_fit
    from .biological_gst_control_v1 import exact_span_loss
    from .observed_variant_residual_v1 import ObservedVariantResidualV1
    torch, np = fit.torch, fit.np
    output = h025.PilotOutputV1(spec["output_directory"], storage, reader)
    device, runtime, lease = configure_fit("cuda", base, output.path)
    output.used = sum(max(path.stat().st_size, path.stat().st_blocks * 512) + 8192
                      for path in output.path.rglob("*") if path.is_file())
    helpers = fit.metric_helpers_v1()
    receipt = dict(schema="observed_variant_residual_evidence_v1", hypothesis_id="H028",
        status="running", input_ref=record, config_ref=spec["refs"]["config"], seed=101, dtype="float32",
        epochs=10, optimizer_steps_per_arm=160, condition=cfg["condition"], runtime=runtime,
        frozen_baseline_checkpoint=spec["refs"]["frozen_baseline_checkpoint"],
        observed_information_controls=list(ARMS), local_recalibration_path=False, fixed_mean_fusion=True,
        development_only=True, TEST_parents=0, models={})
    output.save("started.json", receipt)
    start = time.monotonic()
    try:
        cached, parents, train_ids, val_ids, feature_seal = prepare_v1(
            spec, cfg, baseline_cfg, emitter, variants, reader, output, storage, lease, device)
        receipt["observation_feature_seal"] = feature_seal
        targets = train_targets_after_features_v1(parents, bindings, reader, helpers, feature_seal)
        torch.manual_seed(101)
        initial = ObservedVariantResidualV1().to(device=device, dtype=torch.float32)
        states = {arm: dict(model=deepcopy(initial), steps=0, best=float("inf"), best_epoch=None,
                            current=[], retained_best=[], history=[]) for arm in ARMS}
        for state in states.values():
            state["optimizer"] = torch.optim.AdamW(state["model"].parameters(),
                lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        baseline_seals = [(parent_id, h025.seal_v1(output, "frozen_genomic_smoothing", 0,
            parents[parent_id], parents[parent_id]["frozen_baseline_logits"])) for parent_id in val_ids]
        baseline_score = None
        for epoch in range(1, 11):
            epoch_start = time.monotonic()
            h025.resources_v1(spec, storage, lease)
            order = list(range(16)); random.Random(101 + epoch).shuffle(order)
            losses = {arm: [] for arm in ARMS}
            for index in order:
                tile = cached[train_ids[index]]
                frozen = tile["baseline"].to(device)
                mass = torch.as_tensor(targets[tile["parent_id"]][tile["indexes"]], dtype=torch.float32, device=device)
                for arm in ARMS:
                    state = states[arm]; model = state["model"]; model.train()
                    state["optimizer"].zero_grad(set_to_none=True)
                    logits = model(frozen, tile["context"].field_for_arm(arm).to(device))
                    loss = exact_span_loss(logits, mass)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite exact-span TRAIN loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
                    state["optimizer"].step(); state["steps"] += 1
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                        raise ValueError("Nonfinite fitted context-head parameter")
                    losses[arm].append(float(loss.detach()))
            seals = {arm: [] for arm in ARMS}
            for parent_id, identities in val_ids.items():
                parent = parents[parent_id]
                values = {arm: np.empty_like(parent["frozen_baseline_logits"]) for arm in ARMS}
                for identity in identities:
                    tile = cached[identity]
                    frozen = tile["baseline"].to(device)
                    with torch.no_grad():
                        for arm in ARMS:
                            states[arm]["model"].eval()
                            logits = states[arm]["model"](frozen, tile["context"].field_for_arm(arm).to(device))
                            if not bool(torch.isfinite(logits).all()):
                                raise ValueError("Nonfinite full-support context readout")
                            values[arm][tile["indexes"]] = logits.cpu().numpy()
                for arm in ARMS:
                    seals[arm].append((parent_id, h025.seal_v1(output, arm, epoch, parent, values[arm])))
            if baseline_score is None:
                cells, means = h025.score_v1(baseline_seals, parents, bindings, reader, helpers, cfg)
                baseline_score = dict(arm="frozen_genomic_smoothing", epoch=0, cells=cells, equal_parent_metrics=means)
                output.save("frozen_genomic_smoothing.metrics.json", baseline_score)
            for arm in ARMS:
                cells, means = h025.score_v1(seals[arm], parents, bindings, reader, helpers, cfg)
                primary = means["primary_present_day_64"]
                nll = float("inf") if primary["NLL_is_infinite"] else primary["nll"]
                state = states[arm]
                previous = {r["path"]: r for r in state["current"] + state["retained_best"]}
                checkpoint = output.save(f"{arm}.epoch_{epoch:02d}.pt", {
                    "model_state": {k: v.detach().cpu() for k, v in state["model"].state_dict().items()},
                    "optimizer_state": state["optimizer"].state_dict(), "epoch": epoch, "arm": arm,
                    "seed": 101, "input_ref": record, "frozen_baseline_checkpoint": spec["refs"]["frozen_baseline_checkpoint"]}, "torch")
                state["current"] = [r for _, r in seals[arm]] + [checkpoint]
                row = dict(epoch=epoch, arm=arm, optimizer_steps=state["steps"],
                    TRAIN_mean_tile_query120_NLL=float(np.mean(losses[arm])),
                    cells=cells, equal_parent_metrics=means, checkpoint=checkpoint)
                state["history"].append(row)
                if state["best_epoch"] is None or nll < state["best"]:
                    state.update(best=nll, best_epoch=epoch, selected=row, retained_best=list(state["current"]))
                row["selected_epoch"] = state["best_epoch"]
                output.save(f"{arm}.epoch_{epoch:02d}.metrics.json", row)
                keep = {r["path"] for r in state["current"] + state["retained_best"]}
                for path, product in previous.items():
                    if path not in keep:
                        output.discard(product)
            print(json.dumps({"stage": "observed_variant_epoch", "epoch": epoch, "steps_per_arm": epoch * 16,
                "elapsed_seconds": time.monotonic() - epoch_start,
                "primary_present64": {arm: states[arm]["history"][-1]["equal_parent_metrics"]["primary_present_day_64"]
                                      for arm in ARMS}}), flush=True)
        table = [dict(arm="frozen_genomic_smoothing", selected_epoch=0,
                      **baseline_score["equal_parent_metrics"]["primary_present_day_64"])]
        for arm, state in states.items():
            if state["steps"] != 160:
                raise ValueError("Incomplete fixed160step one-seed fit")
            selected = state["selected"]
            saved = torch.load(io.BytesIO(reader.read(selected["checkpoint"])), map_location="cpu", weights_only=True)
            delta = float(saved["model_state"]["context_head.weight"].abs().max())
            receipt["models"][arm] = dict(selected=selected, steps=160, parameter_count=32,
                selected_context_weight_max_delta=delta,
                retained_records={r["path"]: r for r in state["current"] + state["retained_best"]})
            table.append(dict(arm=arm, selected_epoch=state["best_epoch"],
                              **selected["equal_parent_metrics"]["primary_present_day_64"]))
        output.save("comparison.json", dict(rows=table, frozen_baseline=baseline_score,
            condition=cfg["condition"], novelty_claim=False, ordered_cross_tree_mechanism_tested=False))
        h025.resources_v1(spec, storage, lease)
        reader.read(record)
        receipt.update(status="completed_one_seed_FP32_observed_variant_information_pilot",
                       comparison=table, elapsed_seconds=time.monotonic() - start, output_accounted_bytes=output.used)
    except Exception as error:
        receipt.update(status="failed_partial_preserved", error_type=type(error).__name__,
                       error=str(error), elapsed_seconds=time.monotonic() - start, output_accounted_bytes=output.used)
        output.save("failed.json", receipt)
        raise
    output.save("completed.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "source_only_not_executed", "hypothesis_id": "H028",
            "implementation": {"path": SOURCE, "sha256": fit._hash(SOURCE)},
            "config": {"path": CONFIG, "sha256": fit._hash(CONFIG)}, "seed": 101,
            "epochs": 10, "steps_per_arm": 160, "dtype": "float32"}, indent=2))
        return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Pinned root H028 input required")
    run_v1({"path": args.input, "sha256": args.expected_input_sha256})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""H032 one-seed FP32 static donor-support Gram discovery.

Three new static gate conditions: independent-donor Gram, marginal outer Gram
and genuine tree Gram. Frozen H030 static will supply its selected comparison separately after both fits complete. The selected H025 per-tile skip and original scorer
remain unchanged. No claim of new TGN/DeltaNet primitives or true ancestors.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import resource
from types import SimpleNamespace
import time

from . import census_tsinfer_fit_v1 as fit
from . import train_persistence_trace_pilot_v1 as h025

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/train_reference_dependence_retry_v3.py"
MODULE = "src/arggnn/reference_dependence_gnn_v1.py"
CONFIG = "configs/reference_dependence_retry_v3.json"
ARMS = ("independence_gram", "marginal_outer_gram", "tree_gram")
roster_v1 = h025.roster_v1



def inputs_v1(record):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    lease = validate_lease(); apply_and_validate_title()
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "reference_dependence_gnn_input_v1"
            or spec.get("root_authorizer") != "root_agent" or spec.get("fit_authorized") is not True
            or spec.get("status") != "active" or spec.get("seed") != 101
            or spec.get("device") != "cuda" or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Active root H032 seed101 CUDA input required")
    fit._check_active_v1(spec); fit.require_logical_job_v1(lease, spec)
    refs = spec["refs"]
    cfg = reader.decode_json(refs["config"])
    if cfg.get("execution_guard_ready") is not True:
        raise ValueError("Exact source-ready successor budget adapter/configuration required")
    if (refs["config"]["path"] != CONFIG or cfg["hypothesis_id"] != "H032"
            or cfg["arms"] != list(ARMS) or cfg["seeds"] != [101] or cfg["dtype"] != "float32"
            or spec.get("epochs", 10) != 10):
        raise ValueError("New H032 three-static-arm10epoch160step config required")
    for path in (SOURCE, MODULE):
        if spec["source_sha256"].get(path) != fit._hash(path):
            raise ValueError("Root input must pin both NEW H032 implementations")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    for ref in cfg["helper_refs"].values():
        reader.read(ref)
    for ref in cfg["runtime_recovery"]["predecessor_refs"].values():
        reader.read(ref)
    failed = reader.decode_json(cfg["runtime_recovery"]["predecessor_refs"]["failed"])
    if (failed.get("status") != "failed_partial_preserved"
            or failed.get("error") != "Process inventory changed during writer inspection"
            or failed.get("models") != {} or "first_training_tile" in failed):
        raise ValueError("H032 recovery is only the original pre-training failure")
    base = reader.decode_json(cfg["base_fit_config"])
    if cfg["metrics"] != dict(base["metrics"], aggregation="unweighted_parent_mean_one_seed_no_probability_ensemble"):
        raise ValueError("Original scorer policy with explicit single-fit aggregation required")
    reader.read(refs["runtime"])
    reader.read(refs["frozen_baseline_source"])
    if (ROOT / refs["frozen_baseline_source"]["path"]).resolve() != Path(h025.__file__).resolve():
        raise ValueError("Frozen baseline source must bind the actual H025 factory")
    baseline_cfg = reader.decode_json(refs["frozen_baseline_config"])
    if refs["frozen_baseline_config"]["path"] != h025.CONFIG or baseline_cfg["hypothesis_id"] != "H025":
        raise ValueError("Exact immutable H025 skip configuration required")
    completed = reader.decode_json(refs["frozen_baseline_completed"])
    selected = completed["models"]["genomic_smoothing"]["selected"]
    if (selected["checkpoint"] != refs["frozen_baseline_checkpoint"] or selected["epoch"] != 10
            or completed["seed"] != 101 or not completed["status"].startswith("completed_")):
        raise ValueError("Immutable selected H025 genomic-smoothing epoch10 checkpoint required")
    reader.read(refs["frozen_baseline_checkpoint"])
    storage = reader.decode_json(refs["storage"])
    from .project_budget_v9_consumer_retry_v3 import validate_storage_v9
    if cfg["output_cap_bytes"] != 2000000000 or cfg["optimizer_steps_per_arm"] != 160:
        raise ValueError("Fixed compact2GB/160step discovery budget required")
    validate_storage_v9(spec, cfg, storage,
        source_ref={"path":SOURCE,"sha256":spec["source_sha256"][SOURCE]},
        config_ref=refs["config"], runtime_ref=refs["runtime"])
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
    for key, reference in cfg["frozen_strong_reference_refs"].items():
        if refs[key] != reference:
            raise ValueError("Exact already-completed H026 comparator refs required")
    strong = reader.decode_json(refs["strong_reference_completed"])
    strong_cfg = reader.decode_json(refs["strong_reference_config"])
    strong_input = reader.decode_json(strong["input_ref"])
    selected = strong["models"]["native_reference"]["selected"]
    if (refs["strong_reference_config"]["path"] != "configs/reference_census_innovation_v1.json"
            or strong_cfg["hypothesis_id"] != "H026" or strong["seed"] != 101
            or not strong["status"].startswith("completed_")
            or selected["checkpoint"] != refs["strong_reference_checkpoint"]
            or strong_input["observed_parents"] != spec["observed_parents"]
            or strong_input["refs"]["emitter_bindings"] != refs["emitter_bindings"]
            or strong_input["refs"]["frozen_baseline_checkpoint"] != refs["frozen_baseline_checkpoint"]
            or strong_cfg["metrics"] != cfg["metrics"]):
        raise ValueError("Reuse exact frozen selected H026 native_reference on the same observations/support")
    reader.read(refs["strong_reference_checkpoint"])
    # Immutable launch metadata only, never a changing H030 output.
    prospective_ref = refs["first_moment_reference_input"]
    if prospective_ref != cfg["first_moment_reference_input"]:
        raise ValueError("Exact prospective H030 launched input pin required")
    launched = reader.decode_json(prospective_ref)
    launched_cfg = reader.decode_json(launched["refs"]["config"])
    expected_sources = {item["path"]:item["sha256"] for name,item in cfg["helper_refs"].items()
                        if name in ("H030_first_moment_model","H030_first_moment_driver")}
    if (launched.get("schema") != "reference_logcavity_gnn_input_v1"
            or launched.get("seed") != 101 or launched.get("root_authorizer") != "root_agent"
            or launched.get("fit_authorized") is not True
            or launched["observed_parents"] != spec["observed_parents"]
            or launched["refs"]["emitter_bindings"] != refs["emitter_bindings"]
            or launched["refs"]["target_bindings"] != refs["target_bindings"]
            or launched["refs"]["frozen_baseline_checkpoint"] != refs["frozen_baseline_checkpoint"]
            or launched["refs"]["config"] != cfg["first_moment_reference_config"]
            or launched["output_directory"] != cfg["first_moment_reference_output_directory"]
            or launched_cfg["metrics"] != cfg["metrics"]
            or any(launched["source_sha256"].get(path) != digest for path,digest in expected_sources.items())):
        raise ValueError("Prospective H030 static must bind identical observations/support/skip and exact sources")
    fit._science_v1()
    return spec, cfg, base, baseline_cfg, emitter, bindings, reader, storage


def resources_v1(spec, storage, lease):
    """New V9 boundary without passing new authority through any V4 consumer."""
    from .project_budget_v9_consumer_retry_v3 import budget_check_v9
    lightweight_resources_v1(spec, lease)
    budget_check_v9(storage, boundary=True)



def lightweight_resources_v1(spec, lease):
    """Lease/RSS/allocator guard without BudgetV4 historical consumer rehashes."""
    from .biological_candidate import check_fit_lease
    fit._check_active_v1(spec)
    check_fit_lease(lease)
    fit.require_logical_job_v1(lease, spec)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > spec["max_cpu_rss_bytes"]:
        raise ValueError("H032 CPU RSS peak exceeds root cap")
    if fit.torch.cuda.max_memory_reserved() > spec["max_cuda_reserved_bytes"]:
        raise ValueError("H032 CUDA allocator peak exceeds root cap")


def prepare_v1(spec, cfg, baseline_cfg, emitter, bindings, reader, helpers, output, storage, lease, device):
    import tskit
    from .persistence_trace_kernel_v1 import MetricTreeHoldV1
    from .reference_dependence_gnn_v1 import prepare_reference_dependence_bank_v1
    np, torch = fit.np, fit.torch
    cached, parents, train_ids, val_ids, diagnostics = {}, {}, [], {}, []
    from .project_budget_v9_consumer_retry_v3 import budget_check_v9
    tmp = output.path / "tmp"
    if output.used + 8192 > output.limit:
        raise ValueError("Temporary directory allowance exceeds output cap")
    budget_check_v9(storage, boundary=True, required_write_bytes=8192)
    tmp.mkdir()
    output.used += 8192
    budget_check_v9(storage, boundary=True)
    factory = h025.models_v1(baseline_cfg, torch.device("cpu"))
    teacher = factory["genomic_smoothing"]["model"]
    del factory
    saved = torch.load(io.BytesIO(reader.read(spec["refs"]["frozen_baseline_checkpoint"])),
                       map_location="cpu", weights_only=True)
    if saved["arm"] != "genomic_smoothing" or saved["epoch"] != 10:
        raise ValueError("Frozen H025 checkpoint identity changed")
    teacher.load_state_dict(saved["model_state"], strict=True)
    teacher.to(device=device, dtype=torch.float32).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parent_id in fit.PARENTS:
        resources_v1(spec, storage, lease)
        refs = spec["observed_parents"][parent_id]
        prepared = reader.decode_json(refs["prepared_manifest"])
        entry = reader.decode_json(refs["observed_parent_entry"])
        source_ref = entry["original_observed_sample_manifest"]
        source = reader.decode_json(source_ref)
        tree_ref = {k: entry["inferred_tree"][k] for k in ("path", "sha256")}
        inferred = reader.decode_json(refs["observed_roster"])
        frame = emitter["parents"][parent_id]
        emitted = reader.decode_json(frame["observed_roster"])
        source_map, inferred_map, emitted_map = map(roster_v1, (source, inferred, emitted))
        split = "train" if "_train_" in parent_id else "val"
        if (prepared["parent_id"] != parent_id or prepared["parent_split"] != split
                or prepared["files_sha256"]["observed_roster.json"] != refs["observed_roster"]["sha256"]
                or "tsinfer" not in entry["inferred_tree"]["condition"].lower()
                or set(source_map) != set(inferred_map) or set(source_map) != set(emitted_map)):
            raise ValueError("Bound tsinfer producer and exact cross-emitter observed sample joins required")
        for key, row in source_map.items():
            if any(row[field] != other[field] for other in (inferred_map[key], emitted_map[key])
                   for field in ("role", "age_generations", "observed_cohort", "observed_reference_class")):
                raise ValueError("Observed roles, cohorts, dates or reference labels differ across the hybrid join")
        query_rows = [row for row in source if row["role"] == "query"]
        keys = [(row["sample_id"], row["haplotype"]) for row in query_rows]
        emitter_nodes = np.asarray([emitted_map[key]["inferred_tree_node_id"] for key in keys], dtype=np.int64)
        query_ids = torch.tensor([row["sample_index"] for row in query_rows], dtype=torch.long)
        reader.read(frame["grid_prediction"])
        with np.load(ROOT / frame["grid_prediction"]["path"], allow_pickle=False) as archive:
            grid = {name: archive[name].copy() for name in (
                "probabilities", "supported", "positions_bp", "local_positions_bp",
                "grid_indexes", "bin_edges_bp", "query_nodes")}
        columns = {int(node): column for column, node in enumerate(grid["query_nodes"])}
        if len(columns) != 120 or set(columns) != set(emitter_nodes.tolist()):
            raise ValueError("Frozen emitter Q120 columns must join observed sample identities exactly")
        permutation = [columns[int(node)] for node in emitter_nodes]
        probabilities = grid["probabilities"][:, permutation]
        if (probabilities.dtype != np.float32 or probabilities.shape[1:] != (120, 4)
                or not grid["supported"][:, permutation].all() or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0) or not np.allclose(probabilities.sum(-1), 1, atol=1e-5, rtol=0)):
            raise ValueError("Complete finite positive frozen grid probabilities required; no bin dropping or floor")
        coordinate = prepared["coordinate_contract"]
        coverage = coordinate["coverage_original_bp0"]
        offset = coordinate["local_to_original_offset_bp"]
        roles = [SimpleNamespace(actual_id=row["sample_index"], role=row["role"]) for row in source]
        loaded = (None, roles, 50. + 100. * np.arange(20000), offset, coverage, {"source": source_ref}, None)
        parent = fit.scoring_parent_v1({"parent_id": parent_id, "split": split}, None,
                                      loaded, reader, bindings["parents"][parent_id])
        if (not np.array_equal(grid["positions_bp"], parent["positions"])
                or not np.array_equal(grid["grid_indexes"], parent["retained_indexes"])
                or not np.array_equal(grid["grid_indexes"], coordinate["retained_fixed100bp_indexes"])
                or not np.array_equal(grid["bin_edges_bp"], parent["edges"])
                or not np.array_equal(grid["positions_bp"], 50. + 100. * grid["grid_indexes"])
                or not np.array_equal(grid["positions_bp"] - grid["local_positions_bp"],
                                      np.full(len(grid["positions_bp"]), offset))):
            raise ValueError("Original emitter/tsinfer grid, offset and full clipped-bin support differ")
        import tempfile
        tree_payload = reader.read(tree_ref)
        temporary_allowance = len(tree_payload) + 8192
        if output.used + temporary_allowance > output.limit:
            raise ValueError("Actual bounded native-tree temporary payload exceeds output cap")
        budget_check_v9(storage, boundary=True, required_write_bytes=temporary_allowance)
        try:
            with tempfile.NamedTemporaryFile(dir=tmp, suffix=".observed.trees") as file:
                file.write(tree_payload); file.flush()
                budget_check_v9(storage, boundary=True)
                ts = tskit.load(file.name)
        finally:
            budget_check_v9(storage, boundary=True)
        del tree_payload
        native_samples = [int(inferred_map[(row["sample_id"], row["haplotype"])]["inferred_tree_node_id"])
                          for row in source]
        actual_ids = torch.tensor([row["sample_index"] for row in source], dtype=torch.long)
        if (len(set(native_samples)) != 280 or set(native_samples) != set(map(int, ts.samples()))
                or ts.sequence_length != coverage[1] - coverage[0]
                or any(float(ts.node(node).time) != float(row["age_generations"])
                       for node, row in zip(native_samples, source))):
            raise ValueError("Complete actual/native observed sample and direct-date join differs")
        parent["probabilities"] = probabilities
        parent["unaries"] = torch.from_numpy(np.log(probabilities))
        parents[parent_id] = parent
        parent["frozen_baseline_logits"] = np.empty_like(probabilities)
        pure = [row for row in source if row["role"] == "pure_source_reference"]
        if len(pure) != 96 or any(row["observed_reference_class"] is not None
                                for row in source if row["role"] != "pure_source_reference"):
            raise ValueError("Only the96 observed pure-reference labels may enter model context")
        reference_ids = torch.tensor([row["sample_index"] for row in pure])
        reference_classes = torch.tensor([row["observed_reference_class"] for row in pure])
        namespace = parent_id + ":" + tree_ref["sha256"]
        for tile_id in range(8):
            left = max(tile_id * cfg["tile_bp"], float(parent["edges"][0]))
            right = min((tile_id + 1) * cfg["tile_bp"], float(parent["edges"][-1]))
            indexes = np.flatnonzero((parent["positions"] >= left) & (parent["positions"] < right))
            if not len(indexes):
                raise ValueError("All eight original physical tiles must retain their readouts")
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
            intervals = torch.tensor(intervals, dtype=torch.float64)
            bank = prepare_reference_dependence_bank_v1(
                trees, intervals, torch.from_numpy(parent["positions"][indexes]),
                query_ids, reference_ids, reference_classes, census=cfg["census"],
                sample_identity_namespace=parent_id + ":" + source_ref["sha256"],
                **cfg["geometry"])
            if bank.metadata["younger_reference_count"] != 84 or bank.metadata["older_reference_count"] != 12:
                raise ValueError("Original84 younger/12 older pure-reference policy required")
            with torch.no_grad():
                frozen = teacher(parent["unaries"][indexes].to(device)).detach().cpu()
            parent["frozen_baseline_logits"][indexes] = frozen.numpy()
            identity = f"{parent_id}:{tile_id}"
            cached[identity] = dict(parent_id=parent_id, indexes=indexes, context=bank, baseline=frozen)
            diagnostics.append(dict(identity=identity, split=split,
                native_identity_namespace=namespace, emitter_genealogy="Relate",
                trace_genealogy="independent_observed_tsinfer", **bank.metadata))
            (train_ids if split == "train" else val_ids.setdefault(parent_id, [])).append(identity)
            print(json.dumps({"stage": "reference_dependence_prepare", "identity": identity,
                              "vertices": bank.metadata["total_vertices"], "metadata": bank.metadata}), flush=True)
            resources_v1(spec, storage, lease)
        del ts
    if len(cached) != 32 or len(train_ids) != 16 or len(val_ids) != 2:
        raise ValueError("Original32 tiles,16 TRAIN tiles and two VAL parents required")
    output.save("geometry.json", {"tiles": diagnostics, "source_samples": "observed_R96_proxy_contrasts_only",
        "hybrid_condition": cfg["condition"], "frozen_skip": "exact_selected_H025_smoothing_original_grid_per250kb_tile",
        "TRAIN_emitter_predictions": "in_sample_not_out_of_fold",
        "query_or_context_ancestry_inputs": False, "targets_opened_at_geometry_seal": False})
    targets = {}
    for parent_id, parent in parents.items():
        if parent["split"] == "train":
            target = fit.ExternalTargetV1(parent, bindings["parents"][parent_id], "TRAIN_loss", reader)
            targets[parent_id] = target.target_mass(parent, helpers)
    del teacher, saved
    return cached, parents, train_ids, val_ids, targets


def seal_v1(output, arm, epoch, parent, logits):
    """Unchanged scorer inputs, probability-only compact seals without logits."""
    probabilities = fit.torch.from_numpy(logits).softmax(-1).numpy()
    return output.save(f'{arm}.epoch_{epoch:02d}.{parent["parent_id"]}.npz', {
        "C4_probabilities": probabilities, "positions_bp": parent["positions"],
        "bin_edges_bp": parent["edges"], "retained_fixed100bp_indexes": parent["retained_indexes"],
        "query_sample_indexes": parent["query_slots"]}, "npz")


def strong_reference_v1(spec, parents, reader):
    """Read already-selected H026 seals/metrics without a new fit or selection."""
    completed=reader.decode_json(spec["refs"]["strong_reference_completed"])
    selected=completed["models"]["native_reference"]["selected"]
    cells=selected["cells"]
    if {cell["parent_id"] for cell in cells} != {p for p,v in parents.items() if v["split"]=="val"}:
        raise ValueError("Frozen H026 comparator must cover both original VAL parents")
    for cell in cells:
        parent=parents[cell["parent_id"]]
        with fit.np.load(io.BytesIO(reader.read(cell["prediction_seal"])),allow_pickle=False) as archive:
            checks={"positions_bp":parent["positions"],"bin_edges_bp":parent["edges"],
                "retained_fixed100bp_indexes":parent["retained_indexes"],
                "query_sample_indexes":parent["query_slots"]}
            p=archive["C4_probabilities"]
            if (any(not fit.np.array_equal(archive[k],v) for k,v in checks.items())
                    or p.shape!=(len(parent["positions"]),120,4)
                    or not fit.np.isfinite(p).all() or fit.np.any(p<0)
                    or not fit.np.allclose(p.sum(-1),1,atol=1e-6,rtol=1e-5)):
                raise ValueError("Frozen H026 native reference seal has different support/query identities")
    return dict(arm="frozen_H026_native_reference",epoch=selected["epoch"],
        checkpoint=selected["checkpoint"],cells=cells,equal_parent_metrics=selected["equal_parent_metrics"],
        completed_ref=spec["refs"]["strong_reference_completed"],refitted=False)


def pending_first_moment_reference_v1(spec, cfg):
    """Bind prospective final static comparison without reading live outputs."""
    return dict(arm="frozen_H030_static",status="pending_final_selected_comparison",
        input_ref=spec["refs"]["first_moment_reference_input"],
        output_directory=cfg["first_moment_reference_output_directory"],
        config_ref=cfg["first_moment_reference_config"],selected_epoch=None,
        refitted=False,warmstarted=False,live_checkpoint_or_interim_report_read=False,
        completion_policy="H032_fit_can_complete;separate_bound_metadata_summary_after_both_complete")


def run_v1(record):
    spec, cfg, base, baseline_cfg, emitter, bindings, reader, storage = inputs_v1(record)
    from .biological_candidate import configure_fit
    from .biological_gst_control_v1 import exact_span_loss
    from .reference_dependence_gnn_v1 import ReferenceDependenceGNNV1
    torch, np = fit.torch, fit.np
    from .project_budget_v9_consumer_retry_v3 import PilotOutputV9
    output = PilotOutputV9(spec["output_directory"], storage, reader)
    from .project_budget_v9_consumer_retry_v3 import budget_check_v9
    qualification_allowance = 32000000
    before_qualification = sum(max(path.stat().st_size,path.stat().st_blocks*512)+8192
        for path in output.path.rglob("*") if path.is_file())
    if before_qualification + qualification_allowance > output.limit:
        raise ValueError("Bounded CUDA-qualification allowance exceeds output cap")
    budget_check_v9(storage, boundary=True, required_write_bytes=qualification_allowance)
    try:
        device, runtime, lease = configure_fit("cuda", base, output.path)
    finally:
        budget_check_v9(storage, boundary=True)
    after_qualification = sum(max(path.stat().st_size,path.stat().st_blocks*512)+8192
        for path in output.path.rglob("*") if path.is_file())
    if after_qualification - before_qualification > qualification_allowance:
        raise ValueError("CUDA qualification exceeded its32MB planned allowance")
    output.used = sum(max(path.stat().st_size, path.stat().st_blocks * 512) + 8192
                      for path in output.path.rglob("*") if path.is_file())
    helpers = fit.metric_helpers_v1()
    receipt = dict(schema="reference_dependence_gnn_evidence_v1", hypothesis_id="H032",
        status="running", input_ref=record, config_ref=spec["refs"]["config"], seed=101, dtype="float32",
        epochs=10, optimizer_steps_per_arm=160, condition=cfg["condition"], runtime=runtime,
        frozen_baseline_checkpoint=spec["refs"]["frozen_baseline_checkpoint"],
        graph_context_only=True, local_recalibration_path=False, ordered_cross_tree_reference_route_tested=False, whole_tree_learned=True, donor_dependence_hypothesis=True, recurrent_paths_executed=False,
        development_only=True, TEST_parents=0, models={},
        runtime_recovery=cfg["runtime_recovery"], scientific_model_changed=False)
    output.save("started.json", receipt)
    start = time.monotonic()
    try:
        cached, parents, train_ids, val_ids, targets = prepare_v1(
            spec, cfg, baseline_cfg, emitter, bindings, reader, helpers, output, storage, lease, device)
        strong_reference = strong_reference_v1(spec, parents, reader)
        output.save("frozen_H026_native_reference.metrics.json", strong_reference)
        pending_reference = pending_first_moment_reference_v1(spec,cfg)
        receipt["first_moment_reference"] = pending_reference
        output.save("pending_H030_static_comparison.json",pending_reference)
        torch.manual_seed(101)
        initial = ReferenceDependenceGNNV1(**cfg["model"]).to(device=device, dtype=torch.float32)
        states = {arm: dict(model=deepcopy(initial), steps=0, best=float("inf"), best_epoch=None,
                            current=[], retained_best=[], history=[],
                            initial={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}) for arm in ARMS}
        for state in states.values():
            model = state["model"]
            # Preserve the allocated H030 template but fit only this static route.
            for name, parameter in model.named_parameters():
                active = (name != "raw_ell" and not name.startswith(("kernel_head.",
                    "carrier_gru.", "value.", "key.", "retention.", "write_gate.", "decoder_edge.")))
                parameter.requires_grad_(active)
            state["optimizer"] = torch.optim.AdamW([
                {"params": [p for name,p in model.named_parameters() if p.requires_grad and name != "raw_cost"],
                 "weight_decay": cfg["weight_decay"]},
                {"params": [model.raw_cost], "weight_decay": 0.}], lr=cfg["learning_rate"])
        # Every shared encoder/decoder/gate/head starts bit-identically across arms.
        if any(any(not torch.equal(value, states[ARMS[0]]["initial"][key])
                       for key,value in state["initial"].items()) for state in states.values()):
            raise ValueError("Matched controls lost their shared initialization")
        baseline_seals = [(parent_id, seal_v1(output, "frozen_genomic_smoothing", 0,
            parents[parent_id], parents[parent_id]["frozen_baseline_logits"])) for parent_id in val_ids]
        baseline_score = None
        first_tile_diagnostic = None
        for epoch in range(1, 11):
            epoch_start = time.monotonic()
            # Prune only superseded non-best probability seals before new VAL
            # seals, bounding raw peak below2GB. Last/best checkpoints remain.
            for state in states.values():
                best_paths = {r["path"] for r in state["retained_best"]}
                kept = []
                for product in state["current"]:
                    if product["path"].endswith(".npz") and product["path"] not in best_paths:
                        output.discard(product)
                    else:
                        kept.append(product)
                state["current"] = kept
            resources_v1(spec, storage, lease)
            order = list(range(16)); random.Random(101 + epoch).shuffle(order)
            losses = {arm: [] for arm in ARMS}
            for index in order:
                tile = cached[train_ids[index]]
                frozen = tile["baseline"].to(device)
                reference_bank = tile["context"].to(device)
                mass = torch.as_tensor(targets[tile["parent_id"]][tile["indexes"]], dtype=torch.float32, device=device)
                qualifying = first_tile_diagnostic is None
                diagnostic = dict(identity=train_ids[index], epoch=epoch,
                    holds=tile["context"].metadata["holds"],
                    native_vertices=tile["context"].metadata["native_vertices"],
                    augmented_vertices=tile["context"].metadata["total_vertices"],
                    preserves_first_real_optimizer_update=True, arms={}) if qualifying else None
                for arm in ARMS:
                    lightweight_resources_v1(spec, lease)
                    state = states[arm]; model = state["model"]; model.train()
                    state["optimizer"].zero_grad(set_to_none=True)
                    if qualifying:
                        torch.cuda.synchronize(device)
                        forward_start = time.monotonic()
                    logits = model(reference_bank, frozen, control=arm)
                    if qualifying:
                        torch.cuda.synchronize(device)
                        forward_seconds = time.monotonic() - forward_start
                    loss = exact_span_loss(logits, mass)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite exact-span TRAIN loss")
                    if qualifying:
                        torch.cuda.synchronize(device)
                        backward_start = time.monotonic()
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
                    if qualifying:
                        torch.cuda.synchronize(device)
                        backward_seconds = time.monotonic() - backward_start
                    state["optimizer"].step(); state["steps"] += 1
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                        raise ValueError("Nonfinite fitted reference-trace parameter")
                    losses[arm].append(float(loss.detach()))
                    if qualifying:
                        diagnostic["arms"][arm] = dict(
                            forward_seconds=forward_seconds, backward_including_clip_seconds=backward_seconds,
                            finite_loss=True, finite_gradient_norm=bool(torch.isfinite(gradient_norm)),
                            unclipped_gradient_norm=float(gradient_norm.detach()),
                            TRAIN_exact_span_NLL=float(loss.detach()), optimizer_steps=state["steps"],
                            max_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                            parameter_counts=model.parameter_counts(arm))
                        print(json.dumps({"stage": "reference_dependence_first_training_tile_arm",
                            "identity": train_ids[index], "arm": arm, **diagnostic["arms"][arm]}), flush=True)
                lightweight_resources_v1(spec, lease)
                if qualifying:
                    first_tile_diagnostic = diagnostic
                    output.save("first_training_tile.json", diagnostic)
                    receipt["first_training_tile"] = diagnostic
                del reference_bank, frozen, mass
            seals = {arm: [] for arm in ARMS}
            for parent_id, identities in val_ids.items():
                parent = parents[parent_id]
                values = {arm: np.empty_like(parent["frozen_baseline_logits"]) for arm in ARMS}
                for identity in identities:
                    lightweight_resources_v1(spec, lease)
                    tile = cached[identity]
                    frozen = tile["baseline"].to(device)
                    reference_bank = tile["context"].to(device)
                    with torch.no_grad():
                        for arm in ARMS:
                            states[arm]["model"].eval()
                            logits = states[arm]["model"](reference_bank, frozen, control=arm)
                            if not bool(torch.isfinite(logits).all()):
                                raise ValueError("Nonfinite full-support context readout")
                            values[arm][tile["indexes"]] = logits.cpu().numpy()
                    del reference_bank, frozen
                    lightweight_resources_v1(spec, lease)
                for arm in ARMS:
                    seals[arm].append((parent_id, seal_v1(output, arm, epoch, parent, values[arm])))
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
            resources_v1(spec, storage, lease)
            print(json.dumps({"stage": "reference_dependence_epoch", "epoch": epoch, "steps_per_arm": epoch * 16,
                "elapsed_seconds": time.monotonic() - epoch_start,
                "last_context_diagnostics": {arm: states[arm]["model"].last_diagnostics for arm in ARMS},
                "primary_present64": {arm: states[arm]["history"][-1]["equal_parent_metrics"]["primary_present_day_64"]
                                      for arm in ARMS}}), flush=True)
        table = [dict(arm="frozen_genomic_smoothing",selected_epoch=0,
            **baseline_score["equal_parent_metrics"]["primary_present_day_64"]),
            dict(arm="frozen_H026_native_reference",selected_epoch=strong_reference["epoch"],
            **strong_reference["equal_parent_metrics"]["primary_present_day_64"])]
        for arm, state in states.items():
            if state["steps"] != 160:
                raise ValueError("Incomplete fixed160step one-seed fit")
            selected = state["selected"]
            saved = torch.load(io.BytesIO(reader.read(selected["checkpoint"])), map_location="cpu", weights_only=True)
            delta = max(float((value-state["initial"][key]).abs().max())
                for key,value in saved["model_state"].items())
            receipt["models"][arm] = dict(selected=selected, steps=160, parameter_count=state["model"].parameter_counts(arm),
                selected_parameter_max_delta=delta,
                last_context_diagnostics=state["model"].last_diagnostics,
                retained_records={r["path"]: r for r in state["current"] + state["retained_best"]})
            table.append(dict(arm=arm, selected_epoch=state["best_epoch"],
                              **selected["equal_parent_metrics"]["primary_present_day_64"]))
        output.save("comparison.json", dict(rows=table, frozen_baseline=baseline_score, frozen_strong_reference=strong_reference,
            pending_first_moment_reference=pending_reference, final_H030_static_comparison_complete=False,
            condition=cfg["condition"], novelty_claim=False, ordered_cross_tree_reference_route_tested=False,
            dependence_claim_requires_beating_both_informed_controls=True, alignment_sham_executed=False))
        resources_v1(spec, storage, lease)
        reader.read(record)
        receipt.update(status="completed_one_seed_FP32_static_donor_support_Gram_fit_pending_H030_comparison",
                       comparison=table, elapsed_seconds=time.monotonic() - start, output_accounted_bytes=output.used,
                       new_three_arm_fit_complete=True, final_H030_static_comparison_complete=False)
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
        print(json.dumps({"status": "source_only_not_executed", "hypothesis_id": "H032",
            "implementation": {"path": SOURCE, "sha256": fit._hash(SOURCE)},
            "module": {"path": MODULE, "sha256": fit._hash(MODULE)},
            "config": {"path": CONFIG, "sha256": fit._hash(CONFIG)}, "seed": 101,
            "epochs": 10, "steps_per_arm": 160, "dtype": "float32"}, indent=2))
        return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Pinned root H032 input required")
    run_v1({"path": args.input, "sha256": args.expected_input_sha256})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

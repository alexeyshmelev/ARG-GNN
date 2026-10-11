"""H027 one-seed FP32 reference-driven persistence-trace pilot.

Frozen selected H025 genomic smoothing remains the exact per250kb-tile skip.
A centered reference-only residual with learned spatial/temporal rates is fitted.
The established operator is reused; this is an information-route hypothesis,
not a novel operator/GNN claim. Observed reference classes are proxy inputs.
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
SOURCE = "src/arggnn/train_reference_trace_residual_v1.py"
MODULE = "src/arggnn/reference_trace_residual_v1.py"
CONFIG = "configs/reference_trace_residual_v1.json"
ARMS = ("sample_only", "genuine_ancestor", "sham_ancestor")


def inputs_v1(record):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    lease = validate_lease(); apply_and_validate_title()
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "reference_trace_residual_input_v1"
            or spec.get("root_authorizer") != "root_agent" or spec.get("fit_authorized") is not True
            or spec.get("status") != "active" or spec.get("seed") != 101
            or spec.get("device") != "cuda" or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Active root H027 seed101 CUDA input required")
    fit._check_active_v1(spec); fit.require_logical_job_v1(lease, spec)
    refs = spec["refs"]
    cfg = reader.decode_json(refs["config"])
    if (refs["config"]["path"] != CONFIG or cfg["hypothesis_id"] != "H027"
            or cfg["arms"] != list(ARMS) or cfg["seeds"] != [101] or cfg["dtype"] != "float32"
            or spec.get("epochs", 10) != 10):
        raise ValueError("New H027 three-arm10epoch160step config required")
    for path in (SOURCE, MODULE):
        if spec["source_sha256"].get(path) != fit._hash(path):
            raise ValueError("Root input must pin both NEW H027 implementations")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    for ref in cfg["helper_refs"].values():
        reader.read(ref)
    base = reader.decode_json(cfg["base_fit_config"])
    if cfg["metrics"] != dict(base["metrics"], aggregation="unweighted_parent_mean_one_seed_no_probability_ensemble"):
        raise ValueError("Original scorer policy with explicit single-fit aggregation required")
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
        raise ValueError("Separate root v4 output reservation of at most2GB required")
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
    return spec, cfg, base, baseline_cfg, emitter, bindings, reader, storage


def prepare_v1(spec, cfg, baseline_cfg, emitter, bindings, reader, helpers, output, storage, lease, device):
    import tempfile
    import tskit
    from .persistence_trace_kernel_v1 import MetricTreeHoldV1
    from .reference_trace_residual_v1 import prepare_reference_trace_bank_v1
    np, torch = fit.np, fit.torch
    # Existing H025 preparation supplies the exact observed joins/grid/targets.
    # Its persistence operator is reused; query-source signals are discarded.
    observed_output = SimpleNamespace(path=output.path,
        save=lambda name, value, kind="json": output.save("H025_observation_" + name, value, kind))
    cached, parents, train_ids, val_ids, targets = h025.prepare_v1(
        spec, baseline_cfg, emitter, bindings, reader, helpers, observed_output, storage, lease)
    factory = h025.models_v1(baseline_cfg, torch.device("cpu"))
    teacher = factory["genomic_smoothing"]["model"]
    del factory
    saved = torch.load(io.BytesIO(reader.read(spec["refs"]["frozen_baseline_checkpoint"])),
                       map_location="cpu", weights_only=True)
    if saved["arm"] != "genomic_smoothing" or saved["epoch"] != 10:
        raise ValueError("Frozen smoother checkpoint identity differs")
    teacher.load_state_dict(saved["model_state"], strict=True)
    teacher.to(device=device, dtype=torch.float32).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    diagnostics = []
    for parent_id, parent in parents.items():
        refs = spec["observed_parents"][parent_id]
        prepared = reader.decode_json(refs["prepared_manifest"])
        entry = reader.decode_json(refs["observed_parent_entry"])
        inferred = h025.roster_v1(reader.decode_json(refs["observed_roster"]))
        samples = parent["samples"]
        native_samples = [int(inferred[(row["sample_id"], row["haplotype"])]["inferred_tree_node_id"])
                          for row in samples]
        reference_rows = [row for row in samples if row["role"] == "pure_source_reference"]
        if (len(reference_rows) != 96 or any(row["observed_reference_class"] is not None
                for row in samples if row["role"] != "pure_source_reference")):
            raise ValueError("Only observed pure-reference labels may enter the context")
        reference_ids = torch.tensor([row["sample_index"] for row in reference_rows])
        reference_classes = torch.tensor([row["observed_reference_class"] for row in reference_rows])
        query_rows = [row for row in samples if row["role"] == "query"]
        query_ids = torch.tensor([row["sample_index"] for row in query_rows])
        tree_ref = {key: entry["inferred_tree"][key] for key in ("path", "sha256")}
        with tempfile.NamedTemporaryFile(dir=output.path / "tmp", suffix=".observed.trees") as file:
            file.write(reader.read(tree_ref)); file.flush()
            ts = tskit.load(file.name)
        offset = prepared["coordinate_contract"]["local_to_original_offset_bp"]
        namespace = parent_id + ":" + tree_ref["sha256"]
        baseline = np.empty_like(parent["probabilities"])
        seen = np.zeros(len(parent["positions"]), dtype=bool)
        for tile_id in range(8):
            h025.resources_v1(spec, storage, lease)
            identity = f"{parent_id}:{tile_id}"
            old = cached[identity]
            indexes = old["indexes"]
            with torch.no_grad():
                frozen = teacher(old["direct"].to(device)).detach().cpu()
            baseline[indexes] = frozen.numpy(); seen[indexes] = True
            left = max(tile_id * cfg["tile_bp"], float(parent["edges"][0]))
            right = min((tile_id + 1) * cfg["tile_bp"], float(parent["edges"][-1]))
            trees, intervals = [], []
            actual_ids = torch.tensor([row["sample_index"] for row in samples])
            for tree in ts.trees():
                lo, hi = max(left, tree.interval.left + offset), min(right, tree.interval.right + offset)
                if hi <= lo:
                    continue
                union = set()
                for sample in native_samples:
                    at = sample
                    while at != -1 and at not in union:
                        union.add(at); at = tree.parent(at)
                packed = [int(node) for node in tree.nodes(order="postorder") if node in union]
                local = {node: index for index, node in enumerate(packed)}
                trees.append(MetricTreeHoldV1(
                    torch.tensor([-1 if tree.parent(node) == -1 else local[tree.parent(node)] for node in packed]),
                    torch.tensor([tree.time(node) for node in packed], dtype=torch.float64),
                    torch.tensor(packed), namespace,
                    torch.tensor([local[node] for node in native_samples]), actual_ids))
                intervals.append([lo, hi])
            context = prepare_reference_trace_bank_v1(
                old["bank"], trees, torch.tensor(intervals, dtype=torch.float64),
                query_ids, reference_ids, reference_classes, census=cfg["census"])
            if (context.metadata["younger_reference_count"] != 84
                    or context.metadata["older_reference_count"] != 12):
                raise ValueError("All arms require the same84 younger and12 older observed reference proxies")
            cached[identity] = dict(parent_id=parent_id, indexes=indexes, baseline=frozen, context=context)
            diagnostics.append(dict(identity=identity, split=parent["split"], **context.metadata))
            del old
            print(json.dumps({"stage": "reference_trace_prepare", "identity": identity,
                "reference_sources": len(context.reference_nodes),
                "states": context.operator.num_states}), flush=True)
        if not seen.all():
            raise ValueError("Frozen selected smoother lacks an original readout")
        parent["frozen_baseline_logits"] = baseline
        del ts
    del teacher, saved
    output.save("reference_trace_sources.json", {"tiles": diagnostics,
        "frozen_baseline_checkpoint": spec["refs"]["frozen_baseline_checkpoint"],
        "reference_labels": "observed_proxies_not_census_truth",
        "source_information": "observed_R96_proxy_labels_only",
        "operator": "immutable_established_H025_whole_tree_persistence",
        "ordered_cross_tree_reference_route_tested": True})
    return cached, parents, train_ids, val_ids, targets


def run_v1(record):
    spec, cfg, base, baseline_cfg, emitter, bindings, reader, storage = inputs_v1(record)
    from .biological_candidate import configure_fit
    from .biological_gst_control_v1 import exact_span_loss
    from .reference_trace_residual_v1 import ReferenceTraceResidualV1
    torch, np = fit.torch, fit.np
    output = h025.PilotOutputV1(spec["output_directory"], storage, reader)
    device, runtime, lease = configure_fit("cuda", base, output.path)
    output.used = sum(max(path.stat().st_size, path.stat().st_blocks * 512) + 8192
                      for path in output.path.rglob("*") if path.is_file())
    helpers = fit.metric_helpers_v1()
    receipt = dict(schema="reference_trace_residual_evidence_v1", hypothesis_id="H027",
        status="running", input_ref=record, config_ref=spec["refs"]["config"], seed=101, dtype="float32",
        epochs=10, optimizer_steps_per_arm=160, condition=cfg["condition"], runtime=runtime,
        frozen_baseline_checkpoint=spec["refs"]["frozen_baseline_checkpoint"],
        graph_context_only=True, local_recalibration_path=False, ordered_cross_tree_reference_route_tested=True,
        development_only=True, TEST_parents=0, models={})
    output.save("started.json", receipt)
    start = time.monotonic()
    try:
        cached, parents, train_ids, val_ids, targets = prepare_v1(
            spec, cfg, baseline_cfg, emitter, bindings, reader, helpers, output, storage, lease, device)
        torch.manual_seed(101)
        initial = ReferenceTraceResidualV1(**cfg["model"]).to(device=device, dtype=torch.float32)
        states = {arm: dict(model=deepcopy(initial), steps=0, best=float("inf"), best_epoch=None,
                            current=[], retained_best=[], history=[],
                            initial={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}) for arm in ARMS}
        for state in states.values():
            model=state["model"]
            state["optimizer"] = torch.optim.AdamW([
                {"params":model.context_head.parameters(),"weight_decay":cfg["weight_decay"]},
                {"params":[model.raw_spatial,model.raw_temporal],"weight_decay":0.}],
                lr=cfg["learning_rate"])
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
                reference_bank = tile["context"].to(device)
                mass = torch.as_tensor(targets[tile["parent_id"]][tile["indexes"]], dtype=torch.float32, device=device)
                for arm in ARMS:
                    state = states[arm]; model = state["model"]; model.train()
                    state["optimizer"].zero_grad(set_to_none=True)
                    logits = model(reference_bank, frozen, control=arm)
                    loss = exact_span_loss(logits, mass)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite exact-span TRAIN loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
                    state["optimizer"].step(); state["steps"] += 1
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                        raise ValueError("Nonfinite fitted reference-trace parameter")
                    losses[arm].append(float(loss.detach()))
                del reference_bank, frozen, mass
            seals = {arm: [] for arm in ARMS}
            for parent_id, identities in val_ids.items():
                parent = parents[parent_id]
                values = {arm: np.empty_like(parent["frozen_baseline_logits"]) for arm in ARMS}
                for identity in identities:
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
            h025.resources_v1(spec, storage, lease)
            print(json.dumps({"stage": "reference_trace_epoch", "epoch": epoch, "steps_per_arm": epoch * 16,
                "elapsed_seconds": time.monotonic() - epoch_start,
                "last_readout_reference_mass_min": {arm: states[arm]["model"].last_mass_min for arm in ARMS},
                "primary_present64": {arm: states[arm]["history"][-1]["equal_parent_metrics"]["primary_present_day_64"]
                                      for arm in ARMS}}), flush=True)
        table = [dict(arm="frozen_genomic_smoothing", selected_epoch=0,
                      **baseline_score["equal_parent_metrics"]["primary_present_day_64"])]
        for arm, state in states.items():
            if state["steps"] != 160:
                raise ValueError("Incomplete fixed160step one-seed fit")
            selected = state["selected"]
            saved = torch.load(io.BytesIO(reader.read(selected["checkpoint"])), map_location="cpu", weights_only=True)
            delta = max(float((value-state["initial"][key]).abs().max())
                for key,value in saved["model_state"].items())
            receipt["models"][arm] = dict(selected=selected, steps=160, parameter_count=48,
                selected_parameter_max_delta=delta,
                last_finite_positive_reference_mass=state["model"].last_mass_min,
                retained_records={r["path"]: r for r in state["current"] + state["retained_best"]})
            table.append(dict(arm=arm, selected_epoch=state["best_epoch"],
                              **selected["equal_parent_metrics"]["primary_present_day_64"]))
        output.save("comparison.json", dict(rows=table, frozen_baseline=baseline_score,
            condition=cfg["condition"], novelty_claim=False, ordered_cross_tree_reference_route_tested=True))
        h025.resources_v1(spec, storage, lease)
        reader.read(record)
        receipt.update(status="completed_one_seed_FP32_reference_driven_persistence_pilot",
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
        print(json.dumps({"status": "source_only_not_executed", "hypothesis_id": "H027",
            "implementation": {"path": SOURCE, "sha256": fit._hash(SOURCE)},
            "config": {"path": CONFIG, "sha256": fit._hash(CONFIG)}, "seed": 101,
            "epochs": 10, "steps_per_arm": 160, "dtype": "float32"}, indent=2))
        return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Pinned root H027 input required")
    run_v1({"path": args.input, "sha256": args.expected_input_sha256})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""H036 paired separate-worker FP32 full-parent direct ancestry discovery.

No teacher/emitter arrays. Native observations are prepared and sealed before
TRAIN targets. Both VAL parents are sealed before unchanged external scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import time

from . import census_tsinfer_fit_v1 as fit
from . import train_persistence_trace_pilot_v1 as h025

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/train_mutation_routed_whole_tree_gnn_v1.py"
MODULE = "src/arggnn/mutation_routed_whole_tree_gnn_v1.py"
PREPROCESSOR = "src/arggnn/mutation_routed_tree_observations_v1.py"
LOADER = "src/arggnn/mutation_routed_parent_loader_v1.py"
CONFIG = "configs/mutation_routed_whole_tree_gnn_v1.json"
ARMS = ("aligned", "shuffled")
SELECTION = "earliest_minimum_equal_parent_development_VAL_present64_exact_span_raw_probability_NLL"


def science_v1():
    """Initialize unchanged score helpers, without the old Census/teacher factory."""
    import numpy as np
    import torch
    fit.np, fit.torch = np, torch
    return np, torch


def inputs_v1(record):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    lease = validate_lease()
    apply_and_validate_title()
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "mutation_routed_whole_tree_gnn_input_v1"
            or spec.get("root_authorizer") != "root_agent" or spec.get("fit_authorized") is not True
            or spec.get("status") != "active" or spec.get("seed") != 101
            or spec.get("device") != "cuda" or spec.get("arm") not in ARMS
            or spec.get("epochs", 10) != 10 or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Active root H036 single-arm seed101 FP32 CUDA input required")
    fit._check_active_v1(spec)
    fit.require_logical_job_v1(lease, spec)
    refs = spec["refs"]
    cfg = reader.decode_json(refs["config"])
    if (refs["config"]["path"] != CONFIG or cfg.get("hypothesis_id") != "H036"
            or cfg.get("execution_guard_ready") is not True or cfg["arms"] != list(ARMS)
            or cfg["seeds"] != [101] or cfg["dtype"] != "float32"
            or cfg["epochs"] != 10 or cfg["optimizer_steps_per_arm"] != 20
            or cfg["output_cap_bytes"] != 1000000000 or cfg["selection"] != SELECTION
            or cfg["learning_rate"] != .001 or cfg["weight_decay"] != .0001
            or cfg["gradient_clip_norm"] != 1. or cfg["model"] != {
                "hidden_dim": 32, "expected_haplotypes": 280,
                "site_chunk_size": 256, "checkpoint_sites": True}
            or cfg["parameter_count"] != 29764):
        raise ValueError("Frozen H036 initial20-step full-parent protocol required")
    for name in ("max_cpu_rss_bytes", "max_cuda_reserved_bytes"):
        if type(spec.get(name)) is not int or spec[name] <= 0:
            raise ValueError("Positive root CPU/CUDA bounds required")
    for path in (SOURCE, MODULE, PREPROCESSOR, LOADER):
        if spec["source_sha256"].get(path) != fit._hash(path):
            raise ValueError("Root source closure must pin driver/model/preprocessor/loader")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    for reference in cfg["helper_refs"].values():
        reader.read(reference)
    if cfg["prototype_ref"] != {"path": MODULE, "sha256": fit._hash(MODULE)}:
        raise ValueError("Frozen neural prototype changed")
    if cfg["preprocessor_ref"] != {"path": PREPROCESSOR, "sha256": fit._hash(PREPROCESSOR)}:
        raise ValueError("Frozen observation preprocessing changed")
    base = reader.decode_json(cfg["base_fit_config"])
    if cfg["metrics"] != dict(base["metrics"], aggregation="unweighted_parent_mean_one_seed_no_probability_ensemble"):
        raise ValueError("Unchanged scorer with explicit single-fit parent aggregation required")
    reader.read(refs["runtime"])
    if set(spec["observed_parents"]) != set(fit.PARENTS):
        raise ValueError("Exactly the original two TRAIN and two VAL parents required")
    # Bind truth identities/hashes from JSON metadata only. No truth array opens.
    bindings = reader.decode_json(refs["target_bindings"])
    manifest = reader.decode_json(base["external_target_manifest"])
    rows = {r["parent_id"]: r for r in manifest["replicates"]}
    if set(bindings["parents"]) != set(fit.PARENTS):
        raise ValueError("Original four-parent target bindings required")
    for parent_id, binding in bindings["parents"].items():
        expected = {"path": f"data/biological_paths_execution_pilot_v1/{parent_id}/truth/full_truth_intervals.npz",
                    "sha256": rows[parent_id]["files"]["truth/full_truth_intervals.npz"]["sha256"]}
        if (binding["truth_npz"] != expected or binding["split"] != rows[parent_id]["split"]
                or len(binding["query_sample_indexes"]) != 120
                or len(binding["query_raw_node_ids"]) != 120):
            raise ValueError("Original external C4 target/support binding changed")
    storage = reader.decode_json(refs["storage"])
    from .project_storage_full_reservation_v1 import validate_storage_v9
    launcher = cfg["helper_refs"]["actual_full_storage_launcher"]
    if launcher not in storage["reservation"]["consumer_bindings"]:
        raise ValueError("Actual full-storage launcher must be reserved")
    validate_storage_v9(spec, cfg, storage, source_ref={"path": SOURCE, "sha256": spec["source_sha256"][SOURCE]},
                        config_ref=refs["config"], runtime_ref=refs["runtime"])
    science_v1()
    return spec, cfg, base, bindings, reader, storage


def lightweight_resources_v1(spec, lease):
    from .biological_candidate import check_fit_lease
    fit._check_active_v1(spec)
    check_fit_lease(lease)
    fit.require_logical_job_v1(lease, spec)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > spec["max_cpu_rss_bytes"]:
        raise ValueError("H036 CPU RSS cap exceeded")
    if fit.torch.cuda.max_memory_reserved() > spec["max_cuda_reserved_bytes"]:
        raise ValueError("H036 CUDA reservation cap exceeded")


def resources_v1(spec, storage, lease):
    from .project_storage_full_reservation_v1 import budget_check_v9
    lightweight_resources_v1(spec, lease)
    budget_check_v9(storage, boundary=True)


def state_fingerprint_v1(model):
    """Content hash independent of checkpoint serialization or worker identity."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_state_v1(cfg, device):
    from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeGNNV1
    np, torch = fit.np, fit.torch
    random.seed(101); np.random.seed(101); torch.manual_seed(101)
    # Construct on CPU so paired workers have the same seed101 initial bytes.
    model = MutationRoutedWholeTreeGNNV1(**cfg["model"]).to(device=device, dtype=torch.float32)
    if sum(p.numel() for p in model.parameters()) != cfg["parameter_count"]:
        raise ValueError("Frozen parameter count changed")
    initial_hash = state_fingerprint_v1(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"], betas=(.9, .999), eps=1e-8)
    return dict(model=model, optimizer=optimizer, steps=0, best=float("inf"),
        best_epoch=None, selected=None, history=[], current=[], retained_best=[],
        initial_sha256=initial_hash)


def prepare_v1(spec, bindings, reader, output, helpers):
    from .mutation_routed_parent_loader_v1 import prepare_parents_v1
    from .mutation_routed_whole_tree_gnn_v1 import validate_inputs_v1
    bundle = prepare_parents_v1(spec["observed_parents"], reader,
                                scratch_directory=output.path/"observation_tmp")
    items = bundle["parents"]
    if set(items) != set(fit.PARENTS):
        raise ValueError("Observation bundle changed the frozen parents")
    train_ids, val_ids, parents, banks, receipts = [], [], {}, {}, {}
    for parent_id in fit.PARENTS:
        item = items[parent_id]
        bank, parent = item["model_inputs"], item["scoring"]
        validate_inputs_v1(bank)
        binding = bindings["parents"][parent_id]
        if (parent["parent_id"] != parent_id or parent["split"] != binding["split"]
                or parent["query_slots"].tolist() != binding["query_sample_indexes"]
                or parent["raw_query_nodes"].tolist() != binding["query_raw_node_ids"]
                or not fit.np.array_equal(bank.query_column.numpy(), parent["query_slots"])
                or len(bank.readout_hold) != len(parent["positions"])
                or not fit.np.array_equal(parent["positions"], 50.+100.*parent["retained_indexes"])
                or parent["positions"].shape != (len(parent["edges"])-1,)):
            raise ValueError("Original actual-query identity or exact physical scoring support changed")
        parents[parent_id], banks[parent_id] = parent, bank
        receipts[parent_id] = item["observations_receipt"]
        (train_ids if parent["split"] == "train" else val_ids).append(parent_id)
    if len(train_ids) != 2 or len(val_ids) != 2:
        raise ValueError("Exactly two complete TRAIN/VAL parents required")
    # Durable all-parent observation/preparation seal precedes TRAIN truth access.
    seal = output.save("observations_features_ready.json", dict(
        schema="mutation_routed_all_parent_observation_seal_v1",
        all_four_parents_complete=True, targets_opened=False, teacher_used=False,
        original_native_holds_preserved=True, full_parent_history_detached=False,
        parents=receipts, loader_receipt=bundle["receipt"],
        sham_bin_edges=bundle["sham_bin_edges"]))
    targets = {}
    for parent_id in train_ids:
        target = fit.ExternalTargetV1(parents[parent_id], bindings["parents"][parent_id], "TRAIN_loss", reader)
        targets[parent_id] = target.target_mass(parents[parent_id], helpers)
    output.save("TRAIN_targets_ready.json", dict(observations_features_seal=seal,
        query_truth_features=False, target_purpose="TRAIN_loss_only",
        parents={p: {"target_ref": bindings["parents"][p]["truth_npz"],
                     "shape": list(targets[p].shape)} for p in train_ids}))
    return banks, parents, train_ids, val_ids, targets, seal


def train_parent_v1(state, bank, mass, cfg, device, *, qualifying=False):
    """Exactly one optimizer update from the complete biological parent."""
    from .biological_gst_control_v1 import exact_span_loss
    torch = fit.torch
    model = state["model"]; model.train()
    state["optimizer"].zero_grad(set_to_none=True)
    if qualifying:
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        before = {"allocated": int(torch.cuda.memory_allocated(device)),
                  "reserved": int(torch.cuda.memory_reserved(device))}
    logits = model(bank, control=cfg["_arm"])
    if qualifying:
        torch.cuda.synchronize(device); forward_seconds = time.monotonic()-started
        backward_start = time.monotonic()
    target = torch.as_tensor(mass, dtype=torch.float32, device=device)
    loss = exact_span_loss(logits, target)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite exact-span full-parent TRAIN loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
    if qualifying:
        torch.cuda.synchronize(device); backward_seconds = time.monotonic()-backward_start
        update_start = time.monotonic()
    state["optimizer"].step(); state["steps"] += 1
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite fitted H036 parameter")
    row = {"TRAIN_exact_span_query120_NLL": float(loss.detach()),
           "unclipped_gradient_norm": float(norm.detach()), "optimizer_steps": state["steps"]}
    if qualifying:
        torch.cuda.synchronize(device)
        row.update(forward_seconds=forward_seconds, backward_loss_clip_seconds=backward_seconds,
            optimizer_seconds=time.monotonic()-update_start,
            before_cuda_allocated_bytes=before["allocated"], before_cuda_reserved_bytes=before["reserved"],
            max_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            max_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            max_cpu_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            native_holds=int(bank.sample_index.shape[0]), native_vertices=len(bank.parent),
            observed_haplotypes=int(bank.sample_index.shape[1]), original_sites=len(bank.genotype),
            readout_centers=len(bank.readout_hold), query_columns=len(bank.query_column),
            finite_loss=True, finite_clipped_gradients=True,
            actual_first_full_parent_step_counts_within20=True, dummy_update=False,
            full_parent_history_detached=False, artificial_tile_reset=False)
    return row


def seal_probabilities_v1(output, arm, epoch, parent, logits):
    torch, np = fit.torch, fit.np
    if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Finite FP32 raw logits required")
    probabilities = logits.softmax(-1).cpu().numpy()
    if (probabilities.shape != (len(parent["positions"]), len(parent["query_slots"]), 4)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(-1), 1., rtol=1e-5, atol=1e-6)):
        raise ValueError("Raw complete-support probabilities invalid without floors/renormalization")
    return output.save(f'{arm}.epoch_{epoch:02d}.{parent["parent_id"]}.npz', dict(
        C4_probabilities=probabilities, positions_bp=parent["positions"],
        bin_edges_bp=parent["edges"], retained_fixed100bp_indexes=parent["retained_indexes"],
        query_sample_indexes=parent["query_slots"]), "npz")


def selection_nll_v1(means):
    primary = means["primary_present_day_64"]
    return float("inf") if primary["NLL_is_infinite"] else primary["nll"]


def prune_nonbest_probability_seals_v1(output, state):
    best = {r["path"] for r in state["retained_best"]}
    kept = []
    for record in state["current"]:
        if record["path"].endswith(".npz") and record["path"] not in best:
            output.discard(record)
        else:
            kept.append(record)
    state["current"] = kept


def run_v1(record):
    spec, cfg, base, bindings, reader, storage = inputs_v1(record)
    from .biological_candidate import configure_fit
    from .project_storage_full_reservation_v1 import PilotOutputV9, budget_check_v9
    torch, np = fit.torch, fit.np
    cfg = dict(cfg, _arm=spec["arm"])
    arm = spec["arm"]
    output = PilotOutputV9(spec["output_directory"], storage, reader)
    qualification_allowance = 32000000
    budget_check_v9(storage, boundary=True, required_write_bytes=qualification_allowance)
    device, runtime, lease = configure_fit("cuda", base, output.path)
    output.used = sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192
                      for p in output.path.rglob("*") if p.is_file())
    if output.used > qualification_allowance or output.used > output.limit:
        raise ValueError("CUDA qualification exceeded its admitted32MB allowance")
    helpers = fit.metric_helpers_v1()
    receipt = dict(schema="mutation_routed_whole_tree_gnn_evidence_v1", hypothesis_id="H036",
        status="running", arm=arm, input_ref=record, config_ref=spec["refs"]["config"],
        seed=101, dtype="float32", epochs=10, optimizer_steps_per_arm=20,
        optimizer_step_unit="one_complete_biological_parent", runtime=runtime,
        checkpoint_selection=SELECTION, teacher_used=False, query_age_centering=False,
        zero_residual_head=False, whole_parent_history_detached=False,
        artificial_tile_reset=False, all_native_holds_and280samples=True,
        source_sha256=spec["source_sha256"], novelty_claim=False,
        development_only=True, TEST_parents=0, models={},
        comparison_limitation="same_underlying_physical_observations;additional_original_genotype_inputs_and_tsinfer_tsdate_genealogy_differ_from_Relate_native_ARGMix;not_matched_topology")
    output.save("started.json", receipt)
    started = time.monotonic()
    try:
        resources_v1(spec, storage, lease)
        banks, parents, train_ids, val_ids, targets, observation_seal = prepare_v1(
            spec, bindings, reader, output, helpers)
        receipt["observation_features_seal"] = observation_seal
        state = model_state_v1(cfg, device)
        receipt["initial_state_sha256"] = state["initial_sha256"]
        output.save("initialization.json", dict(arm=arm, seed=101,
            initial_state_sha256=state["initial_sha256"], parameters=cfg["parameter_count"],
            identical_seed_initialization_required_across_workers=True))
        first = None
        for epoch in range(1, 11):
            epoch_start = time.monotonic()
            prune_nonbest_probability_seals_v1(output, state)
            resources_v1(spec, storage, lease)
            order = list(train_ids); random.Random(101+epoch).shuffle(order)
            losses = []
            for parent_id in order:
                lightweight_resources_v1(spec, lease)
                bank = banks[parent_id].to(device)
                row = train_parent_v1(state, bank, targets[parent_id], cfg, device, qualifying=first is None)
                losses.append(row["TRAIN_exact_span_query120_NLL"])
                if first is None:
                    first = dict(parent_id=parent_id, epoch=epoch, arm=arm, **row)
                    output.save("first_training_parent.json", first)
                    receipt["first_training_parent"] = first
                    print(json.dumps({"stage":"H036_first_full_parent_update", **first}), flush=True)
                del bank
                lightweight_resources_v1(spec, lease)
            seals = []
            # Both VAL probabilities are durable before either VAL truth opens.
            for parent_id in val_ids:
                lightweight_resources_v1(spec, lease)
                bank = banks[parent_id].to(device)
                state["model"].eval()
                with torch.no_grad():
                    logits = state["model"](bank, control=arm)
                    seal = seal_probabilities_v1(output, arm, epoch, parents[parent_id], logits)
                seals.append((parent_id, seal))
                del bank, logits
            prediction_manifest = output.save(f"{arm}.epoch_{epoch:02d}.VAL_predictions_sealed.json",
                dict(epoch=epoch, arm=arm, all_VAL_predictions_sealed_before_scoring=True,
                     targets_opened_in_this_prediction_phase=False, seals=dict(seals)))
            cells, means = h025.score_v1(seals, parents, bindings, reader, helpers, cfg)
            nll = selection_nll_v1(means)
            previous = {r["path"]:r for r in state["current"]+state["retained_best"]}
            checkpoint = output.save(f"{arm}.epoch_{epoch:02d}.pt", dict(
                model_state={k:v.detach().cpu() for k,v in state["model"].state_dict().items()},
                optimizer_state=state["optimizer"].state_dict(), epoch=epoch, arm=arm,
                optimizer_steps=state["steps"], seed=101, input_ref=record,
                initial_state_sha256=state["initial_sha256"], teacher_used=False), "torch")
            state["current"] = [r for _,r in seals]+[checkpoint]
            row = dict(epoch=epoch, arm=arm, optimizer_steps=state["steps"],
                TRAIN_mean_full_parent_query120_NLL=float(np.mean(losses)),
                cells=cells, equal_parent_metrics=means, checkpoint=checkpoint,
                prediction_manifest=prediction_manifest)
            state["history"].append(row)
            if state["best_epoch"] is None or nll < state["best"]:
                state.update(best=nll, best_epoch=epoch, selected=row,
                             retained_best=list(state["current"]))
            row["selected_epoch"] = state["best_epoch"]
            output.save(f"{arm}.epoch_{epoch:02d}.metrics.json", row)
            keep = {r["path"] for r in state["current"]+state["retained_best"]}
            for path, product in previous.items():
                if path not in keep:
                    output.discard(product)
            resources_v1(spec, storage, lease)
            print(json.dumps(dict(stage="H036_epoch", arm=arm, epoch=epoch,
                optimizer_steps=state["steps"], elapsed_seconds=time.monotonic()-epoch_start,
                primary_present64=means["primary_present_day_64"])), flush=True)
        if state["steps"] != 20:
            raise ValueError("Initial10epoch/20full-parent-update fit incomplete")
        retained = {r["path"]:r for r in state["current"]+state["retained_best"]}
        receipt["models"][arm] = dict(selected=state["selected"], history=state["history"],
            steps=state["steps"], parameter_count=cfg["parameter_count"], retained_records=retained)
        receipt.update(status="completed_one_seed_FP32_direct_full_parent_initial20updates",
            elapsed_seconds=time.monotonic()-started, output_accounted_bytes=output.used,
            initial_budget_complete=True, continuation_executed=False,
            joint_paired_comparison_pending_other_worker=True)
        reader.read(record)
        resources_v1(spec, storage, lease)
    except Exception as error:
        receipt.update(status="failed_partial_preserved", error_type=type(error).__name__,
            error=str(error), elapsed_seconds=time.monotonic()-started,
            output_accounted_bytes=output.used)
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
        print(json.dumps(dict(status="source_only_not_executed", hypothesis_id="H036",
            implementation={"path":SOURCE,"sha256":fit._hash(SOURCE)},
            config={"path":CONFIG,"sha256":fit._hash(CONFIG)},
            input_schema="mutation_routed_whole_tree_gnn_input_v1", arms=list(ARMS),
            epochs=10, steps_per_arm=20, dtype="float32", teacher_used=False), indent=2))
        return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Pinned active root H036 arm input required")
    run_v1({"path":args.input,"sha256":args.expected_input_sha256})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

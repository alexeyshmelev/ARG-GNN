"""Source-only H037 PyG optimization continuation linked to H036 epoch80/160.

Same seed, learned weights, AdamW, TRAIN loss and physical scoring support.
Root owns registration, budget, admission, cache publication and launches.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields
import io
import json
import os
from pathlib import Path
import random
import time

from . import census_tsinfer_fit_v1 as fit
from . import train_mutation_routed_whole_tree_gnn_v1 as initial
from . import continue_mutation_routed_whole_tree_gnn_v1 as previous

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/continue_mutation_routed_pyg_gnn_v1.py"
CONFIG = "configs/mutation_routed_pyg_continue_v1.json"
PYG_MODEL = "src/arggnn/mutation_routed_pyg_gnn_v1.py"
PYG_DATASET = "src/arggnn/mutation_routed_pyg_dataset_v1.py"
CACHE = "src/arggnn/mutation_routed_observation_cache_v1.py"
FAST_PRIMARY = "src/arggnn/mutation_routed_fast_primary_v1.py"
EFFICIENT = "src/arggnn/mutation_routed_observation_efficiency_v1.py"
ARMS = initial.ARMS
PRIOR_HISTORY_EPOCHS = list(range(1, 11)) + list(range(12, 81, 4))
EVALUATION_EPOCHS = list(range(90, 201, 10))
SOURCE_CLOSURE = (SOURCE, PYG_MODEL, PYG_DATASET, CACHE, FAST_PRIMARY,
    initial.SOURCE, previous.SOURCE, initial.MODULE, initial.PREPROCESSOR,
    initial.LOADER, EFFICIENT)


def validate_config_v1(cfg, old_cfg):
    if (cfg.get("hypothesis_id") != "H037"
            or cfg.get("linked_hypothesis_id") != "H036"
            or cfg.get("kind") != "mutation_routed_pyg_continue_v1"
            or cfg.get("execution_guard_ready") is not True
            or cfg.get("arms") != list(ARMS) or cfg.get("seeds") != [101]
            or cfg.get("dtype") != "float32" or cfg.get("framework") != "torch_plus_pyg"
            or cfg.get("resume_epoch") != 80 or cfg.get("resume_updates_per_arm") != 160
            or cfg.get("additional_epochs") != 120 or cfg.get("additional_updates_per_arm") != 240
            or cfg.get("final_epoch") != 200 or cfg.get("total_updates_per_arm") != 400
            or cfg.get("evaluation_epochs") != EVALUATION_EPOCHS
            or cfg.get("output_cap_bytes") != 500000000
            or cfg.get("data_workers") != 2 or cfg.get("data_worker_start_method") != "spawn"
            or cfg.get("primary_score_backend") != "numpy"
            or type(cfg.get("resident_workspace_reserve_bytes")) is not int
            or cfg["resident_workspace_reserve_bytes"] <= 0
            or any(cfg.get(k) != old_cfg[k] for k in ("model", "parameter_count",
                "metrics", "selection", "learning_rate", "weight_decay",
                "gradient_clip_norm", "prototype_ref", "preprocessor_ref"))):
        raise ValueError("Frozen same-model H036 PyG epoch81to200/240-added-update protocol required")


def inputs_v1(record):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    lease = validate_lease()
    apply_and_validate_title()
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "mutation_routed_pyg_continue_input_v1"
            or spec.get("root_authorizer") != "root_agent"
            or spec.get("fit_authorized") is not True
            or spec.get("continuation_authorized") is not True
            or spec.get("pyg_authorized") is not True or spec.get("status") != "active"
            or spec.get("arm") not in ARMS or spec.get("seed") != 101
            or spec.get("device") != "cuda" or spec.get("epochs") != 120
            or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Fresh active root PyG continuation capability required")
    fit._check_active_v1(spec)
    fit.require_logical_job_v1(lease, spec)
    refs = spec["refs"]
    cfg = reader.decode_json(refs["config"])
    if refs["config"]["path"] != CONFIG or cfg["prior_fit_config_ref"]["path"] != previous.CONFIG:
        raise ValueError("Exact PyG and original160-update config paths required")
    old_cfg = reader.decode_json(cfg["prior_fit_config_ref"])
    validate_config_v1(cfg, old_cfg)
    for name in ("max_cpu_rss_bytes", "max_cuda_reserved_bytes"):
        if type(spec.get(name)) is not int or spec[name] <= 0:
            raise ValueError("Positive root CPU/CUDA bounds required")
    for path in SOURCE_CLOSURE:
        if spec["source_sha256"].get(path) != fit._hash(path):
            raise ValueError("Complete frozen and new PyG source closure required")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    for ref in cfg["helper_refs"].values():
        reader.read(ref)
    registration = reader.decode_json(cfg["registration_ref"])
    if (registration.get("hypothesis_id") != "H037" or registration.get("linked_hypothesis_id") != "H036"
            or registration.get("additional_updates") != 240 or registration.get("total_updates") != 400
            or registration.get("evaluation_epochs") != EVALUATION_EPOCHS
            or registration["continuation_checkpoints"][spec["arm"]] != refs["resume_checkpoint"]
            or registration["prior_completed_refs"][spec["arm"]] != refs["prior_completed"]):
        raise ValueError("New linked H037 root registration/checkpoint/budget condition required")
    for key in ("pyg_model_ref", "pyg_dataset_ref", "observation_cache_ref", "fast_primary_ref",
                "prototype_ref", "preprocessor_ref", "efficiency_ref"):
        reader.read(cfg[key])
    for key, path in (("pyg_model_ref", PYG_MODEL), ("pyg_dataset_ref", PYG_DATASET),
                      ("observation_cache_ref", CACHE), ("fast_primary_ref", FAST_PRIMARY)):
        if cfg[key] != {"path": path, "sha256": spec["source_sha256"][path]}:
            raise ValueError("Config and input implementation pins must agree")
    reader.read(refs["runtime"])
    prior = reader.decode_json(refs["prior_completed"])
    prior_input = reader.decode_json(refs["prior_input"])
    if (prior.get("status") != "completed_same_seed_exact_H036_continuation_total160updates"
            or prior.get("arm") != spec["arm"] or prior.get("seed") != 101
            or prior.get("total_updates") != 160 or prior.get("final_epoch") != 80
            or prior.get("dtype") != "float32" or prior.get("teacher_used") is not False
            or prior.get("input_ref") != refs["prior_input"]
            or prior_input.get("schema") != "mutation_routed_whole_tree_gnn_continue_input_v1"
            or prior_input.get("arm") != spec["arm"] or prior_input.get("seed") != 101
            or prior_input["refs"]["config"] != cfg["prior_fit_config_ref"]
            or prior_input["observed_parents"] != spec["observed_parents"]
            or prior_input["refs"]["target_bindings"] != refs["target_bindings"]):
        raise ValueError("Completed same-arm160-update data/seed/provenance required")
    for path, digest in prior_input["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    model_record = prior["models"][spec["arm"]]
    if (model_record.get("steps") != 160 or model_record.get("last_checkpoint_epoch") != 80
            or model_record.get("last_checkpoint") != refs["resume_checkpoint"]
            or [r["epoch"] for r in model_record["history"]] != PRIOR_HISTORY_EPOCHS
            or [r["optimizer_steps"] for r in model_record["history"]]
               != [2 * e for e in PRIOR_HISTORY_EPOCHS]):
        raise ValueError("Resume same-arm LAST epoch80/160-step checkpoint")
    if set(spec["observed_parents"]) != set(fit.PARENTS):
        raise ValueError("Exactly original four TRAIN/VAL observed parents required")
    cache_sources = spec["observation_cache_source_refs"]
    required_cache_paths = {initial.MODULE, initial.PREPROCESSOR, initial.LOADER, EFFICIENT, CACHE}
    if not required_cache_paths.issubset({ref["path"] for ref in cache_sources.values()}):
        raise ValueError("Cache provenance must pin original observations and cache implementation")
    for ref in cache_sources.values():
        if spec["source_sha256"].get(ref["path"]) != ref["sha256"]:
            raise ValueError("Observation-cache and driver source pins differ")
    base = reader.decode_json(cfg["base_fit_config"])
    bindings = reader.decode_json(refs["target_bindings"])
    manifest = reader.decode_json(base["external_target_manifest"])
    rows = {r["parent_id"]: r for r in manifest["replicates"]}
    if set(bindings["parents"]) != set(fit.PARENTS):
        raise ValueError("Original four-parent target binding required")
    for parent_id, binding in bindings["parents"].items():
        expected = {"path": f"data/biological_paths_execution_pilot_v1/{parent_id}/truth/full_truth_intervals.npz",
                    "sha256": rows[parent_id]["files"]["truth/full_truth_intervals.npz"]["sha256"]}
        if (binding["truth_npz"] != expected or binding["split"] != rows[parent_id]["split"]
                or len(binding["query_sample_indexes"]) != 120 or len(binding["query_raw_node_ids"]) != 120):
            raise ValueError("Original external C4 query/support bindings changed")
    storage = reader.decode_json(refs["storage"])
    from .project_storage_full_reservation_v1 import validate_storage_v9
    if cfg["helper_refs"]["actual_full_storage_launcher"] not in storage["reservation"]["consumer_bindings"]:
        raise ValueError("Actual full-storage launcher must be reserved")
    validate_storage_v9(spec, cfg, storage,
        source_ref={"path": SOURCE, "sha256": spec["source_sha256"][SOURCE]},
        config_ref=refs["config"], runtime_ref=refs["runtime"])
    initial.science_v1()
    import torch_geometric
    if (str(fit.torch.__version__) != cfg["runtime"]["torch"]
            or fit.torch.version.cuda != cfg["runtime"]["torch_cuda"]
            or str(torch_geometric.__version__) != cfg["runtime"]["torch_geometric"]):
        raise ValueError("Root-pinned unchanged Torch/CUDA and PyG runtime required")
    return spec, cfg, base, bindings, reader, storage, prior, prior_input


def restore_v1(spec, cfg, prior, prior_input, reader, device):
    from .mutation_routed_pyg_gnn_v1 import MutationRoutedPyGGNNV1
    torch, np = fit.torch, fit.np
    saved = torch.load(io.BytesIO(reader.read(spec["refs"]["resume_checkpoint"])),
                       map_location="cpu", weights_only=True)
    if (saved["arm"] != spec["arm"] or saved["seed"] != 101
            or saved["epoch"] != 80 or saved["optimizer_steps"] != 160
            or saved["teacher_used"] is not False or saved["input_ref"] != spec["refs"]["prior_input"]
            or saved["initial_state_sha256"] != prior["initial_state_sha256"]
            or saved["original_input_ref"] != prior_input["refs"]["initial_input"]):
        raise ValueError("Exact same-arm LAST epoch80 model/AdamW checkpoint required")
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    model = MutationRoutedPyGGNNV1(**cfg["model"])
    if sum(p.numel() for p in model.parameters()) != cfg["parameter_count"]:
        raise ValueError("PyG must retain the exact learned parameter count")
    initial_hash = initial.state_fingerprint_v1(model)
    if initial_hash != prior["initial_state_sha256"]:
        raise ValueError("Inherited seed101 initialization/state-key order changed")
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"], betas=(.9, .999), eps=1e-8)
    if (len(saved["optimizer_state"]["state"]) != len(list(model.parameters()))
            or any(float(value["step"]) != 160 for value in saved["optimizer_state"]["state"].values())):
        raise ValueError("All trained AdamW counters must resume at exactly160")
    model.load_state_dict(saved["model_state"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    if (not previous.recursive_equal_v1(saved["model_state"], model.state_dict())
            or not previous.recursive_equal_v1(saved["optimizer_state"], optimizer.state_dict())):
        raise ValueError("Restored learned state or AdamW moments/counters/groups differ")
    old = prior["models"][spec["arm"]]
    history = deepcopy(old["history"])
    earliest = min(history, key=lambda row: initial.selection_nll_v1(row["equal_parent_metrics"]))
    selected = deepcopy(old["selected"])
    if (selected["epoch"] != earliest["epoch"] or selected["checkpoint"] != earliest["checkpoint"]
            or initial.selection_nll_v1(selected["equal_parent_metrics"])
               != initial.selection_nll_v1(earliest["equal_parent_metrics"])):
        raise ValueError("Inherited strict earliest-minimum NLL selection changed")
    external = [selected["checkpoint"]] + [c["prediction_seal"] for c in selected["cells"]]
    retained = dict(old["retained_continuation_records"])
    retained.update({r["path"]: r for r in old["external_initial_best"]})
    if any(retained.get(r["path"]) != r for r in external):
        raise ValueError("Inherited selected checkpoint/seals must remain immutable retained refs")
    return dict(model=model, optimizer=optimizer, steps=160, initial_sha256=initial_hash,
        best=initial.selection_nll_v1(selected["equal_parent_metrics"]), best_epoch=selected["epoch"],
        selected=selected, history=history, current=[], retained_best=[],
        external_selected_records=external, TRAIN_log_chunks=[], pending_TRAIN_rows=[])


def bank_bytes_v1(bank):
    total = 0
    for field in fields(bank):
        value = getattr(bank, field.name)
        tensors = value if isinstance(value, tuple) else (value,)
        total += sum(x.numel() * x.element_size() for x in tensors)
    return total


def prepare_v1(spec, cfg, bindings, reader, output, helpers, device, loader_state):
    from .mutation_routed_pyg_dataset_v1 import (
        MutationRoutedCachedParentDatasetV1, make_parent_dataloader_v1,
        shutdown_parent_dataloader_v1,
    )
    from .mutation_routed_observation_cache_v1 import read_observation_cache_index_v1
    index = read_observation_cache_index_v1(spec["refs"]["observation_cache"], reader,
        observed_parent_refs=spec["observed_parents"],
        source_refs=spec["observation_cache_source_refs"], expected_haplotypes=280)
    metadata = index.metadata["bundle_metadata"]
    dataset = MutationRoutedCachedParentDatasetV1(spec["refs"]["observation_cache"],
        observed_parent_refs=spec["observed_parents"], source_refs=spec["observation_cache_source_refs"],
        parent_ids=fit.PARENTS, expected_haplotypes=280,
        worker_tmp_directory=output.path / "data_workers_tmp")
    loader = make_parent_dataloader_v1(dataset, pin_memory=True)
    loader_state["loader"] = loader
    cpu_banks, parents, receipts, worker_rows = {}, {}, {}, []
    for graph in loader:
        parent_id = graph.parent_id
        if parent_id in cpu_banks or parent_id not in fit.PARENTS:
            raise ValueError("Complete observed cache yielded a duplicate/unknown parent")
        bank, parent = graph.to_model_inputs(), graph.scoring
        binding = bindings["parents"][parent_id]
        if (parent["parent_id"] != parent_id or parent["split"] != binding["split"]
                or parent["query_slots"].tolist() != binding["query_sample_indexes"]
                or parent["raw_query_nodes"].tolist() != binding["query_raw_node_ids"]
                or not fit.np.array_equal(bank.query_column.numpy(), parent["query_slots"])
                or len(bank.readout_hold) != len(parent["positions"])
                or not fit.np.array_equal(parent["positions"], 50. + 100. * parent["retained_indexes"])
                or parent["positions"].shape != (len(parent["edges"]) - 1,)):
            raise ValueError("Original actual-query/physical scoring support changed")
        if (graph.worker_pid == os.getpid() or graph.worker_cuda_visible_devices != ""
                or graph.worker_cpu_threads != 1):
            raise ValueError("Observation decoding must use CPU-only one-thread spawn workers")
        cpu_banks[parent_id], parents[parent_id] = bank, parent
        receipts[parent_id] = graph.observations_receipt
        worker_rows.append(dict(parent_id=parent_id, worker_pid=graph.worker_pid,
            cuda_visible_devices=graph.worker_cuda_visible_devices,
            cpu_threads=graph.worker_cpu_threads, tmp_directory=graph.worker_tmp_directory))
    if set(cpu_banks) != set(fit.PARENTS) or len({r["worker_pid"] for r in worker_rows}) != 2:
        raise ValueError("Both CPU workers must finish all four complete parents")
    loader_state["startup_drain"] = shutdown_parent_dataloader_v1(loader)
    if loader_state["startup_drain"].get("drained") is not True:
        raise ValueError("Both startup DataLoader worker PIDs must drain before training")
    train_ids = [p for p in fit.PARENTS if parents[p]["split"] == "train"]
    val_ids = [p for p in fit.PARENTS if parents[p]["split"] == "val"]
    if len(train_ids) != 2 or len(val_ids) != 2:
        raise ValueError("Original two TRAIN/two VAL split required")
    seal = output.save("observations_features_ready.json", dict(
        schema="mutation_routed_pyg_all_parent_observation_seal_v1",
        all_four_parents_complete=True, targets_opened=False, teacher_used=False,
        observation_cache_ref=spec["refs"]["observation_cache"],
        observation_cache_source_refs=spec["observation_cache_source_refs"],
        parents=receipts, loader_receipt=metadata["receipt"], sham_bin_edges=metadata["sham_bin_edges"],
        efficiency_provenance=metadata.get("efficiency_provenance"),
        spawn_workers=worker_rows, startup_worker_drain=loader_state["startup_drain"],
        original_native_holds_preserved=True, full_parent_history_detached=False))
    cpu_targets = {}
    for parent_id in train_ids:
        target = fit.ExternalTargetV1(parents[parent_id], bindings["parents"][parent_id], "TRAIN_loss", reader)
        cpu_targets[parent_id] = fit.np.array(target.target_mass(parents[parent_id], helpers),
                                             dtype=fit.np.float32, copy=True, order="C")
    bank_bytes = sum(bank_bytes_v1(bank) for bank in cpu_banks.values())
    mass_bytes = sum(mass.nbytes for mass in cpu_targets.values())
    free_bytes, total_bytes = fit.torch.cuda.mem_get_info(device)
    current_reserved = fit.torch.cuda.memory_reserved(device)
    reserve = cfg["resident_workspace_reserve_bytes"]
    admission = dict(observed_bank_tensor_bytes=bank_bytes, TRAIN_mass_tensor_bytes=mass_bytes,
        workspace_reserve_bytes=reserve, CUDA_free_bytes=int(free_bytes),
        CUDA_total_bytes=int(total_bytes), current_reserved_bytes=int(current_reserved),
        admitted_max_cuda_reserved_bytes=spec["max_cuda_reserved_bytes"])
    if (bank_bytes + mass_bytes + reserve > free_bytes
            or current_reserved + bank_bytes + mass_bytes + reserve > spec["max_cuda_reserved_bytes"]):
        output.save("resident_memory_not_admitted.json", admission)
        raise ValueError("All-parent GPU residency plus workspace does not fit measured/admitted memory")
    banks = {p: cpu_banks[p].to(device) for p in fit.PARENTS}
    targets = {p: fit.torch.tensor(cpu_targets[p], dtype=fit.torch.float32, device=device) for p in train_ids}
    admission.update(all_four_observed_banks_GPU_resident=True, TRAIN_masses_GPU_resident=True,
        CPU_preparation_repeated_per_update=False, per_update_bank_device_copies=False,
        actual_cuda_allocated_after_residency=int(fit.torch.cuda.memory_allocated(device)),
        actual_cuda_reserved_after_residency=int(fit.torch.cuda.memory_reserved(device)))
    output.save("resident_inputs_ready.json", dict(observation_features_seal=seal,
        targets_purpose="TRAIN_loss_only", query_truth_features=False,
        target_shapes={p: list(targets[p].shape) for p in train_ids}, admission=admission))
    return banks, parents, train_ids, val_ids, targets, seal, admission


def train_parent_v1(state, bank, target, cfg):
    from .biological_gst_control_v1 import exact_span_loss
    torch = fit.torch
    model = state["model"]
    model.train()
    state["optimizer"].zero_grad(set_to_none=True)
    logits = model(bank, control=cfg["_arm"])
    loss = exact_span_loss(logits, target)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite exact-span full-parent TRAIN loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"],
                                         error_if_nonfinite=True)
    state["optimizer"].step()
    state["steps"] += 1
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite fitted H036 PyG parameter")
    return dict(TRAIN_exact_span_query120_NLL=float(loss.detach()),
        unclipped_gradient_norm=float(norm.detach()), optimizer_steps=state["steps"])


def run_v1(record):
    spec, cfg, base, bindings, reader, storage, prior, prior_input = inputs_v1(record)
    from .biological_candidate import configure_fit
    from .project_storage_full_reservation_v1 import PilotOutputV9, budget_check_v9
    from .mutation_routed_fast_primary_v1 import FastPrimaryValidationV1
    from .mutation_routed_pyg_dataset_v1 import shutdown_parent_dataloader_v1
    torch, np = fit.torch, fit.np
    cfg = dict(cfg, _arm=spec["arm"])
    arm = spec["arm"]
    output = PilotOutputV9(spec["output_directory"], storage, reader)
    budget_check_v9(storage, boundary=True, required_write_bytes=32000000)
    device, runtime, lease = configure_fit("cuda", base, output.path)
    runtime["torch_geometric"] = cfg["runtime"]["torch_geometric"]
    output.used = sum(max(p.stat().st_size, p.stat().st_blocks * 512) + 8192
                      for p in output.path.rglob("*") if p.is_file())
    if output.used > 32000000 or output.used > output.limit:
        raise ValueError("CUDA qualification exceeded own32MB allowance")
    helpers = fit.metric_helpers_v1()
    receipt = dict(schema="mutation_routed_pyg_continuation_evidence_v1",
        hypothesis_id="H037", linked_hypothesis_id="H036", condition="same_model_PyG_optimization_continuation",
        status="running", arm=arm, seed=101, dtype="float32", framework="torch_plus_pyg",
        input_ref=record, config_ref=spec["refs"]["config"], registration_ref=cfg["registration_ref"], runtime=runtime,
        prior_completed_ref=spec["refs"]["prior_completed"], prior_input_ref=spec["refs"]["prior_input"],
        resume_checkpoint=spec["refs"]["resume_checkpoint"], resume_epoch=80, resume_updates=160,
        additional_updates=240, total_updates=400, final_epoch=200,
        evaluation_epochs=EVALUATION_EPOCHS, exact_optimizer_resume=True, restarted=False,
        teacher_used=False, whole_parent_history_detached=False, artificial_tile_reset=False,
        checkpoint_selection=initial.SELECTION, source_sha256=spec["source_sha256"],
        novelty_claim=False, mechanism_claim=False, development_only=True, TEST_parents=0,
        comparison_limitation=cfg["comparison_limitation"], models={})
    output.save("started.json", receipt)
    started = time.monotonic()
    state = None
    loader_state = {}
    last_durable_epoch = 80
    last_durable_checkpoint = spec["refs"]["resume_checkpoint"]
    try:
        initial.resources_v1(spec, storage, lease)
        state = restore_v1(spec, cfg, prior, prior_input, reader, device)
        receipt["initial_state_sha256"] = state["initial_sha256"]
        output.save("resume_verified.json", dict(
            resume_checkpoint=spec["refs"]["resume_checkpoint"],
            initial_state_sha256=state["initial_sha256"], restored_steps=160, restored_epoch=80,
            model_state_exact=True, optimizer_state_exact=True, framework="torch_plus_pyg",
            inherited_selected_epoch=state["best_epoch"], external_selected_records=state["external_selected_records"]))
        banks, parents, train_ids, val_ids, targets, observation_seal, admission = prepare_v1(
            spec, cfg, bindings, reader, output, helpers, device, loader_state)
        receipt["observation_features_seal"] = observation_seal
        receipt["resident_admission"] = admission
        validation = FastPrimaryValidationV1(parents, bindings, reader, helpers, cfg, backend="numpy")
        for epoch in range(81, 201):
            epoch_start = time.monotonic()
            evaluation_due = epoch in EVALUATION_EPOCHS
            if evaluation_due:
                initial.prune_nonbest_probability_seals_v1(output, state)
            order = list(train_ids)
            random.Random(101 + epoch).shuffle(order)
            losses = []
            for parent_id in order:
                initial.lightweight_resources_v1(spec, lease)
                result = train_parent_v1(state, banks[parent_id], targets[parent_id], cfg)
                losses.append(result["TRAIN_exact_span_query120_NLL"])
                initial.lightweight_resources_v1(spec, lease)
            train_row = dict(epoch=epoch, arm=arm, total_optimizer_steps=state["steps"],
                additional_optimizer_steps=state["steps"] - 160,
                TRAIN_mean_full_parent_query120_NLL=float(np.mean(losses)),
                evaluation_due=evaluation_due, last_durable_epoch=last_durable_epoch,
                updates_since_last_durable_checkpoint=state["steps"] - 2 * last_durable_epoch,
                elapsed_TRAIN_seconds=time.monotonic() - epoch_start)
            state["pending_TRAIN_rows"].append(train_row)
            if not evaluation_due:
                print(json.dumps(dict(stage="H037_PyG_TRAIN_epoch", **train_row)), flush=True)
                continue
            initial.resources_v1(spec, storage, lease)
            chunk = output.save(f"{arm}.epochs_{epoch-9:03d}through{epoch:03d}.TRAIN_updates.json",
                                state["pending_TRAIN_rows"])
            state["TRAIN_log_chunks"].append(chunk)
            state["pending_TRAIN_rows"] = []
            seals = []
            state["model"].eval()
            for parent_id in val_ids:
                initial.lightweight_resources_v1(spec, lease)
                with torch.no_grad():
                    logits = state["model"](banks[parent_id], control=arm)
                    seal = initial.seal_probabilities_v1(output, arm, epoch, parents[parent_id], logits)
                seals.append((parent_id, seal))
                del logits
            manifest = output.save(f"{arm}.epoch_{epoch:03d}.VAL_predictions_sealed.json", dict(
                epoch=epoch, arm=arm, all_VAL_predictions_sealed_before_scoring=True,
                targets_opened_in_this_prediction_phase=False, seals=dict(seals)))
            cells, means = validation.score(seals)
            nll = initial.selection_nll_v1(means)
            previous_records = {r["path"]: r for r in state["current"] + state["retained_best"]}
            checkpoint = output.save(f"{arm}.epoch_{epoch:03d}.pt", dict(
                model_state={k: v.detach().cpu() for k, v in state["model"].state_dict().items()},
                optimizer_state=state["optimizer"].state_dict(), epoch=epoch, arm=arm,
                optimizer_steps=state["steps"], seed=101, input_ref=record,
                original_input_ref=prior_input["refs"]["initial_input"],
                initial_state_sha256=state["initial_sha256"], teacher_used=False,
                prior_completed_ref=spec["refs"]["prior_completed"], framework="torch_plus_pyg"), "torch")
            last_durable_epoch, last_durable_checkpoint = epoch, checkpoint
            state["current"] = [r for _, r in seals] + [checkpoint]
            row = dict(epoch=epoch, arm=arm, optimizer_steps=state["steps"],
                additional_optimizer_steps=state["steps"] - 160,
                TRAIN_mean_full_parent_query120_NLL=float(np.mean(losses)),
                cells=cells, equal_parent_metrics=means, checkpoint=checkpoint,
                prediction_manifest=manifest, primary_selection_only=True)
            state["history"].append(row)
            if nll < state["best"]:
                state.update(best=nll, best_epoch=epoch, selected=row, retained_best=list(state["current"]))
            row["selected_epoch"] = state["best_epoch"]
            output.save(f"{arm}.epoch_{epoch:03d}.metrics.json", row)
            keep = {r["path"] for r in state["current"] + state["retained_best"]}
            for path, product in previous_records.items():
                if path not in keep:
                    output.discard(product)
            initial.resources_v1(spec, storage, lease)
            print(json.dumps(dict(stage="H037_PyG_selection_epoch", arm=arm, epoch=epoch,
                total_optimizer_steps=state["steps"], additional_optimizer_steps=state["steps"] - 160,
                elapsed_seconds=time.monotonic() - epoch_start,
                selected_epoch=state["best_epoch"], primary_present64=means["primary_present_day_64"])), flush=True)
        if (state["steps"] != 400 or state["pending_TRAIN_rows"]
                or [r["epoch"] for r in state["history"]] != PRIOR_HISTORY_EPOCHS + EVALUATION_EPOCHS
                or len(state["TRAIN_log_chunks"]) != 12):
            raise ValueError("Exact160+240updates or frozen evaluation/log history incomplete")
        selected_seals = [(c["parent_id"], c["prediction_seal"]) for c in state["selected"]["cells"]]
        full_cells, full_means = validation.full_selected_diagnostics(selected_seals)
        diagnostics = output.save("final_selected_full_diagnostics.json", dict(
            selected_epoch=state["best_epoch"], selected_checkpoint=state["selected"]["checkpoint"],
            selected_seals=dict(selected_seals), cells=full_cells, equal_parent_metrics=full_means,
            original_scorer_used=True, selection_revised=False))
        receipt["fast_primary_scoring"] = validation.receipt()
        receipt["data_worker_drain"] = shutdown_parent_dataloader_v1(loader_state["loader"])
        receipt["data_worker_startup_drain"] = loader_state["startup_drain"]
        if receipt["data_worker_drain"].get("drained") is not True:
            raise ValueError("Both DataLoader worker PIDs must drain before completion")
        receipt["models"][arm] = dict(selected=state["selected"], history=state["history"],
            steps=400, additional_steps=240, parameter_count=cfg["parameter_count"],
            TRAIN_log_chunks=state["TRAIN_log_chunks"], last_checkpoint=last_durable_checkpoint,
            last_checkpoint_epoch=last_durable_epoch, final_selected_full_diagnostics=diagnostics,
            external_selected_records=state["external_selected_records"],
            retained_PyG_records={r["path"]: r for r in state["current"] + state["retained_best"]})
        reader.read(record)
        initial.resources_v1(spec, storage, lease)
        receipt.update(status="completed_same_seed_H037_PyG_continuation_total400updates",
            elapsed_seconds=time.monotonic() - started, output_accounted_bytes=output.used,
            original_results_never_modified=True, joint_paired_comparison_pending_other_worker=True)
    except Exception as error:
        drain_error = None
        if "loader" in loader_state:
            try:
                receipt["data_worker_drain"] = shutdown_parent_dataloader_v1(loader_state["loader"])
            except Exception as worker_error:
                drain_error = str(worker_error)
        receipt.update(status="failed_partial_preserved", error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic() - started, output_accounted_bytes=output.used,
            last_durable_epoch=last_durable_epoch, last_durable_checkpoint=last_durable_checkpoint,
            logical_steps_reached=160 if state is None else state["steps"],
            pending_TRAIN_rows=[] if state is None else state["pending_TRAIN_rows"],
            updates_not_checkpointed=0 if state is None else state["steps"] - 2 * last_durable_epoch,
            data_worker_drain_error=drain_error, automatic_replay_or_resume_executed=False)
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
        print(json.dumps(dict(status="source_only_PyG_continuation_not_executed",
            hypothesis_id="H037", linked_hypothesis_id="H036", implementation={"path": SOURCE, "sha256": fit._hash(SOURCE)},
            config={"path": CONFIG, "sha256": fit._hash(CONFIG)}, arms=list(ARMS),
            resume_epoch=80, resume_steps=160, additional_steps=240, total_steps=400,
            evaluation_epochs=EVALUATION_EPOCHS, fresh_root_registration_and_admission_required=True), indent=2))
        return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Fresh pinned root PyG continuation capability required")
    run_v1({"path": args.input, "sha256": args.expected_input_sha256})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Root-only biological GST task adaptation over frozen complete tile graphs.

The existing GST architecture is imported unchanged. Source checks are standard
library only and never deserialize observations, models, labels, or scores.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
PARENTS = [f"bio_paths_pilot_{split}_{i:03d}" for split in ("train", "val") for i in range(2)]
ARMS = {"gst32_primary": 32, "gst64_capacity_sensitivity": 64}
FEATURE_COLUMNS = tuple(i for i in range(48) if i != 7)


def path(value):
    p = (ROOT / value).resolve()
    p.relative_to(ROOT)
    return p


def sha(p):
    result = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked(record):
    p = path(record["path"])
    if not p.is_file() or sha(p) != record["sha256"]:
        raise ValueError("Frozen file changed: " + str(p))
    return p


def verify_sources(cfg):
    for name, digest in cfg["source_sha256"].items():
        if sha(path(name)) != digest:
            raise ValueError("Frozen source drift: " + name)


def parameter_count(hidden_dim):
    """Analytic count for F47/C4, two paper-gate cross-attention blocks."""
    return 39 * hidden_dim * hidden_dim + 105 * hidden_dim + 4


def load_protocol(config_path, expected_hash):
    """Source/config/score-free selection metadata only, without Torch imports."""
    if not expected_hash or sha(config_path) != expected_hash:
        raise ValueError("Root-reviewed exact GST config SHA256 required")
    cfg = json.loads(config_path.read_text())
    expected_architecture = dict(input_dim=47, num_classes=4, num_layers=2, num_heads=4,
        dropout=0.0, routing="cross_attention", token_weighting="span", gate_variant="paper",
        ffn_multiplier=2, self_loop_weight=2.0, validate=True)
    expected_training = dict(epochs=30, patience=8, learning_rate=0.001,
        weight_decay=0.0001, gradient_clip_norm=1.0,
        selection="minimum_mean_parent_VAL_present64_exact_support_raw_probability_NLL",
        supervision="actual_query120_exact_class_span_mass_only",
        VAL_calibration_fit=False, checkpoint_ties="earliest_epoch_strict_less")
    if (cfg["kind"] != "biological_gst_control_v1" or cfg["parents"] != PARENTS
            or cfg["seeds"] != [101, 202, 303] or cfg["models"] != list(ARMS)
            or cfg["architecture"] != expected_architecture
            or cfg["model_widths"] != ARMS or cfg["training"] != expected_training
            or cfg["input"]["whole_fields"] != 48 or cfg["input"]["source_fields"] != 47
            or cfg["input"]["feature_columns"] != list(FEATURE_COLUMNS)
            or cfg["input"]["one_complete_existing_tile_per_forward"] is not True
            or cfg["input"]["graph_subset_batching"] is not False
            or cfg["input"]["Kron_bank_loading_or_reconstruction"] is not False):
        raise ValueError("Frozen GST biological architecture/input/fit policy changed")
    verify_sources(cfg)
    h020_path = checked(cfg["H020_config"])
    h020 = json.loads(h020_path.read_text())
    for name, digest in h020["source_sha256"].items():
        if sha(path(name)) != digest:
            raise ValueError("Frozen H020 dependency drift: " + name)
    if (h020["kind"] != "biological_candidate_pilot_v1"
            or h020["training"]["epochs"] != 30 or h020["training"]["patience"] != 8
            or h020["parents"] != cfg["parents"] or h020["seeds"] != cfg["seeds"]
            or cfg["metrics"] != h020["metrics"] or cfg["storage"] != h020["storage"]
            or cfg["external_truth_manifest"] != h020["external_truth_manifest"]
            or cfg["gpu_runtime"] != dict(h020["gpu_runtime"],
                    launcher_module="arggnn.biological_gst_control_v1")):
        raise ValueError("Common H020 evaluation/storage/GPU policy differs")
    metadata = json.loads(checked(cfg["H009_selection_metadata"]).read_text())
    if (metadata["kind"] != "H009_actual_GST_fixed_configuration_metadata"
            or metadata["model"] != "gst_ca_span" or metadata["hidden_dim"] != 32
            or metadata["num_layers"] != 2 or metadata["dropout"] != 0
            or metadata["learning_rate"] != 0.001 or metadata["weight_decay"] != 0.0001
            or metadata["source_old_input_dim"] != 48 or metadata["source_old_parameters"] != 43332
            or metadata["no_scores_or_weights_in_metadata"] is not True
            or metadata["source_config"] != cfg["H009_original_config"]):
        raise ValueError("Score-free actual H009 fixed-control selection binding differs")
    checked(cfg["H009_original_config"])
    return cfg


def source_only_receipt(cfg, config_sha):
    return dict(status="frozen_source_contract_checked_only",
        config_sha256=config_sha, models={name: dict(hidden_dim=width,
            F47_parameters=parameter_count(width)) for name, width in ARMS.items()},
        H009_selection_metadata=cfg["H009_selection_metadata"],
        truth_opened=False, observed_tree_or_bank_opened=False,
        tensor_or_model_loaded=False, training_executed=False, GPU_accessed=False)


def source47(sequence):
    """Only remove raw parent-branch column 7 on every node, retaining tensors."""
    if sequence.x.ndim != 2 or sequence.x.shape[1] != 48:
        raise ValueError("Frozen whole graph must have exactly 48 source fields")
    return replace(sequence, x=sequence.x[:, FEATURE_COLUMNS])


def model_factory(name, cfg):
    if name not in ARMS:
        raise ValueError("Unknown frozen GST arm")
    from .gst_baseline import GSTAdaptation
    return GSTAdaptation(hidden_dim=cfg["model_widths"][name], **cfg["architecture"])


def actual_query_logits(logits, output_indexes, actual_queries=120):
    import torch
    indexes = torch.as_tensor(output_indexes, dtype=torch.long, device=logits.device)
    if (logits.ndim != 3 or indexes.shape != (actual_queries,)
            or indexes.unique().numel() != actual_queries
            or bool((indexes < 0).any()) or bool((indexes >= logits.shape[1]).any())):
        raise ValueError("Distinct valid actual-query readout indexes required")
    return logits.index_select(1, indexes)


def exact_span_loss(logits, target_mass):
    import torch
    if (logits.shape != target_mass.shape or target_mass.ndim != 3
            or not bool(torch.isfinite(target_mass).all())
            or bool((target_mass < 0).any()) or not bool(target_mass.sum() > 0)):
        raise ValueError("Actual-query exact class-span target mass invalid")
    return -(torch.log_softmax(logits, dim=-1) * target_mass).sum() / target_mass.sum()


def exact_probability_nll(probabilities, target_mass, columns):
    import numpy as np
    p, mass = probabilities[:, columns].astype(np.float64), target_mass[:, columns]
    zero = (p == 0) & (mass > 0)
    if zero.any():
        return float("inf"), float(mass[zero].sum())
    return float(-(mass * np.log(np.where(p > 0, p, 1))).sum() / mass.sum()), 0.0


def validate_whole_tile(sequence, tile, parent):
    """Validate exact source serialization and roles before mechanical projection."""
    import numpy as np
    import torch
    from .models import validate_model_input
    validate_model_input(sequence)
    nodes = parent["observation"]["nodes"]
    indexes = np.asarray(tile["position_indexes"], dtype=int)
    required_flags = ("all_tree_nodes_retained", "all280_leaves_retained",
                      "exact_serialization_roundtrip_checked")
    if (any(tile[k] is not True for k in required_flags)
            or sequence.x.dtype != torch.float32 or sequence.x.shape[1] != 48
            or sequence.intervals.dtype != torch.float64
            or sequence.positions.dtype != torch.float64
            or sequence.window_bounds.dtype != torch.float64
            or sequence.num_graphs != len(tile["original_tree_ids"])
            or sequence.num_queries != 184
            or not torch.equal(sequence.query_node_ids, torch.as_tensor(
                nodes[parent["dynamics_slots"]].copy(), dtype=torch.long))
            or not np.array_equal(sequence.positions.numpy(),
                parent["positions"][indexes] / 2000000)
            or not np.array_equal(sequence.window_bounds.numpy(),
                np.asarray(tile["normalized_interval"], dtype=np.float64))):
        raise ValueError("Complete tile/role/coordinate serialization changed")
    actual = torch.as_tensor(nodes[parent["query_slots"]].copy(), dtype=torch.long)
    references = torch.as_tensor(nodes[parent["reference_slots"]].copy(), dtype=torch.long)
    expected_flags = torch.isin(sequence.node_ids, actual).float()
    if (not torch.equal(sequence.x[:, 4], expected_flags)
            or not torch.equal(sequence.x[:, 5], (sequence.node_ids >= 0).float())
            or not torch.equal(sequence.x[~torch.isin(sequence.node_ids, references), :4],
                               torch.zeros_like(sequence.x[~torch.isin(sequence.node_ids, references), :4]))):
        raise ValueError("Only actual120 query flags and pure96 source classes may enter")
    expected_classes = torch.zeros_like(sequence.x[:, :4])
    for slot in parent["reference_slots"]:
        klass = parent["samples"][slot]["observed_reference_class"]
        expected_classes[sequence.node_ids == int(nodes[slot]), klass] = 1.0
    if not torch.equal(sequence.x[:, :4], expected_classes):
        raise ValueError("Frozen pure96 class identities changed")
    for graph in range(sequence.num_graphs):
        mask = sequence.node_graph == graph
        leaves = sequence.node_ids[mask]
        leaves = leaves[leaves >= 0]
        if not np.array_equal(np.sort(leaves.numpy()), np.sort(nodes)):
            raise ValueError("Every original marginal tree must retain all280 leaves")
        graph_edges = sequence.edge_index[:, sequence.node_graph[sequence.edge_index[0]] == graph]
        edge_pairs = set(map(tuple, graph_edges.T.tolist()))
        if (len(edge_pairs) != 2 * (int(mask.sum()) - 1)
                or graph_edges.shape[1] != len(edge_pairs)
                or any((v, u) not in edge_pairs for u, v in edge_pairs)
                or any(u == v for u, v in edge_pairs)):
            raise ValueError("Full both-direction tree topology required")


def load_whole(tile, parent, device):
    import torch
    from .data import GraphSequence
    p = checked(tile["whole_observed_graph"])
    saved = torch.load(p, map_location="cpu", weights_only=True)
    if set(saved) != {"sequence", "labels_in_input"} or saved["labels_in_input"] is not False:
        raise ValueError("Frozen observed-only whole-graph serialization required")
    sequence = GraphSequence.from_dict(saved["sequence"])
    validate_whole_tile(sequence, tile, parent)
    projected = source47(sequence)
    return projected.to(device)


def prepared_contract(cfg, parents, prepared_path, prepared_sha, observed_sha):
    if (not prepared_sha or prepared_sha != cfg["prepared_manifest"]["sha256"]
            or prepared_path != checked(cfg["prepared_manifest"])):
        raise ValueError("Root-reviewed completed H020 prepared manifest SHA256 required")
    prepared = json.loads(prepared_path.read_text())
    h020 = json.loads(checked(cfg["H020_config"]).read_text())
    if (prepared["status"] != "complete_observed_only_Q184_R96_actualQ120"
            or prepared["config_sha256"] != cfg["H020_config"]["sha256"]
            or prepared["observed_manifest_sha256"] != observed_sha
            or prepared["source_sha256"] != h020["source_sha256"]
            or prepared["truth_opened"] is not False):
        raise ValueError("Frozen H020 prepared source/config/observed binding differs")
    records = {r["parent_id"]: r for r in prepared["parents"]}
    if len(prepared["parents"]) != 4 or set(records) != set(PARENTS):
        raise ValueError("Exactly four prepared biological parents required")
    tree_counts = []
    for parent in parents:
        r = records[parent["parent_id"]]
        expected = dict(query_node_ids=parent["observation"]["nodes"][parent["query_slots"]].tolist(),
            dynamics_query_node_ids=parent["observation"]["nodes"][parent["dynamics_slots"]].tolist(),
            positions_bp=parent["positions"].tolist(), bin_edges_bp=parent["edges"].tolist(),
            actual_query_output_indexes=parent["output_indexes"].tolist(),
            actual_query_sample_indexes=parent["query_slots"].tolist(),
            dynamics_sample_indexes=parent["dynamics_slots"].tolist(),
            reference_sample_indexes=parent["reference_slots"],
            retained_fixed100bp_indexes=parent["retained_indexes"].tolist())
        if r["relate_entry"] != parent["observation"]["entry"] or r["split"] != parent["split"]:
            raise ValueError("Prepared actual inferred parent identity changed")
        for key, value in expected.items():
            if r[key] != value:
                raise ValueError("Prepared role/output/grid join changed: " + key)
        seen_positions = []
        tree_ids = set()
        if len(r["tiles"]) != 8 or r["original_trees_preserved"] not in (955, 897, 946, 920):
            raise ValueError("Existing eight complete physical tiles required")
        for tile in r["tiles"]:
            if (tile["all_tree_nodes_retained"] is not True or tile["all280_leaves_retained"] is not True
                    or tile["query_dynamics"] != 184 or tile["pure_reference_sources"] != 96
                    or tile["supervised_outputs"] != 120 or tile["mixed_latent_outputs"] != 64
                    or tile["mixed_query_flag"] != 0):
                raise ValueError("Complete observed tile role/count contract changed")
            checked(tile["whole_observed_graph"])
            # The Kron bank record remains provenance only and is never opened.
            seen_positions.extend(tile["position_indexes"])
            tree_ids.update(tile["original_tree_ids"])
        if sorted(seen_positions) != list(range(len(parent["positions"]))):
            raise ValueError("Every retained original center must be scored exactly once")
        if tree_ids != set(range(r["original_trees_preserved"])):
            raise ValueError("Every original tree ID must survive, without resampling")
        tree_counts.append(len(tree_ids))
    if tree_counts != [955, 897, 946, 920] or sum(tree_counts) != 3718:
        raise ValueError("All 3718 original marginal records required, with tile boundary copies")
    return records


def sealed_prediction(model, record, parent, output, storage, device):
    import numpy as np
    import torch
    logits = np.empty((len(parent["positions"]), 120, 4), dtype=np.float32)
    seen = np.zeros(len(parent["positions"]), dtype=bool)
    model.eval()
    with torch.no_grad():
        for tile in record["tiles"]:
            sequence = load_whole(tile, parent, device)
            result = actual_query_logits(model(sequence), parent["output_indexes"])
            indexes = np.asarray(tile["position_indexes"], dtype=int)
            if result.shape != (len(indexes), 120, 4) or seen[indexes].any():
                raise ValueError("Actual-query output/grid correspondence changed")
            logits[indexes] = result.detach().cpu().numpy()
            seen[indexes] = True
            del sequence
    if not seen.all() or not np.isfinite(logits).all():
        raise ValueError("Sealed query120 prediction incomplete/nonfinite")
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(-1), 1, atol=1e-6, rtol=1e-5)):
        raise ValueError("Raw C4 probabilities invalid")
    with output.open("xb") as stream:
        np.savez_compressed(stream, logits=logits, C4_probabilities=probabilities,
            positions_bp=parent["positions"], bin_edges_bp=parent["edges"],
            retained_fixed100bp_indexes=parent["retained_indexes"],
            coverage_original_bp0=parent["coverage_original"],
            normalization_length_bp=np.asarray(2000000),
            query_sample_indexes=parent["query_slots"],
            actual_query_output_indexes=parent["output_indexes"],
            query_node_ids=parent["observation"]["nodes"][parent["query_slots"]])
    storage.account(output)
    return probabilities, {"path": str(output), "sha256": sha(output)}


def train(cfg, config_path, expected_config, manifest_path, expected_manifest,
          prepared_path, prepared_sha, seed, output, snapshot_path, snapshot_sha, device_name="cuda"):
    # The runner applies lease/title before reaching any Torch/model import.
    import random
    import time
    import numpy as np
    import torch
    from .biological_candidate import (load_protocol as h020_protocol, observed_contract,
        AccountedStorage, configure_fit, check_fit_lease, ExternalTruthCapability,
        write_json, fingerprint_sha)
    if seed not in cfg["seeds"]:
        raise ValueError("Only frozen seeds101,202,303 supported")
    if expected_manifest != cfg["observed_manifest"]["sha256"] or manifest_path != checked(cfg["observed_manifest"]):
        raise ValueError("Exact H020 observed producer binding required")
    base_cfg, helpers = h020_protocol(checked(cfg["H020_config"]), cfg["H020_config"]["sha256"])
    parents = observed_contract(base_cfg, manifest_path, expected_manifest, helpers)
    records = prepared_contract(cfg, parents, prepared_path, prepared_sha, expected_manifest)
    required = (cfg["storage"]["training_prediction_bytes_per_parent_epoch"]
        * 2 * cfg["training"]["epochs"] * len(cfg["models"])
        + cfg["storage"]["training_fixed_bytes"])
    storage = AccountedStorage.admit(snapshot_path, snapshot_sha, required, cfg)
    output.mkdir(exist_ok=False)
    device, runtime, lease = configure_fit(device_name, cfg, output)
    receipt = dict(kind="biological_gst_control_fit_v1", status="running", seed=seed,
        config_sha256=expected_config, observed_manifest_sha256=expected_manifest,
        prepared_manifest_sha256=prepared_sha, source_sha256=cfg["source_sha256"],
        H009_selection_metadata=cfg["H009_selection_metadata"], models={}, device=str(device),
        storage=storage.record(), TEST_parents=0, GPU_qualification=runtime,
        input="frozen whole observed graphs with all-node F47 projection",
        input_aggregation_difference="GST uses all internal-node F47 and topology, H020 uses leafF47 plus Kron responses",
        target_scope="actual query120 only, mixed64 outputs never labeled",
        runtime=dict(torch=str(torch.__version__), threads=torch.get_num_threads(),
            feature_dtype="float32", coordinate_dtype="float64"),
        baseline_scope="existing independent paper-equation ARG task adaptation, not original-domain reproduction")
    write_json(output / "receipt.json", receipt)
    states, train_truth, val_truth = {}, {}, {}
    started = time.monotonic()
    try:
        for name in cfg["models"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = model_factory(name, cfg).to(device)
            count = sum(p.numel() for p in model.parameters())
            if count != parameter_count(cfg["model_widths"][name]):
                raise ValueError("Actual GST parameter count differs from frozen analytic count")
            states[name] = dict(model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                    weight_decay=cfg["training"]["weight_decay"]),
                best=float("inf"), best_epoch=None, stale=0, active=True, history=[],
                initial_weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, steps=0)
            receipt["models"][name] = dict(status="pending", parameters=count,
                configuration=model.configuration_provenance(),
                initial_parameter_fingerprint_sha256=fingerprint_sha(model.state_dict()),
                query_dynamics=184, reference_sources=96, actual_supervised_outputs=120,
                mixed_latent_outputs=64, mixed_query_feature_flag=0,
                actual_coordinate_units="original physical bp/2000000",
                optimizer_weight_decay=cfg["training"]["weight_decay"])
        train_parents = [p for p in parents if p["split"] == "train"]
        val_parents = [p for p in parents if p["split"] == "val"]
        for parent in train_parents:
            capability = ExternalTruthCapability(parent, cfg, "TRAIN_loss")
            train_truth[parent["parent_id"]] = capability.target_mass(parent, helpers)
        tiles = [(p, t) for p in train_parents for t in records[p["parent_id"]]["tiles"]]
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            check_fit_lease(lease)
            active = [name for name, state in states.items() if state["active"]]
            if not active:
                break
            order = list(range(len(tiles)))
            random.Random(seed + epoch).shuffle(order)
            losses = {name: [] for name in active}
            for index in order:
                parent, tile = tiles[index]
                sequence = load_whole(tile, parent, device)
                positions = np.asarray(tile["position_indexes"], dtype=int)
                mass = torch.from_numpy(train_truth[parent["parent_id"]][positions]).to(device, dtype=torch.float32)
                for name in active:
                    state = states[name]
                    model = state["model"]
                    model.train()
                    state["optimizer"].zero_grad(set_to_none=True)
                    logits = actual_query_logits(model(sequence), parent["output_indexes"])
                    loss = exact_span_loss(logits, mass)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite exact TRAIN query-bp loss")
                    loss.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise ValueError("Nonfinite GST gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
                    state["optimizer"].step()
                    state["steps"] += 1
                    losses[name].append(float(loss.detach()))
                del sequence
            sealed_by_arm = {}
            # Seal all active arms and both VAL parents before any epoch scoring.
            for name in active:
                predictions = []
                for parent in val_parents:
                    pfile = output / f"{name}.epoch_{epoch:02d}.{parent['parent_id']}.probabilities.npz"
                    probabilities, sealed = sealed_prediction(states[name]["model"], records[parent["parent_id"]],
                        parent, pfile, storage, device)
                    predictions.append((parent, probabilities, sealed))
                sealed_by_arm[name] = predictions
            for name in active:
                state = states[name]
                cells = []
                for parent, probabilities, sealed in sealed_by_arm[name]:
                    if parent["parent_id"] not in val_truth:
                        val_truth[parent["parent_id"]] = ExternalTruthCapability(
                            parent, cfg, "VAL_after_sealed_prediction", sealed)
                    mass = val_truth[parent["parent_id"]].target_mass(parent, helpers)
                    primary = np.flatnonzero(parent["cohorts"] == "present_day")
                    nll, infinite = exact_probability_nll(probabilities, mass, primary)
                    cells.append(dict(parent_id=parent["parent_id"], nll=nll,
                        infinite_nll_query_bp=infinite, sealed_prediction=sealed))
                mean_nll = float(np.mean([cell["nll"] for cell in cells]))
                row = dict(epoch=epoch, mean_parent_VAL_present64_NLL=mean_nll, VAL=cells,
                    mean_tile_TRAIN_query120_NLL=float(np.mean(losses[name])),
                    selection_scope="exact retained-center clipped-bin query-bp per parent, unweighted parent mean",
                    optimizer_steps=state["steps"],
                    maximum_parameter_delta_from_initial=max(float((v.detach().cpu() - state["initial_weights"][k]).abs().max())
                        for k, v in state["model"].state_dict().items() if v.is_floating_point()))
                state["history"].append(row)
                if mean_nll < state["best"]:
                    checkpoint = output / f"{name}.best_epoch_{epoch:02d}.pt"
                    torch.save(dict(model_state=state["model"].state_dict(),
                        optimizer_state=state["optimizer"].state_dict(), epoch=epoch, seed=seed,
                        config_sha256=expected_config, observed_manifest_sha256=expected_manifest,
                        prepared_manifest_sha256=prepared_sha, selection_VAL_NLL=mean_nll), checkpoint)
                    storage.account(checkpoint)
                    state.update(best=mean_nll, best_epoch=epoch, stale=0,
                        checkpoint={"path": str(checkpoint), "sha256": sha(checkpoint)})
                else:
                    state["stale"] += 1
                state["active"] = state["stale"] < cfg["training"]["patience"]
                receipt["models"][name].update(status="fitting", history=state["history"],
                    best_epoch=state["best_epoch"], best_mean_parent_VAL_present64_NLL=state["best"],
                    checkpoint=state.get("checkpoint"))
            receipt["storage"] = storage.record()
            write_json(output / "receipt.json", receipt)
            print(json.dumps(dict(stage="fit_epoch", seed=seed, epoch=epoch, active_models=active)), flush=True)
        for name, state in states.items():
            if state["best_epoch"] is None:
                raise ValueError("No finite VAL-selected GST checkpoint for " + name)
            saved = torch.load(checked(state["checkpoint"]), map_location="cpu", weights_only=True)
            state["model"].load_state_dict(saved["model_state"])
            delta = max(float((v.detach().cpu() - state["initial_weights"][k]).abs().max())
                for k, v in state["model"].state_dict().items() if v.is_floating_point())
            if state["steps"] <= 0 or not delta > 0:
                raise ValueError("Selected GST checkpoint has no changed-weight optimizer fit")
            cells = []
            for parent in val_parents:
                pfile = output / f"{name}.selected.{parent['parent_id']}.probabilities.npz"
                probabilities, sealed = sealed_prediction(state["model"], records[parent["parent_id"]],
                    parent, pfile, storage, device)
                capability = val_truth[parent["parent_id"]]
                groups = dict(primary_present_day_64=np.flatnonzero(parent["cohorts"] == "present_day"),
                    secondary_ancient_56=np.flatnonzero(parent["cohorts"] != "present_day"),
                    all_queries_120=np.arange(120))
                metrics = {group: helpers.full_span_score(probabilities, parent, capability, columns, cfg["metrics"])
                    for group, columns in groups.items()}
                result = dict(parent_id=parent["parent_id"], split="val", seed=seed, model=name,
                    selected_epoch=state["best_epoch"], sealed_prediction=sealed,
                    external_truth_provenance=capability.provenance, C4_exact_support_metrics=metrics,
                    no_calibration_fit=True, mixed_latent_outputs_labeled=False)
                metric_path = output / f"{name}.{parent['parent_id']}.metrics.json"
                write_json(metric_path, result)
                storage.account(metric_path)
                cells.append({"parent_id": parent["parent_id"], "path": str(metric_path), "sha256": sha(metric_path)})
            receipt["models"][name].update(status="completed", metrics=cells,
                optimizer_fit_completed=True, optimizer_steps=state["steps"],
                selected_maximum_parameter_delta_from_initial=delta,
                configuration=state["model"].configuration_provenance(),
                actual_coordinate_units="original physical bp/2000000")
        if sha(config_path) != expected_config or sha(manifest_path) != expected_manifest or sha(prepared_path) != prepared_sha:
            raise ValueError("Frozen config/observation/prepared manifest changed during fit")
        verify_sources(cfg)
        h020_protocol(checked(cfg["H020_config"]), cfg["H020_config"]["sha256"])
        checked(cfg["H009_selection_metadata"])
        for record in records.values():
            for tile in record["tiles"]:
                checked(tile["whole_observed_graph"])
        check_fit_lease(lease)
        if device.type == "cuda":
            torch.cuda.synchronize()
            receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt.update(status="completed_two_GST_conditions_one_seed",
            elapsed_seconds=time.monotonic() - started, storage=storage.record())
    except Exception as error:
        receipt.update(status="failed_partial_preserved", error_type=type(error).__name__,
            error=str(error), elapsed_seconds=time.monotonic() - started, storage=storage.record())
        write_json(output / "receipt.json", receipt)
        raise
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    import runpy
    sys.modules.setdefault("arggnn.biological_gst_control_v1", sys.modules[__name__])
    runpy.run_path(str(ROOT / "scripts/train_biological_gst_control_v1.py"), run_name="__main__")


if __name__ == "__main__":
    main()

"""H025 one-seed FP32 persistence pilot over frozen native ARGMix unaries.

Explicit hybrid: a Relate ARGMix emitter plus independently inferred observed
tsinfer graphs. No topology equivalence, refit, OOF prediction or predictive
benefit is claimed. The root supplies pinned inputs, storage and a held GPU
lease. No numerical matrix gate, certificate framework or seed sweep is used.
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

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/train_persistence_trace_pilot_v1.py"
CONFIG = "configs/persistence_trace_pilot_v1.json"
ARMS = ("genomic_smoothing", "sample_only", "genuine_ancestor", "sham_ancestor")
CLASS_ORDER = ["ANA", "CHG", "WHG", "EHG"]


class PilotOutputV1(fit.FitOutputV1):
    """Reuse the existing exclusive writer; prune only its own new products."""
    def __init__(self, directory, storage, reader):
        super().__init__(directory, storage, reader)
        self.owned = {}

    def save(self, name, value, kind="json"):
        record = super().save(name, value, kind)
        self.owned[record["path"]] = (record, (self.path / name).stat().st_size + 8192)
        return record

    def discard(self, record):
        known = self.owned.get(record["path"])
        path = ROOT / record["path"]
        if (known is None or known[0] != record or path.parent != self.path
                or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError("Prune only unchanged predictions/checkpoints created in this new output")
        path.unlink()
        self.used -= known[1]
        del self.owned[record["path"]]


def resources_v1(spec, storage, lease):
    from .biological_candidate import check_fit_lease
    fit._check_active_v1(spec)
    if fit.datetime.now(fit.timezone.utc) >= fit._utc(storage["expires_at_utc"]):
        raise ValueError("Pilot output reservation expired")
    check_fit_lease(lease)
    fit.require_logical_job_v1(lease, spec)
    fit.budget_check_v1(storage)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > spec["max_cpu_rss_bytes"]:
        raise ValueError("Pilot CPU RSS peak exceeds root cap")
    if fit.torch.cuda.max_memory_reserved() > spec["max_cuda_reserved_bytes"]:
        raise ValueError("Pilot CUDA allocator peak exceeds root cap")


def inputs_v1(record):
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    lease = validate_lease()
    apply_and_validate_title()
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "persistence_trace_pilot_input_v1"
            or spec.get("root_authorizer") != "root_agent" or spec.get("fit_authorized") is not True
            or spec.get("status") != "active" or spec.get("seed") != 101
            or spec.get("device") != "cuda" or not fit.canonical_job_v1(spec.get("job_id"))):
        raise ValueError("Pinned active root seed101 pilot input required")
    fit._check_active_v1(spec)
    fit.require_logical_job_v1(lease, spec)
    cfg = reader.decode_json(spec["refs"]["config"])
    if spec["refs"]["config"]["path"] != CONFIG or cfg["hypothesis_id"] != "H025":
        raise ValueError("Exact H025 pilot config required")
    if (cfg["arms"] != list(ARMS) or cfg["seeds"] != [101] or cfg["dtype"] != "float32"
            or type(spec.get("epochs", 10)) is not int or not 1 <= spec.get("epochs", 10) <= 10
            or any(type(spec.get(k)) is not int or spec[k] <= 0
                   for k in ("max_cpu_rss_bytes", "max_cuda_reserved_bytes"))):
        raise ValueError("One FP32 pilot seed and at most10epochs required")
    if spec["source_sha256"].get(SOURCE) != fit._hash(SOURCE):
        raise ValueError("Root must bind this driver source")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    for ref in cfg["helper_refs"].values():
        reader.read(ref)
    base = reader.decode_json(cfg["base_fit_config"])
    expected_metrics = dict(base["metrics"], aggregation="unweighted_parent_mean_one_seed_no_probability_ensemble")
    if cfg["metrics"] != expected_metrics or set(spec["observed_parents"]) != set(fit.PARENTS):
        raise ValueError("Original four-parent split and scoring policy required")
    reader.read(spec["refs"]["runtime"])
    storage = reader.decode_json(spec["refs"]["storage"])
    reservation = storage["reservation"]
    destination = (reader.root / reader.relative(spec["output_directory"])).resolve()
    if (storage["job_id"] != spec["job_id"] or storage["output_directory"] != spec["output_directory"]
            or not 0 < storage["reserved_new_bytes"] <= cfg["output_cap_bytes"]
            or reservation["reserved_bytes"] != storage["reserved_new_bytes"]
            or reservation["output"] != str(destination)
            or any(ref not in reservation["consumer_bindings"] for ref in (
                {"path": SOURCE, "sha256": spec["source_sha256"][SOURCE]},
                spec["refs"]["config"], spec["refs"]["runtime"]))):
        raise ValueError("Separate bounded root v4 pilot reservation required")
    fit.budget_check_v1(storage, first=True)
    emitter = reader.decode_json(spec["refs"]["emitter_bindings"])
    if (emitter["emitter_condition"] != cfg["emitter_condition"]
            or emitter["training_predictions"] != "in_sample_not_out_of_fold"
            or emitter["class_order"] != CLASS_ORDER or set(emitter["parents"]) != set(fit.PARENTS)):
        raise ValueError("Explicit four-parent frozen Relate-emitter hybrid binding required")
    reader.read(emitter["fitted_checkpoint_ref"])
    reader.read(emitter["native_fit_ref"])
    bindings = reader.decode_json(spec["refs"]["target_bindings"])
    metadata = reader.decode_json(base["external_target_manifest"])
    by_id = {row["parent_id"]: row for row in metadata["replicates"]}
    if set(bindings["parents"]) != set(fit.PARENTS):
        raise ValueError("Exact external target joins for the four existing parents required")
    for parent_id, binding in bindings["parents"].items():
        truth = f"data/biological_paths_execution_pilot_v1/{parent_id}/truth/full_truth_intervals.npz"
        if (binding["truth_npz"] != {"path": truth,
                "sha256": by_id[parent_id]["files"]["truth/full_truth_intervals.npz"]["sha256"]}
                or binding["split"] != by_id[parent_id]["split"]):
            raise ValueError("Original external C4 target manifest binding changed")
    # Initializes unchanged target/scoring helpers only, with no numerical run.
    fit._science_v1()
    return spec, cfg, base, emitter, bindings, reader, storage


def roster_v1(rows):
    result = {(row["sample_id"], row["haplotype"]): row for row in rows}
    if len(result) != len(rows) or len(rows) != 280:
        raise ValueError("Unique complete observed280 sample identity roster required")
    return result


def native_sources_v1(ref, emitter_nodes, centers, offset, reader):
    import h5py
    np = fit.np
    reader.read(ref)
    with h5py.File(ROOT / ref["path"], "r") as h5:
        probabilities = np.asarray(h5["refined_probs"], dtype=np.float32)
        nodes = np.asarray(h5["node_id"], dtype=np.int64)
        starts = np.asarray(h5["tree_start"], dtype=np.float64)
        ends = np.asarray(h5["tree_end"], dtype=np.float64)
    if (probabilities.shape != (len(nodes), 4) or starts.shape != nodes.shape or ends.shape != nodes.shape
            or not np.isfinite(probabilities).all() or np.any(probabilities <= 0)
            or not np.allclose(probabilities.sum(-1), 1, atol=1e-5, rtol=0)
            or not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends <= starts)
            or set(nodes.tolist()) != set(emitter_nodes.tolist())):
        raise ValueError("Finite positive native refined C4 query rows required; no probability floor")
    x = centers - offset
    source = np.empty((len(x), 120, 4), dtype=np.float32)
    for query, node in enumerate(emitter_nodes):
        rows = np.flatnonzero(nodes == node)
        rows = rows[np.argsort(starts[rows], kind="stable")]
        left, right = starts[rows], ends[rows]
        if not np.array_equal(right[:-1], left[1:]):
            raise ValueError("Native refined query intervals must have complete nonoverlapping support")
        indexes = np.searchsorted(left, x, side="right") - 1
        if np.any(indexes < 0) or np.any(x >= right[indexes]):
            raise ValueError("Frozen emitter lacks a query source at an exact tsinfer hold center")
        source[:, query] = probabilities[rows[indexes]]
    return fit.torch.from_numpy(np.log(source).reshape(-1, 4))


def prepare_v1(spec, cfg, emitter, bindings, reader, helpers, output, storage, lease):
    import tskit
    from .persistence_trace_kernel_v1 import MetricTreeHoldV1, prepare_persistence_trace_bank_v1
    np, torch = fit.np, fit.torch
    cached, parents, train_ids, val_ids, diagnostics = {}, {}, [], {}, []
    tmp = output.path / "tmp"
    tmp.mkdir()
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
            raise ValueError("Complete actual/native observed sample and direct-date join differs")
        parent["probabilities"] = probabilities
        parent["unaries"] = torch.from_numpy(np.log(probabilities))
        parents[parent_id] = parent
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
            bank = prepare_persistence_trace_bank_v1(
                trees, intervals, [query_ids] * len(trees),
                torch.from_numpy(parent["positions"][indexes]), query_ids,
                sample_identity_namespace=parent_id + ":" + source_ref["sha256"],
                sham_seed=101, **cfg["geometry"])
            source_logits = native_sources_v1(frame["native_refined_HDF5"], emitter_nodes,
                                              intervals.mean(1).numpy(), offset, reader)
            identity = f"{parent_id}:{tile_id}"
            cached[identity] = dict(parent_id=parent_id, indexes=indexes, bank=bank,
                                    source=source_logits, direct=parent["unaries"][indexes])
            diagnostics.append(dict(identity=identity, split=split,
                native_identity_namespace=namespace, emitter_genealogy="Relate",
                trace_genealogy="independent_observed_tsinfer", **bank.metadata))
            (train_ids if split == "train" else val_ids.setdefault(parent_id, [])).append(identity)
            print(json.dumps({"stage": "persistence_pilot_prepare", "identity": identity,
                              "states": bank.num_states, "metadata": bank.metadata}), flush=True)
            resources_v1(spec, storage, lease)
        del ts
    if len(cached) != 32 or len(train_ids) != 16 or len(val_ids) != 2:
        raise ValueError("Original32 tiles,16 TRAIN tiles and two VAL parents required")
    targets = {}
    for parent_id, parent in parents.items():
        if parent["split"] == "train":
            target = fit.ExternalTargetV1(parent, bindings["parents"][parent_id], "TRAIN_loss", reader)
            targets[parent_id] = target.target_mass(parent, helpers)
    output.save("geometry.json", {"tiles": diagnostics, "source_samples": "actual_Q120_only",
        "hybrid_condition": cfg["condition"], "unaries": "canonical_log_frozen_native_averaged_probabilities",
        "TRAIN_emitter_predictions": "in_sample_not_out_of_fold"})
    return cached, parents, train_ids, val_ids, targets


def models_v1(cfg, device):
    torch = fit.torch
    from .persistence_trace_kernel_v1 import PersistenceTraceKernelV1
    class GenomicSmoother(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_projection = torch.nn.Linear(4, cfg["channels"], bias=False)
            self.mixture_logits = torch.nn.Parameter(torch.zeros(cfg["channels"], 3))
            self.residual_output = torch.nn.Linear(4 + cfg["channels"], 4)
            torch.nn.init.zeros_(self.residual_output.weight)
            torch.nn.init.zeros_(self.residual_output.bias)

        def forward(self, direct):
            signal = self.source_projection(direct.detach()).tanh().permute(1, 2, 0)
            messages = torch.stack([torch.nn.functional.avg_pool1d(
                signal, int(width / 100) // 2 * 2 + 1, stride=1,
                padding=int(width / 100) // 2, count_include_pad=False)
                for width in cfg["smoother_window_bp"]], -1)
            message = (messages * self.mixture_logits.softmax(-1)[None, :, None, :]).sum(-1)
            message = message.permute(2, 0, 1)
            return direct.detach() + self.residual_output(torch.cat((direct.detach(), message), -1))
    torch.manual_seed(101); fit.np.random.seed(101); random.seed(101)
    initial = PersistenceTraceKernelV1(num_classes=4, channels=cfg["channels"]).to(device=device)
    models = {arm: deepcopy(initial) for arm in ARMS[1:]}
    smoother = GenomicSmoother().to(device=device)
    smoother.source_projection.load_state_dict(initial.source_projection.state_dict())
    smoother.residual_output.load_state_dict(initial.residual_output.state_dict())
    models[ARMS[0]] = smoother
    states = {}
    for arm in ARMS:
        model = models[arm]
        rate = [p for name, p in model.named_parameters()
                if name.startswith("raw_") or name == "mixture_logits"]
        ids = {id(p) for p in rate}
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in ids], "weight_decay": cfg["weight_decay"]},
            {"params": rate, "weight_decay": 0.}], lr=cfg["learning_rate"])
        states[arm] = dict(model=model, optimizer=optimizer, steps=0, best=float("inf"),
            best_epoch=None, history=[], current=[], retained_best=[],
            initial={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    return states


def seal_v1(output, arm, epoch, parent, logits=None):
    probabilities = (parent["probabilities"] if logits is None
                     else fit.torch.from_numpy(logits).softmax(-1).numpy())
    return output.save(f'{arm}.epoch_{epoch:02d}.{parent["parent_id"]}.npz', {
        "C4_probabilities": probabilities, "positions_bp": parent["positions"],
        "bin_edges_bp": parent["edges"], "retained_fixed100bp_indexes": parent["retained_indexes"],
        "query_sample_indexes": parent["query_slots"], **({} if logits is None else {"logits": logits})}, "npz")


def score_v1(seals, parents, bindings, reader, helpers, cfg):
    cells = []
    for parent_id, record in seals:
        cell = fit.score_sealed_v1(record, parents[parent_id], bindings["parents"][parent_id], reader, helpers, cfg)
        for group in cell["groups"].values():
            group.pop("per_query"); group.pop("retained_original_grid_indexes")
        cells.append(cell)
    means = {}
    for group in cells[0]["groups"]:
        values = [cell["groups"][group] for cell in cells]
        infinite = any(value["infinite_nll_query_bp"] for value in values)
        means[group] = {name: float(fit.np.mean([value[name] for value in values]))
                       for name in ("accuracy", "brier", "calibration_ece")}
        means[group].update(nll=None if infinite else float(fit.np.mean([value["nll"] for value in values])),
                            NLL_is_infinite=infinite)
    return cells, means


def run_v1(record):
    spec, cfg, base, emitter, bindings, reader, storage = inputs_v1(record)
    output = PilotOutputV1(spec["output_directory"], storage, reader)
    from .biological_candidate import configure_fit
    from .biological_gst_control_v1 import exact_span_loss
    device, runtime, lease = configure_fit("cuda", base, output.path)
    output.used = sum(max(path.stat().st_size, path.stat().st_blocks * 512) + 8192
                      for path in output.path.rglob("*") if path.is_file())
    helpers = fit.metric_helpers_v1()
    epochs = spec.get("epochs", cfg["max_epochs"])
    receipt = dict(schema="persistence_trace_pilot_evidence_v1", hypothesis_id="H025",
        status="running", input_ref=record, config_ref=spec["refs"]["config"], condition=cfg["condition"],
        emitter_condition=cfg["emitter_condition"], trace_condition=cfg["trace_condition"],
        seed=101, dtype="float32", epochs=epochs, optimizer_steps_per_arm=epochs * 16,
        runtime=runtime, emitter_binding_ref=spec["refs"]["emitter_bindings"],
        TRAIN_emitter_predictions="in_sample_not_out_of_fold", TEST_parents=0, models={})
    output.save("started.json", receipt)
    start = time.monotonic()
    try:
        cached, parents, train_ids, val_ids, targets = prepare_v1(
            spec, cfg, emitter, bindings, reader, helpers, output, storage, lease)
        states = models_v1(cfg, device)
        plain_seals = [(p, seal_v1(output, "plain_ARGMix", 0, parents[p])) for p in val_ids]
        plain_score = None
        for epoch in range(1, epochs + 1):
            epoch_start = time.monotonic()
            order = list(range(16)); random.Random(101 + epoch).shuffle(order)
            losses = {arm: [] for arm in ARMS}
            for index in order:
                tile = cached[train_ids[index]]
                bank = tile["bank"].to(device)
                source, direct = tile["source"].to(device), tile["direct"].to(device)
                mass = fit.torch.as_tensor(targets[tile["parent_id"]][tile["indexes"]],
                                           dtype=fit.torch.float32, device=device)
                for arm in ARMS:
                    resources_v1(spec, storage, lease)
                    state = states[arm]; model = state["model"]; model.train()
                    state["optimizer"].zero_grad(set_to_none=True)
                    logits = model(direct) if arm == "genomic_smoothing" else model(bank, source, direct, control=arm)
                    loss = exact_span_loss(logits, mass)
                    if not bool(fit.torch.isfinite(loss)):
                        raise ValueError("Nonfinite pilot TRAIN loss")
                    loss.backward()
                    fit.torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
                    state["optimizer"].step(); state["steps"] += 1
                    if any(not bool(fit.torch.isfinite(p).all()) for p in model.parameters()):
                        raise ValueError("Nonfinite fitted pilot parameter")
                    losses[arm].append(float(loss.detach()))
                del bank, source, direct, mass
            seals = {arm: [] for arm in ARMS}
            for parent_id, identities in val_ids.items():
                parent = parents[parent_id]
                values = {arm: fit.np.empty_like(parent["probabilities"]) for arm in ARMS}
                seen = fit.np.zeros(len(parent["positions"]), dtype=bool)
                for identity in identities:
                    tile = cached[identity]
                    if seen[tile["indexes"]].any():
                        raise ValueError("Repeated VAL grid rows")
                    bank = tile["bank"].to(device)
                    source, direct = tile["source"].to(device), tile["direct"].to(device)
                    with fit.torch.no_grad():
                        for arm in ARMS:
                            resources_v1(spec, storage, lease)
                            model = states[arm]["model"]; model.eval()
                            logits = model(direct) if arm == "genomic_smoothing" else model(bank, source, direct, control=arm)
                            if not bool(fit.torch.isfinite(logits).all()):
                                raise ValueError("Nonfinite full-support VAL logits")
                            values[arm][tile["indexes"]] = logits.cpu().numpy()
                    seen[tile["indexes"]] = True
                    del bank, source, direct
                if not seen.all():
                    raise ValueError("Missing VAL bins; denominator cannot be reduced")
                for arm in ARMS:
                    seals[arm].append((parent_id, seal_v1(output, arm, epoch, parent, values[arm])))
            # Every arm and the plain emitter are sealed before the first VAL target read.
            if plain_score is None:
                cells, means = score_v1(plain_seals, parents, bindings, reader, helpers, cfg)
                plain_score = dict(arm="plain_ARGMix", epoch=0, optimizer_steps=0, cells=cells,
                                   equal_parent_metrics=means)
                output.save("plain_ARGMix.metrics.json", plain_score)
            for arm in ARMS:
                resources_v1(spec, storage, lease)
                cells, means = score_v1(seals[arm], parents, bindings, reader, helpers, cfg)
                primary = means["primary_present_day_64"]
                nll = float("inf") if primary["NLL_is_infinite"] else primary["nll"]
                state = states[arm]
                previous = {r["path"]: r for r in state["current"] + state["retained_best"]}
                checkpoint = output.save(f"{arm}.epoch_{epoch:02d}.pt", {
                    "model_state": {k: v.detach().cpu() for k, v in state["model"].state_dict().items()},
                    "optimizer_state": state["optimizer"].state_dict(), "epoch": epoch, "arm": arm,
                    "seed": 101, "input_ref": record, "condition": cfg["condition"]}, "torch")
                state["current"] = [r for _, r in seals[arm]] + [checkpoint]
                row = dict(epoch=epoch, arm=arm, optimizer_steps=state["steps"],
                    TRAIN_mean_tile_query120_NLL=float(fit.np.mean(losses[arm])),
                    cells=cells, equal_parent_metrics=means, checkpoint=checkpoint)
                state["history"].append(row)
                if state["best_epoch"] is None or nll < state["best"]:
                    state.update(best=nll, best_epoch=epoch, selected=row,
                                 retained_best=list(state["current"]))
                row["selected_epoch"] = state["best_epoch"]
                output.save(f"{arm}.epoch_{epoch:02d}.metrics.json", row)
                keep = {r["path"] for r in state["current"] + state["retained_best"]}
                for path, product in previous.items():
                    if path not in keep:
                        output.discard(product)
            print(json.dumps({"stage": "persistence_pilot_epoch", "epoch": epoch,
                "steps_per_arm": epoch * 16, "elapsed_seconds": time.monotonic() - epoch_start,
                "primary_present64": {arm: states[arm]["history"][-1]["equal_parent_metrics"]["primary_present_day_64"]
                                      for arm in ARMS}}), flush=True)
        table = [dict(arm="plain_ARGMix", selected_epoch=0,
                      **plain_score["equal_parent_metrics"]["primary_present_day_64"])]
        for arm, state in states.items():
            if state["steps"] != epochs * 16:
                raise ValueError("Incomplete one-seed pilot budget")
            selected = state["selected"]
            saved = fit.torch.load(io.BytesIO(reader.read(selected["checkpoint"])),
                                   map_location="cpu", weights_only=True)
            delta = max(float((v - state["initial"][key]).abs().max())
                        for key, v in saved["model_state"].items() if v.is_floating_point())
            if not delta > 0:
                raise ValueError("Selected pilot checkpoint has no trained weight change")
            receipt["models"][arm] = dict(selected=selected, steps=state["steps"],
                selected_weight_max_delta=delta,
                parameter_count=sum(p.numel() for p in state["model"].parameters()),
                retained_records={r["path"]: r for r in state["current"] + state["retained_best"]})
            table.append(dict(arm=arm, selected_epoch=state["best_epoch"],
                **selected["equal_parent_metrics"]["primary_present_day_64"]))
        output.save("comparison.json", dict(primary="equal_parent_VAL_present64", rows=table,
            plain_ARGMix=plain_score, condition=cfg["condition"], development_only=True))
        resources_v1(spec, storage, lease)
        reader.read(record)
        receipt.update(status="completed_one_seed_FP32_architecture_pilot",
                       comparison=table, elapsed_seconds=time.monotonic() - start,
                       output_accounted_bytes=output.used)
    except Exception as error:
        receipt.update(status="failed_partial_preserved", error_type=type(error).__name__,
                       error=str(error), elapsed_seconds=time.monotonic() - start,
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
        print(json.dumps({"status": "source_only_not_executed", "hypothesis_id": "H025",
            "implementation": {"path": SOURCE, "sha256": fit._hash(SOURCE)},
            "config": {"path": CONFIG, "sha256": fit._hash(CONFIG)}, "seed": 101,
            "max_epochs": 10, "dtype": "float32"}, indent=2))
        return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Pinned root pilot input required")
    run_v1({"path": args.input, "sha256": args.expected_input_sha256})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

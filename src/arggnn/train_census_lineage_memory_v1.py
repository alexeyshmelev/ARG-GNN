"""Root-launched three-arm observed census-lineage fit.

The root pins a runtime input record and supplies the held GPU lease and v4
storage reservation. Genuine original numerical terminal evidence is required.
All scientific loading, training and scoring use existing unchanged fit APIs.
Only this driver owns its new output; original results are never pruned.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import resource
import time

from . import census_tsinfer_fit_v1 as fit

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/train_census_lineage_memory_v1.py"
ADAPTER = "src/arggnn/census_lineage_memory_v1.py"
CONFIG = "configs/census_lineage_memory_v1.json"
ARMS = ("current_jaccard", "genuine_edge", "sham_edge")
OUTPUT_CAP_BYTES = 20_000_000_000


class RetainedOutputV1(fit.FitOutputV1):
    """Existing exclusive writer, with pruning limited to files it just wrote."""
    def __init__(self, directory, storage, reader):
        super().__init__(directory, storage, reader)
        self.owned = {}

    def save(self, name, value, kind="json"):
        record = super().save(name, value, kind)
        self.owned[record["path"]] = (record, (self.path / name).stat().st_size + 8192)
        return record

    def discard(self, record):
        known = self.owned.get(record["path"])
        if known is None or known[0] != record:
            raise ValueError("Pruning requires an unchanged record created by this driver")
        path = ROOT / record["path"]
        if path.parent != self.path or path.is_symlink():
            raise ValueError("Pruning is restricted to this exclusive output directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Owned output changed before pruning")
        path.unlink()
        self.used -= known[1]
        del self.owned[record["path"]]


def check_resources_v1(spec, storage, lease):
    from .biological_candidate import check_fit_lease
    fit._check_active_v1(spec)
    if fit.datetime.now(fit.timezone.utc) >= fit._utc(storage["expires_at_utc"]):
        raise ValueError("Root output reservation expired")
    check_fit_lease(lease)
    fit.require_logical_job_v1(lease, spec)
    fit.budget_check_v1(storage)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > spec["max_cpu_rss_bytes"]:
        raise ValueError("Process CPU RSS peak exceeds root reservation")
    if fit.torch.cuda.max_memory_reserved() > spec["max_cuda_reserved_bytes"]:
        raise ValueError("CUDA allocator peak exceeds root reservation")


def root_inputs_v1(record):
    """Read the root's explicit bindings; do not activate the old fit admission."""
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    lease = validate_lease()
    apply_and_validate_title()
    reader = fit.ConsumerReaderV1()
    spec = reader.decode_json(record)
    if (spec.get("schema") != "census_lineage_memory_runtime_input_v1"
            or spec.get("status") != "active"
            or spec.get("root_authorizer") != "root_agent"
            or spec.get("fit_authorized") is not True
            or not fit.canonical_job_v1(spec.get("job_id"))
            or spec.get("device") != "cuda"
            or any(type(spec.get(k)) is not int or spec[k] <= 0
                   for k in ("max_cpu_rss_bytes", "max_cuda_reserved_bytes"))):
        raise ValueError("Explicit active root three-arm CUDA runtime input required")
    fit._check_active_v1(spec)
    fit.require_logical_job_v1(lease, spec)
    refs = spec["refs"]
    experiment = reader.decode_json(refs["config"])
    if refs["config"]["path"] != CONFIG or experiment["arms"] != list(ARMS):
        raise ValueError("Pinned three-arm experiment config required")
    base_ref = experiment["base_scientific_config"]
    if base_ref["path"] != fit.CONFIG:
        raise ValueError("Unchanged same-tsinfer scientific config required")
    reader.read(base_ref)
    cfg = fit.load_config_v1()
    if any(experiment[k] != cfg[k] for k in ("training", "model", "seeds", "readout_chunk_size")):
        raise ValueError("Matched original model, optimizer and fixed-budget recipe required")
    if (type(spec.get("seed")) is not int or spec["seed"] not in cfg["seeds"]
            or cfg["training"]["epochs"] != 30
            or cfg["training"]["TRAIN_tiles_per_epoch"] != 16
            or cfg["training"]["optimizer_steps_per_control_seed"] != 480):
        raise ValueError("Original seed and 30-epoch/480-step budget required")
    for path in (SOURCE, ADAPTER, fit.SOURCE):
        if spec["source_sha256"].get(path) != fit._hash(path):
            raise ValueError("Root input must pin driver, adapter and existing fit source")
    for path, digest in spec["source_sha256"].items():
        reader.read({"path": path, "sha256": digest})
    fit.source_report_v1()
    # This requires genuine terminal exit0, complete32 numerical evidence and
    # the original inspection/freeze/lineage; no fabricated numerical success.
    native, qcfg = fit.gate_from_reader_v1(spec, reader)
    if (spec["expected_certificates"] != native["expected_certificates"]
            or len(spec["expected_certificates"]) != 32
            or spec["observed_artifact_sha256"] != native["observed_artifact_sha256"]
            or cfg["model"] != qcfg["model"]):
        raise ValueError("Root runtime input differs from the frozen32 original observations")
    storage = reader.decode_json(refs["storage"])
    reservation = storage["reservation"]
    output_path = (reader.root / reader.relative(spec["output_directory"])).resolve()
    if (storage["output_directory"] != spec["output_directory"]
            or storage["job_id"] != spec["job_id"]
            or type(storage["reserved_new_bytes"]) is not int
            or not 0 < storage["reserved_new_bytes"] <= OUTPUT_CAP_BYTES
            or reservation["reserved_bytes"] != storage["reserved_new_bytes"]
            or reservation["output"] != str(output_path)
            or reservation["auxiliary_scopes"] != []
            or any(ref not in reservation["consumer_bindings"] for ref in (
                {"path": SOURCE, "sha256": spec["source_sha256"][SOURCE]},
                refs["config"], refs["runtime"]))):
        raise ValueError("Separate root v4 reservation of at most20GB required")
    fit.budget_check_v1(storage, first=True)
    fit.verify_observation_closure_v1(native, reader)
    # Existing runtime validation initializes the unchanged _science_v1().
    fit.runtime_v1(reader.decode_json(refs["runtime"]), native, reader,
                   "census_lineage_memory_runtime_v1")
    bindings = reader.decode_json(refs["target_bindings"])
    metadata = reader.decode_json(refs["target_manifest"])
    fit.target_bindings_v1(bindings, metadata, spec, cfg)
    return spec, cfg, native, qcfg, bindings, reader, storage


def edge_summary_v1(memory, original):
    result = memory.summary()
    counts = dict(empty=0, singleton=0, homogeneous=0, permutable=0)
    for query in range(memory.eligible.shape[1]):
        for span_bin in range(4):
            rows = memory.eligible[:, query] & (memory.span_bins == span_bin)
            n = int(rows.sum())
            changes = int(memory.genuine_changed[rows, query].sum())
            category = ("empty" if n == 0 else "singleton" if n == 1
                        else "homogeneous" if changes in (0, n) else "permutable")
            counts[category] += 1
    result["sham_strata_counts"] = counts
    result["degenerate_nonempty_sham_strata"] = counts["singleton"] + counts["homogeneous"]
    rho = original.outer.native_certificate.bank.boundary_rho[list(memory.boundary_events)]
    result["membership_stable_edge_changed_event_queries"] = int(
        ((rho == 1) & memory.genuine_changed).sum())
    return result


def precompute_v1(spec, native, qcfg, bindings, reader, helpers, output, storage, lease):
    from .census_lineage_memory_v1 import prepare_lineage_memory_v1
    paired = fit.read_manifests_v1(native, qcfg, reader)
    tmp = output.path / "tmp"
    tmp.mkdir()
    cached = {}
    parents = {}
    train_tiles = []
    val_tiles = {}
    summaries = []
    for parent, bank in paired:
        check_resources_v1(spec, storage, lease)
        loaded = fit.load_parent_v1(parent, bank, native, qcfg, reader, tmp_directory=tmp)
        parent_id = parent["parent_id"]
        parents[parent_id] = fit.scoring_parent_v1(
            parent, bank, loaded, reader, bindings["parents"][parent_id])
        for tile in sorted(bank["tiles"], key=lambda value: value["tile_id"]):
            check_resources_v1(spec, storage, lease)
            identity = f'{parent_id}:{tile["tile_id"]}'
            original, _, _ = fit.prepare_tile_v1(parent, bank, tile, loaded, native, qcfg, reader)
            expected = spec["expected_certificates"][identity]
            fit.validate_census_biological_certificate_v1(original, expected)
            memory = prepare_lineage_memory_v1(
                original.outer, expected, seed=spec["seed"],
                experiment_config_sha256=spec["refs"]["config"]["sha256"])
            arms = {arm: fit.ConsumerCertificateV1(memory.certificate_for_arm(arm)) for arm in ARMS}
            cached[identity] = {
                "original": original, "arms": arms,
                "indexes": fit.np.asarray(tile["position_indexes"], dtype=int),
                "parent_id": parent_id,
            }
            summaries.append(dict(edge_summary_v1(memory, original), identity=identity,
                                  split=parent["split"],
                                  original_certificate_sha256=expected))
            if parent["split"] == "train":
                train_tiles.append(identity)
            else:
                val_tiles.setdefault(parent_id, []).append(identity)
            print(json.dumps({"stage": "census_lineage_cache_tile", "identity": identity,
                              "cached_tiles": len(cached)}), flush=True)
        del loaded
    if (set(cached) != set(spec["expected_certificates"]) or len(train_tiles) != 16
            or len(val_tiles) != 2 or any(len(tiles) != 8 for tiles in val_tiles.values())):
        raise ValueError("All32 original tiles and two independent parents per split required")
    # Targets enter only after all observed original/derived certificates exist.
    targets = {}
    for parent_id, parent in parents.items():
        if parent["split"] == "train":
            target = fit.ExternalTargetV1(parent, bindings["parents"][parent_id], "TRAIN_loss", reader)
            targets[parent_id] = target.target_mass(parent, helpers)
    output.save("edge_events.json", {"tiles": summaries, "observed_tiles": 32,
        "raw_prediction_retention": "current_and_selected_best_epoch_only",
        "compact_metrics_omit": ["per_query", "retained_original_grid_indexes"]})
    return cached, parents, train_tiles, val_tiles, targets


def compact_score_v1(record, parent, binding, reader, helpers, cfg):
    cell = fit.score_sealed_v1(record, parent, binding, reader, helpers, cfg)
    for group in cell["groups"].values():
        group.pop("per_query")
        group.pop("retained_original_grid_indexes")
    return cell


def run_v1(input_record):
    spec, cfg, native, qcfg, bindings, reader, storage = root_inputs_v1(input_record)
    output = RetainedOutputV1(spec["output_directory"], storage, reader)
    from .biological_candidate import configure_fit
    device, runtime, lease = configure_fit(spec["device"], cfg, output.path)
    output.used = sum(max(p.stat().st_size, p.stat().st_blocks * 512) + 8192
                      for p in output.path.rglob("*") if p.is_file())
    check_resources_v1(spec, storage, lease)
    helpers = fit.metric_helpers_v1()
    receipt = {
        "schema": "census_lineage_memory_fit_evidence_v1", "status": "precomputing",
        "job_id": spec["job_id"], "seed": spec["seed"], "input_ref": input_record,
        "config_ref": spec["refs"]["config"], "source_sha256": spec["source_sha256"],
        "numerical_ref": spec["refs"]["numerical_evidence"], "runtime": runtime,
        "arms": list(ARMS), "model_control_all_arms": "proposed", "models": {},
        "development_VAL_parents": 2, "TEST_parents": 0, "target_inputs": False,
        "raw_prediction_retention": "current_and_selected_best_epoch_only",
    }
    output.save("started.json", receipt)
    start = time.monotonic()
    try:
        cached, parents, train_tiles, val_tiles, targets = precompute_v1(
            spec, native, qcfg, bindings, reader, helpers, output, storage, lease)
        stock = fit.matched_states_v1(cfg, spec["seed"], device)
        states = {arm: stock.pop(control) for arm, control in zip(
            ARMS, ("proposed", "eta0", "permuted_groups"))}
        del stock
        if len({state["initial_sha256"] for state in states.values()}) != 1:
            raise ValueError("Original matched initialization differs")
        receipt["same_initial_weights_sha256"] = states[ARMS[0]]["initial_sha256"]
        for state in states.values():
            state.update(current_records=[], best_records=[])
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            order = list(range(len(train_tiles)))
            random.Random(spec["seed"] + epoch).shuffle(order)
            losses = {arm: [] for arm in ARMS}
            for index in order:
                tile = cached[train_tiles[index]]
                mass = targets[tile["parent_id"]][tile["indexes"]]
                for arm in ARMS:
                    check_resources_v1(spec, storage, lease)
                    cert = tile["arms"][arm]
                    result = fit.train_step_v1(
                        states[arm], cert, cert.outer.certificate_sha256, mass,
                        "proposed", None, cfg, device)
                    losses[arm].append(result["loss"])
            # Seal every arm's two full-support VAL predictions before scoring.
            seals = {arm: [] for arm in ARMS}
            for parent_id, identities in val_tiles.items():
                parent = parents[parent_id]
                logits = {arm: fit.np.empty((len(parent["positions"]), 120, 4), dtype=fit.np.float32)
                          for arm in ARMS}
                seen = fit.np.zeros(len(parent["positions"]), dtype=bool)
                for identity in identities:
                    tile = cached[identity]
                    indexes = tile["indexes"]
                    if seen[indexes].any():
                        raise ValueError("Repeated VAL physical output bins")
                    for arm in ARMS:
                        check_resources_v1(spec, storage, lease)
                        cert = tile["arms"][arm]
                        values, _, ids = fit.raw_prediction_v1(
                            states[arm], cert, cert.outer.certificate_sha256,
                            "proposed", None, cfg)
                        if not fit.np.array_equal(ids, parent["query_ids"]):
                            raise ValueError("VAL actual query output identities differ")
                        logits[arm][indexes] = values
                    seen[indexes] = True
                if not seen.all():
                    raise ValueError("Missing VAL bins, no denominator reduction")
                for arm in ARMS:
                    values = logits[arm]
                    seal = output.save(f"{arm}.epoch_{epoch:02d}.{parent_id}.npz", {
                        "logits": values,
                        "C4_probabilities": fit.torch.from_numpy(values).softmax(-1).numpy(),
                        "positions_bp": parent["positions"], "bin_edges_bp": parent["edges"],
                        "retained_fixed100bp_indexes": parent["retained_indexes"],
                        "query_sample_indexes": parent["query_slots"],
                        "normalization_length_bp": fit.np.array(2000000),
                    }, "npz")
                    seals[arm].append((parent_id, seal))
                del logits
            for arm in ARMS:
                check_resources_v1(spec, storage, lease)
                cells = [compact_score_v1(
                    seal, parents[parent_id], bindings["parents"][parent_id], reader, helpers, cfg)
                    for parent_id, seal in seals[arm]]
                nll = float(fit.np.mean([
                    fit._nll_v1(cell["groups"]["primary_present_day_64"]) for cell in cells]))
                state = states[arm]
                previous = {r["path"]: r for r in state["current_records"] + state["best_records"]}
                checkpoint = output.save(f"{arm}.epoch_{epoch:02d}.pt", {
                    "model_state": {k: v.detach().cpu() for k, v in state["model"].state_dict().items()},
                    "optimizer_state": state["optimizer"].state_dict(),
                    "epoch": epoch, "arm": arm, "control": "proposed",
                    "seed": spec["seed"], "input_ref": input_record,
                }, "torch")
                state["current_records"] = [r for _, r in seals[arm]] + [checkpoint]
                if state["best_epoch"] is None or nll < state["best"]:
                    state.update(best=nll, best_epoch=epoch, checkpoint=checkpoint,
                                 best_records=list(state["current_records"]))
                row = {
                    "epoch": epoch, "arm": arm, "optimizer_steps": state["steps"],
                    "TRAIN_mean_tile_query120_NLL": float(fit.np.mean(losses[arm])),
                    "VAL_equal_parent_present64_NLL": nll if fit.np.isfinite(nll) else None,
                    "VAL_NLL_is_infinite": not fit.np.isfinite(nll), "cells": cells,
                    "current_checkpoint": checkpoint, "selected_epoch": state["best_epoch"],
                    "selected_checkpoint": state["checkpoint"],
                    "raw_prediction_retention": "current_and_selected_best_epoch_only",
                }
                state["history"].append(row)
                output.save(f"{arm}.epoch_{epoch:02d}.metrics.json", row)
                keep = {r["path"] for r in state["current_records"] + state["best_records"]}
                for path, record in previous.items():
                    if path not in keep:
                        output.discard(record)
            print(json.dumps({"stage": "census_lineage_fit_epoch", "seed": spec["seed"],
                "epoch": epoch, "steps_per_arm": epoch * 16,
                "VAL_equal_parent_present64_NLL": {
                    arm: states[arm]["history"][-1]["VAL_equal_parent_present64_NLL"] for arm in ARMS},
                "selected_epochs": {arm: states[arm]["best_epoch"] for arm in ARMS}}), flush=True)
        for arm, state in states.items():
            if state["steps"] != 480 or state["best_epoch"] is None:
                raise ValueError("Incomplete fixed-budget three-arm fit")
            saved = fit.torch.load(io.BytesIO(reader.read(state["checkpoint"])),
                                   map_location="cpu", weights_only=True)
            delta = max(float((v - state["initial"][k]).abs().max())
                        for k, v in saved["model_state"].items() if v.is_floating_point())
            if not delta > 0:
                raise ValueError("Selected checkpoint has no genuine trained weight change")
            selected = state["history"][state["best_epoch"] - 1]
            means = {}
            for group in fit.strata_v1(parents[next(iter(val_tiles))]):
                cells = [cell["groups"][group] for cell in selected["cells"]]
                infinite = any(cell["infinite_nll_query_bp"] for cell in cells)
                means[group] = {key: float(fit.np.mean([cell[key] for cell in cells]))
                                for key in ("accuracy", "brier", "calibration_ece")}
                means[group].update(
                    nll=None if infinite else float(fit.np.mean([cell["nll"] for cell in cells])),
                    NLL_is_infinite=infinite)
            receipt["models"][arm] = {
                "optimizer_steps": state["steps"], "selected_epoch": state["best_epoch"],
                "checkpoint": state["checkpoint"], "current_checkpoint": state["current_records"][-1],
                "selected_weight_max_delta": delta, "selected_metrics": selected,
                "equal_parent_metrics": means,
                "retained_records": {r["path"]: r for r in state["current_records"] + state["best_records"]},
            }
        check_resources_v1(spec, storage, lease)
        reader.read(input_record)
        reader.read(spec["refs"]["config"])
        for path, digest in spec["source_sha256"].items():
            reader.read({"path": path, "sha256": digest})
        receipt.update(status="completed_three_fixed_budget_development_arms_one_seed",
                       optimizer_fit_completed=True, elapsed_seconds=time.monotonic() - start,
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
        print(json.dumps({"schema": "census_lineage_memory_driver_source_v1",
            "implementation": {"path": SOURCE, "sha256": fit._hash(SOURCE)},
            "config": {"path": CONFIG, "sha256": fit._hash(CONFIG)},
            "status": "source_only_not_executed", "actual_arrays_opened": False,
            "models_created": False, "GPU_used": False}, indent=2))
        return 0
    if not args.input or not args.expected_input_sha256:
        raise ValueError("Pinned root runtime input record required")
    result = run_v1({"path": args.input, "sha256": args.expected_input_sha256})
    print(json.dumps({"status": result["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inactive root v2 launch recipe: one genuine observed certificate on CPU.

Fresh original CPU, extra reuse, storage and memory gates are required.
No actual execution occurred during author preparation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
AUTHOR = "research/reviews/census_certified_forest_reuse_CPU_timing_v2_author"
CONTROLS = ("proposed", "eta0", "permuted_groups", "raw_diagonal", "local")
EXPECTED = "85cf9aa984429a460a15b5be94bb3366f4bf06e65a17812dc7dfb59af3403bf5"
CELL = "bio_paths_pilot_train_000:1"

def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timezone-aware capability dates required")
    return result.astimezone(timezone.utc)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for role in ("base", "reuse", "memory"):
        parser.add_argument("--" + role + "-admission", required=True)
        parser.add_argument("--expected-" + role + "-admission-sha256", required=True)
    parser.add_argument("--expected-pins-sha256", required=True)
    args = parser.parse_args()
    if not args.execute:
        raise ValueError("Root execution flag and fresh capabilities required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("CPU-only disabled-bytecode launcher required")
    refs = {role: {"path": getattr(args, role + "_admission"),
                   "sha256": getattr(args, "expected_" + role + "_admission_sha256")}
            for role in ("base", "reuse", "memory")}
    from arggnn.census_native_qualification_runner_v1 import (
        read_manifests_v1, load_parent_v1, prepare_tile_v1, json_bytes_v1,
        _model, BoundedEvidenceOutputV1, VerifiedReaderV1)
    from cpu_resource_bridge_v2 import admit_cpu_timing_v2, sticky_authority_v2
    from arggnn.census_biological_fit_v1 import require_terminal_numerical_v1
    from arggnn.census_certified_forest_reuse_v1 import admit_reuse_v1, CertifiedForestReuseV1
    from arggnn.census_biological_bank_v1 import (
        validate_census_biological_certificate_v1, fingerprint_sha256_v1)
    import torch
    reader = VerifiedReaderV1()
    pins_ref = {"path": AUTHOR + "/pins.json", "sha256": args.expected_pins_sha256}
    pins = reader.decode_json(pins_ref)
    if (pins["expected_certificate_sha256"] != EXPECTED or pins["controls"] != list(CONTROLS)
            or pins["rounds"] != 2 or pins["dtype"] != "float32"
            or pins["cell"] != {"parent_id": "bio_paths_pilot_train_000", "split": "train",
                               "tile_id": 1, "original_tree_count": 132}):
        raise ValueError("Fixed one-certificate recipe changed")
    history = {key: reader.decode_json(value) for key, value in pins["historical_refs"].items()}
    old = require_terminal_numerical_v1(
        history["numerical_evidence"], history["numerical_admission"],
        history["inspection_evidence"], history["inspection_freeze"],
        history["numerical_terminal"], pins["historical_refs"])
    if old["expected_certificates"][CELL] != EXPECTED:
        raise ValueError("Externally pinned genuine certificate differs")
    self_ref = {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    first_storage_read = True
    @sticky_authority_v2
    def authority(required_write_bytes=0):
        nonlocal first_storage_read
        current, qcfg, checked_reader = admit_cpu_timing_v2(
            refs["base"], phase="numerical", reader=reader,
            first_storage_read=first_storage_read, required_write_bytes=required_write_bytes)
        # Fresh authority changes only job/dates/output/storage/launcher.
        immutable = ("schema", "phase", "source_sha256", "observed_artifact_sha256",
                     "expected_certificates", "inspection_manifest",
                     "initial_state_sha256_by_dtype", "group_permutation")
        if any(current[key] != old[key] for key in immutable):
            raise ValueError("Fresh base differs from sealed original lineage")
        for key in old["refs"]:
            if key not in ("storage", "launcher") and current["refs"][key] != old["refs"][key]:
                raise ValueError("Fresh base changes original config/runtime/review/input pins")
        reuse = admit_reuse_v1(refs["reuse"], phase="observed_profile",
                              base_admission_ref=refs["base"])
        if (reuse["fit_authorized"] is not False
                or reuse["source_ref"] != pins["reuse_source_ref"]
                or reuse["config_ref"] != pins["reuse_config_ref"]
                or reuse["review_ref"] != pins["reuse_review_ref"]):
            raise ValueError("Only accepted observed-profile reuse capability permitted")
        memory = reader.decode_json(refs["memory"])
        fields = {"schema", "status", "root_authorizer", "job_id", "created_at_utc",
                  "expires_at_utc", "CPU_total_reservation_bytes", "GPU_admitted",
                  "external_watchdog", "recipe_ref", "pins_ref", "output_directory"}
        now = datetime.now(timezone.utc)
        if (set(memory) != fields
                or memory["schema"] != "census_forest_reuse_CPU_profile_memory_v1"
                or memory["status"] != "active" or memory["root_authorizer"] != "root_agent"
                or memory["job_id"] != current["job_id"]
                or memory["CPU_total_reservation_bytes"] != 64_000_000_000
                or memory["GPU_admitted"] is not False or memory["external_watchdog"] is not True
                or memory["recipe_ref"] != self_ref or memory["pins_ref"] != pins_ref
                or memory["output_directory"] != current["output_directory"]
                or not utc(memory["created_at_utc"]) <= now < utc(memory["expires_at_utc"])
                or utc(memory["created_at_utc"]) < utc(current["created_at_utc"])
                or utc(memory["expires_at_utc"]) > utc(current["expires_at_utc"])):
            raise ValueError("Fresh root CPU memory reservation and watchdog required")
        if (qcfg["seed"] != 41 or qcfg["model"] != pins["model"]
                or qcfg["readout_chunk_size"] != 128
                or current["initial_state_sha256_by_dtype"]["float32"] != pins["initial_state_sha256"]):
            raise ValueError("Original neural configuration or initial state changed")
        first_storage_read = False
        return current, qcfg, checked_reader

    admitted, qcfg, reader = authority()
    output = BoundedEvidenceOutputV1(admitted["output_directory"],
                                    limit_bytes=qcfg["output_cap_bytes"])
    result = {"schema": "census_certified_forest_reuse_one_certificate_CPU_timing_v2",
              "status": "running", "job_id": admitted["job_id"], "device": "cpu",
              "base_admission_ref": refs["base"], "reuse_admission_ref": refs["reuse"],
              "memory_admission_ref": refs["memory"], "recipe_ref": self_ref, "pins_ref": pins_ref,
              "expected_certificate_sha256": EXPECTED, "selected_cell": pins["cell"],
              "controls": list(CONTROLS), "rounds": 2, "dtype": "float32", "records": [],
              "query_truth_accessed": False, "scientific_fit_performed": False,
              "optimizer_created": False, "parameter_updates": 0, "speedup_claim": False,
              "full_original_observation_closure_hashing_retained": True}
    try:
        tmp = output.path / "tmp"
        tmp.mkdir()
        authority()
        begin = time.perf_counter()
        paired = read_manifests_v1(admitted, qcfg, reader)
        result["complete_observation_closure_validation_seconds"] = time.perf_counter() - begin
        parent, bank = next(pair for pair in paired if pair[0]["parent_id"] == pins["cell"]["parent_id"])
        tile = next(item for item in bank["tiles"] if item["tile_id"] == 1)
        if len(tile["original_tree_ids"]) != 132:
            raise ValueError("Prespecified genuine tree count changed")
        authority()
        begin = time.perf_counter()
        loaded = load_parent_v1(parent, bank, admitted, qcfg, reader, tmp_directory=tmp)
        cert, collision, lineage = prepare_tile_v1(parent, bank, tile, loaded, admitted, qcfg, reader)
        result["parent_load_and_certificate_preparation_seconds"] = time.perf_counter() - begin
        validate_census_biological_certificate_v1(cert, EXPECTED)
        frozen_cell = next(c for c in history["inspection_evidence"]["cells"]
                           if c["parent_id"] == parent["parent_id"] and c["tile_id"] == 1)
        if (len(cert.native_trees) != 132 or collision != frozen_cell["event_collision_counts"]
                or lineage != frozen_cell["decoded_lineage"]):
            raise ValueError("Original inspected certificate/input lineage changed")
        result["event_collision_counts"] = collision
        result["decoded_lineage"] = lineage
        initial = _model(qcfg, "float32")
        if (fingerprint_sha256_v1(initial.state_dict()) != pins["initial_state_sha256"]
                or any(p.grad is not None or p.device.type != "cpu" for p in initial.parameters())):
            raise ValueError("Original CPU initialization or absent gradients changed")
        permutation = torch.tensor(admitted["group_permutation"], dtype=torch.long)
        adapter = CertifiedForestReuseV1()
        scored = cert.bank.query_ids[cert.bank.partitions[0].scored_query_mask]
        for round_id in range(2):
            for control in CONTROLS:
                authority()
                model = deepcopy(initial)
                before = adapter.report()
                kwargs = {"group_permutation": permutation} if control == "permuted_groups" else {}
                begin = time.perf_counter()
                prediction = adapter(model, cert, expected_certificate_sha256=EXPECTED,
                                     control=control, readout_chunk_size=128, **kwargs)
                call_seconds = time.perf_counter() - begin
                if (not torch.equal(prediction.scored_query_ids, scored)
                        or prediction.logits.shape != (len(cert.stored_sequence.positions), len(scored), 4)
                        or not bool(torch.isfinite(prediction.logits).all())):
                    raise ValueError("Original output order or finite logits changed")
                begin = time.perf_counter()
                prediction.logits.square().mean().backward()
                backward_seconds = time.perf_counter() - begin
                active_gradients = 0
                for name, parameter in model.named_parameters():
                    inactive = ((name.endswith("raw_eta") and control in ("eta0", "local"))
                                or (name.endswith("raw_damping") and control == "local"))
                    if inactive and parameter.grad is not None:
                        raise ValueError("Inactive control parameter acquired a gradient")
                    if not inactive and (parameter.grad is None
                                         or not bool(torch.isfinite(parameter.grad).all())):
                        raise ValueError("Missing/nonfinite active synthetic gradient")
                    active_gradients += parameter.grad is not None
                    parameter.grad = None
                validate_census_biological_certificate_v1(cert, EXPECTED)
                if (fingerprint_sha256_v1(model.state_dict()) != pins["initial_state_sha256"]
                        or "forward" in model.encoder.__dict__):
                    raise ValueError("Neural weights or scoped encoder restoration changed")
                after = adapter.report()
                result["records"].append({
                    "round": round_id + 1, "control": control, "adapter_call_seconds": call_seconds,
                    "phase_seconds": {key: after["phase_seconds"][key] - before["phase_seconds"][key]
                                      for key in after["phase_seconds"]},
                    "synthetic_backward_seconds": backward_seconds,
                    "forest_bindings_this_call": after["forest_bindings"] - before["forest_bindings"],
                    "forest_reuses_this_call": after["forest_reuses"] - before["forest_reuses"],
                    "active_gradient_tensors": active_gradients,
                    "initial_and_final_state_sha256": pins["initial_state_sha256"],
                    "query_readout_order_checked": True, "all_parameter_gradients_cleared": True,
                    "backward_objective": "observed_logits_squared_mean_no_target_no_optimizer"})
                del prediction, model
        report = adapter.report()
        if (report["calls"], report["forest_bindings"], report["forest_reuses"]) != (10, 1, 9):
            raise ValueError("One first binding and nine actual reuses required")
        result.update(status="completed_observed_only_CPU_timing",
                      forest_reuse_report=report, whole_tree_forward_and_synthetic_backward_measured=True,
                      controls_complete=True, measured_speedup_claimed=False)
    except Exception as error:
        result.update(status="failed_partial_preserved", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        # Denial/expiry prohibits evidence writes; caller stderr carries errors.
        authority(required_write_bytes=len(json_bytes_v1(result)))
        output.write("evidence.json", result)
    print(json.dumps({"status": result["status"], "job_id": result["job_id"],
                      "forest_reuse_report": result["forest_reuse_report"]}, sort_keys=True))

if __name__ == "__main__":
    main()

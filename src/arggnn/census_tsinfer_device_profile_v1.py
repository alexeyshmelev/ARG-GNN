"""Inactive target-free same-tsinfer Census CPU/CUDA parity and backward profile."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
from . import census_tsinfer_fit_v1 as consumer
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"src/arggnn/census_tsinfer_device_profile_v1.py"
CONFIG=ROOT/"configs/census_tsinfer_device_profile_v1.json"
CONTROLS=("proposed","eta0","permuted_groups","raw_diagonal","local")
REFS=consumer.GATE_REFS|{"config","review","runtime","storage","memory","registration"}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def when(value):return consumer._utc(value)
def active(admission):
    now=datetime.now(timezone.utc)
    if not when(admission["created_at_utc"])<=now<when(admission["expires_at_utc"]) or (when(admission["expires_at_utc"])-when(admission["created_at_utc"])).total_seconds()>86400:
        raise ValueError("Fresh bounded root profile authority required")
def require_live_runtime_identity_v1(numerical_runtime):
    import sys,torch,numpy,tskit
    if Path(sys.executable).resolve()!=(ROOT/".runtime/env/bin/python").resolve():raise ValueError("Actual main project interpreter required")
    actual={"python":sys.version.split()[0],"torch":str(torch.__version__),"numpy":str(numpy.__version__),"tskit":str(tskit.__version__)}
    if actual!={k:numerical_runtime["identity"][k] for k in actual}:raise ValueError("Numerically qualified live project packages required")
    return actual
def require_memory_v1(memory,a,cfg):return consumer.memory_v1(memory,a,cfg)

def source_report():
    cfg=json.loads(CONFIG.read_text())
    for p,h in cfg["source_sha256"].items():
        if sha(ROOT/p)!=h:raise ValueError("Pinned device-consumer dependency changed: "+p)
    consumer.source_report_v1()
    old=json.loads((ROOT/"configs/census_biological_device_profile_v1.json").read_text())
    if cfg["model"]!=old["model"] or cfg["tolerances"]!=old["tolerances"]:raise ValueError("Original device profile architecture/tolerances changed")
    current=ast.parse(SOURCE.read_text());original=ast.parse((ROOT/"src/arggnn/census_biological_device_profile_v1.py").read_text())
    def body(tree,name):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        for child in ast.walk(node):
            if isinstance(child,ast.Constant) and child.value=="census_biological_device_profile_evidence_v1":child.value="census_tsinfer_device_profile_evidence_v1"
        node.body=[n for n in node.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        return ast.dump(node,include_attributes=False)
    for name in ("compare_certified_tile_v1","run_profile"):
        if body(current,name)!=body(original,name):raise ValueError("Original device profile scientific body changed: "+name)
    return {"kind":"census_tsinfer_device_profile_v1_source_report","implementation":{"path":str(SOURCE.relative_to(ROOT)),"sha256":sha(SOURCE)},
        "config":{"path":str(CONFIG.relative_to(ROOT)),"sha256":sha(CONFIG)},"observation_arrays_opened":False,"models_created":False,"CUDA_accessed":False,
        "actual_TRAIN_extrema":[446,550],"query_ancestry_input":False}

def admit_profile(record):
    reader=consumer.ConsumerReaderV1();a=reader.decode_json(record)
    fields={"schema","status","root_authorizer","job_id","created_at_utc","expires_at_utc","device","output_directory","refs","source_sha256","selected_cells"}
    if (set(a)!=fields or a["schema"]!="census_tsinfer_device_profile_admission_v1" or a["status"]!="active"
        or a["root_authorizer"]!="root_agent" or a["device"]!="cuda" or set(a["refs"])!=REFS or not consumer.canonical_job_v1(a["job_id"])):
        raise ValueError("Exact fresh root target-free same-tsinfer CUDA profile required")
    active(a)
    from .native_argformer_exec import validate_lease
    require_logical_job_v1(validate_lease(),a)
    cfg=reader.decode_json(a["refs"]["config"]);source_report()
    if cfg!=json.loads(CONFIG.read_text()) or a["refs"]["config"]!={"path":str(CONFIG.relative_to(ROOT)),"sha256":sha(CONFIG)}:raise ValueError("Exact profile configuration required")
    sources=dict(cfg["source_sha256"],**{str(SOURCE.relative_to(ROOT)):sha(SOURCE)})
    if a["source_sha256"]!=sources:raise ValueError("Complete exact profile execution source closure required")
    for p,h in sources.items():reader.read({"path":p,"sha256":h})
    consumer.review_v1(reader.decode_json(a["refs"]["review"]),str(SOURCE.relative_to(ROOT)),str(CONFIG.relative_to(ROOT)))
    consumer.registration_v1(reader.decode_json(a["refs"]["registration"]),a,cfg,"device-profile")
    native,qcfg=consumer.gate_from_reader_v1(a,reader)
    if cfg["model"]!=qcfg["model"] or cfg["tolerances"]!=qcfg["tolerances"]["float32"]:raise ValueError("Fixed original numerical model/tolerances required")
    storage=consumer.storage_v1(reader.decode_json(a["refs"]["storage"]),a,cfg,str(SOURCE.relative_to(ROOT)),str(CONFIG.relative_to(ROOT)))
    consumer.memory_v1(reader.decode_json(a["refs"]["memory"]),a,cfg)
    consumer.verify_observation_closure_v1(native,reader)
    consumer.runtime_v1(reader.decode_json(a["refs"]["runtime"]),native,reader,"census_tsinfer_device_profile_runtime_v1")
    selection=native["_inspection"]["prespecified_TRAIN_numerical_cells"]
    wanted=[{"parent_id":selection[k]["parent_id"],"tile_id":selection[k]["tile_id"],"original_tree_count":selection[k]["num_trees"]} for k in ("minimum","maximum")]
    if wanted!=cfg["actual_TRAIN_extrema"] or a["selected_cells"]!=wanted:raise ValueError("Frozen actual same-tsinfer TRAIN extrema required")
    return a,cfg,native,qcfg,reader,storage["reservation"]


def require_logical_job_v1(lease, admission):
    pool_job = lease["launcher_metadata"]["job"]
    actual_job = os.environ.get("ARGGNN_LOGICAL_JOB_NAME") or os.environ.get("ARGGNN_JOB_NAME")
    if (actual_job != admission["job_id"] or os.environ.get("ARGGNN_JOB_NAME") != pool_job
        or not actual_job):
        raise ValueError("Admitted logical job differs from actual held lease worker")

def compare_certified_tile_v1(cert, expected, initial, permutation, qcfg, device, resource_check):
    """Disposable CPU-float32 versus selected-device-float32 comparison."""
    import torch
    from .census_tsinfer_fit_v1 import forward_certified_census_biological_v1, validate_census_biological_certificate_v1, fingerprint_sha256_v1
    validate_census_biological_certificate_v1(cert, expected)
    tol = qcfg["tolerances"]["float32"]
    records = []
    for control in CONTROLS:
        resource_check()
        cpu = deepcopy(initial).to(device="cpu", dtype=torch.float32)
        selected = deepcopy(initial).to(device=device, dtype=torch.float32)
        kwargs = {"group_permutation": permutation} if control == "permuted_groups" else {}
        pairs = []
        for model, current_device in ((cpu, torch.device("cpu")), (selected, torch.device(device))):
            resource_check()
            if current_device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = forward_certified_census_biological_v1(
                model, cert, expected_certificate_sha256=expected, control=control,
                readout_chunk_size=qcfg["readout_chunk_size"], **kwargs)
            if current_device.type == "cuda": torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - start
            coeff = torch.linspace(-.7, 1.1, result.logits.numel(),
                dtype=torch.float64, device=current_device).reshape_as(result.logits)
            start = time.perf_counter()
            objective = (result.logits.double() * coeff).mean()
            grads = torch.autograd.grad(objective, tuple(model.parameters()), allow_unused=True)
            if current_device.type == "cuda": torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - start
            for (name, parameter), grad in zip(model.named_parameters(), grads):
                inactive = ((name.endswith("raw_eta") and (control in {"eta0", "local"} or len(cert.native_trees) == 1))
                    or (name.endswith("raw_damping") and control == "local"))
                if inactive and grad is not None:
                    raise ValueError("Inactive control parameter acquired gradient")
                if not inactive and (grad is None or not bool(torch.isfinite(grad).all())):
                    raise ValueError("Missing/nonfinite active device VJP")
                parameter.grad = grad
            output = result.logits.detach().cpu().clone()
            gradient_copy = tuple(None if g is None else g.detach().cpu().clone() for g in grads)
            query_ids = result.scored_query_ids.detach().cpu().clone()
            start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), qcfg["gradient_clip"], error_if_nonfinite=True)
            # This synthetic disposable update is never used as a fitted checkpoint.
            torch.optim.AdamW(model.parameters(), lr=qcfg["unit_step_lr"],
                weight_decay=qcfg["unit_step_weight_decay"]).step()
            if current_device.type == "cuda": torch.cuda.synchronize()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError("Nonfinite disposable AdamW step")
            timing = {"forward_wall_seconds_including_host_and_transfers": forward_seconds,
                "backward_wall_seconds": backward_seconds,
                "disposable_AdamW_wall_seconds": time.perf_counter() - start,
                "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated() if current_device.type == "cuda" else None,
                "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved() if current_device.type == "cuda" else None}
            pairs.append((output, gradient_copy, query_ids, timing))
            del result, grads, coeff, objective
        a, b = pairs
        if not torch.equal(a[2], b[2]): raise ValueError("Device changes query output ordering")
        torch.testing.assert_close(a[0], b[0], rtol=tol["output_rtol"], atol=tol["output_atol"])
        errors = []
        for left, right in zip(a[1], b[1]):
            if (left is None) != (right is None): raise ValueError("Active VJP schema differs")
            if left is not None:
                torch.testing.assert_close(left, right, rtol=tol["vjp_rtol"], atol=tol["vjp_atol"])
                errors.append(float((left - right).abs().max()))
        validate_census_biological_certificate_v1(cert, expected)
        records.append({"control": control, "CPU_float32_selected_device_float32_output_VJP_parity": True,
            "output_max_absolute_difference": float((a[0] - b[0]).abs().max()),
            "VJP_max_absolute_difference": max(errors, default=0.),
            "scored_query_count": a[2].numel(), "output_shape": list(a[0].shape),
            "CPU": a[3], "selected_device": b[3], "finite_disposable_AdamW": True,
            "initial_state_sha256": fingerprint_sha256_v1(initial.state_dict()),
            "objective": "fixed mean-logit coefficients, no ancestry targets",
            "optimizer_step_parity_claimed": False})
        del cpu, selected, pairs
    return records

def run_profile(record):
    admission, cfg, native, qcfg, reader, reservation = admit_profile(record)
    import torch
    from .census_tsinfer_fit_v1 import read_manifests_v1, load_parent_v1, prepare_tile_v1
    from .census_native_qualification_runner_v1 import _model, BoundedEvidenceOutputV1
    from .census_biological_bank_v1 import fingerprint_sha256_v1
    from .biological_candidate import configure_fit, check_fit_lease
    from . import project_budget_v4 as budget
    output = BoundedEvidenceOutputV1(admission["output_directory"], limit_bytes=cfg["output_reservation_bytes"])
    result = {"schema": "census_tsinfer_device_profile_evidence_v1", "status": "running", "admission_ref": record,
              "config_ref": admission["refs"]["config"], "source_sha256": admission["source_sha256"],
              "numerical_ref": admission["refs"]["numerical_evidence"], "device": admission["device"],
              "models": [], "selected_cells": [], "query_truth_accessed": False,
              "scientific_fit_performed": False, "synthetic_disposable_steps_only": True}
    try:
        device, runtime, lease = configure_fit("cuda", {"gpu_runtime": {"nvtop_helper": "scripts/qualify_official_argformer_runtime.py"}}, output.path)
        require_logical_job_v1(lease, admission)
        result["runtime"] = dict(runtime, live_package_identity=require_live_runtime_identity_v1(
            reader.decode_json(native["refs"]["runtime"])), CPU_threads=torch.get_num_threads())
        def resources():
            active(admission); reader.read(record); check_fit_lease(lease)
            require_logical_job_v1(lease, admission)
            require_memory_v1(reader.decode_json(admission["refs"]["memory"]), admission, cfg)
            budget.check_reservation(ROOT, reservation,
                expected_source_sha256=cfg["budget_source"]["sha256"],
                expected_config_sha256=cfg["budget_config"]["sha256"], boundary=True)
        resources()
        paired = read_manifests_v1(native, qcfg, reader)
        candidates = [(len(t["original_tree_ids"]), p["parent_id"], t["tile_id"])
                      for p, bank in paired if p["split"] == "train" for t in bank["tiles"]]
        selected = []
        for extreme in (min(x[0] for x in candidates), max(x[0] for x in candidates)):
            chosen = min(x for x in candidates if x[0] == extreme)
            cell = {"parent_id": chosen[1], "tile_id": chosen[2], "original_tree_count": chosen[0]}
            if cell not in selected: selected.append(cell)
        if admission["selected_cells"] != selected:
            raise ValueError("Prespecified TRAIN min/max tree-count selection required")
        tmp = output.path / "tmp"; tmp.mkdir()
        initial = _model(qcfg, "float32")
        if fingerprint_sha256_v1(initial.state_dict()) != native["initial_state_sha256_by_dtype"]["float32"]:
            raise ValueError("Externally frozen original model initialization changed")
        for cell in selected:
            resources()
            parent, bank = next(pair for pair in paired if pair[0]["parent_id"] == cell["parent_id"])
            loaded = load_parent_v1(parent, bank, native, qcfg, reader, tmp_directory=tmp)
            tile = next(t for t in bank["tiles"] if t["tile_id"] == cell["tile_id"])
            cert, _, _ = prepare_tile_v1(parent, bank, tile, loaded, native, qcfg, reader)
            expected = native["expected_certificates"][cell["parent_id"] + ":" + str(cell["tile_id"])]
            checks = compare_certified_tile_v1(cert, expected, initial,
                torch.tensor(native["group_permutation"], dtype=torch.long), qcfg, device, resources)
            result["selected_cells"].append(dict(cell, certificate_sha256=expected, checks=checks))
            for check in checks:
                peak = check["selected_device"]["peak_CUDA_reserved_bytes"]
                if peak > cfg["GPU_total_reservation_mib"] * 1024 * 1024:
                    raise ValueError("Measured profile allocator peak exceeds admitted full-backward reservation")
            del cert, loaded
        resources()
        result.update(status="completed_target_free_device_parity_and_backward_profile",
            all_five_controls_CPU_float32_CUDA_float32_output_VJP_parity=True,
            full_whole_tree_forward_backward_measured=True,
            measured_speedup_or_sustained_utilization_claimed=False)
    except Exception as error:
        result.update(status="failed_partial_preserved", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        output.write("evidence.json", result)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--admission")
    parser.add_argument("--expected-admission-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(source_report(), indent=2)); return 0
    if not args.admission or not args.expected_admission_sha256:
        raise ValueError("Fresh root admission required")
    # Lease/title precede even the lazy Torch import in admission.
    from .native_argformer_exec import validate_lease, apply_and_validate_title
    validate_lease(); apply_and_validate_title()
    run_profile({"path": args.admission, "sha256": args.expected_admission_sha256})
    return 0

if __name__=="__main__":raise SystemExit(main())

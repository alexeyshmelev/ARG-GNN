"""H024 full-support observation-only CPU synthetic numerical qualification.

Default is a stdlib source report. Actual cells require fresh independent source
PASS, exact root admissions, immutable original inputs and truthful v4 resources.
No ancestry target reader, supervised loss, fit loop, GPU selector or scheduler.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = "src/arggnn/anchor_parent_cpu_profile_v1.py"
CONFIG_PATH = "configs/anchor_parent_cpu_profile_v1.json"
CONFIG_SHA256 = "fe978db8dd5fc63c52ec8cb4944b08ccdfb0dc299d549d0ba92323cb030f93d6"

def require(value, message):
    if not value:
        raise ValueError(message)

def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Timezone-aware admission dates required")
    return parsed.astimezone(timezone.utc)

def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()

class ReaderV1:
    def path(self, record):
        require(set(record) == {"path", "sha256"}, "Exact path/SHA256 record required")
        require(isinstance(record["sha256"], str) and len(record["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in record["sha256"]), "Exact lowercase SHA256 required")
        p = Path(record["path"]); p = p if p.is_absolute() else ROOT/p
        require(".." not in p.parts and ROOT in p.parents and not p.is_symlink()
                and p.resolve() == p and p.is_file(), "Canonical repository file required")
        require(not any(any(w in s.lower() for w in ("truth", "oracle", "test")) for s in p.relative_to(ROOT).parts),
                "Truth/oracle/TEST capabilities forbidden")
        require(p.stat().st_size <= 8_000_000_000 and sha(p) == record["sha256"], "Frozen source/input/artifact bytes changed")
        return p
    def json(self, record):
        return json.loads(self.path(record).read_text())

def load_config_v1():
    require(sha(ROOT/CONFIG_PATH) == CONFIG_SHA256, "Frozen profile protocol changed")
    cfg = json.loads((ROOT/CONFIG_PATH).read_text())
    require(cfg["execution_enabled"] is False, "Shipped protocol cannot authorize execution")
    reader = ReaderV1()
    for path, value in cfg["source_pins"].items():
        # Pinned runtime helpers may be binaries, but no GPU context is created.
        require(sha(ROOT/path) == value, "Original source/runtime helper drift: "+path)
    require(cfg["qualification"] == reader.json(cfg["core_config_ref"])["qualification"], "Original tolerances changed")
    for prefix in ("producer", "core"):
        review = reader.json(cfg[prefix+"_review_ref"])
        require(review.get("verdict") == "PASS" and review.get("implementation") == cfg[prefix+"_source_ref"]
                and review.get("config") == cfg[prefix+"_config_ref"], "Exact accepted unchanged producer/core required")
    return cfg

def source_report_v1():
    cfg = load_config_v1()
    return {"kind": cfg["kind"]+"_source_report", "implementation": {"path": SOURCE_PATH, "sha256": sha(Path(__file__))},
            "config": {"path": CONFIG_PATH, "sha256": CONFIG_SHA256}, "cells": cfg["matrix"],
            "serialization_bound": cfg["serialization_bound"], "actual_inputs_opened": False,
            "models_instantiated": False, "actual_profile_fit_GPU_or_budget": False}

def validate_admission_metadata_v1(a, cfg, now):
    fields = {"schema", "status", "root_authorizer", "observed_profile_authorized", "scientific_fit_authorized",
              "supervised_loss_authorized", "GPU_authorized", "job_id", "matrix_job_id", "created_at_utc",
              "expires_at_utc", "cell_id", "matrix_output_directory", "output_directory", "refs",
              "initial_state_sha256", "comparison_refs", "CPU_headroom_bytes", "external_watchdog",
              "sequential_full_parent_cell"}
    require(set(a) == fields and a["schema"] == "anchor_parent_CPU_profile_admission_v1"
            and a["status"] == "active" and a["root_authorizer"] == "root_agent"
            and a["observed_profile_authorized"] is True
            and all(a[k] is False for k in ("scientific_fit_authorized", "supervised_loss_authorized", "GPU_authorized")),
            "Fresh root observed-only synthetic profile authority required")
    require(utc(a["created_at_utc"]) <= now < utc(a["expires_at_utc"])
            and (utc(a["expires_at_utc"])-utc(a["created_at_utc"])).total_seconds() <= 86400,
            "Fresh bounded active root capability required")
    require(isinstance(a["job_id"], str) and a["job_id"] and isinstance(a["matrix_job_id"], str) and a["matrix_job_id"],
            "Distinct root cell/matrix identities required")
    require(a["CPU_headroom_bytes"] == 128_000_000_000 and a["external_watchdog"] is True
            and a["sequential_full_parent_cell"] is True, "One full parent cell and128GB external watchdog required")
    cell = next((c for c in cfg["matrix"] if c["id"] == a["cell_id"]), None)
    require(cell is not None and set(a["comparison_refs"]) == set(cell["reference_cells"]), "Exact fixed cell/reference matrix required")
    require(a["output_directory"] == a["matrix_output_directory"]+"/"+a["cell_id"],
            "New exclusive cell scope within charged matrix scope required")
    require(set(a["refs"]) == {"config", "source", "review", "registration", "runtime", "producer_binding",
                              "observed_acceptance", "observed_report", "storage"},
            "Complete source/runtime/observation/storage refs required")
    for key, expected in {"config":{"path":CONFIG_PATH,"sha256":CONFIG_SHA256},
                          "registration":cfg["root_registration_ref"],"runtime":cfg["runtime_ref"],
                          "producer_binding":cfg["producer_binding_ref"],
                          "observed_acceptance":cfg["accepted_observed_ref"],
                          "observed_report":cfg["accepted_observed_report_ref"]}.items():
        require(a["refs"][key] == expected, "Original exact reference changed: "+key)
    require(isinstance(a["initial_state_sha256"], str) and len(a["initial_state_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in a["initial_state_sha256"]),
            "Externally frozen original seed101 initial state SHA required")
    return cell

class AuthorityV1:
    def __init__(self, admission_ref):
        self.reader = ReaderV1(); self.ref = admission_ref
        self.cfg = load_config_v1(); self.denied = False; self.first = True
        self.a = None; self.cell = None
    def check(self, required_write_bytes=0):
        require(not self.denied, "Sticky profile capability denial")
        try:
            a = self.reader.json(self.ref)
            cell = validate_admission_metadata_v1(a, self.cfg, datetime.now(timezone.utc))
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
                    and Path(sys.executable).resolve() == (ROOT/".runtime/env/bin/python").resolve(),
                    "Original main CPU interpreter and disabled bytecode required")
            require(a["refs"]["source"] == {"path": SOURCE_PATH, "sha256": sha(Path(__file__))}, "Exact running profile source required")
            review = self.reader.json(a["refs"]["review"])
            require(review.get("verdict") == "PASS" and review.get("review_role") == "independent_reviewer"
                    and review.get("actual_admission") is False
                    and review.get("implementation") == a["refs"]["source"]
                    and review.get("config") == a["refs"]["config"], "Fresh independent exact source/config PASS required")
            reg = self.reader.json(a["refs"]["registration"])
            require(reg.get("actual_execution_authorized") is False and reg.get("authorizer") == "root_agent"
                    and reg.get("hypothesis_id") == "H024", "Exact source registration required")
            runtime = self.reader.json(a["refs"]["runtime"])
            import torch, numpy, tskit
            torch.set_num_threads(1)
            identity = {"python":sys.version.split()[0],"torch":str(torch.__version__),"numpy":str(numpy.__version__),
                        "tskit":str(tskit.__version__),"device":"cpu","threads":1,"cuda_visible_devices":""}
            require(runtime["identity"] == identity, "Qualified original main runtime identity changed")
            for path, value in runtime["runtime_file_sha256"].items():
                require(sha(ROOT/path) == value, "Qualified runtime bytes changed: "+path)
            self.reader.json(a["refs"]["producer_binding"])
            accepted = self.reader.json(a["refs"]["observed_acceptance"])
            require(accepted["status"] == "PASS_observed_input_reconstruction_and_assembly_only"
                    and accepted["report"] == a["refs"]["observed_report"]
                    and accepted["binding"] == a["refs"]["producer_binding"]
                    and accepted["source"] == self.cfg["producer_source_ref"]
                    and accepted["config"] == self.cfg["producer_config_ref"], "Original observed producer acceptance changed")
            storage = self.reader.json(a["refs"]["storage"]); reservation = storage["reservation"]
            require(set(storage) == {"schema","status","job_id","created_at_utc","expires_at_utc","reservation","initial_ledger_pins"}
                    and storage["schema"] == "anchor_parent_CPU_profile_storage_v1"
                    and storage["status"] == "active" and storage["job_id"] == a["matrix_job_id"]
                    and utc(storage["created_at_utc"]) <= datetime.now(timezone.utc) < utc(storage["expires_at_utc"]),
                    "Fresh matrix resource capability required")
            require(reservation["key"] == a["matrix_job_id"] and reservation["output"] == str(ROOT/a["matrix_output_directory"])
                    and reservation["auxiliary_scopes"] == [] and reservation["reserved_bytes"] == 40_000_000_000,
                    "Exact complete40GB matrix scope charge required")
            needed = [a["refs"]["source"],a["refs"]["config"],a["refs"]["runtime"],
                      self.cfg["budget_source_ref"],self.cfg["budget_config_ref"],self.cfg["human_authorization_ref"]]
            require(all(r in reservation["consumer_bindings"] for r in needed), "Full own source/runtime/budget/human consumer binding required")
            from . import project_budget_v4 as budget
            account = budget.check_reservation(ROOT,reservation,
                expected_source_sha256=self.cfg["budget_source_ref"]["sha256"],
                expected_config_sha256=self.cfg["budget_config_ref"]["sha256"],
                initial_ledger_pins=storage["initial_ledger_pins"] if self.first else None,
                boundary=True,required_write_bytes=required_write_bytes)
            require(account["limit_bytes"] == 950_000_000_000 and account["hard_limit_bytes"] == 1_000_000_000_000
                    and account["projected_bytes"] == sum(account[k] for k in
                        ("baseline_bytes","external_reserve_bytes","legacy_reserved_bytes","supplemental_reserved_bytes"))
                    and account["projected_bytes"] < 950_000_000_000
                    and account["legacy_ledger_mutated"] is False and account["supplemental_ledger_mutated"] is False,
                    "Genuine readonly combined v4 project accounting required")
            require(peak_RSS_bytes_v1() <= 128_000_000_000, "Measured process peak exceeds128GB; no smaller fallback")
            self.a, self.cell = a, cell; self.first = False
            return a,cell
        except Exception:
            self.denied = True
            raise

def peak_RSS_bytes_v1():
    require(sys.platform.startswith("linux"), "Linux per-process ru_maxrss semantics required")
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024

def snapshot_v1(value):
    import torch
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "Nonfinite recorded tensor")
        return value.detach().cpu().clone()
    if isinstance(value, dict):return {k:snapshot_v1(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):return type(value)(snapshot_v1(v) for v in value)
    require(value is None or isinstance(value,(str,bool,int,float)), "Explicit finite tensor/primitive evidence only")
    if isinstance(value,float):require(math.isfinite(value), "Nonfinite recorded scalar")
    return value

def optimizer_snapshot_v1(model,optimizer):
    names = {id(p):n for n,p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        record = {k:snapshot_v1(v) for k,v in group.items() if k != "params"}
        record["parameter_names"] = [names[id(p)] for p in group["params"]]
        groups.append(record)
    state = {name:snapshot_v1(optimizer.state.get(p,{})) for name,p in model.named_parameters()}
    return {"groups":groups,"state":state}

def assert_tree_close_v1(actual,expected,tolerance,*,tensor_kind="vjp",cross_dtype=False):
    import torch
    if isinstance(actual,torch.Tensor) or isinstance(expected,torch.Tensor):
        require(isinstance(actual,torch.Tensor) and isinstance(expected,torch.Tensor)
                and actual.shape == expected.shape, "Complete tensor shape correspondence required")
        require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), "Nonfinite comparison tensor")
        if not actual.is_floating_point() or not expected.is_floating_point():
            require(torch.equal(actual,expected), "Integer identities/step tensor changed")
        else:
            torch.testing.assert_close(actual,expected,atol=tolerance[tensor_kind+"_atol"],
                rtol=tolerance[tensor_kind+"_rtol"],check_dtype=not cross_dtype)
    elif isinstance(actual,dict) or isinstance(expected,dict):
        require(isinstance(actual,dict) and isinstance(expected,dict) and actual.keys() == expected.keys(), "Full named state/key correspondence required")
        for key in actual:assert_tree_close_v1(actual[key],expected[key],tolerance,tensor_kind=tensor_kind,cross_dtype=cross_dtype)
    elif isinstance(actual,(tuple,list)) or isinstance(expected,(tuple,list)):
        require(type(actual) is type(expected) and len(actual) == len(expected), "Complete sequence structure required")
        for x,y in zip(actual,expected):assert_tree_close_v1(x,y,tolerance,tensor_kind=tensor_kind,cross_dtype=cross_dtype)
    else:
        require(type(actual) is type(expected) and actual == expected, "Exact scalar/group/None state correspondence required")

def synthetic_numerator_v1(logits,widths):
    import torch
    require(logits.ndim == 3 and logits.shape[-1] == 4 and widths.shape == (len(logits),)
            and widths.dtype == torch.float64 and widths.device.type == "cpu"
            and bool(torch.isfinite(widths).all() & (widths>0).all()), "Complete physical readout/Q/class objective required")
    # Fixed observed bp weights, with no ancestry label or target input.
    numerator = (logits.square().mean(-1)*widths.to(logits)[:,None]).sum()
    denominator = float(widths.sum())*logits.shape[1]
    return numerator,denominator

def execute_method_v1(model,optimizer,forward,widths,*,matched,guard=lambda:None):
    """Finite mathematical helper. Actual entry supplies unchanged full producer."""
    import torch
    from .anchor_parent_context_v3 import matched_parent_step_v3
    params = tuple(model.parameters()); names = tuple(n for n,_ in model.named_parameters())
    require(all(p.requires_grad and p.is_leaf for p in params) and not optimizer.state,
            "All original parameters active and original AdamW state initially empty required")
    require(len(params) == len(set(map(id,params))) and all(p.grad is None for p in params), "Fresh full unique parameter set required")
    require(set(id(p) for g in optimizer.param_groups for p in g["params"]) == set(map(id,params))
            and sum(len(g["params"]) for g in optimizer.param_groups) == len(params), "Exact complete original optimizer binding required")
    initial = snapshot_v1(model.state_dict()); raw = {}; handles = []
    for name,p in model.named_parameters():
        def capture(g,name=name):
            require(name not in raw and bool(torch.isfinite(g).all()), "Exactly one finite aggregate VJP per parameter required")
            raw[name] = snapshot_v1(g)
        handles.append(p.register_hook(capture))
    recorded = {}; forward_seconds = 0.
    def cell():
        nonlocal forward_seconds
        guard(); begin = time.perf_counter()
        logits,context = forward()
        forward_seconds = time.perf_counter()-begin
        require(all(p.grad is None for p in params), "Forward callback must not preload gradients")
        require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(context.anchors).all())
                and bool(torch.isfinite(context.targets).all()), "Finite full logits/context required")
        recorded.update(logits=snapshot_v1(logits),anchors=snapshot_v1(context.anchors),targets=snapshot_v1(context.targets))
        return synthetic_numerator_v1(logits,widths)
    try:
        begin = time.perf_counter()
        if matched:
            step = matched_parent_step_v3(optimizer,params,[cell],gradient_clip_norm=1.,inactive_parameters=())
            backward_clip_step_seconds = time.perf_counter()-begin-forward_seconds
        else:
            optimizer.zero_grad(set_to_none=True)
            numerator,denominator = cell(); loss = numerator/denominator
            require(loss.requires_grad and bool(torch.isfinite(loss)), "Connected finite observed scalar required")
            backward = time.perf_counter(); loss.backward()
            backward_seconds = time.perf_counter()-backward
            require(set(raw) == set(names) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),
                    "All original paths are intended active; no discovered exemption")
            clipstep = time.perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            backward_clip_step_seconds = backward_seconds+time.perf_counter()-clipstep
            step = {"loss":float(loss.detach()),"query_bp":denominator,"loss_cells":1,
                    "gradient_norm_before_single_clip":float(norm),"backward_calls":1,"clip_calls":1,"optimizer_steps":1,
                    "declared_inactive_parameters":0}
        require(set(raw) == set(names) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),
                "Complete finite full VJP/clipped gradients required")
        require(step["backward_calls"] == step["clip_calls"] == step["optimizer_steps"] == 1
                and step["declared_inactive_parameters"] == 0, "Exactly one synthetic clipped AdamW step required")
        optimizer_record = optimizer_snapshot_v1(model,optimizer)
        require(all(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                    and isinstance(state["step"],torch.Tensor) and state["step"].numel() == 1
                    and float(state["step"]) == 1. for state in optimizer_record["state"].values()),
                "Exact one-step AdamW state for every original active parameter required")
        recorded.update(initial_model_state=initial,raw_VJP={n:raw[n] for n in names},
                        clipped_gradients={n:snapshot_v1(p.grad) for n,p in model.named_parameters()},
                        final_model_state=snapshot_v1(model.state_dict()),optimizer=optimizer_record)
        recorded["step_scalars"] = {k:step[k] for k in ("loss","query_bp","gradient_norm_before_single_clip")}
        timing = {"cold_forward_seconds":forward_seconds,"backward_clip_step_seconds":backward_clip_step_seconds,
                  "method_total_seconds":time.perf_counter()-begin,"warmup_performed":False,"speedup_claim":False}
        return recorded,timing
    finally:
        for handle in handles:handle.remove()

def tensor_bytes_v1(value):
    import torch
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes_v1(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes_v1(v) for v in value)
    return 0

def bind_reconstructed_parent_v1(assemblies,report,expected,record,metadata):
    import torch
    keys = ("parent_id","namespace","original_tree_count","cache_graph_occurrences","cached_anchor_union",
            "all_leaf_ids","query_ids","reference_ids","tile_source_checks","artifact_refs","feature_columns",
            "exact_anchor_union","exact_weight_union","physical_layout_fingerprint_sha256",
            "target_graph_fingerprint_sha256","unique_original_tree_ids","first_cache_graph_indexes",
            "inverse_cache_graph_indexes","input_fingerprints")
    require(all(report[k] == expected[k] for k in keys), "Complete independently accepted original input/ID/fingerprint reconstruction required")
    require(report["original_tree_count"] == 955 and report["cache_graph_occurrences"] == 962
            and report["cached_anchor_union"] == 256 and report["unique_original_tree_ids"] == list(range(955))
            and len(report["all_leaf_ids"]) == 280 and len(report["query_ids"]) == 184 and len(report["reference_ids"]) == 96
            and report["feature_columns"] == [c for c in range(48) if c != 7], "No tile/reduced-tree/support substitute allowed")
    positions = torch.tensor(metadata["positions"],dtype=torch.float64)
    edges = torch.tensor(metadata["edges"],dtype=torch.float64);widths=edges[1:]-edges[:-1]
    ids = record["retained_fixed100bp_indexes"]
    require(len(ids) == len(positions) == 19988 and len(set(ids)) == len(ids)
            and positions.tolist() == [50.+100.*i for i in ids], "Complete frozen physical grid IDs/order required")
    from .anchor_parent_context_v3 import _tile_evaluations
    from . import biological_candidate as frozen
    for geometry,assembly in assemblies.items():
        p = assembly.parent
        actual = torch.cat([_tile_evaluations(t)[t.layout.target_output] for t in p.tile_contexts])
        require(torch.equal(actual,positions/2000000.)
                and torch.equal(p.layout.anchors,torch.tensor(expected["exact_anchor_union"],dtype=torch.float64))
                and torch.equal(p.layout.weights,torch.tensor(expected["exact_weight_union"],dtype=torch.float64))
                and len(p.tile_contexts) == 8 and len(p.target_graph) == 19988
                and p.original_tree_ids_by_cache_graph == tuple(i for t in record["tiles"] for i in t["original_tree_ids"])
                and frozen.fingerprint_sha(p.layout) == expected["physical_layout_fingerprint_sha256"],
                "Original physical readout/evaluation/tree/anchor correspondence required")
    common = {"parent_id":report["parent_id"],"original_grid_ids":ids,"positions_bp":positions.tolist(),
              "bin_edges_bp":edges.tolist(),"bin_widths_bp":widths.tolist(),"query_ids":report["query_ids"],
              "reference_ids":report["reference_ids"],"leaf_ids":report["all_leaf_ids"],
              "readout_evaluation_ids":assemblies["age"].parent.layout.target_output.tolist(),
              "normalized_readout_coordinates":(positions/2000000.).tolist(),
              "original_unique_tree_ids":report["unique_original_tree_ids"],
              "input_fingerprints":report["input_fingerprints"],"physical_layout_sha256":report["physical_layout_fingerprint_sha256"]}
    return widths,common

def write_artifact_v1(path,value,authority):
    import torch
    planned = (tensor_bytes_v1(value)*5+3)//4+16_777_216
    authority.check(required_write_bytes=planned)
    require(not path.exists() and path.parent.is_dir(), "Fresh exclusive profile artifact required")
    with path.open("xb") as stream:torch.save(value,stream)
    require(path.stat().st_size <= planned, "Conservative serialization bound exceeded")
    authority.check()
    return {"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size}

def compare_records_v1(actual,expected,tolerance,*,cross_dtype=False):
    for key in ("logits","anchors","targets"):
        assert_tree_close_v1(actual[key],expected[key],tolerance,tensor_kind="output",cross_dtype=cross_dtype)
    for key in ("initial_model_state","raw_VJP","clipped_gradients","final_model_state","optimizer"):
        assert_tree_close_v1(actual[key],expected[key],tolerance,cross_dtype=cross_dtype)
    # Scalars are consequences of same recorded full tensors, not relaxed criteria.
    for key in actual["step_scalars"]:
        x,y=actual["step_scalars"][key],expected["step_scalars"][key]
        require(math.isclose(x,y,rel_tol=tolerance["vjp_rtol"],abs_tol=tolerance["vjp_atol"]), "Synthetic step scalar mismatch")

def run_cell_v1(admission_ref):
    authority = AuthorityV1(admission_ref);a,cell=authority.check();cfg=authority.cfg
    import torch
    from . import anchor_parent_producer_v1 as producer
    from . import biological_candidate as frozen
    expected = authority.reader.json(a["refs"]["observed_report"])
    begin=time.perf_counter()
    assemblies,report=producer.qualify_observed_v1(a["refs"]["producer_binding"])
    _,_,record,_,metadata=producer.read_metadata_v1(a["refs"]["producer_binding"])
    widths,common=bind_reconstructed_parent_v1(assemblies,report,expected,record,metadata)
    preparation_seconds=time.perf_counter()-begin
    authority.check()
    out=ROOT/a["output_directory"]
    require(ROOT in out.parents and out.resolve()==out and not out.exists(), "New canonical exclusive cell evidence scope required")
    out.mkdir(parents=True)
    result={"kind":"anchor_parent_full_support_CPU_synthetic_numerical_cell_v1","status":"running",
            "cell":cell,"job_id":a["job_id"],"admission_ref":admission_ref,"source_ref":a["refs"]["source"],"config_ref":a["refs"]["config"],
            "runtime_ref":a["refs"]["runtime"],"original_observed_report_ref":a["refs"]["observed_report"],
            "preparation_seconds":preparation_seconds,"full_support":cfg["full_support"],"common":common,
            "methods":{},"comparisons":[],"supervised_TRAIN_loss_qualified":False,"scientific_fit_performed":False,
            "ancestry_targets_opened":False,"GPU_used":False,"synthetic_disposable_steps_only":True,"speedup_claim":False}
    records={}
    try:
        for method,matched in (("manual_original",False),("matched_parent_v3",True)):
            authority.check()
            model,optimizer=producer.original_model_optimizer_v1(cell["geometry"],101,dtype=getattr(torch,cell["dtype"]))
            require(frozen.fingerprint_sha(model.state_dict()) == a["initial_state_sha256"], "Externally frozen original seed101 initial state changed")
            require(sum(p.numel() for p in model.parameters()) <= 1_000_000
                    and sum(b.numel() for b in model.buffers()) <= 1_000_000, "Frozen conservative state serialization cap exceeded")
            require(len(optimizer.param_groups)==2 and optimizer.param_groups[0]["weight_decay"]==.0001
                    and optimizer.param_groups[1]["weight_decay"]==0.
                    and all(g["lr"]==.001 and tuple(g["betas"])==(.9,.999) and g["eps"]==1e-8 and g["amsgrad"] is False for g in optimizer.param_groups),
                    "Original AdamW group/default/clip contract changed")
            def forward():
                logits,context=producer.forward_parent_v1(assemblies[cell["geometry"]],model,mode=cell["mode"],
                                                         checkpoint_blocks=cell["checkpoint_blocks"])
                require(logits.shape==(19988,184,4) and context.anchors.shape==(256,184,32)
                        and context.targets.shape==(19988,184,32)
                        and torch.equal(context.target_indexes,torch.arange(19988,dtype=torch.long)),
                        "Every full readout/Q context is required")
                return logits,context
            data,timing=execute_method_v1(model,optimizer,forward,widths,matched=matched,guard=authority.check)
            records[method]=data;result["methods"][method]={"timing":timing}
            del model,optimizer;gc.collect()
        tolerance=cfg["qualification"][cell["dtype"]]
        compare_records_v1(records["matched_parent_v3"],records["manual_original"],tolerance)
        result["comparisons"].append({"kind":"unchanged_matched_helper_vs_original_manual_clipped_AdamW","status":"PASS","tolerance":tolerance})
        for refcell,refs in a["comparison_refs"].items():
            authority.check()
            require(set(refs)=={"evidence","terminal","manual_original","matched_parent_v3"}, "Complete root-frozen reference method artifacts required")
            prior=authority.reader.json(refs["evidence"])
            require(prior["status"]=="completed_full_support_observed_only_CPU_synthetic_numerical_cell"
                    and prior["cell"]["id"]==refcell and prior["source_ref"]==a["refs"]["source"]
                    and prior["config_ref"]==a["refs"]["config"] and prior["common"]==common
                    and prior["original_observed_report_ref"]==a["refs"]["observed_report"], "Exact qualified complete prior-cell evidence required")
            require(prior["cell"] == next(c for c in cfg["matrix"] if c["id"] == refcell),
                    "Exact frozen reference arm/dtype/checkpoint cell required")
            terminal = authority.reader.json(refs["terminal"])
            require(terminal == {"kind":"anchor_parent_CPU_profile_root_terminal_v1","status":"complete",
                    "returncode":0,"job_id":prior["job_id"],"cell_id":refcell,"evidence_ref":refs["evidence"],
                    "admission_ref":prior["admission_ref"]}, "Exact root exit0 reference-cell binding required")
            cross=prior["cell"]["dtype"]!=cell["dtype"]
            for method in records:
                require(prior["methods"][method]["artifact"]["path"]==refs[method]["path"]
                        and prior["methods"][method]["artifact"]["sha256"]==refs[method]["sha256"], "Root reference payload binding differs")
                previous=torch.load(authority.reader.path(refs[method]),map_location="cpu",weights_only=True)
                require(previous["cell_id"]==refcell and previous["method"]==method and previous["common"]==common, "Full prior artifact method/support identity required")
                compare_records_v1(records[method],previous["records"],tolerance,cross_dtype=cross)
                del previous;gc.collect()
            result["comparisons"].append({"kind":"cross_dtype" if cross else "checkpoint_true_vs_false",
                                         "reference_cell":refcell,"status":"PASS","tolerance":tolerance})
        for method,data in records.items():
            result["methods"][method]["artifact"]=write_artifact_v1(out/(method+".pt"),
                {"kind":"anchor_parent_full_support_numerical_evidence_v1","cell_id":cell["id"],"method":method,
                 "common":common,"records":data,"ancestry_targets_opened":False},authority)
        result["status"]="completed_full_support_observed_only_CPU_synthetic_numerical_cell"
    except Exception as error:
        result.update(status="failed_partial_full_support_cell_no_fallback",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result["process_peak_RSS_bytes"]=peak_RSS_bytes_v1()
        result["RSS_scope"]="whole isolated cell process including original assembly, numerical models and recorded comparison tensors"
        payload=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        authority.check(required_write_bytes=len(payload))
        with (out/"evidence.json").open("xb") as stream:stream.write(payload)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--admission");parser.add_argument("--expected-admission-sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(source_report_v1(),sort_keys=True,indent=2));return 0
    require(args.admission and args.expected_admission_sha256,"Fresh exact root cell admission required")
    result=run_cell_v1({"path":args.admission,"sha256":args.expected_admission_sha256})
    print(json.dumps({"status":result["status"],"cell":result["cell"]["id"],"process_peak_RSS_bytes":result["process_peak_RSS_bytes"]},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())

"""Selected-checkpoint VAL-only scalar temperature fitting, never model refitting."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
import stat
from pathlib import Path
import tempfile
import importlib.metadata

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
import torch
from .resources import ROOT
from . import development_selection as main
from . import development_selection_bindings as combined
from . import secondary_control_bindings as secondary
from . import confirmatory_lock as provenance

FIT_POLICY = {"schema_version": 1, "kind": "prespecified_validation_scalar_temperature_fit_policy",
    "fit_scope": "fresh_development_validation_only", "training_seeds": [101,202,303],
    "validation_parents": 16, "parameter": "one scalar log_temperature per selected method and seed",
    "temperature_bounds": [.05,20.], "initial_raw_temperature": 1.,
    "solver": "scipy.optimize.minimize_scalar bounded log_temperature",
    "maximum_solver_iterations": 500, "log_temperature_xatol": 1e-12,
    "objective": "equal mean of sixteen parent NLLs; uniform positions and query haplotypes within each parent",
    "nll_policy": "float64 stable log-softmax of raw logits divided by positive temperature; no clipping/floor",
    "final_choice": "exact minimum among bounded solver, both endpoints and raw T=1; exact ties by abs(logT), then logT",
    "model_or_checkpoint_optimization": False, "final_test_access": False}
CODE = ("src/arggnn/validation_temperature.py", "scripts/fit_validation_temperatures.py",
        "src/arggnn/confirmatory_metrics.py")


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def code_references():
    return [{"path":path,"sha256":sha((ROOT/path).read_bytes())} for path in CODE]


_IMPORTED_CODE = code_references()
_IMPORTED_POLICY = canonical(FIT_POLICY)


def _source_stable():
    if code_references()!=_IMPORTED_CODE or canonical(FIT_POLICY)!=_IMPORTED_POLICY:
        raise ValueError("Imported validation-temperature implementation drift")


def _ref(capture, value):
    if not isinstance(value,dict) or set(value)!={"path","sha256"}:
        raise ValueError("Exact development-only path/SHA reference required")
    path=combined.guard_development_path(value["path"],project_root=capture.root)
    path=capture.path(path)
    return {"path":str(path.relative_to(capture.root)),"sha256":main._sha(value["sha256"])}


def _raw_loss(z,y,temperature):
    scaled=z/temperature
    if not np.isfinite(scaled).all():
        raise ValueError("Temperature-scaled logits exceed float64 representable range")
    shifted=scaled-scaled.max(axis=-1,keepdims=True)
    logp=shifted-logsumexp(shifted,axis=-1,keepdims=True)
    if not np.isfinite(logp).all():
        raise ValueError("Float64 log-softmax exceeds representable range")
    return float(-np.take_along_axis(logp,y[...,None],axis=-1)[...,0].mean())


def _arrays(logits,labels):
    if not isinstance(logits,(list,tuple)) or not isinstance(labels,(list,tuple)) or not logits or len(logits)!=len(labels):
        raise ValueError("Explicit aligned nonempty parent arrays required")
    zs=[];ys=[]
    for z,y in zip(logits,labels):
        z=np.asarray(z,dtype=np.float64).copy();y=np.asarray(y).copy()
        if (z.ndim!=3 or z.shape[:2]!=y.shape or not y.size or z.shape[-1]!=4 or
                not np.isfinite(z).all() or y.dtype.kind not in ("i","u") or np.any(y<0) or np.any(y>=4)):
            raise ValueError("Finite aligned four-class raw logits and external integer labels required")
        z.flags.writeable=False;y.flags.writeable=False;zs.append(z);ys.append(y)
    return zs,ys


class TemperatureFitFailure(ValueError):
    def __init__(self,message,record):
        super().__init__(message);self.record=record


def fit_scalar_temperature(logits,labels,*,live_check=None):
    """Pure CPU utility. Production admission enforces the sixteen-parent source."""
    _source_stable();zs,ys=_arrays(logits,labels);trace=[]
    check=(lambda:None) if live_check is None else live_check
    def objective(log_t):
        check();_source_stable()
        temperature=math.exp(float(log_t))
        losses=[_raw_loss(z,y,temperature) for z,y in zip(zs,ys)]
        mean=math.fsum(losses)/len(losses)
        trace.append({"evaluation":len(trace),"log_temperature":float(log_t),
                      "temperature":temperature,"per_parent_nll":losses,"equal_parent_nll":mean})
        check()
        return mean
    lower,upper=(math.log(value) for value in FIT_POLICY["temperature_bounds"])
    try:
        result=minimize_scalar(objective,bounds=(lower,upper),method="bounded",
            options={"maxiter":FIT_POLICY["maximum_solver_iterations"],"xatol":FIT_POLICY["log_temperature_xatol"]})
        if not result.success or not math.isfinite(float(result.x)) or not lower<=float(result.x)<=upper:
            raise ValueError("Bounded scalar temperature optimizer did not converge")
    except BaseException as error:
        failed={"schema_version":1,"kind":"validation_temperature_optimization_record","status":"failed",
                "fit_policy":deepcopy(FIT_POLICY),"evaluations":trace,"external_parent_count":len(zs),
                "reason":str(error),"final_test_access":False}
        raise TemperatureFitFailure(str(error),failed) from error
    candidates=[]
    for value in (lower,0.,upper,float(result.x)):
        loss=objective(value);candidates.append((loss,abs(value),value))
    _,_,chosen=min(candidates);temperature=math.exp(chosen)
    per_parent=[_raw_loss(z,y,temperature) for z,y in zip(zs,ys)]
    raw=[_raw_loss(z,y,1.) for z,y in zip(zs,ys)]
    check();_source_stable()
    return {"schema_version":1,"kind":"validation_temperature_optimization_record","status":"complete",
        "fit_policy":deepcopy(FIT_POLICY),"temperature":temperature,"log_temperature":chosen,
        "raw_equal_parent_nll":math.fsum(raw)/len(raw),"calibrated_equal_parent_nll":math.fsum(per_parent)/len(per_parent),
        "raw_per_parent_nll":raw,"calibrated_per_parent_nll":per_parent,
        "bound_selected":chosen in (lower,upper),"solver_success":bool(result.success),
        "solver_message":str(result.message),"solver_nfev":int(result.nfev),"solver_iterations":int(result.nit),
        "solver_log_temperature":float(result.x),"candidate_log_temperatures":[row[2] for row in candidates],
        "evaluations":trace,"external_parent_count":len(zs),
        "software":{name:importlib.metadata.version(name) for name in ("numpy","scipy","torch")},
        "unchanged_raw_inputs":True,"unchanged_model_weights":True,"final_test_access":False}


def _state(admission):
    return canonical({"references":admission.references,"plan":admission.plan,"selection":admission.selection,
        "secondary_settings":admission.secondary_admission.settings,"secondary_selection":admission.secondary_selection,
        "methods":admission.methods,"fit_policy":admission.fit_policy,"code_artifacts":admission.code_artifacts})


@dataclass
class TemperatureAdmission:
    capture: object
    secondary_admission: object
    references: dict
    plan: dict
    selection: dict
    secondary_selection: dict
    methods: dict
    fit_policy: dict
    code_artifacts: list
    content: bytes=b""

    def check(self,full=True):
        _source_stable()
        if _state(self)!=self.content or self.fit_policy!=FIT_POLICY or self.code_artifacts!=code_references():
            raise ValueError("In-memory temperature admission/policy drift")
        self.capture.verify_stable();self.secondary_admission.check(full=full)


def admit_temperature_fitting(main_plan_ref,main_selection_ref,secondary_config_ref,secondary_selection_ref,
                              scope_amendment_ref,*,project_root=ROOT):
    """Complete factual producer admission and both selector replays before numerical fitting."""
    _source_stable();capture=main._Capture(project_root)
    names=("main_selection_plan","main_selection_receipt","secondary_configuration","secondary_selection_receipt","scope_amendment")
    refs={name:_ref(capture,ref) for name,ref in zip(names,(main_plan_ref,main_selection_ref,secondary_config_ref,secondary_selection_ref,scope_amendment_ref))}
    for ref in code_references():
        capture.read(_ref(capture,ref),json_file=False,allow_source_reuse=True)
    initial=capture.read(refs["main_selection_plan"])
    amendment=capture.read(refs["scope_amendment"]);combined._amendment(amendment)
    if set(initial.get("families",{}))!=set(combined.FAMILIES):
        raise ValueError("Complete ten-family main producer plan required")
    completion=_ref(capture,initial["source_artifacts"]["completion"])
    rebound=combined.bind_development_selection(capture.root/completion["path"],completion["sha256"],
        scope_amendment_path=capture.root/refs["scope_amendment"]["path"],
        expected_scope_amendment_sha256=refs["scope_amendment"]["sha256"],
        family_configs=[{"family_id":family,**spec["protocol"]} for family,spec in initial["families"].items()],
        output_directory=None,project_root=capture.root)
    if canonical(rebound.plan)!=canonical(initial):
        raise ValueError("Main plan differs from genuine complete producer re-admission")
    plan,selection=provenance.replay_main(capture,refs["main_selection_plan"],refs["main_selection_receipt"])
    source=secondary.admit_secondary(capture.root/refs["secondary_configuration"]["path"],
        refs["secondary_configuration"]["sha256"],project_root=capture.root)
    settings=source.settings
    if settings["main_selection_plan"]!=refs["main_selection_plan"] or settings["main_selection_receipt"]!=refs["main_selection_receipt"]:
        raise ValueError("Secondary configuration must bind the exact main selection")
    candidate_config=capture.read(plan["families"]["candidate"]["protocol"])
    candidate_spec=candidate_config["model_specs"][selection["candidate_choice"]["configuration_id"]]
    _,controls=provenance.replay_secondary(capture,refs["secondary_configuration"],refs["secondary_selection_receipt"],
        refs["main_selection_plan"],refs["main_selection_receipt"],candidate_spec)
    if controls["source_binding"]!=plan["source_binding"]:
        raise ValueError("Main/secondary source binding differs")
    candidate=provenance._bundle(capture,selection["candidate_choice"],selection["cells"],plan["families"]["candidate"]["protocol"])
    baseline_choice=selection["strongest_required_baseline"];family=baseline_choice["family_id"]
    baseline=provenance._bundle(capture,baseline_choice,selection["cells"],plan["families"][family]["protocol"],
        smoother=family=="reference_smoother")
    for bundle in (candidate,baseline):
        provenance.validate_primary_bundle(capture,bundle,selection)
    methods={"candidate":candidate,"strongest_required_baseline":baseline}
    methods.update({control:provenance._bundle(capture,row["choice"],controls["cells"],refs["secondary_configuration"],family=control)
                    for control,row in controls["controls"].items()})
    expected=secondary.controls_for(candidate_spec)
    if set(methods)!={"candidate","strongest_required_baseline",*expected} or selection["ordered_parent_ids"]!=[p["parent_id"] for p in plan["validation_parents"]] or len(plan["validation_parents"])!=16:
        raise ValueError("Exact selected method matrix and all sixteen ordered VAL parents required")
    admission=TemperatureAdmission(capture,source,refs,plan,selection,controls,methods,deepcopy(FIT_POLICY),code_references())
    admission.content=_state(admission);admission.check()
    return admission


def _numerical_inputs(admission):
    """Selected logits and external labels are only loaded here, after full replay/admission."""
    admission.check()
    cases=admission.secondary_admission.dataset.cases("val")
    by_id={case.parent_id:case for case in cases};parents=admission.plan["validation_parents"]
    if len(cases)!=16 or len(by_id)!=16 or set(by_id)!={row["parent_id"] for row in parents}:
        raise ValueError("All sixteen actual source VAL parents required")
    cases=[by_id[row["parent_id"]] for row in parents];labels=[];fingerprints=[]
    for case,contract in zip(cases,parents):
        if (main.tensor_fingerprint(case.positions)!=contract["positions_sha256"] or
                main.tensor_fingerprint(case.query_node_ids)!=contract["query_ids_sha256"] or
                list(case.labels.shape)!=contract["logit_shape"][:2]):
            raise ValueError("External VAL labels/coordinate/query contracts differ")
        labels.append(case.labels.numpy().copy())
        fingerprints.append(main.tensor_fingerprint(case.labels))
    records={};capture=main._Capture(admission.capture.root)
    for method,bundle in admission.methods.items():
        for seed in bundle["seed_bundles"]:
            ref=_ref(capture,seed["development_prediction"])
            content=capture.read(ref,json_file=False)
            value=torch.load(io.BytesIO(content),map_location="cpu",weights_only=True)
            if (value.get("seed")!=seed["seed"] or value.get("protocol_sha256")!=bundle["protocol"]["sha256"] or
                    value.get("parent_ids")!=[row["parent_id"] for row in parents] or
                    len(value.get("logits",[]))!=16 or len(value.get("positions",[]))!=16 or
                    len(value.get("query_node_ids",[]))!=16):
                raise ValueError("Selected raw DEV logits must bind exact seed/protocol/all ordered parents")
            zs=[]
            for contract,z,x,q in zip(parents,value["logits"],value["positions"],value["query_node_ids"]):
                if (not isinstance(z,torch.Tensor) or z.device.type!="cpu" or z.dtype not in (torch.float32,torch.float64) or
                        list(z.shape)!=contract["logit_shape"] or not bool(torch.isfinite(z).all()) or
                        main.tensor_fingerprint(x)!=contract["positions_sha256"] or main.tensor_fingerprint(q)!=contract["query_ids_sha256"]):
                    raise ValueError("Selected DEV raw logit/coordinate/query arrays differ")
                zs.append(z.numpy().copy())
            zs,ys=_arrays(zs,labels)
            raw=[_raw_loss(z,y,1.) for z,y in zip(zs,ys)]
            if not math.isclose(math.fsum(raw)/16,seed["validation_nll"],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("Selected raw validation NLL differs from admitted external labels")
            records[(method,seed["seed"])]=(zs,ys,seed)
    def check():
        admission.check(full=False);capture.verify_stable()
        if [main.tensor_fingerprint(case.labels) for case in cases]!=fingerprints:
            raise ValueError("In-memory external VAL labels drift")
        if any(main.tensor_fingerprint(case.positions)!=row["positions_sha256"] or
               main.tensor_fingerprint(case.query_node_ids)!=row["query_ids_sha256"] for case,row in zip(cases,parents)):
            raise ValueError("In-memory external VAL coordinate/query drift")
    check()
    return records,capture,check,fingerprints


def _file_stamp(s):
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _verify_staged(path,identity,content,expected_stamp=None):
    lexical=path.lstat()
    if (lexical.st_dev,lexical.st_ino)!=identity:raise ValueError("Owned fit artifact inode changed")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if (before.st_dev,before.st_ino)!=identity or not stat.S_ISREG(before.st_mode):
            raise ValueError("Owned fit artifact inode changed")
        if expected_stamp is not None and _file_stamp(before)!=expected_stamp:
            raise ValueError("Owned staged fit artifact stat drift")
        digest=hashlib.sha256()
        with os.fdopen(fd,"rb",closefd=False) as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b""):digest.update(chunk)
        after=os.fstat(fd)
        if _file_stamp(before)!=_file_stamp(after) or before.st_size!=len(content) or digest.hexdigest()!=sha(content):
            raise ValueError("Owned staged fit artifact byte/hash drift")
    finally:os.close(fd)


def _unlink_owned(path,identity):
    try:current=path.lstat()
    except FileNotFoundError:return
    if (current.st_dev,current.st_ino)==identity:path.unlink()


def _publish(output, content, check):
    # Hold original FD through cleanup so an unlinked inode cannot be reused by a foreign replacement.
    with tempfile.NamedTemporaryFile(mode="wb",dir=output.parent,prefix=".val_temperature_pending_",delete=False) as handle:
        pending=Path(handle.name);current=os.fstat(handle.fileno());identity=(current.st_dev,current.st_ino)
        published=False
        try:
            handle.write(content);handle.flush();os.fsync(handle.fileno());stamp=_file_stamp(os.fstat(handle.fileno()))
            check();_verify_staged(pending,identity,content,stamp)
            os.link(pending,output,follow_symlinks=False);published=True
            _verify_staged(output,identity,content)
        except BaseException:
            if published:_unlink_owned(output,identity)
            raise
        finally:_unlink_owned(pending,identity)


def fit_validation_temperatures(admission,*,output_directory):
    """Emit actual scalar fit records, exact lock-compatible receipts and complete manifest."""
    if not isinstance(admission,TemperatureAdmission):
        raise ValueError("Genuine main and secondary admission required")
    admission.check()
    root=admission.capture.root
    output=combined.guard_development_path(output_directory,project_root=root)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh exclusive development fitting directory with existing parent required")
    inputs,numerical_capture,check,label_fingerprints=_numerical_inputs(admission)
    # All selected inputs have been verified before making a fitting output.
    output.mkdir();s=output.lstat();owned=(s.st_dev,s.st_ino)
    def live():
        check();s=output.lstat()
        if (s.st_dev,s.st_ino)!=owned:raise ValueError("Owned fitting output directory changed")
    refs=admission.references;records=[];active=None
    try:
        for (method,seed),(zs,ys,bundle) in inputs.items():
            active={"method_id":method,"seed":seed};live()
            optimization=fit_scalar_temperature(zs,ys,live_check=live)
            optimization.update(method_id=method,seed=seed,checkpoint=bundle["checkpoint"],
                development_prediction=bundle["development_prediction"],
                validation_parent_ids=admission.selection["ordered_parent_ids"],
                external_label_sha256=label_fingerprints,
                source_binding=admission.plan["source_binding"],input_artifacts=numerical_capture.provenance(),
                main_selection_receipt_sha256=refs["main_selection_receipt"]["sha256"],
                secondary_selection_receipt_sha256=refs["secondary_selection_receipt"]["sha256"],
                code_artifacts=admission.code_artifacts)
            record_path=output/f"{method}_s{seed}_optimization.json";content=canonical(optimization)
            _publish(record_path,content,live)
            record_ref={"path":str(record_path.relative_to(root)),"sha256":sha(content)}
            evidence={"schema_version":1,"kind":"validation_temperature_fit_receipt","status":"complete",
                "method_id":method,"seed":seed,"temperature":optimization["temperature"],
                "checkpoint":bundle["checkpoint"],"development_prediction":bundle["development_prediction"],
                "main_selection_receipt_sha256":refs["main_selection_receipt"]["sha256"],
                "secondary_selection_receipt_sha256":refs["secondary_selection_receipt"]["sha256"],
                "source_binding":admission.plan["source_binding"],"validation_parents":admission.plan["validation_parents"],
                "fit_scope":"fresh_development_validation_only","unchanged_model_weights":True,"final_test_access":False,
                "fit_code_artifacts":admission.code_artifacts,"fit_record_artifacts":[record_ref]}
            path=output/f"{method}_s{seed}_fit.json";content=canonical(evidence);_publish(path,content,live)
            evidence_ref={"path":str(path.relative_to(root)),"sha256":sha(content)}
            admission.capture.read(record_ref);admission.capture.read(evidence_ref)
            records.append({key:evidence[key] for key in ("method_id","seed","temperature","checkpoint","development_prediction","fit_scope")}
                |{"validation_parent_ids":admission.selection["ordered_parent_ids"],"fit_evidence":evidence_ref})
        manifest={"schema_version":1,"kind":"prespecified_validation_temperature_manifest",
            "status":"frozen_before_test_preparation","main_selection_receipt_sha256":refs["main_selection_receipt"]["sha256"],
            "secondary_selection_receipt_sha256":refs["secondary_selection_receipt"]["sha256"],
            "training_seeds":[101,202,303],"records":records,"final_test_access":False}
        # Same frozen lock validator must accept all emitted evidence before manifest publication.
        provenance.validate_temperature_manifest(admission.capture,manifest,
            {key:admission.methods[key] for key in ("candidate","strongest_required_baseline")},
            {key:value for key,value in admission.methods.items() if key not in ("candidate","strongest_required_baseline")},
            admission.selection,admission.secondary_selection,refs,admission.plan)
        admission.check(full=True)
        content=canonical(manifest);path=output/"manifest.json";_publish(path,content,live)
        return {"values":manifest,"bytes":content,"sha256":sha(content),"path":str(path)}
    except BaseException as error:
        # Earlier complete fits are real failed-attempt evidence. Never delete them or emit a manifest.
        failure={"schema_version":1,"kind":"validation_temperature_failed_attempt","status":"failed",
            "active_cell":active,"error_type":type(error).__name__,"reason":str(error),
            "main_selection_receipt_sha256":refs["main_selection_receipt"]["sha256"],
            "secondary_selection_receipt_sha256":refs["secondary_selection_receipt"]["sha256"],
            "manifest_emitted":(output/"manifest.json").exists(),"final_test_access":False}
        if isinstance(error,TemperatureFitFailure):failure["optimization_record"]=error.record
        try:
            s=output.lstat()
            if (s.st_dev,s.st_ino)==owned:_publish(output/"failure.json",canonical(failure),lambda:None)
        except (OSError,ValueError):pass
        raise


def main_cli(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("main-plan","main-selection","secondary-config","secondary-selection","scope-amendment"):
        parser.add_argument("--"+name,required=True);parser.add_argument("--expected-"+name+"-sha256",required=True)
    parser.add_argument("--fit",action="store_true");parser.add_argument("--output-directory")
    args=parser.parse_args(argv)
    if args.fit and not args.output_directory:raise ValueError("Explicit new output directory required for actual fitting")
    if args.output_directory:combined.guard_development_path(args.output_directory)
    def ref(name):
        return {"path":getattr(args,name.replace("-","_")),"sha256":getattr(args,"expected_"+name.replace("-","_")+"_sha256")}
    admission=admit_temperature_fitting(ref("main-plan"),ref("main-selection"),ref("secondary-config"),
        ref("secondary-selection"),ref("scope-amendment"))
    if not args.fit:
        print(json.dumps({"event":"validation_temperature_admitted","scalar_fits":len(admission.methods)*3,"manifest_written":False}),flush=True)
        return
    result=fit_validation_temperatures(admission,output_directory=args.output_directory)
    print(json.dumps({"event":"validation_temperatures_fitted","sha256":result["sha256"],"path":result["path"],
                      "records":len(result["values"]["records"]),"final_test_access":False}),flush=True)


if __name__=="__main__":main_cli()

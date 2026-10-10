"""Conditional matched attribution controls after complete main development selection."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import torch
from .resources import ROOT
from . import development_training_bindings as common
from . import development_selection as main
from . import development_selection_bindings as combined
from .development_execution import DETERMINISTIC_POLICY

PROPOSAL="configs/secondary_development_search_plan_v1.json"
EXTRA=("anchor_controls.py","secondary_control_bindings.py","train_secondary_development.py",
       "development_selection.py","development_selection_bindings.py")
RATES=(.0003,.001)
FIXED={"input_dim":47,"num_classes":4,"hidden_dim":32,"num_layers":2,"latent_dim":8,
       "num_scales":3,"num_anchors":32,"epochs":48,"patience":8,"weight_decay":.0001,
       "rate_weight_decay":0.,"initial_coupling":[5.,20.,80.],"initial_damping":1.}
SCOPE=common.family_scope("candidate")["input_scope"]
INTERPRETATION="Matched attribution comparisons retain observed graph-local exchange. They are not graph-free nulls. Parameter-schema equality does not establish equal effective capacity. No primary re-selection, main-family omission or final-test promotion."


def implementation_hashes():return common.snapshot(extra=EXTRA)


def proposal_values():
    return {"schema_version":1,"kind":"conditional_secondary_development_search_proposal",
        "status":"non_executable_until_genuine_completed_main_selection_and_preparation_are_bound",
        "splits":{"train":64,"val":16,"test":0},"training_seeds":[101,202,303],
        "fixed":deepcopy(FIXED),"learning_rates":list(RATES),"nll_policy":main.RAW_NLL,
        "checkpoint_rule":"earliest exact minimum equal-parent raw-logit validation NLL",
        "control_selection_rule":"all three seed cells succeed; global best+0.0001 then parameter count and frozen lexical ID within each control",
        "required_main_families":list(combined.FAMILIES),
        "conditional_rule":{"always":["gain_matched_diagonal","static_local"],
            "selected_age":"matched_scalar_age_full","selected_plain_depth2":"noncontextual",
            "selected_plain_depth1":None},
        "maximum_secondary_optimizer_starts":18,"existing_secondary_reserve":72,
        "maximum_combined_prescribed_starts":288,"deterministic_execution_policy":deepcopy(DETERMINISTIC_POLICY),
        "input_scope":SCOPE,"interpretation":INTERPRETATION,"final_test_access":False,
        "reviewed_implementation_sha256":implementation_hashes(),
        "binder_script":{"path":"scripts/bind_secondary_development.py","sha256":common.sha(ROOT/"scripts/bind_secondary_development.py")}}


def controls_for(spec):
    if not isinstance(spec,dict) or set(spec)!={"geometry","kernel_depth","coupling","rate_parameterization"} or spec["geometry"] not in ("plain","age") or type(spec["kernel_depth"]) is not int or spec["kernel_depth"] not in (1,2) or spec["coupling"]!="full" or spec["rate_parameterization"]!="log":
        raise ValueError("Exact selected plain/age full/log candidate specification required")
    controls={"gain_matched_diagonal":{**spec,"control":"gain_matched_diagonal","coupling":"gain_matched_diagonal"},
              "static_local":{**spec,"control":"static_local"}}
    if spec["geometry"]=="age":controls["matched_scalar_age_full"]={**spec,"control":"matched_scalar_age_full","geometry":"matched_scalar_age"}
    elif spec["kernel_depth"]==2:controls["noncontextual"]={**spec,"control":"noncontextual"}
    return controls


def _ref(capture,ref):
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:raise ValueError("Exact path/SHA reference required")
    path=combined.guard_development_path(ref["path"],project_root=capture.root)
    return combined._ref(capture,path,ref["sha256"])


def verify_main_selection(plan_ref,receipt_ref,capture,*,completion,completion_path,completion_sha256,source_snapshot):
    """Reconstruct and replay complete main evidence, never trust winner JSON alone."""
    plan_ref=_ref(capture,plan_ref);receipt_ref=_ref(capture,receipt_ref)
    plan=capture.read(plan_ref);receipt=capture.read(receipt_ref)
    main.load_development_selection_plan(capture.root/plan_ref["path"],plan_ref["sha256"],project_root=capture.root)
    if set(plan["families"])!=set(combined.FAMILIES) or set(plan["required_baseline_families"])!=set(combined.FAMILIES)-{"candidate"}:
        raise ValueError("Complete prescribed ten-family main study required")
    for family,spec in plan["families"].items():
        config=capture.read(spec["protocol"])
        combined.validate_producer_settings(family,config,completion=completion,completion_path=completion_path,
            completion_sha256=completion_sha256,capture=capture,source_snapshot=source_snapshot)
        contract=combined._family_contract(family)
        if any(config.get(k)!=contract[k] for k in ("input_scope","cross_tree_aggregation")) or config.get("bound_implementation_sha256")!=contract["pins"]:
            raise ValueError("Main protocol reviewed scope/source pins differ")
        policy=config.get(combined.POLICY_FIELDS[family])
        if not isinstance(policy,dict) or set(policy)!=set(DETERMINISTIC_POLICY) or any(type(policy[k]) is not type(v) or policy[k]!=v for k,v in DETERMINISTIC_POLICY.items()):
            raise ValueError("Main strict numerical policy differs")
    if receipt.get("kind")!="immutable_fresh_development_selection_receipt" or receipt.get("plan_sha256")!=plan_ref["sha256"] or receipt.get("matrix_resolved") is not True or receipt.get("baseline_selection_eligible") is not True or receipt.get("final_test_access") is not False or receipt.get("promotes_final_test_phase") is not False or receipt.get("required_baseline_families")!=plan["required_baseline_families"]:
        raise ValueError("Genuine resolved main selection receipt required")
    if not receipt.get("candidate_choice") or not receipt.get("strongest_required_baseline"):
        raise ValueError("Candidate and strongest required baseline choices required")
    provenance=receipt.get("source_artifacts")
    if not isinstance(provenance,list):raise ValueError("Captured main evidence provenance required")
    by_sha={}
    for record in provenance:
        ref=_ref(capture,{k:record[k] for k in ("path","sha256")})
        by_sha.setdefault(ref["sha256"],{})[ref["path"]]=ref
    def evidence(digest):
        rows=by_sha.get(digest,{})
        if len(rows)!=1:raise ValueError("Unique actual main run/ledger artifact required")
        return next(iter(rows.values()))
    cells=receipt.get("cells")
    if not isinstance(cells,list) or len(cells)!=240 or any(cell.get("status")=="missing" for cell in cells):
        raise ValueError("Every prescribed main cell must be resolved")
    runs={}
    for cell in cells:
        if cell.get("status")=="success":
            key=(cell["family_id"],cell["run_receipt_sha256"])
            runs[key]={"family_id":key[0],**evidence(key[1])}
    ledger_hashes=list(dict.fromkeys(entry["ledger_sha256"] for entry in receipt.get("failures_and_retry_ledger",[])))
    for cell in cells:
        if cell.get("status")=="failed" and cell["ledger_sha256"] not in ledger_hashes:ledger_hashes.append(cell["ledger_sha256"])
    replay=main.synthesize_development_selection(capture.root/plan_ref["path"],plan_ref["sha256"],
        run_receipts=list(runs.values()),failure_ledgers=[evidence(digest) for digest in ledger_hashes],project_root=capture.root).payload
    actual=json.dumps({k:v for k,v in replay.items() if k!="created_utc"},sort_keys=True,allow_nan=False)
    expected=json.dumps({k:v for k,v in receipt.items() if k!="created_utc"},sort_keys=True,allow_nan=False)
    if actual!=expected:
        raise ValueError("Main selection receipt does not replay from exact captured evidence")
    # Bind every artifact consumed by replay, including selected weights/logits.
    for record in provenance:
        capture.read({k:record[k] for k in ("path","sha256")},json_file=False,allow_source_reuse=True) if str(record["path"]).endswith((".pt",".py")) else capture.read({k:record[k] for k in ("path","sha256")})
    return plan,receipt


def _assemble(proposal_ref,plan_ref,receipt_ref,plan,receipt,capture,completion):
    chosen=receipt["candidate_choice"]
    if chosen["family_id"]!="candidate":raise ValueError("Main candidate family differs")
    protocol=plan["families"]["candidate"]["protocol"]
    candidate=capture.read(protocol)
    for key,value in FIXED.items():
        actual=candidate.get(key,47 if key=="input_dim" else None)
        if actual!=value or isinstance(actual,bool):raise ValueError("Selected source dimensions/optimization budget differ")
    spec=candidate["model_specs"].get(chosen["configuration_id"])
    controls=controls_for(spec);models=[];specs={};rates={}
    for control,control_spec in controls.items():
        for rate in RATES:
            name=f"{control}_lr{int(rate*1e6):04d}"
            models.append(name);specs[name]=deepcopy(control_spec);rates[name]=rate
    selected=[cell for cell in receipt["cells"] if cell["family_id"]=="candidate" and cell["configuration_id"]==chosen["configuration_id"]]
    if len(selected)!=3 or sorted(cell["seed"] for cell in selected)!=[101,202,303] or any(cell["status"]!="success" for cell in selected):
        raise ValueError("All three chosen-candidate checkpoint cells required")
    main_binding=plan["source_binding"]
    settings={"schema_version":1,"kind":"fresh_development_secondary_controls_bound_config",
        "family":"secondary_controls","development_namespace":"development-v1","splits":{"train":64,"val":16,"test":0},
        "training_seeds":[101,202,303],**deepcopy(FIXED),**deepcopy(main_binding),
        "completion_receipt":str(capture.root/plan["source_artifacts"]["completion"]["path"]),
        "preparation_execution_freeze":str(capture.root/plan["source_artifacts"]["preparation_execution_freeze"]["path"]),
        "boundary_data":str((capture.root/plan["source_artifacts"]["boundary_manifest"]["path"]).parent),
        "age_data":str((capture.root/plan["source_artifacts"]["age_manifest"]["path"]).parent),
        "conditional_proposal":proposal_ref,"main_selection_plan":plan_ref,"main_selection_receipt":receipt_ref,
        "candidate_protocol":protocol,"selected_candidate":deepcopy(chosen),"selected_candidate_spec":deepcopy(spec),
        "selected_candidate_checkpoints":[{k:cell[k] for k in ("seed","checkpoint","checkpoint_sha256","protocol_sha256","implementation_sha256")} for cell in sorted(selected,key=lambda c:c["seed"])],
        "primary_strongest_baseline":deepcopy(receipt["strongest_required_baseline"]),
        "validation_parents":deepcopy(plan["validation_parents"]),"controls":list(controls),
        "models":models,"model_specs":specs,"model_learning_rates":rates,"optimizer_starts":len(models)*3,
        "maximum_secondary_optimizer_starts":18,"existing_secondary_reserve":72,"maximum_combined_prescribed_starts":288,
        "nll_policy":main.RAW_NLL,"selection":common.SELECTION,"tie_tolerance":.0001,
        "deterministic_execution_policy":deepcopy(DETERMINISTIC_POLICY),"input_scope":SCOPE,
        "interpretation":INTERPRETATION,"bound_implementation_sha256":implementation_hashes(),"final_test_access":False}
    common.check_completion_settings(settings,completion)
    if settings["optimizer_starts"]>18:raise ValueError("Secondary eighteen-start cap exceeded")
    return settings


@dataclass
class SecondaryAdmission:
    settings:dict
    content:bytes
    sha256:str
    capture:object
    source_lock:object
    dataset:object
    ages:object
    def check(self,full=True):
        if (json.dumps(self.settings,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()!=self.content:raise ValueError("In-memory secondary settings drift")
        if self.settings["bound_implementation_sha256"]!=implementation_hashes():raise ValueError("Secondary dependency drift")
        if full:combined._check_all(self.capture,self.source_lock)
        else:self.capture.verify_stable();self.source_lock.check(full=False)
        return dict(self.settings["bound_implementation_sha256"])


def _bind(proposal_ref,plan_ref,receipt_ref,*,project_root=ROOT):
    capture=main._Capture(project_root)
    proposal_ref=_ref(capture,proposal_ref);plan_ref=_ref(capture,plan_ref);receipt_ref=_ref(capture,receipt_ref)
    proposal=capture.read(proposal_ref)
    if proposal!=proposal_values():raise ValueError("Frozen conditional proposal/code/policy differs")
    for name,digest in proposal["reviewed_implementation_sha256"].items():combined._dependency_ref(capture,name,digest)
    capture.read(proposal["binder_script"],json_file=False,allow_source_reuse=True)
    initial_plan=capture.read(plan_ref)
    completion_ref=_ref(capture,initial_plan["source_artifacts"]["completion"])
    completion_path=capture.root/completion_ref["path"]
    completion,dataset,ages,source_lock=common.verify_completion(completion_path,completion_ref["sha256"],with_snapshot=True)
    plan,receipt=verify_main_selection(plan_ref,receipt_ref,capture,completion=completion,
        completion_path=completion_path,completion_sha256=completion_ref["sha256"],source_snapshot=source_lock)
    cases=dataset.cases("val")
    by_id={case.parent_id:case for case in cases}
    if len(cases)!=16 or len(by_id)!=16:raise ValueError("Every actual observed validation parent required")
    actual=[]
    for index in range(16):
        parent_id=plan["source_binding"]["source_dataset_manifest_sha256"]+":val:"+str(index)
        if parent_id not in by_id:raise ValueError("Observed parent/source identity differs")
        case=by_id[parent_id]
        actual.append({"parent_id":parent_id,"positions_sha256":main.tensor_fingerprint(case.positions),
            "query_ids_sha256":main.tensor_fingerprint(case.query_node_ids),"logit_shape":[len(case.positions),len(case.query_node_ids),4]})
    if actual!=plan["validation_parents"]:raise ValueError("Main contracts differ from verified observed coordinate/query order")
    settings=_assemble(proposal_ref,plan_ref,receipt_ref,plan,receipt,capture,completion)
    content=(json.dumps(settings,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    admission=SecondaryAdmission(settings,content,hashlib.sha256(content).hexdigest(),capture,source_lock,dataset,ages)
    admission.check()
    return admission


def bind_secondary_development(proposal_ref,plan_ref,receipt_ref,*,output_path=None,project_root=ROOT):
    capture=main._Capture(project_root)
    output=None if output_path is None else combined.guard_development_path(output_path,project_root=capture.root)
    if output is not None and (output.exists() or not output.parent.is_dir()):raise ValueError("Fresh exclusive secondary config path required")
    result=_bind(proposal_ref,plan_ref,receipt_ref,project_root=project_root)
    if output is not None:
        with output.open("xb") as handle:handle.write(result.content)
        try:result.check()
        except BaseException:output.unlink();raise
    return result


def admit_secondary(config_path,expected_sha256,*,project_root=ROOT):
    capture=main._Capture(project_root);ref=_ref(capture,{"path":str(config_path),"sha256":expected_sha256})
    settings=capture.read(ref)
    if not isinstance(settings,dict) or settings.get("kind")!="fresh_development_secondary_controls_bound_config":raise ValueError("Only a genuine completion/selection-bound secondary config is executable")
    result=_bind(settings["conditional_proposal"],settings["main_selection_plan"],settings["main_selection_receipt"],project_root=project_root)
    if settings!=result.settings or capture.files[str(capture.root/ref["path"])]["sha256"]!=result.sha256:
        raise ValueError("Captured secondary config differs from factual selected/proposal/source settings")
    result.capture.read(ref)
    result.check()
    return result


def synthesize_secondary_selection(config_path,expected_sha256,*,run_receipts=(),failure_ledgers=(),output_path=None,project_root=ROOT):
    """Strict independent two-LR/three-seed attribution selection, no main re-selection."""
    preflight=main._Capture(project_root)
    _ref(preflight,{"path":str(config_path),"sha256":expected_sha256})
    for ref in (*run_receipts,*failure_ledgers):_ref(preflight,ref)
    if output_path is not None:combined.guard_development_path(output_path,project_root=project_root)
    admission=admit_secondary(config_path,expected_sha256,project_root=project_root)
    settings=admission.settings;capture=admission.capture
    expected={(name,seed) for name in settings["models"] for seed in (101,202,303)}
    cells={};run_artifacts=[];failures=[]
    spec={"role":"secondary_control","protocol":{"path":str(config_path),"sha256":expected_sha256},
        "expected_implementation_sha256":settings["bound_implementation_sha256"]}
    cell_plan={"families":{"secondary_controls":spec},"source_binding":{key:settings[key] for key in main.BINDINGS},"validation_parents":settings["validation_parents"]}
    cases={case.parent_id:case for case in admission.dataset.cases("val")}
    from .confirmatory_metrics import logit_metrics
    from .train_secondary_development import make_control
    for ref in run_receipts:
        run=capture.read(_ref(capture,ref))
        if run.get("kind")!="secondary_development_run" or run.get("protocol_sha256")!=expected_sha256 or run.get("main_selection_receipt_sha256")!=settings["main_selection_receipt"]["sha256"] or run.get("source_binding")!=cell_plan["source_binding"] or run.get("initial_implementation_sha256")!=settings["bound_implementation_sha256"] or json.dumps(run.get("deterministic_execution_policy"),sort_keys=True)!=json.dumps(DETERMINISTIC_POLICY,sort_keys=True):
            raise ValueError("Secondary run source/config/main/policy binding differs")
        seed=run.get("training_seed")
        if type(seed) is not int or seed not in (101,202,303) or not isinstance(run.get("models"),dict) or not run["models"]:
            raise ValueError("Prescribed nonempty secondary run required")
        for name,body in run["models"].items():
            key=(name,seed)
            if key not in expected or key in cells:raise ValueError("Unexpected or duplicated secondary cell")
            if body.get("control_spec")!=settings["model_specs"][name] or body.get("learning_rate")!=settings["model_learning_rates"][name] or body.get("ending_source_binding")!=cell_plan["source_binding"] or body.get("ending_input_artifact_sha256")!=admission.source_lock.hashes:
                raise ValueError("Matched control/spec/source record differs")
            model=make_control(name,settings)
            cell=main._cell(capture,cell_plan,{"secondary_controls":settings},"secondary_controls",name,seed,body,ref["sha256"])
            checkpoint=main._artifact(main._Capture(capture.root),body["checkpoint"],body["checkpoint_sha256"])
            expected_state=model.state_dict()
            if set(checkpoint["model"])!=set(expected_state) or any(checkpoint["model"][key].shape!=value.shape or checkpoint["model"][key].dtype!=value.dtype for key,value in expected_state.items()):raise ValueError("Exact checkpoint parameter schema/dtype required")
            model.load_state_dict(checkpoint["model"],strict=True)
            if body.get("model_configuration")!=model.configuration_manifest() or body.get("parameters")!=sum(p.numel() for p in model.parameters()):
                raise ValueError("Control manifest/capacity record differs from frozen implementation")
            if json.dumps(checkpoint.get("deterministic_execution_policy"),sort_keys=True)!=json.dumps(DETERMINISTIC_POLICY,sort_keys=True):raise ValueError("Checkpoint numerical policy differs")
            if checkpoint.get("main_selection_receipt_sha256")!=settings["main_selection_receipt"]["sha256"] or checkpoint.get("control_spec")!=settings["model_specs"][name] or checkpoint.get("source_binding")!=cell_plan["source_binding"]:
                raise ValueError("Checkpoint matched-main/source/spec binding differs")
            score=cell["validation_nll"]
            if body.get("original_selected_validation_nll")!=score or body.get("reloaded_validation_nll")!=score or body.get("checkpoint_replay_nll_delta")!=0.:
                raise ValueError("Exact zero deterministic checkpoint replay delta required")
            prediction=main._artifact(main._Capture(capture.root),body["validation_predictions"],body["validation_predictions_sha256"])
            if prediction.get("source_binding")!=cell_plan["source_binding"] or prediction.get("control_spec")!=settings["model_specs"][name] or prediction.get("implementation_sha256")!=settings["bound_implementation_sha256"] or prediction.get("main_selection_receipt_sha256")!=settings["main_selection_receipt"]["sha256"]:
                raise ValueError("Prediction matched-main/source/spec binding differs")
            if checkpoint.get("original_selected_logit_sha256")!=[main.tensor_fingerprint(z) for z in prediction["logits"]] or body.get("checkpoint_replay_max_abs_logit_delta")!=0.:raise ValueError("Exact selected/reloaded raw-logit replay required")
            if json.dumps(prediction.get("deterministic_execution_policy"),sort_keys=True)!=json.dumps(DETERMINISTIC_POLICY,sort_keys=True):raise ValueError("Prediction numerical policy differs")
            for parent,z,reported in zip(settings["validation_parents"],prediction["logits"],cell["per_parent_nll"]):
                actual=logit_metrics(z.numpy(),cases[parent["parent_id"]].labels.numpy())["nll"]
                if not math.isclose(actual,reported,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("Raw prediction NLL differs from verified external parent labels")
            cells[key]=cell
        run_artifacts.append(ref)
    for ref in failure_ledgers:
        ledger=capture.read(_ref(capture,ref))
        if set(ledger)!={"kind","config_sha256","main_selection_receipt_sha256","entries"} or ledger["kind"]!="secondary_development_failure_ledger" or ledger["config_sha256"]!=expected_sha256 or ledger["main_selection_receipt_sha256"]!=settings["main_selection_receipt"]["sha256"]:
            raise ValueError("Secondary failure ledger differs")
        for entry in ledger["entries"]:
            if set(entry)!={"configuration_id","seed","attempt_id","terminal","reason","test_access"} or type(entry["seed"]) is not int or entry["terminal"] is not True or entry["test_access"] is not False or not isinstance(entry["reason"],str) or not entry["reason"]:
                raise ValueError("One terminal recorded start per secondary cell required")
            main._safe(entry["attempt_id"]);key=(entry["configuration_id"],entry["seed"])
            if key not in expected or key in cells:raise ValueError("Duplicate/extra failed secondary cell")
            cells[key]={"status":"failed",**entry};failures.append({**entry,"ledger_sha256":ref["sha256"]})
    matrix=[cells.get((name,seed),{"status":"missing","configuration_id":name,"seed":seed}) for name in settings["models"] for seed in (101,202,303)]
    controls={}
    for control in settings["controls"]:
        rows=[]
        for name in settings["models"]:
            if settings["model_specs"][name]["control"]!=control:continue
            group=[cells.get((name,seed),{}) for seed in (101,202,303)]
            success=[row for row in group if row.get("status")=="success"]
            if len({row["parameters"] for row in success})>1:raise ValueError("Secondary parameter count changes across seeds")
            rows.append({"family_id":control,"configuration_id":name,"eligible":len(success)==3,
                "parameters":success[0]["parameters"] if success else None,
                "mean_three_seed_nll":math.fsum(row["validation_nll"] for row in success)/3 if len(success)==3 else None})
        resolved=all(cells.get((row["configuration_id"],seed),{}).get("status") in ("success","failed") for row in rows for seed in (101,202,303))
        controls[control]={"matrix_resolved":resolved,"configurations":rows,"choice":main._winner(rows) if resolved else None}
    admission.check()
    result={"kind":"secondary_development_attribution_selection_receipt","schema_version":1,"config_sha256":expected_sha256,
        "main_selection_plan":settings["main_selection_plan"],"main_selection_receipt":settings["main_selection_receipt"],
        "selected_candidate":settings["selected_candidate"],"primary_strongest_baseline":settings["primary_strongest_baseline"],
        "controls":controls,"cells":matrix,"failures":failures,"optimizer_starts_recorded":len(cells),
        "maximum_optimizer_starts":settings["optimizer_starts"],"matrix_resolved":all(row["status"]!="missing" for row in matrix),
        "source_binding":cell_plan["source_binding"],"deterministic_execution_policy":deepcopy(DETERMINISTIC_POLICY),
        "captured_artifacts":capture.provenance(),"interpretation":INTERPRETATION,"primary_selection_changed":False,
        "final_test_access":False,"authorizes_confirmatory_generation":False}
    content=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if output_path is not None:
        output=combined.guard_development_path(output_path,project_root=capture.root)
        if not output.parent.is_dir():raise ValueError("Existing project output parent required")
        with output.open("xb") as handle:handle.write(content)
    return {"values":result,"bytes":content,"sha256":hashlib.sha256(content).hexdigest()}

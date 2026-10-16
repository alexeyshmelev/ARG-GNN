"""CPU checkpoint metadata compatibility checks. No model/data/prediction execution."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,os,resource,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v1"
OLD="research/reviews/C3_CPU_actual_plan_lineage_integration_source_20261016_v1"
SCANNER="research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1"
P=ROOT/HERE
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
checks=[]
def require(ok,message):
    if not ok:raise ValueError(message)
def good(name,ok=True):
    require(ok,name);checks.append(dict(name=name,status="PASS"))
def rejects(name,fn,kind=(ValueError,KeyError,TypeError)):
    try:fn()
    except kind as error:
        checks.append(dict(name=name,status="PASS_REJECTED",exception_type=type(error).__name__));return
    raise AssertionError(name)
def ref(path):
    p=ROOT/path;require(p.is_file() and not p.is_symlink(),"Existing exact regular file")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load_contract():
    spec=importlib.util.spec_from_file_location("_source_only_nested_schema_C3_contract",P/"acceptance_contract_v1.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def functions(path):
    return {x.name:x for x in ast.parse((ROOT/path).read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
def same_ast(name,a,b):
    good(name+" AST unchanged",ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
def F(path):
    return dict(path=path,sha256=hashlib.sha256(("SYNTHETIC_ONLY:"+path).encode()).hexdigest())
def source_checks():
    c=load_contract();manifest=c.manifest_v1()
    good("All original81 frozen source mappings unchanged",
        all(manifest["source_sha256"].get(k)==v for k,v in c.legacy.manifest_v1()["source_sha256"].items()))
    good("Original live000to002 input hashes unchanged", c.OLD_INPUT_REFS==json.loads((ROOT/OLD/"source_manifest.v1.json").read_text())["fixed_old_prediction_input_refs"])
    for phase,record in c.OLD_INPUT_REFS.items():
        old=c.stage_source_v1(dict(input_ref=record,root_driver_ref=c.LEGACY_DRIVER_REF),phase)
        good("Real original input/source lineage unchanged "+phase,len(old["source_sha256"])==81)
    for filename,names in [
        ("worker_entry_v1.py",("execute","matrix_v1","score_v1","validate_saved_v1")),
        ("acceptance_contract_v1.py",("plan_producer_lineage_v1","plan_ready_v1","training_ready_v1",
            "prediction_metadata_v1","matrix_metadata_v1","evidence_request_fields_v1","stage_source_v1",
            "validate_request","validate_input_v1","required_direct_refs_v1")),
        ("root_driver_v1.py",("bind_reviewed_supervisor_v1","main"))]:
        a=functions(HERE+"/"+filename);b=functions(OLD+"/"+filename)
        for name in names:same_ast(filename+" "+name,a[name],b[name])
    good("New source cannot prepare old live parents",c.PHASES==("predict_val_003","predict_val_004","matrix","score"))
    a=functions(HERE+"/prediction_bridge_v1.py");b=functions(SCANNER+"/prediction_bridge_v1.py")
    same_ast("Candidate initialization/RNG scheduling",a["new_candidate_models_v1"],b["new_candidate_models_v1"])
    new=a["predict_v1"];old=b["predict_v1"]
    # Remove only the documented metadata extraction/assertion delta.
    new2=copy.deepcopy(new);old2=copy.deepcopy(old)
    loop_new=next(n for n in new2.body if isinstance(n,ast.For))
    loop_old=next(n for n in old2.body if isinstance(n,ast.For))
    require(len(loop_new.body)==len(loop_old.body)+1,"Exactly one metadata extraction statement added")
    idx=next(i for i,n in enumerate(loop_new.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="loaded_metadata" for t in n.targets))
    require(isinstance(loop_new.body[idx+1],ast.Expr),"Only adjacent candidate assertion is replaced")
    del loop_new.body[idx:idx+2]
    oldidx=next(i for i,n in enumerate(loop_old.body) if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="require"
        and any(isinstance(x,ast.Name) and x.id=="loaded" for x in ast.walk(n)))
    del loop_old.body[oldidx]
    same_ast("Entire interleaved predict body outside metadata compatibility block",new2,old2)
    # The scanner is still the exact original sealed module, never copied or edited.
    good("Original scanner source digest unchanged",ref(c.SCANNER_REF["path"])==c.SCANNER_REF)
    evidence=json.loads((ROOT/OLD/"actual_VAL003_evidence.v1.json").read_text())
    env,spec,request,wait=c.plan_producer_lineage_v1(evidence,"american_c3_val_003")
    plan=c.plan_ready_v1(evidence,"american_c3_val_003")
    good("Real closed VAL003 plan lineage unchanged",plan["haplotypes"]==1600 and plan["focal_records"]==10000
        and plan["sequence_length_bp"]==30000000 and "driver_ref" not in wait)
    failed=json.loads((P/"preserved_prediction_failure.v1.json").read_text())
    actual=c.read(failed["actual_wait_ref"])
    good("Actual failed prediction direct1 with owned drain and no completion retained",
        actual["returncode"]==1 and actual["all_owned_processes_and_groups_drained"] is True and actual["completed_ref"] is None)
    child_ref=ref("research/reviews/native7_H050_stopped_retention_ROOT_activation_20261016_v1/explicit_child_selector.root_active_v1.json")
    selected=c.select_storage_metadata_v1(child_ref)
    good("Actual child metadata selection accepted without importing/configuring guard",
        selected["policy_ref"]==c.ACTIVE_POLICY_REF and selected["source_ref"]==json.loads((P/"approved_selector_sources.v1.json").read_text())["approved"][0]["source_ref"])
    rejects("Previous policy selector cannot replace active child metadata",
        lambda:c.select_storage_metadata_v1(ref("research/reviews/C3_CPU_actual_plan_ROOT_VAL003_20261016_v1/explicit_current_policy_selector.root_active_v1.json")))
    # Conservative immutable role union uses source-only returned child fixture,
    # not the actual guard. No inspection, journal lock or mutation is invoked.
    child=json.loads((ROOT/"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/inert_fixtures.final.v1.json").read_text())["full_role_models"]["guard_state"]
    known=json.loads((P/"approved_selector_sources.v1.json").read_text())["approved"][0]
    for row in child:
        if row["path"]==known["source_manifest_ref"]["path"]:row.update(known["source_manifest_ref"])
    req=dict(caps_ref=F("SYNTHETIC/caps.json"),training_evidence_ref=c.SAVED_ACCEPTANCE_REF,
        integration_review_ref=F("SYNTHETIC/review.json"),ROOT_directive_ref=F("SYNTHETIC/directive.json"),
        actual_stage_evidence_ref=F("SYNTHETIC/evidence.json"),storage_policy_selector_ref=child_ref)
    spec=dict(root_stage_request_ref=F("SYNTHETIC/request.json"),refs=dict(config=F("SYNTHETIC/config.json")))
    required=c.required_direct_refs_v1(spec,req)
    rows={}
    for row in [*child,*required,F(c.HANDOFF)]:
        require(row["path"] not in rows or rows[row["path"]]==row,"No conflicting modelled role")
        rows[row["path"]]=row
    good("Complete child role union fits64 and preserves every returned role",len(rows)<=64 and all(r in rows.values() for r in child))
    good("No actual guard module imported",not any(
        "stopped_retention" in getattr(v,"__file__","") for k,v in list(sys.modules.items()) if getattr(v,"__file__",None)))
    return dict(source_map_count=len(manifest["source_sha256"]),modelled_child_unique_bindings=len(rows),actual_capacity_or_admission_proved=False),new.body,old.body
def metadata_checks(new_body,old_body):
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only explicit environment")
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from arggnn.american_c3_gnn_predict_isolated_v2 import completed_training_v1,eligible_checkpoint_v1
    from arggnn.american_c3_gnn_train_v1 import scheduled_checkpoint_metadata_v1
    acceptance=json.loads((ROOT/"research/reviews/C3_saved600_validator_ROOT_actual_20261016_v1/artifact_acceptance.root_active_v1.json").read_text())
    training_ref=acceptance["training_completion_ref"]
    require(ref(training_ref["path"])==training_ref,"Exact accepted saved600 training metadata")
    training=completed_training_v1(training_ref)
    loop_new=next(n for n in new_body if isinstance(n,ast.For));loop_old=next(n for n in old_body if isinstance(n,ast.For))
    idx=next(i for i,n in enumerate(loop_new.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="loaded_metadata" for t in n.targets))
    newcode=compile(ast.fix_missing_locations(ast.Module(body=loop_new.body[idx:idx+2],type_ignores=[])),"actual_patched_metadata_statements","exec")
    oldidx=next(i for i,n in enumerate(loop_old.body) if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="require"
        and any(isinstance(x,ast.Name) and x.id=="loaded" for x in ast.walk(n)))
    oldcode=compile(ast.fix_missing_locations(ast.Module(body=[loop_old.body[oldidx]],type_ignores=[])),"original_failed_metadata_assertion","exec")
    summaries=[]
    def apply(loaded,checked,candidate,train=training):
        ns=dict(loaded=loaded,checked=checked,candidate=candidate,training=train,
            scheduled_checkpoint_metadata_v1=scheduled_checkpoint_metadata_v1,require=require)
        exec(newcode,ns);return ns["loaded_metadata"]
    for candidate in training["checkpoint_candidates"]:
        r=candidate["checkpoint_ref"];p=ROOT/r["path"]
        require(p.stat().st_size<=2*2**20 and ref(r["path"])==r,"Small exact accepted checkpoint only")
        before=ref(r["path"])
        value=torch.load(p,map_location="cpu",weights_only=True,mmap=True)
        loaded={k:v for k,v in value.items() if k not in ("model_state","optimizer_state")}
        same,checked=eligible_checkpoint_v1(r,training_ref)
        good("Accepted original completed training for epoch%d"%candidate["epoch"],same==training)
        good("Checkpoint really uses nested progress at epoch%d"%candidate["epoch"],
            "epoch" not in loaded and "global_step" not in loaded
            and loaded["trainer_state"]["global_step"]==candidate["global_step"]
            and loaded["candidate"]["completed_epoch"]==candidate["epoch"])
        rejects("Original bridge reproduces KeyError epoch%d"%candidate["epoch"],
            lambda:exec(oldcode,dict(loaded=loaded,checked=checked,candidate=candidate,require=require)),(KeyError,))
        derived=apply(loaded,checked,candidate)
        good("Actual patched statements accept exact metadata epoch%d"%candidate["epoch"],derived==checked)
        wrong=dict(candidate);wrong["epoch"]+=5
        rejects("Wrong scheduled candidate epoch rejects%d"%candidate["epoch"],lambda:apply(loaded,checked,wrong))
        wrong=dict(candidate);wrong["global_step"]+=1
        rejects("Wrong scheduled candidate step rejects%d"%candidate["epoch"],lambda:apply(loaded,checked,wrong))
        wrong=dict(checked);wrong["observation_seal_sha256"]="0"*64
        rejects("Checked data provenance mismatch rejects%d"%candidate["epoch"],lambda:apply(loaded,wrong,candidate))
        wrong=dict(training);wrong["protocol_sha256"]="0"*64
        rejects("Wrong original training protocol rejects%d"%candidate["epoch"],lambda:apply(loaded,checked,candidate,wrong))
        wrong=dict(loaded);wrong["candidate"]=dict(loaded["candidate"]);wrong["candidate"]["completed_epoch"]+=5
        rejects("Malformed checkpoint candidate rejects%d"%candidate["epoch"],lambda:apply(wrong,checked,candidate))
        wrong=dict(loaded);wrong.pop("trainer_state")
        rejects("Missing genuine nested trainer_state rejects%d"%candidate["epoch"],lambda:apply(wrong,checked,candidate))
        good("Accepted checkpoint bytes unchanged epoch%d"%candidate["epoch"],ref(r["path"])==before)
        summaries.append(dict(checkpoint_ref=r,size_bytes=p.stat().st_size,top_level_keys=sorted(loaded),
            nested_trainer_global_step=loaded["trainer_state"]["global_step"],
            nested_candidate=loaded["candidate"],validated_metadata=derived,model_instantiated=False))
        del value,loaded
    good("No CUDA initialized and no model instantiated",not torch.cuda.is_initialized())
    return dict(training_completion_ref=training_ref,checkpoints=summaries,
        checkpoint_bytes_max=1387710,checkpoint_metadata_only=True,
        strict_model_state_loading_source_unchanged_but_not_executed_here=True,
        numerical_inference_training_or_truth_executed=False,torch_version=str(torch.__version__))
def main():
    source,new,old=source_checks()
    meta=metadata_checks(new,old)
    return dict(schema="C3_nested_checkpoint_schema_source_checks_v1",status="PASS_SOURCE_ONLY",
        source=source,real_accepted_checkpoint_metadata=meta,checks=checks,check_count=len(checks),
        peak_main_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        no_guard_probe_or_accounting_or_payload_or_prediction_or_truth_or_training_or_GPU=True,
        source_only=True)
if __name__=="__main__":
    result=main()
    print(json.dumps(result,sort_keys=True,indent=2))

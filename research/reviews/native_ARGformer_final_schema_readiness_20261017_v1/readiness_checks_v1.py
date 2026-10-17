"""Narrow source/metadata proof. No module imports, checkpoint reads or live calls."""
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace
import ast,copy,hashlib,json,sys,traceback

BASE=Path("research/reviews/native_ARGformer_final_schema_readiness_20261017_v1")
SRC=Path("research/reviews/native_VAL000_full_state_resume_source_20261016_v12")
ADMIT=Path("research/reviews/native_VAL000_fresh_continuation_ROOT_v12_root01_oct16")
OBS=Path("research/reviews/user_progress_20261017_v5/saved_native_optimizer_progress.root_v1.json")
def ref(path):
    path=Path(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def require(ok,message):
    if not ok:raise ValueError(message)
def function(text,name):
    return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
def isolated(nodes,ns,filename):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),filename,"exec"),ns)

def main():
    manifest=json.loads((SRC/"source_manifest.v1.json").read_text())
    require(ref(SRC/"source_manifest.v1.json")["sha256"]=="734c8f55fc6a06c20fba9b63b9d72bf733ee9e6b49261f8da26c8bb6b27ae664","Pinned admitted manifest")
    source_refs=manifest["new_executed_source_refs"][:]
    for suffix in ("/common_v1.py","/native_child_v1.py","/composer/core/state.py","/composer/core/time.py"):
        matches=[r for r in manifest["source_authentication_refs"] if r["path"].endswith(suffix) and
            (suffix.startswith("/composer") or "native_VAL000_GPU0_borrowed32_source_20261016_v2/" in r["path"])]
        require(len(matches)==1,"One exact source binding "+suffix)
        source_refs+=matches
    for r in source_refs:
        require(Path(r["path"]).suffix==".py" and ref(r["path"])==r,"Pinned source only "+r["path"])
    admitted=json.loads((ADMIT/"admitted.json").read_text())
    amendment=json.loads((ADMIT/"amendment.root_active_v12.json").read_text())
    inp=json.loads((ADMIT/"input.json").read_text())
    consumer=json.loads((ADMIT/"consumer_manifest.json").read_text())
    request=json.loads((ADMIT/"request.json").read_text())
    require(admitted["amendment_ref"]==ref(ADMIT/"amendment.root_active_v12.json"),"Admitted amendment")
    require(admitted["input_ref"]==amendment["fresh_continuation_input_ref"]==ref(ADMIT/"input.json"),"Admitted input")
    require(inp["consumer_manifest_ref"]==ref(ADMIT/"consumer_manifest.json"),"Input consumer manifest")
    require(inp["root_stage_request_ref"]==ref(ADMIT/"request.json"),"Input ROOT request")
    require(amendment["adapter_manifest_ref"]==inp["source_manifest_ref"]==consumer["source_manifest_ref"]==request["source_manifest_ref"]==ref(SRC/"source_manifest.v1.json"),"All current source identities agree")
    require(ref(OBS)["sha256"]=="4116a44f011705f0f45cea862179a0bc2cbc0b5acfad48ffc6c7e795c864ad54","Pinned prior observation")
    observed=json.loads(OBS.read_text())
    require(observed["timestamp_keys"]==["Timestamp"] and observed["saved_batch"]==156500 and observed["optimizer_parameter_state_count"]==77 and observed["saved_epoch"] is None and observed["full_scientific_training_complete"] is False,"Existing observation only; no epoch/completion inference")
    driver=(SRC/"execution_only_v1.py").read_text()
    bridge=(SRC/"resume_scientific_bridge_v5.py").read_text()
    native_path=Path("research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py")
    native=native_path.read_text()
    common=Path("research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/common_v1.py").read_text()
    top=ast.parse(common).body
    context=next(n for n in top if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=="CONTEXT" for v in n.targets))
    require(ast.unparse(context.value)=="HERE + '/native_child_v1.py'","Original context resolves to exact native helper")
    worker=ast.get_source_segment(driver,function(driver,"worker_v1"))
    run=ast.get_source_segment(bridge,function(bridge,"run_resumed_native_v5"))
    require("native_VAL000_full_state_resume_source_20261016_v12 import resume_scientific_bridge_v5" in worker and "resume_scientific_bridge_v5.run_resumed_native_v5(input_ref,s,q,cfg,ticket,entry_guard)" in worker,"Actual v12 worker route")
    require("_final_checkpoint=checkpoint_metadata_bridge_v1(native)" in run and "types.FunctionType(native.run_v1.__code__,ns" in run,"Original native code bound to adapted final helper")
    require("finally:session.close()" in run,"Original resume teardown")
    native_run=ast.get_source_segment(native,function(native,"run_v1"))
    require('observed_trainer["epoch"]==5 and observed_trainer["batch"]==227990' in native_run and 'observed_optimizer["unique_step_values"]==[227990]' in native_run,"Original genuine trainer budget")
    require('final=_final_checkpoint(directory/"checkpoints",epochs=5,updates=227990,' in native_run and '_save_json(directory/"completed.json",result);return 0' in native_run,"Final helper precedes genuine native completion and zero return")
    require(run.index('c.require(session.restore_receipt is not None')<run.index('original_save(path,value)'),"Genuine restore precedes completion save")
    execute=ast.get_source_segment(driver,function(driver,"execute_v1"))
    require('drained=raw_status is not None and roster_complete and identity_absence is True and pipe_eof' in execute,"Genuine wait and full ownership drain")
    require('if not drained or raw_status is None or not evidence_ready:' in execute and execute.index('if not drained or raw_status is None or not evidence_ready:')<execute.index('actual=read(completed_ref)'),"Unresolved drain blocks terminal acceptance")
    require('if rc is not None and rc!=0:failure=failure or "Actual worker exit "+str(rc)' in execute and 'failure=failure or "Strict terminal evidence failed: "' in execute and 'returncode=0 if failure is None else' in execute,"Original nonzero/parser failures preserved")
    require('actual.get("completed_epochs")==5 and actual.get("actual_optimizer_updates")==227990' in execute and 'actual.get("observed_counters_verified") is True' in execute,"Supervisor exact completion predicates")
    state_path=next(Path(r["path"]) for r in source_refs if r["path"].endswith("/composer/core/state.py"))
    time_path=next(Path(r["path"]) for r in source_refs if r["path"].endswith("/composer/core/time.py"))
    state_tree=ast.parse(state_path.read_text())
    attrs=next(n for n in state_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_STATE_DICT_SERIALIZED_ATTRIBUTES" for t in n.targets))
    require("timestamp" in ast.literal_eval(attrs.value),"Installed State serializes timestamp by class-qualified key")
    state_class=next(n for n in state_tree.body if isinstance(n,ast.ClassDef) and n.name=="State")
    state_serial=ast.get_source_segment(state_path.read_text(),next(n for n in state_class.body if isinstance(n,ast.FunctionDef) and n.name=="state_dict"))
    require("serialized_value = {type(obj).__qualname__: obj.state_dict() for obj in ensure_tuple(attribute_value)}" in state_serial,"Installed State wraps Timestamp state_dict")
    time_class=next(n for n in ast.parse(time_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="Timestamp")
    time_serial=next(n for n in time_class.body if isinstance(n,ast.FunctionDef) and n.name=="state_dict")
    serializer_keys=set(ast.literal_eval(k) for k in time_serial.body[0].value.keys)
    counters={"iteration","epoch","batch","sample","token","epoch_in_iteration","token_in_iteration","batch_in_epoch","sample_in_epoch","token_in_epoch"}
    walls={"total_wct","iteration_wct","epoch_wct","batch_wct"}
    require(serializer_keys==counters|walls,"Installed serializer exactly matches parser keys")
    native_ref=ref(native_path);bridge_ref=ref(SRC/"resume_scientific_bridge_v5.py")
    raw_calls=[]
    def raw_source(r):
        require(r==native_ref,"Isolated bridge may read only pinned native source")
        raw_calls.append(r)
        return native_path.read_bytes()
    fake_c=SimpleNamespace(HERE=str(SRC),require=require,original=SimpleNamespace(CONTEXT=str(native_path),ref=lambda p:native_ref),
        raw=raw_source,local=lambda p:Path(p),ref=lambda p:bridge_ref)
    ns=dict(c=fake_c,timedelta=timedelta,ast=ast,copy=copy,SOURCE=str(SRC/"resume_scientific_bridge_v5.py"))
    isolated([function(bridge,"validated_composer_timestamp_v1"),function(bridge,"checkpoint_metadata_bridge_v1")],ns,"isolated_metadata_functions")
    records={};paths=[];alias=None
    class FakePath:
        def __init__(self,name,size=1):self.name=name;self.size=size
        def lstat(self):return SimpleNamespace(st_size=self.size)
        def __str__(self):return self.name
    def entries(unused):return paths,alias
    original_ns=dict(_checkpoint_entries=entries,Path=Path)
    isolated([function(native,"_int_time"),function(native,"optimizer_counter_evidence"),function(native,"_final_checkpoint")],original_ns,"isolated_original_helper")
    wrapped=ns["checkpoint_metadata_bridge_v1"](SimpleNamespace(_final_checkpoint=original_ns["_final_checkpoint"]))
    require(raw_calls==[native_ref],"Actual adapter authenticates only exact native source")
    original_final=function(native,"_final_checkpoint")
    expected=copy.deepcopy(original_final)
    timestamp_assign=[n for n in ast.walk(expected) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="timestamp" for t in n.targets)]
    require(len(timestamp_assign)==1,"One original timestamp assignment")
    timestamp_assign[0].value=ast.Call(func=ast.Name(id="_actual_native_nested_timestamp_v1",ctx=ast.Load()),args=[ast.Name(id="state",ctx=ast.Load())],keywords=[])
    ens=dict(original_ns,_actual_native_nested_timestamp_v1=ns["validated_composer_timestamp_v1"])
    isolated([expected],ens,str(SRC/"resume_scientific_bridge_v5.py")+":nested_timestamp_metadata_bridge")
    adapted=next(x.cell_contents for x in wrapped.__closure__ if callable(x.cell_contents) and getattr(x.cell_contents,"__name__","")=="_final_checkpoint")
    require(adapted.__code__.co_code==ens["_final_checkpoint"].__code__.co_code and adapted.__code__.co_consts==ens["_final_checkpoint"].__code__.co_consts and adapted.__code__.co_names==ens["_final_checkpoint"].__code__.co_names,"Actual adapted helper preserves entire original body except timestamp expression")
    def payload():
        stamp={k:0 for k in counters}
        stamp.update(epoch=5,batch=227990)
        stamp.update({k:timedelta(seconds=1) for k in walls})
        optim={i:dict(step=227990,exp_avg="handwritten-presence-marker",exp_avg_sq="handwritten-presence-marker") for i in range(77)}
        return dict(state=dict(timestamp={"Timestamp":stamp},model={"handwritten_presence":True},optimizers={"DecoupledAdamW":{"state":optim}}))
    fake_torch=SimpleNamespace(load=lambda path,**kw:copy.deepcopy(records[str(path)]))
    def reference(path):return dict(path=str(path),sha256="a"*64 if str(path)!="distinct.pt" else "b"*64)
    def invoke(value=None,names=None,alias_value=None,max_files=2,max_bytes=100):
        nonlocal paths,alias,records
        paths=[FakePath(n) for n in (names or ["final.pt"])]
        records={str(p):copy.deepcopy(value if value is not None else payload()) for p in paths}
        alias=alias_value
        return wrapped("handwritten_directory",epochs=5,updates=227990,max_files=max_files,max_bytes=max_bytes,torch=fake_torch,reference=reference)
    positive=invoke()
    require(positive["completed_epochs"]==5 and positive["actual_optimizer_updates"]==227990 and positive["checkpoint_optimizer_counters"]["parameter_state_count"]==77 and positive["actual_checkpoint_timestamp_schema"]=="state.timestamp.Timestamp","Nested final fixture accepted with original full counters")
    rejected=[]
    def reject(label,change=None,**kw):
        val=payload()
        if change:change(val)
        try:invoke(val,**kw)
        except (ValueError,TypeError,KeyError,AttributeError) as e:
            rejected.append(dict(case=label,error=str(e)));return
        raise AssertionError("Unexpected acceptance "+label)
    reject("flat_timestamp",lambda v:v["state"].update(timestamp=v["state"]["timestamp"]["Timestamp"]))
    reject("mixed_outer_timestamp",lambda v:v["state"]["timestamp"].update(epoch=5))
    reject("missing_inner_key",lambda v:v["state"]["timestamp"]["Timestamp"].pop("token_in_epoch"))
    reject("extra_inner_key",lambda v:v["state"]["timestamp"]["Timestamp"].update(extra=0))
    reject("boolean_counter",lambda v:v["state"]["timestamp"]["Timestamp"].update(epoch=True))
    reject("negative_counter",lambda v:v["state"]["timestamp"]["Timestamp"].update(token=-1))
    reject("float_counter",lambda v:v["state"]["timestamp"]["Timestamp"].update(batch=227990.0))
    reject("non_timedelta_wall",lambda v:v["state"]["timestamp"]["Timestamp"].update(epoch_wct=1))
    reject("negative_wall",lambda v:v["state"]["timestamp"]["Timestamp"].update(batch_wct=timedelta(seconds=-1)))
    reject("partial_epoch",lambda v:v["state"]["timestamp"]["Timestamp"].update(epoch=4))
    reject("partial_updates",lambda v:v["state"]["timestamp"]["Timestamp"].update(batch=227989))
    reject("wrong_optimizer_step",lambda v:v["state"]["optimizers"]["DecoupledAdamW"]["state"][0].update(step=227989))
    reject("missing_optimizer_moment",lambda v:v["state"]["optimizers"]["DecoupledAdamW"]["state"][0].pop("exp_avg_sq"))
    reject("empty_model",lambda v:v["state"].update(model={}))
    reject("ambiguous_final_files",names=["final.pt","distinct.pt"])
    reject("wrong_latest_alias",alias_value={"target_text":"other.pt"})
    reject("file_cap",max_bytes=0)
    reject("file_count_cap",names=["final.pt","same.pt","third.pt"])
    require(not any(x in sys.modules for x in ("torch","composer","numpy","torch_geometric")),"No numerical imports")
    return dict(schema="native_ARGformer_final_nested_schema_readiness_checks_v1",verdict="SCHEMA_READY_NO_BRIDGE_REQUIRED",
        source_only=True,authenticated_relevant_source_count=len(source_refs),source_refs=source_refs,
        manifest_ref=ref(SRC/"source_manifest.v1.json"),
        admitted_metadata_refs=[ref(ADMIT/n) for n in ("admitted.json","amendment.root_active_v12.json","input.json","consumer_manifest.json","request.json")],
        prior_observation_ref=ref(OBS),actual_v12_route_to_nested_parser_proven=True,installed_Composer_serializer_matches_parser=True,
        original_helper_only_timestamp_expression_adapted=True,positive_fixture=dict(completed_epochs=positive["completed_epochs"],actual_optimizer_updates=positive["actual_optimizer_updates"],parameter_state_count=positive["checkpoint_optimizer_counters"]["parameter_state_count"],schema=positive["actual_checkpoint_timestamp_schema"]),
        negative_cases=rejected,original_trainer_and_supervisor_full_budget_checks_preserved=True,original_genuine_wait_drain_and_failure_paths_preserved=True,
        actual_checkpoint_or_payload_reads=False,actual_entrypoints_or_guards_or_state_or_process_resource_probes=False,numerical_imports=False,
        new_acceptance_or_producer_exit_or_completion_fabricated=False,live_files_or_running_code_modified=False,
        limitation="This narrow source proof establishes timestamp-schema readiness, not actual completion or final wait/outer acceptance. It does not infer an epoch from the 156500 observation. Original final helper reports optimizer parameter-state count; it does not explicitly compare that count to 77, while resume restore requires 77.")

if __name__=="__main__":
    try: print(json.dumps(main(),indent=2,sort_keys=True))
    except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)

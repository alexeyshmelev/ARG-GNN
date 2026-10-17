"""Actual extracted B001/D001 RAM regressions. No production imports."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,json,hashlib,io,contextlib
HERE=Path(__file__).parent;ROOT=HERE.parents[2]
def need(ok,message="RAM assertion"):
    if not ok:raise ValueError(message)
def ref(name):return dict(path="RAM/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
def functions(path,names,ns):
    nodes=[copy.deepcopy(n)for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
    need({n.name for n in nodes}==set(names),"Exact extracted actual source")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<actual B001/D001 RAM definitions>","exec"),ns);return ns
def rejected(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return True
    return False
def run_v2():
    names=["bundle","completed","wait","outer","registry","input","foreign_input","request","manifest","counts","prep","cap","review","wrong_manifest"]
    r={n:ref(n)for n in names}
    records={
    "bundle":dict(root_authorizer="root_agent",completed_ref=r["completed"],actual_wait_ref=r["wait"],pool_outer_consumed_ref=r["outer"]),
    "completed":dict(parent_id="american_c3_val_001",preparation_ref=r["prep"],original_parent_cardinality_ref=r["counts"],completed_epochs=5,actual_optimizer_updates=10,initializer_checkpoint_ref=None,resume_checkpoint_ref=None,query_ancestry_truth_opened=False,fresh_optimizer_evidence=dict(optimizer_state_empty=True),active_input_ref=r["input"]),
    "wait":dict(input_ref=r["input"],returncode=0,actual_OS_wait_status=0,actual_direct_worker_wait=True,complete_owned_roster_known=True,all_owned_processes_and_groups_drained=True,operational_failure=None,completed_ref=r["completed"],actual_authority_seconds_remaining=200,adapter_manifest_ref=r["manifest"]),
    "outer":dict(immutable_pool_registry_snapshot_ref=r["registry"],root_authorizer="root_agent",actual_pool_logical_outer_OS_wait_consumed=True,actual_exit_code=0,job_id="N1",actual_wait_ref=r["wait"],pool_capability_ref=r["cap"],original_standalone_or_historical_wait_relabelled=False,actual_terminal_consumed=True),
    "input":dict(job_id="N1",root_stage_request_ref=r["request"],source_sha256={}),
    "request":dict(parent_id="american_c3_val_001",preparation_ref=r["prep"],counts_ref=r["counts"],adapter_manifest_ref=r["manifest"],postdrain_pool_capability_ref=r["cap"],source_review_ref=r["review"]),
    "registry":dict(capability_ref=r["cap"],jobs=[dict(logical_job="N1",state="succeeded",exit_code=0,leader_wait_status_known=True,owned_drain_observed=True)]),
    "review":dict(independent=True,verdict="SOURCE_PASS",source_manifest_ref=r["manifest"])}
    def read(record):return copy.deepcopy(records[record["path"][4:]])
    c=SimpleNamespace(read=read,require=need,science_v1=lambda _: (dict(scientific_protocol=dict(optimizer_updates=10)),None,None,None),source_refs_v1=lambda:[],ref=lambda _:r["manifest"],MANIFEST="RAMmanifest")
    old=functions(ROOT/"research/reviews/native_ARGformer_postdrain_pool_source_20261017_v1/benchmark_barrier_v1.py",["completed_native_fit_bundle_v1"],dict(c=c))["completed_native_fit_bundle_v1"]
    new=functions(HERE/"benchmark_barrier_v1.py",["completed_native_fit_bundle_v1"],dict(c=c))["completed_native_fit_bundle_v1"]
    need(new(r["bundle"])["expected_optimizer_updates"]==10,"True fresh source-bound baseline")
    fresh=[]
    for name,key,value in [("completed","active_input_ref",r["foreign_input"]),("wait","adapter_manifest_ref",r["wrong_manifest"]),("outer","actual_terminal_consumed",False)]:
        previous=records[name][key];records[name][key]=value
        need(bool(old(r["bundle"])),"Reproduce exact sealed-v1 omission")
        need(rejected(lambda:new(r["bundle"])),"Reject exact demonstrated B001 mutation")
        records[name][key]=previous;fresh.append(name+"."+key)
    records["completed"]["child_input_ref"]=r["foreign_input"]
    need(rejected(lambda:new(r["bundle"])),"No conflicting child input accepted for a fresh fit")
    del records["completed"]["child_input_ref"]
    need(bool(new(r["bundle"])),"Fresh positive remains accepted after independent negatives")
    # Genuine legacy v12 has a newer child input and preserves its older active
    # input/history plus non-null full-state resume. No checkpoint bytes are used.
    lr={n:ref("legacy_"+n)for n in ("completion","wait","outer","input","old_active","amendment","review","checkpoint","prep","author","resume")}
    legacy_manifest=dict(path="research/reviews/native_VAL000_full_state_resume_source_20261016_v12/source_manifest.v1.json",sha256="734c8f55fc6a06c20fba9b63b9d72bf733ee9e6b49261f8da26c8bb6b27ae664")
    child=dict(path="research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py",sha256="f2770222f02979b6064a7f1ffea80953cd234114c96ddde0845372479292379e")
    roster=ref("legacy_roster");parent="american_c3_val_000"
    data={
    lr["completion"]["path"]:dict(parent_id=parent,stage="mlm",completed_epochs=5,actual_optimizer_updates=227990,observed_counters_verified=True,
        trainer_optimizer_counters=dict(unique_step_values=[227990]),checkpoint_optimizer_counters=dict(unique_step_values=[227990]),
        initializer_checkpoint_ref=None,checkpoint_ref=lr["checkpoint"],preparation_ref=lr["prep"],author_source_manifest_ref=lr["author"],query_ancestry_truth_opened=False,
        child_input_ref=lr["input"],active_input_ref=lr["old_active"],actual_execution_amendment_ref=lr["amendment"],adapter_manifest_ref=legacy_manifest,
        actual_original_full_state_resume=True,original_fresh_run_initializer_checkpoint_ref=None,resume_checkpoint_ref=lr["resume"],
        full_state_resume_input=dict(checkpoint_ref=lr["resume"],preserved_optimizer_updates=22000),actual_Composer_restore=dict(full_state_restored=True),original_history_preserved=True),
    lr["wait"]["path"]:dict(returncode=0,actual_OS_wait_status=0,actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,complete_owned_roster_known=True,
        completed_ref=lr["completion"],operational_failure=None,group_observations=[dict(absent=True)],input_ref=lr["input"],owned_roster_policy_ref=roster,
        actual_execution_amendment_ref=lr["amendment"],adapter_manifest_ref=legacy_manifest),
    lr["outer"]["path"]:dict(actual_exit_code=0,actual_terminal_consumed=True,actual_direct_worker_returncode=0,all_owned_processes_and_groups_drained=True,
        actual_wait_ref=lr["wait"],actual_tool_session_id=70445),
    lr["amendment"]["path"]:dict(root_authorizer="root_agent",independent_source_review_passed=True,adapter_manifest_ref=legacy_manifest,
        independent_source_review_ref=lr["review"],resume_checkpoint_ref=lr["resume"]),
    legacy_manifest["path"]:dict(owned_roster_policy_ref=roster,original_scientific_child_ref=child),
    lr["review"]["path"]:dict(RAM_only=True)}
    lc=SimpleNamespace(read=lambda v:copy.deepcopy(data[v["path"]]),require=need,PARENT=parent,OWNED_ROSTER_POLICY_REF=roster,
        raw=lambda v:json.dumps(data[v["path"]]).encode(),ref=lambda _:r["manifest"],MANIFEST="RAM_new_manifest")
    legacy=functions(HERE/"downstream_context_v1.py",["_bundle_v1"],dict(c=lc,pool_prerequisite_bundle_v1=lambda *a:(_ for _ in()).throw(ValueError("No cross-branch fallback"))))["_bundle_v1"]
    bundle=dict(completion_ref=lr["completion"],wait_ref=lr["wait"],outer_terminal_ref=lr["outer"])
    baseline=legacy(bundle,"mlm")
    need(baseline[0]["active_input_ref"]!=baseline[1]["input_ref"] and baseline[0]["resume_checkpoint_ref"]is not None,"Legacy child takes precedence; full state/history kept")
    stage_ns=dict(require=need,REFS=dict(preparation=lr["prep"],author_manifest=lr["author"]))
    strict=functions(ROOT/"research/reviews/native_VAL000_downstream_scientific_stages_source_20261016_v1/scientific_stages_v1.py",["full_mlm_metadata_prerequisite_v1"],stage_ns)["full_mlm_metadata_prerequisite_v1"]
    strict(*baseline,completion_ref=lr["completion"],wait_ref=lr["wait"],checkpoint_ref=lr["checkpoint"])
    legacy_negatives=[]
    for record,key,value in [(lr["completion"],"child_input_ref",lr["old_active"]),(lr["wait"],"adapter_manifest_ref",r["manifest"]),
        (lr["completion"],"actual_execution_amendment_ref",r["request"]),(lr["completion"],"actual_original_full_state_resume",False),
        (lr["completion"],"resume_checkpoint_ref",r["wrong_manifest"]),(lr["outer"],"actual_terminal_consumed",False)]:
        previous=data[record["path"]][key];data[record["path"]][key]=value
        need(rejected(lambda:legacy(bundle,"mlm")),"Reject invalid legacy provenance without fresh-manifest fallback")
        data[record["path"]][key]=previous;legacy_negatives.append(key)
    need(bool(legacy(bundle,"mlm")),"True full-state legacy baseline still accepted")
    diagnostics=[]
    for name,prefix,direct_name in (("native_root_driver_v1.py","native","native_direct_gpu0_v1.py"),("downstream_execution_v1.py","downstream","downstream_direct_gpu0_v1.py")):
        tree=ast.parse((HERE/name).read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="execute_v1")
        actual_try=next(n for n in execute.body if isinstance(n,ast.Try)and any(isinstance(h.type,ast.Name)and h.type.id=="BaseException"and any(isinstance(v,ast.If)and "capacity_sample"in ast.unparse(v.test)for v in h.body)for h in n.handlers))
        handler=next(h for h in actual_try.handlers if isinstance(h.type,ast.Name)and h.type.id=="BaseException")
        need(bool(actual_try.finalbody),"Original owned cleanup still follows handler")
        emitter=functions(HERE/name,["emit_capacity_failure_v3"],dict(json=json,require=need))["emit_capacity_failure_v3"]
        bounded=functions(HERE/direct_name,["bounded_capacity_failure_v3"],dict(json=json))["bounded_capacity_failure_v3"]
        sample=bounded(dict(index=0,uuid="RAM_GPU",total_mib=81920,used_mib=80000,compute=[dict(pid=i+1,used_mib=1,name="RAM"*300)for i in range(100)]),1.,dict(classification_complete=True),ValueError("RAM failure"*200))
        need(len(json.dumps(sample,sort_keys=True,ensure_ascii=True,allow_nan=False).encode())<=4096,"Existing failing sample bound")
        class CapacityFailure(ValueError):pass
        error=CapacityFailure("RAM actual failing capacity");error.capacity_sample=sample
        for emit_fails in (False,True):
            events=[];stdout=io.StringIO()
            def emit(value):
                events.append("bounded_sample")
                if emit_fails:raise OSError("RAM broken stdout")
                emitter(value)
            ns=dict(failure=None,failed_capacity_sample=None,pre_stop_capacity_emitted=False,capacity_emit_error=None,error=error,
                emit_capacity_failure_v3=emit,events=events,last_capacity=dict(RAM_previous_success=True))
            block=ast.Try(body=[ast.Raise(exc=ast.Name(id="error",ctx=ast.Load()),cause=None)],handlers=[copy.deepcopy(handler)],orelse=[],
                finalbody=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id="events",ctx=ast.Load()),attr="append",ctx=ast.Load()),args=[ast.Constant("owned_cleanup")],keywords=[]))])
            with contextlib.redirect_stdout(stdout):exec(compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),"<actual capacity exception RAM>","exec"),ns)
            need(ns["failed_capacity_sample"]==sample and events==["bounded_sample","owned_cleanup"],"Sample precedes cleanup even if emission fails")
            need(ns["pre_stop_capacity_emitted"]is(not emit_fails),"Truthful emission flag")
            need((ns["capacity_emit_error"]is not None)is emit_fails,"Bounded emission failure retained")
            if not emit_fails:need(json.loads(stdout.getvalue())[prefix+"_capacity_failure_before_owned_stop"]==sample,"Truthful original failing sample")
            obs=next(n.value for n in execute.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="observation"for t in n.targets))
            fields=[copy.deepcopy(k)for k in obs.keywords if k.arg.startswith("failing_GPU")]
            need(len(fields)==3,"All three diagnostic fields reach terminal observation")
            value=eval(compile(ast.fix_missing_locations(ast.Expression(body=ast.Call(func=ast.Name(id="dict",ctx=ast.Load()),args=[],keywords=fields))),"<actual diagnostic fields RAM>","eval"),ns)
            need(value["failing_GPU_capacity_sample"]==sample,"Failure sample retained in terminal metadata")
        diagnostics.append(name)
    return dict(schema="native_postdrain_B001_D001_RAM_regressions_v2",status="PASS_STATIC_RAM_ONLY",
        sealed_v1_all_three_counterexamples_reproduced=True,fresh_baseline_accepted=True,fresh_three_provenance_mutations_rejected=fresh,
        conflicting_fresh_child_input_rejected=True,legacy_v12_full_state_baseline_and_original_stage_prerequisite_accepted=True,
        legacy_original_active_history_and_nonnull_resume_preserved=True,legacy_provenance_negatives_rejected=legacy_negatives,
        capacity_diagnostic_supervisors=diagnostics,actual_bounded_sample_emitted_and_retained_before_cleanup=True,
        broken_stdout_still_enters_cleanup_and_records_error=True,production_imports=0,numeric_payload_checkpoint_truth_reads=0,
        real_guard_process_resource_runtime_admission_journal_signal_install_Git_operations=0)
if __name__=="__main__":
    result=run_v2();out=HERE/"provenance_diagnostic_checks.v2.json"
    with out.open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps(dict(status=result["status"],result_path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),sort_keys=True))

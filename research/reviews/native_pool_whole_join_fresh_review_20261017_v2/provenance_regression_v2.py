"""Exact source-only regression for the completed-native-fit provenance joins."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
PACKET=ROOT/"research/reviews/native_ARGformer_postdrain_pool_source_20261017_v2"
OUT=Path(__file__).parent
def need(ok,message):
 if not ok:raise ValueError(message)
def ref(name):return dict(path="RAM/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
r={n:ref(n)for n in ["bundle","completed","wait","outer","registry","input","foreign_input","request","manifest","counts","prep","cap","review","wrong_manifest"]}
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
source=PACKET/"benchmark_barrier_v1.py"
fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="completed_native_fit_bundle_v1")
ns=dict(c=c);exec(compile(ast.Module(body=[fn],type_ignores=[]),"<actual completed-native-fit source>","exec"),ns)
baseline=ns[fn.name](r["bundle"]);cases=[]
for name,key,value in [("completed","active_input_ref",r["foreign_input"]),("wait","adapter_manifest_ref",r["wrong_manifest"]),("outer","actual_terminal_consumed",False)]:
 previous=records[name][key];records[name][key]=value
 try:actual=ns[fn.name](r["bundle"]);accepted=True;error=None
 except Exception as e:actual=None;accepted=False;error=repr(e)
 records[name][key]=previous
 cases.append(dict(mutation=name+"."+key,mutated_value=value,unexpectedly_accepted=accepted,result=actual,error=error))
need(all(not x["unexpectedly_accepted"]for x in cases),"V2 rejects all three demonstrated omissions")
result=dict(schema="independent_exact_fit_bundle_provenance_rejections_v2",source_function="benchmark_barrier_v1.completed_native_fit_bundle_v1",source_ref=dict(path=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),synthetic_baseline_accepted=bool(baseline),counterexamples=cases,production_imports=0,real_resource_or_runtime_operations=False,all_references_are_in_RAM_fixture=True)
target=OUT/"native_fit_bundle_provenance_rejections.v2.json";target.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(source_blocker_reproduced=False,unexpected_acceptances=sum(x["unexpectedly_accepted"]for x in cases),result_path=str(target)),sort_keys=True))

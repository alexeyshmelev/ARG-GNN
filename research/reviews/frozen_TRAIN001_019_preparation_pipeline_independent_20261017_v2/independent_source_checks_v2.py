"""Independent source/AST/RAM review of exact sealed CPU pipeline v2. No production imports in this checker."""
from pathlib import Path
from types import SimpleNamespace,FunctionType,CodeType
from datetime import datetime,timezone,timedelta
from contextlib import contextmanager,redirect_stdout
import ast,argparse,copy,hashlib,io,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");REL="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2"
W=REL+"/worker_entry_v1.py";B=REL+"/batch_ROOT_v1.py";C=REL+"/close_unit_ROOT_v1.py";M=REL+"/source_manifest.v1.json"
OLD="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3";OBS="research/reviews/frozen_TRAIN001_observe_ready_20261017_v2"
checks=[]
def need(ok,msg="Independent RAM condition"):
 if not ok:raise ValueError(msg)
def yes(n,ok=True):need(ok,n);checks.append(n)
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def doc(p):return json.loads((ROOT/p).read_text())
def fn(path,name,e):
 text=(ROOT/path).read_text();n=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 f=FunctionType(next(c for c in compile(text,path,"exec").co_consts if isinstance(c,CodeType)and c.co_name==name),e,name,tuple(ast.literal_eval(x)for x in n.args.defaults))
 f.__kwdefaults__={k.arg:ast.literal_eval(v)for k,v in zip(n.args.kwonlyargs,n.args.kw_defaults)if v is not None};return f
def astfn(path,name):
 return ast.dump(next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name),include_attributes=False)
m=doc(M);h=doc(REL+"/HANDOFF.v2.json")
yes("sealed_handoff_exact",ref(REL+"/HANDOFF.v2.json")["sha256"]=="9f105e9d5e3581532241ab61f694fc205c028cd21d94f06c9e214082535ec5b4")
yes("sealed_manifest_exact",ref(M)["sha256"]=="2c9a08835aa60dafa0a25ccf51ac1933655797e89f9e1aaa49a25ff97d8d9e8b")
for rr in m["refs"]:need(ref(rr["path"])==rr)
yes("all19_concrete_refs_authenticated",len(m["refs"])==19)
for name in("original_owned","available_original_slots","available_memory","terminal","argv","group_call","state","guard"):
 yes("same_reviewed_v3_"+name,astfn(B,name)==astfn(OLD+"/batch_ROOT_v1.py",name))
current_group=astfn(B,"run_group");old_group=astfn(OLD+"/batch_ROOT_v1.py","run_group")
current_assignment=ast.dump(ast.parse("end_work=deadline_epoch-180").body[0],include_attributes=False)
old_assignment=ast.dump(ast.parse("end_work=min(deadline_epoch,time.time()+10800)-180").body[0],include_attributes=False)
yes("aggregate_monitor_only_stage_specific_outer_deadline_delta",current_group.replace(current_assignment,old_assignment)==old_group)
yes("finite_group_call_deadline_inherited",astfn(B,"group_call")==astfn(OLD+"/batch_ROOT_v1.py","group_call"))
yes("unchanged_original_writer_wrapper",astfn(W,"_OriginalWriterCurrentGuard")==astfn(OBS+"/worker_entry_v1.py","_OriginalWriterCurrentGuard"))
condition=m["condition"];oldcond=doc(OBS+"/source_manifest.v1.json")["condition"]
yes("original_inference_template_runtime_caps_unchanged",condition["stage_templates"]["infer"]==oldcond["inference_template_inactive"]and condition["stage_runtime_refs"]["infer"]==oldcond["inference_runtime_ref"]and condition["stage_caps"]["infer"]==oldcond["caps_proposed"])
yes("original_native_template_provenance_unchanged",condition["stage_templates"]["native_paths"]==dict(doc(condition["original_native_config_provenance_ref"]["path"]),execution_enabled=False,observation_ref=None))
yes("original_registered_seed_registry_unchanged",condition["registry"]==doc(condition["registry_provenance_ref"]["path"]))
yes("unchanged_shared_two_slot_pool",m["synchronization_pool"]==doc(OLD+"/source_manifest.v1.json")["synchronization_pool"])
# Exact queue wrappers: acquisition-only timeout queues; identical error after yield is fatal.
for path,name,core in((W,"admit","admit_under_coordinator"),(C,"close_once","close_under_coordinator")):
 for case in("before_yield","after_yield","other_error","fresh","aged"):
  events=[];now=datetime.now(timezone.utc);q=dict(batch_control_input_ref={},created_at_utc=(now-timedelta(seconds=300 if case=="aged"else 1)).isoformat())
  @contextmanager
  def coordinator(rr):
   events.append("enter")
   if case=="before_yield":raise ValueError("Finite shared ROOT accounting coordinator wait")
   if case=="other_error":raise ValueError("Different source/accounting error")
   yield
   events.append("exit")
  def operation(rr):
   events.append("operation")
   if case=="after_yield":raise ValueError("Finite shared ROOT accounting coordinator wait")
   return dict(status="CORE_RAN")
  env=dict(read=lambda rr:q,accounting_transaction=coordinator,support=lambda:None,datetime=datetime,timezone=timezone,**{core:operation})
  if case in("after_yield","other_error"):
   try:fn(path,name,env)({})
   except ValueError:yes(name+"_no_replay_after_yield_or_other_error_"+case)
   else:raise AssertionError("Unsafe queue reclassification")
  else:
   result=fn(path,name,env)({})
   if case=="before_yield"or(case=="aged"and path==W):yes(name+"_queued_before_actual_operation_"+case,result["status"]=="QUEUED_BEFORE_OPERATION"and result["accounting_operation_started"]is False and "operation"not in events)
   else:yes(name+"_actual_core_once_"+case,result["status"]=="CORE_RAN"and events.count("operation")==1)
# Native focal identity helper is exact AST extracted and the original science receives original cfg.
src=(ROOT/W).read_text();bridge=ast.unparse(next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)and n.name=="execute_in_admitted_context_v1"))
yes("original_science_execute_and_writer_no_replacement","science.execute(cfg, writer)"in bridge and "american_c3_observe_v3 as science"in bridge and "american_c3_paths_v1 as science"in bridge)
yes("focal_identity_exact_original_helper_and_all10000_fields",'n.name=="identity_sha"'in src and 'v.id=="IDENTITY"'in src and 'namespace["identity_sha"](records)'in src and 'len(records["record_ids"])==10000'in src and 'publish_focal_seal(read(result),writer)'in src)
# Every class/source is explicit and all default CLIs inert without initializers.
for path in(W,B,C):
 env=dict(argparse=argparse,json=json,__doc__="RAM")
 with redirect_stdout(io.StringIO())as stream:r=fn(path,"main",env)([])
 yes("default_inactive_"+Path(path).name,r==0 and json.loads(stream.getvalue())["status"]=="SOURCE_ONLY_INACTIVE")

# Exact v1->v2 source delta: namespaces plus complete predecessor source-closure join only.
V1="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v1"
for path in("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py"):
 oldtree=ast.parse((ROOT/V1/path).read_text());newtree=ast.parse((ROOT/REL/path).read_text().replace(REL,V1))
 if path=="worker_entry_v1.py":
  oldbody=[n for n in oldtree.body if not(isinstance(n,ast.FunctionDef)and n.name=="predecessor_capsule")]
  newbody=[n for n in newtree.body if not(isinstance(n,ast.FunctionDef)and n.name=="predecessor_capsule")]
  yes("all_worker_AST_outside_predecessor_unchanged_after_namespace",ast.dump(ast.Module(body=oldbody,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=newbody,type_ignores=[]),include_attributes=False))
  op=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name=="predecessor_capsule")
  npred=next(n for n in newtree.body if isinstance(n,ast.FunctionDef)and n.name=="predecessor_capsule")
  oldindex=next(i for i,n in enumerate(op.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="predecessor_request"for t in n.targets))
  newindex=next(i for i,n in enumerate(npred.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="predecessor_request"for t in n.targets))
  dump=lambda ns:ast.dump(ast.Module(body=ns,type_ignores=[]),include_attributes=False)
  yes("predecessor_prefix_every_original_metadata_role_unchanged",dump(op.body[:oldindex+1])==dump(npred.body[:newindex+1]))
  yes("predecessor_every_postclosure_science_wait_drain_roster_requirement_unchanged",dump(op.body[oldindex+3:])==dump(npred.body[newindex+5:]))
  exact=r"""previous="simulate"if STAGE=="infer"else"infer"
expected_manifest=condition["original_simulation_batch_manifest_ref"]if previous=="simulate"else condition["original_inference_manifest_ref"]if PARENT=="american_c3_train_001"else metadata_ref(MANIFEST)
expected_sources={r["path"]:r["sha256"]for r in read(expected_manifest)["refs"]}
need(inp["source_sha256"]==predecessor_request["source_sha256"]==expected_sources and predecessor_request["source_manifest_ref"]==expected_manifest,"Exact complete immutable original predecessor manifest/source closure")
"""
  yes("new_predecessor_join_exact_pinned_three_route_manifests_complete_maps",dump(npred.body[newindex+1:newindex+5])==dump(ast.parse(exact).body))
 else:yes("full_"+path+"_AST_unchanged_after_namespace",ast.dump(oldtree,include_attributes=False)==ast.dump(newtree,include_attributes=False))
yes("initializer_condition_pool_exact_top_level_original_pool",condition["synchronization_pool"]==m["synchronization_pool"])
oldcondition=doc(V1+"/source_manifest.v1.json")["condition"]
yes("every_original_condition_unchanged_except_exact_pool_addition",{k:v for k,v in condition.items()if k!="synchronization_pool"}==oldcondition)
# Capacity freshness is exact controller/wrapper composition: one clock, queued if request >240s before operation.
for parent,stage in(("american_c3_train_002","infer"),("american_c3_train_002","native_paths"),("american_c3_train_001","native_paths")):
 now=datetime.now(timezone.utc);prior=dict(path="RAM/capsule",sha256="0"*64)
 context=SimpleNamespace(manifest=m,authority=dict(expires_at_utc=(now+timedelta(hours=40)).isoformat(),source_review_ref={},CPU_lanes=[list(range(8)),list(range(8,16))],scope_id="RAM"),authority_ref={},control_input_ref={},
  w=SimpleNamespace(scientific_config=lambda p,s,r:dict(parent=p,stage=s,observation_ref=r)))
 env=dict(datetime=datetime,timezone=timezone,timedelta=timedelta,need=need,read=lambda rr:dict(refs=dict(observation_ref={})),ref=ref,MANIFEST=M,REL=REL)
 q=fn(B,"unit_request",env)(context,dict(parent_id=parent,stage=stage,unit_key=parent+stage,prior_capsule_ref=prior),0,2**50,2**50)
 yes("exact_controller_request_capacity_clock_join_"+parent+stage,q["created_at_utc"]==q["actual_capacity"]["created_at_utc"]and q["expires_at_utc"]==q["actual_capacity"]["expires_at_utc"])
 yes("exact_controller_observed_capacity_values_join_"+parent+stage,q["actual_capacity"]["admissible_new_bytes"]==2**50 and q["actual_capacity"]["available_owned_RSS_bytes"]==2**50)
print(json.dumps(dict(schema="CPU_preparation_pipeline_independent_source_checks_v2",status="SOURCE_PASS",checks_passed=len(checks),checks=checks,
source_manifest_ref=ref(M),reviewed_sources=h["executed_sources"],actual_guard_imports_configurations_resource_process_GPU_probes_admissions_launches_closes=0,scientific_payload_truth_checkpoint_reads_or_hashes=0),sort_keys=True,indent=2))

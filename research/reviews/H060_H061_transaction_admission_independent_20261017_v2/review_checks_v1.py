"""Independent whole v2 source review: no module entries, guard, arrays or resources."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,runpy,sys,traceback,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H060_H061_transaction_admission_source_20261017_v2"
OLD="research/reviews/H060_H061_transaction_admission_source_20261017_v1"
OWN="research/reviews/H060_H061_transaction_admission_independent_20261017_v2"
def need(ok,message):
 if not ok:raise ValueError(message)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):
 need(ref(record["path"])==record,"exact source/metadata SHA")
 return(ROOT/record["path"]).read_bytes()
def read(record):return json.loads(raw(record))
def tree(path):return ast.parse((ROOT/path).read_bytes())
def fn(t,name):return copy.deepcopy(next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name))
def iso(nodes,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),"<independent-isolated-AST>","exec"),scope)
 return scope
def reject(action):
 try:action()
 except(ValueError,KeyError,TypeError):return
 raise AssertionError("expected rejection")
def main():
 base=ROOT/HERE
 expected={"root_driver_v1.py":"8889977254c90c33c8498a5e40abb4887920813a828b5ce27f5e1c74b81fc1b1","source_manifest.v1.json":"a17a17a097aaf7e5099b40cc9ed27dc9236fb1a0f335605a328a7af493b2d879","source_handoff.v1.json":"6d24d61b86c1cd55ff63da7e915cd0f9080db3ccf20a4dc8f9e26b24c7e8f2d2","audit_manifest.v1.json":"f212e5d5531a4b8ae069b553e1d80ee031e35d4b182d81c42ca5ec228c6afaa0"}
 for name,sha in expected.items():need(ref(HERE+"/"+name)["sha256"]==sha,"supplied seal "+name)
 manifest=read(ref(HERE+"/source_manifest.v1.json"));audit=read(ref(HERE+"/audit_manifest.v1.json"))
 for mapping in(manifest["source_sha256"],audit["files"]):
  for path,sha in mapping.items():
   need(Path(path).suffix in(".json",".py",".md",".patch"),"source/control metadata only")
   data=raw(dict(path=path,sha256=sha))
   if path.endswith(".py"):compile(data,path,"exec")
 need(len(manifest["source_sha256"])==330 and len(audit["files"])==12,"entire seal")
 b=read(ref(HERE+"/source_bindings.v1.json"));oldb=read(ref(OLD+"/source_bindings.v1.json"))
 current=tree(HERE+"/root_driver_v1.py");old=tree(OLD+"/root_driver_v1.py")
 norm=copy.deepcopy(current)
 norm.body=[n for n in norm.body if not(isinstance(n,ast.FunctionDef)and n.name=="_install_H061_prior_freeze_join")]
 for n in norm.body:
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="HERE"for t in n.targets):n.value.value=OLD
 cnode=next(n for n in norm.body if isinstance(n,ast.FunctionDef)and n.name=="contract")
 cnode.body=[n for n in cnode.body if not(isinstance(n,ast.If)and len(n.body)==1 and isinstance(n.body[0],ast.Expr)and isinstance(n.body[0].value,ast.Call)and ast.unparse(n.body[0].value.func)=="_install_H061_prior_freeze_join")]
 validator=next(n for n in cnode.body if isinstance(n,ast.FunctionDef)and n.name=="validate_request")
 validator.body=[n for n in validator.body if not(isinstance(n,ast.If)and ast.unparse(n.test)=="consumer == 'H061'")]
 need(ast.dump(norm,include_attributes=False)==ast.dump(old,include_attributes=False),"independent exact common AST delta")
 a,bcopy=copy.deepcopy(oldb),copy.deepcopy(b)
 for x in(a,bcopy):
  for key in("schema","transaction_memo","prior_integration_component","H061_prior_freeze"):x.pop(key,None)
 need(a==bcopy,"all original consumer/caps/science/namespace bindings unchanged")
 for name in("worker_entry_v1.py","scientific_config.INACTIVE.template.v1.json","storage_profiles.v1.json","handwritten_role_fixture.v1.json"):
  need((base/name).read_bytes()==(ROOT/OLD/name).read_bytes(),"exact worker/asset")
 memo=b["transaction_memo"];mr=read(memo["independent_review_ref"])
 need(memo["source_ref"]["sha256"]=="487fcd93b73f6668126522b67720041abbbaa8b95ee54c3bc69adffba1a765b2"and memo["independent_review_ref"]["sha256"]=="5f57cd4b46c3575901f660f47ab1c556aabdb2f30b8c15d7ad339b2cf0a29978","actual memo2/review")
 need(mr["verdict"]=="SOURCE_PASS"and mr["independent"]is True and not mr["blockers"]and mr["source_ref"]==memo["source_ref"]and mr["source_manifest_ref"]==memo["source_manifest_ref"],"separate memo2 pass")
 for record in memo.values():need(manifest["source_sha256"][record["path"]]==record["sha256"],"memo/review transitive closure")
 prior=b["H061_prior_freeze"]
 need(prior["acceptance_ref"]["sha256"]=="70c8c47468b69cc0b6977f1e80cb898cb7b773b0096de881188e61c5613e2146"and prior["source_ref"]==ref(OLD+"/root_driver_v1.py")and prior["source_manifest_ref"]==ref(OLD+"/source_manifest.v1.json"),"exact actual freeze identities")
 # Actual original acceptance body with independently built metadata fixtures.
 contract=tree(b["consumers"]["H061"]["contract_ref"]["path"]);views={};auth=[]
 def reader(record):return copy.deepcopy(views[record["path"]])if record["path"]in views else read(record)
 scope=dict(read=reader,require=need,ref=ref,MANIFEST=HERE+"/source_manifest.v1.json",DRIVER=HERE+"/root_driver_v1.py",STATUSES={"postselection":"completed_H061_postselection_CPU_stage"},PHASE="postselection")
 iso([fn(contract,"_in_output"),fn(contract,"accepted_stage")],scope)
 original=scope["accepted_stage"];c=SimpleNamespace(accepted_stage=original,require=need,raw=lambda record:auth.append(record)or raw(record))
 hook=iso([fn(current,"_install_H061_prior_freeze_join")],dict(bindings=lambda:b,types=types,copy=copy))["_install_H061_prior_freeze_join"];hook(c)
 cells=dict(zip(c.accepted_stage.__code__.co_freevars,c.accepted_stage.__closure__))
 frozen=cells["frozen"].cell_contents
 need(frozen.__code__ is original.__code__ and frozen.__defaults__==original.__defaults__ and frozen.__kwdefaults__==original.__kwdefaults__,"entire original acceptance code")
 need({k for k,v in scope.items()if frozen.__globals__[k]is not v}=={"MANIFEST","DRIVER"},"exact two expected globals only")
 need(cells["original"].cell_contents is original and original.__globals__["MANIFEST"]==HERE+"/source_manifest.v1.json","predict branch stays original current-v2 identity")
 acceptance=read(prior["acceptance_ref"]);envelope=read(acceptance["completed_ref"])
 q=dict(freeze_acceptance_ref=prior["acceptance_ref"],train_acceptance_ref=envelope["upstream"]["train_acceptance_ref"],scientific_config=envelope["scientific_config"])
 need(c.accepted_stage(q,"freeze")==read(envelope["result_ref"]),"actual pinned freeze metadata join")
 need(auth==[prior[k]for k in("acceptance_ref","source_ref","source_manifest_ref")],"all3 prior pins read uncached")
 records={"acceptance":prior["acceptance_ref"],"wait":acceptance["actual_wait_ref"],"envelope":acceptance["completed_ref"],"result":envelope["result_ref"]}
 mutations=[("acceptance","accepted",False),("acceptance","root_authorizer","other"),("acceptance","source_manifest_ref",ref(HERE+"/source_manifest.v1.json")),("acceptance","outer_terminal_consumed",False),("acceptance","outer_terminal_exit_code",1),("acceptance","stage","predict"),("wait","returncode",1),("wait","actual_direct_worker_wait",False),("wait","all_owned_processes_and_groups_drained",False),("wait","operational_failure","failed"),("wait","completed_ref",{}),("envelope","root_driver_ref",ref(HERE+"/root_driver_v1.py")),("envelope","scientific_config",{}),("envelope","upstream",{}),("envelope","input_ref",{}),("result","new_core_fits",1),("result","new_optimizer_updates",1),("result","status","not_completed")]
 negative=[]
 for target,key,value in mutations:
  record=records[target];changed=read(record);changed[key]=value;views[record["path"]]=changed
  reject(lambda:c.accepted_stage(q,"freeze"));views.clear();negative.append(target+":"+key)
 reject(lambda:c.accepted_stage(dict(q,freeze_acceptance_ref=dict(path=prior["acceptance_ref"]["path"],sha256="0"*64)),"freeze"))
 wrapper=next(n for n in fn(current,"contract").body if isinstance(n,ast.FunctionDef)and n.name=="validate_request")
 stage_if=next(n for n in wrapper.body if isinstance(n,ast.If)and ast.unparse(n.test)=="consumer == 'H061'")
 for stage in("freeze","other"):reject(lambda stage=stage:iso([stage_if],dict(consumer="H061",q=dict(stage=stage),c=SimpleNamespace(require=need))))
 for stage in("predict","score"):iso([stage_if],dict(consumer="H061",q=dict(stage=stage),c=SimpleNamespace(require=need)))
 # Authored checker inspected as AST/metadata-only before independent rerun.
 authored=runpy.run_path(str(base/"source_check_v1.py"))["check"]()
 need(authored==read(ref(HERE+"/source_check.result.v1.json")),"author result exact rerun")
 need([x["distinct_union"]for x in authored["handwritten_role_cases"]]==[61,62,63,64],"all role boundaries")
 # Actual derived original admission reaches rejecting memo before any staging put.
 builder=iso([fn(current,"_memo_admission_node")],dict(ast=ast,copy=copy,need=need))["_memo_admission_node"]
 class FP:
  def __init__(self,value):self.value=str(value);self.name=self.value.rsplit("/",1)[-1]
  def exists(self):return False
 prewrite=[]
 for who,sp in b["consumers"].items():
  events=[];request=dict(packet_directory="fixture/packet",output_directory="fixture/output",cpu_ids=[0])
  def guard(flag):events.append("guard_false");return object()
  def failure(*args):events.append("memo_reject");raise ValueError("prewrite fixture")
  cc=SimpleNamespace(validate_request=lambda record:(request,{},{}),local=FP,require=need,guard=guard)
  body=builder(fn(tree(sp["driver_ref"]["path"]),"admit"),who)
  function=iso([body],dict(c=cc,need=need,local=FP,guard=guard,os=SimpleNamespace(sched_getaffinity=lambda _:set([0])),_make_root_mutation_memo=failure))["admit"]
  reject(lambda:function({}));need(events==["guard_false","memo_reject"],"no put before memo preflight")
  prewrite.append(dict(consumer=who,events=events))
 worker=tree(HERE+"/worker_entry_v1.py")
 need(any(isinstance(n,ast.ImportFrom)and n.level==1 and n.module is None and n.names[0].name=="root_driver_v1"for n in worker.body),"relative worker")
 need(not any(n.startswith(("arggnn","scripts.project_budget","_child_source_repair","_fresh_ROOT_transaction_memo"))for n in sys.modules),"no project/guard/memo imports")
 return dict(schema="independent_H060_H061_txv2_checks_v1",source_only=True,source_ref=ref(HERE+"/root_driver_v1.py"),source_manifest_ref=ref(HERE+"/source_manifest.v1.json"),authenticated_source_refs=330,authenticated_audit_files=12,whole_common_AST_delta_exact=True,bindings_workers_assets_science_caps_executors_barriers_unchanged=True,memo2_separate_SOURCE_PASS=True,actual_freeze_join=True,original_acceptance_code_only_expected_MANIFEST_DRIVER_changed=True,freeze_predicate_negative_cases=negative,only_exact_acceptance_ref=True,prior_refs_uncached=True,new_H061_predict_score_only=True,predict_remains_current_v2_identity=True,memo_prewrite_failures=prewrite,author_metadata_AST_checker_exact_rerun=True,author_results=authored,actual_entries_guards_memo_state_mutex_resources_arrays_data_numerics_qualification_jobs_Git=False)
if __name__=="__main__":
 try:print(json.dumps(main(),indent=2,sort_keys=True))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)

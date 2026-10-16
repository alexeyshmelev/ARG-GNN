"""Actual imported admit lifecycle with a clearly synthetic reservation guard."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib,json,os,shutil,stat,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v4"
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads((ROOT/p).read_text())
def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 assert sys.dont_write_bytecode and sys.flags.no_user_site
 d=importlib.import_module(HERE.replace("/",".")+".root_driver_v1");w=importlib.import_module(HERE.replace("/",".")+".worker_entry_v1");c=d.c
 assert w.c is c and c.HERE==HERE and c.SOURCE==HERE+"/root_contract_v1.py"and c.DRIVER==HERE+"/root_driver_v1.py"and c.ENTRY==HERE+"/worker_entry_v1.py"
 assert c._h._GUARD is None and c.SELECTOR_REF is None
 guardtree=ast.parse((ROOT/c.GUARD_REF["path"]).read_text());driver=ast.parse((ROOT/c.DRIVER).read_text())
 pred=next(n.args[0]for n in ast.walk(fn(guardtree,"apply_reserve_v1"))if isinstance(n,ast.Call)and len(n.args)>1 and isinstance(n.args[1],ast.Constant)and n.args[1].value=="New key and fresh exclusive paths")
 actualpred=next(n.args[0]for n in ast.walk(fn(driver,"admit"))if isinstance(n,ast.Call)and len(n.args)>1 and isinstance(n.args[1],ast.Constant)and n.args[1].value=="New key and fresh exclusive paths")
 class Normalize(ast.NodeTransformer):
  def visit_Name(self,n):
   if n.id=="own":n.id="reserve"
   return n
  def visit_Call(self,n):
   self.generic_visit(n)
   if isinstance(n.func,ast.Name)and n.func.id=="_policy":n.func=ast.Attribute(value=ast.Name(id="g",ctx=ast.Load()),attr="_policy",ctx=ast.Load())
   return n
 assert ast.dump(Normalize().visit(copy.deepcopy(pred)),include_attributes=False)==ast.dump(actualpred,include_attributes=False)
 v4=ast.parse((ROOT/c.V4["path"]).read_text());names={"integer","canonical","file_bytes","parse","bound","paths_overlap","accounting_paths","scope_spec"}
 env=dict(Path=Path,stat=stat,os=os,hashlib=hashlib,json=json,MAX_JSON=32*1024*1024,LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock")
 exec(compile(ast.Module(body=[n for n in v4.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[]),"<original_readonly_scope_predicates>","exec"),env)
 actualguard=read("research/reviews/TreeLoom_matched_CPU_ROOT_actual_20261017_v3/reserve_authority.json")
 guards=list(actualguard["bindings"].values());assert len(guards)==45
 pure=dict(require=need,copy=copy,_raw=c.raw)
 exec(compile(ast.Module(body=[fn(guardtree,"consumer_bindings_v1")],type_ignores=[]),"<pure_guard_ref_union>","exec"),pure)
 fixture=ROOT/HERE/"SOURCE_ONLY_lifecycle_fixture";assert not fixture.exists();preparation=fixture/"preparation";preparation.mkdir(parents=True)
 packet=fixture/"future_packet";output=fixture/"future_output"
 q=read("research/reviews/TreeLoom_matched_CPU_ROOT_preparation_20261017_v3/ROOT_request.root_active_v1.json")
 q.update(preparation_directory=str(preparation.relative_to(ROOT)),packet_directory=str(packet.relative_to(ROOT)),output_directory=str(output.relative_to(ROOT)),active=False,execution_authorized=False,root_authorizer="SOURCE_ONLY_FIXTURE_NOT_AUTHORITY")
 request_path=preparation/"request.fixture.json";request_path.write_text(json.dumps(q));request_ref=ref(request_path)
 timeline=[];reserved=[False]
 def staged_put(path,value,*a,**k):
  path=Path(path)
  if not reserved[0]:assert path.parent==preparation and not packet.exists()and not output.exists()
  else:assert packet.exists()and not output.exists()and path.parent==packet
  r=original_put(path,value,*a,**k);timeline.append(dict(event="metadata_write",phase="post_receipt"if reserved[0]else"pre_reservation",name=path.name,parent=path.parent.name));return r
 class FakeGuard:
  MODE="SOURCE_ONLY_FAKE_GUARD_NO_ACCOUNTING"
  budget=types.SimpleNamespace(SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py",v4=types.SimpleNamespace(mutex=lambda root:contextlib.nullcontext(),scope_spec=env["scope_spec"]))
  def _state_locked(self,*a,**k):return ({},None,{},actualguard["bindings"])
  def _policy(self):return dict(never_credit_keys=[])
  def _pins(self,x):return dict(source_only_fixture=True)
  def consumer_bindings_v1(self,*groups):return pure["consumer_bindings_v1"](*groups)
  def operation_draft_v1(self,operation,reserve,**k):
   assert operation=="reserve"and not packet.exists()and not output.exists();timeline.append(dict(event="draft",paths_absent=True));return dict(request=copy.deepcopy(reserve),source_only_fixture=True)
  def apply_reserve_v1(self,authority_ref,**k):
   authority=c.read(authority_ref);reserve=authority["request"];paths=env["scope_spec"](ROOT,reserve)
   assert paths==[output,packet]and not any(p.exists()for p in paths)
   assert reserve["reserved_bytes"]==32*2**20 and len(reserve["consumer_bindings"])==61 and all(r in reserve["consumer_bindings"]for r in guards)
   assert Path(authority_ref["path"]).parent==Path(q["preparation_directory"])
   timeline.append(dict(event="fake_apply_returning_receipt",packet_absent=True,output_absent=True,complete_roles=61));reserved[0]=True
   return dict(kind="SOURCE_ONLY_FAKE_RECEIPT_NOT_A_RESERVATION",value=dict(reserve,root_admission=authority_ref),initial_ledger_pins=dict(source_only_fixture=True))
 fake=FakeGuard();original_put=d.put;original_validate=c.validate_request;original_guard=d.guard;original_input=c.input_spec;original_affinity=d.os.sched_getaffinity;original_selector=c.SELECTOR_REF
 try:
  c.validate_request=lambda r:(q,c.CAPS,{})
  d.guard=lambda consumer:fake
  c.input_spec=lambda r:None
  c.SELECTOR_REF=q["storage_policy_selector_ref"]
  d.os.sched_getaffinity=lambda pid:set(q["cpu_ids"])
  d.put=staged_put
  result=d.admit(request_ref)
  assert reserved[0]and packet.is_dir()and not output.exists()
  staged={x["name"]for x in timeline if x.get("phase")=="pre_reservation"}
  assert staged=={packet.name+".config.json",packet.name+".source_and_input_closure.json",packet.name+".reserve_authority.json"}
  after={x["name"]for x in timeline if x.get("phase")=="post_receipt"}
  assert after=={"reserve_receipt.json","storage.json","input.json","admitted.json"}
  assert HERE.replace("/",".")+".root_driver_v1"in result["execution_argv"]
 finally:
  d.put=original_put;c.validate_request=original_validate;d.guard=original_guard;c.input_spec=original_input;d.os.sched_getaffinity=original_affinity;c.SELECTOR_REF=original_selector
  shutil.rmtree(fixture)
 assert c._h._GUARD is None and c.SELECTOR_REF is None
 assert not any(name in sys.modules for name in("torch","numpy","torch_geometric","scripts.project_budget_v9","arggnn.project_budget_v4"))
 return dict(schema="H060_actual_admit_metadata_lifecycle_check_v4",status="PASS_SOURCE_ONLY_ACTUAL_ADMIT_WITH_GUARD_STUB",actual_driver=d.__name__,actual_contract=c.__name__,worker_contract_identical=True,
  original_fresh_key_path_predicate_AST_exact=True,actual_admit_body_exercised=True,fake_apply_asserted_both_reserved_paths_absent=True,
  staging_in_separate_existing_preparation=True,packet_and_receipt_storage_input_admitted_only_after_fake_receipt=True,output_left_absent=True,
  complete_scope_output_plus_packet_32MiB=True,complete_roles=61,all45raw38unique_guards_retained=True,
  real_guard_policy_mutex_capacity_admission_or_science_called=False,request_validation_and_final_input_authority_stubs_explicit=True,
  fixture_artifacts_removed=True,metadata_timeline=timeline)
if __name__=="__main__":print(json.dumps(main(),indent=2,sort_keys=True))

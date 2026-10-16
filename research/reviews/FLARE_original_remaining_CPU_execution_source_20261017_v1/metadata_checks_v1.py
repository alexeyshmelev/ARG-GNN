"""Inert actual-admit checks. Explicit stubs, no guard/resource/native/payload/truth calls."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib,json,os,shutil,stat,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1"
def need(ok,msg):
 if not ok:raise ValueError(msg)
def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
def main():
 assert sys.dont_write_bytecode and sys.flags.no_user_site
 d=importlib.import_module(HERE.replace("/",".")+".root_driver_v1");c=d.c
 assert c._h._GUARD is None and c.SELECTOR_REF is None
 # Extract only the original pure union and read-only scope predicates.
 guardtree=ast.parse((ROOT/c.GUARD_REF["path"]).read_text())
 v4=ast.parse((ROOT/"src/arggnn/project_budget_v4.py").read_text())
 names={"integer","canonical","file_bytes","parse","bound","paths_overlap","accounting_paths","scope_spec"}
 env=dict(Path=Path,stat=stat,os=os,hashlib=hashlib,json=json,MAX_JSON=32*2**20,
  LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock")
 exec(compile(ast.Module(body=[n for n in v4.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[]),"<original_readonly_scope>","exec"),env)
 pure=dict(require=need,copy=copy,_raw=c.raw)
 exec(compile(ast.Module(body=[fn(guardtree,"consumer_bindings_v1")],type_ignores=[]),"<original_pure_ref_union>","exec"),pure)
 snapshot=json.loads((ROOT/c.DEPS["GUARD_ROLE_SNAPSHOT"]["path"]).read_text())
 guards=snapshot["bindings"];rawroles=list(guards.values())
 assert len(rawroles)==45 and len({(r["path"],r["sha256"])for r in rawroles})==38
 framework=d.framework()
 assert framework.execute.__code__.co_filename==str(ROOT/c.FRAMEWORK["path"])
 assert framework.c is c and framework.validate_completion is d.validate_completion
 fixture=ROOT/HERE/"SOURCE_ONLY_metadata_lifecycle_fixture";assert not fixture.exists();fixture.mkdir()
 result=[]
 originals={k:getattr(c,k)for k in ("validate_request","guard","input_spec","SELECTOR_REF","completion","sources","put")}
 original_affinity=os.sched_getaffinity
 def inert_json(path,value):path.write_text(json.dumps(value));return c.ref(path)
 try:
  for stage in c.STAGES:
   cell=fixture/stage;cell.mkdir();prep=cell/"preparation";prep.mkdir();packet=cell/"future_packet";output=cell/"future_output"
   refs={name:inert_json(prep/(name+".json"),dict(source_only_fixture=True,active=False,role=name))for name in
    ("request","review","runtime","directive","context","count","registration","selector","previous","conversion","native")}
   q=dict(stage=stage,parent_id="american_c3_val_001",phase=stage+"_american_c3_val_001",
    preparation_directory=str(prep.relative_to(ROOT)),packet_directory=str(packet.relative_to(ROOT)),output_directory=str(output.relative_to(ROOT)),
    source_review_ref=refs["review"],runtime_ref=refs["runtime"],root_directive_ref=refs["directive"],context_ref=refs["context"],
    registration_ref=refs["registration"],storage_policy_selector_ref=refs["selector"],cpu_ids=[1,2,3,4],
    conversion_ref=None if stage=="convert"else refs["conversion"],native_run_ref=refs["native"]if stage=="seal"else None,
    previous_stage_evidence_ref=None if stage=="convert"else refs["previous"],created_at_utc="2000-01-01T00:00:00+00:00",
    expires_at_utc="2000-01-01T00:00:01+00:00",active=False,execution_authorized=False,root_authorizer="SOURCE_ONLY_FIXTURE_NOT_AUTHORITY")
   # The test is lifecycle/roles only. Full request/input authority deliberately stubbed.
   caps=c.stage_caps(stage,dict(output_cap_bytes=881252800 if stage=="convert"else(8536100352 if stage=="native"else 32000000)))
   binary=c.DEPS["BINARY_PROVENANCE"]
   # Unique original native completion metadata needed only for its binary-ref role.
   (prep/"native.json").write_text(json.dumps(dict(source_only_fixture=True,binary_provenance_ref=binary)))
   refs["native"]=c.ref(prep/"native.json");q["native_run_ref"]=refs["native"]if stage=="seal"else None
   class Core:
    def _context(self,*a,**k):return dict(targets={q["parent_id"]:dict(count_manifest_ref=refs["count"])})
    def inactive_stage_configs_v1(self,*a,**k):
     value=dict(execution_enabled=False,output_cap_bytes=caps["output_bytes"],binary_provenance_ref=binary)
     return dict(conversion=copy.deepcopy(value),native=copy.deepcopy(value),seal=copy.deepcopy(value))
   c.completion=lambda:Core()
   c.validate_request=lambda r:(q,caps,{})
   c.sources=lambda:dict(source_sha256={c.DRIVER:c.ref(c.DRIVER)["sha256"]})
   c.input_spec=lambda r:None
   c.SELECTOR_REF=refs["selector"]
   os.sched_getaffinity=lambda pid:{1,2,3,4}
   timeline=[];reserved=[False]
   def staged_put(path,value,*a,**k):
    path=Path(path)
    if not reserved[0]:assert path.parent==prep and not packet.exists()and not output.exists()
    else:assert packet.is_dir()and not output.exists()and path.parent==packet
    r=originals["put"](path,value,*a,**k)
    timeline.append(dict(event="write",phase="after_fake_receipt"if reserved[0]else"pre_reservation",name=path.name));return r
   c.put=staged_put
   class FakeGuard:
    MODE="SOURCE_ONLY_FAKE_NO_ACCOUNTING"
    budget=types.SimpleNamespace(SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py",
     v4=types.SimpleNamespace(mutex=lambda root:contextlib.nullcontext(),scope_spec=env["scope_spec"]))
    def _state_locked(self,*a,**k):return ({},None,{},guards)
    def _policy(self):return dict(never_credit_keys=[])
    def _pins(self,x):return dict(source_only_fixture=True)
    def consumer_bindings_v1(self,*groups):return pure["consumer_bindings_v1"](*groups)
    def operation_draft_v1(self,operation,reserve,**k):
     assert operation=="reserve"and not packet.exists()and not output.exists()
     timeline.append(dict(event="fake_draft",paths_absent=True))
     return dict(request=copy.deepcopy(reserve),source_only_fixture=True,not_real_authority=True)
    def apply_reserve_v1(self,ar,**k):
     reserve=c.read(ar)["request"];paths=env["scope_spec"](ROOT,reserve)
     assert paths==[output,packet]and not any(p.exists()for p in paths)
     assert reserve["reserved_bytes"]==caps["reservation_bytes"]and reserve["auxiliary_scopes"]==[str(packet)]
     assert all(r in reserve["consumer_bindings"]for r in rawroles)
     assert Path(ar["path"]).parent==Path(q["preparation_directory"])
     timeline.append(dict(event="fake_receipt_only",paths_absent=True,roles=len(reserve["consumer_bindings"])))
     reserved[0]=True
     return dict(kind="SOURCE_ONLY_FAKE_NOT_RESERVATION",value=dict(reserve,root_admission=ar),initial_ledger_pins=dict(source_only_fixture=True))
   c.guard=lambda consumer:FakeGuard()
   answer=d.admit(refs["request"])
   assert reserved[0]and packet.is_dir()and not output.exists()
   assert {x["name"]for x in timeline if x.get("phase")=="pre_reservation"}=={
    packet.name+".config.json",packet.name+".source_and_input_closure.json",packet.name+".reserve_authority.json"}
   assert {x["name"]for x in timeline if x.get("phase")=="after_fake_receipt"}=={"reserve_receipt.json","storage.json","input.json","admitted.json"}
   assert HERE.replace("/",".")+".root_driver_v1"in answer["execution_argv"]
   result.append(dict(stage=stage,direct_guard_union=answer["consumer_binding_count"],whole_charge=caps["reservation_bytes"],
    output_bytes=caps["output_bytes"],timeline=timeline))
 finally:
  for k,v in originals.items():setattr(c,k,v)
  os.sched_getaffinity=original_affinity
  shutil.rmtree(fixture)
 # Actual lifecycle predicate with clearly synthetic metadata, including exact close digest.
 reserve=dict(key="SOURCE_ONLY",reserved_bytes=1,output=str(ROOT/"data/SOURCE_ONLY"),auxiliary_scopes=[],consumer_bindings=[])
 completion_ref=dict(path="data/SOURCE_ONLY/completed.json",sha256="0"*64)
 input_ref=dict(path="SOURCE_ONLY_input.json",sha256="0"*64);wait_ref=dict(path="SOURCE_ONLY_wait.json",sha256="0"*64)
 e=dict(schema="FLARE_remaining_stage_ROOT_terminal_evidence_v1",root_authorizer="root_agent",stage="convert",parent_id="american_c3_val_001",
  completion_ref=completion_ref,input_ref=input_ref,actual_wait_ref=wait_ref,
  outer_terminal_ref=dict(path="SOURCE_ONLY_outer.json",sha256="0"*64),ordinary_close_receipt_ref=dict(path="SOURCE_ONLY_close.json",sha256="0"*64))
 inp=dict(schema="FLARE_remaining_CPU_admitted_input_v1",parent_id="american_c3_val_001",stage="convert",output_directory="data/SOURCE_ONLY",storage=dict(reservation=reserve),root_driver_ref=c.ref(c.DRIVER),main_source_ref=c.ref(c.ENTRY),source_sha256=c.sources()["source_sha256"])
 wait=dict(actual_direct_worker_wait=True,returncode=0,all_owned_processes_and_groups_drained=True,operational_failure=None,input_ref=input_ref,completed_ref=completion_ref)
 outer=dict(actual_terminal_consumed=True,actual_exit_code=0,completed_ref=completion_ref,direct_wait_ref=wait_ref)
 closed=dict(kind="concurrent_completed_scope_close_receipt_v1",key="SOURCE_ONLY",artifacts_deleted=0,value=dict(future_writes_and_relaunch_revoked=True,request=dict(completed_ref=completion_ref,direct_wait_ref=wait_ref),reservation_sha256=hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()))
 records={"e":e,input_ref["path"]:inp,wait_ref["path"]:wait,e["outer_terminal_ref"]["path"]:outer,e["ordinary_close_receipt_ref"]["path"]:closed}
 records[c.MANIFEST]=c.read(c.ref(c.MANIFEST))
 original_read=c.read;c.read=lambda r:records[r["path"]]
 try:
  c.stage_evidence(dict(path="e"),stage="convert",parent="american_c3_val_001",completion_ref=completion_ref)
  rejected=[]
  for obj,key,bad in [(wait,"returncode",1),(wait,"all_owned_processes_and_groups_drained",False),
   (outer,"actual_terminal_consumed",False),(closed["value"],"reservation_sha256","1"*64)]:
   old=obj[key];obj[key]=bad
   try:c.stage_evidence(dict(path="e"),stage="convert",parent="american_c3_val_001",completion_ref=completion_ref)
   except ValueError:rejected.append(key)
   else:raise AssertionError("Invalid lifecycle accepted: "+key)
   finally:obj[key]=old
 finally:c.read=original_read
 assert c._h._GUARD is None and c.SELECTOR_REF is None
 assert not any(x in sys.modules for x in ("torch","numpy","torch_geometric","scripts.project_budget_v9","arggnn.project_budget_v4"))
 return dict(schema="FLARE_runtime_inert_metadata_check_v1",status="PASS_SOURCE_ONLY",actual_imported_admit_exercised=True,
  existing_execute_function_reused_exact=True,raw_guards=45,unique_guards=38,stages=result,
  actual_lifecycle_predicate_positive_and_negative=True,rejected=rejected,
  no_real_guard_capacity_admission_process_payload_truth_or_Git=True,request_and_input_authority_stubs_explicit=True,
  selected_context_and_scientific_config_stubs_explicit=True,fixture_removed=True)
if __name__=="__main__":print(json.dumps(main(),indent=2,sort_keys=True))

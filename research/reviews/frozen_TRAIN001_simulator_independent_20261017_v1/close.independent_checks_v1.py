"""Independent TRAIN001 close source/AST/RAM review. No guard or operations imports."""
from pathlib import Path, PurePosixPath
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from contextlib import contextmanager,redirect_stdout
import ast,copy,hashlib,io,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE="research/reviews/frozen_TRAIN001_simulator_close_ready_20261017_v1"
GUARD="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py"
CHECKS=[]
def need(ok,message):
 if not ok:raise ValueError(message)
def check(name,fn):fn();CHECKS.append(name)
def rejects(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError,FileExistsError):return
 raise AssertionError("Expected failstop")
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def actual(r):
 data=(ROOT/r["path"]).read_bytes();assert hashlib.sha256(data).hexdigest()==r["sha256"]
 return json.loads(data)
def function(path,name,scope):
 t=ast.parse((ROOT/path).read_bytes());n=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),path+"#RAM_AST","exec"),scope);return scope[name]
def encoded(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
digest=lambda value:hashlib.sha256(encoded(value)).hexdigest()
def schema_RAM(request,blobs):
 handles={};counter=[10];root=None
 def st(path):
  key=str(path.relative_to(root));data=blobs[key]
  return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_size=len(data),st_dev=1,
   st_ino=int(hashlib.sha256(key.encode()).hexdigest()[:12],16),st_mtime_ns=1,st_ctime_ns=1)
 class RAMPath(PurePosixPath):
  def resolve(self):return self
  def lstat(self):return st(self)
 root=RAMPath(str(ROOT))
 class Memory(io.BytesIO):
  def fileno(self):return self.fd
 def open_(path,flags):
  assert flags==os.O_RDONLY|os.O_NOFOLLOW
  path=RAMPath(path);st(path);fd=counter[0];counter[0]+=1;handles[fd]=path;return fd
 def fdopen(fd,mode):
  assert mode=="rb";f=Memory(blobs[str(handles[fd].relative_to(root))]);f.fd=fd;return f
 env=dict(Path=RAMPath,LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock",
  MAX_JSON=32*2**20,json=json,hashlib=hashlib,stat=stat,os=SimpleNamespace(open=open_,fdopen=fdopen,
  fstat=lambda fd:st(handles[fd]),O_RDONLY=os.O_RDONLY,O_NOFOLLOW=os.O_NOFOLLOW))
 for name in ("encoded","integer","canonical","file_bytes","parse","paths_overlap","accounting_paths","bound","scope_spec"):
  function("src/arggnn/project_budget_v4.py",name,env)
 return env["scope_spec"](root,copy.deepcopy(request))
def fixtures():
 manifest=actual(ref(BASE+"/source_manifest.v1.json"));pins=actual(ref(BASE+"/pins.v1.json"));blobs={}
 for r in manifest["source_refs"]:
  b=(ROOT/r["path"]).read_bytes();assert hashlib.sha256(b).hexdigest()==r["sha256"];blobs[r["path"]]=b
 CHECKS.append("all21_manifest_source_and_compact_metadata_refs_authenticated")
 ir=dict(path="research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/ROOT_actual_TRAIN001_root02_oct17_v5/input.json",
  sha256="91743348259ade65ccf83017ece27a01ba61df0147c8f5ee192e9f05b289c491")
 inp=actual(ir);docs={ir["path"]:inp}
 for r in (inp["refs"]["storage"],inp["root_stage_request_ref"],inp["refs"]["config"],inp["caps_ref"]):
  docs[r["path"]]=actual(r);blobs[r["path"]]=(ROOT/r["path"]).read_bytes()
 own=docs[inp["refs"]["storage"]["path"]]["reservation"]
 assert own["reserved_bytes"]==7000000000 and len(own["consumer_bindings"])==61
 for r in own["consumer_bindings"]:
  b=(ROOT/r["path"]).read_bytes();assert hashlib.sha256(b).hexdigest()==r["sha256"];blobs[r["path"]]=b
 CHECKS.append("actual_original_input_full7GB_and_all61_bindings_authenticated")
 def ram(path,value):
  data=encoded(value);r=dict(path=path,sha256=hashlib.sha256(data).hexdigest());docs[path]=copy.deepcopy(value);blobs[path]=data;return r
 observed=ram("RAM_ONLY/observed.json",dict(schema="american_c3_observed_v1",parent_id="american_c3_train_001",
  sequence_length_bp=30000000,counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),
  query_ancestry_truth_in_inputs=False,migrations_and_internal_population_labels_removed=True))
 completed=ram(inp["output_directory"]+"/completed.json",dict(status="completed_C3_simulation_reconstruction",
  parent_id="american_c3_train_001",sample_unit="haploid",class_order=["EUR","EAS","AFR"],
  sites=200001,actual_tables_bytes=100001,observation_ref=observed))
 launch=ram("RAM_ONLY/launch.json",dict(pid=6200,input_ref=ir,ownership=dict(signals_use_pidfd_only=True,
  no_numeric_group_signals=True,owned_identities=[dict(pid=6200,start=101)])))
 wait=ram("RAM_ONLY/wait.json",dict(schema="American_C3_full_original_CPU_actual_wait_v1",parent_id="american_c3_train_001",
  phase="simulate_train_001",input_ref=ir,pid=6200,completed_ref=completed,returncode=0,actual_OS_wait_status=0,
  actual_direct_worker_wait=True,actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,
  cleanup_errors=[],operational_failure=None,adopted_waits=[dict(pid=6201,status=0,returncode=0)],
  group_observations=[dict(pgid=6200,absent=True),dict(pgid=6201,absent=True)],
  known_owned_pids=[6200,6201],known_owned_process_groups=[6200,6201],supervisor_pid=6100,supervisor_process_group=6100))
 supervisor=ram("RAM_ONLY/supervisor.json",dict(pid=6100,actual_direct_wait=True,actual_returncode=0))
 outer=ram("RAM_ONLY/outer.json",dict(schema="ROOT_frozen_TRAIN001_simulator_actual_outer_terminal_v1",root_authorizer="root_agent",
  actual_terminal_consumed=True,actual_exit_code=0,actual_tool_session_id=123,actual_worker_wait_ref=wait,
  actual_supervisor_wait_ref=supervisor,completed_ref=completed,input_ref=ir,all_owned_processes_and_groups_drained=True,new_ancestry_score=False))
 p=dict(pins,input=ir,storage=inp["refs"]["storage"],original_request=inp["root_stage_request_ref"],config=inp["refs"]["config"],
  completed=completed,launch=launch,wait=wait,supervisor_wait=supervisor,outer=outer,reservation_key=own["key"],reservation_sha256=digest(own))
 env=dict(ROOT=ROOT,Path=Path,datetime=datetime,timezone=timezone,need=need,read=lambda r:copy.deepcopy(docs[r["path"]]),
  raw=lambda r:blobs[r["path"]],local=lambda path:Path(path)if Path(path).is_absolute()else ROOT/path)
 for name in ("metadata_join","request_value","current"):
  function(BASE+"/close_ROOT_v1.py",name,env)
 return SimpleNamespace(env=env,docs=docs,blobs=blobs,ram=ram,p=p,own=own,inp=inp,manifest=manifest,pins=pins)
def join_checks(f):
 env=f.env;p=f.p
 result=env["metadata_join"](p);assert result[5]==f.own and result[6]==[6100,6200,6201]and result[7]==[6100,6200,6201]
 CHECKS.append("complete_original_input_future_RAM_success_join_with_adopted_helper")
 cases=[
 ("wait","returncode",1),("wait","actual_OS_wait_status",15),("wait","actual_direct_worker_wait",False),
 ("wait","all_owned_processes_and_groups_drained",False),("wait","cleanup_errors",["RAM"]),
 ("wait","operational_failure","RAM"),("wait","adopted_waits",[dict(pid=6201,status=256,returncode=1)]),
 ("wait","group_observations",[dict(pgid=6200,absent=False)]),
 ("outer","actual_terminal_consumed",False),("outer","actual_exit_code",1),("outer","actual_tool_session_id",None),
 ("supervisor_wait","actual_returncode",1),("supervisor_wait","pid",999),
 ("completed","parent_id","american_c3_train_000"),("completed","status","FAILED"),
 ("input","GPU_authorized",True),("input","GNN_optimization_authorized",True),("input","query_truth_access_authorized",True)]
 for label,key,value in cases:
  path=p[label]["path"];saved=copy.deepcopy(f.docs[path]);f.docs[path][key]=value
  check("metadata_join_reject_"+label+"_"+key,lambda:rejects(lambda:env["metadata_join"](p)));f.docs[path]=saved
 return result
def full_close_RAM(f,case):
 docs=copy.deepcopy(f.docs);events=[];saved={};created=[False];p=f.p;own=f.own
 now=datetime.now(timezone.utc);end=(now+timedelta(minutes=10)).isoformat()
 def ram(path,value):
  data=encoded(value);r=dict(path=path,sha256=hashlib.sha256(data).hexdigest());docs[path]=copy.deepcopy(value);return r
 source_ref=ref(BASE+"/close_ROOT_v1.py");manifest_ref=ref(BASE+"/source_manifest.v1.json")
 review=ram("RAM_ONLY/independent_review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],source_ref=source_ref,source_manifest_ref=manifest_ref))
 rev=ram("RAM_ONLY/revocation.json",dict(schema="ROOT_frozen_TRAIN001_simulator_permanent_route_revocation_v1",
  root_authorizer="root_agent",active=True,input_ref=p["input"],direct_wait_ref=p["wait"],completed_ref=p["completed"],
  actual_outer_terminal_ref=p["outer"],reservation_key=own["key"],reservation_sha256=digest(own),output_directory=f.inp["output_directory"],
  worker_auxiliary_scope=f.inp["worker_auxiliary_scope"],all_original_launch_resume_retry_write_routes_permanently_revoked=True,
  preserve_all_files_original_clocks_failures_and_reservations=True,revoked_at_utc=(now-timedelta(seconds=1)).isoformat()))
 auth=dict(schema="ROOT_frozen_TRAIN001_simulator_ordinary_close_once_authorization_v1",root_authorizer="root_agent",active=True,
  once_authorized=True,operation="close_completed",source_manifest_ref=manifest_ref,source_review_ref=review,
  reservation_key=own["key"],reservation_sha256=digest(own),storage_policy_selector_ref=f.pins["selector"],
  ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions=True,preserve_all_files_and_original_reservations=True,
  no_retry_retirement_or_new_quota_framework=True,created_at_utc=(now-timedelta(seconds=10)).isoformat(),expires_at_utc=end,
  evidence_refs={name:p[name]for name in ("input","launch","wait","completed","outer","supervisor_wait","storage","original_request","config")},
  permanent_route_revocation_ref=rev,run_directory=BASE+"/ROOT_actual_success_RAM_ONLY")
 if case=="inactive":auth["active"]=False
 if case=="not_once":auth["once_authorized"]=False
 if case=="expired":auth["expires_at_utc"]=(now-timedelta(seconds=1)).isoformat()
 if case=="wide_window":auth["expires_at_utc"]=(now+timedelta(hours=2)).isoformat()
 if case=="missing_evidence":auth["evidence_refs"].pop("outer")
 if case=="wrong_review_manifest":docs[review["path"]]["source_manifest_ref"]=p["producer_manifest"]
 if case=="revocation_not_permanent":docs[rev["path"]]["all_original_launch_resume_retry_write_routes_permanently_revoked"]=False
 if case=="future_revocation":docs[rev["path"]]["revoked_at_utc"]=(now+timedelta(seconds=60)).isoformat()
 if case=="wrong_digest":auth["reservation_sha256"]="0"*64
 ar=ram("RAM_ONLY/once.json",auth)
 docs[manifest_ref["path"]]=f.manifest
 rows=actual(ref("research/reviews/FLARE_VAL004_native_ROOT_close_Census_root01_oct17/ordinary_close_authority.json"))["bindings"]
 state=[{own["key"]:copy.deepcopy(own)},{},dict(base_pins={"RAM":"ONLY"},completed_overlay_prefix={"events":1,"tail_sha256":"0"*64},
  new_reserved_bytes=7000000000,completed_credit_bytes=0),rows,[],[]]
 if case=="changed_open":state[0][own["key"]]["reserved_bytes"]+=1
 if case=="role_omission":state[3]=dict(list(rows.items())[:-1])
 g=SimpleNamespace();gs=dict(ROOT=ROOT,Path=Path,datetime=datetime,timezone=timezone,copy=copy,require=need,
  _json=lambda r:copy.deepcopy(docs[r["path"]]),_completed_json=lambda r:copy.deepcopy(docs[r["path"]]),
  _raw=lambda r:f.blobs[r["path"]],MODE="Census_observation_two_scope_retention_v1",RETIRE_OPERATION="retire_stopped",
  ORDINARY_EXCLUSIONS=[],MALFORMED_KEY="RAM_MALFORMED",BOOTSTRAP_CHARGE=0,META_CHARGE=512*1024,
  CEILING_BYTES=980*10**9,HUMAN_LIMIT_BYTES=10**12,WARNING_BYTES=950*10**9)
 def canonical(root,path):return Path(path)if Path(path).is_absolute()else root/path
 def integer(value,label,minimum=0):need(type(value)is int and value>=minimum,label);return value
 @contextmanager
 def mutex(root):
  events.append("mutex_enter");yield;events.append("mutex_exit")
 def file_bytes(*args):raise AssertionError("No real file boundary")
 for fn in (canonical,mutex,file_bytes):fn.__module__="RAM_original_v4"
 v4=SimpleNamespace(__name__="RAM_original_v4",canonical=canonical,file_bytes=file_bytes,mutex=mutex,encoded=encoded,integer=integer)
 gs["budget"]=SimpleNamespace(v4=v4,digest=digest)
 def absent(pid):
  events.append("absent_"+str(pid))
  need(not(case=="present_PID"and pid==6201),"RAM present PID")
  return dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=now.isoformat())
 def paths(value):return [Path(value["output"])]+[Path(x)for x in value["auxiliary_scopes"]]
 def measure(root,scope):
  events.append("measure_whole_primary_and_AUX");assert scope==paths(own)
  return dict(conservative_scope_bytes=1000001,entries=20,scope_paths=[str(x)for x in scope],stat_only=True)
 v4.measure_scopes=measure
 gs.update(_utc=lambda:now.isoformat(),_absent=absent,_paths=paths,_policy=lambda:dict(never_credit_keys=[],base_projected_bytes=0),
  _state_locked=lambda *args,**kwargs:state)
 for name in ("_pointer","_identity","_completed_identity","operation_draft_v1","_authority","_expire","_proof","_pins",
  "consumer_bindings_v1","assert_open","_measurement","charge","warning_fields","apply_close_completed_v1"):
  function(GUARD,name,gs)
 def append(operation,authority,value,state):
  events.append("append_RAM")
  fn=next(n for n in ast.parse((ROOT/GUARD).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=="_append")
  assignment=next(n for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="event"for t in n.targets))
  scope=dict(events=state[4],hashes=state[5],operation=operation,authority_ref=authority,value=value,state=state)
  exec(compile(ast.Module(body=[assignment],type_ignores=[]),"RAM_original_event_assignment","exec"),scope)
  event_bytes=encoded(scope["event"])+b"\n";assert len(event_bytes)<=65536
  events.append("event_bytes_"+str(len(event_bytes)));return digest(scope["event"])
 gs["_append"]=append
 for name,value in gs.items():
  if callable(value)or name=="budget":setattr(g,name,value)
 g.MODE=gs["MODE"];g._POLICY_REF=p["policy"];g.load_selected_guard_v1=lambda selector:g
 class Loader:
  def exec_module(self,module):assert module is g;events.append("guard_import_RAM")
 spec=SimpleNamespace(name="RAM_FAKE_GUARD",loader=Loader())
 class Directory:
  def relative_to(self,root):return PurePosixPath(auth["run_directory"])
  def mkdir(self):
   if created[0]or case=="existing_attempt":raise FileExistsError()
   created[0]=True;events.append("exclusive_invocation")
 directory=Directory()
 def local(path):
  if path==auth["run_directory"]:return directory
  return canonical(ROOT,path)
 def read(r):
  if r["path"]in docs:return copy.deepcopy(docs[r["path"]])
  return actual(r)
 def save(directory,name,value):
  events.append("save_"+name);saved[name]=copy.deepcopy(value);return ram("RAM_ONLY/close/"+name,value)
 def raw(r):
  if r["path"]in f.blobs:return f.blobs[r["path"]]
  return (ROOT/r["path"]).read_bytes()
 def killpg(group,signal):
  events.append("group_"+str(group));assert signal==0
  if case=="present_group"and group==6201:return
  raise ProcessLookupError()
 env=dict(ROOT=ROOT,HERE=BASE,SOURCE=BASE+"/close_ROOT_v1.py",MANIFEST=BASE+"/source_manifest.v1.json",Path=Path,
  sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1,optimize=0),modules={}),need=need,
  datetime=datetime,timezone=timezone,json=json,pins=lambda:copy.deepcopy(f.pins),ref=ref,read=read,raw=raw,local=local,save=save,
  metadata_join=gs.get("metadata_join"),importlib=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda *a:spec,module_from_spec=lambda sp:g)),
  os=SimpleNamespace(killpg=killpg))
 for name in ("metadata_join","request_value","current","close_once"):function(BASE+"/close_ROOT_v1.py",name,env)
 with redirect_stdout(io.StringIO()):
  if case=="healthy":
   result=env["close_once"](ar);assert result["status"]=="ORDINARY_CLOSE_APPLIED"and result["credit_bytes"]==7000000000-1000001
   assert events.index("exclusive_invocation")<events.index("guard_import_RAM")<events.index("measure_whole_primary_and_AUX")<events.index("append_RAM")
   assert events.count("measure_whole_primary_and_AUX")==1 and events.count("append_RAM")==1
   assert all(events.count("absent_"+str(pid))==3 for pid in (6100,6200,6201))
   assert set(saved)=={"invocation.json","targeted_completion_observation.json","ordinary_close_authority.json","ordinary_close_receipt.json","result.json"}
   event_size=next(int(s.removeprefix("event_bytes_"))for s in events if s.startswith("event_bytes_"))
   return dict(event_bytes=event_size,records=len(saved))
  else:
   rejects(lambda:env["close_once"](ar));assert "append_RAM"not in events
if __name__=="__main__":
 f=fixtures();join_checks(f)
 roles=actual(ref("research/reviews/FLARE_VAL004_native_ROOT_close_Census_root01_oct17/ordinary_close_authority.json"))["bindings"]
 assert len(roles)==47 and all(r in f.own["consumer_bindings"]for r in roles.values())
 CHECKS.append("all47_original_roles_already_retained_in_exact61union")
 own_no_admission={k:v for k,v in f.own.items()if k!="root_admission"}
 schema_RAM(own_no_admission,f.blobs);CHECKS.append("complete_original_scope_spec_and_exact_file_bytes_helpers_RAM")
 bad=copy.deepcopy(own_no_admission);bad["consumer_bindings"].append(dict(path="RAM_ONLY/forbidden.lock",sha256="0"*64))
 check("full_original_scope_spec_rejects_raw_lock",lambda:rejects(lambda:schema_RAM(bad,f.blobs)))
 summary=full_close_RAM(f,"healthy");CHECKS.append("complete_close_once_original_draft_proof_authority_apply_measurement_event_RAM")
 for case in ("inactive","not_once","expired","wide_window","missing_evidence","wrong_review_manifest","revocation_not_permanent",
  "future_revocation","wrong_digest","changed_open","role_omission","present_PID","present_group","existing_attempt"):
  check("complete_close_failstop_"+case,lambda case=case:full_close_RAM(f,case))
 print(json.dumps(dict(schema="independent_TRAIN001_ordinary_close_source_AST_RAM_checks_v1",status="PASS",count=len(CHECKS),
  checks=CHECKS,summary=summary,actual_original_input_ref=f.p["input"],actual_original_reservation_bytes=7000000000,
  actual_original_bindings=61,all47roles_preserved=True,future_success_authority_absence_measurement_and_append_are_RAM_only=True,
  actual_guard_imports_process_state_scope_accounting_or_science_operations=0),sort_keys=True,indent=2))

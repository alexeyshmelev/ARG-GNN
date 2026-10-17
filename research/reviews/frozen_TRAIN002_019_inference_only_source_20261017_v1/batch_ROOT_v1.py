"""Finite ROOT simulation batch. Default invocation is inert."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import argparse,ctypes,fcntl,hashlib,importlib.util,json,os,signal,stat,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/frozen_TRAIN002_019_inference_only_source_20261017_v1"
SOURCE=REL+"/batch_ROOT_v1.py";MANIFEST=REL+"/source_manifest.v1.json"
WORKER=REL+"/worker_entry_v1.py";CLOSE=REL+"/close_unit_ROOT_v1.py"
PYTHON=ROOT/".runtime/env/bin/python";CANCEL=[None]
def need(ok,message):
 if not ok:raise ValueError(message)
def local(value):
 p=Path(value);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and".."not in p.parts and p.resolve()==p,"Canonical contained path");return p
def ref(value):
 p=local(value);st=p.lstat()
 need(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and st.st_size<=2**20,"Bounded immutable metadata")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(rr):
 need(type(rr)is dict and set(rr)=={"path","sha256"}and ref(rr["path"])==rr,"Exact metadata ref")
 return local(rr["path"]).read_bytes()
def read(rr):return json.loads(raw(rr))
def load(rr,name):
 raw(rr);sp=importlib.util.spec_from_file_location(name,local(rr["path"]))
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def raw_save(directory,name,value):
 need(Path(name).name==name,"Flat exclusive metadata")
 data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();need(len(data)<=2**20,"Bounded metadata")
 with(directory/name).open("xb")as f:need(f.write(data)==len(data),"Complete write");f.flush();os.fsync(f.fileno())
 return ref(directory/name)
def qualified(authority_ref):
 m=read(ref(MANIFEST))
 for rr in m["refs"]:raw(rr)
 w=load(ref(WORKER),"_batch_static_worker");a=w.validate_pipeline_authority(authority_ref);review=read(a["source_review_ref"])
 need(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
  and review.get("source_manifest_ref")==ref(MANIFEST)
  and all(ref(p)in review.get("reviewed_sources",[])for p in(WORKER,SOURCE,CLOSE)),"Complete independent batch source review")
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0
  and Path(sys.executable).resolve()==PYTHON.resolve(),"Exact ROOT runtime")
 need(a["run_directory"].startswith(REL+"/ROOT_actual_pipeline_")and a["admission_directory"].startswith(REL+"/ROOT_pipeline_admission_")
  and not local(a["run_directory"]).is_relative_to(local(a["admission_directory"]))
  and not local(a["admission_directory"]).is_relative_to(local(a["run_directory"])),"Disjoint fresh namespaces")
 lanes=a["CPU_lanes"]
 need(type(lanes)is list and len(lanes)==2
  and all(type(v)is list and len(v)==8 and all(type(i)is int and i>=0 for i in v)for v in lanes),"Two8CPU lanes")
 need(len(set(lanes[0]+lanes[1]))==16 and set(lanes[0]+lanes[1])<=os.sched_getaffinity(0),"Actual16 distinct available CPUs")
 return w,a,m
def guard(m):
 p=m["close_pins"];mod=load(p["guard"],"_batch_exact_current_Census");g=mod.load_selected_guard_v1(p["selector"])
 need(g is mod and g.MODE=="Census_observation_two_scope_retention_v1"and g._POLICY_REF==p["policy"],"Exact current guard");return g
def state(g,m):
 p=m["close_pins"]
 with g.budget.v4.mutex(ROOT):
  s=g._state_locked(p["implementation"]["sha256"],p["launcher"]["sha256"],fresh=True)
  need(len(s[3])==47,"All47 current roles");return s
def put(ctx,name,value):
 ctx.w.validate_pipeline_authority(ctx.authority_ref)
 ctx.g.budget_check_v9(ctx.storage,boundary=True,required_write_bytes=2**20+8192)
 need(sum(1 for _ in ctx.directory.iterdir())<512,"Bounded batch admin entries")
 return raw_save(ctx.directory,name,value)
def admit_batch(authority_ref):
 w,a,m=qualified(authority_ref)
 with w.accounting_transaction(a["simulation_batch_control_input_ref"]):
  g=guard(m);directory=local(a["admission_directory"]);capacity=a["actual_initial_capacity"]
  need(not directory.exists()and not local(a["run_directory"]).exists(),"Fresh exclusive preparation admin")
  now=datetime.now(timezone.utc);stamp=datetime.fromisoformat(capacity["created_at_utc"])
  need(stamp.tzinfo and 0<=(now-stamp).total_seconds()<=300
   and capacity.get("root_authorizer")=="root_agent"and capacity.get("actual_current_observation")is True
   and capacity.get("source_only_estimate")is False and capacity.get("admissible_new_bytes",0)>=33554432,"Actual ROOT32MiB preparation admin capacity")
  s=state(g,m);directory.mkdir()
  bindings=g.consumer_bindings_v1(list(s[3].values()),m["refs"],list(m["close_pins"].values()),
   [ref(MANIFEST),authority_ref,a["source_review_ref"],m["controller_runtime_ref"],a["simulation_batch_control_input_ref"]])
  need(len(bindings)<=64,"Unchanged full47 roles within original64 direct ceiling")
  reserve=dict(key="H041/TreeLoom-TRAIN002-019-inference-admin-"+a["scope_id"],output=str(local(a["run_directory"])),
   auxiliary_scopes=[],reserved_bytes=33554432,consumer_bindings=bindings)
  auth=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(s[2]),bindings=s[3])
  auth.update(active=True,expires_at_utc=a["expires_at_utc"]);ar=raw_save(directory,"reserve_authority.json",auth)
  receipt=g.apply_reserve_v1(ar,expected_source_sha256=m["close_pins"]["implementation"]["sha256"],
   expected_launcher_sha256=m["close_pins"]["launcher"]["sha256"])
  rr=raw_save(directory,"reserve_receipt.json",receipt)
  storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
   job_id="TreeLoom-TRAIN002-019-inference-admin-"+a["scope_id"],output_directory=a["run_directory"],reserved_new_bytes=33554432,
   reservation=receipt["value"],initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=a["expires_at_utc"],
   completed_overlay_policy_ref=m["close_pins"]["policy"],
   budget_components=dict(implementation=m["close_pins"]["implementation"],launcher=m["close_pins"]["launcher"],consumer_adapter=m["close_pins"]["guard"]))
  g.budget_check_v9(storage,boundary=True,required_write_bytes=16384);local(a["run_directory"]).mkdir()
  ir=raw_save(directory,"pipeline_input.json",dict(schema="TreeLoom_TRAIN_preparation_admin_input_v1",authority_ref=authority_ref,
   storage=storage,reserve_receipt_ref=rr,source_manifest_ref=ref(MANIFEST),simulation_batch_control_input_ref=a["simulation_batch_control_input_ref"]))
  return dict(status="PREPARATION_ADMIN_ADMITTED",input_ref=ir,execution_started=False)
def original_owned(m):
 load(m["pidfd_ref"],"pidfd_backend_v1");owned=load(m["owned_ref"],"_batch_original_owned")
 sys.modules["pidfd_backend_v1"].self_probe_v1()
 need(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,"ROOT subreaper")
 need(not owned._children(os.getpid()),"No unrelated children");return owned
def run_group(commands,*,owned,deadline_epoch,rss_cap):
 need(1<=len(commands)<=2 and not owned._children(os.getpid()),"At most2 authorized roots")
 roots=[];tree=None;failure=None;adopted=[];peak=0;end_work=deadline_epoch-180
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=".:src:scripts",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
  OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
 def refresh():
  if tree is not None:tree.refresh();need(len(tree.known)<=128,"Finite aggregate roster")
 def consume():
  for r in roots:
   if r["status"]is None:
    pid,st=os.waitpid(r["proc"].pid,os.WNOHANG)
    if pid:r["status"]=st;r["proc"].returncode=os.waitstatus_to_exitcode(st)
   if not r["eof"]and r["nonblocking"]:
    try:data=os.read(r["proc"].stdout.fileno(),65536)
    except BlockingIOError:data=None
    if data==b"":r["eof"]=True
    elif data:r["output"].extend(data);need(len(r["output"])<=262144,"Bounded complete child terminal")
  if tree is not None:
   roots_set={r["proc"].pid for r in roots}
   for pid,row in list(tree.known.items()):
    if pid in roots_set:continue
    current=owned._stat(pid)
    if not owned._same(current,row)or current["ppid"]!=os.getpid():continue
    try:actual,st=os.waitpid(pid,os.WNOHANG)
    except ChildProcessError:continue
    if actual:adopted.append(dict(pid=actual,status=st,returncode=os.waitstatus_to_exitcode(st)))
 def absent():
  return tree is not None and tree.identities_absent()and not owned._children(os.getpid())
 try:
  for command in commands:
   need(CANCEL[0]is None and time.time()<end_work,"Current finite work before spawn")
   proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
   roots.append(dict(proc=proc,status=None,eof=False,output=bytearray(),argv=command,nonblocking=False,seed=owned._stat(proc.pid)))
   if tree is None:tree=owned.OwnedTree(proc.pid)
   else:refresh()
   os.set_blocking(proc.stdout.fileno(),False);roots[-1]["nonblocking"]=True
  while True:
   refresh();consume();rss=0
   for pid,row in list(tree.known.items()):
    if not owned._same(owned._stat(pid),row):continue
    try:rss+=int(Path("/proc/%d/statm"%pid).read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    except(FileNotFoundError,ProcessLookupError):pass
   peak=max(peak,rss);need(rss<=rss_cap,"Aggregate owned RSS")
   need(all(r["status"]is None or os.waitstatus_to_exitcode(r["status"])==0 for r in roots),"Genuine nonzero direct wait")
   if all(r["status"]is not None for r in roots):
    need(all(os.waitstatus_to_exitcode(r["status"])==0 for r in roots),"Genuine nonzero direct wait")
    if absent()and all(r["eof"]for r in roots):break
   need(CANCEL[0]is None and time.time()<end_work,"Canceled or finite work cutoff");time.sleep(.2)
 except BaseException as exc:failure=type(exc).__name__+": "+str(exc)[:2000]
 finally:
  until=time.monotonic()+30
  while roots and time.monotonic()<until:
   try:refresh();consume()
   except BaseException as exc:failure=failure or"Cleanup observation: "+str(exc)[:500]
   try:
    if absent()and all(r["status"]is not None and r["eof"]for r in roots):break
   except BaseException as exc:failure=failure or"Cleanup absence: "+str(exc)[:500]
   rows=list(tree.known.values())if tree is not None else[]
   for r in roots:
    if not any(row["pid"]==r["proc"].pid for row in rows):
     row=r["seed"]
     if row is not None and row["ppid"]==os.getpid():rows.append(row)
   for row in rows:
    try:owned.signal_identity(row,signal.SIGTERM if time.monotonic()<until-20 else signal.SIGKILL)
    except BaseException as exc:failure=failure or"pidfd cleanup: "+str(exc)[:500]
   time.sleep(.05)
  try:refresh();consume();drained=absent()
  except BaseException as exc:drained=False;failure=failure or"Final drain unknown: "+str(exc)[:500]
  groups=[]
  if tree is not None:
   for pgid in sorted({r["group"]for r in tree.known.values()}):
    try:os.killpg(pgid,0)
    except ProcessLookupError:groups.append(dict(pgid=pgid,absent=True))
    except OSError:groups.append(dict(pgid=pgid,absent=None))
    else:groups.append(dict(pgid=pgid,absent=False))
  drained=bool(drained and groups and all(g["absent"]is True for g in groups)
   and all(r["status"]is not None and r["eof"]for r in roots))
  for r in roots:r["proc"].stdout.close()
 result=[dict(pid=r["proc"].pid,actual_direct_wait=r["status"]is not None,actual_OS_wait_status=r["status"],
  actual_returncode=None if r["status"]is None else os.waitstatus_to_exitcode(r["status"]),
  all_owned_processes_and_groups_drained=drained,stdout=bytes(r["output"]).decode(errors="replace"),actual_argv=r["argv"],
  ownership=None if tree is None else tree.evidence(),group_observations=groups,adopted_waits=adopted,operational_failure=failure)for r in roots]
 success=bool(len(roots)==len(commands)and drained and failure is None
  and all(r["actual_returncode"]==0 for r in result)and all(r["returncode"]==0 for r in adopted))
 return dict(status="OS_TERMINALS_AND_OWNED_DRAIN"if success else"FAILED_OR_UNKNOWN_PRESERVED",success=success,
  roots=result,peak_owned_RSS_bytes=peak,UI_tool_terminal_claimed=False)
def terminal(result):
 values=[]
 for line in result["stdout"].splitlines():
  try:value=json.loads(line)
  except(ValueError,TypeError):continue
  if type(value)is dict:values.append(value)
 need(values,"Genuine child JSON terminal");return values[-1]
def argv(module,flag,rr):
 return[str(PYTHON),"-B","-s","-m",module[:-3].replace("/","."),flag,json.dumps(rr,separators=(",",":"))]
def available_memory():
 for line in Path("/proc/meminfo").read_text().splitlines():
  row=line.split()
  if row[0]=="MemAvailable:":return int(row[1])*1024
 raise ValueError("Actual MemAvailable absent")
def available_original_slots(ctx):
 # Observe only the original two synchronization inodes. No new pool is created.
 pool=ctx.manifest["synchronization_pool"];free=0;held=[]
 try:
  for value in pool["slot_paths"]:
   p=local(value);before=p.lstat()
   need(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size==0,"Original empty singly linked slot")
   fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW);held.append(fd);actual=os.fstat(fd)
   need((actual.st_dev,actual.st_ino,actual.st_size)==(before.st_dev,before.st_ino,0),"Original synchronization inode unchanged")
   try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   free+=1
  return free
 finally:
  for fd in held:os.close(fd)

def wait_wave_capacity(ctx,candidates):
 announced=False
 while True:
  ctx.w.validate_pipeline_authority(ctx.authority_ref)
  need(CANCEL[0]is None and time.time()+11550+1800<datetime.fromisoformat(ctx.authority["expires_at_utc"]).timestamp(),
   "Finite full original stage and genuine close headroom before admission")
  s=state(ctx.g,ctx.manifest);remaining=s[2]["limit_bytes"]-s[2]["projected_bytes"]
  ram=available_memory();slots=available_original_slots(ctx)
  selected=[]
  for unit in candidates:
   cap=ctx.manifest["condition"]["stage_caps"][unit["stage"]]
   if len(selected)<min(2,slots)and remaining>=cap["reservation_bytes"]+65536 and ram>=cap["rss_bytes"]:
    selected.append(unit);remaining-=cap["reservation_bytes"]+65536;ram-=cap["rss_bytes"]
  if selected:return selected
  if not announced:
   print(json.dumps(dict(status="QUEUED_NOT_ADMITTED",candidate_units=[u["unit_key"]for u in candidates],
    available_original_CPU_slots=slots,no_attempt_consumed=True)),flush=True);announced=True
  need(CANCEL[0]is None and time.time()+13350<datetime.fromisoformat(ctx.authority["expires_at_utc"]).timestamp(),
   "Finite queue before storage/slot admission, no consumed attempt")
  time.sleep(30)
def unit_request(ctx,unit,lane,capacity,ram):
 parent=unit["parent_id"];stage=unit["stage"];caps=ctx.manifest["condition"]["stage_caps"][stage]
 now=datetime.now(timezone.utc);end=min(now+timedelta(seconds=caps["wall_seconds"]+3600),datetime.fromisoformat(ctx.authority["expires_at_utc"]))
 need((end-now).total_seconds()>caps["wall_seconds"]+150,"Complete original wall and terminal evidence")
 clock=dict(created_at_utc=now.isoformat(),expires_at_utc=end.isoformat())
 prior=read(unit["prior_capsule_ref"]);cfg=ctx.w.scientific_config(parent,stage,prior["refs"]["observation_ref"])
 q=dict(clock,schema="TreeLoom_TRAIN_preparation_ROOT_request_v1",root_authorizer="root_agent",active=True,
  execution_authorized=True,GPU_authorized=False,parent_id=parent,stage=stage,phase=stage+"_"+parent.removeprefix("american_c3_"),
  fresh_unfinished_stage=True,automatic_retry=False,source_manifest_ref=ref(MANIFEST),storage_policy_selector_ref=ctx.manifest["close_pins"]["selector"],
  query_truth_access_authorized=False,GNN_optimization_authorized=False,TEST_access_authorized=False,
  current_Census_original_reserve_execute_authorized=True,pipeline_authority_ref=ctx.authority_ref,
  batch_control_input_ref=ctx.control_input_ref,source_review_ref=ctx.authority["source_review_ref"],caps=caps,
  cpu_ids=ctx.authority["CPU_lanes"][lane],scope_id=ctx.authority["scope_id"]+"_"+unit["unit_key"],
  runtime_ref=ctx.manifest["condition"]["stage_runtime_refs"][stage],scientific_config=cfg,native_MLM_dependency=False,
  prior_capsule_ref=unit["prior_capsule_ref"],bindings=dict(parent_id=parent,stage=stage,scientific_config=cfg,prior_capsule_ref=unit["prior_capsule_ref"]),
  packet_directory=REL+"/ROOT_actual_"+parent.removeprefix("american_c3_").upper()+"_"+stage+"_"+ctx.authority["scope_id"],
  output_directory="experiments/results/frozen_TreeLoom_full30Mb_"+parent.removeprefix("american_c3_").upper()+"_"+stage+"_"+ctx.authority["scope_id"],
  source_sha256={r["path"]:r["sha256"]for r in ctx.manifest["refs"]})
 q["actual_capacity"]=dict(clock,root_authorizer="root_agent",actual_current_observation=True,source_only_estimate=False,
  parent_id=parent,cpu_ids=q["cpu_ids"],admissible_new_bytes=capacity,available_owned_RSS_bytes=ram)
 return q
def group_call(ctx,commands,seconds,rss,name):
 deadline=min(time.time()+seconds,datetime.fromisoformat(ctx.authority["expires_at_utc"]).timestamp())
 result=run_group(commands,owned=ctx.owned,deadline_epoch=deadline,rss_cap=rss);rr=put(ctx,name,result)
 need(result["success"],"Actual failure/unknown preserved: "+rr["path"]);return result
def prepare_close(ctx,unit,input_ref,supervisor):
 parent=unit["parent_id"];stage=unit["stage"];name=unit["unit_key"]
 inp=read(input_ref);v=terminal(supervisor)
 need(v.get("status")=="PREPARED"and v.get("returncode")==0,"Original prepared terminal")
 wr=v["actual_wait_ref"];cp=v["completed_ref"];wait=read(wr)
 need(wait["input_ref"]==input_ref and wait["completed_ref"]==cp and wait["supervisor_pid"]==supervisor["pid"]
  and supervisor["actual_OS_wait_status"]==0 and supervisor["actual_returncode"]==0
  and supervisor["all_owned_processes_and_groups_drained"]is True,"Same real supervisor and workerwait")
 sw=put(ctx,name+".supervisor_wait.json",dict(supervisor,stdout=None,input_ref=input_ref,completed_ref=cp,direct_worker_wait_ref=wr))
 outer=put(ctx,name+".OS_terminal.json",dict(schema="ROOT_batch_unit_actual_OS_terminal_v1",root_authorizer="root_agent",
  actual_supervisor_OS_terminal_consumed=True,actual_exit_code=0,UI_tool_terminal_claimed=False,batch_authority_ref=ctx.authority_ref,
  actual_worker_wait_ref=wr,actual_supervisor_wait_ref=sw,completed_ref=cp,input_ref=input_ref,all_owned_processes_and_groups_drained=True,new_ancestry_score=False))
 own=inp["storage"]["reservation"];digest=ctx.g.budget.digest(own)
 rev=put(ctx,name+".permanent_revocation.json",dict(schema="ROOT_batch_unit_permanent_route_revocation_v1",
  root_authorizer="root_agent",active=True,input_ref=input_ref,direct_wait_ref=wr,completed_ref=cp,actual_outer_terminal_ref=outer,
  reservation_key=own["key"],reservation_sha256=digest,output_directory=inp["output_directory"],worker_auxiliary_scope=inp["worker_auxiliary_scope"],
  all_original_launch_resume_retry_write_routes_permanently_revoked=True,preserve_all_files_original_clocks_failures_and_reservations=True,
  revoked_at_utc=datetime.now(timezone.utc).isoformat()))
 now=datetime.now(timezone.utc);end=min(now+timedelta(seconds=1800),datetime.fromisoformat(ctx.authority["expires_at_utc"]))
 need((end-now).total_seconds()>300,"Finite close window")
 evidence=dict(input=input_ref,launch=ref(inp["worker_auxiliary_scope"]+"/launch.json"),wait=wr,completed=cp,outer=outer,
  supervisor_wait=sw,storage=inp["refs"]["storage"],original_request=inp["root_stage_request_ref"],config=inp["refs"]["config"])
 return put(ctx,name+".close_once.json",dict(schema="ROOT_batch_unit_ordinary_close_once_authorization_v1",root_authorizer="root_agent",
  active=True,once_authorized=True,operation="close_completed",source_manifest_ref=ref(MANIFEST),parent_id=parent,stage=stage,unit_key=name,batch_authority_ref=ctx.authority_ref,
  source_review_ref=ctx.authority["source_review_ref"],batch_control_input_ref=ctx.control_input_ref,evidence_refs=evidence,reservation_key=own["key"],reservation_sha256=digest,
  storage_policy_selector_ref=ctx.manifest["close_pins"]["selector"],ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions=True,
  preserve_all_files_and_original_reservations=True,no_retry_retirement_or_new_quota_framework=True,permanent_route_revocation_ref=rev,
  run_directory=ctx.authority["run_directory"]+"/close_"+name+"_"+ctx.authority["scope_id"],created_at_utc=now.isoformat(),expires_at_utc=end.isoformat()))
def accept_close(ctx,unit,root_result):
 v=terminal(root_result);need(v.get("status")=="ORDINARY_CLOSE_APPLIED","Actual successful ordinary close terminal")
 receipt=read(v["receipt_ref"]);s=state(ctx.g,ctx.manifest)
 cap=ctx.manifest["condition"]["stage_caps"][unit["stage"]]["reservation_bytes"]
 need(receipt["key"]in s[1]and receipt["completed_overlay_prefix"]["tail_sha256"]in s[5]
  and s[0][receipt["key"]]["reserved_bytes"]==cap,"Original journal accepted genuine stage close")
 return put(ctx,unit["unit_key"]+".close_accepted.json",dict(parent_id=unit["parent_id"],stage=unit["stage"],
  actual_direct_OS_wait0=True,all_owned_processes_and_groups_drained=True,receipt_ref=v["receipt_ref"],
  accepted_completed_overlay_prefix=s[2]["completed_overlay_prefix"],UI_tool_terminal_claimed=False,automatic_retry=False))
def capsule_from_actual(ctx,parent,next_stage,evidence,accepted_ref,revocation_ref,name):
 required={"input_ref","config_ref","completion_ref","wait_ref","outer_ref","observation_ref","roster_ref"}
 need(set(evidence)==required,"Seven genuine actual predecessor roles, no omission")
 values={k:read(v)for k,v in evidence.items()}
 capsule=dict(schema="TreeLoom_preparation_predecessor_capsule_v1",parent_id=parent,next_stage=next_stage,
  refs=evidence,metadata=values,close_accepted_ref=accepted_ref,permanent_revocation_ref=revocation_ref,query_truth_opened=False)
 # Parent/stage are selected only inside a fresh scientific worker; this writes metadata only.
 return put(ctx,name+".predecessor_capsule.json",capsule)

def simulated_candidate(ctx,parent):
 control=read(ctx.control_input_ref);sa=read(control["authority_ref"]);run=sa["run_directory"]
 accept_path=run+"/"+parent+".close_accepted.json"
 if not local(accept_path).exists():return None
 accepted=ref(accept_path);accepted_value=read(accepted);receipt=read(accepted_value["receipt_ref"])
 close_ref=ref(run+"/"+parent+".close_once.json");ca=read(close_ref);e=ca["evidence_refs"];inp=read(e["input"]);done=read(e["completed"])
 need(accepted_value["parent_id"]==parent and accepted_value["actual_direct_OS_wait0"]is True
  and accepted_value["all_owned_processes_and_groups_drained"]is True
  and receipt["key"]==inp["storage"]["reservation"]["key"],"Actual same-parent simulation ordinary-close receipt")
 state_now=ctx.readiness_state
 if receipt["key"]not in state_now[1]or receipt["completed_overlay_prefix"]["tail_sha256"]not in state_now[5]:return None
 obs=read(done["observation_ref"])
 prior=dict(input_ref=e["input"],config_ref=e["config"],completion_ref=e["completed"],wait_ref=e["wait"],outer_ref=e["outer"],
  observation_ref=done["observation_ref"],roster_ref=obs["roster_ref"])
 name=parent.removeprefix("american_c3_")+"_infer"
 capsule=capsule_from_actual(ctx,parent,"infer",prior,accepted,ca["permanent_route_revocation_ref"],name)
 return dict(parent_id=parent,stage="infer",unit_key=name,prior_capsule_ref=capsule,simulation_completion_ref=e["completed"])





def preoperation_queue(ctx,unit_key,result):
 need(result.get("status")=="QUEUED_BEFORE_OPERATION"and result.get("accounting_operation_started")is False,
  "Only exact no-operation-yet coordinator queue terminal")
 ctx.preoperation_deferrals+=1
 need(ctx.preoperation_deferrals<=32 and CANCEL[0]is None
  and time.time()+13350<datetime.fromisoformat(ctx.authority["expires_at_utc"]).timestamp(),
  "Finite overall no-operation queue, no consumed scientific/admission attempt")
 print(json.dumps(dict(status="QUEUED_BEFORE_OPERATION",unit_key=unit_key,actual_admission_attempt_consumed=False,
  reason=result.get("reason"),deferrals=ctx.preoperation_deferrals)),flush=True)

def close_when_ready(ctx,unit,authorization_ref):
 number=0
 while True:
  auth=read(authorization_ref)
  need(time.time()+450<datetime.fromisoformat(auth["expires_at_utc"]).timestamp(),
   "Original finite one-shot close authority remains current before any close invocation")
  close=group_call(ctx,[argv(CLOSE,"--authorization-ref",authorization_ref)],1500,4*2**30,
   unit["unit_key"]+".close_OS_%02d.json"%number)
  value=terminal(close["roots"][0])
  if value.get("status")!="QUEUED_BEFORE_OPERATION":return close
  preoperation_queue(ctx,unit["unit_key"]+"_close",value)
  number+=1;time.sleep(30)

def publish_parent_ready(ctx,unit,input_ref,supervisor,accepted):
 inp=read(input_ref);v=terminal(supervisor);done=read(v["completed_ref"]);obs=read(done["observation_ref"])
 need(unit["stage"]=="infer"and inp["parent_id"]==obs["parent_id"]==unit["parent_id"]
  and done["status"]=="completed_current_C3_inference"
  and obs["sequence_length_bp"]==30000000 and obs["query_ancestry_truth_in_inputs"]is False
  and obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
  "Actual full30Mb same-parent inferred observation; no focal/native readiness claim")
 name=unit["unit_key"]
 row=dict(schema="TreeLoom_inference_only_parent_ready_v1",root_authorizer="root_agent",parent_id=unit["parent_id"],
  input_ref=input_ref,config_ref=inp["refs"]["config"],completed_ref=v["completed_ref"],direct_wait_ref=v["actual_wait_ref"],
  actual_supervisor_wait_ref=ref(ctx.directory/(name+".supervisor_wait.json")),
  actual_outer_terminal_ref=ref(ctx.directory/(name+".OS_terminal.json")),observation_ref=done["observation_ref"],
  roster_ref=obs["roster_ref"],inference_close_accepted_ref=accepted,
  permanent_revocation_ref=ref(ctx.directory/(name+".permanent_revocation.json")),
  simulation_completion_ref=unit["simulation_completion_ref"],source_manifest_ref=ref(MANIFEST),inference_source_ref=ref(WORKER),
  class_order=["EUR","EAS","AFR"],sequence_length_bp=30000000,query_truth_opened=False,output_frozen=True,
  actual_original_worker_and_supervisor_OS_wait0=True,all_owned_processes_and_groups_drained=True,
  UI_tool_terminal_claimed=False,native_path_corpus_prepared=False,compact_focals_still_required=True)
 return put(ctx,"parent_ready."+unit["parent_id"]+".json",row)

def run_batch(input_ref):
 inp=read(input_ref);w,a,m=qualified(inp["authority_ref"]);g=guard(m);s=state(g,m)
 need(inp["schema"]=="TreeLoom_TRAIN_preparation_admin_input_v1"and inp["source_manifest_ref"]==ref(MANIFEST)
  and inp["simulation_batch_control_input_ref"]==a["simulation_batch_control_input_ref"]
  and inp["storage"]["reservation"]==s[0][inp["storage"]["reservation"]["key"]]
  and inp["storage"]["reservation"]["reserved_bytes"]==33554432
  and inp["storage"]["reservation"]["key"]not in s[1],"Actual open charged preparation admin")
 directory=local(a["run_directory"]);need(directory.is_dir()and not list(directory.iterdir()),"Fresh one-shot preparation run")
 with w.accounting_transaction(a["simulation_batch_control_input_ref"]):pass
 ctx=SimpleNamespace(w=w,control_input_ref=a["simulation_batch_control_input_ref"],authority=a,authority_ref=inp["authority_ref"],
  manifest=m,g=g,storage=inp["storage"],directory=directory)
 put(ctx,"invocation.json",dict(input_ref=input_ref,one_shot=True,automatic_retry=False,UI_tool_terminal_claimed=False))
 ctx.owned=original_owned(m);ctx.preoperation_deferrals=0;ctx.admission_requests=0;completed=[];attempted=[];pending={};unready=list(w.ALLOWED_PARENTS)
 rows={};simulation_completions={}
 old={sig:signal.signal(sig,lambda n,f:CANCEL.__setitem__(0,n))for sig in(signal.SIGINT,signal.SIGTERM)}
 try:
  wave_number=0
  while len(rows)<18:
   w.validate_pipeline_authority(ctx.authority_ref)
   ctx.readiness_state=state(g,m)
   for parent in list(unready):
    unit=simulated_candidate(ctx,parent)
    if unit is not None:pending[unit["unit_key"]]=unit;simulation_completions[parent]=unit["simulation_completion_ref"];unready.remove(parent)
   if not pending:
    need(CANCEL[0]is None and time.time()+13350<datetime.fromisoformat(a["expires_at_utc"]).timestamp(),"Finite queue for genuine simulator readiness")
    print(json.dumps(dict(status="QUEUED_FOR_GENUINE_SIMULATION_CLOSE",remaining_parents=unready,no_admission=True)),flush=True)
    time.sleep(30);continue
   candidates=wait_wave_capacity(ctx,list(pending.values()));inputs=[];wave=[]
   for lane,unit in enumerate(candidates):
    name=unit["unit_key"];need(name not in attempted,"No replay of original stage")
    state_now=state(g,m);number=ctx.admission_requests;ctx.admission_requests+=1
    qr=put(ctx,name+".request_%03d.json"%number,unit_request(ctx,unit,lane,state_now[2]["limit_bytes"]-state_now[2]["projected_bytes"],available_memory()))
    result=group_call(ctx,[argv(WORKER,"--admit-ref",qr)],900,4*2**30,name+".admission_OS_%03d.json"%number)
    admitted=terminal(result["roots"][0]);need(admitted["execution_started"]is False,"No science in accounting admission")
    if admitted.get("status")=="QUEUED_BEFORE_OPERATION":
     preoperation_queue(ctx,name,admitted);continue
    attempted.append(name);inputs.append(admitted["input_ref"]);wave.append(unit)
   if not inputs:time.sleep(30);continue
   seconds=max(m["condition"]["stage_caps"][u["stage"]]["wall_seconds"]for u in wave)+750
   result=group_call(ctx,[argv(WORKER,"--execute-ref",ir)for ir in inputs],seconds,len(inputs)*68719476736,"wave_%02d.preparation_OS.json"%wave_number)
   for unit,ir,sup in zip(wave,inputs,result["roots"]):
    ca=prepare_close(ctx,unit,ir,sup);close=close_when_ready(ctx,unit,ca)
    accepted=accept_close(ctx,unit,close["roots"][0]);completed.append(dict(parent_id=unit["parent_id"],stage=unit["stage"],input_ref=ir,close_accepted_ref=accepted))
    del pending[unit["unit_key"]]
    rows[unit["parent_id"]]=publish_parent_ready(ctx,unit,ir,sup,accepted)
   put(ctx,"wave_%02d.completed.json"%wave_number,dict(completed=completed[-len(wave):],inferred_ready_parent_ids=sorted(rows),new_score=False))
   print(json.dumps(dict(status="INFERENCE_WAVE_COMPLETED_AND_CLOSED",inferred_ready_parents=sorted(rows),completed_stages=len(completed),new_score=False)),flush=True)
   wave_number+=1
  parents=list(w.ALLOWED_PARENTS)
  need(set(rows)==set(parents)and len(completed)==len(attempted)==18,"Exactly18original inference units, no native corpus units")
  ready=put(ctx,"TRAIN002_019_inference_readiness.json",dict(schema="TreeLoom_TRAIN002_019_inference_only_readiness_v1",
   root_authorizer="root_agent",TRAIN_parent_ids=parents,parents={p:rows[p]for p in parents},all18_complete=True,
   class_order=["EUR","EAS","AFR"],sequence_length_bp=30000000,query_truth_opened=False,native_path_corpus_prepared=False,
   compact_focals_still_required=True,all20_training_readiness_claimed=False,UI_tool_terminal_claimed=False))
  done=put(ctx,"completed.json",dict(status="TRAIN002_019_ORIGINAL_INFERENCES_CLOSED",readiness_ref=ready,
   completed=completed,attempted_units=attempted,automatic_retry=False,new_ancestry_score=False,query_truth_opened=False,
   native_path_corpus_prepared=False,all20_training_readiness_claimed=False,
   UI_tool_terminal_claimed=False,admin_close_requires_ROOT_actual_final_terminal=True))
  return dict(status="INFERENCE_ONLY_COMPLETE",completed_ref=done,readiness_ref=ready,returncode=0)
 except BaseException as exc:
  failure=put(ctx,"failed_preserved.json",dict(status="PREPARATION_STOPPED_NO_REPLAY",failure=type(exc).__name__+": "+str(exc)[:3000],
   attempted_units=attempted,completed=completed,automatic_retry=False,no_unknown_or_failed_credit=True))
  return dict(status="PREPARATION_STOPPED_NO_REPLAY",failure_ref=failure,returncode=1)
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)
def main(argv_=None):
 p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
 mode.add_argument("--admit-batch-ref");mode.add_argument("--run-batch-ref");a=p.parse_args(argv_)
 if not(a.admit_batch_ref or a.run_batch_ref):print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False)));return 0
 result=admit_batch(json.loads(a.admit_batch_ref))if a.admit_batch_ref else run_batch(json.loads(a.run_batch_ref))
 print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())

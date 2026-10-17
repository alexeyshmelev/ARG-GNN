"""Finite ROOT simulation batch. Default invocation is inert."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import argparse,ctypes,fcntl,hashlib,importlib.util,json,os,signal,stat,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v2"
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
 w=load(ref(WORKER),"_batch_static_worker");a=w.validate_batch_authority(authority_ref);review=read(a["source_review_ref"])
 need(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
  and review.get("source_manifest_ref")==ref(MANIFEST)
  and all(ref(p)in review.get("reviewed_sources",[])for p in(WORKER,SOURCE,CLOSE)),"Complete independent batch source review")
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0
  and Path(sys.executable).resolve()==PYTHON.resolve(),"Exact ROOT runtime")
 need(a["run_directory"].startswith(REL+"/ROOT_actual_batch_")and a["admission_directory"].startswith(REL+"/ROOT_batch_admission_")
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
 ctx.w.validate_batch_authority(ctx.authority_ref)
 ctx.g.budget_check_v9(ctx.storage,boundary=True,required_write_bytes=2**20+8192)
 need(sum(1 for _ in ctx.directory.iterdir())<512,"Bounded batch admin entries")
 return raw_save(ctx.directory,name,value)
def admit_batch(authority_ref):
 w,a,m=qualified(authority_ref);g=guard(m);directory=local(a["admission_directory"]);capacity=a["actual_initial_capacity"]
 need(not directory.exists()and not local(a["run_directory"]).exists(),"Fresh batch")
 need(capacity.get("root_authorizer")=="root_agent"and capacity.get("actual_current_observation")is True
  and capacity.get("source_only_estimate")is False and capacity.get("admissible_new_bytes",0)>=33554432,"Actual ROOT32MiB initial capacity")
 s=state(g,m);directory.mkdir()
 bindings=g.consumer_bindings_v1(list(s[3].values()),m["refs"],list(m["close_pins"].values()),
  [ref(MANIFEST),authority_ref,a["source_review_ref"],m["controller_runtime_ref"]])
 reserve=dict(key="H041/TreeLoom-TRAIN002-019-batch-admin-"+a["scope_id"],output=str(local(a["run_directory"])),
  auxiliary_scopes=[],reserved_bytes=33554432,consumer_bindings=bindings)
 auth=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(s[2]),bindings=s[3])
 auth.update(active=True,expires_at_utc=a["expires_at_utc"]);ar=raw_save(directory,"reserve_authority.json",auth)
 receipt=g.apply_reserve_v1(ar,expected_source_sha256=m["close_pins"]["implementation"]["sha256"],
  expected_launcher_sha256=m["close_pins"]["launcher"]["sha256"])
 rr=raw_save(directory,"reserve_receipt.json",receipt)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
  job_id="TreeLoom-TRAIN002-019-batch-admin-"+a["scope_id"],output_directory=a["run_directory"],reserved_new_bytes=33554432,
  reservation=receipt["value"],initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=a["expires_at_utc"],
  completed_overlay_policy_ref=m["close_pins"]["policy"],
  budget_components=dict(implementation=m["close_pins"]["implementation"],launcher=m["close_pins"]["launcher"],consumer_adapter=m["close_pins"]["guard"]))
 run=local(a["run_directory"])
 g.budget_check_v9(storage,boundary=True,required_write_bytes=16384);run.mkdir()
 lock=run/"accounting_coordinator.lock";fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  st=os.fstat(fd);need(stat.S_ISREG(st.st_mode)and st.st_size==0 and st.st_nlink==1,"Actual exclusive coordinator inode")
  coordinator=dict(path=str(lock.relative_to(ROOT)),st_dev=st.st_dev,st_ino=st.st_ino)
 finally:os.close(fd)
 ir=raw_save(directory,"batch_input.json",dict(coordinator=coordinator,schema="ROOT_TRAIN002_019_batch_admitted_input_v1",
  authority_ref=authority_ref,storage=storage,reserve_receipt_ref=rr,source_manifest_ref=ref(MANIFEST)))
 return dict(status="BATCH_ADMIN_ADMITTED",input_ref=ir,execution_started=False)
def original_owned(m):
 load(m["pidfd_ref"],"pidfd_backend_v1");owned=load(m["owned_ref"],"_batch_original_owned")
 sys.modules["pidfd_backend_v1"].self_probe_v1()
 need(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,"ROOT subreaper")
 need(not owned._children(os.getpid()),"No unrelated children");return owned
def run_group(commands,*,owned,deadline_epoch,rss_cap):
 need(1<=len(commands)<=2 and not owned._children(os.getpid()),"At most2 authorized roots")
 roots=[];tree=None;failure=None;adopted=[];peak=0;end_work=min(deadline_epoch,time.time()+10800)-180
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

def wait_wave_capacity(ctx,count):
 announced=False
 while True:
  ctx.w.validate_batch_authority(ctx.authority_ref)
  need(CANCEL[0]is None and time.time()+16200<datetime.fromisoformat(ctx.authority["expires_at_utc"]).timestamp(),
   "Full finite wave, admission and close headroom remains before any admission")
  s=state(ctx.g,ctx.manifest)
  remaining=s[2]["limit_bytes"]-s[2]["projected_bytes"];ram=available_memory();slots=available_original_slots(ctx)
  if remaining>=count*7000000000+count*65536 and ram>=count*68719476736 and slots>=count:return remaining,ram
  if not announced:
   print(json.dumps(dict(status="QUEUED_NOT_ADMITTED",next_parent_count=count,available_new_bytes=remaining,
    actual_available_memory_bytes=ram,available_original_CPU_slots=slots,no_attempt_consumed=True)),flush=True);announced=True
  need(CANCEL[0]is None and time.time()+10800+600<datetime.fromisoformat(ctx.authority["expires_at_utc"]).timestamp(),
   "Finite queue ended before admission, no failed replay");time.sleep(30)
def unit_request(ctx,parent,lane,capacity,ram):
 now=datetime.now(timezone.utc);end=min(now+timedelta(seconds=14400),datetime.fromisoformat(ctx.authority["expires_at_utc"]))
 need((end-now).total_seconds()>10950,"Full original wall and evidence remains")
 clock=dict(created_at_utc=now.isoformat(),expires_at_utc=end.isoformat());cfg=ctx.w.scientific_config_for_parent(parent,True)
 q=dict(clock,schema="frozen_TRAIN002_019_full30Mb_simulator_ROOT_request_v1",root_authorizer="root_agent",active=True,
  execution_authorized=True,GPU_authorized=False,parent_id=parent,stage="simulate",phase="simulate_"+parent.removeprefix("american_c3_"),
  fresh_unfinished_parent=True,automatic_retry=False,source_manifest_ref=ref(MANIFEST),storage_policy_selector_ref=ctx.manifest["close_pins"]["selector"],
  query_truth_access_authorized=False,TRAIN_truth_creation_only=True,TEST_access_authorized=False,
  current_Census_original_reserve_execute_authorized=True,batch_authority_ref=ctx.authority_ref,batch_control_input_ref=ctx.control_input_ref,source_review_ref=ctx.authority["source_review_ref"],
  caps=ctx.manifest["caps_proposed"],cpu_ids=ctx.authority["CPU_lanes"][lane],scope_id=ctx.authority["scope_id"]+"_"+parent.removeprefix("american_c3_"),
  runtime_ref=ctx.manifest["controller_runtime_ref"],scientific_config=cfg,native_MLM_dependency=False,
  bindings=dict(parent_id=parent,scientific_config=cfg),packet_directory=REL+"/ROOT_actual_"+parent.removeprefix("american_c3_").upper()+"_"+ctx.authority["scope_id"],
  output_directory="experiments/results/frozen_TreeLoom_full30Mb_"+parent.removeprefix("american_c3_").upper()+"_"+ctx.authority["scope_id"],
  source_sha256={r["path"]:r["sha256"]for r in ctx.manifest["refs"]})
 q["actual_capacity"]=dict(clock,root_authorizer="root_agent",actual_current_observation=True,source_only_estimate=False,
  parent_id=parent,cpu_ids=q["cpu_ids"],admissible_new_bytes=capacity,available_owned_RSS_bytes=ram);return q
def group_call(ctx,commands,seconds,rss,name):
 deadline=min(time.time()+seconds,datetime.fromisoformat(ctx.authority["expires_at_utc"]).timestamp())
 result=run_group(commands,owned=ctx.owned,deadline_epoch=deadline,rss_cap=rss);rr=put(ctx,name,result)
 need(result["success"],"Actual failure/unknown preserved: "+rr["path"]);return result
def prepare_close(ctx,parent,input_ref,supervisor):
 inp=read(input_ref);v=terminal(supervisor)
 need(v.get("status")=="PREPARED"and v.get("returncode")==0,"Original prepared terminal")
 wr=v["actual_wait_ref"];cp=v["completed_ref"];wait=read(wr)
 need(wait["input_ref"]==input_ref and wait["completed_ref"]==cp and wait["supervisor_pid"]==supervisor["pid"]
  and supervisor["actual_OS_wait_status"]==0 and supervisor["actual_returncode"]==0
  and supervisor["all_owned_processes_and_groups_drained"]is True,"Same real supervisor and workerwait")
 sw=put(ctx,parent+".supervisor_wait.json",dict(supervisor,stdout=None,input_ref=input_ref,completed_ref=cp,direct_worker_wait_ref=wr))
 outer=put(ctx,parent+".OS_terminal.json",dict(schema="ROOT_batch_unit_actual_OS_terminal_v1",root_authorizer="root_agent",
  actual_supervisor_OS_terminal_consumed=True,actual_exit_code=0,UI_tool_terminal_claimed=False,batch_authority_ref=ctx.authority_ref,
  actual_worker_wait_ref=wr,actual_supervisor_wait_ref=sw,completed_ref=cp,input_ref=input_ref,all_owned_processes_and_groups_drained=True,new_ancestry_score=False))
 own=inp["storage"]["reservation"];digest=ctx.g.budget.digest(own)
 rev=put(ctx,parent+".permanent_revocation.json",dict(schema="ROOT_batch_unit_permanent_route_revocation_v1",
  root_authorizer="root_agent",active=True,input_ref=input_ref,direct_wait_ref=wr,completed_ref=cp,actual_outer_terminal_ref=outer,
  reservation_key=own["key"],reservation_sha256=digest,output_directory=inp["output_directory"],worker_auxiliary_scope=inp["worker_auxiliary_scope"],
  all_original_launch_resume_retry_write_routes_permanently_revoked=True,preserve_all_files_original_clocks_failures_and_reservations=True,
  revoked_at_utc=datetime.now(timezone.utc).isoformat()))
 now=datetime.now(timezone.utc);end=min(now+timedelta(seconds=1800),datetime.fromisoformat(ctx.authority["expires_at_utc"]))
 need((end-now).total_seconds()>300,"Finite close window")
 evidence=dict(input=input_ref,launch=ref(inp["worker_auxiliary_scope"]+"/launch.json"),wait=wr,completed=cp,outer=outer,
  supervisor_wait=sw,storage=inp["refs"]["storage"],original_request=inp["root_stage_request_ref"],config=inp["refs"]["config"])
 return put(ctx,parent+".close_once.json",dict(schema="ROOT_batch_unit_ordinary_close_once_authorization_v1",root_authorizer="root_agent",
  active=True,once_authorized=True,operation="close_completed",source_manifest_ref=ref(MANIFEST),parent_id=parent,batch_authority_ref=ctx.authority_ref,
  source_review_ref=ctx.authority["source_review_ref"],batch_control_input_ref=ctx.control_input_ref,evidence_refs=evidence,reservation_key=own["key"],reservation_sha256=digest,
  storage_policy_selector_ref=ctx.manifest["close_pins"]["selector"],ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions=True,
  preserve_all_files_and_original_reservations=True,no_retry_retirement_or_new_quota_framework=True,permanent_route_revocation_ref=rev,
  run_directory=ctx.authority["run_directory"]+"/close_"+parent+"_"+ctx.authority["scope_id"],created_at_utc=now.isoformat(),expires_at_utc=end.isoformat()))
def accept_close(ctx,parent,root_result):
 v=terminal(root_result);need(v.get("status")=="ORDINARY_CLOSE_APPLIED","Actual successful close terminal")
 receipt=read(v["receipt_ref"]);s=state(ctx.g,ctx.manifest)
 need(receipt["key"]in s[1]and receipt["completed_overlay_prefix"]["tail_sha256"]in s[5]
  and s[0][receipt["key"]]["reserved_bytes"]==7000000000,"Original journal accepted genuine close")
 return put(ctx,parent+".close_accepted.json",dict(parent_id=parent,actual_direct_OS_wait0=True,all_owned_processes_and_groups_drained=True,
  receipt_ref=v["receipt_ref"],accepted_completed_overlay_prefix=s[2]["completed_overlay_prefix"],UI_tool_terminal_claimed=False,automatic_retry=False))
def run_batch(input_ref):
 inp=read(input_ref);w,a,m=qualified(inp["authority_ref"]);g=guard(m);s=state(g,m)
 need(inp["schema"]=="ROOT_TRAIN002_019_batch_admitted_input_v1"and inp["source_manifest_ref"]==ref(MANIFEST)
  and inp["storage"]["reservation"]==s[0][inp["storage"]["reservation"]["key"]]
  and inp["storage"]["reservation"]["reserved_bytes"]==33554432 and inp["storage"]["reservation"]["key"]not in s[1],"Actual open batch admin charge")
 directory=local(a["run_directory"])
 need(directory.is_dir()and sorted(p.name for p in directory.iterdir())==["accounting_coordinator.lock"],"Fresh batch has only its actual coordinator")
 with w.accounting_transaction(input_ref):pass
 g.budget_check_v9(inp["storage"],boundary=True,required_write_bytes=2**20)
 ctx=SimpleNamespace(w=w,control_input_ref=input_ref,authority=a,authority_ref=inp["authority_ref"],manifest=m,g=g,storage=inp["storage"],directory=directory)
 put(ctx,"invocation.json",dict(batch_input_ref=input_ref,one_shot=True,automatic_retry=False,UI_tool_terminal_claimed=False))
 ctx.owned=original_owned(m);completed=[];attempted=[]
 old={sig:signal.signal(sig,lambda n,f:CANCEL.__setitem__(0,n))for sig in(signal.SIGINT,signal.SIGTERM)}
 try:
  parents=list(w.ALLOWED_PARENTS)
  for offset in range(0,len(parents),2):
   wave=parents[offset:offset+2];wait_wave_capacity(ctx,len(wave));inputs=[]
   for lane,parent in enumerate(wave):
    need(parent not in attempted,"No automatic replay")
    s=state(g,m);qr=put(ctx,parent+".request.json",unit_request(ctx,parent,lane,s[2]["limit_bytes"]-s[2]["projected_bytes"],available_memory()))
    attempted.append(parent)
    result=group_call(ctx,[argv(WORKER,"--admit-ref",qr)],900,4*2**30,parent+".admission_OS.json")
    admitted=terminal(result["roots"][0]);need(admitted.get("execution_started")is False,"No science during admission");inputs.append(admitted["input_ref"])
   result=group_call(ctx,[argv(WORKER,"--execute-ref",ir)for ir in inputs],11550,len(inputs)*68719476736,"wave_%02d.simulation_OS.json"%(offset//2))
   for parent,ir,sup in zip(wave,inputs,result["roots"]):
    ca=prepare_close(ctx,parent,ir,sup);close=group_call(ctx,[argv(CLOSE,"--authorization-ref",ca)],1500,4*2**30,parent+".close_OS.json")
    accepted=accept_close(ctx,parent,close["roots"][0]);completed.append(dict(parent_id=parent,input_ref=ir,close_accepted_ref=accepted))
   put(ctx,"wave_%02d.completed.json"%(offset//2),dict(completed=completed[-len(wave):],new_score=False))
   print(json.dumps(dict(status="WAVE_COMPLETED_AND_GENUINELY_CLOSED",completed_parents=[x["parent_id"]for x in completed],new_score=False)),flush=True)
  done=put(ctx,"completed.json",dict(status="REGISTERED_TRAIN002_019_SIMULATIONS_COMPLETED_AND_CLOSED",completed=completed,
   attempted_parents=attempted,automatic_retry=False,new_ancestry_score=False,UI_tool_terminal_claimed=False,batch_admin_close_requires_ROOT_actual_final_terminal=True))
  return dict(status="BATCH_COMPLETE",completed_ref=done,returncode=0)
 except BaseException as exc:
  failure=put(ctx,"failed_preserved.json",dict(status="BATCH_STOPPED_NO_REPLAY",failure=type(exc).__name__+": "+str(exc)[:3000],
   attempted_parents=attempted,completed=completed,automatic_retry=False,no_unknown_or_failed_credit=True))
  return dict(status="BATCH_STOPPED_NO_REPLAY",failure_ref=failure,returncode=1)
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)
def main(argv_=None):
 p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
 mode.add_argument("--admit-batch-ref");mode.add_argument("--run-batch-ref");a=p.parse_args(argv_)
 if not(a.admit_batch_ref or a.run_batch_ref):print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False)));return 0
 result=admit_batch(json.loads(a.admit_batch_ref))if a.admit_batch_ref else run_batch(json.loads(a.run_batch_ref))
 print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())

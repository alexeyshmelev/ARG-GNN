"""Inactive original VAL002–004 native/seal: Census selection and same-source owned worker."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,copy,ctypes,hashlib,importlib,importlib.util,json,os,signal,stat,subprocess,sys,time,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/FLARE_original_VAL002004_native_seal_Census_execution_source_20261017_v1'
SOURCE=HERE+'/root_driver_v1.py';ENTRY=SOURCE
MANIFEST=HERE+'/source_manifest.v1.json';HANDOFF=HERE+'/source_handoff.v1.json'
SETUP_SECONDS=7200;EVIDENCE_RESERVE=300;ORIGINAL_MARGIN=600
CAPS={};OUTPUT=PARENT=CONFIG=None

def need(ok,message):
 if not ok:raise ValueError(message)
def local(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and '..'not in p.parts and p.resolve()==p,'Canonical contained source/metadata');return p
def ref(path):
 p=local(path);st=p.lstat();need(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and st.st_size<=32*2**20,'Compact singly linked source/metadata')
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):need(isinstance(record,dict)and set(record)=={'path','sha256'}and ref(record['path'])==record,'Exact source/metadata SHA');return local(record['path']).read_bytes()
def read(record):return json.loads(raw(record))
def bindings():return read(ref(HERE+'/source_bindings.v1.json'))
def load(record,name):
 raw(record);sp=importlib.util.spec_from_file_location(name,local(record['path']));m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def phase_budget(stage):
 need(stage in('convert','native','seal'),'Original scientific stage')
 science=10800 if stage=='convert'else 14400 if stage=='native'else 3600;wall=science+600
 return dict(setup_seconds=SETUP_SECONDS,scientific_seconds=science,wall_seconds=wall,combined_seconds=SETUP_SECONDS+wall,evidence_reserve_seconds=300,authority_margin_seconds=600)

def contract():
 b=bindings();c=load(b['original_contract_ref'],'_FLARE_original_VAL002004_contract')
 c.HERE=HERE;c.SOURCE=c.DRIVER=SOURCE;c.ENTRY=ENTRY;c.MANIFEST=MANIFEST;c.HANDOFF=HANDOFF
 c.STAGES=('native','seal');c.VAL=tuple(sorted(b['original_targets']))
 selector=load(b['selector_bridge_ref'],'_FLARE_original_VAL002004_Census_selector')
 selector.install_v1(c,b)
 operational=load(b['operational_join_ref'],'_FLARE_original_VAL002004_source_joins')
 operational.install_v1(c,b,phase_budget)
 return c

def initial_storage_transaction(s,q,g):
 """One existing consumer-memo original-lock storage transaction; no new memo layer."""
 raw(bindings()['selected_child_source_ref']);tree=ast.parse(raw(bindings()['selected_child_source_ref']))
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_storage_v9')
 expected=ast.parse('return budget_check_v9(storage, first=True, boundary=True)').body[0]
 need(ast.dump(node.body[-1],include_attributes=False)==ast.dump(expected,include_attributes=False),'Exact original initial-storage return')
 prelude=copy.deepcopy(node);prelude.name='_FLARE_original_storage_prelude';prelude.body=prelude.body[:-1];scope=dict(g.__dict__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[prelude],type_ignores=[])),SOURCE+'#original-storage-prelude','exec'),scope)
 with g.budget.v4.mutex(ROOT):
  scope[prelude.name](s,dict(output_cap_bytes=c.CAPS['reservation_bytes']),s['storage'],source_ref=c.ref(c.DRIVER),config_ref=s['refs']['config'],runtime_ref=s['runtime_ref'])
  view=g.check_storage_locked_v1(s['storage'],first=True,boundary=True)
  need(isinstance(view.get('bindings'),dict),'Actual returned current bindings view')
  complete=g.consumer_bindings_v1(list(view['bindings'].values()),c.direct_roles(s['root_stage_request_ref'],q,s['refs']['config'],s['transitive_source_closure_ref']))
  c.require(all(rr in s['storage']['reservation']['consumer_bindings']for rr in complete),'Every actual returned/current guard/direct role retained; original64cap')
  return view

class PhaseStop(RuntimeError):
 def __init__(self,kind):super().__init__(kind);self.kind=kind
class PhaseClock:
 """No extension of old clocks. One fresh execution has fixed setup and original work caps."""
 def __init__(self):
  self.started=time.monotonic();self.started_utc=datetime.now(timezone.utc).isoformat();self.setup_deadline=self.started+7200
  self.phase_origin=None;self.scientific_deadline=None;self.work_deadline=None;self.combined_deadline=None;self.authority=None;self.cancel=[None];self.failure_kind=None;self.timeline=[];self.closing=False
  self.old_handlers={n:signal.getsignal(n)for n in(signal.SIGINT,signal.SIGTERM,signal.SIGALRM)}
  need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'Fresh interpreter without competing timer')
  signal.signal(signal.SIGINT,self.cancelled);signal.signal(signal.SIGTERM,self.cancelled);signal.signal(signal.SIGALRM,self.alarm)
  self.arm(self.setup_deadline,'SETUP_CUTOFF');self.mark('execution_entry')
 def mark(self,event,**fields):
  row=dict(event=event,monotonic=time.monotonic(),utc=datetime.now(timezone.utc).isoformat(),**fields)
  if len(self.timeline)<32:self.timeline.append(row)
  print(json.dumps(dict(FLARE_phase_event=row),sort_keys=True),flush=True)
 def arm(self,deadline,kind):
  self.alarm_kind=kind;remaining=deadline-time.monotonic()
  if remaining<=0:self.failure_kind=kind;raise PhaseStop(kind)
  signal.setitimer(signal.ITIMER_REAL,remaining,1.0) # Remains a failstop during exception unwinding until owned cleanup begins.
 def cancelled(self,number,frame):
  self.cancel[0]=number;self.failure_kind='CANCELLED'
  if not self.closing:
   self.alarm_kind='CANCELLED';signal.setitimer(signal.ITIMER_REAL,1.0,1.0);raise PhaseStop('CANCELLED')
 def alarm(self,number,frame):self.failure_kind=self.alarm_kind;raise PhaseStop(self.alarm_kind)
 def configure(self,q):
  self.budget=phase_budget(q['stage']);self.combined_deadline=self.started+self.budget['combined_seconds'];self.authority=datetime.fromisoformat(q['expires_at_utc']).timestamp()
  c.current(q,self.budget['combined_seconds']+600)
  authority_setup=time.monotonic()+self.authority-time.time()-self.budget['wall_seconds']-600
  self.arm(min(self.setup_deadline,authority_setup),'SETUP_CUTOFF'if self.setup_deadline<=authority_setup else'AUTHORITY_CUTOFF')
 def check_setup(self,s):
  if self.cancel[0]is not None:raise PhaseStop('CANCELLED')
  if time.monotonic()>=self.setup_deadline:raise PhaseStop('SETUP_CUTOFF')
  c.current(s,self.budget['wall_seconds']+600)
 def release(self,s,frame):
  self.check_setup(s);now=time.monotonic();need(now+self.budget['wall_seconds']<=self.combined_deadline,'Full original work window inside combined lifetime')
  self.phase_origin=now;self.scientific_deadline=now+self.budget['scientific_seconds'];self.work_deadline=min(now+self.budget['wall_seconds']-300,self.combined_deadline-300,time.monotonic()+self.authority-time.time()-300)
  self.mark('worker_ready_phase_boundary',worker_ready=frame);self.arm(self.work_deadline,'WORK_CUTOFF')
  return dict(schema='FLARE_scientific_release_v2',input_ref=frame['input_ref'],worker_pid=frame['pid'],phase_origin_monotonic=now,scientific_deadline_monotonic=self.scientific_deadline,supervisor_work_deadline_monotonic=self.work_deadline)
 def active(self):
  if self.cancel[0]is not None:self.failure_kind='CANCELLED';return False
  if time.monotonic()>=(self.setup_deadline if self.phase_origin is None else self.work_deadline):self.failure_kind='SETUP_CUTOFF'if self.phase_origin is None else'WORK_CUTOFF';return False
  return True
 def deadline_utc(self,value):
  return None if value is None else datetime.fromtimestamp(datetime.fromisoformat(self.started_utc).timestamp()+value-self.started,timezone.utc).isoformat()
 def evidence(self):return dict(setup_deadline_utc=self.deadline_utc(self.setup_deadline),scientific_deadline_monotonic=self.scientific_deadline,scientific_deadline_utc=self.deadline_utc(self.scientific_deadline),work_deadline_utc=self.deadline_utc(self.work_deadline),combined_deadline_utc=self.deadline_utc(self.combined_deadline),execution_started_monotonic=self.started,execution_started_utc=self.started_utc,setup_deadline_monotonic=self.setup_deadline,phase_origin_monotonic=self.phase_origin,work_deadline_monotonic=self.work_deadline,combined_deadline_monotonic=self.combined_deadline,authority_expiry_epoch=self.authority,cancellation_signal=self.cancel[0],failure_kind=self.failure_kind,timeline=self.timeline)
 def stop_work_alarm(self):self.closing=True;signal.setitimer(signal.ITIMER_REAL,0)
 def evidence_alarm(self):
  self.closing=False
  if self.combined_deadline is not None:
   deadline=min(self.combined_deadline,time.monotonic()+self.authority-time.time()-150)
   deadline=min(deadline,self.setup_deadline+300 if self.phase_origin is None else self.phase_origin+self.budget['wall_seconds'])
   self.arm(deadline,'EVIDENCE_CUTOFF')
 def close(self):
  signal.setitimer(signal.ITIMER_REAL,0)
  for number,handler in self.old_handlers.items():signal.signal(number,handler)

def activate():
 global c,need,local,ref,raw,read,load,put,current,guard,PIDFD,OWNED
 c=contract();need=c.require;local=c.local;ref=c.ref;raw=c.raw;read=c.read;load=c.load;put=c.put;current=c.current;guard=c.guard;PIDFD=c.PIDFD;OWNED=c.OWNED
 source=ast.parse(raw(c.FRAMEWORK));wanted=('input_spec','worker_python','sample_scope','boundary');nodes=[n for n in source.body if isinstance(n,ast.FunctionDef)and n.name in wanted]
 need(len(nodes)==len(wanted),'Exact original lifecycle helpers');exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/c.FRAMEWORK['path']),'exec'),globals())

def admit(record):
 activate();q=read(record);current(q,phase_budget(q['stage'])['combined_seconds']+600)
 d=load(bindings()['original_driver_ref'],'_FLARE_phase_original_admission_driver');d.c=c
 memo_bridge=load(bindings()['ROOT_memo_bridge_ref'],'_FLARE_original_Census_reserve_memo_bridge')
 memo_bridge.install_admission_v1(d,c,bindings())
 return d.admit(record)

def validate_completion(value,q):
 source=ast.parse(raw(bindings()['original_driver_ref']));node=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='validate_completion')
 scope=dict(c=c);exec(compile(ast.Module(body=[node],type_ignores=[]),SOURCE+'#original-completion','exec'),scope);return scope['validate_completion'](value,q)

def owned_execute(r,clock):
    need(os.getpid()==os.getpgrp()==os.getsid(0),"ROOT supervisor must use a dedicated actual session")
    s,q,cfg=input_spec(r);g=guard(True)
    need(Path.cwd()==ROOT and Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve(),"Original CPU supervisor runtime")
    need(not local(OUTPUT).exists() and not local(s["worker_auxiliary_scope"]).exists(),"Fresh one-shot execution only")
    initial_storage_transaction(s,q,g);clock.check_setup(s);clock.mark("supervisor_initial_storage_ready")
    aux=local(s["worker_auxiliary_scope"]);g.budget_check_v9(s["storage"],boundary=True,required_write_bytes=32768)
    aux.mkdir();(aux/"tmp").mkdir()
    put(aux/"attempt_consumed.json",dict(input_ref=r,consumed_utc=datetime.now(timezone.utc).isoformat(),one_shot=True,phase_clock=clock.evidence()),cap=4096)
    load(PIDFD,"pidfd_backend_v1");owned=load(OWNED,"_European_outer_owned")
    sys.modules["pidfd_backend_v1"].self_probe_v1()
    need(not owned._children(os.getpid()),"No unrelated children")
    need(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,"Owned subreaper")
    gate_read,gate_write=os.pipe();ready_read,ready_write=os.pipe();proc=tree=seed=None;status=None;failure=None;tail=bytearray();eof=False;nonblocking=False;cancel=clock.cancel;ready_buffer=bytearray();ready_count=0;scientific_entry=None
    old={} # PhaseClock installed handlers before all setup metadata, restores them at outer exit.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]),
        TMPDIR=str(aux/"tmp"),TMP=str(aux/"tmp"),TEMP=str(aux/"tmp"),OMP_NUM_THREADS="32",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="32",ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(c.POLICY_REF))
    for name,key in (("cache","XDG_CACHE_HOME"),("config","XDG_CONFIG_HOME"),("data","XDG_DATA_HOME"),("torch","TORCH_HOME"),
        ("cuda","CUDA_CACHE_PATH"),("inductor","TORCHINDUCTOR_CACHE_DIR"),("triton","TRITON_CACHE_DIR"),
        ("matplotlib","MPLCONFIGDIR"),("huggingface","HF_HOME"),("numba","NUMBA_CACHE_DIR")):
        env[key]=str(aux/"tmp"/name)
    env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",WANDB_MODE="disabled")
    clock.check_setup(s);clock.mark("worker_fork_requested")
    try:
        proc=subprocess.Popen([worker_python(q),"-B","-m",HERE.replace("/",".")+".root_driver_v1",
            "--worker-ref",json.dumps(r),"--gate-fd",str(gate_read),"--ready-fd",str(ready_write)],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(gate_read,ready_write),bufsize=0)
        os.close(gate_read);gate_read=None;os.close(ready_write);ready_write=None;os.set_blocking(ready_read,False);os.set_blocking(proc.stdout.fileno(),False);nonblocking=True
        seed=owned._stat(proc.pid);tree=owned.OwnedTree(proc.pid)
        put(aux/"launch.json",dict(pid=proc.pid,input_ref=r,ownership=tree.evidence(),gate_closed=True,worker_fork_observed_utc=datetime.now(timezone.utc).isoformat(),monotonic=time.monotonic()),cap=4096)
        clock.check_setup(s)
        need(os.write(gate_write,b"G")==1,"Original owned bootstrap gate write");clock.mark("worker_bootstrap_gate_released")
        while clock.active():
            if ready_read is not None:
                try:chunk=os.read(ready_read,4096)
                except BlockingIOError:chunk=None
                if chunk:
                    ready_buffer.extend(chunk);need(len(ready_buffer)<=8192,"Bounded actual worker status pipe")
                    while b"\n"in ready_buffer:
                        line,_,rest=ready_buffer.partition(b"\n");ready_buffer=bytearray(rest);need(len(line)<=4096,"Compact owned status frame")
                        frame=json.loads(line);ready_count+=1
                        validate_worker_frame(frame,r,proc.pid,ready_count,clock)
                        if ready_count==1:
                            grant=clock.release(s,frame);payload=(json.dumps(grant,sort_keys=True)+"\n").encode();need(len(payload)<=4096,"Bounded scientific gate grant")
                            need(os.write(gate_write,payload)==len(payload),"Full parent scientific-release grant");os.close(gate_write);gate_write=None
                            clock.mark("scientific_release_sent")
                            put(aux/"worker_ready.json",dict(ready=frame,release=grant),cap=8192)
                        else:
                            scientific_entry=frame;clock.mark("actual_worker_scientific_entry",receipt=frame);os.close(ready_read);ready_read=None
                elif chunk==b"":raise PhaseStop("WORKER_READY_PROTOCOL_EOF")
            event=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            tree.refresh();need(len(tree.known)<=64,"Bounded own process roster")
            rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
            for row in tree.known.values():
                if owned._same(owned._stat(row["pid"]),row):
                    try:rss+=int(Path("/proc/%d/statm"%row["pid"]).read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
                    except (FileNotFoundError,ProcessLookupError):pass
            need(rss<=CAPS["rss_bytes"],"Owned RSS fail-stop")
            current(s,300);sample_scope(s)
            try:data=os.read(proc.stdout.fileno(),65536)
            except BlockingIOError:data=None
            if data==b"":eof=True
            elif data:tail.extend(data);del tail[:-CAPS["log_bytes"]]
            if event is not None:
                # An exit can race the second already-written status frame. Consume it/EOF next.
                if ready_count<2:continue
                break
            time.sleep(.2)
        else:failure=clock.failure_kind or "UNCLASSIFIED_PHASE_CUTOFF"
    except BaseException as exc:
        if isinstance(exc,PhaseStop):clock.failure_kind=exc.kind
        elif "Current actual ROOT authority/margin"in str(exc):clock.failure_kind="AUTHORITY_CUTOFF"
        else:clock.failure_kind=clock.failure_kind or"EXCEPTION"
        failure=type(exc).__name__+": "+str(exc)[:2000]
    finally:
        clock.stop_work_alarm()
        for fd in (gate_read,gate_write,ready_read,ready_write):
            if fd is not None:
                try:os.close(fd)
                except OSError:pass
        stop=time.monotonic();end=stop+CAPS["cleanup_seconds"];absence=False;groups=[];adopted=[];errors=[]
        while proc is not None and time.monotonic()<end:
            try:
                if status is None:
                    pid,actual=os.waitpid(proc.pid,os.WNOHANG)
                    if pid==proc.pid:status=actual;proc.returncode=os.waitstatus_to_exitcode(actual)
                if tree is not None:tree.refresh();need(len(tree.known)<=64,"Bounded cleanup roster")
                rows=list(tree.known.values())[:64] if tree is not None else ([seed] if seed else [])
                for row in rows:
                    if time.monotonic()>=end:break
                    owned.signal_identity(row,signal.SIGTERM if time.monotonic()-stop<10 else signal.SIGKILL)
                if status is not None:
                    for _ in range(64):
                        if time.monotonic()>=end:break
                        try:pid,actual=os.waitpid(-1,os.WNOHANG)
                        except ChildProcessError:break
                        if not pid:break
                        if len(adopted)<64:adopted.append(dict(pid=pid,status=actual,returncode=os.waitstatus_to_exitcode(actual)))
                        else:failure=failure or "Adopted roster overflow"
                for _ in range(4):
                    if time.monotonic()>=end or not nonblocking:break
                    try:data=os.read(proc.stdout.fileno(),65536)
                    except BlockingIOError:break
                    if not data:eof=True;break
                    tail.extend(data);del tail[:-CAPS["log_bytes"]]
                if status is not None and tree is None:break
                if status is not None and tree is not None and time.monotonic()<end:
                    absence=tree.identities_absent() and not owned._children(os.getpid())
                    if absence and eof:
                        groups=[]
                        for pgid in sorted({x["group"] for x in tree.known.values()}):
                            gone=None
                            if time.monotonic()<end:
                                try:os.killpg(pgid,0)
                                except ProcessLookupError:gone=True
                                else:gone=False
                            groups.append(dict(pgid=pgid,absent=gone))
                        if groups and all(x["absent"] is True for x in groups):break
            except BaseException as exc:
                failure=failure or "Cleanup observation failed"
                if len(errors)<16:errors.append(type(exc).__name__+": "+str(exc)[:500])
            time.sleep(min(.05,max(0,end-time.monotonic())))
        if proc is not None and proc.stdout is not None:proc.stdout.close()
        for sig,handler in old.items():signal.signal(sig,handler)
    if clock.cancel[0] is not None:failure=failure or"CANCELLED";clock.failure_kind="CANCELLED"
    rc=None if status is None else os.waitstatus_to_exitcode(status)
    drained=status is not None and absence and eof and bool(groups) and all(x["absent"] is True for x in groups)
    if not drained:failure=failure or "Unresolved owned drain/wait"
    if rc!=0 or any(x["returncode"]!=0 for x in adopted):failure=failure or "Genuine nonzero/unknown child wait"
    observation=dict(schema="FLARE_phase_CPU_actual_wait_v2",input_ref=r,pid=None if proc is None else proc.pid,
        actual_direct_worker_wait=status is not None,actual_OS_wait_status=status,returncode=rc,
        all_owned_processes_and_groups_drained=drained,group_observations=groups,adopted_waits=adopted,
        operational_failure=failure,cleanup_errors=errors,outer_truth_scored=False,automatic_close_or_credit=False,phase_clock=clock.evidence(),actual_worker_scientific_entry=scientific_entry)
    print(json.dumps(observation,sort_keys=True),flush=True)
    if not drained or time.time()+150>=datetime.fromisoformat(s["expires_at_utc"]).timestamp():
        print(json.dumps(dict(retained_worker_tail=bytes(tail[-8192:]).decode(errors="replace"),no_future_stage_authorized=True)),flush=True)
        return dict(status="FAILED_UNRESOLVED_OR_EVIDENCE_MARGIN",returncode=1,actual_wait_ref=None,observation=observation)
    try:
        # Genuine drain precedes every new source/accounting/product inspection.
        clock.evidence_alarm()
        boundary(g,s,32768,target="aux")
        cp=None
        if local(OUTPUT+"/completed.json").is_file():
            cp=ref(OUTPUT+"/completed.json");completed=read(cp);validate_completion(completed,q)
        else:failure=failure or "No genuine completed matched training envelope"
        observation.update(completed_ref=cp,phase=q["phase"],parent_id=PARENT,actual_direct_launcher_wait=status is not None,
            known_owned_pids=sorted(tree.known) if tree is not None else [],known_owned_process_groups=[x["pgid"] for x in groups],
            supervisor_pid=os.getpid(),supervisor_process_group=os.getpgrp())
        observation["operational_failure"]=failure;observation["phase_clock"]=clock.evidence()
        need(len(json.dumps(observation,sort_keys=True).encode())<=32768,"Compact precharged complete wait evidence")
        wr=put(aux/"actual_wait.json",observation)
        boundary(g,s,len(tail)*6+8192,target="aux");put(aux/"worker_log.json",dict(tail=bytes(tail).decode(errors="replace"),retained_raw_bytes=len(tail)),cap=128*2**20)
        return dict(status="PREPARED" if failure is None else "FAILED",returncode=0 if failure is None else 1,
            actual_wait_ref=wr,completed_ref=cp,outer_tool_terminal_consumed=False,outer_truth_scored=False)
    except BaseException as exc:
        clock.failure_kind=exc.kind if isinstance(exc,PhaseStop)else('AUTHORITY_CUTOFF'if'Current actual ROOT authority/margin'in str(exc)else clock.failure_kind or'EXCEPTION')
        observation.update(phase_clock=clock.evidence(),prior_operational_failure=observation.get("operational_failure"),
            operational_failure=type(exc).__name__+": "+str(exc)[:2000])
        print(json.dumps(dict(actual_wait_observation=observation,evidence_failure=str(exc)[:2000])),flush=True)
        return dict(status="FAILED_EVIDENCE",returncode=1)

def validate_worker_frame(frame,record,pid,sequence,clock):
 expected='FLARE_worker_bootstrap_ready_v2'if sequence==1 else'FLARE_worker_scientific_entry_v2'
 need(sequence in(1,2)and isinstance(frame,dict)and frame.get('schema')==expected and frame.get('input_ref')==record and type(frame.get('pid'))is int and frame['pid']==pid,'Exact owned worker status identity/sequence')
 stamp=frame.get('monotonic');need(type(stamp)in(int,float)and clock.started<=stamp<=time.monotonic()+1,'Real ordered worker monotonic timestamp')
 utc=datetime.fromisoformat(frame['utc']);need(utc.tzinfo is not None and datetime.fromisoformat(clock.started_utc)<=utc and utc.timestamp()<=time.time()+1,'Actual bounded worker UTC timestamp')
 if sequence==1:
  need(frame.get('initial_metadata_storage_and_all_roles_validated')is True and type(frame.get('actual_role_count'))is int and 1<=frame['actual_role_count']<=64 and frame.get('worker_source_ref')==ref(ENTRY),'Only fully validated exact worker can become ready')
 else:need(clock.phase_origin is not None and clock.phase_origin<=stamp<clock.scientific_deadline and frame.get('phase_origin_monotonic')==clock.phase_origin and frame.get('scientific_deadline_monotonic')==clock.scientific_deadline,'Science entry follows exact actual parent release before original science cutoff')

def execute(record):
 clock=PhaseClock()
 try:
  activate();s=read(record);q=read(s['root_stage_request_ref']);clock.configure(q)
  return owned_execute(record,clock)
 except BaseException as error:
  clock.failure_kind=error.kind if isinstance(error,PhaseStop)else('AUTHORITY_CUTOFF'if'Current actual ROOT authority/margin'in str(error)else clock.failure_kind or'EXCEPTION')
  print(json.dumps(dict(status='FAILED_BEFORE_OWNED_EXECUTION_OR_EVIDENCE',returncode=1,phase_clock=clock.evidence(),error_type=type(error).__name__,error=str(error)[:2000],no_completion_or_automatic_retry_claim=True)),flush=True)
  return dict(status='FAILED',returncode=1)
 finally:clock.close()

def _driver_main_v1(argv=None):
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--admit-ref');g.add_argument('--execute-ref');a=p.parse_args(argv)
 if not(a.admit_ref or a.execute_ref):print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',admission=False,execution=False,guard_imported=False)));return 0
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,'Exact ROOT -B -s repository interpreter')
 result=admit(json.loads(a.admit_ref))if a.admit_ref else execute(json.loads(a.execute_ref));print(json.dumps(result,sort_keys=True),flush=True);return result.get('returncode',0)



# Both functional entry roles are implemented in these same authenticated bytes.
ORIGINAL_WORKER_REF={'path': 'research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4/worker_entry_v1.py', 'sha256': 'cc6a9b5b98034985876a88ecff6c0da6d0b16f2d23ed9195b11b4a0c0a78751b'}
EMBEDDED_ORIGINAL_WORKER_UTF8='"""Exact FLARE scientific call after owned bootstrap and genuine parent scientific release."""\nfrom pathlib import Path\nimport argparse,importlib,json,os,resource,signal,sys,time\nfrom datetime import datetime,timezone\nfrom . import root_driver_v1 as d\n\ndef compact_status(fd,value):\n data=(json.dumps(value,sort_keys=True)+\'\\n\').encode();d.need(len(data)<=4096,\'Bounded owned worker status\')\n d.need(os.write(fd,data)==len(data),\'Whole owned status-pipe frame\')\ndef grant_from_parent(fd):\n data=bytearray()\n while b\'\\n\'not in data:\n  chunk=os.read(fd,4096);d.need(chunk,\'Parent scientific gate did not authorize work\');data.extend(chunk);d.need(len(data)<=4096,\'Bounded scientific release\')\n d.need(data.endswith(b\'\\n\')and data.count(b\'\\n\')==1,\'One exact scientific release frame\');return json.loads(data)\ndef worker(record,gate_fd,ready_fd):\n d.activate();c=d.c;s,q,cfg=c.input_spec(record)\n c.require(type(gate_fd)is int and gate_fd>=0 and type(ready_fd)is int and ready_fd>=0 and ready_fd!=gate_fd,\'Two distinct actual owned startup pipes\')\n try:\n  c.require(os.read(gate_fd,1)==b\'G\',\'Original owned pidfd bootstrap gate\')\n  c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.prefix==str(c.ROOT/\'.runtime/env\')and os.environ.get(\'CUDA_VISIBLE_DEVICES\')==\'\',\'Exact qualified CPU interpreter\')\n  private=c.local(s[\'worker_auxiliary_scope\'])/\'tmp\'\n  for key in(\'TMPDIR\',\'TMP\',\'TEMP\'):c.require(os.environ.get(key)==str(private),\'Charged private TMP\')\n  c.require(set(q[\'cpu_ids\'])<=os.sched_getaffinity(0),\'Four actual CPU identities\');os.sched_setaffinity(0,set(q[\'cpu_ids\']))\n  for key in(\'OMP_NUM_THREADS\',\'NUMBA_NUM_THREADS\'):os.environ[key]=\'4\'\n  for key in(\'MKL_NUM_THREADS\',\'OPENBLAS_NUM_THREADS\'):os.environ[key]=\'1\'\n  g=c.guard(True);view=d.initial_storage_transaction(s,q,g)\n  # Initial package metadata and unchanged writer-class import belong to setup.\n  from arggnn.american_c3_common_v1 import runtime_report\n  c.require(runtime_report()==c.read(c.BASE_RUNTIME),\'Exact prequalified package report\');c.raw(c.WAIT_WRITER)\n  context=importlib.import_module(\'research.reviews.FLARE_consumer_namespace_root_20261015_v1.root_context_980_v1\')\n  c.current(s,c.CAPS[\'wall_seconds\']+600)\n  ready=dict(schema=\'FLARE_worker_bootstrap_ready_v2\',input_ref=record,pid=os.getpid(),monotonic=time.monotonic(),utc=datetime.now(timezone.utc).isoformat(),worker_source_ref=c.ref(c.ENTRY),initial_metadata_storage_and_all_roles_validated=True,actual_role_count=len(g.consumer_bindings_v1(list(view[\'bindings\'].values()),c.direct_roles(s[\'root_stage_request_ref\'],q,s[\'refs\'][\'config\'],s[\'transitive_source_closure_ref\']))))\n  compact_status(ready_fd,ready);grant=grant_from_parent(gate_fd)\n  c.require(grant.get(\'schema\')==\'FLARE_scientific_release_v2\'and grant.get(\'input_ref\')==record and grant.get(\'worker_pid\')==os.getpid(),\'Actual parent releases this exact ready worker\')\n  origin=grant[\'phase_origin_monotonic\'];deadline=grant[\'scientific_deadline_monotonic\'];now=time.monotonic()\n  c.require(type(origin)in(int,float)and type(deadline)in(int,float)and deadline-origin==c.CAPS[\'scientific_seconds\']and origin<=now<deadline,\'Original full scientific cap begins only at parent ready boundary\')\n  entry=dict(schema=\'FLARE_worker_scientific_entry_v2\',input_ref=record,pid=os.getpid(),monotonic=now,utc=datetime.now(timezone.utc).isoformat(),phase_origin_monotonic=origin,scientific_deadline_monotonic=deadline)\n  compact_status(ready_fd,entry)\n finally:\n  os.close(gate_fd);os.close(ready_fd)\n def scientific_timeout(number,frame):raise d.PhaseStop(\'WORKER_SCIENTIFIC_CUTOFF\')\n previous=signal.getsignal(signal.SIGALRM);c.require(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),\'Fresh worker scientific timer\')\n signal.signal(signal.SIGALRM,scientific_timeout);signal.setitimer(signal.ITIMER_REAL,max(0.000001,deadline-time.monotonic()))\n count=[0]\n def progress(event,**metadata):\n  if count[0]<512:\n   count[0]+=1;print(json.dumps(dict(FLARE_worker_progress=dict(event=event,parent_id=q[\'parent_id\'],stage=q[\'stage\'],monotonic=time.monotonic(),utc=datetime.now(timezone.utc).isoformat(),**metadata)),sort_keys=True),flush=True)\n def fast():\n  c.current(s,150);c.require(time.monotonic()<deadline,\'Original stage scientific deadline\')\n  c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=c.CAPS[\'rss_bytes\'],\'Original CPU RSS failstop\')\n  c.require(os.environ.get(\'CUDA_VISIBLE_DEVICES\')==\'\'and set(os.sched_getaffinity(0))==set(q[\'cpu_ids\']),\'CPU-only exact4CPU affinity\')\n class Adapter:\n  def __getattr__(self,name):return getattr(g,name)\n  def budget_check_v9(self,*args,**kwargs):\n   fast();progress(\'original_guard_boundary_begin\')\n   wall_start=time.monotonic();cpu_start=time.process_time()\n   try:value=g.budget_check_v9(*args,**kwargs)\n   except BaseException:\n    progress(\'original_guard_boundary_failed\',boundary_wall_seconds=time.monotonic()-wall_start,boundary_process_cpu_seconds=time.process_time()-cpu_start)\n    raise\n   wall_seconds=time.monotonic()-wall_start;process_cpu_seconds=time.process_time()-cpu_start\n   progress(\'original_guard_boundary_end\',boundary_wall_seconds=wall_seconds,boundary_process_cpu_seconds=process_cpu_seconds)\n   fast();return value\n try:\n  progress(\'scientific_entry\',phase_origin_monotonic=origin,scientific_deadline_monotonic=deadline)\n  class ProgressWriter(context.WaitWriter):\n   def prepare(self,name,budget_bytes):\n    progress(\'writer_prepare_begin\',product=name);p=super().prepare(name,budget_bytes);progress(\'writer_prepare_end\',product=name);return p\n   def finish(self,p,planned):\n    progress(\'writer_finish_begin\',product=p.name);r=super().finish(p,planned);progress(\'writer_finish_end\',product=p.name);return r\n  writer=ProgressWriter(s,Adapter());module=importlib.import_module(c.MODULES[q[\'stage\']])\n  c.require(c.ref(module.SOURCE)==c.ref(c.PRODUCERS[q[\'stage\']]),\'Exact unchanged original stage callable\')\n  fast();result=module.execute(cfg,writer);fast();progress(\'scientific_callable_completed\');return result\n except BaseException as error:\n  progress(\'worker_failed\',exception_type=type(error).__name__,reason=str(error)[:1000]);raise\n finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)\ndef main(argv=None):\n p=argparse.ArgumentParser();p.add_argument(\'--worker-ref\');p.add_argument(\'--gate-fd\',type=int);p.add_argument(\'--ready-fd\',type=int);a=p.parse_args(argv)\n if not a.worker_ref:print(json.dumps(dict(status=\'SOURCE_ONLY_INACTIVE\',guard_imported=False,models_imported=False)));return 0\n d.need(a.gate_fd is not None and a.ready_fd is not None,\'Actual two-phase owned worker gate\')\n result=worker(json.loads(a.worker_ref),a.gate_fd,a.ready_fd);print(json.dumps(dict(completion_ref=result),sort_keys=True),flush=True);return 0\nif __name__==\'__main__\':raise SystemExit(main())\n'
def worker_entry_same_source_v1(argv):
 need(hashlib.sha256(EMBEDDED_ORIGINAL_WORKER_UTF8.encode()).hexdigest()==ORIGINAL_WORKER_REF["sha256"],
  "Byte-exact original scientific worker embedded in same authenticated driver source")
 tree=ast.parse(EMBEDDED_ORIGINAL_WORKER_UTF8)
 imports=[n for n in tree.body if isinstance(n,ast.ImportFrom)and n.level==1 and n.module is None
  and len(n.names)==1 and n.names[0].name=="root_driver_v1"and n.names[0].asname=="d"]
 need(len(imports)==1,"Only original worker driver import binds to this same module")
 tree.body=[ast.Assign(targets=[ast.Name(id="d",ctx=ast.Store())],value=ast.Name(id="_same_authenticated_entry_module",ctx=ast.Load()))if n is imports[0]else n for n in tree.body]
 ast.fix_missing_locations(tree)
 module=types.ModuleType("_FLARE_same_source_private_original_worker")
 module.__dict__.update(__file__=str(ROOT/SOURCE),__package__=HERE.replace("/","."),
  _same_authenticated_entry_module=sys.modules[__name__])
 exec(compile(tree,str(ROOT/SOURCE)+"#embedded_original_worker","exec",dont_inherit=True,optimize=0),module.__dict__)
 return module.main(argv)
def main(argv=None):
 args=list(sys.argv[1:]if argv is None else argv)
 if any(a=="--worker-ref"or a.startswith("--worker-ref=")for a in args):
  return worker_entry_same_source_v1(args)
 return _driver_main_v1(args)
if __name__=="__main__":raise SystemExit(main())

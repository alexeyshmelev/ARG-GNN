"""Inactive fresh VAL001 conversion: authorized3h work condition, unchanged guarded converter."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,copy,ctypes,hashlib,importlib,importlib.util,json,os,signal,stat,subprocess,sys,time,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4'
SOURCE=HERE+'/root_driver_v1.py';ENTRY=HERE+'/worker_entry_v1.py'
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
 b=bindings();c=load(b['original_contract_ref'],'_FLARE_phase_original_contract');selector=load(b['selector_protocol_contract_ref'],'_FLARE_phase_selector_protocol')
 c.HERE=HERE;c.SOURCE=c.DRIVER=SOURCE;c.ENTRY=ENTRY;c.MANIFEST=MANIFEST;c.HANDOFF=HANDOFF
 c.STAGES=('convert',);c.VAL=('american_c3_val_001',)
 original_stage_caps=c.stage_caps
 def stage_caps(stage,cfg):
  caps=original_stage_caps(stage,cfg)
  c.require(stage=='convert'and caps==b['predecessor_convert_caps'],'Exact unchanged full VAL001 conversion resources')
  caps.update(scientific_seconds=10800,wall_seconds=11400)
  return caps
 c.stage_caps=stage_caps
 def selected(record):
  value=selector.selected(record);c.require(value['source_ref']==b['selected_child_source_ref']and value['selector_source_ref']==b['selected_child_source_ref']and value['accounting_mode']=='Relate_failed64GiB_retention_v1','Exact actual current child only');return value
 def bind_selector(record):
  v=selected(record);selector.bind_selector(record)
  for key in('SELECTOR_REF','_CHOSEN','SELECTOR','GUARD_REF','POLICY_REF'):setattr(c,key,copy.deepcopy(getattr(selector,key)))
  return v
 def guard(consumer):
  c.require(c.SELECTOR_REF==selector.SELECTOR_REF and c._CHOSEN==selector._CHOSEN and c.GUARD_REF==selector.GUARD_REF and c.POLICY_REF==selector.POLICY_REF,'Same selected guard lifetime')
  return selector.guard(consumer) # Existing guard(True) consumer memo unchanged; no added memo.
 c.selected=selected;c.bind_selector=bind_selector;c.guard=guard
 original_roles=c.direct_roles
 def direct_roles(request_ref,q,cfgref,closure):
  roles=original_roles(request_ref,q,cfgref,closure)
  for rr in(c.DEPS['SELECTOR'],c.GUARD_REF,q['prior_failure_acceptance_ref']):
   if rr not in roles:roles.append(copy.deepcopy(rr))
  c.require(c.GUARD_REF in roles and c.GUARD_REF==selected(q['storage_policy_selector_ref'])['source_ref'],'Exact child remains direct; no omitted static selector')
  return roles
 c.direct_roles=direct_roles
 def failure_acceptance(record):
  v=c.read(record);old=b['old_failure']
  c.require(v.get('schema')=='ROOT_FLARE_failed_unsealed_acceptance_v2'and v.get('root_authorizer')=='root_agent'and v.get('accepted')is True
   and all(type(v.get(k))is type(old[k])and v.get(k)==old[k]for k in('original_input_ref','original_request_ref','old_created_at_utc','old_expires_at_utc','old_reservation_key','actual_wait_ref','outer_wait_ref','outer_tool','outer_returncode','direct_worker_returncode','owned_drain','completed_ref','old_charge_bytes','recorded_operational_failure','actual_cancel_signal','exact_sites_written'))
   and v.get('old_launch_resume_write_authority_revoked')is True and v.get('old_files_scopes_key_charge_clocks_and_failure_unchanged')is True
   and v.get('original_conversion_unsealed')is True,'Genuine exact failed12977 acceptance, no completion/cause/credit relabel')
  for key in('original_input_ref','original_request_ref','actual_wait_ref','outer_wait_ref'):c.raw(old[key])
  latest=b['setupv3_failure'];accepted=v.get('latest_setupv3_failure_acceptance')
  c.require(isinstance(accepted,dict)and accepted.get('schema')=='ROOT_FLARE_setupv3_failed_unsealed_acceptance_v1'
   and accepted.get('root_authorizer')=='root_agent'and accepted.get('accepted')is True
   and all(type(accepted.get(k))is type(value)and accepted.get(k)==value for k,value in latest.items())
   and accepted.get('old_launch_resume_write_authority_revoked')is True
   and accepted.get('old_files_scopes_key_charge_clocks_and_failure_unchanged')is True
   and accepted.get('original_conversion_unsealed')is True,
   'Genuine exact97739outer1 worker1/drain acceptance; no old clock extension, reuse, credit or success')
  for key in('original_input_ref','original_request_ref','actual_wait_ref','outer_wait_ref','failure_diagnosis_ref'):c.raw(latest[key])
 original_validate=c.validate_request
 def validate_request(record):
  q=c.read(record)
  c.require(q.get('stage')=='convert'and q.get('parent_id')=='american_c3_val_001',
   'Only one fresh full VAL001 conversion condition; no native, seal, scoring or other parent')
  budget=phase_budget(q.get('stage'))
  c.require(q.get('storage_policy_selector_ref')==b['actual_selector_metadata_ref'],'Exact current Relate selector only')
  c.require(all(q.get(k)==value for k,value in b['fresh_namespaces'].items()),
   'Exact distinct never-reserved v4 preparation/packet/output only')
  c.require(q.get('proposed_reservation_key')==b['fresh_reservation_key']
   and b['fresh_reservation_key']=='H041/FLARE-original-convert-'+Path(q['packet_directory']).name,
   'Exact fresh key derived by unchanged admission')
  condition=b['fresh_condition']
  c.require(all(type(q.get(k))is type(value)and q.get(k)==value for k,value in condition.items()),
   'Fresh3h operational completion condition; no partial reuse or efficiency/runtime-result claim')
  for prior in b['all_failed_namespaces']:
   c.require(all(not(c.local(q[k])==c.local(v)or c.local(q[k]).is_relative_to(c.local(v))or c.local(v).is_relative_to(c.local(q[k])))
    for k in('preparation_directory','packet_directory','output_directory')for v in prior.values()),
    'Both historical failed scopes remain fully disjoint and unchanged')
  c.require(q.get('phase_budget')==budget,'Exact7200setup+10800conversion+600walloverhead=18600combined; original evidence/cleanup margins')
  for key in('preparation_directory','packet_directory','output_directory'):
   p=c.local(q[key])
   for old in b['excluded_namespaces'].values():
    o=c.local(old);c.require(p!=o and not p.is_relative_to(o)and not o.is_relative_to(p),'Every fresh namespace disjoint from every old namespace')
  failure_acceptance(q['prior_failure_acceptance_ref']);directive=c.read(q['root_directive_ref'])
  c.require(directive.get('phase_budget')==budget and directive.get('prior_failure_acceptance_ref')==q['prior_failure_acceptance_ref']
   and directive.get('proposed_reservation_key')==b['fresh_reservation_key']
   and all(type(directive.get(k))is type(value)and directive.get(k)==value for k,value in condition.items()),
   'Separate genuine ROOT directive binds fresh3h condition, exact key and both failures')
  registration=c.read(q['registration_ref']);registered=registration.get('fresh_completion_condition_registration')
  c.require(isinstance(registered,dict)and registered.get('schema')=='FLARE_fresh3h_VAL001_completion_condition_registration_v4'
   and registered.get('root_authorizer')=='root_agent'and registered.get('registered_before_request')is True
   and registered.get('source_manifest_ref')==c.ref(MANIFEST)
   and registered.get('source_review_ref')==q['source_review_ref']
   and registered.get('proposed_reservation_key')==b['fresh_reservation_key']
   and registered.get('phase_budget')==budget
   and registered.get('original_failed_input_refs')==[b['old_failure']['original_input_ref'],b['setupv3_failure']['original_input_ref']]
   and all(type(registered.get(k))is type(value)and registered.get(k)==value for k,value in condition.items()),
   'Genuine prospective fresh3h operational condition registration inside original direct target registration')
  for key in('decision_snapshot_ref','ledger_snapshot_ref'):c.raw(registered[key])
  registered_at=datetime.fromisoformat(registered['registered_utc'])
  created=datetime.fromisoformat(q['created_at_utc']);expires=datetime.fromisoformat(q['expires_at_utc'])
  c.require(registered_at.tzinfo is not None and registered_at<=created,'Fresh3h registration precedes request')
  c.require(created.tzinfo is not None and expires.tzinfo is not None and(expires-created).total_seconds()>budget['combined_seconds']+600,'Fresh finite authority covers full combined setup/work and original margin')
  return original_validate(record) # All original science/schema/stage/runtime/directive/margin predicates.
 c.validate_request=validate_request
 original_input_spec=c.input_spec
 def input_spec(record):
  result=original_input_spec(record);s,q,cfg=result
  c.require(s['storage']['reservation']['key']==b['fresh_reservation_key']
   and s['job_id']==b['fresh_reservation_key'].split('/',1)[1],
   'Actual admitted storage/job binds the exact never-reserved fresh3h key')
  return result
 c.input_spec=input_spec
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
        proc=subprocess.Popen([worker_python(q),"-B","-m",HERE.replace("/",".")+".worker_entry_v1",
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

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--admit-ref');g.add_argument('--execute-ref');a=p.parse_args(argv)
 if not(a.admit_ref or a.execute_ref):print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',admission=False,execution=False,guard_imported=False)));return 0
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,'Exact ROOT -B -s repository interpreter')
 result=admit(json.loads(a.admit_ref))if a.admit_ref else execute(json.loads(a.execute_ref));print(json.dumps(result,sort_keys=True),flush=True);return result.get('returncode',0)
if __name__=='__main__':raise SystemExit(main())

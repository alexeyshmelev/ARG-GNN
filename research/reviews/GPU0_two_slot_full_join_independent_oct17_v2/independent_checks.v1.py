"""Independent v2 whole-join verification, static metadata and RAM-only OS doubles.
No project imports, production resource/control/admission/journal/science/Git calls.
"""
import ast,copy,difflib,hashlib,json,os,pathlib,stat
from types import SimpleNamespace
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_postdrain_two_slot_pool_source_20261017_v2"
PRIOR="research/reviews/GPU0_postdrain_two_slot_pool_source_20261017_v1"
OUT=ROOT/"research/reviews/GPU0_two_slot_full_join_independent_oct17_v2"
def need(ok,message):
 if not ok:raise ValueError(message)
def raw(name):
 p=ROOT/name;s=p.lstat();need(p.resolve()==p and ROOT in p.parents and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=32*2**20,"Canonical bounded static source "+name)
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as f:opened=os.fstat(f.fileno());data=f.read(32*2**20+1);after=os.fstat(f.fileno())
 ident=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 need(ident(s)==ident(opened)==ident(after)==ident(p.lstat())and len(data)==s.st_size,"Stable source "+name);return data
def sha(data):return hashlib.sha256(data).hexdigest()
def ref(name):return dict(path=name,sha256=sha(raw(name)))
def bound(r):
 data=raw(r["path"]);need(sha(data)==r["sha256"],"Exact SHA "+r["path"]);return data
def find(tree,name):
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name];need(len(nodes)==1,"One AST "+name);return nodes[0]
def dump(n):return ast.dump(n,include_attributes=False)
manifest_ref=ref(HERE+"/source_manifest.v1.json");handoff_ref=ref(HERE+"/source_handoff.v1.json")
need(manifest_ref["sha256"]=="e8ed58689b76c0f1db73aa4517ead0220d2832f133a88a6580086b9f6d11fca8"and handoff_ref["sha256"]=="b1336b355e9be05e5bbc76048ccd4608f1ee5c41870d29da8a9203b04b1c99b6","Requested immutable v2 inputs")
manifest=json.loads(bound(manifest_ref));need(len(manifest["refs"])==253 and manifest["source_sha256"]=={r["path"]:r["sha256"]for r in manifest["refs"]},"Complete253 source refs")
total=parsed=0
for r in manifest["refs"]:
 data=bound(r);total+=len(data)
 if pathlib.Path(r["path"]).suffix==".py":ast.parse(data,filename=r["path"]);parsed+=1
packet=json.loads(raw(HERE+"/MANIFEST.v1.json"));need(len(packet["files"])==50,"Complete50 packet files");packet_bytes=0
for r in packet["files"]:
 data=bound(dict(path=r["path"],sha256=r["sha256"]));need(len(data)==r["bytes"],"Exact packet size");packet_bytes+=len(data)
need(packet_bytes==packet["total_bytes"]==364174,"Exact full packet byte count")
producer=json.loads(raw(HERE+"/producer_manifest.v1.json"));need(len(producer["refs"])==124,"Complete124 producer refs")
for r in producer["refs"]:bound(r)
source_map=json.loads(raw(HERE+"/SOURCE_MAP.v1.json"));delta_count=0
for row in source_map["files"]:
 after=bound(row["successor_ref"]).decode()
 if "predecessor_ref"in row:
  before=bound(row["predecessor_ref"]).decode()
  expected="".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=row["predecessor_ref"]["path"],tofile=row["successor_ref"]["path"]))
  need(expected==bound(row["delta_ref"]).decode(),"Mapped exact predecessor diff");delta_count+=1
old_H058="research/reviews/H058_GPU0_full_training_successor_source_20261017_v2"
for name in ("producer_v1.py","observations_gpu_v1.py","context_factory_v1.py"):
 normalized=raw(old_H058+"/"+name).decode().replace(old_H058,HERE).replace(old_H058.replace("/","."),HERE.replace("/","."))
 need(dump(ast.parse(normalized))==dump(ast.parse(raw(HERE+"/"+name))),"Original H058 scientific/context AST "+name)
unchanged=[]
for name in ("__init__.py","common_v1.py","direct_gpu0_v1.py","context_factory_v1.py","observations_gpu_v1.py","producer_v1.py","root_driver_v1.py","pool_contract_v1.py","pool_admission_v1.py","native_attachment_seam_v1.py"):
 normalized=raw(PRIOR+"/"+name).decode().replace(PRIOR,HERE).replace(PRIOR.replace("/","."),HERE.replace("/","."))
 need(dump(ast.parse(normalized))==dump(ast.parse(raw(HERE+"/"+name))),"Entire prior reviewed component retained "+name);unchanged.append(name)
need(ref(PRIOR+"/source_manifest.v1.json")["sha256"]=="396e656eab6196110d26427667191cbaed129e8fb7273c16f52ed02bb7752b8c"and ref(PRIOR+"/source_handoff.v1.json")["sha256"]=="dc7cd9408be5819bc8a180884f072c634501293db07f3163535a0673cdf5205a","V1 remains immutable")
pooltext=raw(HERE+"/pool_v1.py").decode();pooltree=ast.parse(pooltext)
legacy_name="src/arggnn/gpu45_c3cache_h050_pool_v1.py";legacytext=raw(legacy_name).decode();legacytree=ast.parse(legacytext);legacy_class=find(legacytree,"Shared45Controller")
sharedtree=ast.parse(raw("research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_shared_pool_v1.py"));shared_class=find(sharedtree,"Shared0Controller")
clock=[100.0];ACTIVE={"launching","running","draining"};fake_time=SimpleNamespace(time=lambda:clock[0])
uuid="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
external=dict(pid=999999,name="preserved external work",used_mib=51200)
gpu=dict(index=0,uuid=uuid,total_mib=81920,used_mib=51200,compute=[external])
memory_ns=dict(base=SimpleNamespace(checked_inventory=lambda inventory:[inventory()]),require=need,GPU0_UUID=uuid,ACTIVE=ACTIVE,TITLE="AI Auto Research Pipeline Test")
for node in (find(sharedtree,"stale_compute_classification_v1"),find(shared_class,"_memory")):
 exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),"<v2_independent_original_memory_RAM>","exec"),memory_ns)
context={"app":None}
def fake_waitpid(pid,options):
 app=context["app"];need(pid==333 and options==1 and not app.adopted_reaped,"Only exact RAM adopted child wait")
 app.adopted_reaped=True;app.adopted_OS_wait_calls+=1;return(pid,0)
fake_os=SimpleNamespace(getpid=lambda:111,WNOHANG=1,waitpid=fake_waitpid,waitstatus_to_exitcode=lambda status:0)
tick_ns=dict(time=fake_time,ACTIVE=ACTIVE,os=fake_os,require=need)
for name in ("tick","_fail_owned_supervision","_service_termination","_reap_owned_adopted"):
 exec(compile(ast.Module(body=[copy.deepcopy(find(legacy_class,name))],type_ignores=[]),"<v2_independent_original_tick_RAM>","exec"),tick_ns)
class FakeShared:
 _memory=memory_ns["_memory"];tick=tick_ns["tick"];_fail_owned_supervision=tick_ns["_fail_owned_supervision"];_service_termination=tick_ns["_service_termination"]
 def _lease_check(self):pass
 def _drain_logs(self):pass
 def _process_scan(self):
  self.scan_calls+=1;self.pid_jobs={}if self.child.exited else{222:"H058"}
  self.processes={}if self.child.exited else{222:dict(pid=222,uid=7,ppid=111,state="S",start_ticks=1000)}
  if not self.adopted_reaped:self.processes[333]=dict(pid=333,uid=7,ppid=111,state="Z",start_ticks=1001)
 def _reap_owned_adopted(self):
  self.adopted_passes+=1;return tick_ns["_reap_owned_adopted"](self)
 def _persist(self):self.persist_calls+=1
 def _signal_job(self,name):self.signal_names.append(name)
spy=SimpleNamespace(popen_calls=0)
def forbidden_popen(*args,**kwargs):spy.popen_calls+=1;raise AssertionError("RAM fixture must never launch")
legacy=SimpleNamespace(ACTIVE=ACTIVE,subprocess=SimpleNamespace(Popen=forbidden_popen))
reviewed=SimpleNamespace(legacy=legacy,base=SimpleNamespace(),Shared0Controller=FakeShared)
sources={"down":raw("research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py"),"old":raw(legacy_name)}
fake_h=SimpleNamespace(source_refs_v1=lambda:None,load=lambda *args:reviewed,raw=lambda r:sources[r],POLICY={})
admission_spy=[]
def forbidden_submission(*args,**kwargs):admission_spy.append("called");raise AssertionError("Closed/draining fixture must reject before input/admission")
fake_c=SimpleNamespace(SHARED="shared",DOWN="down",OLD="old",guard_v1=lambda:None,guarded_check_v1=lambda *args:None,validate_cap_v1=lambda *args,**kwargs:None,validate_submission_v1=forbidden_submission)
factory_ns=dict(h=fake_h,c=fake_c,ROOT=ROOT,MODULE="exact.future.module",require=need,ast=ast,os=None,time=fake_time,json=json,types=None,_verify_new_exclusive_v1=lambda *args:None,_open_old_inode_v1=lambda *args:None)
factory=find(pooltree,"build_controller_class_v1")
exec(compile(ast.Module(body=[copy.deepcopy(factory)],type_ignores=[]),"<v2_independent_exact_factory_RAM>","exec"),factory_ns)
cls,_=factory_ns["build_controller_class_v1"]()
# Independently check exact generated tick, including every wait/exception call.
original_tick=ast.get_source_segment(legacytext,find(legacy_class,"tick"))
replacement='                if self.lifecycle=="running":self.safety_error=None'
expected_tick=original_tick.replace("                self.safety_error=None",replacement)
need(original_tick.count("                self.safety_error=None")==1,"Exactly one original reset site")
compiled={};exec(compile(expected_tick,cls.tick.__code__.co_filename,"exec"),compiled)
need(compiled["tick"].__code__==cls.tick.__code__,"Actual generated tick matches independent exact one-change source")
restored=ast.parse(expected_tick);matches=[]
for n in ast.walk(restored):
 if isinstance(n,ast.If)and ast.unparse(n.test)=="self.lifecycle == 'running'"and len(n.body)==1 and ast.unparse(n.body[0])=="self.safety_error = None":matches.append(n)
need(len(matches)==1,"One added lifecycle predicate")
for parent in ast.walk(restored):
 for field,values in ast.iter_fields(parent):
  if isinstance(values,list):
   for i,value in enumerate(values):
    if value is matches[0]:values[i]=value.body[0]
need(dump(restored)==dump(ast.parse(original_tick)),"All original tick AST recovered after removing only reset predicate")
class NoMutex:
 def __enter__(self):return self
 def __exit__(self,*args):return False
class FakeChild:
 pid=222;stdout=None
 def __init__(self,returns):self.returns=list(returns);self.poll_calls=0;self.exited=False
 def poll(self):
  self.poll_calls+=1;result=self.returns.pop(0);self.exited=result is not None;return result
class FakeLog:
 def __init__(self):self.closed=False
 def close(self):self.closed=True
def make_app(returns):
 app=object.__new__(cls);child=FakeChild(returns);app.child=child
 app.lock=NoMutex();app.lifecycle="running";app.safety_error=None;app.terminate_requested=False
 app.persist_calls=app.scan_calls=app.adopted_passes=app.adopted_OS_wait_calls=0;app.signal_names=[];app.adopted_reaped=False;app.orphan_waits=[]
 app.cap=dict(expires_unix=1000,drain_expires_unix=2000,maximum_concurrent_jobs=2)
 app.jobs={"H058":dict(logical_job="H058",state="running",requested=dict(peak_mib=30720),process_identities=[dict(pid=222,start_ticks=1000),dict(pid=333,start_ticks=1001)],descendant_waits=[]),"queued":dict(logical_job="queued",state="queued",requested=dict(peak_mib=30720))}
 app.children={"H058":child};app.logs={"H058":FakeLog()};app.pid_jobs={};app.processes={};app.ambiguous=[];app.metadata=dict(uuid=uuid);app.device=dict(used_mib=0);app.headroom=4096;app.inventory=lambda:gpu
 context["app"]=app;return app,child
external_before=copy.deepcopy(gpu);reports=[]
for label,returns in (("already_exited_zero",[0]),("alive_then_exited_zero",[None,None,0]),("alive_then_SIGTERM",[None,-15])):
 app,child=make_app(returns);snapshots=[]
 for i in range(len(returns)):
  app.tick(True);clock[0]+=1
  snapshots.append(dict(tick=i,poll_calls=child.poll_calls,children=len(app.children),lifecycle=app.lifecycle,job_state=app.jobs["H058"]["state"],signal_names=list(app.signal_names),external_used_mib=gpu["used_mib"],safety_error=app.safety_error))
  need(child.poll_calls==i+1,"Every blocked tick reaches actual inherited leader poll")
  need(app.jobs["queued"]["state"]=="cancelled"and app.safety_error.startswith("Periodic active pool capacity failed:"),"Queue cancelled and cause sticky throughout drain")
  if app.lifecycle=="draining":
   rejected=False
   try:app.submit_postdrain(None)
   except ValueError:rejected=True
   need(rejected and not admission_spy,"Draining cannot admit fresh work")
 need(not app.children and app.lifecycle=="closed"and app.jobs["H058"]["state"]=="failed"and app.jobs["H058"]["leader_wait_status_known"]is True and app.jobs["H058"]["owned_drain_observed"]is True,"Original genuine wait/drain finalized as failed")
 need(app.adopted_OS_wait_calls==1 and len(app.orphan_waits)==1 and app.orphan_waits[0]["wait_status_known"]is True and app.jobs["H058"]["descendant_waits"]==app.orphan_waits,"Original adopted-child wait body records exact known wait")
 need(app.jobs["H058"]["exit_code"]==returns[-1]and all(n=="H058"for n in app.signal_names),"Exact actual leader returncode; owned-only stop request")
 need(gpu==external_before,"External occupation/process rows untouched")
 incoming=app._memory(30720);need(incoming["blocked_by"]==["actual_free_memory_or_reserved_headroom"]and incoming["external_compute"]==[external],"Unchanged incoming global capacity still rejects")
 app._launch_queue();need(spy.popen_calls==0,"Closed queue cannot reach Popen")
 rejected=False
 try:app.submit_postdrain(None)
 except ValueError:rejected=True
 need(rejected and not admission_spy,"Closed cannot admit fresh work")
 reports.append(dict(case=label,ticks=snapshots,actual_adopted_waits=copy.deepcopy(app.orphan_waits),incoming_still_blocked=True,closed_submit_rejected=True,queue_Popen_calls=spy.popen_calls))
gpu.update(used_mib=0,compute=[]);before_polls=child.poll_calls;app.tick(True);app._launch_queue()
need(child.poll_calls==before_polls and spy.popen_calls==0 and app.lifecycle=="closed"and app.jobs["queued"]["state"]=="cancelled","Later capacity recovery cannot relaunch failed authority")
contract=ast.parse(raw(HERE+"/pool_contract_v1.py"));seam=ast.parse(raw(HERE+"/native_attachment_seam_v1.py"))
need("execution_enabled') is False"in ast.unparse(contract)or "execution_enabled'] is False"in ast.unparse(contract),"Native remains unconditionally disabled")
need(not any(isinstance(n,ast.Call)and ast.unparse(n.func)in("subprocess.Popen","os.system","os.execv","os.execve")for n in ast.walk(seam)),"No native runner or launch")
result=dict(schema="GPU0_postdrain_v2_independent_whole_join_checks_v1",status="SOURCE_PASS_RAM_STATIC_ONLY",source_manifest_ref=manifest_ref,source_handoff_ref=handoff_ref,source_refs_verified=253,static_bytes_verified=total,python_ASTs_parsed=parsed,packet_files_verified=50,packet_bytes_verified=packet_bytes,producer_refs_verified=124,mapped_predecessor_deltas_verified=delta_count,previous_reviewed_components_namespace_AST_equal=unchanged,original_H058_scientific_context_AST_preserved=True,v1_inputs_unchanged=True,actual_factory_start_queue_tick_compile=True,actual_derived_tick_matches_independent_one_change_code=True,entire_original_tick_AST_restored=True,persistent_capacity_repair_cases=reports,actual_original_adopted_wait_body_executed_only_on_RAM_OS_double=True,admission_rejected_while_draining_and_closed=True,incoming_capacity_still_rejected=True,capacity_recovery_cannot_relaunch=True,native_slot_unconditionally_disabled=True,project_modules_imported=False,real_process_signal_resource_mutex_guard_probe_admission_journal_Git_operations=False,scientific_or_numerical_execution=False)
p=OUT/"independent_checks.v1.json"
with p.open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(dict(status=result["status"],result_ref=ref(str(p.relative_to(ROOT))),cases=len(reports),source_refs_verified=253,packet_files_verified=50)))

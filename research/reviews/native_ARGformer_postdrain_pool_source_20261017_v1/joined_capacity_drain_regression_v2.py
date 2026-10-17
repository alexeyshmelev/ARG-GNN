"""Author joined RAM-only capacity/drain regression, not an independent verdict.
No target imports, scientific/model execution or real resource/control operations.
"""
import ast,copy,difflib,hashlib,json,os,pathlib,stat
from types import SimpleNamespace
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_ARGformer_postdrain_pool_source_20261017_v1"
OUT=ROOT/HERE
def need(ok,message):
 if not ok:raise ValueError(message)
def raw(name):
 p=ROOT/name;s=p.lstat();need(p.resolve()==p and ROOT in p.parents and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=32*2**20,"Exact static source/metadata "+name)
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as f:opened=os.fstat(f.fileno());data=f.read(32*2**20+1);after=os.fstat(f.fileno())
 ident=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 need(ident(s)==ident(opened)==ident(after)==ident(p.lstat())and len(data)==s.st_size,"Stable static bytes "+name);return data
def digest(data):return hashlib.sha256(data).hexdigest()
def ref(name):return dict(path=name,sha256=digest(raw(name)))
def bound(r):
 data=raw(r["path"]);need(digest(data)==r["sha256"],"Bound SHA "+r["path"]);return data
def find(tree,name):
 x=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name];need(len(x)==1,"Single "+name);return x[0]
pooltree=ast.parse(raw(HERE+"/pool_v1.py"));sharedtree=ast.parse(raw("research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_shared_pool_v1.py"));legacytree=ast.parse(raw("src/arggnn/gpu45_c3cache_h050_pool_v1.py"))
clock=[100.0];ACTIVE={"launching","running","draining"};fake_time=SimpleNamespace(time=lambda:clock[0])
legacy_class=find(legacytree,"Shared45Controller");shared_class=find(sharedtree,"Shared0Controller")
uuid="GPU-ff06ca1f-403a-f2d0-b584-71e177452667";gpu=dict(index=0,uuid=uuid,total_mib=81920,used_mib=51200,compute=[dict(pid=999999,name="external preserved job",used_mib=51200)])
memory_ns=dict(base=SimpleNamespace(checked_inventory=lambda inventory:[inventory()]),require=need,GPU0_UUID=uuid,ACTIVE=ACTIVE,TITLE="AI Auto Research Pipeline Test")
for n in (find(sharedtree,"stale_compute_classification_v1"),find(shared_class,"_memory")):
 exec(compile(ast.Module(body=[copy.deepcopy(n)],type_ignores=[]),"<independent_RAM_original_memory>","exec"),memory_ns)
tick_ns=dict(time=fake_time,ACTIVE=ACTIVE)
for name in ("tick","_fail_owned_supervision","_service_termination"):
 exec(compile(ast.Module(body=[copy.deepcopy(find(legacy_class,name))],type_ignores=[]),"<independent_RAM_original_tick>","exec"),tick_ns)
class FakeShared:
 _memory=memory_ns["_memory"]
 tick=tick_ns["tick"]
 _fail_owned_supervision=tick_ns["_fail_owned_supervision"]
 _service_termination=tick_ns["_service_termination"]
 def _lease_check(self):pass
 def _drain_logs(self):pass
 def _process_scan(self):self.scan_calls+=1
 def _reap_owned_adopted(self):self.adopted_wait_calls+=1
 def _persist(self):self.persist_calls+=1
 def _signal_job(self,name):self.signal_calls+=1;self.signal_names.append(name)
legacy=SimpleNamespace(ACTIVE=ACTIVE);reviewed=SimpleNamespace(legacy=legacy,base=SimpleNamespace(Controller=type("FakeOriginalBaseController",(),{})),Shared0Controller=FakeShared)
sources={"base":(ROOT/"src/arggnn/gpu_pool_v1.py").read_bytes(),"down":raw("research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py"),"old":raw("src/arggnn/gpu45_c3cache_h050_pool_v1.py")}
fake_h=SimpleNamespace(BASE_POOL="base",source_refs_v1=lambda:None,load=lambda *a:reviewed,raw=lambda r:sources[r],POLICY={})
fake_c=SimpleNamespace(SHARED="shared",DOWN="down",OLD="old",guard_v1=lambda:None,guarded_check_v1=lambda *a:None,validate_cap_v1=lambda *a,**k:None)
factory_ns=dict(h=fake_h,c=fake_c,ROOT=ROOT,MODULE="exact.future.module",require=need,ast=ast,os=None,time=fake_time,json=json,types=None,_verify_new_exclusive_v1=lambda *a:None,_open_old_inode_v1=lambda *a:None)
factory=find(pooltree,"build_controller_class_v1")
exec(compile(ast.Module(body=[copy.deepcopy(factory)],type_ignores=[]),"<independent_RAM_factory>","exec"),factory_ns)
cls,_=factory_ns["build_controller_class_v1"]()
class NoMutex:
 def __enter__(self):return self
 def __exit__(self,*a):return False
class FakeChild:
 def __init__(self,returns):self.poll_calls=0;self.stdout=None;self.returns=list(returns)
 def poll(self):self.poll_calls+=1;return self.returns.pop(0)
class FakeLog:
 def close(self):pass

def application(returns,queued=True):
 child=FakeChild(returns);app=object.__new__(cls);job="H058"
 app.lock=NoMutex();app.lifecycle="running";app.safety_error=None;app.terminate_requested=False
 app.persist_calls=app.signal_calls=app.scan_calls=app.adopted_wait_calls=0;app.signal_names=[]
 app.cap=dict(expires_unix=1000,drain_expires_unix=2000,maximum_concurrent_jobs=2)
 app.jobs={job:dict(logical_job=job,state="running",requested=dict(peak_mib=30720),process_identities=[],descendant_waits=[])}
 if queued:app.jobs["queued_H058"]=dict(logical_job="queued_H058",state="queued",requested=dict(peak_mib=30720))
 app.children={job:child};app.logs={job:FakeLog()};app.pid_jobs={};app.processes={};app.ambiguous=[]
 app.metadata=dict(uuid=uuid);app.device=dict(used_mib=0);app.headroom=4096;app.inventory=lambda:gpu
 return app,child
external_before=copy.deepcopy(gpu);app,child=application([0])
app.tick(True)
need(child.poll_calls==1 and not app.children and app.lifecycle=="closed","Exited owned leader drained in same persistent-capacity tick")
need(app.jobs["H058"]["leader_wait_status_known"] is True and app.jobs["H058"]["owned_drain_observed"] is True and app.jobs["H058"]["state"]=="failed","Genuine wait0 plus resource abort remains failed")
need(app.jobs["queued_H058"]["state"]=="cancelled" and app.safety_error.startswith("Periodic active pool capacity failed:"),"Sticky fail-stop cancels queue and preserves capacity cause")
need(app.adopted_wait_calls>=2 and app.scan_calls>=2,"Original discovery/adopted wait/drain path reached")
need(gpu==external_before and app.signal_names==[],"Unrelated external inventory preserved; exited child needs no signal")
immediate=dict(poll_calls=child.poll_calls,children=len(app.children),lifecycle=app.lifecycle,job_state=app.jobs["H058"]["state"],safety_error=app.safety_error,queued_state=app.jobs["queued_H058"]["state"],external_used_mib=gpu["used_mib"])
app,child=application([None,None,0]);snapshots=[]
for i in range(3):
 app.tick(True)
 snapshots.append(dict(tick=i,poll_calls=child.poll_calls,children=len(app.children),lifecycle=app.lifecycle,job_state=app.jobs["H058"]["state"],signals=app.signal_calls,safety_error=app.safety_error,external_used_mib=gpu["used_mib"]))
 clock[0]+=1
need([r["poll_calls"] for r in snapshots]==[1,2,3] and snapshots[-1]["children"]==0 and snapshots[-1]["lifecycle"]=="closed","Poll progresses during every persistent external-capacity tick, including exit after drain")
need(app.signal_calls>0 and set(app.signal_names)=={"H058"},"Only exact owned logical job termination requested")
need(gpu==external_before and app.jobs["queued_H058"]["state"]=="cancelled","No external mutation or queued launch")
# The global admission accounting still rejects a new30GiB slot against the
# same external50GiB plus4GiB headroom, even after the owned worker drained.
remaining=app._memory(30720)
need(remaining["blocked_by"]==["actual_free_memory_or_reserved_headroom"],"Unweakened global/incoming capacity after drain")
need(remaining["external_compute"]==external_before["compute"],"Complete external usage still charged")
# Removing unrelated occupancy after closure cannot resume or relaunch failed work.
gpu.update(used_mib=0,compute=[]);app.tick(True)
need(app.lifecycle=="closed" and app.jobs["queued_H058"]["state"]=="cancelled" and child.poll_calls==3,"Capacity recovery cannot revive closed failed authority")
result=dict(schema="GPU0_postdrain_capacity_drain_joined_regression_v2",status="PASS_RAM_ONLY",source_ref=ref(HERE+"/pool_v1.py"),unchanged_original_memory_ref=ref("research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_shared_pool_v1.py"),unchanged_original_tick_and_termination_ref=ref("src/arggnn/gpu45_c3cache_h050_pool_v1.py"),reviewer_blocker_ref=ref("research/reviews/GPU0_two_slot_full_join_independent_oct17_v1/independent_checks.v1.json"),exact_v2_factory_compiled_with_actual_inherited_memory_tick_failure_and_termination_bodies=True,immediate_exited_child_persistent_capacity=immediate,live_then_exited_child_persistent_capacity=snapshots,owned_wait_and_adopted_wait_progress=True,owned_termination_only=True,queue_cancelled_and_launch_failstop=True,sticky_failure_cause_preserved=True,global_admission_capacity_not_weakened=True,external_processes_and_usage_preserved=True,capacity_recovery_cannot_relaunch=True,production_modules_imported=False,real_process_signal_mutex_guard_device_admission_journal_Git_operations=False,scientific_or_numerical_execution=False)
with (OUT/"joined_capacity_drain_regression.final.v2.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=result["status"],result_ref=ref(HERE+"/joined_capacity_drain_regression.final.v2.json"),immediate=immediate,snapshots=snapshots),sort_keys=True))

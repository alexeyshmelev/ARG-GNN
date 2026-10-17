"""Independent static closure and RAM-only inherited-drain reproduction.
No target module import, production callable invocation, mutex, process, device,
admission, storage/journal mutation or scientific/numerical execution.
"""
import ast,copy,difflib,hashlib,json,os,pathlib,stat
from types import SimpleNamespace
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_postdrain_two_slot_pool_source_20261017_v1"
OUT=ROOT/"research/reviews/GPU0_two_slot_full_join_independent_oct17_v1"
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
manifest_ref=ref(HERE+"/source_manifest.v1.json");handoff_ref=ref(HERE+"/source_handoff.v1.json")
need(manifest_ref["sha256"]=="396e656eab6196110d26427667191cbaed129e8fb7273c16f52ed02bb7752b8c" and handoff_ref["sha256"]=="dc7cd9408be5819bc8a180884f072c634501293db07f3163535a0673cdf5205a","Exact requested packet")
manifest=json.loads(bound(manifest_ref));need(manifest["source_sha256"]=={r["path"]:r["sha256"]for r in manifest["refs"]}and len(manifest["refs"])==248,"Exact complete248-ref map")
total=parsed=0
for r in manifest["refs"]:
 data=bound(r);total+=len(data)
 if pathlib.Path(r["path"]).suffix==".py":ast.parse(data,filename=r["path"]);parsed+=1
packet=json.loads(raw(HERE+"/MANIFEST.v1.json"));need(len(packet["files"])==46,"Exact46-file packet");packet_bytes=0
for r in packet["files"]:
 data=bound(dict(path=r["path"],sha256=r["sha256"]));need(len(data)==r["bytes"],"Exact packet byte count");packet_bytes+=len(data)
need(packet_bytes==packet["total_bytes"]==340300,"Complete sealed packet bytes")
source_map=json.loads(raw(HERE+"/SOURCE_MAP.v1.json"))
delta_results=[]
for row in source_map["files"]:
 successor=row["successor_ref"];after=bound(successor).decode()
 if "predecessor_ref"in row:
  before=bound(row["predecessor_ref"]).decode();patch=bound(row["delta_ref"]).decode()
  expected="".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=row["predecessor_ref"]["path"],tofile=successor["path"]))
  need(expected==patch,"Exact mapped source delta "+successor["path"]);delta_results.append(successor["path"])
old="research/reviews/H058_GPU0_full_training_successor_source_20261017_v2"
for name in ("producer_v1.py","observations_gpu_v1.py","context_factory_v1.py"):
 source=raw(old+"/"+name).decode().replace(old,HERE).replace(old.replace("/","."),HERE.replace("/","."))
 need(ast.dump(ast.parse(source),include_attributes=False)==ast.dump(ast.parse(raw(HERE+"/"+name)),include_attributes=False),"Scientific/context AST preserved "+name)
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
 def _process_scan(self):pass
 def _reap_owned_adopted(self):pass
 def _persist(self):self.persist_calls+=1
 def _signal_job(self,name):self.signal_calls+=1
legacy=SimpleNamespace(ACTIVE=ACTIVE);reviewed=SimpleNamespace(legacy=legacy,base=SimpleNamespace(),Shared0Controller=FakeShared)
sources={"down":raw("research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py"),"old":raw("src/arggnn/gpu45_c3cache_h050_pool_v1.py")}
fake_h=SimpleNamespace(source_refs_v1=lambda:None,load=lambda *a:reviewed,raw=lambda r:sources[r],POLICY={})
fake_c=SimpleNamespace(SHARED="shared",DOWN="down",OLD="old",guard_v1=lambda:None,guarded_check_v1=lambda *a:None,validate_cap_v1=lambda *a,**k:None)
factory_ns=dict(h=fake_h,c=fake_c,ROOT=ROOT,MODULE="exact.future.module",require=need,ast=ast,os=None,time=fake_time,json=json,types=None,_verify_new_exclusive_v1=lambda *a:None,_open_old_inode_v1=lambda *a:None)
factory=find(pooltree,"build_controller_class_v1")
exec(compile(ast.Module(body=[copy.deepcopy(factory)],type_ignores=[]),"<independent_RAM_factory>","exec"),factory_ns)
cls,_=factory_ns["build_controller_class_v1"]()
class NoMutex:
 def __enter__(self):return self
 def __exit__(self,*a):return False
class FakeChild:
 def __init__(self):self.poll_calls=0;self.stdout=None
 def poll(self):self.poll_calls+=1;return 0
class FakeLog:
 def close(self):pass
child=FakeChild();app=object.__new__(cls);job="H058"
app.lock=NoMutex();app.lifecycle="running";app.safety_error=None;app.terminate_requested=False;app.persist_calls=app.signal_calls=0
app.cap=dict(expires_unix=1000,drain_expires_unix=2000,maximum_concurrent_jobs=2)
app.jobs={job:dict(logical_job=job,state="running",requested=dict(peak_mib=30720),process_identities=[],descendant_waits=[])}
app.children={job:child};app.logs={job:FakeLog()};app.pid_jobs={};app.processes={};app.ambiguous=[];app.metadata=dict(uuid=uuid);app.device=dict(used_mib=0);app.headroom=4096;app.inventory=lambda:gpu
snapshots=[]
for i in range(3):
 app.tick(False);snapshots.append(dict(tick=i,lifecycle=app.lifecycle,job_state=app.jobs[job]["state"],poll_calls=child.poll_calls,children=len(app.children),signals=app.signal_calls,safety_error=app.safety_error));clock[0]+=1
need(child.poll_calls==0 and len(app.children)==1 and app.lifecycle=="draining"and app.signal_calls>0,"Persistent capacity loss blocks genuine wait/poll despite owned signals")
gpu.update(used_mib=0,compute=[])
app.tick(False)
need(child.poll_calls==1 and not app.children and app.lifecycle=="closed","Wait/drain resumes only after unrelated external capacity recovers")
contract=ast.parse(raw(HERE+"/pool_contract_v1.py"));seam=ast.parse(raw(HERE+"/native_attachment_seam_v1.py"))
need("execution_enabled') is False"in ast.unparse(contract)or "execution_enabled'] is False"in ast.unparse(contract),"Native capability cannot enable")
need(not any(isinstance(n,ast.Call)and ast.unparse(n.func)in("subprocess.Popen","os.system","os.execv","os.execve")for n in ast.walk(seam)),"No native launch")
result=dict(schema="GPU0_postdrain_independent_full_join_checks_v1",status="SOURCE_BLOCKER_REPRODUCED_RAM_ONLY",source_manifest_ref=manifest_ref,source_handoff_ref=handoff_ref,source_refs_verified=248,static_bytes_verified=total,python_ASTs_parsed=parsed,packet_files_verified=46,packet_bytes_verified=packet_bytes,mapped_H058_deltas_verified=len(delta_results),scientific_context_namespace_AST_preserved=True,actual_factory_extracted_start_and_queue_compile=True,production_modules_imported=False,real_resource_process_mutex_admission_journal_Git_operations=False,numerical_scientific_execution=False,reproduction=dict(scenario="Owned H058 child has exited; unrelated preserved external50GiB on80GiB leaves insufficient30GiB reservation+4GiB headroom. Actual inherited memory/tick plus exact new factory execute only against RAM doubles.",persistent_capacity_ticks=snapshots,after_external_capacity_recovers=dict(poll_calls=child.poll_calls,children=len(app.children),lifecycle=app.lifecycle),conclusion="New _memory exception prevents inherited tick reaching child.poll on every draining tick while external capacity remains blocked. Owned signals proceed, but genuine wait/drain/final close cannot progress independently of unrelated capacity."),findings=[dict(priority="P1",path=HERE+"/pool_v1.py",line=find(find(factory,"PostdrainController"),"_memory").lineno,title="Keep actual wait/drain progress after periodic capacity failure")])
with (OUT/"independent_checks.v1.json").open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(dict(result_path=str(OUT/"independent_checks.v1.json"),sha256=digest((OUT/"independent_checks.v1.json").read_bytes()),status=result["status"],snapshots=snapshots)))

"""Independent bounded source/RAM tests. No production imports or OS probes."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
PACKET=ROOT/"research/reviews/native_ARGformer_postdrain_pool_source_20261017_v2"
OUT=Path(__file__).parent
CASES=[]
def need(ok,message):
 if not ok:raise ValueError(message)
def checked(name,fn):
 fn();CASES.append(name)
def rejects(call):
 try:call()
 except (ValueError,KeyError,TypeError):return True
 return False
def extract(name,names,ns):
 tree=ast.parse((PACKET/name).read_text());nodes=[copy.deepcopy(n)for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 need({n.name for n in nodes}==set(names),"Exact extraction")
 exec(compile(ast.Module(body=nodes,type_ignores=[]),"<independent exact source>","exec"),ns);return ns
registry=json.loads((PACKET/"parent_registry.v1.json").read_text())
slots={"H058":dict(cuda_cap_bytes=28*2**30,margin_mib=2048,peak_mib=30720),"native":dict(cuda_cap_bytes=32*2**30,margin_mib=2048,peak_mib=34816)}
ns=extract("pool_contract_v1.py",["slot_spec_v1","validate_native_roster_v1"],dict(require=need,read=lambda _:registry,ref=lambda x:x,h=SimpleNamespace(HERE="RAM"),SLOTS=slots))
phases=["contrastive","embedding_contrastive_train","embedding_contrastive_val","embedding_query_paths_train","embedding_query_paths_val","RAM_retrieve_and_seal","finalize"]
jobs={"H058":dict(worker_kind="H058")}
for parent in registry["parents"]:
 for phase in phases+(["mlm"]if parent in registry["remaining_fresh_fit_parents"]else []):
  jobs[parent+"--"+phase]=dict(worker_kind="native",parent_id=parent,native_phase=phase,resource_profile="explicit_unqualified_full_stage_32GiB")
def roster():
 need(len(jobs)==40,"Full four plus35 downstream plus H058")
 ns["validate_native_roster_v1"](jobs)
 for change in ("VAL000mlm","duplicate","noH058","wrongParent","unknownPhase"):
  j=copy.deepcopy(jobs)
  if change=="VAL000mlm":j["bad"]=dict(worker_kind="native",parent_id="american_c3_val_000",native_phase="mlm")
  elif change=="duplicate":j["bad"]=dict(j["american_c3_val_001--mlm"])
  elif change=="noH058":j.pop("H058")
  elif change=="wrongParent":j["american_c3_val_001--mlm"]["parent_id"]="foreign"
  else:j["american_c3_val_001--mlm"]["native_phase"]="retry"
  need(rejects(lambda:ns["validate_native_roster_v1"](j)),change)
checked("full40_unique_roster_and_five_negative_conditions",roster)
uid=123;uuid="RAM-GPU0";clock=SimpleNamespace(time=lambda:100.,monotonic=lambda:100.)
inventory=dict(index=0,uuid=uuid,total_mib=81920,used_mib=42288,compute=[dict(pid=1,name="AI Auto Research Pipeline Test",used_mib=10000),dict(pid=2,name="AI Auto Research Pipeline Test",used_mib=20000),dict(pid=9,name="external",used_mib=12288)])
peer_jobs=[dict(logical_job="H058",state="running",requested=dict(peak_mib=30720),process_identities=[dict(pid=1,start_ticks=11,uid=uid)]),dict(logical_job="N",state="running",requested=dict(peak_mib=34816),process_identities=[dict(pid=2,start_ticks=22,uid=uid)])]
def identity(pid):return dict(pid=pid,start_ticks=pid*11,uid=uid,state="S")
base=SimpleNamespace(real_inventory=lambda:[copy.deepcopy(inventory)],proc_identity=identity)
for file in ("native_direct_gpu0_v1.py","downstream_direct_gpu0_v1.py"):
 cn=extract(file,["GPUCapacityFailureV3","bounded_capacity_failure_v3","capacity_v1"],dict(require=need,GPU0_UUID=uuid,time=clock,os=SimpleNamespace(getuid=lambda:uid),json=json,controller_v1=lambda q,b:(None,dict(jobs=peer_jobs),None,None)))
 def capacity():
  q=dict(job_id="N",postdrain_pool_capability_ref="RAM");snapshot=copy.deepcopy(inventory)
  r=cn["capacity_v1"](q,base);need(r["metrics"]["projected_mib"]==81920 and r["metrics"]["actual_free_required_mib"]==39632,"Exact80GiB boundary including all unfilled reservations")
  inventory["used_mib"]+=1;inventory["compute"][-1]["used_mib"]+=1
  try:cn["capacity_v1"](q,base);raise AssertionError("NextMiB must fail")
  except cn["GPUCapacityFailureV3"]as e:need(e.capacity_sample["computed_metrics"]["projected_mib"]==81921,"Real failed decision bound")
  inventory.update(copy.deepcopy(snapshot))
  inventory["compute"][0]["name"]="wrong title";need(rejects(lambda:cn["capacity_v1"](q,base)),"Owned title")
  inventory.update(copy.deepcopy(snapshot));inventory["uuid"]="wrong";need(rejects(lambda:cn["capacity_v1"](q,base)),"GPU0 UUID")
  inventory.update(copy.deepcopy(snapshot));need(inventory==snapshot,"External data preserved")
 checked(file+":actual_capacity_boundary_nextMiB_title_UUID",capacity)
# Direct-role union checks exercise actual code against RAM references.
for name in ("common_v1.py","native_common_v1.py"):
 u=extract(name,["unique"],dict(require=need,ref=lambda path:dict(path=path,sha256=path)))
 rows=[dict(path=str(i),sha256=str(i))for i in range(64)]
 need(len(u["unique"](rows+rows))==64,"Duplicates preserve original64")
 need(rejects(lambda:u["unique"](rows+[dict(path="65",sha256="65")])),"65 rejected")
 need(rejects(lambda:u["unique"](rows+[dict(path="1",sha256="wrong")])),"Conflicting role rejected")
 CASES.append(name+":64_union65_and_hash_conflict")
# Whole downstream per-group census, using RAM sizes only.
master=dict(output_directory="out",worker_auxiliary_scope="aux")
ram_sizes={"out":[(("contrastive","payload"),8*2**30,"regular"),(("embedding_contrastive_train","payload"),56*2**30,"regular"),(("RAM_retrieve_and_seal","payload"),2**30,"regular")],"aux":[]}
ct=extract("downstream_contract_v1.py",["charged_census_v1"],dict(require=need,STATE=(master,),local=lambda p:p,original=SimpleNamespace(_sizes=lambda path,**kw:copy.deepcopy(ram_sizes[path]))))
spec=dict(phase="finalize")
ct["charged_census_v1"](spec)
need(rejects(lambda:ct["charged_census_v1"](spec,required_write_bytes=1)),"No scope allowance above65")
ram_sizes["out"][1]=(("embedding_query_paths_val","payload"),56*2**30+1,"regular")
need(rejects(lambda:ct["charged_census_v1"](spec)),"Embedding group56GiB remains")
CASES.append("full65GiB_whole_and56GiB_embedding_census_failstops")
# Use exact pool factory with entirely synthetic scheduler/path/Popen surfaces.
class RamPath:
 def __truediv__(self,other):return self
 def __str__(self):return "RAM/path"
 def open(self,*a,**k):return SimpleNamespace(close=lambda:None)
launches=[]
def popen(argv,**kw):
 launches.append((list(argv),kw));pid=100+len(launches)
 return SimpleNamespace(pid=pid,stdout=SimpleNamespace(fileno=lambda:99))
base_scan=(ROOT/"src/arggnn/gpu_pool_v1.py").read_bytes()
down=(ROOT/"research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py").read_bytes()
old=(ROOT/"src/arggnn/gpu45_c3cache_h050_pool_v1.py").read_bytes()
fake_os=SimpleNamespace(environ={},pathsep=":",getuid=lambda:uid,set_blocking=lambda *a:None)
fake_base=SimpleNamespace(Controller=type("RamController",(),{}),proc_identity=lambda pid:dict(uid=uid,pid=pid,pgid=pid,sid=pid,start_ticks=pid*100))
legacy=SimpleNamespace(ACTIVE={"launching","running","draining"},time=clock,os=fake_os,subprocess=SimpleNamespace(Popen=popen,PIPE="RAM",STDOUT="RAM",DEVNULL="RAM"),bound=lambda r:json.dumps(dict(budget_components=dict(implementation=dict(sha256="a"),launcher=dict(sha256="b")))),require=need,json=json,ROOT=RamPath())
legacy.base=fake_base
class FakeShared:pass
reviewed=SimpleNamespace(legacy=legacy,base=fake_base,Shared0Controller=FakeShared)
h=SimpleNamespace(BASE_POOL="base",source_refs_v1=lambda:None,load=lambda *a:reviewed,raw=lambda r:{"base":base_scan,"down":down,"old":old}[r],POLICY="RAM")
c=SimpleNamespace(SHARED="shared",DOWN="down",OLD="old",guard_v1=lambda:None,guarded_check_v1=lambda *a:None,validate_cap_v1=lambda *a,**k:None,validate_submission_v1=lambda *a:None)
factory=extract("pool_v1.py",["build_controller_class_v1"],dict(h=h,c=c,ROOT=RamPath(),MODULE="RAM.module",require=need,ast=ast,os=fake_os,time=clock,json=json,resource=None,_verify_new_exclusive_v1=lambda *a:None,_open_old_inode_v1=lambda *a:None))
cls,_=factory["build_controller_class_v1"]()
def queued(kind):return dict(state="queued",requested=dict(worker_kind=kind,peak_mib=1,storage_ref="RAM",env={},log_limit_bytes=100,submission_ref="RAM",absolute_expires_unix=1000))
def queue_case(active_kind,expected):
 launches.clear();app=object.__new__(cls);app.lifecycle="running";app.safety_error=None
 app.jobs={}
 if active_kind:app.jobs["active"]=dict(state="running",requested=dict(worker_kind=active_kind))
 app.jobs.update(N1=queued("native"),N2=queued("native"),H1=queued("H058"),H2=queued("H058"))
 app._process_scan=lambda:None;app._memory=lambda incoming:dict(blocked_by=[]);app._lease_check=lambda:None;app._persist=lambda:None;app._check=lambda *a:None
 app.children={};app.logs={};app.processes={};app.pid_jobs={};app.guard=None;app.cap_ref={"path":"RAM","sha256":"RAM"};app.metadata=dict(uuid="RAM");app.gpu=0;app.pool_job="RAM";app.pool="RAM";app.python="RAM/python";app.lease=SimpleNamespace(fileno=lambda:7);app.directory=RamPath()
 app._launch_queue()
 actual=[name for name,j in app.jobs.items()if name!="active"and j["state"]=="running"]
 need(actual==expected,"Exactly one per kind and skip active same kind: "+str(actual))
 need(len(launches)==len(expected),"No real process calls")
 for argv,kw in launches:need(argv[:5]==["RAM/python","-B","-s","-m","RAM.module"]and kw["shell"]is False,"Exact fresh module dispatch")
queue_case(None,["N1","H1"]);queue_case("native",["H1"]);queue_case("H058",["N1"])
CASES.append("actual_generated_queue_one_active_per_kind_and_exact_dispatch")
result=dict(schema="independent_native_H058_downstream_pool_RAM_regressions_v1",status="PASS_STATIC_RAM_ONLY",cases=CASES,all_real_process_device_mutex_guard_admission_journal_operations=False,scientific_imports_or_payload_reads=False,CUDA_VISIBLE_DEVICES="",source_only=True)
(OUT/"independent_checks.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result,sort_keys=True))

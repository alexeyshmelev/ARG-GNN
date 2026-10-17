"""Independent narrow inference-only control review. RAM fixtures only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,hashlib,json
BASE=Path("research/reviews/frozen_TRAIN002_019_inference_only_source_20261017_v1")
OLD=Path("research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2")
checks=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def need(ok,label):
 if not ok:raise ValueError(label)
def reject(fn,label):
 try:fn()
 except(ValueError,KeyError,TypeError):checks.append(label);return
 raise AssertionError("Invalid metadata accepted: "+label)
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rr(p):return dict(path=str(p),sha256=hashlib.sha256(str(p).encode()).hexdigest())
def load(p):return json.loads(Path(p).read_text())
m=load(BASE/"source_manifest.v1.json");old=load(OLD/"source_manifest.v1.json")
check(ref(BASE/"source_manifest.v1.json")["sha256"]=="4a96cba030ea11b302bbca0bd0eed2c5f8da7f716f5b5e448cec242bb06a8a95","Exact sealed manifest")
expected={"worker_entry_v1.py":"4128d6de8a8f965d3bf749c30ca1e8447fe263b2102af4fc9f2d50e0f5de1f2a","batch_ROOT_v1.py":"b5c1e3041a19090131c097fbc8146f5fa5a0d269574bc1486e33b1f159763cf6","close_unit_ROOT_v1.py":"4e31cce52d8dbaf48ee964c6d5c1cf05d62a24e288e1f617a34c9dd0882e0ca2"}
trees={}
for name,h in expected.items():
 check(ref(BASE/name)["sha256"]==h,"Exact sealed "+name)
 nt=ast.parse((BASE/name).read_text());ot=ast.parse((OLD/name).read_text());trees[name]=nt
 nf={n.name:ast.dump(n,include_attributes=False)for n in nt.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 of={n.name:ast.dump(n,include_attributes=False)for n in ot.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 changed={k for k in of if k in nf and of[k]!=nf[k]}
 change={"worker_entry_v1.py":{"source_paths","stage_status","registered_row","validate_pipeline_authority","execute_in_admitted_context_v1"},"batch_ROOT_v1.py":{"admit_batch","run_batch"},"close_unit_ROOT_v1.py":set()}[name]
 added={"publish_parent_ready"}if name=="batch_ROOT_v1.py"else set()
 removed={"downstream_candidate","readiness_row","seal_all20"}if name=="batch_ROOT_v1.py"else set()
 check(changed==change and set(nf)-set(of)==added and set(of)-set(nf)==removed,"Exact limited control delta "+name)
 check([ast.dump(n)for n in nt.body if isinstance(n,(ast.Import,ast.ImportFrom))]==[ast.dump(n)for n in ot.body if isinstance(n,(ast.Import,ast.ImportFrom))],"Original imports unchanged "+name)
 check(len([k for k in of if k in nf and of[k]==nf[k]])>0,"Inherited unchanged functions "+name)
prior=m["original_pipeline2_review_ref"]
check(ref(prior["path"])==prior and load(prior["path"])["verdict"]=="SOURCE_PASS"and load(prior["path"])["independent"]is True,"Inherited pipeline2 independent SOURCE_PASS authenticated")
check(m["coordinator_source_ref"]==old["coordinator_source_ref"]and m["close_pins"]==old["close_pins"],"Exact original coordinator and ordinary accounting pins")
old_science=[r for r in old["refs"]if not r["path"].startswith(str(OLD)+"/")]
new_science=[r for r in m["refs"]if not r["path"].startswith(str(BASE)+"/")]
check(new_science==old_science,"All original non-control source refs unchanged")
for key in["synchronization_pool","registry","stage_caps","stage_templates","stage_runtime_refs"]:
 oldval=old["condition"][key];newval=m["condition"][key]
 if key.startswith("stage_"):oldval={"infer":oldval["infer"]}
 check(newval==oldval,"Original inference condition preserved "+key)
check(m["condition"]["synchronization_pool"]==m["synchronization_pool"],"Actual initializer still receives original shared slots")
check(m["admin_unique_union"]==old["admin_unique_union"]==63 and m["scientific_unit_unique_union"]==old["scientific_unit_unique_union"]==64,"No source-binding expansion or original ceiling change")
parents=tuple("american_c3_train_%03d"%i for i in range(2,20))
ram={};calls=[]
def read(r):
 if r["path"].startswith("ram/"):return copy.deepcopy(ram[r["path"]])
 check(ref(r["path"])==r,"Authenticated real metadata fixture read")
 return load(r["path"])
def extract(name,file,ns):
 n=next(n for n in trees[file].body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),str(BASE/file)+"#independentRAM","exec"),ns)
ns=dict(need=need,read=read,metadata_ref=ref,MANIFEST=str(BASE/"source_manifest.v1.json"),ALLOWED_PARENTS=parents,datetime=datetime,timezone=timezone)
for name in["stage_status","registered_row","validate_pipeline_authority"]:extract(name,"worker_entry_v1.py",ns)
check(ns["stage_status"]("infer")=="completed_current_C3_inference","Only original inference stage accepts")
reject(lambda:ns["stage_status"]("native_paths"),"Native paths stage rejects")
for p in parents:check(ns["registered_row"](p)["parent_id"]==p,"Original registered TRAIN row "+p)
for p in["american_c3_train_000","american_c3_train_001","american_c3_val_000","american_c3_test_000"]:
 reject(lambda p=p:ns["registered_row"](p),"Excluded parent rejects "+p)
now=datetime.now(timezone.utc)
controlref=rr("ram/sim_control");sim_auth=rr("ram/sim_authority")
ram[controlref["path"]]={"authority_ref":sim_auth};ram[sim_auth["path"]]={"expires_at_utc":(now+timedelta(days=2)).isoformat()}
ns["coordinator_support"]=lambda:SimpleNamespace(validate_batch_authority=lambda r:need(r==sim_auth,"Actual pinned coordinator authority metadata"))
a=dict(schema="TreeLoom_TRAIN002_019_finite_inference_authority_v1",created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(days=1)).isoformat(),root_authorizer="root_agent",active=True,source_only=False,source_manifest_ref=ref(BASE/"source_manifest.v1.json"),allowed_parents=list(parents),stages=["infer"],native_path_corpus_preparation_authorized=False,inference_only=True,maximum_producers=2,maximum_open_unit_reservations=2,stage_reservation_bytes=dict(infer=13000000000),admin_reservation_bytes=33554432,automatic_retry=False,GPU_authorized=False,query_truth_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False,preserve_original_science_and_accounting=True,reuse_original_simulation_accounting_coordinator=True,new_unit_terminal_semantics="actual_direct_OS_wait_and_owned_drain_no_UI_terminal_claim",simulation_batch_control_input_ref=controlref)
ar=rr("ram/authority");ram[ar["path"]]=a
check(ns["validate_pipeline_authority"](ar)==a,"Valid explicit inference-only authority accepts")
for key,bad in[("schema","TreeLoom_TRAIN001_019_finite_preparation_authority_v1"),("allowed_parents",["american_c3_train_001"]+list(parents)),("stages",["infer","native_paths"]),("native_path_corpus_preparation_authorized",True),("inference_only",False),("maximum_producers",3),("maximum_open_unit_reservations",3),("stage_reservation_bytes",dict(infer=13000000000,native_paths=16500000000)),("automatic_retry",True),("GPU_authorized",True),("query_truth_access_authorized",True),("TEST_access_authorized",True),("GNN_optimization_authorized",True),("source_manifest_ref",rr("ram/wrongmanifest")),("created_at_utc",(now+timedelta(hours=1)).isoformat()),("expires_at_utc",(now+timedelta(days=3)).isoformat())]:
 b=copy.deepcopy(a);b[key]=bad;ram[ar["path"]]=b
 reject(lambda:ns["validate_pipeline_authority"](ar),"Reject expanded/invalid authority "+key)
ram[ar["path"]]=a;oldend=ram[sim_auth["path"]]["expires_at_utc"];ram[sim_auth["path"]]["expires_at_utc"]=(now+timedelta(hours=1)).isoformat()
reject(lambda:ns["validate_pipeline_authority"](ar),"Coordinator must outlive inference continuation")
ram[sim_auth["path"]]["expires_at_utc"]=oldend
# Exercise new frontier publisher with independent RAM metadata, no actual terminals.
inp=rr("ram/input");done=rr("ram/done");obs=rr("ram/obs");accepted=rr("ram/accepted")
ram[inp["path"]]=dict(parent_id=parents[0],refs=dict(config=rr("ram/config")))
ram[done["path"]]=dict(status="completed_current_C3_inference",observation_ref=obs)
ram[obs["path"]]=dict(parent_id=parents[0],sequence_length_bp=30000000,query_ancestry_truth_in_inputs=False,genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",roster_ref=rr("ram/roster"))
unit=dict(stage="infer",parent_id=parents[0],unit_key="train_002_infer",simulation_completion_ref=rr("ram/simdone"))
sup=dict(completed_ref=done,actual_wait_ref=rr("ram/directwait"));written={}
def put(ctx,name,v):written[name]=v;return rr("ram/published/"+name)
bn=dict(need=need,read=read,terminal=lambda v:v,ref=rr,MANIFEST=str(BASE/"source_manifest.v1.json"),WORKER=str(BASE/"worker_entry_v1.py"),put=put)
extract("publish_parent_ready","batch_ROOT_v1.py",bn)
ctx=SimpleNamespace(directory=Path("ram/directory"))
bn["publish_parent_ready"](ctx,unit,inp,sup,accepted)
row=next(iter(written.values()))
check(row["native_path_corpus_prepared"]is False and row["compact_focals_still_required"]is True and row["query_truth_opened"]is False,"Frontier honestly requires compact focals and claims no corpus or truth")
check(row["inference_close_accepted_ref"]==accepted and row["direct_wait_ref"]==sup["actual_wait_ref"]and row["simulation_completion_ref"]==unit["simulation_completion_ref"],"Frontier retains original accepted close, wait and simulator refs")
for where,key,bad in[(unit,"stage","native_paths"),(ram[obs["path"]],"parent_id",parents[1]),(ram[obs["path"]],"sequence_length_bp",2000000),(ram[obs["path"]],"query_ancestry_truth_in_inputs",True),(ram[obs["path"]],"genealogy_kind","simulated_true_ARG"),(ram[done["path"]],"status","completed_native_C3_path_preparation")]:
 save=copy.deepcopy(where[key]);where[key]=bad
 reject(lambda:bn["publish_parent_ready"](ctx,unit,inp,sup,accepted),"Invalid frontier rejects "+key);where[key]=save
# Structural orchestration checks cover only changed control, inherited operations are not invoked.
bt=trees["batch_ROOT_v1.py"]
run=next(n for n in bt.body if isinstance(n,ast.FunctionDef)and n.name=="run_batch")
text=ast.unparse(run)
check("list(w.ALLOWED_PARENTS)"in text and "len(rows) < 18"in text and "len(completed) == len(attempted) == 18"in text,"Exactly18 inference controls, no TRAIN000/001 reuse or native staging")
check("downstream_candidate"not in text and "seal_all20"not in text and "native_paths"not in text,"No native or all20 dispatch remains in orchestration")
check(text.index("accepted = accept_close")<text.index("publish_parent_ready"),"Frontier follows genuine inherited close acceptance")
check("all20_training_readiness_claimed=False"in text.replace(" ","")and "native_path_corpus_prepared=False"in text.replace(" ",""),"Aggregate explicitly withholds all20 and native-corpus claims")
wt=trees["worker_entry_v1.py"]
dispatch=next(n for n in wt.body if isinstance(n,ast.FunctionDef)and n.name=="execute_in_admitted_context_v1")
dt=ast.unparse(dispatch)
check("need(STAGE == 'infer'"in dt and "from arggnn import american_c3_observe_v3 as science"in dt and "american_c3_paths_v1 as science"not in dt,"Scientific execution dispatch pins original inference only")
check("science.execute(cfg, writer)"in dt,"Original full inference execution remains")
result=dict(schema="TreeLoom_TRAIN002_019_inference_only_independent_checks_v1",passed=True,check_count=len(checks),checks=checks,narrow_control_delta_only=True,author_fixtures_reused=False,actual_guard_imported=False,actual_operations=False,actual_process_socket_lock_GPU_resource_probes=False,scientific_execution=False)
print(json.dumps(result,sort_keys=True,indent=2))

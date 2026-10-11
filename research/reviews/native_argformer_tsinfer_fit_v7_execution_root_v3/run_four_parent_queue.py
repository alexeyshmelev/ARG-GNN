"""Root-owned exclusive GPU2 native queue with independent CPU retrieval."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root"
RESULT=ROOT/"experiments/results/native_argformer_tsinfer_fit_v7_root"
PY=ROOT/".runtime/env/bin/python"
SOURCE=ROOT/"src/arggnn/native_argformer_tsinfer_fit_v7.py"
CONFIG=ROOT/"configs/native_argformer_tsinfer_fit_v7.json"
SOURCE_SHA="b94a4e16a6a9dc230256974b9a0e45a6edd1b41b9ecc8ac4171c4461ec514712"
CONFIG_SHA="d744aad4883062d23f874d15be65068b5f1b5e8a44db20ed2e8a425a4fd336c1"
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {"path":str(p.relative_to(ROOT)),"sha256":sha(p)}
def save(p,x):
 with p.open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def ticks(pid):return int(Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()[19])
def read_pinned(reference,expected_path):
 assert reference["path"]==str(expected_path.relative_to(ROOT))
 raw=expected_path.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==reference["sha256"]
 return json.loads(raw)
def stage_proposal(plan,parent,stage):
 entries=[entry for entry in plan["stage_proposals"] if entry["parent"]==parent and entry["stage"]==stage]
 assert len(entries)==1
 entry=entries[0]
 a=read_pinned(entry["admission_proposal"],BASE/f"{parent}_{stage}_admission_proposal.json")
 registration=read_pinned(entry["experiment_registration"],BASE/f"{parent}_{stage}_registration.json")
 assert a["experiment_registration"]==entry["experiment_registration"]
 assert a["scope_parents"]==registration["scope_parents"]==[parent] and a["stage"]==registration["stage"]==stage
 return a
def preflight_bindings(plan,activation):
 assert activation["budget_receipt"]==plan["budget_receipt"]
 reserve=read_pinned(plan["budget_receipt"],BASE/"budget_receipt.json")
 barrier=read_pinned(plan["release_barrier"],BASE/"release_barrier.json")
 expected={(parent,stage) for parent in ("bio_paths_pilot_train_000","bio_paths_pilot_train_001","bio_paths_pilot_val_000","bio_paths_pilot_val_001") for stage in ("probe-memory","fit","retrieve")}
 entries=plan["stage_proposals"]
 assert len(entries)==len(expected) and {(entry["parent"],entry["stage"]) for entry in entries}==expected
 for entry in entries:stage_proposal(plan,entry["parent"],entry["stage"])
 return reserve,barrier
cfg=read_pinned({"path":str(CONFIG.relative_to(ROOT)),"sha256":CONFIG_SHA},CONFIG)
assert sha(SOURCE)==SOURCE_SHA and sha(CONFIG)==CONFIG_SHA
# Root creates the bound activation packet after applying the actual budget.
activation=read(BASE/"queue_activation.json")
assert activation["root_authorizer"]=="root_agent" and activation["active"] is True
assert activation["queue_source"]==ref(Path(__file__))
plan_ref=activation["execution_plan"];plan_path=ROOT/plan_ref["path"]
plan=read_pinned(plan_ref,plan_path)
assert plan["queue_source"]==activation["queue_source"] and plan["physical_GPU"]==2
review_ref=activation["independent_review"];review_path=ROOT/review_ref["path"]
review=read_pinned(review_ref,review_path)
assert review["verdict"]=="PASS" and review["review_role"]=="independent_reviewer" and review["execution_plan"]==plan_ref and review["queue_source"]==activation["queue_source"]

reserve,barrier=preflight_bindings(plan,activation)
own_entries=reserve["admitted_reservations"]
legacy=ROOT/".runtime/gpup/storage_reservations.json"
supp=ROOT/".runtime/storage/project_budget_v4.json"
retrievals=[]
assert barrier["root_authorizer"]=="root_agent" and barrier["active"] is True
waiting=time.monotonic()
def identity_alive(record):
 path=Path(f"/proc/{record['pid']}")
 if not path.exists():return False
 if path.stat().st_uid!=record["uid"]:raise ValueError("release-barrier UID changed")
 return ticks(record["pid"])==record["start_ticks"]
save(BASE/"queue_wait_started.json",{"utc":utc(),"pid":os.getpid(),"start_ticks":ticks(os.getpid()),"barrier":plan["release_barrier"]})
while any(identity_alive(r) for r in barrier["waiters"]):
 if time.monotonic()-waiting>barrier["maximum_wait_seconds"]:
  save(BASE/"queue_wait_timeout.json",{"utc":utc(),"no_process_signaled":True})
  raise SystemExit(2)
 time.sleep(barrier["poll_interval_seconds"])
save(BASE/"queue_wait_completed.json",{"utc":utc(),"all_bound_old_identities_gone":True,"no_process_signaled":True})

def admission(parent,stage,run,probe=None,fit=None):
 a=stage_proposal(plan,parent,stage)
 assert a["implementation"]["sha256"]==SOURCE_SHA and a["config"]["sha256"]==CONFIG_SHA
 key="native_argformer_tsinfer_fit_v7/"+parent
 own=own_entries[key]
 a.update(run_id=run,logical_job_name="argformer7_"+run,output=str(Path(own["output"])/stage/run),launch_log=str(Path(own["output"])/"launch_logs"/(stage+"_"+run+".log")))
 storage={"kind":"native_argformer_tsinfer_fit_v7_root_storage","authorizer":"root_agent",
 "active":True,"implementation":a["implementation"],"config":a["config"],
 "scope_parents":[parent],"reservation_key":key,"reserved_bytes":20000000000,
 "output":own["output"],"write_scopes":[own["output"],*own["auxiliary_scopes"]],
 "stage_boundary_cap_not_OS_quota":True,"mode":"separately_audited_successor",
 "migration_authority":own["root_admission"],"companion":cfg["storage"]["companion_contract"],
 "reservation":own,"legacy_ledger_sha256_at_admission":sha(legacy),
 "supplemental_ledger_sha256_at_admission":sha(supp)}
 sp=BASE/f"{parent}_{stage}_{run}_storage_active.json";save(sp,storage)
 a.update(active=True,status="root_authorized_exact_native_stage",storage=ref(sp))
 if probe:a["full_batch_memory_receipt"]=ref(probe)
 if fit:a["completed_fit_receipt"]=ref(fit)
 ap=BASE/f"{parent}_{stage}_{run}_admission_active.json";save(ap,a)
 return a,ap
def start(parent,stage,run,probe=None,fit=None):
 a,ap=admission(parent,stage,run,probe,fit)
 args=["--execute","--parent",parent,"--stage",stage,"--run-id",run,
 "--expected-source-sha256",SOURCE_SHA,"--expected-config-sha256",CONFIG_SHA,
 "--admission",str(ap.relative_to(ROOT)),"--expected-admission-sha256",sha(ap)]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(ROOT/"src"),
 ARGGNN_LOGICAL_JOB_NAME=a["logical_job_name"])
 if stage=="retrieve":
  env["CUDA_VISIBLE_DEVICES"]=""
  command=[str(PY),"-B","-m","arggnn.native_argformer_tsinfer_fit_v7",*args]
 else:
  command=[str(PY),"-B",str(ROOT/"scripts/launch_gpu.py"),"--gpu","2","--job",
  a["logical_job_name"],"arggnn.native_argformer_tsinfer_fit_v7",*args]
 log=Path(a["launch_log"]);log.parent.mkdir(parents=True,exist_ok=True)
 stdout=log.open("xb")
 child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
 record={"parent":parent,"stage":stage,"run_id":run,"pid":child.pid,
 "start_ticks":ticks(child.pid),"started_utc":utc(),"root_admission":ref(ap),
 "argv":command,"GPU":None if stage=="retrieve" else 2}
 save(BASE/f"{parent}_{stage}_{run}_started.json",record)
 return child,stdout,record,Path(a["output"])/"receipt.json",a["maximum_walltime_seconds"]+900
def finish(item):
 child,stdout,record,rp,timeout=item
 try:code=child.wait(timeout=timeout);watchdog=False
 except subprocess.TimeoutExpired:
  watchdog=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
 stdout.close()
 result=dict(record,returncode=code,watchdog=watchdog,ended_utc=utc())
 if rp.is_file():result["receipt"]=ref(rp);result["scientific_status"]=read(rp).get("status")
 save(BASE/f"{record['parent']}_{record['stage']}_{record['run_id']}_terminal.json",result)
 return code==0 and rp.is_file() and read(rp).get("status")=="passed",rp
save(BASE/"queue_started.json",{"started_utc":utc(),"pid":os.getpid(),"start_ticks":ticks(os.getpid()),
 "physical_GPU":2,"other_project_GPU":3,"maximum_physical_GPUs":2,"root_activation":ref(BASE/"queue_activation.json")})
outcomes=[]
parents=[("train000","bio_paths_pilot_train_000"),("train001","bio_paths_pilot_train_001"),("val000","bio_paths_pilot_val_000"),("val001","bio_paths_pilot_val_001")]
probes={}
for label,parent in parents:
 ok,probe=finish(start(parent,"probe-memory",label+"_probe_memory_r001"))
 outcomes.append({"parent":parent,"probe_passed":ok})
 if not ok:
  save(BASE/"queue_terminal.json",{"ended_utc":utc(),"outcomes":outcomes,"stopped_on_first_failed_probe":True,"remaining_stages_unexecuted":True,"paper_reproduction_claimed":False,"silent_fallback":False})
  raise SystemExit(1)
 probes[parent]=probe
for label,parent in parents:
 ok,fit=finish(start(parent,"fit",label+"_fit_r001",probe=probes[parent]))
 outcomes.append({"parent":parent,"fit_passed":ok})
 if ok:retrievals.append(start(parent,"retrieve",label+"_retrieve_r001",fit=fit))
for item in retrievals:
 ok,_=finish(item);outcomes.append({"parent":item[2]["parent"],"retrieval_passed":ok})
save(BASE/"queue_terminal.json",{"ended_utc":utc(),"outcomes":outcomes,"paper_reproduction_claimed":False,"silent_fallback":False})

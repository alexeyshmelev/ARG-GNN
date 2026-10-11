"""Root-owned exclusive GPU3 native queue with independent CPU retrieval."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/native_argformer_tsinfer_fit_v6_execution_root"
RESULT=ROOT/"experiments/results/native_argformer_tsinfer_fit_v6_root"
PY=ROOT/".runtime/env/bin/python"
SOURCE=ROOT/"src/arggnn/native_argformer_tsinfer_fit_v6.py"
CONFIG=ROOT/"configs/native_argformer_tsinfer_fit_v6.json"
SOURCE_SHA="024b98957be0b86d86e97a7c9fef1f22a492709f652eaf3f7900f6f651d4c77f"
CONFIG_SHA="e0dfe65b93ce614a13bbfc390b9f79988956e3724e3c697ad840fc2dc0196318"
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {"path":str(p.relative_to(ROOT)),"sha256":sha(p)}
def save(p,x):
 with p.open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def ticks(pid):return int(Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()[19])
cfg=read(CONFIG)
assert sha(SOURCE)==SOURCE_SHA and sha(CONFIG)==CONFIG_SHA
# Root creates the bound activation packet after applying the actual budget.
activation=read(BASE/"queue_activation.json")
assert activation["root_authorizer"]=="root_agent" and activation["active"] is True
reserve_ref=activation["budget_receipt"]
reserve_path=ROOT/reserve_ref["path"]
assert sha(reserve_path)==reserve_ref["sha256"]
reserve=read(reserve_path)
own_entries=reserve["admitted_reservations"]
legacy=ROOT/".runtime/gpup/storage_reservations.json"
supp=ROOT/".runtime/storage/project_budget_v4.json"
retrievals=[]
def admission(parent,stage,run,probe=None,fit=None):
 proposal=BASE/f"{parent}_{stage}_admission_proposal.json"
 a=read(proposal)
 assert a["implementation"]["sha256"]==SOURCE_SHA and a["config"]["sha256"]==CONFIG_SHA
 key="native_argformer_tsinfer_fit_v6/"+parent
 own=own_entries[key]
 storage={"kind":"native_argformer_tsinfer_fit_v6_root_storage","authorizer":"root_agent",
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
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH="",
 ARGGNN_LOGICAL_JOB_NAME=a["logical_job_name"])
 if stage=="retrieve":
  env["CUDA_VISIBLE_DEVICES"]=""
  command=[str(PY),"-B","-m","arggnn.native_argformer_tsinfer_fit_v6",*args]
 else:
  command=[str(PY),"-B",str(ROOT/"scripts/launch_gpu.py"),"--gpu","3","--job",
  a["logical_job_name"],"arggnn.native_argformer_tsinfer_fit_v6",*args]
 log=Path(a["launch_log"]);log.parent.mkdir(parents=True,exist_ok=True)
 stdout=log.open("xb")
 child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
 record={"parent":parent,"stage":stage,"run_id":run,"pid":child.pid,
 "start_ticks":ticks(child.pid),"started_utc":utc(),"root_admission":ref(ap),
 "argv":command,"GPU":None if stage=="retrieve" else 3}
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
 "physical_GPU":3,"other_project_GPU":2,"maximum_physical_GPUs":2,"root_activation":ref(BASE/"queue_activation.json")})
outcomes=[]
for label,parent in [("train000","bio_paths_pilot_train_000"),("train001","bio_paths_pilot_train_001"),
 ("val000","bio_paths_pilot_val_000"),("val001","bio_paths_pilot_val_001")]:
 ok,probe=finish(start(parent,"probe-memory",label+"_probe_memory_r001"))
 if not ok:
  outcomes.append({"parent":parent,"probe_passed":False});continue
 ok,fit=finish(start(parent,"fit",label+"_fit_r001",probe=probe))
 outcomes.append({"parent":parent,"probe_passed":True,"fit_passed":ok})
 if ok:retrievals.append(start(parent,"retrieve",label+"_retrieve_r001",fit=fit))
for item in retrievals:
 ok,_=finish(item);outcomes.append({"parent":item[2]["parent"],"retrieval_passed":ok})
save(BASE/"queue_terminal.json",{"ended_utc":utc(),"outcomes":outcomes,
 "paper_reproduction_claimed":False,"silent_fallback":False})

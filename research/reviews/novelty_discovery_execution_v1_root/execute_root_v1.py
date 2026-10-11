"""Root-only integration for H030/H031. Review, closure, preparation and launch are separate commands."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, json, os, sys, traceback, subprocess, fcntl, stat

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
PACKET=ROOT/"research/reviews/novelty_discovery_execution_v1_root"
CLOSE=ROOT/"research/reviews/project_budget_v7_H026_close_root"
sys.path.insert(0,str(ROOT))
from scripts import project_budget_v7 as b

def ref(p): return b.v4.reference(ROOT,p)
def read(p): return json.loads((ROOT/p).read_text())
def save(p,v):
 p=Path(p)
 if not p.is_absolute(): p=ROOT/p
 with p.open("x") as f: json.dump(v,f,sort_keys=True,indent=2,allow_nan=False); f.write("\n")
 return ref(p)
def pins():
 return ref("scripts/project_budget_v7.py")["sha256"],ref("scripts/launch_reserved_gpu_v4.py")["sha256"]
def reviewed():
 source,launcher=pins()
 c,bindings=b.reviewed(ROOT,source,launcher)
 review_ref=ref("research/reviews/project_budget_v7_independent/receipt.json")
 review=b.v4.bound(ROOT,review_ref,True)
 assert review["verdict"]=="PASS" and review["actual_apply_performed"] is False
 assert all(review[k]==v for k,v in bindings.items())
 return source,launcher,bindings,review_ref
def authority(operation,view,bindings,review_ref):
 return dict(kind="project_budget_v7_root_admission",root_authorizer="root_agent",active=True,operation=operation,
     **bindings,independent_review=review_ref,expected_before=b.pins(view),
     all_budget_writers_use_scheduler_mutex=True,all_new_consumers_use_v7_guard=True,
     legacy_and_v4_new_admissions_frozen=True,no_historical_record_mutation_authorized=True,
     global_rebase_authorized=False,GPU_authorized=False,scientific_execution_authorized=False,
     v6_never_activated_root_attested=True)

def close_h026():
 source,launcher,bindings,review_ref=reviewed()
 assert not (ROOT/".runtime/storage/project_budget_v6").exists()
 view=b.inspect(ROOT,expected_source_sha256=source,expected_launcher_sha256=launcher)
 old=read(str(b.v4.SUPPLEMENTAL))["reservations"]
 own=old["H026/reference_census_innovation_v1_seed101_GPU3"]
 paths=b.v4.scope_spec(ROOT,own,True)
 completed_ref=ref(paths[0]/"completed.json")
 completed=b.v4.bound(ROOT,completed_ref,True);runtime=completed["runtime"];lease=runtime["lease"]
 terminal_pids=[runtime["pid"],lease["launcher_metadata"]["launcher_pid"]]
 assert len(set(terminal_pids))==2 and all(not Path("/proc",str(x)).exists() for x in terminal_pids)
 now=datetime.now(timezone.utc).isoformat()
 process=save(CLOSE/"process_observation.json",dict(observed_utc=now,recorded_pids=terminal_pids,
       recorded_pids_absent=True,launcher_exit_status_known=False))
 lease_path=Path(lease["lease_path"]);s=lease_path.lstat()
 expected=dict(device=lease["lease_device"],inode=lease["lease_inode"])
 assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and expected==dict(device=s.st_dev,inode=s.st_ino)
 fd=os.open(lease_path,os.O_RDWR|os.O_NOFOLLOW)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  current=lease_path.lstat();opened=os.fstat(fd)
  assert (s.st_dev,s.st_ino)==(current.st_dev,current.st_ino)==(opened.st_dev,opened.st_ino)
 finally:os.close(fd)
 lo=save(CLOSE/"lease_observation.json",dict(observed_utc=now,path=str(lease_path),
       original_lease_identity=expected,exclusive_nonblocking_acquire_release_verified=True,lease_bytes_modified=False))
 proc=subprocess.run(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
       "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
 rows=[]
 for line in proc.stdout.splitlines():
  x=[z.strip() for z in line.split(",")]
  rows.append(dict(pid=int(x[0]),gpu_uuid=x[1],process_name=x[2],used_gpu_memory_mib=x[3]))
 assert all(row["pid"] not in terminal_pids for row in rows)
 go=save(CLOSE/"gpu_observation.json",dict(observed_utc=now,recorded_pids=terminal_pids,
       no_recorded_gpu_context=True,compute_apps=rows,nvidia_smi_returncode=proc.returncode))
 terminal=save(CLOSE/"terminal_receipt.json",dict(kind="project_budget_v7_terminal_receipt",
    root_authorizer="root_agent",reservation_sha256=b.digest(own),completed=completed_ref,pids=terminal_pids,
    observed_utc=now,recorded_pids_absent=True,original_lease_exclusive_acquire_release_verified=True,
    no_recorded_gpu_context=True,external_NFS_writers_outside_observation=True,lease_path=str(lease_path),
    original_lease_identity=expected,process_observation=process,lease_observation=lo,gpu_observation=go,
    evidence_method="root_observed_absence",status="driver_completed_launcher_exit_unknown",returncode=None,
    exit_status_known=False))
 measurement=b.measure(ROOT,paths)
 a=authority("close",view,bindings,review_ref)
 a.update(scope_write_authority_revoked_for_all_cooperative_project_jobs=True,
      closed_scope_future_writes_and_relaunch_revoked=True,external_NFS_writers_outside_observation=True,
      opaque_credential_daemons=ref(CLOSE/"opaque_credential_daemons.json"),
      closure=dict(key=own["key"],reservation_sha256=b.digest(own),completed=completed_ref,
      completed_identity={k:completed[k] for k in ["status","hypothesis_id","seed","config_ref","input_ref"]},
      terminal_receipt=terminal,scope_paths=[str(p) for p in paths],expected_scope_measurement=measurement))
 ar=save(CLOSE/"budget_admission.json",a)
 result=b.apply(ROOT,ar,expected_source_sha256=source,expected_launcher_sha256=launcher)
 save(CLOSE/"apply_receipt.json",result)
 print(json.dumps(dict(stage="H026_retained_readonly",retained_bytes=measurement["conservative_scope_bytes"],
       projected_bytes=result["projected_bytes"],journal_tail_sha256=result["journal_tail_sha256"])),flush=True)

def prepare():
 source,launcher,bindings,review_ref=reviewed()
 adapter=ref("src/arggnn/project_budget_v7_consumer.py")
 h30review=read("research/reviews/novelty_discovery_consumers_v1_independent/H030_V7_receipt.json")
 h31review=read("research/reviews/novelty_discovery_consumers_v1_independent/H031_V7_receipt.json")
 assert h30review["verdict"]=="PASS" and h31review["verdict"]=="PASS"
 now=datetime.now(timezone.utc);expires=(now+timedelta(hours=23)).isoformat()
 jobs={}
 for hyp,device,cap,job,driver,cfgpath in [
  ("H030","cuda",2000000000,"reference-logcavity-gnn-v1","src/arggnn/train_reference_logcavity_gnn_v1.py","configs/reference_logcavity_gnn_v1.json"),
  ("H031","cpu",300000000,"observed-variant-signal-diagnostic-v1","src/arggnn/observed_variant_signal_diagnostic_v1.py","configs/observed_variant_residual_v1.json")]:
  spec=read(str((PACKET/f"{hyp}_inactive_input_draft.json").relative_to(ROOT)))
  cfg=read(cfgpath)
  if hyp=="H030":
   assert cfg["execution_guard_ready"] is True
   assert h30review["driver"]["sha256"]==ref(driver)["sha256"]
  else: assert h31review["driver"]["sha256"]==ref(driver)["sha256"]
  directory=PACKET/hyp;directory.mkdir()
  runtime=dict(schema=f"{hyp}_runtime_v1",status="active",job_id=job,device=device,created_at_utc=now.isoformat(),
    expires_at_utc=expires,output_directory=spec["output_directory"],training_dtype="float32" if device=="cuda" else "no_fit",
    physical_gpu=3 if device=="cuda" else None,maximum_project_physical_GPUs=2,
    process_title="AI Auto Research Pipeline Test" if device=="cuda" else None,
    max_cpu_rss_bytes=128000000000 if device=="cuda" else 8000000000,
    max_cuda_reserved_bytes=34359738368 if device=="cuda" else 0,
    launcher=bindings["launcher"],scientific_source=ref(driver),config=ref(cfgpath),
    independent_scientific_source_review=ref("research/reviews/novelty_discovery_consumers_v1_independent/"+hyp+"_V7_receipt.json"),
    first_tile_qualification="actual_first_TRAIN_FP32_update_included" if device=="cuda" else None)
  rr=save(directory/"runtime.json",runtime)
  view=b.inspect(ROOT,expected_source_sha256=source,expected_launcher_sha256=launcher)
  consumers=[ref(driver),ref(cfgpath),rr,adapter,bindings["implementation"],bindings["launcher"]]
  if hyp=="H030":consumers.append(ref("src/arggnn/reference_logcavity_gnn_v1.py"))
  a=authority("reserve",view,bindings,review_ref)
  a["reservation"]=dict(key=hyp+"/"+Path(spec["output_directory"]).name,output=str(ROOT/spec["output_directory"]),
       auxiliary_scopes=[],reserved_bytes=cap,consumer_bindings=consumers)
  ar=save(directory/"budget_admission.json",a)
  applied=b.apply(ROOT,ar,expected_source_sha256=source,expected_launcher_sha256=launcher)
  save(directory/"budget_receipt.json",applied)
  jobs[hyp]=dict(spec=spec,config=cfg,runtime=rr,reservation=applied["value"],job_id=job,cap=cap,directory=directory,driver=driver,cfgpath=cfgpath)
 # Both first-read pins refer to the complete two-job admission state.
 view=b.inspect(ROOT,expected_source_sha256=source,expected_launcher_sha256=launcher)
 for hyp,job in jobs.items():
  spec=job["spec"];directory=job["directory"]
  storage=dict(kind="project_budget_v7_storage_record",active=True,consumer_uses_v7_guard=True,
      job_id=job["job_id"],output_directory=spec["output_directory"],reserved_new_bytes=job["cap"],
      expires_at_utc=expires,reservation=job["reservation"],initial_ledger_pins=b.pins(view),
      budget_components=dict(implementation=bindings["implementation"],launcher=bindings["launcher"],consumer_adapter=adapter))
  sr=save(directory/"storage.json",storage)
  spec.update(status="active",root_authorizer="root_agent",job_id=job["job_id"],device="cuda" if hyp=="H030" else "cpu",
      created_at_utc=now.isoformat(),expires_at_utc=expires)
  spec["refs"].update(runtime=job["runtime"],storage=sr)
  if hyp=="H030":
   spec.update(fit_authorized=True,seed=101,epochs=10)
   spec["refs"]["config"]=ref(job["cfgpath"])
   spec["source_sha256"]={job["driver"]:ref(job["driver"])["sha256"],
       "src/arggnn/reference_logcavity_gnn_v1.py":ref("src/arggnn/reference_logcavity_gnn_v1.py")["sha256"]}
  else:
   spec.update(diagnostic_authorized=True,payload_reads_authorized=True)
   spec["refs"]["protocol"]=ref("docs/observed_variant_signal_diagnostic_v1.md")
   spec["source_sha256"]={job["driver"]:ref(job["driver"])["sha256"]}
  spec["source_sha256"].update({adapter["path"]:adapter["sha256"],
      bindings["implementation"]["path"]:bindings["implementation"]["sha256"],
      bindings["launcher"]["path"]:bindings["launcher"]["sha256"]})
  inp=save(directory/"input.json",spec)
  print(json.dumps(dict(stage="prepared",hypothesis=hyp,input_ref=inp,storage_ref=sr,projected_bytes=view["projected_bytes"])),flush=True)

def launch(hyp):
 source,launcher,bindings,review_ref=reviewed()
 directory=PACKET/hyp;ir=ref(directory/"input.json");spec=b.v4.bound(ROOT,ir,True)
 sr=spec["refs"]["storage"];storage=b.v4.bound(ROOT,sr,True)
 if hyp=="H030":
  cmd=[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/"scripts/launch_reserved_gpu_v4.py"),"--gpu","3",
       "--job",spec["job_id"],"--storage-record",sr["path"],"--expected-storage-sha256",sr["sha256"],
       "--expected-budget-source-sha256",source,"--expected-launcher-sha256",launcher,
       "arggnn.train_reference_logcavity_gnn_v1","--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]]
 else:
  cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m","arggnn.observed_variant_signal_diagnostic_v1",
       "--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]]
 env=dict(os.environ);env["PYTHONDONTWRITEBYTECODE"]="1";env["OMP_NUM_THREADS"]="1";env["MKL_NUM_THREADS"]="1"
 if hyp=="H031":env["CUDA_VISIBLE_DEVICES"]=""
 logfile=directory/"launch.log"
 with logfile.open("xb") as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(directory/"launch.json",dict(root_authorizer="root_agent",hypothesis_id=hyp,input_ref=ir,storage_ref=sr,
      launched_utc=datetime.now(timezone.utc).isoformat(),pid=proc.pid,physical_gpu=3 if hyp=="H030" else None,
      command=cmd,log_path=str(logfile.relative_to(ROOT)),exit_status_known=False))
 print(json.dumps(dict(stage="launched",hypothesis=hyp,launcher_or_CPU_pid=proc.pid,log_path=str(logfile.relative_to(ROOT)))),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("action",choices=["close","prepare","launchH030","launchH031"]);args=p.parse_args()
 try:
  if args.action=="close":close_h026()
  elif args.action=="prepare":prepare()
  else:launch(args.action[-4:])
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)

"""Root-only ARGMix v3 CPU preparation: admit 50GB, launch, wait and preserve logs.

Run with the ordinary project Python after scripts/project_env.sh. Creation/import
does not execute this helper. Its main entry point performs the root launch.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import importlib.util,json,os,subprocess,sys,traceback
R=Path("/workspace-SR003.nfs2/ARG-GNN");P=R/"research/reviews/expanded_full_panel_v1_root"
SOURCE="research/reviews/expanded_full_panel_v1_root/native_launch_helpers_v1/argmix_v3.py"
UTILITY="research/reviews/expanded_full_panel_v1_root/retire_inactive_scopes_before_fast_cache_v1.py"
DRAFT="research/reviews/expanded_native_argmix_v3_author/preparation_admission_draft.json"
DRAFT_SHA="531d19a76dd1c909c19fdedfe0bc9ebe68879fdc8f3e611bf16eb0865cf35a3b"
sp=importlib.util.spec_from_file_location("root_retired_storage",R/UTILITY)
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)


def run():
 from arggnn import prepare_expanded_native_argmix_v3 as prep
 if not (P/"obsolete_scope_retirements_v1/completed.json").is_file():
  raise ValueError("Root obsolete-scope retirement completion required")
 if m.ref(DRAFT)!={"path":DRAFT,"sha256":DRAFT_SHA}:raise ValueError("Frozen v3 preparation draft changed")
 spec=m.read(DRAFT)
 for name,sha in spec["source_sha256"].items():
  if m.ref(name)!={"path":name,"sha256":sha}:raise ValueError("Pinned v3 source changed: "+name)
 for name in ("config_ref","runtime_ref","generation_ref","registration_ref","build_ref","CPU_library_admission_ref"):
  m.m._bound(spec[name])
 if spec["source_ref"]!=m.ref(prep.SOURCE) or spec["storage_adapter_ref"]!=m.ref(m.m.SOURCE):
  raise ValueError("Exact frozen v3/retired sources required")
 packet=P/"native_argmix_preparation_v3";output=spec["output_directory"];cap=50000000000
 if (R/output).exists() or packet.exists():raise ValueError("Fresh native v3 output/root packet required; no replay")
 now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=23)).isoformat();job=spec["job_id"]
 packet.mkdir()
 cfg=m.m._bound(spec["config_ref"])
 cfg.update(execution_guard_ready=True,status="active_root_cpu_preparation",root_storage_and_GPU_guard_integration_pending=False)
 cfg_ref=m.save(packet,"config.json",cfg)
 runtime=m.m._bound(spec["runtime_ref"])
 runtime.update(status="active",root_must_freeze_at_admission=False,created_at_utc=now.isoformat(),
  expires_at_utc=expiry,job_id=job,source_ref=spec["source_ref"],device="cpu",CUDA_VISIBLE_DEVICES="",
  GPU_authorized=False,fit_authorized=False,output_directory=output)
 runtime_ref=m.save(packet,"runtime.json",runtime)
 spec.update(status="active",execution_authorized=True,created_at_utc=now.isoformat(),expires_at_utc=expiry,
  config_ref=cfg_ref,runtime_ref=runtime_ref,output_cap_bytes=cap,GPU_authorized=False,fit_authorized=False,
  root_must_supply_successor_reservation_initial5pins_and_active_storage=False)
 a=m.authority("reserve")
 if a["bindings"]!=spec["accounting_bindings_required"]:raise ValueError("All six frozen accounting refs required")
 bindings=list(a["bindings"].values())+[m.ref("src/arggnn/project_budget_v9_consumer.py"),m.ref(SOURCE),m.ref(UTILITY),m.ref(DRAFT)]
 bindings += [spec[name] for name in ("source_ref","storage_adapter_ref","config_ref","runtime_ref","generation_ref","registration_ref","build_ref","CPU_library_admission_ref")]
 bindings += [dict(path=p,sha256=h) for p,h in spec["source_sha256"].items()]
 unique=[]
 for b in bindings:
  if b not in unique:unique.append(b)
 spec["reservation_consumer_bindings_required"]=unique
 a["reservation"]=dict(key="H038/"+job,output=str(R/output),auxiliary_scopes=[],reserved_bytes=cap,consumer_bindings=unique)
 ar=m.save(packet,"reserve_authority.json",a)
 res=m.m.apply_reserve_v1(ar,expected_source_sha256=m.CORE,expected_launcher_sha256=m.BASE)
 m.save(packet,"reserve_receipt.json",res)
 spec["storage"]=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
  accounting_mode=m.m.MODE,job_id=job,output_directory=output,reserved_new_bytes=cap,reservation=res["value"],
  expires_at_utc=expiry,initial_ledger_pins=m.m._pins(m.inspect()),
  budget_components=dict(implementation=m.ref(m.m.budget.SOURCE_REL),launcher=m.ref(m.m.budget.LAUNCHER_REL),consumer_adapter=m.ref(m.m.SOURCE)))
 ir=m.save(packet,"input.json",spec)
 # The root log is deliberately outside every native reservation scope.
 command=[str(R/".runtime/env/bin/python"),"-B","-m","arggnn.prepare_expanded_native_argmix_v3",
  "--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",
  MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",RAYON_NUM_THREADS="1")
 with (packet/"launch.log").open("xb") as log:
  child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  launch_error=None
  try:
   m.save(packet,"launch.json",dict(pid=child.pid,process_group=child.pid,command=command,input_ref=ir,
    source_ref=spec["source_ref"],runtime_ref=runtime_ref,config_ref=cfg_ref,GPU_used=False,fit_performed=False,utc=now.isoformat()))
   print(json.dumps(dict(stage="native_argmix_v3_started",pid=child.pid,input_ref=ir)),flush=True)
   rc=child.wait()
  except BaseException as error:
   launch_error=error
   if child.poll() is None:child.terminate()
   rc=child.wait()
  finally:log.flush();os.fsync(log.fileno())
 done=R/output/"completed.json";failed=R/output/"failed.json"
 wait=m.save(packet,"root_wait.json",dict(pid=child.pid,returncode=rc,input_ref=ir,
  completed_ref=m.ref(done) if done.is_file() else None,failed_ref=m.ref(failed) if failed.is_file() else None,
  log_ref=m.ref(packet/"launch.log"),status="complete" if rc==0 and done.is_file() else "failed_partial_preserved",
  GPU_used=False,fit_performed=False,actual_direct_process_wait=True,error_type=None if launch_error is None else type(launch_error).__name__,
  error=None if launch_error is None else str(launch_error),finished_utc=datetime.now(timezone.utc).isoformat()))
 print(json.dumps(dict(stage="native_argmix_v3_exited",returncode=rc,root_wait_ref=wait)),flush=True)
 if launch_error is not None:raise launch_error
 if rc or not done.is_file():raise RuntimeError("Native preparation failed; partial products preserved without replay")
 return wait


if __name__=="__main__":
 try:run()
 except BaseException:print(traceback.format_exc(),flush=True);raise SystemExit(1)

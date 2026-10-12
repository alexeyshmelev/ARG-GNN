"""Root-reviewed draft: genuine prep -> free GPU2/3 -> native fit/eight VAL."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,fcntl,hashlib,importlib.util,json,math,os,signal,subprocess,time,traceback,uuid
R=Path("/workspace-SR003.nfs2/ARG-GNN");P=R/"research/reviews/expanded_full_panel_v1_root"
SOURCE="research/reviews/expanded_full_panel_v1_root/native_launch_helpers_v1/argmix_native_gpu_fit_v1.py"
UTILITY="research/reviews/expanded_full_panel_v1_root/retire_inactive_scopes_before_fast_cache_v1.py"
sp=importlib.util.spec_from_file_location("root_native_fit_storage",R/UTILITY)
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
PREP=P/"native_argmix_preparation_v3";PREP_OUTPUT=R/"data/expanded_native_argmix_prepared_v3"
PYG=P/"full_panel_pyg_fits_v4"


def wait_prepared(deadline):
 while not ((PREP_OUTPUT/"completed.json").is_file() and (PREP/"root_wait.json").is_file()):
  if (PREP_OUTPUT/"failed.json").exists():raise RuntimeError("Preserved native preparation failure")
  if (PREP/"root_wait.json").exists() and m.read(PREP/"root_wait.json").get("returncode")!=0:raise RuntimeError("Native prep root wait failed")
  if time.monotonic()>deadline:raise TimeoutError("Genuine native preparation completion wait expired")
  time.sleep(10)
 done_ref=m.ref(PREP_OUTPUT/"completed.json");done=m.m._bound(done_ref)
 input_ref=m.ref(PREP/"input.json");spec=m.m._bound(input_ref);wait_ref=m.ref(PREP/"root_wait.json");wait=m.m._bound(wait_ref)
 if (done["input_ref"]!=input_ref or done["TRAIN_parents"]!=32 or done["VAL_parents"]!=8
  or done["outer_parents"]!=0 or done["GPU_used"] is not False or done["fit_performed"] is not False
  or wait.get("returncode")!=0 or wait.get("completed_ref")!=done_ref or wait.get("actual_direct_process_wait") is not True):
  raise ValueError("Genuine exact all40 preparation and direct root wait required")
 return spec,input_ref,done_ref,wait_ref,done["native_prepared_ref"]


def wait_GPU(deadline):
 from arggnn.resources import ALLOWED_GPUS,eligible_free,gpu_inventory
 while True:
  # The current priority dispatcher must have launched both H038 arms first.
  if all((PYG/arm/"relaunch_v6.json").is_file() for arm in ("aligned","zero_concordance")):
   with m.m.budget.v4.mutex(R):
    inventory={r["index"]:r for r in gpu_inventory()};handles=[];held=[];free=[]
    try:
     for gpu in ALLOWED_GPUS:
      handle=(R/(".runtime/locks/gpu_%d.lock"%gpu)).open("r+");handles.append(handle)
      try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:held.append(gpu)
      else:
       if gpu in (2,3) and eligible_free(inventory[gpu]):free.append(inventory[gpu])
     if len(held)<2 and free:return free[0]
    finally:
     for handle in handles:handle.close()
  if time.monotonic()>deadline:raise TimeoutError("Priority-respecting free GPU2/3 wait expired")
  time.sleep(10)


def size_bounds(prepared_ref):
 prepared=m.m._bound(prepared_ref);frames=m.m._bound(prepared["frames"])["parents"]
 vals={p:f for p,f in frames.items() if f["split"]=="val"}
 if len(frames)!=40 or len(vals)!=8:raise ValueError("Full32TRAIN/eight VAL metadata required")
 per={}
 for p,f in vals.items():
  t=f["native_tree_count"]
  if type(t) is not int or t<1 or f["queries"]!=686 or f["normalization_length_bp"]!=2000000:raise ValueError("Original full VAL shape required")
  rows=686*t;chunks=686*(math.ceil(t/512)+2)
  per[p]=dict(native_trees=t,rows_upper=rows,
   HDF5_two_files_upper_bytes=rows*512,
   raw_chunk_logits_and_indexes_upper_bytes=chunks*1024*25+rows*24,
   full_grid_probabilities_mask_coordinates_upper_bytes=20000*686*17+2_000_000)
 val_upper=sum(sum(v[k] for k in ("HDF5_two_files_upper_bytes","raw_chunk_logits_and_indexes_upper_bytes",
  "full_grid_probabilities_mask_coordinates_upper_bytes")) for v in per.values())
 checkpoint=64_000_000;fit=checkpoint+64_000_000;workspace=1_000_000_000
 cap=math.ceil((fit+val_upper+workspace)/1_000_000_000)*1_000_000_000
 return dict(schema="native_fit_all8_raw_output_bound_v1",VAL=per,checkpoint_upper_bytes=checkpoint,
  checkpoint_count=1,fit_upper_bytes=fit,VAL_upper_bytes=val_upper,workspace_upper_bytes=workspace,
  output_cap_bytes=cap,model_state_size_checked_before_fit=True,no_archive_copy=True)


def run(wait_seconds):
 from arggnn import expanded_native_argmix_gpu_fit_root_v1 as worker
 deadline=time.monotonic()+wait_seconds
 prep,prep_input,prep_done,prep_wait,prepared=wait_prepared(deadline)
 bounds=size_bounds(prepared);gpu=wait_GPU(deadline)
 packet=P/"native_argmix_GPU_fit_v1";output="experiments/results/expanded_native_argmix_full32TRAIN8VAL_fit_v1"
 short=".runtime/tmp/naf_"+hashlib.sha256(output.encode()).hexdigest()[:12]
 if packet.exists() or (R/output).exists() or (R/short).exists():raise ValueError("Fresh non-overlapping native fit scopes required; no replay")
 if m.ref(worker.SOURCE)["sha256"]!=m.ref(worker.DRAFT)["sha256"]:raise ValueError("Canonical consumer differs from reviewed draft")
 packet.mkdir()
 now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=23)).isoformat();job="expanded-native-argmix-full32TRAIN8VAL-fit-v1"
 source_sha={p:m.ref(p)["sha256"] for p in worker.SOURCE_PATHS}
 for p,sha in prep["source_sha256"].items():
  if source_sha[p]!=sha:raise ValueError("Prepared native/source closure drift")
 cfg_ref=prep["config_ref"];m.m._bound(cfg_ref)
 # Reuse exact preparation config. Copying/changing it would break product identity.
 rt=m.save(packet,"runtime.json",dict(status="active",device="cuda",physical_GPU=gpu["index"],
  canonical_uuid=str(uuid.UUID(gpu["uuid"].removeprefix("GPU-"))),native_prefix=".runtime/official_argmix_abi210_v1/cu129/env",
  native_python="3.12",torch="2.10.0+cu129",process_title="AI Auto Research Pipeline Test",source_ref=m.ref(worker.SOURCE),
  job_id=job,output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry))
 br=m.save(packet,"size_bounds.json",bounds)
 refs=dict(config_ref=cfg_ref,runtime_ref=rt,prepared_ref=prepared,prep_completed_ref=prep_done,
  prep_input_ref=prep_input,prep_root_wait_ref=prep_wait,build_ref=prep["build_ref"],
  CPU_library_admission_ref=prep["CPU_library_admission_ref"],size_bounds_ref=br)
 a=m.authority("reserve");consumers=list(a["bindings"].values())+[m.ref("src/arggnn/project_budget_v9_consumer.py")]
 consumers+=list(refs.values())+[dict(path=p,sha256=h) for p,h in source_sha.items()]+[m.ref(PREP/"runtime.json"),prep["generation_ref"],prep["registration_ref"]]
 unique=[]
 for ref in consumers:
  if ref not in unique:unique.append(ref)
 a["reservation"]=dict(key="H038/"+job,output=str(R/output),auxiliary_scopes=[str(R/short)],
  reserved_bytes=bounds["output_cap_bytes"],consumer_bindings=unique)
 ar=m.save(packet,"reserve_authority.json",a);res=m.m.apply_reserve_v1(ar,expected_source_sha256=m.CORE,expected_launcher_sha256=m.BASE)
 m.save(packet,"reserve_receipt.json",res)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.m.MODE,
  job_id=job,output_directory=output,reserved_new_bytes=bounds["output_cap_bytes"],reservation=res["value"],
  expires_at_utc=expiry,initial_ledger_pins=m.m._pins(m.inspect()),budget_components=dict(
   implementation=m.ref(m.m.budget.SOURCE_REL),launcher=m.ref(m.m.budget.LAUNCHER_REL),consumer_adapter=m.ref(m.m.SOURCE)))
 sr=m.save(packet,"storage.json",storage)
 spec=dict(schema=worker.SCHEMA,status="active",root_authorizer="root_agent",fit_authorized=True,GPU_authorized=True,
  all8_VAL_inference_authorized=True,outer_included=False,source_sha256=source_sha,
  job_id=job,output_directory=output,short_tmp_directory=short,output_cap_bytes=bounds["output_cap_bytes"],
  max_cpu_rss_bytes=512000000000,created_at_utc=now.isoformat(),expires_at_utc=expiry,storage=storage,**refs)
 ir=m.save(packet,"input.json",spec)
 command=[str(R/".runtime/env/bin/python"),"-B","-m","scripts.launch_reserved_gpu_terminal_retired_scopes_v1",
  "--gpu",str(gpu["index"]),"--job",job,"--storage-record",sr["path"],"--expected-storage-sha256",sr["sha256"],
  "--expected-budget-source-sha256",m.CORE,"--expected-launcher-sha256",m.BASE,
  "arggnn.expanded_native_argmix_gpu_fit_root_v1","--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]]
 env=dict(os.environ,PYTHONPATH=str(R)+":"+str(R/"scripts")+":"+str(R/"src"),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
 with (packet/"launch.log").open("xb") as log:
  child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  error=None
  try:
   m.save(packet,"launch.json",dict(pid=child.pid,command=command,input_ref=ir,gpu=gpu["index"],UUID=gpu["uuid"],cap_bytes=bounds["output_cap_bytes"]))
   while child.poll() is None:
    if datetime.now(timezone.utc)>=datetime.fromisoformat(expiry):raise TimeoutError("Native fit admission expired")
    rss,_=worker.prep.owned_rss_v1(child.pid)
    if rss>spec["max_cpu_rss_bytes"]:raise MemoryError("Owned native fit process tree exceeds RSS bound")
    m.m.budget_check_v9(storage,boundary=False);time.sleep(5)
   rc=child.wait()
  except BaseException as e:
   error=e
   if child.poll() is None:child.terminate()
   rc=child.wait()
  finally:log.flush();os.fsync(log.fileno())
 done=R/output/"completed.json"
 wait=m.save(packet,"root_wait.json",dict(pid=child.pid,returncode=rc,input_ref=ir,gpu=gpu["index"],
  completed_ref=m.ref(done) if done.is_file() else None,log_ref=m.ref(packet/"launch.log"),
  actual_direct_launcher_wait=True,error_type=None if error is None else type(error).__name__,
  status="complete" if rc==0 and done.is_file() else "failed_partial_preserved",finished_utc=datetime.now(timezone.utc).isoformat()))
 print(json.dumps(dict(stage="native_GPU_fit_waited",returncode=rc,root_wait_ref=wait)),flush=True)
 if error is not None:raise error
 if rc or not done.is_file():raise RuntimeError("Native fit failed; preserve artifacts without scientific replay")
 return wait


def main():
 p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");p.add_argument("--wait-seconds",type=int,default=172800);a=p.parse_args()
 if not a.execute:print(json.dumps(dict(status="source_only_root_native_fit_draft",worker_promotion_required=True)));return
 run(a.wait_seconds)
if __name__=="__main__":
 try:main()
 except BaseException:print(traceback.format_exc(),flush=True);raise SystemExit(1)

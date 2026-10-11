import pathlib,json,os,subprocess,signal,time,datetime
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN");BASE=ROOT/"research/reviews/anchor_parent_producer_actual_train000_v1_root";OUT=ROOT/"experiments/results/anchor_parent_observed_train000_v1"
reg=json.loads((BASE/"observed_registration.json").read_text());OUT.mkdir(exist_ok=False)
env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),CUDA_VISIBLE_DEVICES="",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1")
for key in ["XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME","HF_HOME","TORCH_HOME","TRITON_CACHE_DIR","CUDA_CACHE_PATH","TORCHINDUCTOR_CACHE_DIR","NUMBA_CACHE_DIR","MPLCONFIGDIR","WANDB_DIR","WANDB_CACHE_DIR","WANDB_CONFIG_DIR","TMPDIR","TMP","TEMP"]:
 path=OUT/"write_scopes"/key.lower();path.mkdir(parents=True);env[key]=str(path)
start=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (OUT/"stdout.json").open("xb") as out,(OUT/"stderr.log").open("xb") as err:
 p=subprocess.Popen([str(ROOT/".runtime/env/bin/python"),"-B",str(BASE/"observed_child.py")],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
 tick=int(pathlib.Path(f"/proc/{p.pid}/stat").read_text().rsplit(") ",1)[1].split()[19])
 (BASE/"observed_started.json").write_text(json.dumps({"supervisor_PID":os.getpid(),"child_PID":p.pid,"child_start_ticks":tick,"UTC":started},indent=2)+"\n")
 failure=None
 while p.poll() is None:
  used=sum(max(x.stat().st_size,x.stat().st_blocks*512)+8192 for x in OUT.rglob("*") if x.is_file())
  status=pathlib.Path(f"/proc/{p.pid}/status")
  rss=next((int(l.split()[1])*1024 for l in status.read_text().splitlines() if l.startswith("VmRSS:")),0) if status.exists() else 0
  if used>20000000 or rss>128000000000 or time.monotonic()-start>3600:
   failure="owned_output_or_RSS_or_time_limit";os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(.5)
 rc=p.wait()
record={"returncode":rc,"watchdog_failure":failure,"started_utc":started,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":"completed_observed_inputs_no_model" if rc==0 and not failure else "failed_partial_preserved","model_truth_GPU_or_fit_authorized":False}
(OUT/"exit.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record),flush=True)

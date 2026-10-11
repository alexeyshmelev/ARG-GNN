import datetime,hashlib,json,os,pathlib,signal,subprocess,time
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN");BASE=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_runtime_capture_root"
OUT=ROOT/"experiments/results/native_argformer_tsinfer_fit_v7_root"
reg=json.loads((BASE/"registration.json").read_text());limit=reg["maximum_total_output_bytes"]
OUT.mkdir(exist_ok=False);started=datetime.datetime.now(datetime.timezone.utc).isoformat()
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES="",PYTHONPATH="",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",WANDB_MODE="disabled",HF_HUB_DISABLE_TELEMETRY="1")
for key in ["XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME","HF_HOME","TORCH_HOME","TRITON_CACHE_DIR","CUDA_CACHE_PATH","TORCHINDUCTOR_CACHE_DIR","NUMBA_CACHE_DIR","MPLCONFIGDIR","WANDB_DIR","WANDB_CACHE_DIR","WANDB_CONFIG_DIR"]:
 path=OUT/"write_scopes"/key.lower();path.mkdir(parents=True);env[key]=str(path)
tmp=OUT/"tmp";tmp.mkdir()
for key in ["TMPDIR","TMP","TEMP"]:env[key]=str(tmp)
for key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","BLIS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:env[key]="4"
env.pop("PYTHONHOME",None)
argv=[str(ROOT/".runtime/conda/envs/argformer_native_v1/bin/python"),"-B",str(BASE/"capture_child.py")]
with (OUT/"stdout.log").open("xb") as stdout,(OUT/"stderr.log").open("xb") as stderr:
 process=subprocess.Popen(argv,env=env,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
 def ticks(pid):return int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()[19])
 (BASE/"started.json").write_text(json.dumps({"supervisor_PID":os.getpid(),"child_PID":process.pid,"child_start_ticks":ticks(process.pid),"started_utc":started,"argv":argv,"timeout_seconds":reg["timeout_seconds"]},indent=2)+"\n")
 launch=time.monotonic();failure=None
 while process.poll() is None:
  used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in OUT.rglob("*") if p.is_file())
  if used>limit or time.monotonic()-launch>reg["timeout_seconds"]:
   failure="owned_output_limit" if used>limit else "owned_capture_timeout";os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   break
  time.sleep(.5)
 status=process.wait()
stdout=(OUT/"stdout.log").read_text();stderr=(OUT/"stderr.log").read_text()
receipt={"status":"complete" if status==0 and not failure else "failed_partial_preserved","returncode":status,"root_watchdog_failure":failure,"started_utc":started,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"stdout":stdout[:20000],"stderr":stderr[:20000],"models_instantiated":False,"GPU_used":False,"prefix_mutation_authorized":False,"registration_sha256":hashlib.sha256((BASE/"registration.json").read_bytes()).hexdigest(),"actual_complete_runtime_inventory":status==0 and not failure}
(OUT/"capture_exit.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"status":receipt["status"],"returncode":status}),flush=True)

import os,subprocess,pathlib,json,time,datetime as dt,signal
root=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
packet=root/"experiments/results/census_tsinfer_observation_inspect_v1_root"
admission=packet/"admission.json"
import hashlib
args=[str(root/".runtime/env/bin/python"),"-B","-m","arggnn.census_tsinfer_observation_v1","--execute","--phase","inspect","--admission",str(admission.relative_to(root)),"--expected-admission-sha256",hashlib.sha256(admission.read_bytes()).hexdigest()]
env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
def identity(pid):
 s=pathlib.Path(f"/proc/{pid}/stat").read_text();return int(s[s.rfind(")")+2:].split()[19])
started=dt.datetime.now(dt.timezone.utc).isoformat()
with (packet/"worker.stdout").open("xb") as log:
 child=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record={"kind":"root_Census_tsinfer_CPU_inspection_start_v1","started_utc":started,"supervisor_pid":os.getpid(),"supervisor_start_ticks":identity(os.getpid()),"worker_pid":child.pid,"worker_start_ticks":identity(child.pid),"argv":args,"external_watchdog_seconds":21600,"CUDA_VISIBLE_DEVICES":"","scientific_fit_authorized":False}
 with (packet/"started.json").open("x") as f:json.dump(record,f,indent=2);f.write("\n")
 timeout=False
 try:returncode=child.wait(timeout=21600)
 except subprocess.TimeoutExpired:
  timeout=True
  if identity(child.pid)==record["worker_start_ticks"]:os.killpg(child.pid,signal.SIGTERM)
  try:returncode=child.wait(timeout=20)
  except subprocess.TimeoutExpired:
   if identity(child.pid)==record["worker_start_ticks"]:os.killpg(child.pid,signal.SIGKILL)
   returncode=child.wait()
 result={"kind":"root_Census_tsinfer_CPU_inspection_exit_v1","started_utc":started,"completed_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"returncode":returncode,"watchdog_triggered":timeout,"worker_pid":child.pid,"worker_start_ticks":record["worker_start_ticks"]}
 with (packet/"exit.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")

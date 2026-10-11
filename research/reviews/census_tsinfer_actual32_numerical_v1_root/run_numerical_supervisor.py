"""Root external watchdog for the frozen actual32 numerical qualification."""
from pathlib import Path
import os,subprocess,json,datetime,hashlib,signal
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
PACKET=ROOT/"research/reviews/census_tsinfer_actual32_numerical_v1_root"
BASE=ROOT/"experiments/results/census_tsinfer_actual32_numerical_v1_supervisor"
BASE.mkdir(exist_ok=False)
admission=PACKET/"admission_active.json"
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ticks(pid):return int(Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()[19])
def save(name,d):
 with (BASE/name).open("x") as f:json.dump(d,f,indent=2);f.write("\n")
args=[str(ROOT/".runtime/env/bin/python"),"-B","-m","arggnn.census_tsinfer_observation_v1",
 "--execute","--phase","numerical","--admission",str(admission.relative_to(ROOT)),
 "--expected-admission-sha256",hashlib.sha256(admission.read_bytes()).hexdigest()]
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
with (BASE/"worker.stdout").open("xb") as log:
 child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 started={"schema":"census_tsinfer_actual32_numerical_root_start_v1","started_utc":utc(),
 "worker_pid":child.pid,"worker_start_ticks":ticks(child.pid),"supervisor_pid":os.getpid(),
 "supervisor_start_ticks":ticks(os.getpid()),"argv":args,"external_watchdog_seconds":21600,
 "GPU_authorized":False,"fit_authorized":False,"query_truth_authorized":False}
 save("started.json",started)
 timeout=False
 try:code=child.wait(timeout=21600)
 except subprocess.TimeoutExpired:
  timeout=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
 save("exit.json",dict(started,completed_utc=utc(),returncode=code,watchdog_triggered=timeout))

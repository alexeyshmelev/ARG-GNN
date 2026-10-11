"""Root supervisor for unchanged seed202 seven-control parallel run."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"experiments/results/biological_mechanisms_seed202_parallel_v1_root"
def write(name,obj):
 (BASE/name).write_text(json.dumps(obj,indent=2)+"\n")
def ticks(pid):return int(Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()[19])
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
admission=BASE/"launch_admission.json"
bound=json.loads((BASE/"launch_command.json").read_text())
if hashlib.sha256(admission.read_bytes()).hexdigest()!=bound["admission_sha256"]:
 raise ValueError("Frozen root admission changed")
command=[str(ROOT/".runtime/env/bin/python"),"-B","-m","arggnn.budgeted_gpu_launcher_v1","--launch",
 "--admission",str(admission.relative_to(ROOT)),"--expected-admission-sha256",bound["admission_sha256"]]
with (BASE/"launch.stdout").open("xb") as stdout,(BASE/"launch.stderr").open("xb") as stderr:
 child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=stdout,stderr=stderr,start_new_session=True)
 launch={"kind":"root_parallel_training_supervisor_v1","started_utc":utc(),"launcher_pid":child.pid,
 "launcher_start_ticks":ticks(child.pid),"command":command,"physical_gpu":2,"seed":202,
 "supervisor_pid":os.getpid(),"supervisor_start_ticks":ticks(os.getpid()),"watchdog_seconds":86400}
 write("launch.json",launch)
 try:
  code=child.wait(timeout=86400)
  result=dict(launch,status="terminal",returncode=code,completed_utc=utc(),watchdog_triggered=False)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=30)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGKILL);code=child.wait()
  result=dict(launch,status="watchdog_terminal",returncode=code,completed_utc=utc(),watchdog_triggered=True)
 write("exit.json",result)

from pathlib import Path
import os,json,hashlib,subprocess,sys,time,datetime,traceback
sys.excepthook=lambda t,v,b:traceback.print_exception(t,v,b,file=sys.stdout)
R=Path.cwd();H=R/"research/reviews/H056_three_bank_application_actual_root_20261016_v2"
def ref(p):return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def put(n,v):
 with (H/n).open("x") as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
 return ref(H/n)
def identity(pid):
 a=Path("/proc",str(pid),"stat").read_text().rsplit(")",1)[1].split()
 return dict(pid=pid,ppid=int(a[1]),pgid=int(a[2]),sid=int(a[3]),start_ticks=int(a[19]))
assert R==Path("/workspace-SR003.nfs2/ARG-GNN")
ip=H/"input.json";ir=ref(ip);spec=json.loads(ip.read_bytes())
assert spec["execution_authorized"] and spec["root_host_resource_enforcement_active"]
observer=identity(os.getpid());put("root_supervisor_identity.root_v1.json",observer)
argv=[str(R/".runtime/env/bin/python"),"-B","research/reviews/H056_bank_application_runtime_source_20261016_v1/runtime_v2.py","--launch","--input",ir["path"],"--expected-sha256",ir["sha256"]]
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
start=time.monotonic()
with (H/"worker_stdout.log").open("xb") as log:
 proc=subprocess.Popen(argv,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch=identity(proc.pid)
 put("actual_launch.root_v1.json",dict(schema="H056_frozen_application_actual_launch_root_v1",argv=argv,input_ref=ir,launcher_identity=launch,root_supervisor_identity=observer,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 rc=proc.wait()
 put("actual_direct_wait_minimal.root_v1.json",dict(actual_direct_wait_observed=True,returncode=rc,launcher_pid=proc.pid,launcher_start_ticks=launch["start_ticks"],seconds=time.monotonic()-start,input_ref=ir))
absent=not Path("/proc",str(proc.pid)).exists()
try:os.killpg(proc.pid,0);group_absent=False
except ProcessLookupError:group_absent=True
wait=dict(schema="H056_frozen_application_actual_direct_wait_and_absence_v1",root_supervisor_identity=observer,launcher_pid=proc.pid,launcher_start_ticks=launch["start_ticks"],actual_direct_wait_observed=True,returncode=rc,actual_launcher_PID_absent=absent,actual_owned_group_absent=group_absent,seconds=time.monotonic()-start,stdout_ref=ref(H/"worker_stdout.log"))
put("actual_wait.root_v1.json",wait)
print(json.dumps(wait),flush=True)
print((H/"worker_stdout.log").read_text()[-16000:],flush=True)
assert rc==0 and absent and group_absent

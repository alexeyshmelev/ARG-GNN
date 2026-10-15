"""Directly wait the unmodified root wave supervisor and persist its real result."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent
PARENTS=["american_c3_short_train_012","american_c3_short_train_013"]
def ref(path):
 p=Path(path)
 if not p.is_absolute(): p=ROOT/p
 return {"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def save(name,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+"\n").encode()
 assert len(data)<=65536
 with (HERE/name).open("xb") as f:
  f.write(data)
  f.flush()
  os.fsync(f.fileno())
def start(pid):
 return Path("/proc/%s/stat"%pid).read_text().rsplit(")",1)[1].split()[19]
def main():
 assert Path.cwd().resolve()==ROOT
 pins=json.loads((HERE/"pins.json").read_text())
 for r in pins.values(): assert ref(r["path"])==r
 cmd=[str(ROOT/".runtime/env/bin/python"),"-B",pins["driver"]["path"],"--run-parents",*PARENTS,
      "--policy",pins["policy"]["path"],"--policy-sha",pins["policy"]["sha256"]]
 env=os.environ.copy()
 env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
 proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
 save("launch_identity.json",dict(schema="C3_real_supervisor_wait_launch_v1",command=cmd,
      wrapper_pid=os.getpid(),wrapper_start=start(os.getpid()),direct_supervisor_pid=proc.pid,
      direct_supervisor_start=start(proc.pid),direct_supervisor_process_group=proc.pid,
      pins=pins,wrapper_source_ref=ref(__file__),GPU_used=False,
      created_utc=datetime.now(timezone.utc).isoformat()))
 # The unchanged supervisor retains scientific logs in its declared stage scopes.
 # Here it emits only bounded two-parent summaries or an exception traceback.
 captured=proc.communicate()[0]
 rc=proc.wait()
 log=captured[:1048576]
 with (HERE/"supervisor_terminal.log").open("xb") as f:
  f.write(log);f.flush();os.fsync(f.fileno())
 stage_refs=[]
 roster=[dict(role="direct_wait_wrapper",pid=os.getpid()),dict(role="original_wave_supervisor",pid=proc.pid)]
 for parent in PARENTS:
  base=ROOT/"research/reviews/American_C3_short_train2mb_wave_supervision_v2/admitted"/parent
  for stage in ("simulate","infer"):
   w=base/stage/"actual_wait.json"
   if w.exists():
    value=json.loads(w.read_text())
    stage_refs.append(dict(parent_id=parent,stage=stage,wait_ref=ref(w)))
    roster.extend(value.get("owned_process_roster",[]))
 save("actual_supervisor_wait.json",dict(schema="C3_real_supervisor_wait_v1",
      direct_supervisor_pid=proc.pid,wrapper_pid=os.getpid(),actual_direct_supervisor_wait=True,
      returncode=rc,finished_utc=datetime.now(timezone.utc).isoformat(),stage_wait_refs=stage_refs,
      observed_original_process_roster=roster,terminal_log_ref=ref(HERE/"supervisor_terminal.log"),
      original_captured_bytes=len(captured),retained_log_bytes=len(log),log_truncated=len(captured)>len(log),
      parent_preparations_completed=[p for p in PARENTS if (ROOT/"research/reviews/American_C3_short_train2mb_wave_supervision_v2/admitted"/p/"parent_preparation_completed.json").exists()],
      scientific_driver_and_config_unchanged=True,GPU_used=False))
 print(json.dumps({"actual_supervisor_wait":ref(HERE/"actual_supervisor_wait.json"),"returncode":rc}),flush=True)
 return rc
if __name__=="__main__": raise SystemExit(main())

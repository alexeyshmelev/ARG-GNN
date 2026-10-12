"""Resume never-started H038 fits via module launcher, preserving admissions/sources."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,traceback
R=Path("/workspace-SR003.nfs2/ARG-GNN");P=R/"research/reviews/expanded_full_panel_v1_root/full_panel_pyg_fits_v4"
def save(path,value):
 with path.open("x") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
def run():
 runs=[]
 for arm in ["aligned","zero_concordance"]:
  packet=P/arm;old=json.loads((packet/"launch.json").read_text());wait=json.loads((packet/"root_wait.json").read_text())
  assert wait["returncode"]==1 and wait["completed_ref"] is None
  assert "ModuleNotFoundError: No module named 'scripts'" in (packet/"launch.log").read_text()
  spec=json.loads((R/old["input_ref"]["path"]).read_text());assert not (R/spec["output_directory"]).exists()
  command=old["command"];assert command[2]=="scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"
  command=command[:2]+["-m","scripts.launch_reserved_gpu_terminal_retired_scopes_v1"]+command[3:]
  with (packet/"relaunch_v5.log").open("xb") as log:
   child=subprocess.Popen(command,cwd=R,env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4"),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(packet/"relaunch_v5.json",dict(pid=child.pid,command=command,input_ref=old["input_ref"],utc=datetime.now(timezone.utc).isoformat(),repeat_reason="Both earlier wrappers failed at import before lease/output/training. Run frozen wrapper as a module to expose repository scripts namespace",new_reservation_applied=False,scientific_settings_changed=False))
  print(json.dumps(dict(stage="H038_relaunched_as_module",arm=arm,pid=child.pid,gpu=old["gpu"])),flush=True);runs.append((packet,arm,child,spec))
 while runs:
  for packet,arm,child,spec in list(runs):
   rc=child.poll()
   if rc is None:continue
   done=R/spec["output_directory"]/"completed.json"
   value=dict(pid=child.pid,returncode=rc,completed_exists=done.is_file(),utc=datetime.now(timezone.utc).isoformat())
   save(packet/"relaunch_v5_wait.json",value);print(json.dumps(dict(stage="H038_relaunch_exited",arm=arm,**value)),flush=True);runs.remove((packet,arm,child,spec))
  if runs:
   import time;time.sleep(5)
if __name__=="__main__":
 try:run()
 except Exception:print(traceback.format_exc(),flush=True);raise SystemExit(1)

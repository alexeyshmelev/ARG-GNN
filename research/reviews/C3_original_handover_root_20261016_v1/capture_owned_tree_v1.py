"""Capture only the exact original C3 root and its current owned descendants."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path("research/reviews/C3_original_handover_root_20261016_v1")
SEEDS={3042044:476961689,3042104:476963326,3042221:476966238}
DRIVER=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/root_driver_v1.py",sha256="7724766b65ad4463af9c2f8f108ce063a31564c2cd07a044e944fe65b6dc6e46")
INPUT=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/admitted_train_v1/input.json",sha256="e0142a909e411afd7e54403ce36e68200e66123752751b7532e54686643ec23f")
def ref(name):
 p=ROOT/name
 return dict(path=str(name),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def stat(pid):
 p=Path("/proc")/str(pid)
 values=(p/"stat").read_text().rsplit(")",1)[1].split()
 cmd=(p/"cmdline").read_bytes()
 return dict(pid=pid,UID=p.stat().st_uid,start_ticks=int(values[19]),state=values[0],PPID=int(values[1]),PGID=int(values[2]),SID=int(values[3]),
  title=cmd.split(b"\0",1)[0].decode(errors="replace"),cmdline_sha256=hashlib.sha256(cmd).hexdigest())
def capture():
 known={};queue=[(3042044,None)];edges=[]
 while queue:
  pid,parent=queue.pop()
  if pid in known:continue
  row=stat(pid)
  if row["UID"]!=os.getuid():raise ValueError("Unexpected UID in exact owned tree")
  if parent is not None and row["PPID"]!=parent:raise ValueError("Owned child relation changed")
  if pid in SEEDS and row["start_ticks"]!=SEEDS[pid]:raise ValueError("Original PID identity changed")
  known[pid]=row
  if parent is not None:edges.append([parent,pid])
  if len(known)>256:raise ValueError("Bounded known project tree exceeded256")
  children=set()
  for task in (Path("/proc")/str(pid)/"task").iterdir():
   try:children.update(map(int,(task/"children").read_text().split()))
   except FileNotFoundError:raise ValueError("Owned task changed during capture")
  queue.extend((child,pid) for child in sorted(children))
 for pid,start in SEEDS.items():
  if pid not in known or known[pid]["start_ticks"]!=start:raise ValueError("Known original seed absent from owned tree")
 root=known[3042044]
 if root["PGID"]!=3042044 or root["SID"]!=3042044:raise ValueError("Original root dedicated session")
 return known,sorted(edges)
def main():
 p=argparse.ArgumentParser();p.add_argument("--capture",action="store_true");a=p.parse_args()
 if not a.capture:print(json.dumps(dict(status="source_only_no_process_probe")));return 0
 if Path.cwd()!=ROOT:raise ValueError("Exact project cwd")
 if ref(DRIVER["path"])!=DRIVER or ref(INPUT["path"])!=INPUT:raise ValueError("Original source/input bytes changed")
 first,edges=capture();time.sleep(.25);second,edges2=capture()
 keys=("pid","UID","start_ticks","PPID","PGID","SID","cmdline_sha256")
 if edges!=edges2 or {p:tuple(r[k] for k in keys) for p,r in first.items()}!={p:tuple(r[k] for k in keys) for p,r in second.items()}:raise ValueError("Owned identities did not stabilize")
 result=dict(schema="C3_original_pre_cancellation_owned_roster_root_v1",root_authorizer="root_agent",observed_utc=datetime.now(timezone.utc).isoformat(),
  root_pid=3042044,root_start_ticks=SEEDS[3042044],launcher_pid=3042104,launcher_start_ticks=SEEDS[3042104],
  model_worker_pid=3042221,model_worker_start_ticks=SEEDS[3042221],root_UID=os.getuid(),physical_gpu=4,native_GPU5_preserved=True,
  exact_original_driver_ref=DRIVER,exact_original_input_ref=INPUT,original_exec_session_id=50244,
  source_ref=ref(HERE/"capture_owned_tree_v1.py"),known_seeds=[dict(pid=p,start_ticks=t) for p,t in SEEDS.items()],
  complete_current_owned_task_children_capture=True,two_stable_identity_snapshots=True,owned_only=True,unrelated_host_inventory_performed=False,
  identities=list(second.values()),owned_edges=edges2,owned_process_groups=sorted(set(r["PGID"] for r in second.values())),
  signal_sent=False,drain_claimed=False)
 output=ROOT/HERE/"pre_cancellation_owned_roster.root_v1.json"
 with output.open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(dict(roster_ref=ref(output.relative_to(ROOT)),identities=len(second),groups=result["owned_process_groups"])),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(main())

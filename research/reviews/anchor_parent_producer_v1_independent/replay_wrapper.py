from pathlib import Path
import json,os,sys
O=Path("/workspace-SR003.nfs2/ARG-GNN/research/reviews/anchor_parent_producer_v1_independent")
R=O.parents[2];events=[];allowed_cpu_library_maps=[]
def audit(event,args):
 if event=="open" and not isinstance(args[0],int):
  p=Path(os.path.abspath(os.fsdecode(args[0])));mode,flags=args[1:3];writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  own=p==O or O in p.parents
  if writing and not own:events.append("outside write "+str(p));raise RuntimeError(events[-1])
  if not writing and p==Path("/proc/self/maps"):
   allowed_cpu_library_maps.append("original configure_cpu/threadpoolctl own CPU library inspection");return
  if not writing and not own:
   for base in (R/"data",R/"experiments/results",R/".runtime/conda/envs",R/".runtime/gpup",R/".runtime/locks",R/".runtime/storage",Path("/proc")):
    if p==base or base in p.parents:events.append("actual dataset/runtime/resource read "+str(p));raise RuntimeError(events[-1])
 if event in ("subprocess.Popen","os.system","os.posix_spawn","socket.connect","socket.bind"):events.append(event);raise RuntimeError(event)
sys.addaudithook(audit)
p=Path(sys.argv[1]);exec(compile(p.read_text(),str(p),"exec"),{"__file__":str(p),"__name__":"__main__"})
(p.parent/"blocked_events.json").write_text(json.dumps(events,indent=2)+"\n")
assert not events
(p.parent/"reviewer_audit.json").write_text(json.dumps({"forbidden_events":events,"allowed_own_CPU_library_map_reads":len(allowed_cpu_library_maps),"actual_dataset_GPU_fit_budget_or_Git":False},indent=2)+"\n")

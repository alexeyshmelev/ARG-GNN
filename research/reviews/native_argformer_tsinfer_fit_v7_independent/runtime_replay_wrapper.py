from pathlib import Path
import builtins,json,os,sys
R=Path("/workspace-SR003.nfs2/ARG-GNN");O=R/"research/reviews/native_argformer_tsinfer_fit_v7_independent";events=[]
def audit(event,args):
 if event=="open" and not isinstance(args[0],int):
  p=Path(os.path.abspath(os.fsdecode(args[0])));mode,flags=args[1:3]
  writing=(isinstance(mode,str) and any(x in mode for x in "wax+")) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  own=p==O or O in p.parents
  if writing and not own:events.append("outside write "+str(p));raise RuntimeError(events[-1])
  if not writing and not own:
   for base in (R/"data",R/"experiments/results",R/".runtime/conda/envs",R/".runtime/gpup",R/".runtime/locks",R/".runtime/storage",Path("/proc")):
    if p==base or base in p.parents:events.append("actual read "+str(p));raise RuntimeError(events[-1])
 if event in ("socket.connect","subprocess.Popen","os.system","os.posix_spawn"):events.append(event);raise RuntimeError(event)
sys.addaudithook(audit)
original_import=builtins.__import__
def imported(name,*a,**k):
 if name=="torch" and getattr(sys.modules.get("torch"),"__file__",None):
  fake=Path(sys.modules["torch"].__file__)
  if O in fake.parents:return original_import(name,*a,**k)
 if name.split(".")[0] in ("torch","numpy","tskit","tsinfer","composer","transformers","streaming"):
  events.append("actual scientific import "+name);raise RuntimeError(events[-1])
 return original_import(name,*a,**k)
builtins.__import__=imported
p=O/"runtime_parse_replay.py"
exec(compile(p.read_text(),str(p),"exec"),{"__file__":str(p),"__name__":"__main__"})
assert not events
(O/"runtime_replay_audit.json").write_text(json.dumps({"forbidden_events":events,"actual_native_prefix_scientific_data_model_GPU_budget_or_lease_access":False},indent=2)+"\n")

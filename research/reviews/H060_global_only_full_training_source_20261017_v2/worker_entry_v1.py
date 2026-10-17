"""Inactive actual CPU worker. No scientific import before admitted ROOT gate."""
from pathlib import Path
import argparse,hashlib,importlib,json,io,os,re,resource,stat,sys,time
from research.reviews.H060_global_only_full_training_source_20261017_v2 import root_driver_v1 as common
c=None # Bound to the one fresh private contract by common.worker after source authentication.
ROOT=common.ROOT
def jsonable_runtime_v1(value, torch, np):
 """Bind serialization to the actual initialized local runtime, never lazy legacy globals."""
 if isinstance(value,torch.Tensor):return jsonable_runtime_v1(value.detach().cpu().numpy(),torch,np)
 if isinstance(value,np.ndarray):return jsonable_runtime_v1(value.tolist(),torch,np)
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {str(k):jsonable_runtime_v1(v,torch,np) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [jsonable_runtime_v1(v,torch,np) for v in value]
 return value

class CappedBuffer(io.BytesIO):
 def __init__(self,cap):super().__init__();self.cap=cap
 def write(self,data):
  c.require(self.tell()+len(data)<=self.cap,"Actual serialization exceeds prospective file byte cap; no disk write")
  return super().write(data)
class BoundedWriter:
 """Original exclusive V9 writer boundaries with capped actual serialization."""
 def __init__(self,s,q,g,fast,torch,jsonable):
  self.s=s;self.q=q;self.g=g;self.fast=fast;self.torch=torch;self.jsonable=jsonable
  self.path=c.local(s["output_directory"]);self.tmp=q["packet_directory"]+"/worker_aux/tmp"
  self.used=8192;self.json_used=0;self.torch_files=0;self.names=set()
  c.require(not self.path.exists(),"Exclusive fresh matched output")
  g.budget_check_v9(s["storage"],boundary=True,required_write_bytes=8192);self.path.mkdir()
 def save(self,name,value,kind="json"):
  self.fast("before_actual_bounded_serialization")
  c.require(isinstance(name,str)and Path(name).name==name and name not in self.names,"One exclusive flat product name")
  allowed_json={"started.json","TRAIN_observation_barrier.json","TRAIN_external_target_joins.json","matched_initialization.json",
   "completed_training.json","failed_training.json","worker_failure.json","completed.json"}
  if kind=="torch":
   c.require(re.fullmatch(r"checkpoint_pair_epoch_(005|010|015|020|025|030)\.pt",name)is not None or name=="failure_last_complete_pair.pt",
    "Six paired checkpoints and at most one failure boundary only")
   c.require(self.torch_files<c.CAPS["maximum_torch_files"],"At most7full-state files; no per-update series")
   buffer=CappedBuffer(c.CAPS["checkpoint_serialized_bytes"]);self.torch.save(value,buffer);data=buffer.getvalue()
  else:
   c.require(kind=="json"and name in allowed_json,"Exact bounded training/operational metadata products")
   data=(json.dumps(self.jsonable(value),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
   c.require(self.json_used+len(data)<=c.CAPS["total_JSON_bytes"],"Aggregate actual JSON serialization cap; no partial write")
  planned=len(data)+8192
  c.require(self.used+planned<=c.CAPS["output_bytes"],"Prospective complete output+metadata byte cap")
  self.fast("before_actual_guarded_product_write")
  self.g.budget_check_v9(self.s["storage"],boundary=True,required_write_bytes=planned)
  available=os.statvfs(self.path);c.require(planned<=available.f_bavail*available.f_frsize,"Actual available filesystem bytes")
  path=self.path/name;fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,"wb")as f:
   f.write(data);f.flush();os.fsync(f.fileno());identity=os.fstat(f.fileno())
  named=path.lstat()
  c.require(stat.S_ISREG(named.st_mode)and named.st_nlink==1 and(named.st_dev,named.st_ino)==(identity.st_dev,identity.st_ino),"Exact newly created output object")
  self.names.add(name);self.used+=planned
  if kind=="torch":self.torch_files+=1
  else:self.json_used+=len(data)
  return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
def worker(record,gate_fd):
 s,q,cfg=c.input_spec(record)
 c.require(type(gate_fd)is int and gate_fd>=0,"Actual parent-owned startup gate")
 try:c.require(os.read(gate_fd,1)==b"G","ROOT registered pidfd ownership before scientific work")
 finally:os.close(gate_fd)
 c.require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
  and sys.prefix==str(ROOT/".runtime/env")and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Qualified -B -s CPU-only interpreter")
 for key in ("TMPDIR","TMP","TEMP"):c.require(os.environ.get(key)==str(c.local(cfg["short_tmp_directory"])),"Charged private TMP only")
 c.require(set(q["cpu_ids"])<=os.sched_getaffinity(0),"ROOT supplied actual32CPU capacity")
 os.sched_setaffinity(0,set(q["cpu_ids"]))
 g=c.guard(True);c.verify_actual_roles(s,q,g)
 g.validate_storage_v9(s,dict(output_cap_bytes=c.CAPS["reservation_bytes"]),s["storage"],source_ref=c.ref(c.DRIVER),
  config_ref=s["refs"]["config"],runtime_ref=s["runtime_ref"])
 # All model/data/checkpoint imports are after real source/runtime/reservation/owned gate checks.
 import numpy as np,torch,torch_geometric
 from arggnn.american_c3_common_v1 import runtime_report
 c.require(runtime_report()==c.read(c.BASE_RUNTIME),"Exact prequalified shared CPU package report")
 torch.set_num_threads(32);torch.set_num_interop_threads(1)
 c.require(torch.get_default_dtype()==torch.float32 and not torch.cuda.is_initialized(),"Full FP32/zeroCUDA; no physical lease")
 started=time.monotonic()
 def fast(event="TreeLoom_actual_matched_CPU_boundary"):
  c.current(s,150)
  c.require(time.monotonic()-started<c.CAPS["scientific_seconds"],"Full24h scientific failstop; no shortcut")
  c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=c.CAPS["rss_bytes"],"128GiB main RSS; finite outer covers complete owned tree")
  c.require(not torch.cuda.is_initialized()and torch.get_num_threads()==32 and set(os.sched_getaffinity(0))==set(q["cpu_ids"]),"CPU/32threads/affinity remain exact")
  return dict(event=event,no_storage_scan_or_journal_call=True)
 writer=BoundedWriter(s,q,g,fast,torch,lambda value:jsonable_runtime_v1(value,torch,np))
 writer.save("started.json",dict(status="started_TreeLoom_actual_full_matched_CPU_training",input_ref=record,
  additional_updates_per_arm=600,total_additional_updates=1200,first_scheduled_updates_timed_inline=True,
  standalone_token_qualification=False,device="cpu",dtype="float32",CPU_threads=32,
  output_serialization_is_failstop_not_fit_guarantee=True,physical_GPU_lease=None,TEST_deferred=True))
 try:
  b=c.binding();c.raw(c.TARGET_HOOK);hooks=importlib.import_module(b["target_hook_module"])
  ctx=c.context(q["training_context_ref"])
  # Original metadata resolver remains TRAIN-only; new target factory is deferred until runner all20 cache barrier.
  meta=hooks.actual_train_metadata_v1(ctx["actual_train_bindings"])
  def train_truth(parent):
   c.require(parent in c.TRAIN,"Original20TRAIN-only truth capability; VAL/TEST impossible")
   return meta[parent]["private_truth_ref"]
  runner=importlib.import_module(b["runner_module"])
  fast("before_actual_full_matched_runner")
  result=runner.run_train_v1(q["scientific_config"],parent_plan_refs=ctx["parent_plan_refs"],
   artifact_acceptance_ref=c.ACCEPTANCE,writer=writer,device="cpu",worker_tmp_directory=cfg["short_tmp_directory"],
   resolve_train_truth=train_truth,runtime_guard=fast,actual_train_bindings=ctx["actual_train_bindings"],resume_ref=None)
  fast("after_full600additional_eacharm")
  return writer.save("completed.json",dict(status=c.STATUSES[c.PHASE],parent_id=c.PARENT,phase=c.PHASE,
   input_ref=record,root_driver_ref=c.ref(c.DRIVER),result_ref=result,additional_updates_per_arm=600,total_additional_updates=1200,
   device="cpu",physical_gpu=None,TEST_deferred=True,automatic_close_or_credit=False))
 except BaseException as error:
  try:writer.save("worker_failure.json",dict(status="failed_TreeLoom_actual_matched_CPU_training",exception_type=type(error).__name__,
   message=str(error)[:2000],no_success_or_automatic_retry_claimed=True))
  except BaseException:pass
  raise
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worker-ref");p.add_argument("--gate-fd",type=int);a=p.parse_args(argv)
 if not a.worker_ref:
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",models_imported=False,actual_training_updates=0)));return 0
 common.need(a.gate_fd is not None,"Actual owned startup gate required")
 result=common.worker(json.loads(a.worker_ref),a.gate_fd);print(json.dumps(dict(completed_ref=result)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())

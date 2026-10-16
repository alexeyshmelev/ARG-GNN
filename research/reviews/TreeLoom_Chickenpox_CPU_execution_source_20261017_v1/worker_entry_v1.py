"""Inactive actual CPU worker. No scientific import before admitted ROOT gate."""
from pathlib import Path
import argparse,hashlib,importlib,json,io,os,re,resource,stat,sys,time
from research.reviews.TreeLoom_Chickenpox_CPU_execution_source_20261017_v1 import root_contract_v1 as c
ROOT=c.ROOT
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
  allowed_json={"started.json","six_selection_barrier.json","completed_TRAIN.json","worker_failure.json","completed.json"}
  allowed_json|={"selection_seed%d_%s.json"%(seed,arm)for seed in c.SEEDS for arm in c.ARMS}
  if kind=="torch":
   c.require(re.fullmatch(r"selected_seed(101|202|303)_(full|no_memory)\.pt",name)is not None,
    "Exactly the six selected CPU states; no optimizer or failure checkpoints")
   c.require(self.torch_files<c.CAPS["maximum_torch_files"],"At most six selected-state files")
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
 started=time.monotonic()
 c.require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
  and sys.prefix==str(ROOT/".runtime/env")and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Qualified -B -s CPU-only interpreter")
 for key in ("TMPDIR","TMP","TEMP"):c.require(os.environ.get(key)==str(c.local(cfg["short_tmp_directory"])),"Charged private TMP only")
 c.require(q["cpu_ids"]==[223]and set(q["cpu_ids"])<=os.sched_getaffinity(0),"ROOT supplied exact CPU223 capacity")
 os.sched_setaffinity(0,{223})
 g=c.guard(True);c.verify_actual_roles(s,q,g)
 g.validate_storage_v9(s,dict(output_cap_bytes=c.CAPS["reservation_bytes"]),s["storage"],source_ref=c.ref(c.DRIVER),
  config_ref=s["refs"]["config"],runtime_ref=s["runtime_ref"])
 # Rehash the admitted source closure after the real owned gate, before all scientific imports.
 c.sources()
 from arggnn.american_c3_common_v1 import runtime_report
 c.require(runtime_report()==c.read(c.BASE_RUNTIME),"Exact prequalified shared CPU package report; no numerical probe")
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 c.require(torch.get_default_dtype()==torch.float32 and not torch.cuda.is_initialized(),"FP32/zero CUDA; no physical lease")
 def fast(event="TreeLoom_actual_Chickenpox_CPU_boundary"):
  c.current(s,150)
  c.require(time.monotonic()-started<c.CAPS["scientific_seconds"],"Finite6900s work failstop; no scientific shortcut")
  c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=c.CAPS["rss_bytes"],
   "4GiB worker RSS failstop; finite outer also covers whole owned tree")
  c.require(not torch.cuda.is_initialized()and torch.get_num_threads()==1 and torch.get_num_interop_threads()==1
   and set(os.sched_getaffinity(0))=={223},"CPU223/one Torch thread/zero CUDA remain exact")
  return dict(event=event,no_storage_scan_or_journal_call=True)
 writer=BoundedWriter(s,q,g,fast,torch,lambda value:value)
 writer.save("started.json",dict(status="started_TreeLoom_Chickenpox_TRAIN_selection",input_ref=record,
  seeds=list(c.SEEDS),arms=list(c.ARMS),epochs_each=40,updates_each=800,total_optimizer_updates=4800,
  first_scheduled_updates_timed_inline=True,standalone_token_qualification=False,
  device="cpu",dtype="float32",CPU_threads=1,cpu_ids=[223],physical_GPU_lease=None,
  output_serialization_is_failstop_not_fit_guarantee=True,TEST_deferred=True,scoring_authorized=False,
  legacy_GRU_status="unavailable_payload_not_accepted"))
 try:
  b=c.binding();runner=importlib.import_module(b["runner_module"])
  fast("before_real_full_six_fit_runner")
  result=runner.run_train_v1(q["scientific_config"],writer=writer,runtime_guard=fast)
  fast("after_all4800real_updates_and_six_selections")
  return writer.save("completed.json",dict(status=c.STATUSES[c.PHASE],parent_id=c.PARENT,phase=c.PHASE,
   input_ref=record,root_driver_ref=c.ref(c.DRIVER),result_ref=result,total_optimizer_updates=4800,
   device="cpu",physical_gpu=None,TEST_deferred=True,scoring_authorized=False,
   separate_ROOT_TEST_release_required=True,automatic_close_or_credit=False))
 except BaseException as error:
  try:writer.save("worker_failure.json",dict(status="failed_TreeLoom_Chickenpox_TRAIN_selection",
   exception_type=type(error).__name__,message=str(error)[:2000],no_success_or_automatic_retry_claimed=True))
  except BaseException:pass
  raise

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worker-ref");p.add_argument("--gate-fd",type=int);a=p.parse_args(argv)
 if not a.worker_ref:
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",models_imported=False,actual_training_updates=0)));return 0
 c.require(a.gate_fd is not None,"Actual owned startup gate required")
 result=worker(json.loads(a.worker_ref),a.gate_fd);print(json.dumps(dict(completed_ref=result)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())

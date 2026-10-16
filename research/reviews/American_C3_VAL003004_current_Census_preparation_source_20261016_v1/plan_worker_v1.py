"""Executable current980 C3 GNN stages. Default inert; no science imports before authorization."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,resource,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/REL)]
from research.reviews.American_C3_VAL003004_current_Census_preparation_source_20261016_v1 import root_contract_v1 as c
def execute(record):
 spec=c.read(record,cap=33554432);req,caps,_=c.validate_request(spec["root_stage_request_ref"]);stage=req["phase"];c.active(spec)
 c.require(spec["root_driver_ref"]==c.ref(c.DRIVER) and spec["job_id"] and spec["phase"]==stage,"Exact current input/driver")
 for path,sha in spec["source_sha256"].items():c.require(c.ref(path)["sha256"]==sha,"Actual admitted scientific/runtime source closure")
 GPU=stage in c.GPU_PHASES;lease=None
 if GPU:
  from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
  lease=validate_lease();apply_and_validate_title()
  c.require(lease["physical_gpu"]==req["physical_gpu"] and lease["visible_gpu_uuid"]==req["GPU_UUID"],"Actual root-selected UUID/lease/title before Torch")
 else:c.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Explicit CPU stage visibility")
 from arggnn.american_c3_common_v1 import runtime_report
 expected=dict(c.read(c.RUNTIME_REF),CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
 c.require(runtime_report()==expected,"Exact existing qualified shared runtime metadata")
 guard,memo=c.consumer_guard();storage=c.read(spec["refs"]["storage"]);cfg=c.read(spec["refs"]["config"])
 c.require(cfg==c.make_config(req,caps),"Exact operational config")
 closure=c.read(spec["transitive_source_closure_ref"],cap=33554432)
 c.require(closure["source_sha256"]==spec["source_sha256"] and closure["root_stage_request_ref"]==spec["root_stage_request_ref"]
  and closure["bindings"]==req["bindings"] and closure["native_MLM_dependency"] is False,"Exact admitted transitive source/input closure")
 required=[spec["transitive_source_closure_ref"],spec["root_stage_request_ref"],spec["caps_ref"],
  spec["refs"]["config"],spec["refs"]["runtime"],c.ref(c.SOURCE),c.ref(c.DRIVER),c.ref(c.ENTRY)]
 c.require(all(r in storage["reservation"]["consumer_bindings"] for r in required),"Actual root input/config/runtime/caps/closure reserved")
 main_ref=c.verify_shim() if GPU else c.ref(c.ENTRY)
 guard.validate_storage_v9(spec,dict(output_cap_bytes=caps["reservation_cap_bytes"]),storage,
  source_ref=main_ref,config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
 # Original writer and runtime. Imports follow actual title/lease for GPU phases.
 import numpy as np,torch,torch_geometric
 from arggnn import census_tsinfer_fit_v1 as fit
 fit.np=np;fit.torch=torch
 c.require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Pinned current Torch/PyG")
 writer=guard.PilotOutputV9(spec["output_directory"],storage,fit.ConsumerReaderV1())
 from arggnn.biological_candidate import configure_fit,check_fit_lease
 guard.budget_check_v9(storage,boundary=True,required_write_bytes=8*2**20)
 device,runtime,qualified=configure_fit("cuda" if GPU else "cpu",c.read(c.BASE_CONFIG_REF),writer.path)
 if GPU:
  c.require(qualified==lease,"Original GPU runtime retains inherited lease")
  torch.cuda.set_per_process_memory_fraction(min(1.,caps["max_cuda_reserved_bytes"]/torch.cuda.get_device_properties(device).total_memory),device)
 writer.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in writer.path.rglob("*") if p.is_file())
 c.require(writer.used<=8*2**20,"Bounded existing runtime qualification artifact allowance")
 writer.limit=caps["phase_write_budget_bytes"]
 start=time.monotonic()
 def fast(event="C3_GNN_runtime_boundary"):
  c.active(spec);c.require(time.monotonic()-start<caps["operational_deadline_seconds"],"Full fixed-condition deadline; no shortcut")
  c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=caps["max_cpu_rss_bytes"],"Current-process RSS cap; workers remain0")
  if GPU:
   check_fit_lease(qualified);fit.require_logical_job_v1(qualified,spec)
   c.require(torch.cuda.max_memory_reserved()<=caps["max_cuda_reserved_bytes"],"Actual Torch CUDA reserved cap")
  return dict(event=event,no_storage_scan_or_journal_call=True)
 class Writer:
  path=writer.path
  def save(self,name,value,kind="json"):
   fast("before_GNN_synchronous_product_write")
   c.require(kind=="json","Plan emits JSON metadata only")
   planned=len(guard.budget.v4.encoded(fit._jsonable_v1(value)))+1
   if name==c.PARENT+"_block_index.json":
    c.require(planned<=b["plan_config"]["output_cap_bytes"],"Original20,000,000-byte plan JSON cap")
   c.require(writer.used+planned+8192<=caps["phase_write_budget_bytes"],"Original aggregate20MiB stage write cap")
   return writer.save(name,value,kind=kind)
  def json(self,name,value):return self.save(name,value)
  def npz(self,name,**arrays):return self.save(name,arrays,kind="npz")
 w=Writer();h=c.hooks();b=req["bindings"]
 fast();w.save("started.json",dict(status="started_C3_GNN_current980",stage=stage,input_ref=record,
  runtime=runtime,native_MLM_dependency=False,TEST_deferred=True,root_driver_ref=c.ref(c.DRIVER)))
 c.require(stage==c.PHASE,"No TRAIN/prediction/score/label/checkpoint/GPU capability")
 result=h.build_plan_stage_v1(b["plan_config"],parent_id=b["parent_id"],writer=w,runtime_guard=fast)
 fast("completed_GNN_stage")
 return w.save("completed.json",dict(status=c.STATUSES[stage],stage=stage,parent_id=c.PARENT,
  input_ref=record,root_driver_ref=c.ref(c.DRIVER),result_ref=result,native_MLM_dependency=False,TEST_deferred=True))
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--input");p.add_argument("--expected-input-sha256");a=p.parse_args(argv)
 if not a.execute:
  print(json.dumps(dict(status="source_only_current980_C3_GNN",phases=list(c.PHASES),native_MLM_dependency=False,actual_execution=False)));return 0
 c.require(a.input and a.expected_input_sha256,"Actual admitted root input path/SHA")
 result=execute(dict(path=a.input,sha256=a.expected_input_sha256));print(json.dumps(dict(completed_ref=result)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())

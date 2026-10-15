"""Executable current980 C3 GNN stages. Default inert; no science imports before authorization."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,resource,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/REL)]
import root_contract_v1 as c
def execute(record):
 spec=c.read(record,cap=33554432);c.active(spec);req=c.read(spec["root_stage_request_ref"],cap=33554432);stage=req["phase"]
 c.require(spec["root_driver_ref"]==c.ref(c.DRIVER) and spec["job_id"] and spec["phase"]==stage,"Exact current input/driver")
 for path,sha in spec["source_sha256"].items():c.require(c.ref(path)["sha256"]==sha,"Actual admitted scientific/runtime source closure")
 GPU=stage in c.GPU_PHASES;lease=None
 if GPU:
  from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
  lease=validate_lease();apply_and_validate_title()
  c.require(lease["physical_gpu"]==req["physical_gpu"] and lease["visible_gpu_uuid"]==req["GPU_UUID"],"Actual root-selected UUID/lease/title before Torch")
 else:c.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Explicit CPU stage visibility")
 from research.reviews.C3_GNN_prefetch_provider_20261015_v1.worker_entry_integration_api_v1 import bootstrap_before_torch_v1
 ipc_receipt=bootstrap_before_torch_v1()
 req,caps,_=c.validate_request(spec["root_stage_request_ref"])
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
  c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=caps["max_cpu_rss_bytes"],"Main-process peak RSS cap; provider and outer supervisor separately cover owned workers")
  if GPU:
   check_fit_lease(qualified);fit.require_logical_job_v1(qualified,spec)
   c.require(torch.cuda.max_memory_reserved()<=caps["max_cuda_reserved_bytes"],"Actual Torch CUDA reserved cap")
  return dict(event=event,no_storage_scan_or_journal_call=True)
 class Writer:
  path=writer.path
  def save(self,name,value,kind="json"):
   fast("before_GNN_synchronous_product_write");return writer.save(name,value,kind=kind)
  def json(self,name,value):return self.save(name,value)
  def npz(self,name,**arrays):return self.save(name,arrays,kind="npz")
 w=Writer();h=c.hooks();b=req["bindings"]
 fast();w.save("started.json",dict(status="started_C3_GNN_prefetch_isolated980",stage=stage,input_ref=record,
  runtime=runtime,IPC_bootstrap=ipc_receipt,native_MLM_dependency=False,TEST_deferred=True,root_driver_ref=c.ref(c.DRIVER)))
 if stage=="train":
  config=c.read(cfg["scientific_training_config_ref"]);config["execution_enabled"]=True
  meta=h.actual_train_metadata_v1(b["actual_train_bindings"])
  def train_truth(parent):
   c.require(parent in c.TRAIN,"Only all20 short TRAIN truth");return meta[parent]["private_truth_ref"]
  from research.reviews.C3_GNN_prefetch_isolated980_root_20261015_v2.tracking_factory_v1 import TrackingObservationProviderFactoryV1
  factory=TrackingObservationProviderFactoryV1(operational_caps=cfg["observation_prefetch"],resource_check=fast)
  training_failure=None
  try:
   result=h.run_train_stage_v1(config,parent_plan_refs=b["parent_plan_refs"],actual_train_bindings=b["actual_train_bindings"],
    writer=w,device=device,worker_tmp_directory=cfg["short_tmp_directory"],resolve_train_truth=train_truth,
    runtime_guard=fast,resume_ref=None,observation_provider_factory=factory)
  except BaseException as error:
   training_failure=error;raise
  finally:
   try:w.save("observation_prefetch_all_worker_drains.json",factory.finalize(training_failed=training_failure is not None))
   except BaseException as drain_error:
    if training_failure is None:raise
    if hasattr(training_failure,"add_note"):training_failure.add_note("Prefetch drain manifest failure: "+str(drain_error)[:1000])
 elif stage.startswith("plan_val_"):
  result=h.build_plan_stage_v1(b["plan_config"],parent_id=b["parent_id"],writer=w,runtime_guard=fast)
 elif stage.startswith("predict_val_"):
  train=c.read(b["training"]["completion_ref"])["result_ref"]
  result=h.predict_one_original_VAL_v1(b["plan_ref"],parent_id=b["parent_id"],training_completion_ref=train,
   writer=w,device=device,worker_tmp_directory=cfg["short_tmp_directory"],runtime_guard=fast)
 elif stage=="matrix":
  train=c.read(b["training"]["completion_ref"])["result_ref"];seals=[]
  for parent in c.VAL:
   envelope=c.read(b["VAL_predictions"][parent]["completion_ref"]);six=c.read(envelope["result_ref"])
   c.require(six["parent_id"]==parent and six["training_completion_ref"]==train and len(six["prediction_seal_refs"])==6,"Exact own original5VAL six seals")
   seals.extend(six["prediction_seal_refs"])
  result=h.seal_original_VAL_matrix_stage_v1(seals,training_completion_ref=train,writer=w,runtime_guard=fast)
 else:
  matrix=c.read(b["matrix"]["completion_ref"])["result_ref"]
  def val_truth(parent):
   c.require(parent in c.VAL,"Only original5VAL truth, TEST deferred")
   return c.read(b["VAL_truth_capability"][parent])["private_truth_ref"]
  result=h.score_and_select_stage_v1(matrix,writer=w,resolve_val_truth=val_truth,runtime_guard=fast)
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

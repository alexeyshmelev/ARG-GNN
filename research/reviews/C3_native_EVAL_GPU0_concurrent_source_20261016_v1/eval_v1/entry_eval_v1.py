"""Narrow8GiB operational entry; existing exact C3 prediction is unchanged."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,resource,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v1/eval_v1"
SOURCE=HERE+"/entry_eval_v1.py"
CONTRACT=HERE+"/contract_eval_v1.py"
def _contract():
    p=ROOT/CONTRACT
    sp=importlib.util.spec_from_file_location("_C3_EVAL_exact_contract",p)
    c=importlib.util.module_from_spec(sp);sys.modules[sp.name]=c;sp.loader.exec_module(c);return c
def invoke_job_v1(method):
    c=_contract();spec,caps,storage=c.validate_method_v1(method);q,_=c.validate_request_v1(spec["root_stage_request_ref"])
    start=time.monotonic()
    c.require(Path(__file__).resolve()==ROOT/SOURCE and Path.cwd().resolve()==ROOT,"Exact runtime source/cwd")
    c.require("torch" not in sys.modules and "torch_geometric" not in sys.modules,"EVAL8GiB allocator must precede scientific imports")
    c.require(os.getpid()==os.getpgrp()==os.getsid(0),"Dedicated owned pool method session leader")
    c.require(os.environ.get("ARGGNN_PHYSICAL_GPU")=="0" and os.environ.get("CUDA_VISIBLE_DEVICES")==c.GPU_UUID and os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==method["job_id"] and os.environ.get("ARGGNN_JOB_NAME","").startswith("gpu0_shared_v1:"),"Inherited logical GPU0 pool job lineage")
    capref=json.loads(os.environ.get("ARGGNN_NATIVE_GPU0_MONITOR_CAPABILITY_REF","null"));cap=c.read(capref)
    c.require(cap.get("schema")=="GPU0_declared_logical_pool_capability_v1" and cap.get("status")=="active" and cap.get("root_authorizer")=="root_agent" and cap.get("physical_gpu")==0 and cap.get("gpu_uuid")==c.GPU_UUID and time.time()<cap["expires_unix"] and cap.get("prior_controller_completion_ref")==q["prior_controller_completion_ref"],"Actual unexpired fresh successor capability")
    envelope=c.read(cap["jobs"][method["job_id"]]["input_ref"]);request=c.read(envelope["request_ref"])
    c.require(request["method_job"]==method and request["slot"]=="C3_EVAL_"+q["parent"]["parent_id"][-3:] and c.ref(SOURCE) in cap["source_refs"].values(),"Exact finite declared EVAL method/capability/source")
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    import arggnn.native_argformer_exec as inherited_guard
    c.require(getattr(inherited_guard,"GPU0_ADAPTER_SOURCE",None)==c.GPU_GUARD_REF["path"] and Path(inherited_guard.__file__).resolve()==ROOT/c.GPU_GUARD_REF["path"],"Exact inherited reviewed GPU0 guard module")
    lease=validate_lease();apply_and_validate_title()
    c.require(lease["physical_gpu"]==0 and lease["visible_gpu_uuid"]==c.GPU_UUID,"Inherited exact GPU0 lease before Torch")
    guard=c.capacity_guard_v1();storage=c.read(method["storage_ref"]);cfg=c.read(spec["refs"]["config"])
    guard.validate_storage_v9(spec,dict(output_cap_bytes=caps["reservation_cap_bytes"]),storage,source_ref=c.ref(SOURCE),config_ref=spec["refs"]["config"],runtime_ref=c.RUNTIME_REF)
    import numpy as np,torch,torch_geometric
    c.require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Pinned current Torch/PyG")
    c.require(torch.cuda.device_count()==1,"Exactly leased single visible CUDA0")
    device=torch.device("cuda:0")
    total=torch.cuda.get_device_properties(device).total_memory
    c.require(0<caps["max_cuda_reserved_bytes"]<=total,"Explicit8GiB device allocator bound")
    # Installed before configure_fit creates even its qualification tensor.
    # Never invoke oldworker or its50GB set_per_process_memory_fraction call.
    torch.cuda.set_per_process_memory_fraction(caps["max_cuda_reserved_bytes"]/total,device)
    from arggnn import census_tsinfer_fit_v1 as fit
    fit.np=np;fit.torch=torch
    writer=guard.PilotOutputV9(spec["output_directory"],storage,fit.ConsumerReaderV1())
    from arggnn.biological_candidate import configure_fit,check_fit_lease
    device,runtime,qualified=configure_fit("cuda",c.read(c.BASE_CONFIG_REF),writer.path)
    c.require(qualified==lease,"Qualification retains actual inherited lease")
    from arggnn.american_c3_common_v1 import runtime_report
    expected=dict(c.read(c.RUNTIME_REF),CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    c.require(runtime_report()==expected,"Exact qualified shared scientific runtime")
    writer.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in writer.path.rglob("*") if p.is_file())
    c.require(writer.used<=8*2**20,"Bounded own runtime qualification artifacts")
    def fast(event="C3_EVAL_boundary"):
        c.active(spec);c.active(q)
        c.require(time.monotonic()-start<caps["operational_deadline_seconds"],"EVAL operational deadline")
        c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=caps["max_cpu_rss_bytes"],"EVAL owned-process RSS cap; loaders use workers0")
        check_fit_lease(qualified);fit.require_logical_job_v1(qualified,spec)
        c.require(torch.cuda.max_memory_reserved(device)<=8*2**30,"Additional strict8GiB EVAL CUDA cap")
        return dict(event=event,no_accounting_or_project_scan=True)
    class Writer:
        path=writer.path
        def save(self,name,value,kind="json"):
            fast("before_EVAL_product_write")
            record=writer.save(name,value,kind=kind)
            c.require(writer.used<=caps["phase_write_budget_bytes"],"Bounded EVAL product bytes")
            return record
        def json(self,name,value):return self.save(name,value)
    w=Writer()
    fast("before_original_C3_EVAL_hook")
    h=c.load(c.HOOK_REF,"_C3_EVAL_unchanged_prediction_hooks")
    result=h.predict_one_original_VAL_v1(q["parent"]["plan_ref"],parent_id=q["parent"]["parent_id"],training_completion_ref=q["training"]["training_completion_ref"],writer=w,device=device,worker_tmp_directory=q["short_tmp_directory"],runtime_guard=fast)
    fast("after_original_all6_EVAL")
    done=dict(status="completed_C3_full30Mb_EVAL_one_original_VAL_all6",schema="C3_full30Mb_EVAL_completion_v1",phase="eval",original_prediction_stage="predict_val_"+q["parent"]["parent_id"][-3:],parent_id=q["parent"]["parent_id"],result_ref=result,training_completion_ref=q["training"]["training_completion_ref"],root_stage_request_ref=spec["root_stage_request_ref"],root_driver_ref=c.ref(c.ADMISSION),main_source_ref=c.ref(SOURCE),input_ref=method["input_ref"],private_truth_opened=False,original5VAL_all30_barrier_still_required=True,TEST_deferred=True)
    w.save("completed.json",done)
    return 0
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--input");p.add_argument("--expected-input-sha256");a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="source_only_C3_EVAL",execution=False)));return 0
    if not a.input or not a.expected_input_sha256:raise ValueError("Exact actual EVAL input required")
    c=_contract();return invoke_job_v1(c.method_v1(dict(path=a.input,sha256=a.expected_input_sha256)))
if __name__=="__main__":raise SystemExit(main())

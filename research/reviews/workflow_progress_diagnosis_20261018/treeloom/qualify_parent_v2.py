"""ROOT-invoked one existing full AMtrain000 CPU qualification; never launches work."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"src"))


def qualify(spec,*,resource_check):
    from types import SimpleNamespace
    args=SimpleNamespace(parent=spec["parent_id"],output_directory=spec["output_directory"],
        max_rss_bytes=spec["max_rss_bytes"],max_seconds=spec["max_seconds"],
        max_output_bytes=spec["max_output_bytes"])
    from .resident_pool_v2 import need,exact_signature,pack_bundle,restore_bundle
    from .prepare_integration_v2 import install_runtime_preparation
    need(args.parent=="argformer_american_train_000","Exactly the existing full AMtrain000 parent")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and 0<args.max_rss_bytes<=128*2**30
         and 0<args.max_seconds<=7200 and 0<args.max_output_bytes<=16*2**20,
         "CPU-only qualification requires explicit ROOT RAM/wall/output limits")
    out=Path(args.output_directory).resolve()
    need((out.is_relative_to(Path(__file__).resolve().parent)
          or out.is_relative_to(ROOT/"experiments/results")) and not out.exists(),
         "Fresh admitted qualification output required")
    # Carry only the three observed refs. TRAIN truth is never resolved or opened.
    refs=spec["observed_input_refs"]
    need(set(refs)=={"observation_ref","focal_registry_ref","native_preparation_ref"},
         "Only observed input bindings are permitted")
    out.mkdir()
    started=time.monotonic();last_emitted=[None,started];counts={};timings={};peak=[0]
    def rss():
        return int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    def write(name,value):
        encoded=(json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode()
        need(sum(p.stat().st_size for p in out.iterdir())+len(encoded)<=args.max_output_bytes,
             "Tiny qualification output cap")
        with (out/name).open("ab") as stream:stream.write(encoded);stream.flush()
    def check(stage):
        resource_check(stage)
        elapsed=time.monotonic()-started;current=rss();peak[0]=max(peak[0],current)
        need(elapsed<=args.max_seconds and current<=args.max_rss_bytes,"CPU qualification RAM/wall failstop")
        counts[stage]=counts.get(stage,0)+1
        now=time.monotonic()
        if stage!=last_emitted[0] or now-last_emitted[1]>=30:
            write("progress.jsonl",dict(stage=stage,elapsed_seconds=elapsed,rss_bytes=current))
            last_emitted[:]=[stage,now]
    def alarm(signum,frame):raise TimeoutError("CPU qualification wall limit")
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.max_seconds)
    result=dict(schema="one_full_AMtrain000_exact_compact_CPU_qualification_v2",
                parent_id=args.parent,input_job_ref=spec.get("input_job_ref"),
                observed_input_refs=refs,limits=vars(args),private_truth_opened=False,
                models_executed=False,CUDA_called=False,scientific_outputs_created=0,candidate_source_refs=spec.get("candidate_source_refs",[]),
                full_parent_preparation_count=0)
    try:
        import torch
        torch.set_num_threads(8);torch.set_num_interop_threads(1)
        from arggnn.american_c3_gnn_dataset_v1 import load_observation
        config=dict(native_preparation_ref=refs["native_preparation_ref"],
                    class_order=["AFR","EUR","NAT"],expected_haplotypes=1600,
                    reference_counts=[200,200,200],input_reference_class_mask=None,
                    input_reference_sample_indices=None)
        # Use the original frozen class order, never an invented label mapping.
        from arggnn.benchmarking.treeloom_fresh import CLASSES,FreshParentRegistry,FreshTreeLoom
        config["class_order"]=list(CLASSES)
        registry=FreshParentRegistry.from_ref(spec["parent_registry_ref"])
        # Namespace construction defines the original classes; no model is instantiated.
        runtime=install_runtime_preparation(FreshTreeLoom(registry))
        check("load_original_observation");at=time.monotonic()
        state=load_observation(refs["observation_ref"],refs["focal_registry_ref"],config)
        timings["load_observation_seconds"]=time.monotonic()-at
        need(state["manifest"]["parent_id"]==args.parent,"Same exact existing full parent")
        external={"bank.genotype":state["manifest"]["genotypes_ref"]}
        at=time.monotonic();result["full_parent_preparation_count"]+=1
        bundle,metadata=runtime.prepare_observation(state,split="train",resource_check=check,**refs,
                                                    **spec["preparation_limits"])
        timings["complete_prepare_seconds"]=time.monotonic()-at
        result.update(native_holds=bundle.native_holds,original_sites=bundle.original_sites,
                      packed_nodes=bundle.bank.parent.numel(),haplotypes=bundle.bank.sample_index.shape[1])
        at=time.monotonic();signature=exact_signature(bundle,metadata,check)
        timings["source_signature_seconds"]=time.monotonic()-at
        at=time.monotonic()
        record=pack_bundle(bundle,metadata,input_refs=refs,external_arrays=external,resource_check=check)
        need(record["charged_bytes"]<=spec["max_parent_compact_bytes"],"Complete compact parent must fit metadata-only upper bound")
        timings["lossless_pack_seconds"]=time.monotonic()-at
        result.update(compact_charged_bytes=record["charged_bytes"],
                      compact_payload_bytes=record["payload_bytes"],
                      expanded_tensor_bytes=record["expanded_tensor_bytes"],
                      restore_scratch_upper_bound_bytes=record["read_peak_upper_bound_bytes"],
                      disk_cache_bytes=0)
        del bundle,metadata,state;gc.collect()
        check("original_expanded_parent_released")
        for attempt in (1,2):
            need(rss()+record["read_peak_upper_bound_bytes"]<=args.max_rss_bytes,
                 "Full exact restore must fit admitted RSS with compact master retained")
            at=time.monotonic()
            restored,restored_metadata=restore_bundle(record,input_refs=refs,resource_check=check,verify_integrity=True)
            timings["exact_restore_%d_seconds"%attempt]=time.monotonic()-at
            at=time.monotonic()
            need(exact_signature(restored,restored_metadata,check)==signature,
                 "Every original bank/episode/cut/focal byte, dtype, shape and scalar must match")
            timings["restored_signature_%d_seconds"%attempt]=time.monotonic()-at
            del restored,restored_metadata;gc.collect()
            check("exact_restore_%d_verified_and_released"%attempt)
        result.update(status="passed_complete_parent_exact_preparation_and_two_reuses",
                      original_complete_parent_retained=True,
                      all_original_tensor_and_metadata_signatures_equal=True,
                      exact_restores=2,full_parent_preparation_count=1)
    except BaseException as error:
        result.update(status="failed_bounded_CPU_qualification",error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc()[-16384:])
    finally:
        signal.alarm(0)
        result.update(elapsed_seconds=time.monotonic()-started,observed_peak_rss_bytes=peak[0],
                      timings=timings,resource_check_counts=counts)
        write("qualification.json",result)
    print(json.dumps(dict(status=result["status"],output_directory=str(out),
                          elapsed_seconds=result["elapsed_seconds"],peak_rss_bytes=peak[0]),sort_keys=True))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-job",required=True)
    parser.add_argument("--input-job-sha256",required=True)
    parser.add_argument("--parent",required=True)
    parser.add_argument("--output-directory",required=True)
    parser.add_argument("--max-rss-bytes",type=int,required=True)
    parser.add_argument("--max-seconds",type=int,required=True)
    parser.add_argument("--max-output-bytes",type=int,required=True)
    args=parser.parse_args(argv)
    from .resident_pool_v2 import need
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only visibility required")
    job_path=ROOT/args.input_job
    raw=job_path.read_bytes()
    need(len(raw)<=16*2**20 and hashlib.sha256(raw).hexdigest()==args.input_job_sha256,
         "Exact bounded existing input-job metadata required")
    training=json.loads(raw)["call"]["spec"]["training_spec"]
    item=training["train_inputs"][args.parent]
    spec=dict(parent_id=args.parent,output_directory=args.output_directory,
              max_rss_bytes=args.max_rss_bytes,max_seconds=args.max_seconds,
              max_output_bytes=args.max_output_bytes,preparation_limits=training["preparation_limits"],
              parent_registry_ref=training["parent_registry_ref"],
              max_parent_compact_bytes=next(r["complete_compact_cache_upper_bound_bytes"]+160000
                  for r in json.loads((Path(__file__).parent/"storage_bound_v2.json").read_text())["parents"]
                  if r["parent_id"]==args.parent),
              observed_input_refs={k:item[k] for k in
                                  ("observation_ref","focal_registry_ref","native_preparation_ref")},
              input_job_ref=dict(path=args.input_job,sha256=args.input_job_sha256))
    from arggnn.benchmarking.treeloom_preparation_v4 import candidate_source_refs
    spec["candidate_source_refs"]=candidate_source_refs()
    result=qualify(spec,resource_check=lambda stage:None)
    return 0 if result["status"].startswith("passed_") else 1


if __name__=="__main__":
    try:raise SystemExit(main())
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        raise

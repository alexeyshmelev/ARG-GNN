"""CPU context factory for ROOT's existing admitted finite CPU worker.

The caller must authenticate admission/source/runtime/charged scopes before entry.
This module creates no controller, guard, authority, lease, storage entry or process.
"""
from pathlib import Path
import os,resource
from research.reviews.ordered_marker_learning_integration_source_20261017_v2.protocol_v2 import require,TRAIN,VAL,H050_INPUT_REF,H050_TRAIN_BINDINGS_REF,H050_VAL_BINDINGS_REF,OBSERVATION


def build_context_v1(spec,*,reader,output,guard,worker_tmp_directory):
    require(callable(guard)and callable(getattr(output,"save",None)),"Actual ROOT bounded guard/writer callbacks")
    guard() # Existing ROOT worker verifies real admission/source/runtime before imports.
    require(spec.get("root_authorizer")=="root_agent" and spec.get("admitted")is True
        and spec.get("hypothesis_id")=="ordered_marker_placement_v2" and spec.get("phase")in("train","evaluate")
        and spec.get("device")=="cpu" and spec.get("physical_gpu")is None
        and spec.get("max_cuda_reserved_bytes")==0 and os.environ.get("CUDA_VISIBLE_DEVICES")=="",
        "Already admitted current ROOT CPU-only spec; metadata is not independent authority")
    cfg=reader.decode_json(spec["refs"]["config"])
    original=reader.decode_json(H050_INPUT_REF)
    generation=reader.decode_json(original["refs"]["generation_config"])
    registration=reader.decode_json(original["refs"]["registration"])
    require(registration["config_ref"]==original["refs"]["generation_config"]
        and registration["parent_registry"]==generation["parent_registry"],"Exact original generation/registration")
    phase=spec["phase"]
    bindings=reader.decode_json(H050_TRAIN_BINDINGS_REF if phase=="train"else H050_VAL_BINDINGS_REF)["parents"]
    require(set(bindings)==set(TRAIN if phase=="train"else VAL),"Exact original phase target authority")
    counts=reader.decode_json(cfg["source_count_ref"])["parents"]
    import torch
    from arggnn import train_ordered_clade_episode_v1 as h45
    require(torch.get_num_threads()==32 and torch.get_num_interop_threads()==1
        and torch.get_default_dtype()==torch.float32 and not torch.cuda.is_initialized(),
        "ROOT configures exact original CPU32/interop1/FP32 before this factory")
    h45.science()
    def before_episode(pid,bank,declared):
        guard();actual=h45.episode_upper_bounds_v1(bank);row=counts[pid]
        require(declared==actual and actual["native_nodes"]==row["native_nodes"]
            and actual["native_holds"]==row["native_holds"] and actual["haplotypes"]==952,
            "Original complete-bank episode bounds")
        require(actual["compact_payload_upper_bound_bytes"]<=min(cfg["episode_payload_cap_bytes"],cfg["episode_cache_cap_bytes"]),
            "Whole original episode inside reviewed caps")
        estimate=actual["preparation_named_array_allowance_bytes"]+cfg["episode_python_scratch_allowance_bytes"]
        with open("/proc/self/statm")as stream:rss=int(stream.read().split()[1])*os.sysconf("SC_PAGE_SIZE")
        require(estimate<=cfg["episode_preparation_cap_bytes"]and rss+estimate<=cfg["max_cpu_rss_bytes"],
            "Prospective complete episode arrays plus scratch/RSS; no measured-fit claim")
    ctx=dict(root_authorizer="root_agent",admitted=True,hypothesis_id="ordered_marker_placement_v2",phase=phase,
        device=torch.device("cpu"),physical_gpu=None,cfg=cfg,config_ref=spec["refs"]["config"],
        reader=reader,output=output,guard=guard,before_episode_build=before_episode,
        registry=generation["parent_registry"],entry_refs=original["observation_entry_refs"],
        observation_source_refs=original["observation_cache_source_refs"],observation_cache_ref=OBSERVATION,
        bindings=bindings,worker_tmp_directory=str(worker_tmp_directory))
    if phase=="evaluate":ctx["train_completion_ref"]=spec["train_completion_ref"]
    guard()
    return ctx

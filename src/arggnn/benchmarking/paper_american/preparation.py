"""Readable full native corpus preparation with static unchanged author functions."""
import os
from arggnn.benchmarking.american_competitors import (need, read, ref, ROOT, Writer)
from . import protocol as p

def run_prepare(spec, *, resource_check):
    from arggnn.benchmarking.american_competitors import build_argformer_spec
    expected=build_argformer_spec(stage="prepare",inference_ref=spec["inference_ref"],
        focal_preparation_ref=spec["focal_preparation_ref"],output_directory=spec["output_directory"],
        limits=spec["limits"],path_limits=spec["path_limits"])
    need(spec==expected,"Exact bounded original native corpus scientific spec required")
    for reference in spec["requirements"]["source_refs"]:
        need(ref(reference["path"])==reference,"Exact historical native scientific dependency required")
    need(callable(resource_check),"ROOT worker resource check required")
    return _prepare(spec,p.path(spec["output_directory"]),resource_check)

def _prepare(spec,out,check):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Native path preparation is CPU-only")
    from . import paths_static as original
    cfg=dict(original.DEFAULTS,**spec["path_limits"],execution_enabled=True,observation_ref=spec["observation_ref"],
             path_sampling_seed=read(spec["inference_ref"])["seeds"]["sampling"],query_focal_count=10000,
             output_cap_bytes=spec["limits"]["max_output_bytes"]-4*2**20)
    need(original.storage_bound(cfg)<=cfg["output_cap_bytes"],"Whole original full-corpus production bound exceeds cap")
    out.mkdir(parents=True,exist_ok=False)
    scratch=out/"scratch"
    w=Writer(scratch,cfg["output_cap_bytes"],check,completion_name="original_preparation.json")
    completion_ref=original.execute(cfg,w)
    p=read(completion_ref)
    need(p["status"]=="completed_native_C3_path_preparation" and p["parent_id"]==spec["parent_id"]
         and p["observation_ref"]==spec["observation_ref"] and p["query_ancestry_truth_opened"] is False,
         "Genuine full native corpus completion required")
    import numpy as np
    from arggnn.american_c3_common_v1 import bound
    with np.load(bound(p["focals_ref"]),allow_pickle=False) as a, np.load(bound(spec["focal_registry_ref"]),allow_pickle=False) as b:
        need(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),
             "Full native focal registry must exactly equal shared compact original IDs and coordinates")
    datasets={}
    for key,r in p["full_path_id_order_refs"].items():
        check("native_full_corpus_row_count")
        with np.load(bound(r),allow_pickle=False) as z: rows=len(z["record_ids"])
        datasets[key]=dict(rows=rows,dataset_ref=p["native_dataset_refs"][key],row_order_ref=r)
    labels={s:ref(scratch/"contrastive"/s/"labels.pkl") for s in ("train","val")}
    result=dict(p,schema="american_fresh_ARGformer_preparation_v1",focals_ref=spec["focal_registry_ref"],
                native_generated_focals_ref=p["focals_ref"],native_focals_equal_shared_registry=True,
                datasets=datasets,reference_label_refs=labels,MLM_optimizer_updates=5*(datasets["mlm/train"]["rows"]//512),
                contrastive_optimizer_updates=datasets["contrastive/train"]["rows"]//4096,
                disposable_scratch_directory=str(scratch.relative_to(ROOT)),
                disposable_only_after_final_prediction_and_successful_outer_wait=True,
                source_refs=spec["requirements"]["source_refs"]+[ref(__file__),
                    ref("src/arggnn/benchmarking/paper_american/paths_static.py"),
                    ref("src/arggnn/benchmarking/paper_american/author_path_functions.py")],native_config=cfg)
    need(result["MLM_optimizer_updates"]>0 and result["contrastive_optimizer_updates"]>0,"No zero-update native fit")
    return Writer(out,spec["limits"]["max_output_bytes"],check,create=False).json("preparation_completed.json",result)


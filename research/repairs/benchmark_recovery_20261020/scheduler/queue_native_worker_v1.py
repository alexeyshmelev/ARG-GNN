"""Readable native numerical child for the separately approved shared GPU0 route.

No scheduler, admission, lease or subprocess launch. The outer worker owns the
process group. Source imports are inert until main/run is explicitly invoked.
"""
from __future__ import annotations
import argparse
from collections.abc import Mapping
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

from arggnn.benchmarking import american_competitors as api


def checkpoint_counters(state,epochs,updates):
    api.need(isinstance(state,Mapping) and isinstance(state.get("model"),Mapping) and state["model"],
             "Actual nonempty saved native model state required")
    ts=state.get("timestamp",{}).get("Timestamp",{})
    api.need(ts.get("epoch")==epochs and ts.get("batch")==updates,
             "Actual complete native Composer epoch/update counters required")
    optimizers=state.get("optimizers",{})
    api.need(isinstance(optimizers,Mapping) and set(optimizers)=={"DecoupledAdamW"},"Original native AdamW state required")
    states=optimizers["DecoupledAdamW"].get("state",{})
    api.need(isinstance(states,Mapping) and states,"Every actual optimizer parameter state required")
    for item in states.values():
        api.need(isinstance(item,Mapping) and {"step","exp_avg","exp_avg_sq"}<=set(item),"Actual optimizer moments/counter required")
        step=item["step"].item() if hasattr(item["step"],"item") else item["step"]
        api.need(step==updates,"Every saved native Adam counter must equal the full declared update budget")
    return dict(completed_epochs=epochs,actual_optimizer_updates=updates,
                optimizer_parameter_state_count=len(states),optimizer_unique_step_values=[updates])


def _native(spec,out):
    api.need(Path(sys.executable).resolve()==(api.ROOT/api.NATIVE_PYTHON).resolve()
             and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact isolated native interpreter -B -s required")
    api.need(os.environ.get("CUDA_VISIBLE_DEVICES")==api.GPU0_UUID and os.environ.get("ARGGNN_PHYSICAL_GPU")=="0",
             "Only admitted physical GPU0 may become visible")
    api.need(int(os.environ.get("WORLD_SIZE","1"))==1,"Single-device native world size must remain one")
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    for name in ("TMPDIR","TMP","TEMP","HF_HOME","XDG_CACHE_HOME","TORCH_HOME","TRITON_CACHE_DIR","CUDA_CACHE_PATH"):
        p=out/(".tmp" if name in ("TMPDIR","TMP","TEMP") else ".cache/"+name.lower())
        p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
    tempfile.tempdir=str(out/".tmp")
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1",WANDB_MODE="disabled",TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
                      CUDA_CACHE_MAXSIZE=str(256*2**20))
    ipc=importlib.import_module("research.reviews.American_C3_native_MLM_amendment_v12.abstract_unix_bootstrap_v1")
    ipc.install_abstract_unix_v1("fresh_american_native")
    sys.path[:0]=[str(api.ROOT/api.AUTHOR),str(api.ROOT)]
    os.chdir(out)
    from .queue_native_admission_v1 import validate_worker_lease
    validate_worker_lease()
    import torch
    validate_worker_lease(first_cuda=True)
    api.need(torch.cuda.device_count()==1,"Exactly one admitted visible CUDA device required")
    cap=int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES","0"))
    total=torch.cuda.get_device_properties(0).total_memory
    api.need(0<cap<=total,"Finite ROOT-admitted CUDA allocator cap required")
    torch.cuda.set_per_process_memory_fraction(cap/total,0)
    return torch


def _author(name):
    m=importlib.import_module(name)
    api.need(Path(m.__file__).resolve()==api.ROOT/api.AUTHOR/(name+".py"),"Pinned unchanged author module origin required")
    for key,mod in tuple(sys.modules.items()):
        origin=getattr(mod,"__file__",None)
        if origin and (key=="src" or key.startswith("src.")):
            api.need(Path(origin).resolve().is_relative_to(api.ROOT/api.AUTHOR),"Isolated author src namespace required")
    return m


def _configuration(prep,out,phase,*,checkpoint=None,data_view=None):
    from omegaconf import OmegaConf
    filename="mlm" if phase=="mlm" else "contrastive"
    cfg=OmegaConf.load(api.ROOT/api.AUTHOR/("yamls/"+filename+".yaml"))
    root=api.path(prep["native_dataset_refs"]["mlm/train"]["path"]).parents[2]
    cfg.data_local=str(root/("mlm" if phase=="mlm" else data_view or "contrastive"))
    cfg.run_name=prep["parent_id"]+"_native_"+phase
    cfg.save_folder=str(out/"checkpoints")
    cfg.load_path=None;cfg.autoresume=False;cfg.loggers={};cfg.device="gpu"
    cfg.save_num_checkpoints_to_keep=1;cfg.save_overwrite=False
    cfg.model.pretrained_model_name=str((api.ROOT/api.BASE_CONFIG["path"]).parent)
    if phase!="mlm": cfg.model.pretrained_checkpoint=str(api.path(checkpoint["path"])) if checkpoint else None
    # Only I/O, logging and finite retention differ. All numerical YAML fields stay intact.
    return cfg


def _verify_encoder_transfer(source,model,torch):
    mapped={k.removeprefix("model."):v for k,v in source.items()}
    target=model.state_dict()
    core={k for k in target if k.startswith("bert.")}
    api.need(core and core=={k for k in mapped if k.startswith("bert.")},"Every MLM encoder key must transfer to contrastive initialization")
    for k in core:
        a,b=mapped[k],target[k]
        api.need(tuple(a.shape)==tuple(b.shape) and a.dtype==b.dtype
                 and bool(torch.equal(a.detach().cpu(),b.detach().cpu())),"Exact MLM encoder tensor transfer required: "+k)
    return len(core)


def _train(spec,prep,out,torch,check):
    from omegaconf import OmegaConf
    phase=spec["stage"]
    checkpoint=spec.get("checkpoint_ref")
    cfg=_configuration(prep,out,phase,checkpoint=checkpoint)
    module=_author("main" if phase=="mlm" else "sequence_contrastive")
    trainer_box={}
    actual_trainer=module.Trainer
    def capture_trainer(*args,**kwargs):
        api.need(not trainer_box,"Exactly one original native Trainer is permitted")
        from composer.core import Callback
        from .native_telemetry_v1 import memory_callback
        kwargs["callbacks"]=[*kwargs.get("callbacks",[]),memory_callback(Callback,torch,out,check)]
        t=actual_trainer(*args,**kwargs);trainer_box["trainer"]=t
        api.need(len(t.state.train_dataloader)==spec["expected_updates"]//(5 if phase=="mlm" else 1),
                 "Actual complete native loader length must equal the admitted data-derived budget")
        return t
    bindings=dict(Trainer=capture_trainer)
    initialization={}
    if phase=="contrastive":
        actual_build=module.build_contrastive_model
        def checked_build(*args,**kwargs):
            model=actual_build(*args,**kwargs)
            saved=torch.load(api.path(checkpoint["path"]),map_location="cpu",weights_only=False,mmap=True)
            count=_verify_encoder_transfer(saved["state"]["model"],model.base.backbone,torch)
            api.need(model.base.temperature==.3 and model.base.class_weights is None,
                     "Preserve explicitly named release temperature and uniform fallback")
            initialization.update(complete_encoder_keys_equal=count,initializer_checkpoint_ref=checkpoint,
                                  verified_before_optimizer=True,no_forward_or_optimizer_probe=True)
            from .activation_checkpoint_v1 import install_native
            initialization["activation_checkpoint_execution"]=install_native(model,torch,api.ROOT/api.AUTHOR)
            del saved
            return model
        bindings["build_contrastive_model"]=checked_build
    OmegaConf.save(cfg,out/"resolved_io_config.yaml",resolve=True)
    previous_bindings={name:getattr(module,name) for name in bindings}
    check("before_full_original_native_training")
    try:
        for name,value in bindings.items():setattr(module,name,value)
        (module.main if phase=="mlm" else module.train)(cfg)
    finally:
        for name,value in previous_bindings.items():setattr(module,name,value)
    check("after_full_original_native_training")
    trainer=trainer_box["trainer"]
    epochs=5 if phase=="mlm" else 1
    api.need(int(trainer.state.timestamp.epoch.value)==epochs
             and int(trainer.state.timestamp.batch.value)==spec["expected_updates"],"Actual live full training counters required")
    final=out/"checkpoints/final-full-budget.pt"
    api.need(not final.exists(),"No final-checkpoint overwrite/replay")
    trainer.save_checkpoint(str(final),weights_only=False)
    saved=torch.load(final,map_location="cpu",weights_only=False,mmap=True)
    counters=checkpoint_counters(saved["state"],epochs,spec["expected_updates"])
    del saved
    return dict(checkpoint_ref=api.ref(final),initialization=initialization,**counters,
                settings=api.SETTINGS["MLM" if phase=="mlm" else "contrastive"],
                all_original_training_rows_used=True,additional_final_checkpoint=True)


def _embeddings(spec,prep,out,torch,check):
    from omegaconf import OmegaConf
    from arggnn.american_c3_common_v1 import bound
    outputs={}
    module=_author("embeddings")
    for key in api.VIEWS:
        view,split=key.split("/")
        cfg=_configuration(prep,out,"embeddings",data_view=view)
        config_path=out/(view+"_"+split+"_resolved_io_config.yaml")
        OmegaConf.save(cfg,config_path,resolve=True)
        model,loaded=module.load_checkpoint_model(str(config_path),str(api.path(spec["checkpoint_ref"]["path"])))
        dest=out/"scratch"/view/split
        dest.mkdir(parents=True,exist_ok=False)
        vectors=module.embed_arg_dataset(model,loaded,output_path=str(dest/"embeddings.pt"),batch_size=256,
            pooling_strategy="head_norm",layer_index=None,max_samples=0,is_classification=True,
            include_splits=split,use_split_labels=view=="contrastive")
        rows=prep["datasets"][key]["rows"]
        api.need(tuple(vectors.shape)==(rows,768) and vectors.dtype==torch.float32 and bool(torch.isfinite(vectors).all()),
                 "Every full native split embedding must be finite and complete")
        outputs[key]=dict(embedding_ref=api.ref(dest/"embeddings.pt"),rows=rows,
                          raw_dataset_ref=prep["native_dataset_refs"][key],row_order_ref=prep["full_path_id_order_refs"][key])
        del vectors,model,loaded
        torch.cuda.empty_cache()
        check("full_native_single_split_embedding_complete")
    return dict(checkpoint_ref=spec["checkpoint_ref"],outputs=outputs,pooling="head_norm",max_samples=0,
                disposable_scratch_directory=str((out/"scratch").relative_to(api.ROOT)),
                disposable_only_after_final_prediction_and_successful_outer_wait=True)


def _retrieval(spec,prep,out,torch,check):
    import numpy as np
    from arggnn import american_c3_native_concat_v1 as concat
    from arggnn import american_c3_evaluate_v1 as evaluator
    producer=api.read(spec["producer_ref"])
    values={}
    for view in ("query_paths","contrastive"):
        parts=[]
        for split in ("train","val"):
            key=view+"/"+split;item=producer["outputs"][key]
            api.need(item["raw_dataset_ref"]==prep["native_dataset_refs"][key]
                     and item["row_order_ref"]==prep["full_path_id_order_refs"][key],"Same native full raw-path row-order sidecars required")
            order=concat.row_order(item["row_order_ref"],parent_id=spec["parent_id"],reference=view=="contrastive")
            a=torch.load(api.path(item["embedding_ref"]["path"]),map_location="cpu",weights_only=True,mmap=True)
            n=prep["datasets"][key]["rows"];meta=a["metadata"]
            keys={"embeddings","input_ids","attention_masks","extant_node_ids","metadata"}
            if view=="contrastive":keys.add("labels")
            api.need(set(a)==keys and meta["splits"]==[split] and meta["num_samples"]==n
                     and meta["embedding_dim"]==768 and meta["pooling_strategy"]=="head_norm" and meta["layer_index"] is None
                     and tuple(a["embeddings"].shape)==(n,768) and a["embeddings"].dtype==torch.float32
                     and bool(torch.isfinite(a["embeddings"]).all())
                     and a["input_ids"].shape==a["attention_masks"].shape==(n,128)
                     and a["input_ids"].dtype==a["attention_masks"].dtype==torch.long
                     and a["extant_node_ids"].dtype==torch.long
                     and np.array_equal(a["extant_node_ids"].reshape(-1).numpy(),order["sample_nodes"]),
                     "Exact complete native tensors, metadata and original sample order required")
            if view=="contrastive":
                api.need(a["labels"].dtype==torch.long and np.array_equal(a["labels"].reshape(-1).numpy(),order["reference_classes_0based"]),
                         "Only observed source reference labels may enter retrieval")
            parts.append(a);check("complete_native_embedding_read")
        api.need({k:v for k,v in parts[0]["metadata"].items() if k not in ("splits","num_samples")}==
                 {k:v for k,v in parts[1]["metadata"].items() if k not in ("splits","num_samples")},"Compatible original train/val embeddings required")
        combined={k:torch.cat([a[k] for a in parts],dim=0) for k in parts[0] if k!="metadata"}
        combined["metadata"]=dict(parts[0]["metadata"],splits=["train","val"],num_samples=sum(a["metadata"]["num_samples"] for a in parts))
        values[view]=combined
        del parts,combined
    class RAM:
        def __init__(self):self.values=values
        def descriptor(self,view,kind):
            return dict(schema="american_fresh_native_RAM_embeddings_v1",storage_kind=kind,view=view,
                        parent_id=spec["parent_id"],rows=len(values[view]["embeddings"]),splits=["train","val"],
                        source_embedding_refs=[producer["outputs"][view+"/"+s]["embedding_ref"] for s in ("train","val")],
                        actual_combined_payload_file_exists=False)
    ram=RAM();tokens={"RAM_native_references":values["contrastive"],"RAM_native_queries":values["query_paths"]}
    def load_ram(name):
        api.need(name in tokens,"Only complete validated native RAM inputs may be retrieved")
        return tokens[name]
    module=_author("retrieve")
    original_loader=module.load_embeddings
    previous=sys.argv
    try:
        sys.argv=[str(api.ROOT/api.AUTHOR/"retrieve.py"),"--corpus","RAM_native_references","--queries","RAM_native_queries",
                  "--out",str(out/"retrieval.pt"),"--topk","1","--q_batch","1024","--e_chunk","100000"]
        module.load_embeddings=load_ram
        module.main()
    finally:
        module.load_embeddings=original_loader
        sys.argv=previous
    check("original_complete_native_retrieval_finished")
    retrieval_ref=api.ref(out/"retrieval.pt")
    registry=evaluator.registry(spec["focal_registry_ref"])
    api.need(len(registry["record_ids"])==10000,"Every and only common focal record required")
    cfg=dict(evaluator.DEFAULTS,mode="native_top1_seal",method="native_ARGformer_fresh_author_release_per_ARG",
             preparation_ref=spec["preparation_ref"],focal_registry_ref=spec["focal_registry_ref"],
             native_query_embeddings_ref=ram.descriptor("query_paths","RAM_tensors"),
             native_corpus_embeddings_ref=ram.descriptor("contrastive","RAM_tensors"),
             native_retrieval_ref=retrieval_ref,external_truth_ref=None,prediction_ref=None)
    writer=api.Writer(out/"seal",64*2**20,check,completion_name="prediction_seal.json")
    from .native_seal_v1 import native_top1_seal
    def load_embedding(record):
        for view in ("query_paths","contrastive"):
            if record==ram.descriptor(view,"RAM_tensors"):return ram.values[view]
        raise ValueError("Only the exact validated RAM embedding descriptors may load")
    seal_ref=native_top1_seal(cfg,writer,registry,load_embedding)
    return dict(checkpoint_ref=spec["checkpoint_ref"],retrieval_ref=retrieval_ref,prediction_seal_ref=seal_ref,
                records=10000,topk=1,all_reference_rows_used=True,native_scores_and_ties_unchanged=True,
                combined_embedding_files_written=False,scoring_performed=False,
                disposable_scratch_directories=[prep["disposable_scratch_directory"],producer["disposable_scratch_directory"]])


def run(spec):
    out=api.validate_spec(spec,fresh=False)
    api.need(spec["method"]=="argformer" and spec["stage"] in ("mlm","contrastive","embeddings","retrieval"),"One native numerical stage required")
    deadline=time.monotonic()+spec["limits"]["max_seconds"]
    def check(boundary):
        api.need(time.monotonic()<deadline and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=spec["limits"]["max_rss_bytes"],
                 "Native stage wall/RSS failstop reached")
    from .queue_native_admission_v1 import validate_worker_lease
    request=validate_worker_lease()
    limit=request["task"]["max_output_bytes"]
    resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
    check("native_child_before_import")
    prep=api._prep(spec["preparation_ref"],spec)
    # Authenticate every actual corpus/checkpoint/embedding payload before its use.
    records=[prep["vocab_ref"]]
    if spec["stage"] in ("mlm","contrastive","embeddings"):
        views=("mlm",) if spec["stage"]=="mlm" else ("contrastive",) if spec["stage"]=="contrastive" else ("contrastive","query_paths")
        for view in views:
            for split in ("train","val"):
                key=view+"/"+split
                records.extend((prep["native_dataset_refs"][key],prep["full_path_id_order_refs"][key]))
                local=api.path(prep["native_dataset_refs"][key]["path"]).parent
                api.need(not (local/("tokenized_"+split+"_sequences_and_vocab.pkl")).exists(),"No silent tokenized-file selection")
        if "contrastive" in views:records.extend(prep["reference_label_refs"].values())
    if "checkpoint_ref" in spec:records.append(spec["checkpoint_ref"])
    if spec["stage"]=="retrieval":records.extend(x["embedding_ref"] for x in api.read(spec["producer_ref"])["outputs"].values())
    for record in records:
        check("authenticate_native_input")
        api.need(api.ref(record["path"])==record,"Exact actual same-parent native payload bytes required")
    torch=_native(spec,out)
    if spec["stage"] in ("mlm","contrastive"):result=_train(spec,prep,out,torch,check)
    elif spec["stage"]=="embeddings":result=_embeddings(spec,prep,out,torch,check)
    else:result=_retrieval(spec,prep,out,torch,check)
    result.update(stage=spec["stage"],parent_id=spec["parent_id"],preparation_ref=spec["preparation_ref"],
                  focal_registry_ref=spec["focal_registry_ref"],observation_ref=spec["observation_ref"],
                  query_ancestry_truth_opened=False,exact_paper_reproduction=False,
                  variant=api.SETTINGS["variant"],author_source_modified=False,outer_owned_group_cleanup_pending=True)
    check("native_child_before_completion")
    return api.Writer(out,spec["limits"]["max_output_bytes"],check,create=False).json("scientific_completed.json",result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec-ref",required=True);a=p.parse_args()
    result=run(api.read(json.loads(a.spec_ref)))
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":main()

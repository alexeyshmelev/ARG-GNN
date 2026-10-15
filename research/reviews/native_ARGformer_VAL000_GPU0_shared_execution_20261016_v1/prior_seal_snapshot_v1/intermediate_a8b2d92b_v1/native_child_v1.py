"""Native postMLM child: exact author contrastive science and RAM I/O only."""
from __future__ import annotations
import argparse,hashlib,importlib,importlib.util,io,json,os,resource,stat,sys,traceback
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL='research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1'
sys.path[:0]=[str(ROOT/REL),str(ROOT/"research/reviews/C3_native_post_MLM_root_handoff_20261015_v1"),
 str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),
 str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),
 str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12")]
SOURCE=REL+"/native_child_v1.py"
GUARD='research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py'
GUARD_SHA='3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69'
def _early_read(record):
    p=ROOT/record["path"]
    if p.resolve()!=p or not p.is_relative_to(ROOT) or p.stat().st_size>1048576:
        raise ValueError("Canonical bounded root metadata")
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record["sha256"]:raise ValueError("Exact root input SHA required")
    return json.loads(raw)

def _title_lease():
    path=ROOT/GUARD
    if hashlib.sha256(path.read_bytes()).hexdigest()!=GUARD_SHA:raise ValueError("Pinned new GPU0 lease/title guard changed")
    loaded=importlib.util.spec_from_file_location("_C3_existing_native_title_lease",path)
    guard=importlib.util.module_from_spec(loaded);loaded.loader.exec_module(guard)
    guard.install_before_torch_v1()
    lease=guard.validate_lease();guard.apply_and_validate_title()
    if lease["physical_gpu"] not in (0,):raise ValueError("Only current policy-bound physical GPU0")
    return guard,lease

def _save_json(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if len(raw)>1048576:raise ValueError("Bounded native child completion")
    with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())

def _int_time(value):
    value=getattr(value,"value",value)
    if type(value) is not int:raise ValueError("Actual integer Composer timestamp")
    return value

def _verify_author_import_origins(author_root):
    for name,module in list(sys.modules.items()):
        if name=="src" or name.startswith("src.") or name in ("main","sequence_contrastive","embeddings","retrieve"):
            origin=getattr(module,"__file__",None)
            if origin is None:continue
            path=Path(origin).resolve()
            if not path.is_relative_to(author_root):raise ValueError("Native author import shadowed: "+name)

def _io_config(mode,*,data_root,checkpoint,directory,base_config,OmegaConf,author_root):
    name="mlm" if mode=="mlm" else "contrastive"
    cfg=OmegaConf.load(author_root/"yamls"/(name+".yaml"))
    cfg.data_local=str(data_root);cfg.run_name="C3_native_"+mode
    cfg.save_folder=str(directory/"checkpoints");cfg.load_path=None;cfg.autoresume=False
    cfg.loggers={};cfg.device="gpu";cfg.model.pretrained_model_name=str(base_config.parent)
    if name=="contrastive":cfg.model.pretrained_checkpoint=str(checkpoint) if checkpoint is not None else None
    # No cfg.model.train_loader injection: preserve current uniform fallback.
    if "train_loader" in cfg.model:raise ValueError("Primary native config-plumbing correction forbidden")
    return cfg
def optimizer_counter_evidence(optimizers):
    if not isinstance(optimizers,dict) or set(optimizers)!={"DecoupledAdamW"}:
        raise ValueError("Actual author DecoupledAdamW serialized state required")
    values=[];count=0
    for name,optimizer in optimizers.items():
        state=optimizer.get("state")
        if not isinstance(state,dict) or not state:raise ValueError("Nonempty genuine optimizer parameter state")
        for parameter,value in state.items():
            if not isinstance(value,dict) or "step" not in value or "exp_avg" not in value or "exp_avg_sq" not in value:
                raise ValueError("Actual Adam step and moment state required for every retained parameter")
            step=value["step"]
            if hasattr(step,"item"):step=step.item()
            if type(step) not in (int,float) or step!=int(step) or step<=0:
                raise ValueError("Finite positive genuine integer optimizer step")
            values.append(int(step));count+=1
    return dict(optimizer_names=sorted(optimizers),parameter_state_count=count,
        unique_step_values=sorted(set(values)),minimum_step=min(values),maximum_step=max(values),
        source="actual_author_optimizer_state_dict")

def _checkpoint_entries(directory):
    paths=[];alias=None
    for path in directory.iterdir():
        s=path.lstat()  # Never stat/open through the Composer alias.
        if path.name=="latest-rank0.pt" and stat.S_ISLNK(s.st_mode):
            target_text=os.readlink(path);rel=Path(target_text)
            if rel.is_absolute() or rel.parent!=Path(".") or rel.name in ("",".",".."):
                raise ValueError("Only relative in-directory own Composer latest alias")
            target=directory/rel;ts=target.lstat()
            if not stat.S_ISREG(ts.st_mode) or ts.st_nlink!=1:
                raise ValueError("Composer alias target must already be an own regular checkpoint")
            alias=dict(alias_name=path.name,target_text=target_text,
                alias_lstat=[s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns],
                target_lstat=[ts.st_dev,ts.st_ino,ts.st_mode,ts.st_nlink,ts.st_size,ts.st_mtime_ns,ts.st_ctime_ns],
                status="observed_validated_alias_pending_postdrain_normalization",
                alias_followed_or_opened=False,normalized_future_bytes_claimed=False)
        elif stat.S_ISREG(s.st_mode) and s.st_nlink==1 and path.suffix==".pt" and not path.name.startswith("latest"):
            paths.append(path)
        elif stat.S_ISLNK(s.st_mode) or stat.S_ISREG(s.st_mode) and s.st_nlink!=1:
            raise ValueError("Unexpected checkpoint symlink/hardlink")
    return sorted(paths),alias

def _final_checkpoint(directory,*,epochs,updates,max_files,max_bytes,torch,reference):
    paths,alias=_checkpoint_entries(directory)
    if not paths or len(paths)>max_files:raise ValueError("Bounded actual native checkpoint files, never latest-alias selection")
    candidates=[]
    for path in paths:
        if path.lstat().st_size>max_bytes:raise ValueError("Actual native checkpoint exceeds admitted file bound")
        # Only current child-generated, source-bound Composer files are read.
        value=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        state=value.get("state",{});timestamp=state.get("timestamp",{})
        epoch=_int_time(timestamp.get("epoch"));batch=_int_time(timestamp.get("batch"))
        if epoch==epochs and batch==updates:
            if not isinstance(state.get("model"),dict) or not state["model"] or not state.get("optimizers"):
                raise ValueError("Final Composer model and optimizer state required")
            counters=optimizer_counter_evidence(state["optimizers"])
            if counters["unique_step_values"]!=[updates]:raise ValueError("Actual checkpoint Adam step counters differ from full budget")
            candidates.append(dict(checkpoint_ref=reference(path),completed_epochs=epoch,
                actual_optimizer_updates=counters["unique_step_values"][0],checkpoint_optimizer_counters=counters,Composer_timestamp={k:_int_time(timestamp[k])
                    for k in ("epoch","batch","batch_in_epoch") if k in timestamp},
                model_state_keys=len(state["model"]),optimizer_state_present=True))
        del value
    if not candidates:raise ValueError("No actual final completed checkpoint at exact release epoch/update budget")
    # Multiple regular aliases are accepted only if their entire bytes agree.
    if len({r["checkpoint_ref"]["sha256"] for r in candidates})!=1:
        raise ValueError("Ambiguous distinct final Composer checkpoints")
    selected=candidates[0]
    if alias is not None:
        if Path(selected["checkpoint_ref"]["path"]).name!=alias["target_text"]:
            raise ValueError("Actual latest alias must name the verified final regular checkpoint")
    selected["checkpoint_latest_alias"]=alias
    return selected

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--child-input",required=True)
    parser.add_argument("--expected-child-input-sha256",required=True);args=parser.parse_args(argv)
    packet_ref=dict(path=args.child_input,sha256=args.expected_child_input_sha256);packet=_early_read(packet_ref)
    root_ref=packet["active_input_ref"];spec=_early_read(root_ref)
    # No project or model/runtime import occurs before existing title and lease.
    guard,lease=_title_lease()
    if lease["lease_metadata_sha256"]!=packet["lease_metadata_sha256"]:raise ValueError("Inherited original lease changed")
    sys.path.insert(0,str(ROOT/"src"))
    from arggnn.american_c3_common_v1 import require,local,ref,bound,CLASSES
    from consumer_contract_v1 import (
        SOURCE as CONTRACT_SOURCE,MAIN,AUTHOR_ROOT,AUTHOR_MANIFEST_REF,metadata,
        validate_active_input,load_root_context,preparation_v1,counts_v1)
    cfg,_=validate_active_input(root_ref,spec,MAIN,GPU=True)
    require(packet["schema"]=="american_c3_native_child_input_v1"
        and packet["stage"] in ("contrastive","RAM")
        and spec["source_sha256"][SOURCE]==ref(SOURCE)["sha256"]
        and packet["stage"] in {"fit":("contrastive",),"embeddings":("RAM",)}[spec["phase"]],
        "Exact canonical native child/source/phase")
    directory=local(packet["directory"]);root_output=local(spec["output_directory"])
    directory.relative_to(root_output)
    require(directory==root_output/packet["name"]
        and local(packet_ref["path"])==root_output/(packet["name"]+"_child_input.json")
        and local(packet["short_tmp_directory"])==local(cfg["short_tmp_directory"])/packet["name"],
        "Exact canonical parent-owned stage/input/temp paths")
    require(directory.is_dir() and not (directory/"completed.json").exists(),"Owned fresh child stage directory")
    resource.setrlimit(resource.RLIMIT_FSIZE,(cfg["max_checkpoint_bytes"],cfg["max_checkpoint_bytes"]))
    ctx=load_root_context(root_ref,spec,native_child=True)
    ctx["fast_guard"]("before_native_runtime")
    runtime=metadata(spec["refs"]["runtime"]);python=bound(runtime["native_python_ref"])
    prefix=local(runtime["native_prefix"])
    require(Path(sys.prefix).resolve()==prefix and Path(sys.executable).resolve()==python
        and python.parent==prefix/"bin" and sys.dont_write_bytecode and sys.flags.no_user_site
        and os.environ.get("PYTHONPATH","")=="" and os.environ.get("PYTHONNOUSERSITE")=="1",
        "Actual isolated SHA-bound native interpreter, no user-site/PYTHONPATH/bytecode writes")
    require(callable(ctx.get("validate_native_runtime")) and callable(ctx.get("configure_native_device")),
        "Root must inject existing complete runtime inventory and device/resource guards")
    runtime_receipt=ctx["validate_native_runtime"](spec["refs"]["runtime"],runtime)
    require(runtime_receipt.get("status")=="passed" and runtime_receipt.get("runtime_ref")==spec["refs"]["runtime"],
        "Complete actual native runtime inventory must match before model import")
    prep,_,prepared_dir=preparation_v1(spec);counts=counts_v1(spec,prep)
    base=bound(runtime["bert_base_config_ref"]);base_value=metadata(runtime["bert_base_config_ref"])
    require(base.name=="config.json" and runtime["bert_base_config_release_name"]=="bert-base-uncased"
        and base_value["hidden_size"]==768 and base_value["intermediate_size"]==3072
        and base_value["num_hidden_layers"]==12 and base_value["num_attention_heads"]==12,
        "Qualified local copy of release BERT base configuration, preserving numerical defaults")
    from abstract_unix_bootstrap_v1 import bootstrap_native_execution_v1
    bootstrap=bootstrap_native_execution_v1("c3valpostpred",physical_gpu=spec["physical_gpu"],gpu_uuid=spec["execution_gpu_uuid"])
    _save_json(directory/"operational_IPC_bootstrap.json",bootstrap)
    author_root=local(AUTHOR_ROOT);sys.path.insert(0,str(author_root))
    import torch
    from omegaconf import OmegaConf
    ctx["configure_native_device"](runtime,lease)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,"One actual admitted visible CUDA UUID; no CPU fallback")
    from composer.utils import dist
    require(dist.get_world_size()==1,"Exact release one-GPU full-batch condition")
    guard.apply_and_validate_title();require(guard.validate_lease()==lease,"Actual title/lease before author model import")
    os.chdir(directory);stage=packet["stage"];ctx["fast_guard"]("before_native_author_call")
    result=dict(schema="american_c3_native_child_completion_v1",status="completed_C3_native_child",
        stage=stage,parent_id=spec["parent_id"],active_input_ref=root_ref,child_input_ref=packet_ref,
        preparation_ref=spec["refs"]["preparation"],counts_ref=spec["refs"]["counts"],
        author_source_manifest_ref=AUTHOR_MANIFEST_REF,runtime_ref=spec["refs"]["runtime"],
        runtime_validation=runtime_receipt,physical_gpu=lease["physical_gpu"],
        query_ancestry_truth_opened=False,author_code_modified=False)
    if stage=="contrastive":
        checkpoint=bound(packet["initializer_checkpoint_ref"])
        config=_io_config(stage,data_root=prepared_dir/"contrastive",checkpoint=checkpoint,
            directory=directory,base_config=base,OmegaConf=OmegaConf,author_root=author_root)
        config.save_num_checkpoints_to_keep=cfg["checkpoint_retention_IO_override"]
        resolved=OmegaConf.to_container(config,resolve=True)
        require((resolved["max_duration"],resolved["global_train_batch_size"],resolved["device_train_microbatch_size"],
            resolved["global_eval_batch_size"],resolved["device_eval_microbatch_size"],resolved["precision"],resolved["seed"])==
            ("1ep",4096,4096,4096,4096,"fp32",36)
            and resolved["model"]["contrastive"]["temperature"]==.30
            and resolved["model"]["contrastive"]["use_class_weights"] is True
            and "train_loader" not in resolved["model"]
            and resolved["save_interval"]=="20ba" and resolved["save_num_checkpoints_to_keep"]==1
            and resolved.get("train_subset_num_batches",-1)==-1,
            "Exact native supervised in-batch science with separately disclosed retention1")
        OmegaConf.save(config,directory/"resolved_io_config.yaml",resolve=True)
        module=importlib.import_module("sequence_contrastive");_verify_author_import_origins(author_root)
        require(Path(module.__file__).resolve()==author_root/"sequence_contrastive.py","Exact pinned author train API")
        module.train(config)
        final=_final_checkpoint(directory/"checkpoints",epochs=1,updates=counts["contrastive_optimizer_updates"],
            max_files=cfg["max_checkpoint_files"],max_bytes=cfg["max_checkpoint_bytes"],torch=torch,reference=ref)
        result.update(final,resolved_config_ref=ref(directory/"resolved_io_config.yaml"),
            initializer_checkpoint_ref=packet["initializer_checkpoint_ref"],
            uniform_class_weight_fallback_preserved=True,training_config_plumbing_correction=False,
            release_checkpoint_retention=-1,actual_checkpoint_retention=1,save_interval_unchanged=True,
            live_optimizer_counters_independently_observed=False,saved_optimizer_counters_verified=True)
    else:
        from arggnn.american_c3_native_concat_v1 import row_order
        from arggnn.american_c3_evaluate_v1 import registry,DEFAULTS as EVAL_DEFAULTS
        import ram_downstream_v1 as ram
        import embeddings as emb
        import retrieve
        _verify_author_import_origins(author_root)
        checkpoint=bound(packet["checkpoint_ref"])
        orders={key:row_order(record,parent_id=spec["parent_id"],reference=key.startswith("contrastive/"))
            for key,record in prep["full_path_id_order_refs"].items() if key.startswith(("query_paths/","contrastive/"))}
        frozen=registry(prep["focals_ref"]);require(len(frozen["record_ids"])==10000,"Exactly10000 frozen common focal records")
        combined={}
        for view in ("query_paths","contrastive"):
            parts={}
            for split in ("train","val"):
                config=_io_config("embedding",data_root=prepared_dir/view,checkpoint=None,
                    directory=directory,base_config=base,OmegaConf=OmegaConf,author_root=author_root)
                config_path=directory/(view+"_"+split+"_resolved_io_config.yaml");OmegaConf.save(config,config_path,resolve=True)
                model,loaded_cfg=emb.load_checkpoint_model(str(config_path),str(checkpoint))
                parts[split]=ram.embed_single_split_ram_v1(torch=torch,embeddings_module=emb,model=model,cfg=loaded_cfg,
                    view=view,split=split,expected_rows=counts["datasets"][view+"/"+split]["rows"],fast_guard=ctx["fast_guard"])
                del model
            combined[view],_=ram.concatenate_embeddings_ram_v1(torch=torch,view=view,parts=parts,
                orders={sp:orders[view+"/"+sp] for sp in ("train","val")},counts=counts,
                parent_id=spec["parent_id"],fast_guard=ctx["fast_guard"])
            del parts
        native=ram.retrieve_ram_v1(retrieve_module=retrieve,query=combined["query_paths"],corpus=combined["contrastive"],
            device=torch.device("cuda",0),fast_guard=ctx["fast_guard"])
        manifest=ram.tensor_manifest_v1(torch=torch,values=dict(query_paths=combined["query_paths"],
            contrastive=combined["contrastive"],retrieval=native),fast_guard=ctx["fast_guard"])
        manifest.update(checkpoint_ref=packet["checkpoint_ref"],counts_ref=spec["refs"]["counts"],
            preparation_ref=spec["refs"]["preparation"],full_path_id_order_refs=prep["full_path_id_order_refs"])
        manifest_path=directory/"RAM_tensor_manifest.json";_save_json(manifest_path,manifest)
        seal_cfg=dict(EVAL_DEFAULTS,mode="native_top1_seal",preparation_ref=spec["refs"]["preparation"],
            focal_registry_ref=prep["focals_ref"],native_query_embeddings_ref=None,native_corpus_embeddings_ref=None,
            native_retrieval_ref=None,external_truth_ref=None,prediction_ref=None,native_RAM_tensor_manifest_ref=ref(manifest_path))
        sealed_ref=ram.native_top1_seal_ram_v1(seal_cfg,_RAMWriter(directory),frozen,prep=prep,orders=orders,
            query_arrays=combined["query_paths"],corpus_arrays=combined["contrastive"],native=native,torch=torch)
        sealed=metadata(sealed_ref)
        result.update(checkpoint_ref=packet["checkpoint_ref"],seal_completion_ref=sealed_ref,
            prediction_ref=sealed["prediction_ref"],native_top1_preserved_ref=sealed["native_top1_preserved_ref"],
            RAM_tensor_manifest_ref=ref(manifest_path),embedding_file_artifacts_written=False,
            native_normalization_topk_and_C3_postprocessing_unchanged=True)
        del combined,native
    ctx["verify_runtime_imports"]();_verify_author_import_origins(author_root)
    ctx["fast_guard"]("after_native_author_call")
    _save_json(directory/"completed.json",result)
    return 0

class _RAMWriter:
    """Only compact final native prediction/score products, never embeddings."""
    def __init__(self,directory):self.directory=directory
    def npz(self,name,**arrays):
        import numpy as np
        if Path(name).name!=name:raise ValueError("Flat child product")
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);raw=buffer.getvalue()
        if len(raw)>16*1048576:raise ValueError("Compact sealed native product cap")
        path=self.directory/name
        with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        from arggnn.american_c3_common_v1 import ref
        return ref(path)
    def json(self,name,value):
        if name!="completed.json":raise ValueError("Only original native seal completion")
        path=self.directory/"sealed_predictions_completed.json";_save_json(path,value)
        from arggnn.american_c3_common_v1 import ref
        return ref(path)
if __name__=="__main__":raise SystemExit(main())

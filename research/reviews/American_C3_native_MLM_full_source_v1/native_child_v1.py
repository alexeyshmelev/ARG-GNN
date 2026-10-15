"""Isolated native-interpreter child. Standard library only before title/lease.

Author functions/configuration are unchanged. Only per-parent I/O bindings,
offline base-config location, fresh-run state, device and disabled external
logging are adapted. Parent owns direct wait/group drain and delayed log writes.
"""
from __future__ import annotations
import argparse,hashlib,importlib,importlib.util,json,os,sys,traceback
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/American_C3_native_MLM_full_source_v1/native_child_v1.py"
GUARD="src/arggnn/native_argformer_exec.py"
GUARD_SHA="84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0"
def _early_read(record):
    p=ROOT/record["path"]
    if p.resolve()!=p or not p.is_relative_to(ROOT) or p.stat().st_size>1048576:
        raise ValueError("Canonical bounded root metadata")
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record["sha256"]:raise ValueError("Exact root input SHA required")
    return json.loads(raw)

def _title_lease():
    path=ROOT/GUARD
    if hashlib.sha256(path.read_bytes()).hexdigest()!=GUARD_SHA:raise ValueError("Pinned existing lease/title guard changed")
    loaded=importlib.util.spec_from_file_location("_C3_existing_native_title_lease",path)
    guard=importlib.util.module_from_spec(loaded);loaded.loader.exec_module(guard)
    lease=guard.validate_lease();guard.apply_and_validate_title()
    if lease["physical_gpu"] not in (4,5):raise ValueError("Only currently authorized physical GPUs4/5")
    return guard,lease

def _save_json(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if len(raw)>1048576:raise ValueError("Bounded native child completion")
    with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())

def _int_time(value):
    value=getattr(value,"value",value)
    if type(value) is not int:raise ValueError("Actual integer Composer timestamp")
    return value

def _final_checkpoint(directory,*,epochs,updates,max_files,max_bytes,torch,reference):
    paths=sorted(p for p in directory.iterdir() if p.is_file() and not p.is_symlink()
        and p.suffix==".pt" and not p.name.startswith("latest"))
    if not paths or len(paths)>max_files:raise ValueError("Bounded actual native checkpoint files, never latest-alias selection")
    candidates=[]
    for path in paths:
        if path.stat().st_size>max_bytes:raise ValueError("Actual native checkpoint exceeds admitted file bound")
        # Only current child-generated, source-bound Composer files are read.
        value=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        state=value.get("state",{});timestamp=state.get("timestamp",{})
        epoch=_int_time(timestamp.get("epoch"));batch=_int_time(timestamp.get("batch"))
        if epoch==epochs and batch==updates:
            if not isinstance(state.get("model"),dict) or not state["model"] or not state.get("optimizers"):
                raise ValueError("Final Composer model and optimizer state required")
            candidates.append(dict(checkpoint_ref=reference(path),completed_epochs=epoch,
                actual_optimizer_updates=batch,Composer_timestamp={k:_int_time(timestamp[k])
                    for k in ("epoch","batch","batch_in_epoch") if k in timestamp},
                model_state_keys=len(state["model"]),optimizer_state_present=True))
        del value
    if not candidates:raise ValueError("No actual final completed checkpoint at exact release epoch/update budget")
    # Multiple regular aliases are accepted only if their entire bytes agree.
    if len({r["checkpoint_ref"]["sha256"] for r in candidates})!=1:
        raise ValueError("Ambiguous distinct final Composer checkpoints")
    return candidates[0]

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

def main(argv=None):
    import resource
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--child-input",required=True)
    parser.add_argument("--expected-child-input-sha256",required=True);args=parser.parse_args(argv)
    packet_ref=dict(path=args.child_input,sha256=args.expected_child_input_sha256);packet=_early_read(packet_ref)
    root_ref=packet["active_input_ref"];spec=_early_read(root_ref);guard,lease=_title_lease()
    if lease["lease_metadata_sha256"]!=packet["lease_metadata_sha256"]:raise ValueError("Inherited original lease changed")
    sys.path.insert(0,str(ROOT/"src"))
    from arggnn.american_c3_common_v1 import require,local,ref,bound,CLASSES
    from consumer_contract_v1 import (MAIN,AUTHOR_ROOT,AUTHOR_MANIFEST_REF,metadata,validate_active_input,
        load_root_context,preparation_v1,counts_v1)
    cfg,_=validate_active_input(root_ref,spec,MAIN,GPU=True)
    require(packet["schema"]=="american_c3_native_child_input_v1" and packet["stage"]==spec["phase"]
        and packet["stage"] in ("qualify_mlm","mlm"),"One exact MLM-only child stage")
    directory=local(packet["directory"]);root_output=local(spec["output_directory"])
    require(directory==root_output/packet["name"] and local(packet_ref["path"])==root_output/(packet["name"]+"_child_input.json")
        and local(packet["short_tmp_directory"])==local(cfg["short_tmp_directory"])/packet["name"],
        "Exact own parent-supplied child/input/temp paths")
    require(directory.is_dir() and not (directory/"completed.json").exists(),"Fresh child directory")
    # Per-file limit is an additional fail-stop, never an aggregate cache quota.
    resource.setrlimit(resource.RLIMIT_FSIZE,(cfg["max_checkpoint_bytes"],cfg["max_checkpoint_bytes"]))
    ctx=load_root_context(root_ref,spec,native_child=True);ctx["fast_guard"]("before_native_runtime")
    runtime=metadata(spec["refs"]["runtime"]);python=bound(runtime["native_python_ref"]);prefix=local(runtime["native_prefix"])
    require(Path(sys.prefix).resolve()==prefix and Path(sys.executable).resolve()==python and python.parent==prefix/"bin"
        and sys.dont_write_bytecode and sys.flags.no_user_site and os.environ.get("PYTHONPATH","")==""
        and os.environ.get("PYTHONNOUSERSITE")=="1","Isolated exact native executable, no user site/bytecode")
    runtime_receipt=ctx["validate_native_runtime"](spec["refs"]["runtime"],runtime)
    require(runtime_receipt.get("status")=="passed","Complete native inventory before imports")
    prep,_,prepared_dir=preparation_v1(spec);counts=counts_v1(spec,prep)
    base=bound(runtime["bert_base_config_ref"]);base_value=metadata(runtime["bert_base_config_ref"])
    require(base.name=="config.json" and runtime["bert_base_config_release_name"]=="bert-base-uncased"
        and base_value["hidden_size"]==768 and base_value["intermediate_size"]==3072
        and base_value["num_hidden_layers"]==12 and base_value["num_attention_heads"]==12,"Exact local base configuration")
    author_root=local(AUTHOR_ROOT);sys.path.insert(0,str(author_root))
    import torch
    from omegaconf import OmegaConf
    ctx["configure_native_device"](runtime,lease)
    from composer.utils import dist
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and dist.get_world_size()==1,
        "One actual held GPU, no CPU/distributed fallback")
    guard.apply_and_validate_title();require(guard.validate_lease()==lease,"Title/lease before author import")
    os.chdir(directory)
    config=_io_config("mlm",data_root=prepared_dir/"mlm",checkpoint=None,directory=directory,base_config=base,
        OmegaConf=OmegaConf,author_root=author_root)
    resolved=OmegaConf.to_container(config,resolve=True)
    require((resolved["max_duration"],resolved["global_train_batch_size"],resolved["device_train_microbatch_size"],
        resolved["precision"],resolved["seed"])==("5ep",512,512,"amp_bf16",17)
        and resolved.get("train_subset_num_batches",-1)==-1
        and resolved["model"]["model_config"]["num_hidden_layers"]==12
        and resolved["model"]["model_config"]["num_attention_heads"]==12
        and resolved["max_seq_len"]==128,"Unchanged full author numeric budget")
    module=importlib.import_module("main")
    require(Path(module.__file__).resolve()==author_root/"main.py","Exact author main origin")
    _verify_author_import_origins(author_root)
    result=dict(schema="american_c3_native_mlm_child_completion_v1",status="completed_C3_native_MLM_child",
        stage=packet["stage"],parent_id=spec["parent_id"],active_input_ref=root_ref,child_input_ref=packet_ref,
        preparation_ref=spec["refs"]["preparation"],author_source_manifest_ref=AUTHOR_MANIFEST_REF,
        runtime_ref=spec["refs"]["runtime"],runtime_validation=runtime_receipt,physical_gpu=lease["physical_gpu"],
        query_ancestry_truth_opened=False,author_code_modified=False,initializer_checkpoint_ref=None)
    if packet["stage"]=="qualify_mlm":
        from qualifier_v1 import qualify
        result["qualification"]=qualify(module,config,directory,torch,OmegaConf,ctx,runtime)
    else:
        OmegaConf.save(config,directory/"resolved_io_config.yaml",resolve=True)
        trainer=module.main(config,return_trainer=True)
        require(_int_time(trainer.state.timestamp.epoch)==5 and _int_time(trainer.state.timestamp.batch)==224775,
            "Actual full epoch5/update224775 Trainer timestamp")
        if hasattr(trainer,"close"):trainer.close()
        final=_final_checkpoint(directory/"checkpoints",epochs=5,updates=224775,
            max_files=cfg["max_checkpoint_files"],max_bytes=cfg["max_checkpoint_bytes"],torch=torch,reference=ref)
        result.update(final)
    OmegaConf.save(config,directory/"resolved_io_config.yaml",resolve=True) if packet["stage"]=="qualify_mlm" else None
    result["resolved_config_ref"]=ref(directory/"resolved_io_config.yaml")
    ctx["verify_runtime_imports"]();_verify_author_import_origins(author_root);ctx["fast_guard"]("after_native_author_call")
    _save_json(directory/"completed.json",result);return 0
if __name__=="__main__":raise SystemExit(main())

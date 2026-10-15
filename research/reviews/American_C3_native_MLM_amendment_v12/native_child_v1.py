"""Isolated native-interpreter child. Standard library only before title/lease.

Author functions/configuration are unchanged. Only per-parent I/O bindings,
offline base-config location, fresh-run state, device and disabled external
logging are adapted. Parent owns direct wait/group drain and delayed log writes.
"""
from __future__ import annotations
import argparse,hashlib,importlib,importlib.util,json,os,stat,sys,traceback
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[1:1]=[str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5")]
import amendment_v12 as amendment
SOURCE="research/reviews/American_C3_native_MLM_amendment_v12/native_child_v1.py"
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
        and packet["stage"]=="mlm","One exact MLM-only child stage")
    directory=local(packet["directory"]);root_output=local(spec["output_directory"])
    require(directory==root_output/packet["name"] and local(packet_ref["path"])==root_output/(packet["name"]+"_child_input.json")
        and local(packet["short_tmp_directory"])==amendment.TMP/packet["name"],
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
    from abstract_unix_bootstrap_v1 import bootstrap_native_execution_v1
    ipc_bootstrap=bootstrap_native_execution_v1("c3mlm-v12",physical_gpu=spec["physical_gpu"],gpu_uuid=spec["execution_gpu_uuid"])
    _save_json(directory/"operational_IPC_bootstrap.json",ipc_bootstrap)
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
        runtime_ref=spec["refs"]["runtime"],runtime_validation=runtime_receipt,physical_gpu=lease["physical_gpu"],operational_IPC_bootstrap=ipc_bootstrap,
        query_ancestry_truth_opened=False,author_code_modified=False,initializer_checkpoint_ref=None,corpus_authentication_ref=None)
    # Once, at the quiescent pre-load boundary; no decode/copy or per-update hash.
    from corpus_auth_v2 import authenticate_inputs,confirm_unchanged
    authenticated=authenticate_inputs(prep,prepared_dir)
    authenticated["preparation_ref"]=spec["refs"]["preparation"]
    _save_json(directory/"corpus_authentication.json",authenticated)
    result["corpus_authentication_ref"]=ref(directory/"corpus_authentication.json")
    OmegaConf.save(config,directory/"resolved_io_config.yaml",resolve=True)
    confirm_unchanged(authenticated,prepared_dir)
    trainer=module.main(config,return_trainer=True)
    observed_trainer=dict(epoch=_int_time(trainer.state.timestamp.epoch),batch=_int_time(trainer.state.timestamp.batch),
        batch_in_epoch=_int_time(trainer.state.timestamp.batch_in_epoch))
    observed_optimizer=optimizer_counter_evidence({type(opt).__qualname__:opt.state_dict() for opt in trainer.state.optimizers})
    require(observed_trainer["epoch"]==5 and observed_trainer["batch"]==224775
        and observed_optimizer["unique_step_values"]==[224775],
        "Actual full epoch5/update224775 Trainer timestamp")
    if hasattr(trainer,"close"):trainer.close()
    final=_final_checkpoint(directory/"checkpoints",epochs=5,updates=224775,
        max_files=cfg["max_checkpoint_files"],max_bytes=cfg["max_checkpoint_bytes"],torch=torch,reference=ref)
    result.update(final)
    result.update(trainer_timestamp=observed_trainer,trainer_optimizer_counters=observed_optimizer,
        expected_training_budget=dict(epochs=5,optimizer_updates=224775),observed_counters_verified=True)
    result["resolved_config_ref"]=ref(directory/"resolved_io_config.yaml")
    ctx["verify_runtime_imports"]();_verify_author_import_origins(author_root);ctx["fast_guard"]("after_native_author_call")
    _save_json(directory/"completed.json",result);return 0
if __name__=="__main__":raise SystemExit(main())

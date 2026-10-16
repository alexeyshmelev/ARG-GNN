"""One fresh native ARGformer MLM worker. Operational adaptation only.
The author I/O config, main call, numeric definitions, full counters and checkpoint
verification are copied unchanged from the sealed native predecessor.
"""
from pathlib import Path
import hashlib,importlib,importlib.util,json,os,resource,stat,sys
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import common_v1 as c
ROOT=c.ROOT

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

def _actual_context_uuid(c,lease,torch):
    """Reuse the current-context driver method in existing biological_candidate.

    Torch2.4 lacks a UUID property. This I/O-only adaptation retains evidence in
    the returned runtime receipt rather than writing a qualification directory.
    """
    import ctypes,uuid
    torch.cuda.set_device(0)
    keep_context=torch.empty(1,device=torch.device("cuda",0));torch.cuda.synchronize()
    c.require(torch.cuda.current_device()==0,"Exact single logical CUDA0 context")
    drivers=set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)==6 and Path(parts[5]).name.startswith("libcuda.so."):
            path=Path(parts[5])
            c.require(path.is_absolute() and path.is_file(),"Actual own loaded CUDA driver file")
            drivers.add(path.resolve())
    c.require(len(drivers)==1,"Exactly one own loaded CUDA driver")
    driver_path=drivers.pop();driver=ctypes.CDLL(str(driver_path));device=ctypes.c_int(-1)
    lookup=driver.cuCtxGetDevice;lookup.argtypes=[ctypes.POINTER(ctypes.c_int)];lookup.restype=ctypes.c_int
    c.require(lookup(ctypes.byref(device))==0,"Actual driver context device lookup")
    class CudaUUID(ctypes.Structure):
        _fields_=[("bytes",ctypes.c_ubyte*16)]
    identity=CudaUUID();name="cuDeviceGetUuid_v2" if hasattr(driver,"cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    lookup=getattr(driver,name);lookup.argtypes=[ctypes.POINTER(CudaUUID),ctypes.c_int];lookup.restype=ctypes.c_int
    c.require(lookup(ctypes.byref(identity),device.value)==0,"Actual driver context UUID lookup")
    actual=str(uuid.UUID(bytes=bytes(identity.bytes)))
    c.require(actual==lease["canonical_uuid"],"Actual CUDA context UUID equals held original GPU0 lease")
    del keep_context
    return dict(actual_context_uuid=actual,driver_library=str(driver_path),uuid_function=name,
        context_device=device.value,method_source_ref=c.ref("src/arggnn/biological_candidate.py"))

def run_v1(root_ref,spec,q,cfg,ticket,guard):
    require0=c.require
    require0("torch" not in sys.modules and "torch_geometric" not in sys.modules,"Install native direct lease/title before science")
    require0(os.getpid()==os.getpgrp()==os.getsid(0),"One actual owned native session")
    lease=guard.validate_lease();guard.apply_and_validate_title()
    require0(lease["physical_gpu"]==0 and lease["borrowed_physical_inode"] is True,"Actual native borrowed GPU0 attachment")
    runtime=c.read(q["runtime_ref"])
    require0(Path(sys.executable).resolve()==c.local(c.NATIVE_PYTHON_REF["path"])
        and Path(sys.prefix).resolve()==c.local(runtime["native_prefix"])
        and sys.dont_write_bytecode and sys.flags.no_user_site and os.environ.get("PYTHONPATH","")==""
        and os.environ.get("PYTHONNOUSERSITE")=="1","Exact isolated native interpreter and no user site")
    sys.path[:0]=[str(ROOT/"src"),str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12"),
        str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5")]
    from arggnn.american_c3_common_v1 import require,local,ref,bound,CLASSES
    from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import native_runtime_v1, gpu0_abstract_ipc_adapter_v3
    runtime_receipt,verify_runtime=native_runtime_v1.validate_native_runtime_v1(q["runtime_ref"],runtime)
    require(runtime_receipt.get("status")=="passed","Exact full native inventory before scientific imports")
    gpu0_abstract_ipc_adapter_v3.install_private_module_before_native_child_native32_v1()
    directory=local(spec["output_directory"])
    require(not directory.exists(),"Fresh native scientific output, no replay")
    directory.mkdir()
    resource.setrlimit(resource.RLIMIT_FSIZE,(cfg["max_checkpoint_bytes"],cfg["max_checkpoint_bytes"]))
    from abstract_unix_bootstrap_v1 import bootstrap_native_execution_v1
    ipc_bootstrap=bootstrap_native_execution_v1("c3-val000-native32-mlm-v1",physical_gpu=0,gpu_uuid=c.GPU0_UUID)
    _save_json(directory/"operational_IPC_bootstrap.json",ipc_bootstrap)
    _,prep,counts,_=c.science_v1(q);prepared_dir=local(c.PREP_OUTPUT)
    AUTHOR_ROOT=c.AUTHOR_ROOT;AUTHOR_MANIFEST_REF=c.AUTHOR_MANIFEST_REF
    metadata=c.read
    packet={"stage":"mlm"};packet_ref=root_ref
    spec=dict(spec,parent_id=c.PARENT,physical_gpu=0,execution_gpu_uuid=c.GPU0_UUID,
        refs=dict(spec["refs"],preparation=c.PREP_REF,preparation_config=c.PREP_CONFIG_REF,counts=c.COUNTS_REF))
    def fast_guard(event):
        c.active(spec);guard.apply_and_validate_title()
        require(guard.validate_lease()==lease,"Stable native32 borrowed physical and owned logical attachment")
    def configure_device(value,actual_lease):
        require(value==runtime and actual_lease==lease,"Exact native runtime and attachment")
        fast_guard("before_first_cuda_tensor")
        import torch
        require(torch.cuda.device_count()==1,"Only one actual visible CUDA device")
        device=torch.device("cuda",0);props=torch.cuda.get_device_properties(device)
        # This precedes the first tensor, including the UUID context tensor.
        torch.cuda.set_per_process_memory_fraction(c.CAPS["max_cuda_reserved_bytes"]/props.total_memory,device)
        identity=_actual_context_uuid(c,lease,torch)
        runtime_receipt.update(actual_device_identity=identity,actual_total_memory_bytes=props.total_memory,
            max_cuda_reserved_bytes=c.CAPS["max_cuda_reserved_bytes"])
        verify_runtime();fast_guard("after_native_device")
    ctx=dict(configure_native_device=configure_device,verify_runtime_imports=verify_runtime,fast_guard=fast_guard)
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
        and resolved["max_seq_len"]==128 and resolved["save_interval"]=="500ba"
        and resolved["save_num_checkpoints_to_keep"]==1,"Unchanged full author numeric budget and checkpoint retention")
    module=importlib.import_module("main")
    require(Path(module.__file__).resolve()==author_root/"main.py","Exact author main origin")
    _verify_author_import_origins(author_root)
    result=dict(schema="american_c3_native_mlm_child_completion_v1",status="completed_C3_native_MLM_child",
        stage=packet["stage"],parent_id=spec["parent_id"],active_input_ref=root_ref,actual_native_root_input_ref=packet_ref,
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
    observed_vocab=len(trainer.state.model.tokenizer)
    require(observed_vocab==107144,"Actual unchanged author tokenizer uses this ARG vocabulary")
    result["observed_tokenizer_vocab_size"]=observed_vocab
    observed_trainer=dict(epoch=_int_time(trainer.state.timestamp.epoch),batch=_int_time(trainer.state.timestamp.batch),
        batch_in_epoch=_int_time(trainer.state.timestamp.batch_in_epoch))
    observed_optimizer=optimizer_counter_evidence({type(opt).__qualname__:opt.state_dict() for opt in trainer.state.optimizers})
    require(observed_trainer["epoch"]==5 and observed_trainer["batch"]==227990
        and observed_optimizer["unique_step_values"]==[227990],
        "Actual full epoch5/update227990 Trainer timestamp")
    if hasattr(trainer,"close"):trainer.close()
    final=_final_checkpoint(directory/"checkpoints",epochs=5,updates=227990,
        max_files=cfg["max_checkpoint_files"],max_bytes=cfg["max_checkpoint_bytes"],torch=torch,reference=ref)
    result.update(final)
    result.update(trainer_timestamp=observed_trainer,trainer_optimizer_counters=observed_optimizer,
        expected_training_budget=dict(epochs=5,optimizer_updates=227990),observed_counters_verified=True)
    result["resolved_config_ref"]=ref(directory/"resolved_io_config.yaml")
    ctx["verify_runtime_imports"]();_verify_author_import_origins(author_root);ctx["fast_guard"]("after_native_author_call")
    _save_json(directory/"completed.json",result);return 0

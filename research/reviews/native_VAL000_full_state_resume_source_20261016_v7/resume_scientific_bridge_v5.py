"""Inactive full-state operational resume bridge over unchanged author functions.
No checkpoint, Torch, process, device or runtime access occurs at import time.
"""
from pathlib import Path
from datetime import timedelta
import ast,copy,hashlib,os,stat,sys,types
from research.reviews.native_VAL000_full_state_resume_source_20261016_v6 import amendment_contract_v1 as c
SOURCE=c.HERE+"/resume_scientific_bridge_v5.py"
def validated_composer_timestamp_v1(state):
    outer=state.get("timestamp")
    c.require(type(outer) is dict and set(outer)=={"Timestamp"},"Exact installed Composer nested Timestamp serialization")
    value=outer["Timestamp"]
    counters={"iteration","epoch","batch","sample","token","epoch_in_iteration","token_in_iteration","batch_in_epoch","sample_in_epoch","token_in_epoch"}
    walls={"total_wct","iteration_wct","epoch_wct","batch_wct"}
    c.require(type(value) is dict and set(value)==counters|walls
        and all(type(value[k]) is int and value[k]>=0 for k in counters)
        and all(isinstance(value[k],timedelta) and value[k].total_seconds()>=0 for k in walls),
        "Exact actual native timestamp keys, nonnegative integer counters and timedeltas")
    return value

def checkpoint_metadata_bridge_v1(helper):
    # One metadata-load expression only. Original full model/optimizer counters,
    # epoch/update budget, file/alias selection and ambiguity checks stay intact.
    record=c.original.ref(c.original.CONTEXT);c.raw(record)
    source=c.local(record["path"]).read_text();tree=ast.parse(source)
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_final_checkpoint"))
    matches=[n for n in ast.walk(function) if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="timestamp"]
    expected=ast.parse("timestamp=state.get("+chr(34)+"timestamp"+chr(34)+",{})").body[0]
    c.require(len(matches)==1 and ast.dump(matches[0],include_attributes=False)==ast.dump(expected,include_attributes=False),
        "Only original direct timestamp lookup is adapted for installed Composer schema")
    matches[0].value=ast.Call(func=ast.Name(id="_actual_native_nested_timestamp_v1",ctx=ast.Load()),args=[ast.Name(id="state",ctx=ast.Load())],keywords=[])
    ns=dict(helper._final_checkpoint.__globals__,_actual_native_nested_timestamp_v1=validated_composer_timestamp_v1)
    ast.fix_missing_locations(function)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(c.local(SOURCE))+":nested_timestamp_metadata_bridge","exec"),ns)
    adapted=ns["_final_checkpoint"]
    def checked(*args,**kwargs):
        result=adapted(*args,**kwargs)
        return dict(result,actual_checkpoint_timestamp_schema="state.timestamp.Timestamp",
            checkpoint_metadata_parser_bridge_source_ref=c.ref(SOURCE),original_checkpoint_model_optimizer_and_selection_checks_preserved=True)
    return checked

def _stamp(value):
    return dict(device=value.st_dev,inode=value.st_ino,bytes=value.st_size,mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,link_count=value.st_nlink)

class ResumeSession:
    def __init__(self,native,spec):
        self.native=native;self.spec=spec;self.fd=None;self.path=None
        self.verified=None;self.restore_receipt=None;self.installed_checkpoint=None;self.original_loader=None
    def stable(self):
        c.active_amendment_v1()
        expected=c.RESUME_STATE["checkpoint_identity"]
        c.require(self.fd is not None and _stamp(os.fstat(self.fd))==expected
            and _stamp(c.local(c.RESUME_STATE["checkpoint_ref"]["path"]).lstat())==expected,
            "Exact retained immutable original checkpoint identity throughout restore")
    def prepare(self,torch):
        c.require(self.fd is None,"One exact full-state checkpoint restore per worker")
        qualified=c.RESUME_STATE
        path=c.local(qualified["checkpoint_ref"]["path"])
        before=path.lstat()
        c.require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=2*2**30
            and _stamp(before)==qualified["checkpoint_identity"],"ROOT-qualified immutable single-link original checkpoint")
        self.fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
        self.path="/proc/self/fd/"+str(self.fd);self.stable()
        sha=hashlib.sha256()
        while True:
            c.active_amendment_v1()
            block=os.read(self.fd,8*2**20)
            if not block:break
            sha.update(block)
        self.stable()
        c.require(sha.hexdigest()==qualified["checkpoint_ref"]["sha256"],"Actual full checkpoint SHA through own stable read-only FD")
        os.lseek(self.fd,0,os.SEEK_SET)
        value=torch.load(self.path,map_location="cpu",weights_only=False,mmap=True)
        self.stable()
        state=value.get("state",{})
        timestamp=validated_composer_timestamp_v1(state)
        counters=self.native.optimizer_counter_evidence(state.get("optimizers"))
        progress=qualified["actual_saved_progress"]
        c.require(timestamp["epoch"]==progress["epoch"] and timestamp["batch"]==progress["batch"]
            and timestamp["batch_in_epoch"]==progress["batch_in_epoch"]
            and counters["unique_step_values"]==progress["optimizer_unique_steps"]
            and counters["parameter_state_count"]==77
            and isinstance(state.get("model"),dict) and bool(state["model"])
            and isinstance(state.get("schedulers"),dict) and bool(state["schedulers"])
            and isinstance(value.get("rng"),list) and bool(value["rng"])
            and all(isinstance(row,dict) and bool(row) for row in value["rng"]),
            "Actual full model/Adam moments/steps/scheduler/timestamp/RNG, not weights-only or guessed progress")
        self.verified=dict(schema="native_VAL000_actual_full_state_restore_input_v5",
            checkpoint_ref=qualified["checkpoint_ref"],ROOT_qualification_ref=c.read(c.current_amendment_ref_v1())["ROOT_resume_qualification_ref"],
            original_input_ref=c.ORIGINAL_INPUT,source_ref=c.ref(SOURCE),actual_FD_full_SHA_verified=True,
            checkpoint_identity=qualified["checkpoint_identity"],saved_progress=progress,optimizer_counters=counters,
            model_optimizer_scheduler_RNG_timestamp_present=True,checkpoint_payload_copied_to_disk=False,
            query_ancestry_truth_opened=False)
        del value,state
        return self.verified
    def install_original_Composer_FD_loader(self):
        from composer.utils import checkpoint
        c.require(self.original_loader is None,"One private exact Composer full-state loader bridge")
        original=checkpoint.load_checkpoint
        def download(path,node_checkpoint_folder,object_store,progress_bar,fsdp_sharded_state_dict_enabled=False,deepspeed_sharded_checkpoint=False):
            self.stable()
            c.require(path==self.path and object_store is None
                and fsdp_sharded_state_dict_enabled is False and deepspeed_sharded_checkpoint is False,
                "Only exact one-rank local immutable FD checkpoint, no TMP symlink/copy or remote fallback")
            return self.path,None,False
        original_code=types.FunctionType(original.__code__,dict(original.__globals__,download_checkpoint=download),
            original.__name__,original.__defaults__,original.__closure__)
        original_code.__kwdefaults__=original.__kwdefaults__
        def full_state(*args,**kwargs):
            import inspect
            bound=inspect.signature(original).bind(*args,**kwargs);bound.apply_defaults()
            values=bound.arguments
            c.require(values["path"]==self.path and values["load_weights_only"] is False
                and values["strict_model_weights"] is True and values["ignore_keys"] is None
                and values["exclude_algorithms"] is None and values["object_store"] is None,
                "Original complete strict full-state Composer restore, no state omission")
            self.stable();rng=original_code(*args,**kwargs);self.stable()
            state=values["state"];progress=c.RESUME_STATE["actual_saved_progress"]
            counters=self.native.optimizer_counter_evidence({type(opt).__qualname__:opt.state_dict() for opt in state.optimizers})
            c.require(state.timestamp.epoch.value==progress["epoch"] and state.timestamp.batch.value==progress["batch"]
                and state.timestamp.batch_in_epoch.value==progress["batch_in_epoch"]
                and counters["unique_step_values"]==progress["optimizer_unique_steps"]
                and bool(state.schedulers) and isinstance(rng,list) and bool(rng),
                "Actual restored optimizer/scheduler/time/RNG matches full saved state before any resumed update")
            self.restore_receipt=dict(schema="native_VAL000_actual_Composer_full_state_restore_v5",
                checkpoint_ref=c.RESUME_STATE["checkpoint_ref"],load_weights_only=False,
                saved_progress=progress,actual_restored_optimizer_counters=counters,
                actual_scheduler_and_RNG_returned=True,data_position_restored_by_original_Composer_spin=True,
                bitwise_worker_mask_replay_equivalence_claimed=False,source_ref=c.ref(SOURCE))
            return rng
        self.original_loader=original;self.installed_checkpoint=checkpoint;checkpoint.load_checkpoint=full_state
    def config(self,original,*args,**kwargs):
        import torch
        cfg=original(*args,**kwargs)
        from omegaconf import OmegaConf
        before=OmegaConf.to_container(cfg,resolve=True)
        c.require((args[0] if args else kwargs.get("mode"))=="mlm","MLM full-state resume only")
        c.require(str(cfg.max_duration)=="5ep" and cfg.load_path is None and cfg.autoresume is False,
            "Original scientific duration retained before explicit operational restore")
        self.prepare(torch);self.install_original_Composer_FD_loader()
        cfg.load_path=self.path;cfg.load_weights_only=False;cfg.autoresume=False
        cfg.restart_override=False
        c.require(OmegaConf.to_container(cfg,resolve=True)==dict(before,load_path=self.path,
            load_weights_only=False,autoresume=False,restart_override=False),
            "Only four explicit operational full-state resume config fields differ")
        c.require(cfg.save_num_checkpoints_to_keep==1 and cfg.save_interval=="500ba"
            and cfg.global_train_batch_size==cfg.device_train_microbatch_size==512
            and cfg.precision=="amp_bf16" and cfg.seed==17
            and cfg.train_loader.num_workers==cfg.eval_loader.num_workers==8,
            "Exact original batch/precision/seed/workers/checkpoint events, no speed or science fallback")
        self.native._save_json(c.local(self.spec["output_directory"])/"actual_full_state_resume_input.json",self.verified)
        return cfg
    def close(self):
        if self.installed_checkpoint is not None:self.installed_checkpoint.load_checkpoint=self.original_loader
        if self.fd is not None:os.close(self.fd);self.fd=None

def run_resumed_native_v5(root_ref,spec,q,cfg,ticket,guard):
    c.require(c.RESUME_STATE is not None and cfg["max_checkpoint_bytes"]==2*2**30,
        "Exact inactive ROOT-bound original full-state qualifier and new-product hard file cap")
    from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import native_child_v1 as native
    session=ResumeSession(native,spec);io_config=native._io_config
    def io(*args,**kwargs):return session.config(io_config,*args,**kwargs)
    original_save=native._save_json
    def save(path,value):
        if Path(path).name=="completed.json":
            c.require(session.restore_receipt is not None,"Genuine original Composer full-state restore occurred")
            value=dict(value,child_input_ref=root_ref,actual_execution_amendment_ref=c.current_amendment_ref_v1(),
                adapter_manifest_ref=c.ref(c.MANIFEST),actual_original_full_state_resume=True,
                original_fresh_run_initializer_checkpoint_ref=None,resume_checkpoint_ref=c.RESUME_STATE["checkpoint_ref"],
                full_state_resume_input=session.verified,actual_Composer_restore=session.restore_receipt,
                original_failed_v4_wait_and_outer_preserved=True,original_full_epochs=5,original_full_updates=227990,
                original_checkpoint_timestamp_schema_bridge=True,original_author_scientific_calls_modified=False)
        c.active_amendment_v1()
        original_save(path,value)
    ns=dict(native.run_v1.__globals__,c=c,_io_config=io,_final_checkpoint=checkpoint_metadata_bridge_v1(native),_save_json=save)
    run=types.FunctionType(native.run_v1.__code__,ns,native.run_v1.__name__,native.run_v1.__defaults__,native.run_v1.__closure__)
    try:return run(root_ref,spec,q,cfg,ticket,guard)
    finally:session.close()

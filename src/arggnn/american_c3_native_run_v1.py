"""Canonical guarded arggnn CLI for fresh per-ARG native C3 phases.

Standard-library-only supervisor. Root injects existing title/lease/resource,
runtime and storage authority. No accounting/admission/launcher is implemented.
While a native child can write, only fast resource checks and bounded RAM pipe
reads occur. Output scans, ref hashes, log writes and boundary guards wait for
actual direct wait, adopted-child reaping and process-group ESRCH.
"""
from __future__ import annotations
import argparse,collections,ctypes,importlib.util,json,os,selectors,signal,subprocess,sys,time
from pathlib import Path
from .american_c3_common_v1 import ROOT,CLASSES,require,local,ref,bound
from .american_c3_native_contract_v1 import (
    SOURCE as CONTRACT_SOURCE,AUTHOR_MANIFEST_REF,NATIVE_GUARD_REF,DEFAULTS,GPU_PHASES,
    metadata,validate_active_input,load_root_context,preparation_v1,counts_v1,fit_completion_v1)
SOURCE="src/arggnn/american_c3_native_run_v1.py"
CHILD="src/arggnn/american_c3_native_child_v1.py"
CACHE_VARS=("XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME","HF_HOME","TORCH_HOME",
    "TRITON_CACHE_DIR","CUDA_CACHE_PATH","TORCHINDUCTOR_CACHE_DIR","NUMBA_CACHE_DIR",
    "MPLCONFIGDIR","WANDB_DIR","WANDB_CACHE_DIR","WANDB_CONFIG_DIR","HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE","HF_DATASETS_CACHE","TRANSFORMERS_CACHE","TORCH_EXTENSIONS_DIR")
class NativeChildFailure(RuntimeError):
    def __init__(self,message,*,returncode,wait_ref):
        super().__init__(message);self.returncode=returncode;self.wait_ref=wait_ref
class _RAMLog:
    def __init__(self,cap):self.cap=cap;self.parts=collections.deque();self.kept=0;self.total=0
    def add(self,data):
        self.total+=len(data);self.parts.append(data);self.kept+=len(data)
        while self.parts and self.kept>self.cap:
            first=self.parts.popleft();over=self.kept-self.cap
            if len(first)>over:self.parts.appendleft(first[over:]);self.kept-=over
            else:self.kept-=len(first)
    def record(self):
        return dict(tail=b"".join(self.parts).decode("utf-8",errors="replace"),
            total_pipe_bytes=self.total,retained_raw_bytes=self.kept,cap_bytes=self.cap,
            discarded_earlier_bytes=self.total-self.kept,buffered_in_RAM_until_actual_drain=True)
def _title_lease():
    p=bound(NATIVE_GUARD_REF);loaded=importlib.util.spec_from_file_location("_C3_native_existing_guard",p)
    module=importlib.util.module_from_spec(loaded);loaded.loader.exec_module(module)
    lease=module.validate_lease();module.apply_and_validate_title()
    require(lease["physical_gpu"] in (4,5),"Only current authorized physical GPUs4/5")
    return module,lease
def _subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,"Linux subreaper needed to adopt/drain native DataLoader descendants")
def _children():
    raw=Path("/proc/self/task/%d/children"%os.getpid()).read_text().strip()
    return [int(p) for p in raw.split()] if raw else []
def _group_exists(pgid):
    try:os.killpg(pgid,0);return True
    except ProcessLookupError:return False
def _signal_group(pgid,signum):
    try:os.killpg(pgid,signum)
    except ProcessLookupError:pass
def _reap_group(pgid,records):
    while True:
        try:pid,status=os.waitpid(-pgid,os.WNOHANG)
        except ChildProcessError:return
        if pid==0:return
        if len(records)<1024:
            records.append(dict(pid=pid,raw_wait_status=status,exitcode=os.waitstatus_to_exitcode(status)))
def _environment(directory,tmp,runtime,lease):
    env=os.environ.copy();env.pop("PYTHONHOME",None)
    for key in ("RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","NODE_RANK","MASTER_ADDR","MASTER_PORT"):
        env.pop(key,None)
    env.update(PYTHONPATH="",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",WANDB_MODE="disabled",
        CONDA_PREFIX=runtime["native_prefix"],TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp))
    env["PATH"]=str(local(runtime["native_prefix"])/"bin")+os.pathsep+env.get("PATH","")
    for name in CACHE_VARS:
        folder=directory/"write_scopes"/name.lower();folder.mkdir(parents=True,exist_ok=False);env[name]=str(folder)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[name]="4"
    return env
def _run_child(packet,*,record,spec,cfg,ctx,lease):
    name=packet["name"];writer=ctx["writer"]
    ctx["boundary_guard"]("before_native_child",required_write_bytes=0)
    directory=local(writer.path/name);require(not directory.exists(),"Fresh exclusive native stage scope")
    directory.mkdir()
    tmp_base=local(cfg["short_tmp_directory"]);tmp=tmp_base/name
    storage=metadata(spec["refs"]["storage"])
    auxiliary=[local(x) for x in storage["reservation"].get("auxiliary_scopes",[])]
    require(any(tmp_base==a or a in tmp_base.parents for a in auxiliary),"Short native TMPDIR must be an admitted auxiliary scope")
    require(len(os.fsencode(tmp/"pymp-12345678/listener-12345678"))<=107,"Repository-local native IPC socket bound")
    require(not tmp.exists(),"Fresh exclusive stage temporary directory");tmp.mkdir(parents=True)
    runtime=metadata(spec["refs"]["runtime"]);python=bound(runtime["native_python_ref"])
    require(python.parent==local(runtime["native_prefix"])/"bin","Actual admitted isolated native interpreter")
    env=_environment(directory,tmp,runtime,lease)
    child_input=dict(packet,schema="american_c3_native_child_input_v1",active_input_ref=record,
        directory=str(directory.relative_to(ROOT)),short_tmp_directory=str(tmp.relative_to(ROOT)),
        lease_metadata_sha256=lease["lease_metadata_sha256"])
    child_ref=writer.save(name+"_child_input.json",child_input)
    command=[str(python),"-B","-s",str(ROOT/CHILD),"--child-input",child_ref["path"],
        "--expected-child-input-sha256",child_ref["sha256"]]
    ctx["fast_guard"]("before_native_spawn")
    require(not _children(),"No unrelated pre-existing child process in native supervisor")
    _subreaper();proc=None;interruption=None;guard_error=None;old_handlers={};adopted=[]
    logs=_RAMLog(cfg["child_log_ram_bytes"]);selector=selectors.DefaultSelector();drained=False
    def stop(signum,_frame):
        nonlocal interruption
        interruption=signum
        if proc is not None:_signal_group(proc.pid,signum)
    try:
        for signum in (signal.SIGTERM,signal.SIGINT):
            old_handlers[signum]=signal.signal(signum,stop)
        proc=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,
            pass_fds=(lease["lease_fd"],),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
        os.set_blocking(proc.stdout.fileno(),False);selector.register(proc.stdout,selectors.EVENT_READ)
        stop_at=None;next_guard=0.
        # No writer/boundary/scope measurement exists in this live-child loop.
        while proc.poll() is None or selector.get_map():
            for key,_ in selector.select(timeout=.2):
                data=os.read(key.fileobj.fileno(),65536)
                if data:logs.add(data)
                else:selector.unregister(key.fileobj)
            now=time.monotonic()
            if guard_error is None and now>=next_guard:
                try:ctx["fast_guard"]("native_child_active",child_pid=proc.pid,process_group=proc.pid)
                except BaseException as error:guard_error=error
                next_guard=now+float(cfg["guard_poll_seconds"])
            if guard_error is not None or interruption is not None:
                if stop_at is None:stop_at=now;_signal_group(proc.pid,signal.SIGTERM)
                if now-stop_at>=float(cfg["termination_grace_seconds"]):_signal_group(proc.pid,signal.SIGKILL)
            # If direct child exited while descendants hold its pipe open, drain
            # its owned group instead of waiting forever for a forgotten worker.
            if proc.poll() is not None and selector.get_map():
                for child_pid in _children():
                    try:
                        if os.getpgid(child_pid)!=proc.pid:os.kill(child_pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                if stop_at is None:stop_at=now;_signal_group(proc.pid,signal.SIGTERM)
                if now-stop_at>=float(cfg["termination_grace_seconds"]):_signal_group(proc.pid,signal.SIGKILL)
        returncode=proc.wait()  # actual direct wait, also on failure/cancellation
        end_started=time.monotonic();_signal_group(proc.pid,signal.SIGTERM)
        while _group_exists(proc.pid) or _children():
            _reap_group(proc.pid,adopted)
            # Child packages must not escape the owned session. Any adopted
            # escape is still our direct descendant, never another user's PID.
            for child_pid in _children():
                try:
                    if os.getpgid(child_pid)!=proc.pid:os.kill(child_pid,signal.SIGKILL)
                except ProcessLookupError:pass
            if time.monotonic()-end_started>=float(cfg["termination_grace_seconds"]):
                _signal_group(proc.pid,signal.SIGKILL)
            # Reap adopted escaped descendants too, after direct wait.
            while True:
                try:pid,status=os.waitpid(-1,os.WNOHANG)
                except ChildProcessError:break
                if pid==0:break
                if len(adopted)<1024:
                    adopted.append(dict(pid=pid,raw_wait_status=status,exitcode=os.waitstatus_to_exitcode(status)))
            if not _group_exists(proc.pid) and not _children():break
            time.sleep(.1)
        drained=not _group_exists(proc.pid) and not _children()
        require(drained,"Actual process-group ESRCH and no adopted live writer required")
    finally:
        # On an unexpected supervisor exception, keep ownership and finish
        # termination/directwait/reaping before any final scope access.
        if proc is not None and not drained:
            _signal_group(proc.pid,signal.SIGKILL)
            proc.wait()
            while _group_exists(proc.pid) or _children():
                _reap_group(proc.pid,adopted)
                for child_pid in _children():
                    try:os.kill(child_pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                while True:
                    try:pid,status=os.waitpid(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if pid==0:break
                    if len(adopted)<1024:adopted.append(dict(pid=pid,raw_wait_status=status,exitcode=os.waitstatus_to_exitcode(status)))
                if _group_exists(proc.pid) or _children():time.sleep(.1)
            drained=True
        selector.close()
        if proc is not None and proc.stdout is not None:proc.stdout.close()
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
    # Native child and every adopted descendant are now terminal. Only here may
    # existing journal/scope checks, output hashes and buffered log writes run.
    ctx["boundary_guard"]("after_native_child_actual_wait_and_drain",required_write_bytes=0)
    log_ref=writer.save(name+"_native_log.json",logs.record())
    completion_path=directory/"completed.json";completion_ref=ref(completion_path) if completion_path.is_file() else None
    wait=dict(schema="american_c3_native_direct_wait_and_drain_v1",parent_id=spec["parent_id"],
        native_child_pid=proc.pid,process_group=proc.pid,returncode=returncode,
        actual_direct_child_wait=True,process_group_ESRCH=drained,no_adopted_live_writer=not _children(),
        adopted_descendant_waits=adopted,adopted_wait_records_may_be_truncated=(len(adopted)>=1024),
        child_input_ref=child_ref,child_completion_ref=completion_ref,
        log_ref=log_ref,guard_failure_type=type(guard_error).__name__ if guard_error else None,
        interruption_signal=interruption,all_logs_buffered_until_terminal=True)
    wait_ref=writer.save(name+"_actual_wait.json",wait)
    if guard_error is not None:
        if hasattr(guard_error,"add_note"):guard_error.add_note("Actual native wait/drain: "+str(wait_ref))
        raise guard_error
    if interruption is not None:raise NativeChildFailure("Native phase interrupted",returncode=returncode or -interruption,wait_ref=wait_ref)
    if returncode!=0 or completion_ref is None:
        raise NativeChildFailure("Native child failed; artifacts preserved",returncode=returncode or 1,wait_ref=wait_ref)
    return metadata(completion_ref),completion_ref,wait_ref

def run_context_v1(record,spec,cfg,ctx,lease):
    prep,_,_=preparation_v1(spec);counts=counts_v1(spec,prep);writer=ctx["writer"]
    ctx["boundary_guard"]("before_complete_native_phase",required_write_bytes=cfg["phase_write_budget_bytes"])
    if spec["phase"]=="fit":
        phases={}
        for mode in ("mlm","contrastive"):
            packet=dict(name=mode,stage=mode)
            if mode=="contrastive":packet["initializer_checkpoint_ref"]=phases["mlm"]["checkpoint_ref"]
            receipt,completion_ref,wait_ref=_run_child(packet,record=record,spec=spec,cfg=cfg,ctx=ctx,lease=lease)
            require(receipt["stage"]==mode and receipt["parent_id"]==spec["parent_id"]
                and receipt["actual_optimizer_updates"]==counts["MLM_optimizer_updates" if mode=="mlm" else "contrastive_optimizer_updates"],
                "Actual same-parent exact-budget native phase completion")
            phases[mode]=dict(child_completion_ref=completion_ref,actual_wait_ref=wait_ref,checkpoint_ref=receipt["checkpoint_ref"])
        return writer.save("completed.json",dict(status="completed_C3_fresh_native_fit",parent_id=spec["parent_id"],
            class_order=list(CLASSES),preparation_ref=spec["refs"]["preparation"],counts_ref=spec["refs"]["counts"],
            author_source_manifest_ref=AUTHOR_MANIFEST_REF,phases=phases,uniform_class_weight_fallback_preserved=True,
            training_config_plumbing_correction=False,query_ancestry_truth_opened=False,
            model_reused_from_other_parent=False,author_Table1_exact_reproduction=False))
    fit=fit_completion_v1(spec["refs"]["fit_completion"],spec)
    checkpoint=fit["phases"]["contrastive"]["checkpoint_ref"]
    if spec["phase"]=="embeddings":
        outputs={}
        for view in ("query_paths","contrastive"):
            for split in ("train","val"):
                packet=dict(name=view+"_"+split,stage="embedding",view=view,split=split,checkpoint_ref=checkpoint)
                receipt,completion_ref,wait_ref=_run_child(packet,record=record,spec=spec,cfg=cfg,ctx=ctx,lease=lease)
                require(receipt["checkpoint_ref"]==checkpoint and receipt["view"]==view and receipt["split"]==split,
                    "Four exact single-split embedding producer completions")
                outputs[view+"/"+split]=dict(embedding_ref=receipt["embedding_ref"],
                    child_completion_ref=completion_ref,actual_wait_ref=wait_ref)
        return writer.save("completed.json",dict(status="completed_C3_native_four_single_split_embeddings",
            parent_id=spec["parent_id"],preparation_ref=spec["refs"]["preparation"],counts_ref=spec["refs"]["counts"],
            fit_completion_ref=spec["refs"]["fit_completion"],checkpoint_ref=checkpoint,outputs=outputs,
            random_multi_split_interleaving=False,query_ancestry_truth_opened=False))
    combined=metadata(spec["refs"]["combined_completion"])
    require(combined["status"]=="completed_C3_native_CPU_row_preserving_concat"
        and combined["parent_id"]==spec["parent_id"] and combined["fit_completion_ref"]==spec["refs"]["fit_completion"]
        and combined["preparation_ref"]==spec["refs"]["preparation"] and combined["counts_ref"]==spec["refs"]["counts"]
        and combined["checkpoint_ref"]==checkpoint and combined["splits"]==["train","val"]
        and combined["all_tensor_values_preserved_bit_for_bit"] is True,
        "Complete row-preserving CPU concat before unchanged retrieval")
    receipt,completion_ref,wait_ref=_run_child(dict(name="retrieve",stage="retrieve",
        corpus_ref=combined["combined_embeddings_refs"]["contrastive"],
        queries_ref=combined["combined_embeddings_refs"]["query_paths"]),
        record=record,spec=spec,cfg=cfg,ctx=ctx,lease=lease)
    return writer.save("completed.json",dict(status="completed_C3_native_top1_retrieval",
        parent_id=spec["parent_id"],preparation_ref=spec["refs"]["preparation"],counts_ref=spec["refs"]["counts"],
        fit_completion_ref=spec["refs"]["fit_completion"],combined_completion_ref=spec["refs"]["combined_completion"],
        checkpoint_ref=checkpoint,retrieval_ref=receipt["retrieval_ref"],child_completion_ref=completion_ref,
        actual_wait_ref=wait_ref,topk=1,q_batch=1024,e_chunk=100000,query_ancestry_truth_opened=False))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input");parser.add_argument("--expected-input-sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_inactive_canonical_C3_native_consumer",
            source_ref=ref(SOURCE),phases=list(GPU_PHASES),actual_execution=False),indent=2));return 0
    require(args.input and args.expected_input_sha256,"Actual active root native input path/SHA required")
    record=dict(path=args.input,sha256=args.expected_input_sha256);spec=metadata(record)
    # Existing standard-library title/lease guard precedes root context/runtime.
    _,lease=_title_lease()
    cfg,_=validate_active_input(record,spec,SOURCE,GPU=True)
    require(spec["phase"] in GPU_PHASES,"Canonical GPU consumer accepts native GPU phases only")
    ctx=load_root_context(record,spec)
    try:
        completion=run_context_v1(record,spec,cfg,ctx,lease)
        print(json.dumps(dict(completed_ref=completion)),flush=True);return 0
    except NativeChildFailure as error:
        print(json.dumps(dict(status="failed_native_child_preserved",returncode=error.returncode,
            actual_wait_ref=error.wait_ref)),flush=True)
        return error.returncode if error.returncode>0 else 128-error.returncode
if __name__=="__main__":raise SystemExit(main())

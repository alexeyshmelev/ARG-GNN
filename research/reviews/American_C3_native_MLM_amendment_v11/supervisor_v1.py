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
import amendment_v11 as amendment
from arggnn.american_c3_common_v1 import ROOT,CLASSES,require,local,ref,bound
from consumer_contract_v1 import (
    SOURCE as CONTRACT_SOURCE,AUTHOR_MANIFEST_REF,NATIVE_GUARD_REF,DEFAULTS,GPU_PHASES,
    metadata,validate_active_input,load_root_context,preparation_v1,counts_v1)
SOURCE=amendment.ENTRY
CHILD="research/reviews/American_C3_native_MLM_amendment_v11/native_child_v1.py"
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
    except PermissionError:return True


def _environment(directory,tmp,runtime,lease):
    env=os.environ.copy();env.pop("PYTHONHOME",None)
    for key in ("RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","NODE_RANK","MASTER_ADDR","MASTER_PORT"):
        env.pop(key,None)
    env.update(PYTHONPATH="",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",WANDB_MODE="disabled",
        CONDA_PREFIX=runtime["native_prefix"],TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp))
    env.update(HF_DATASETS_OFFLINE="1",PIP_NO_INDEX="1",PIP_DISABLE_PIP_VERSION_CHECK="1",CUDA_CACHE_MAXSIZE=str(2**30))
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
    tmp_base=amendment.TMP;tmp=tmp_base/name
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

    from owned_lifecycle_v2 import OwnedTree,signal_identity,_stat as owned_stat
    from pidfd_backend_v1 import self_probe_v1
    self_probe_v1()  # exact own-pidfd signal0/CLOEXEC backend probe before Popen
    require(not _children(),"No unrelated pre-existing child process in native supervisor")
    _subreaper();proc=None;ownership=None;interruption=None;guard_error=None;old_handlers={};adopted=[]
    from progress_v8 import Progress
    progress=Progress()
    terminal=amendment.capture("native");direct_captured=False
    logs=_RAMLog(cfg["child_log_ram_bytes"]);selector=selectors.DefaultSelector();drained=False;group_esrch=False
    def capture_direct(rc):
        nonlocal direct_captured,guard_error
        if not direct_captured:
            try:
                print(json.dumps(dict(event="actual_direct_native_wait_v9",pid=proc.pid,returncode=rc,
                    actual_direct_child_wait=True,drain_state="not_yet_certified"),separators=(",",":")),flush=True)
            except BaseException as error:
                guard_error=guard_error or error
        try:
            if not direct_captured:
                terminal.event(0,dict(schema="native_MLM_genuine_native_direct_wait_v9",pid=proc.pid,returncode=rc,
                    actual_direct_child_wait=True,drain_state="not_yet_certified",finished_unix=time.time()))
                direct_captured=True
            terminal.tail(logs)
        except BaseException as error:
            guard_error=guard_error or error  # Evidence failure never prevents actual writer drain.
    def stop(signum,_frame):
        nonlocal interruption
        interruption=signum  # loop signals only verified original identities via pidfd
    def reap_adopted():
        while True:
            try:pid,status=os.waitpid(-1,os.WNOHANG)
            except ChildProcessError:return
            if pid==0:return
            if len(adopted)<1024:adopted.append(dict(pid=pid,raw_wait_status=status,exitcode=os.waitstatus_to_exitcode(status)))
    def drain(force=False):
        nonlocal drained
        started=time.monotonic()
        while _children() or not ownership.identities_absent():
            signum=signal.SIGKILL if force or time.monotonic()-started>=float(cfg["termination_grace_seconds"]) else signal.SIGTERM
            ownership.signal_all(signum);reap_adopted()
            if _children() or not ownership.identities_absent():time.sleep(.1)
        drained=not _children() and ownership.identities_absent()
    try:
        for signum in (signal.SIGTERM,signal.SIGINT):old_handlers[signum]=signal.signal(signum,stop)
        proc=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,
            pass_fds=(lease["lease_fd"],amendment.lifetime_fd()),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
        ownership=OwnedTree(proc.pid)  # capture exact start identity before any poll/reap
        os.set_blocking(proc.stdout.fileno(),False);selector.register(proc.stdout,selectors.EVENT_READ)
        stop_at=None;next_guard=0.
        # No scope scan/writer/hash is present while the native process can write.
        while proc.poll() is None or selector.get_map():
            ownership.refresh()
            for key,_ in selector.select(timeout=.2):
                data=os.read(key.fileobj.fileno(),65536)
                if data:logs.add(data);progress.feed(data)
                else:selector.unregister(key.fileobj)
            now=time.monotonic()
            if guard_error is None and now>=next_guard:
                try:ctx["fast_guard"]("native_child_active",child_pid=proc.pid,process_group=proc.pid)
                except BaseException as error:guard_error=error
                next_guard=now+float(cfg["guard_poll_seconds"])
            if guard_error is not None or interruption is not None or (proc.poll() is not None and selector.get_map()):
                if stop_at is None:stop_at=now
                ownership.signal_all(signal.SIGKILL if now-stop_at>=float(cfg["termination_grace_seconds"]) else signal.SIGTERM)
        returncode=proc.wait();capture_direct(returncode)  # actual direct wait durable before any postprocessing
        drain()
        require(drained,"All original owned identities absent and adopted children reaped")
    finally:
        if proc is not None and not drained:
            if ownership is not None:
                ownership.signal_all(signal.SIGKILL);returncode=proc.wait();capture_direct(returncode);drain(force=True)
            else:
                # Failed startup before tracking: direct child remains unreaped.
                row=owned_stat(proc.pid)
                if row is not None and row["ppid"]==os.getpid():signal_identity(row,signal.SIGKILL)
                returncode=proc.wait();capture_direct(returncode)
                require(not _children(),"Startup failed before full ownership tracking; outer supervisor retains descendants")
        selector.close()
        if proc is not None and proc.stdout is not None:proc.stdout.close()
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
    require(proc is not None and ownership is not None and drained,"Actual direct child and complete identity drain required")
    capture_direct(returncode)
    tail_evidence=terminal.tail(logs)
    terminal.event(1,dict(schema="native_MLM_actual_native_drain_v9",pid=proc.pid,returncode=returncode,
        owned_identities_drained=drained,no_adopted_live_writer=not _children(),
        owned_identity_drain_proof=ownership.evidence(),raw_stdout_tail=tail_evidence))
    terminal.close()
    from terminal_namespace_v11 import clean_dead_sockets,normalize_latest
    prove=lambda:drained and not _children() and ownership.identities_absent()
    counter=[0]
    def preserve_namespace(value):
        counter[0]+=1;amendment.admin_save("native_namespace_%02d.json"%counter[0],value)
    try:
        clean_dead_sockets(tmp,prove_drained=prove,preserve=preserve_namespace)
        normalize_latest(directory/"checkpoints",prove_drained=prove,preserve=preserve_namespace,max_bytes=cfg["max_checkpoint_bytes"])
    except BaseException as error:
        guard_error=guard_error or error
        amendment.admin_save("native_normalization_failure.json",dict(type=type(error).__name__,message=str(error)[:2000]))

    # Observation only. A reused numeric group is never signaled or claimed absent.
    group_esrch=not _group_exists(proc.pid)
    if not group_esrch:guard_error=guard_error or ValueError("Numeric process group remains after original identities drained; no group ownership assumed")
    # Native child and every adopted descendant are now terminal. Only here may
    # existing journal/scope checks, output hashes and buffered log writes run.
    ctx["boundary_guard"]("after_native_child_actual_wait_and_drain",required_write_bytes=0)
    log_ref=writer.save(name+"_native_log.json",logs.record())
    completion_path=directory/"completed.json";completion_ref=ref(completion_path) if completion_path.is_file() else None
    wait=dict(schema="american_c3_native_direct_wait_and_drain_v1",parent_id=spec["parent_id"],
        native_child_pid=proc.pid,process_group=proc.pid,returncode=returncode,
        actual_direct_child_wait=True,process_group_ESRCH=group_esrch,no_adopted_live_writer=not _children(),
        owned_identities_drained=drained,owned_identity_drain_proof=ownership.evidence(),
        adopted_descendant_waits=adopted,adopted_wait_records_may_be_truncated=(len(adopted)>=1024),
        child_input_ref=child_ref,child_completion_ref=completion_ref,
        log_ref=log_ref,guard_failure_type=type(guard_error).__name__ if guard_error else None,
        interruption_signal=interruption,all_logs_buffered_until_terminal=True)
    wait_ref=writer.save(name+"_actual_wait.json",wait)
    from terminal_bounds_v1 import verify
    caps=metadata(spec["caps_ref"])
    writer.save(name+"_terminal_bounds.json",verify(directory,tmp,caps))
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
    mode=spec["phase"]
    receipt,completion_ref,wait_ref=_run_child(dict(name=mode,stage=mode),record=record,spec=spec,cfg=cfg,ctx=ctx,lease=lease)
    require(receipt["stage"]==mode and receipt["parent_id"]==spec["parent_id"],"Actual same-parent MLM-only child")
    result=dict(status="completed_C3_native_full_MLM_v1",execution_amendment_ref=amendment.authority_ref(),actual_producer_ref=amendment.ref(SOURCE),
        parent_id=spec["parent_id"],preparation_ref=spec["refs"]["preparation"],runtime_ref=spec["refs"]["runtime"],
        author_source_manifest_ref=AUTHOR_MANIFEST_REF,child_completion_ref=completion_ref,actual_wait_ref=wait_ref,
        terminal_bounds_ref=ref(writer.path/(mode+"_terminal_bounds.json")),query_ancestry_truth_opened=False,
        model_reused_from_other_parent=False,author_Table1_exact_reproduction=False,contrastive_executed=False)
    if mode=="mlm":
        require(receipt["actual_optimizer_updates"]==224775 and receipt["completed_epochs"]==5
            and receipt["initializer_checkpoint_ref"] is None,"Full fresh same-ARG MLM completion")
        authentication=metadata(receipt["corpus_authentication_ref"])
        expected={"mlm/train":prep["native_dataset_refs"]["mlm/train"],"mlm/val":prep["native_dataset_refs"]["mlm/val"],"vocab":prep["vocab_ref"]}
        require(authentication["status"]=="authenticated_full_MLM_inputs"
            and authentication["preparation_ref"]==spec["refs"]["preparation"]
            and {k:v["ref"] for k,v in authentication["inputs"].items()}==expected
            and all(v["sha256_verified"] and v["stable_read_verified"] for v in authentication["inputs"].values()),
            "Actual exact full-MLM input byte authentication evidence")
        result.update(checkpoint_ref=receipt["checkpoint_ref"],actual_optimizer_updates=receipt["actual_optimizer_updates"],completed_epochs=receipt["completed_epochs"],
            trainer_timestamp=receipt["trainer_timestamp"],trainer_optimizer_counters=receipt["trainer_optimizer_counters"],
            checkpoint_optimizer_counters=receipt["checkpoint_optimizer_counters"],observed_counters_verified=receipt["observed_counters_verified"],
            corpus_authentication_ref=receipt["corpus_authentication_ref"])
    return writer.save("completed.json",result)
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

"""One explicit H050 CPU EVALUATE caller and original owned-process monitor.

Import is standard-library only. No generic scientific dispatcher, admission,
GPU lease, numerical qualification or scientific work occurs on import.
"""
from pathlib import Path
import argparse, ast, datetime, hashlib, importlib, importlib.util, json, os
import resource, selectors, signal, subprocess, sys, time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H050_CPU_evaluation_successor_source_20261016_v1"
SOURCE=HERE+"/runtime_v1.py"
METADATA=HERE+"/metadata_v1.py"
METADATA_SHA="d2a82daecfb35a93b249acf0cdb1ba4cc417bde6f89d94bda6e0b02c84d7d57f"
RSS=64*2**30
DEADLINE=86400
AUX=16*2**20
LOG_CAP=2*2**20
_GUARD=None
_META=None
_TIMER=None
_RAN=False

def need(ok,message):
    if not ok:raise ValueError(message)

def meta():
    global _META
    if _META is None:
        p=ROOT/METADATA
        code=p.read_bytes()
        need(p.resolve()==p and len(code)<=2**20 and hashlib.sha256(code).hexdigest()==METADATA_SHA,"Exact bounded CPU metadata source before execution")
        spec=importlib.util.spec_from_file_location("_H050_CPU_specific_metadata",p)
        _META=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_META
        exec(compile(code,str(p),"exec"),_META.__dict__)
    return _META

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project-only path")
    return p
def ref(value):return meta().ref(local(value).relative_to(ROOT).as_posix())
def metadata(record):return meta().read(record)
def immutable_bytes(record,cap):return meta().raw(record,cap)
def source_load(record,label):return meta().load(record,label)
def cpu():
    need(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Project cwd and explicit empty CUDA visibility")
def monitor_helpers():
    r=meta().FIXED["monitor"];immutable_bytes(r,2**20)
    return source_load(r,"_H050_CPU_original_owned_monitor")
def put(p,value,cap=65536):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(data)<=cap,"Compact precharged root metadata")
    fd=os.open(local(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    return ref(p)

def arm_timer(started,spec):
    global _TIMER
    end=datetime.datetime.fromisoformat(spec["expires_at_utc"]).timestamp()
    remaining=min(DEADLINE-(time.monotonic()-started),end-time.time())
    need(remaining>0,"Actual CPU execution authority remains")
    old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    need(old_timer==(0.0,0.0),"One fresh process-wide wall timer")
    _TIMER=dict(started=started,absolute=end,expired=False,old_handler=old_handler)
    def timeout(signum,frame):
        _TIMER["expired"]=True
        raise TimeoutError("H050 CPU24h/absolute execution deadline")
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,remaining)

def check_deadline():
    need(_TIMER is not None and not _TIMER["expired"]
        and time.monotonic()-_TIMER["started"]<DEADLINE and time.time()<_TIMER["absolute"],
        "Original finite CPU deadline includes final writes/checks")
    need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RSS,"Own process peak RSS within64GiB")
    cpu()
    t=sys.modules.get("torch")
    if t is not None:need(not t.cuda.is_initialized(),"CPU evaluation must never initialize CUDA")

def restore_timer():
    if _TIMER is not None:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,_TIMER["old_handler"])

def configured_guard():
    global _GUARD
    if _GUARD is None:
        f=meta().FIXED
        for r in f.values():immutable_bytes(r,2**20)
        renewal=source_load(f["renewal"],"_H050_CPU_original_marker_renewal")
        renewal.install_renewal_v1(f["renewal_certificate"])
        m=source_load(f["guard"],"_H050_CPU_original980_guard")
        m.configure_v2(f["repair"],policy_ref=f["policy"])
        memo=source_load(f["memo"],"_H050_CPU_original_transaction_memo2")
        memo.install(m.budget.v4,expected_source_path=local(f["v4"]["path"]),expected_source_sha256=f["v4"]["sha256"])
        need(m.MODE=="prospective_980gb_v1","Original prospective980guard")
        _GUARD=m
    return _GUARD

def configure_environment(threads,aux=None):
    os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads),
        NUMEXPR_NUM_THREADS=str(threads))
    if aux is not None:
        os.environ.update(TMPDIR=str(aux),TMP=str(aux),TEMP=str(aux),XDG_CACHE_HOME=str(aux),
            TORCH_EXTENSIONS_DIR=str(aux/"torch_extensions"),TORCHINDUCTOR_CACHE_DIR=str(aux/"torchinductor"))
    os.environ.pop("ARGGNN_GPU_LEASE",None);os.environ.pop("ARGGNN_GPU_LEASE_FD",None)
    os.environ.pop("ARGGNN_PHYSICAL_GPU",None)
    os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(meta().POLICY,separators=(",",":"))
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]

def load_input(args):
    cpu();need(args.input and args.expected_sha256,"Exact externally supplied root input")
    record=dict(path=local(args.input).relative_to(ROOT).as_posix(),sha256=args.expected_sha256)
    spec=metadata(record);storage=metadata(spec["refs"]["storage"])
    cfg=meta().validate_spec(spec,storage)
    out,aux=local(spec["output_directory"]),local(spec["worker_auxiliary_scope"])
    need(out!=aux and not out.is_relative_to(aux) and not aux.is_relative_to(out)
        and not out.exists(),"Disjoint fresh scientific output/auxiliary, no replay")
    # Authenticate frozen science source bytes before numerical packages/payloads.
    for p,h in spec["source_sha256"].items():immutable_bytes(dict(path=p,sha256=h),2**20)
    for r in meta().required_bindings(cfg,spec):immutable_bytes(r,32*2**20)
    return record,spec,cfg,storage

class Reader:
    """Same immutable pinned-reader discipline, finite exact payload authorities."""
    root=ROOT
    def __init__(self,payloads):
        self.allowed={v["ref"]["path"]:(v["ref"],v["max_file_bytes"]) for v in payloads["rows"]}
        need(len(self.allowed)==48,"Exactly all24 checkpoints,8cache,8cut,8VAL truths")
        self.evidence={}
    def read(self,record,**kwargs):
        if Path(record["path"]).suffix in (".json",".py",".sh"):
            cap=32*2**20
        else:
            row=self.allowed.get(record["path"])
            need(row is not None and row[0]==record,"Exact selected H050 EVAL scientific payload only")
            cap=row[1]
        data=immutable_bytes(record,cap)
        self.evidence[record["path"]]=dict(sha256=record["sha256"],bytes=len(data))
        return data
    def decode_json(self,record):
        need(Path(record["path"]).suffix==".json","Named immutable JSON only")
        return json.loads(self.read(record))

def cpu_context(record,spec,cfg,storage,m):
    """Explicit CPU context. Original producer science is imported unmodified."""
    check_deadline()
    import torch,torch_geometric,setproctitle
    need(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0",
        "Same original pinned Torch/PyG versions")
    torch.set_num_threads(spec["CPU_threads"]);torch.set_num_interop_threads(1)
    need(torch.get_default_dtype()==torch.float32 and not torch.cuda.is_initialized(),"Ordinary CPU FP32, no CUDA initialization")
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    need(Path("/proc/self/cmdline").read_bytes().split(b"\0")[0].decode().strip()=="AI Auto Research Pipeline Test",
        "Actual clearly named CPU process")
    payloads=metadata(spec["CPU_payload_plan_ref"]);reader=Reader(payloads)
    # Import exact original producer package, retaining all scientific code objects.
    producer=importlib.import_module("research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.producer_v1")
    common=importlib.import_module("research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1.bridge_common_v1")
    original_context=importlib.import_module("research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1.context_factory_v1")
    need(producer.run_evaluate.__code__.co_filename==str(ROOT/meta().PRODUCER)
        and producer.load_checkpoints.__code__.co_filename==str(ROOT/meta().PRODUCER),
        "Original compiled scientific functions, no replacement or monkey patch")
    producer.validate_protocol(cfg);producer.bind_registration(cfg,reader);producer.h45.science()
    original=reader.decode_json(cfg["original_H038_input_ref"])
    generation=reader.decode_json(spec["refs"]["generation_config"]);registration=reader.decode_json(spec["refs"]["registration"])
    need(registration["config_ref"]==spec["refs"]["generation_config"]
        and registration["parent_registry"]==generation["parent_registry"],"Original immutable simulation registration/generation")
    preparation=common.framework.completed_marker_preparation_v1(reader,spec)
    counts=reader.decode_json(cfg["count_manifest_ref"])
    need(all(counts["parents"][p]["observation_entry_ref"]==spec["observation_entry_refs"][p]
        for p in producer.TRAIN+producer.VAL),"Original40 actual count identities")
    plan=common.count_plan_v1(counts,"evaluate",preparation,cfg)
    common.validate_plan_caps_v1(plan,cfg,storage["reserved_new_bytes"])
    cut_preparation=common.authenticate_actual_cut_preparation_v1(reader,spec,cfg)
    need(cfg["cut_preparation_completion_ref"]["path"]==cfg["expected_future_preparation_output"]+"/completed.json",
        "Same genuinely completed corrected v5 recipes")
    targets=reader.decode_json(spec["refs"]["target_bindings"]);originals=reader.decode_json(original["refs"]["target_bindings"])
    bindings=targets["parents"]
    need(set(bindings)==set(producer.VAL) and targets["phase"]=="evaluate"
        and targets["outer_included"] is False and targets["TRAIN_included"] is False and targets["VAL_included"] is True
        and bindings=={p:originals["parents"][p] for p in producer.VAL},"Same eight original VAL truth metadata, no TRAIN/outer targets")
    for pid,binding in bindings.items():
        need(binding["truth_npz"]["path"]==generation["output_root"]+"/"+pid+"/truth/full_truth_intervals.npz",
            "Original exact same-parent truth metadata path")
        provenance=reader.decode_json(binding["generation_receipt"])
        need(provenance["parent_id"]==pid and provenance["files"]["truth/full_truth_intervals.npz"]["sha256"]==binding["truth_npz"]["sha256"],
            "Same simulated-parent truth provenance, no payload read here")
    def guard():
        check_deadline()
        need(torch.get_default_dtype()==torch.float32 and not torch.cuda.is_initialized(),"FP32 CPU runtime remains unchanged")
    # Extract the original two prospective preparation callbacks without changing
    # any AST node. Only their bound operational guard is the CPU monitor.
    tree=ast.parse(immutable_bytes(ref(meta().CONTEXT),2**20))
    outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="build_context_v1")
    callbacks=[n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name in ("before_episode","before_cut")]
    need(len(callbacks)==2,"Exactly original two preparation callbacks")
    ns=dict(original_context.__dict__)
    ns.update(guard=guard,cfg=cfg,counts=counts,preparation=preparation)
    exec(compile(ast.Module(body=callbacks,type_ignores=[]),"<unchanged_H050_original_CPU_preparation_callbacks>","exec"),ns)
    output=m.PilotOutputV9(spec["output_directory"],storage,reader)
    ctx=dict(root_authorizer="root_agent",admitted=True,hypothesis_id="H050",arm="joint_four_arm",
        phase="evaluate",cfg=cfg,reader=reader,output=output,config_ref=spec["refs"]["config"],
        source_manifest_ref=cfg["source_manifest_ref"],registry=generation["parent_registry"],
        entry_refs=spec["observation_entry_refs"],observation_cache_ref=spec["refs"]["observation_cache"],
        observation_source_refs=spec["observation_cache_source_refs"],preparation=preparation,
        preparation_completion_ref=spec["preparation_completion_ref"],preparation_actual_wait_ref=spec["preparation_actual_wait_ref"],
        device=torch.device("cpu"),guard=guard,physical_gpu=None,bindings=bindings,cut_preparation=cut_preparation,
        before_episode_build=ns["before_episode"],before_cut_decode=ns["before_cut"],
        cut_preparation_completion_ref=spec["cut_preparation_completion_ref"],
        cut_preparation_actual_wait_ref=spec["cut_preparation_actual_wait_ref"],train_completion_ref=spec["train_completion_ref"])
    output.save("started.json",dict(schema="H050_CPU_EVAL_started_v1",input_ref=record,
        CPU_runtime_ref=ref(SOURCE),original_producer_ref=ref(meta().PRODUCER),device="cpu",physical_gpu=None,
        dtype="float32",CPU_threads=spec["CPU_threads"],output_plan=plan,checkpoints=24,VAL_products=192,
        same_four_arms_six_checkpoints_eight_VAL=True,optimization_performed=False,
        shared_train_terminal_certificate_ref=spec["train_actual_wait_ref"],
        CPU_FP32_qualification_ref=spec["CPU_FP32_qualification_ref"],GPU_pool_retirement_required=False))
    return producer,ctx

def worker(args):
    need(args.monitor_parent_pid==os.getppid() and args.monitor_read_fd is not None,"Actual outer CPU monitor parent")
    with os.fdopen(args.monitor_read_fd,"rb") as stream:ticket=stream.read(65)
    need(ticket==(args.expected_sha256+"\n").encode(),"Owned PID/RSS seeded before releasing child")
    def stop(signum,frame):raise RuntimeError("Original outer CPU monitor requested termination")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic();record,spec,cfg,storage=load_input(args)
    configure_environment(spec["CPU_threads"],local(spec["worker_auxiliary_scope"]))
    arm_timer(started,spec);m=configured_guard()
    m.validate_storage_v9(spec,cfg,storage,source_ref=ref(SOURCE),config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
    producer,ctx=cpu_context(record,spec,cfg,storage,m)
    try:
        # EXACT original evaluation. Its own PhaseReader denies truth until the
        # all192 original RAM-matrix verify() succeeds. No subset or optimization.
        result=producer.run_evaluate(ctx,record)
    except Exception as error:
        ctx["output"].save("failed.json",dict(hypothesis_id="H050",phase="evaluate",device="cpu",
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),outer_truth_read=False,automatic_replay=False))
        raise
    check_deadline()
    need(result["schema"]=="H050_four_arm_VAL_completion_v1" and result["all192_RAM_seals_before_any_VAL_truth"] is True
        and result["optimization_performed"] is False and result["outer_truth_read"] is False,
        "Actual unchanged complete H050 scientific evaluation")
    ctx["output"].save("root_CPU_runtime_receipt.json",dict(schema="H050_CPU_EVAL_worker_receipt_v1",
        input_ref=record,original_scientific_completion_ref=ref(local(spec["output_directory"])/"completed.json"),
        CPU_runtime_ref=ref(SOURCE),device="cpu",physical_gpu=None,dtype="float32",optimizer_updates=0,
        original_science_unmodified=True,all192_original_seals_before_truth=True,CUDA_initialized=False,
        peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_seconds=time.monotonic()-started))
    m.budget_check_v9(storage,boundary=True)
    check_deadline()
    print(json.dumps(dict(scientific_completion_ref=ref(local(spec["output_directory"])/"completed.json"))),flush=True)
    return 0

def launch(args):
    launch_started=time.monotonic();record,spec,cfg,storage=load_input(args)
    configure_environment(spec["CPU_threads"])
    arm_timer(launch_started,spec);b=monitor_helpers();m=configured_guard()
    m.validate_storage_v9(spec,cfg,storage,source_ref=ref(SOURCE),config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
    aux=local(spec["worker_auxiliary_scope"])
    need(not aux.exists() and not local(spec["output_directory"]).exists(),"Single launch, fresh output and auxiliary")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=AUX);aux.mkdir()
    runtime=metadata(spec["refs"]["runtime"]);env=os.environ.copy()
    env.update(TMPDIR=str(aux),TMP=str(aux),TEMP=str(aux),XDG_CACHE_HOME=str(aux),
        TORCH_EXTENSIONS_DIR=str(aux/"torch_extensions"),TORCHINDUCTOR_CACHE_DIR=str(aux/"torchinductor"))
    rfd,wfd=os.pipe()
    argv=[runtime["executable"],"-B",str(ROOT/SOURCE),"--worker",
        "--input",record["path"],"--expected-sha256",record["sha256"],"--monitor-parent-pid",str(os.getpid()),"--monitor-read-fd",str(rfd)]
    proc=None;selector=None;owned={};groups={};peak=races=discarded=0;log=bytearray();error=None
    started=utc();clock=time.monotonic();term_at=None;killed=False;rc=None;minimal_ref=None;seed_ref=None;released=False
    def retain(chunk):
        nonlocal discarded
        keep=min(len(chunk),LOG_CAP-len(log));log.extend(chunk[:keep]);discarded+=len(chunk)-keep
    try:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(rfd,))
        row=b.process_info(proc.pid)
        need(row is not None and row["pgid"]==proc.pid and row["session"]==proc.pid,"Seed actual owned PID before child release")
        owned[proc.pid]=row["start_ticks"];groups[proc.pid]={proc.pid:row["start_ticks"]}
        initial,misses=b.observe(owned,groups);peak=initial;races+=misses
        need(initial<=RSS,"Initial aggregate worker+monitor RSS before child compute")
        seed_ref=put(aux/"launch_monitor_seed.json",dict(schema="H050_CPU_precompute_owned_monitor_seed_v1",worker_pid=proc.pid,
            worker_start_ticks=row["start_ticks"],worker_pgid=row["pgid"],worker_session=row["session"],supervisor_pid=os.getpid(),
            initial_aggregate_RSS_bytes=initial,actual_owned_roster_seeded=True,child_compute_released=False,input_ref=record,checked_utc=utc()))
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        os.write(wfd,(record["sha256"]+"\n").encode());released=True
        os.close(wfd);wfd=None;os.close(rfd);rfd=None
        while proc.poll() is None:
            now=time.monotonic()
            try:
                rss,misses=b.observe(owned,groups);peak=max(peak,rss);races+=misses
                need(rss<=RSS and now-clock<=DEADLINE,"Aggregate64GiB RSS soft bound or24h deadline exceeded")
                check_deadline()
            except BaseException as exc:
                if error is None:error=dict(type=type(exc).__name__,message=str(exc)[:2048])
            if error is not None and term_at is None:
                b.signal_owned(groups,signal.SIGTERM);term_at=now
            if term_at is not None and now-term_at>=30 and not killed:
                b.signal_owned(groups,signal.SIGKILL);killed=True
            for key,_ in selector.select(.25):
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
    except BaseException as exc:
        error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:b.signal_owned(groups,signal.SIGTERM)
    finally:
        for fd in (rfd,wfd):
            if fd is not None:
                try:os.close(fd)
                except OSError as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:
            try:rc=proc.wait(timeout=30 if error else 5)
            except subprocess.TimeoutExpired:
                b.signal_owned(groups,signal.SIGKILL);rc=proc.wait()
            minimal_ref=put(aux/"actual_wait_minimal.json",dict(schema="H050_CPU_actual_direct_wait_minimal_v1",actual_direct_wait=True,
                returncode=rc,pid=proc.pid,worker_pgid=proc.pid,input_ref=record,
                started_utc=started,waited_utc=utc(),elapsed_seconds=time.monotonic()-clock),cap=8192)
    if proc is None:
        put(aux/"spawn_failed.json",dict(actual_direct_wait=False,error=error));raise RuntimeError("Popen failed, no actual wait")
    live=b.absent_groups(groups)
    if live:
        b.signal_owned(groups,signal.SIGKILL);until=time.monotonic()+10
        while live and time.monotonic()<until:time.sleep(.1);live=b.absent_groups(groups)
    if selector is not None:
        for _ in range(16):
            ready=selector.select(0)
            if not ready:break
            for key,_ in ready:
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
        selector.close()
    proc.stdout.close();need(not live,"Owned groups unresolved, minimal wait retained, no final success claim")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=len(log)+65536)
    fd=os.open(aux/"worker_stdout.txt",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(log);f.flush();os.fsync(f.fileno())
    terminal=put(aux/"outer_wait.json",dict(schema="H050_CPU_actual_outer_monitor_wait_v1",actual_wait_ref=minimal_ref,
        input_ref=record,returncode=rc,error=error,argv=argv,launch_monitor_seed_ref=seed_ref,child_compute_released=released,
        observed_owned_groups_ESRCH=True,observed_owned_groups=sorted(groups),observed_process_identities=owned,
        observed_peak_aggregate_RSS_bytes=peak,RSS_limit_bytes=RSS,RSS_is_polling_soft_bound_not_kernel_quota=True,
        aggregate_includes_outer_monitor=True,poll_seconds=.25,proc_exit_or_task_read_races=races,
        unobserved_between_poll_processes_possible=True,wall_deadline_seconds=DEADLINE,termination_grace_seconds=30,
        elapsed_seconds=time.monotonic()-clock,retained_stdout_ref=ref(aux/"worker_stdout.txt"),discarded_stdout_bytes=discarded))
    m.budget_check_v9(storage,boundary=True);check_deadline()
    need(rc==0 and error is None,"Actual H050 CPU worker failed, preserve partials/direct wait, no fallback")
    return dict(status="completed_H050_CPU_EVAL_worker_wait0_root_terminal_still_required",
        scientific_completion_ref=ref(local(spec["output_directory"])/"completed.json"),
        actual_wait_ref=minimal_ref,outer_wait_ref=terminal,automatic_completed_scope_credit=False)

def main(argv=None):
    global _RAN
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument("--launch",action="store_true");g.add_argument("--worker",action="store_true")
    p.add_argument("--input");p.add_argument("--expected-sha256")
    p.add_argument("--monitor-parent-pid",type=int);p.add_argument("--monitor-read-fd",type=int)
    args=p.parse_args(argv)
    if not(args.launch or args.worker):
        print(json.dumps(dict(status="source_only_inactive",hypothesis_id="H050",device="cpu",optimization_steps=0)));return 0
    need(not _RAN and sys.platform.startswith("linux") and not any(
        n in sys.modules for n in ("torch","torch_geometric","tensorflow","jax","cupy")),"One fresh Linux CPU-only process")
    _RAN=True
    os.environ["CUDA_VISIBLE_DEVICES"]="";sys.dont_write_bytecode=True;cpu()
    try:
        result=worker(args) if args.worker else launch(args)
        check_deadline()
    finally:restore_timer()
    if not args.worker:print(json.dumps(result,sort_keys=True),flush=True)
    return result if args.worker else 0

if __name__=="__main__":raise SystemExit(main())

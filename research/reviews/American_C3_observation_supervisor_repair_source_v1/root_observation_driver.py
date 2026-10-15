"""Root-only C3 full30Mb supervisor recovery; unchanged scientific worker and unary policy.

Read-only modes never configure a guard. Root activates/credits/reserves.
Full inference reuses successful v5 short qualification; all v5 failed outputs remain preserved.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, errno, hashlib, importlib.util, json, os, selectors, signal, stat, subprocess, sys, time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_observation_supervisor_repair_source_v1"
HERE=ROOT/REL
SOURCE=REL+"/root_observation_driver.py"
V3="research/reviews/American_C3_observation_root_execution_20261015_v3"
OVERLAY="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py"
OVERLAY_SHA="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef"
WORKER="src/arggnn/american_c3_observe_v3.py"
CLOSURE=["src/arggnn/american_c3_common_v1.py",WORKER,"src/arggnn/american_c3_simulate_v1.py"]
PYTHON=ROOT/".runtime/envs/biological_tsinfer_v1/bin/python"
STAGES=("full30mb",)
OUTPUTS={"full30mb":"experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1"}
PACKETS={s:HERE/("admitted_"+s+"_repair_v1") for s in STAGES}
PREPROCESSING_POLICY="direct_tskit_simplify_all_samples_keep_unary_false_no_site_filter_no_trimming_no_gap_split_v1"
LOG_CAP=16777216
WORK_CAP=134217728
METADATA_ALLOWANCE=1048576
MAX_JSON=65536
sys.path[:0]=[str(ROOT),str(ROOT/"src")]


def require(ok,message):
    if not ok:raise ValueError(message)


def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(".." not in p.parts and p.resolve()==p,"Canonical nonsymlink project path required")
    p.relative_to(ROOT)
    return p


def ref(value):
    p=local(value);digest=hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda:stream.read(8388608),b""):digest.update(block)
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=digest.hexdigest())


def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"}
            and ref(record["path"])==record,"Exact SHA-bound metadata required")
    data=local(record["path"]).read_bytes()
    require(len(data)<=MAX_JSON,"Overlay operation/evidence metadata must be at most64KiB")
    return json.loads(data)


def save(folder,name,value,guard=None,storage=None):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    require(len(raw)<=MAX_JSON,"Compact root metadata must fit64KiB bound")
    if guard is not None:guard.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw)+8192)
    path=folder/name
    with path.open("xb") as stream:stream.write(raw)
    return ref(path)


def frozen():
    closure_ref=ref(REL+"/frozen_refs.json");value=read(closure_ref)
    for record in value["refs"].values():
        require(ref(record["path"])==record,"Frozen integration source/config/runtime changed: "+record["path"])
    return closure_ref,value["refs"]


def helper(pins):
    spec=importlib.util.spec_from_file_location("C3_pinned_v3_readonly_helpers",local(pins["v3_driver"]["path"]))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def qualification_v5(pins):
    # Reuse exact prior unary-policy short qualification without rerunning science.
    original=pins["v5_driver"]
    require(ref(original["path"])==original,"Frozen successful v5 qualification supervisor changed")
    spec=importlib.util.spec_from_file_location("C3_successful_v5_qualification",local(original["path"]))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.qualification_v5(pins)


def policy_metadata_v5(policy_ref,pins):
    if policy_ref is None:return False,"Root must supply exact active overlay policy ref"
    require(policy_ref==pins["policy"],"Exact root-activated immutable policy ref required")
    p=read(policy_ref)
    require(p["source_ref"]==pins["overlay"] and p["marker_repair_authority_ref"]==pins["repair_authority"]
            and p["implementation_ref"]==pins["budget"] and p["launcher_ref"]==pins["launcher"]
            and p["eligible_physical_GPU_ids"]==[4,5]
            and p["new_GPU_occupancy_on_2_3_authorized"] is False,
            "Exact reviewed overlay/base/repair/GPU45 policy metadata required")
    return p.get("active") is True, "Root overlay policy is active" if p.get("active") is True else "Root overlay policy is inactive"


def preflight_v5(stage,policy_ref=None):
    require(stage in STAGES,"Exact short/full unary-policy stage required")
    frozen_ref,pins=frozen();v3=helper(pins)
    cfg=read(pins["config_"+stage]);inputs=read(pins["inputs_"+stage])
    require(cfg["execution_enabled"] is False and type(cfg.get("num_threads")) is int
            and cfg["num_threads"]==8 and cfg["observation_ref"]==inputs["observation_ref"]
            and cfg["unary_preprocessing_policy"]==PREPROCESSING_POLICY,
            "Exact explicit unary-policy/eight-thread inactive config required")
    m=read(cfg["observation_ref"]);expected=inputs["expected_identity"]
    length,sites,h,counts,parent=(100000,244,64,{"AFR":8,"EUR":8,"ASIA":8,"ADMIX":40},"american_c3_qualification_000") if stage=="qualification100kb" else (30000000,226185,1600,{"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000},"american_c3_train_000")
    require(m["parent_id"]==parent and m["sequence_length_bp"]==length
            and m["counts_haploid"]==counts and m["class_order"]==["EUR","EAS","AFR"]
            and m["sample_unit"]=="haploid" and m["query_ancestry_truth_in_inputs"] is False
            and expected["sites"]==sites,"Exact original observed subject/site/length identity required")
    require(set(inputs["consumed_data_refs"])=={"genotypes_ref","positions_ref","alleles_ref","roster_ref"},
            "Only observed inputs allowed")
    for key,record in inputs["consumed_data_refs"].items():
        require(m[key]==record and ref(record["path"])==record,"Original observed input changed: "+key)
    g=v3.npy_header(m["genotypes_ref"]["path"]);p=v3.npy_header(m["positions_ref"]["path"])
    require(g["shape"]==(sites,h) and g["descr"]=="|i1" and not g["fortran_order"]
            and p["shape"]==(sites,) and p["descr"]=="<f8" and not p["fortran_order"],
            "Exact original array metadata required; no array loads")
    serialized=cfg["max_sample_store_bytes"]+2*(cfg["max_tables_bytes"]+16777216)+cfg["max_node_map_bytes"]+16777216
    require(serialized+WORK_CAP+LOG_CAP+METADATA_ALLOWANCE<=cfg["output_cap_bytes"],
            "Raw/dated tree, store, node-map and work/log plan must fit cap")
    require(not local(OUTPUTS[stage]).exists(),"Fresh exclusive versioned stage output required")
    qready,qreason,evidence=(True,"This is the root's new explicit-policy short qualification",{})
    if stage=="full30mb":qready,qreason,evidence=qualification_v5(pins)
    pready,preason=policy_metadata_v5(policy_ref,pins)
    return dict(status="read_only_explicit_unary_policy_preflight",stage=stage,ready_for_root_admission=qready and pready,
        qualification_ready=qready,qualification_status=qreason,qualification_evidence_refs=evidence,
        policy_metadata_ready=pready,policy_status=preason,policy_ref=policy_ref,
        frozen_refs_ref=frozen_ref,config_ref=pins["config_"+stage],runtime_ref=pins["runtime"],
        input_closure_ref=pins["inputs_"+stage],source_closure_refs=[ref(path) for path in CLOSURE],
        source_ref=pins["worker"],driver_ref=ref(SOURCE),output_directory=OUTPUTS[stage],
        output_cap_bytes=cfg["output_cap_bytes"],serialized_count_bound_bytes=serialized,
        planned_total_bytes=serialized+WORK_CAP+LOG_CAP+METADATA_ALLOWANCE,
        tsinfer_num_threads=8,unary_preprocessing_policy=PREPROCESSING_POLICY,
        guard_configured=False,admission_created=False,scientific_arrays_loaded=False,
        oracle_or_private_truth_payload_opened=False)

def overlay_guard_v5(policy_ref,pins):
    require(policy_ref is not None and pins["overlay"]==dict(path=OVERLAY,sha256=OVERLAY_SHA),
            "Exact independently reviewed overlay/policy required")
    spec=importlib.util.spec_from_file_location("C3_reviewed_completed_credit_overlay_v3",local(OVERLAY))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.configure_v2(pins["repair_authority"],policy_ref=policy_ref)
    return module


def admit_v5(stage,policy_ref):
    require(Path.cwd().resolve()==ROOT,"Root must run from repository")
    plan=preflight_v5(stage,policy_ref);require(plan["ready_for_root_admission"],plan["qualification_status"]+"; "+plan["policy_status"])
    _,pins=frozen();report=runtime_report_v5()
    packet,output=PACKETS[stage],OUTPUTS[stage]
    require(report["runtime"]==read(pins["runtime"])
            and report["source_closure_refs"]==plan["source_closure_refs"],"Existing inference runtime/worker changed")
    require(not packet.exists(),"Fresh root stage packet required")
    packet.mkdir()
    cfg=dict(read(pins["config_"+stage]),execution_enabled=True)
    config_ref=save(packet,"config.json",cfg)
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=22)).isoformat()
    job="American-C3-observe-"+stage+"-supervisor-repair-20261015-v1"
    m=overlay_guard_v5(policy_ref,pins)
    with m.budget.v4.mutex(ROOT):
        state=m._state_locked(pins["budget"]["sha256"],pins["launcher"]["sha256"],fresh=True)
    require(len(state)==7,"Reviewed overlay seven-part state API required")
    view,bindings=state[2],state[3]
    consumers=[*plan["source_closure_refs"],config_ref,pins["runtime"],ref(SOURCE),pins["v3_driver"],
        plan["frozen_refs_ref"],plan["input_closure_ref"],pins["config_"+stage],pins["v5_driver"],policy_ref,
        *bindings.values(),*plan["qualification_evidence_refs"].values()]
    consumers=list({(r["path"],r["sha256"]):r for r in consumers}.values())
    require(len(consumers)<=64 and all(Path(r["path"]).suffix in (".py",".sh",".json") for r in consumers),
            "Existing64 direct source/JSON consumer limit required")
    request=dict(key="H041/"+job,output=str(ROOT/output),auxiliary_scopes=[str(packet/"worker_aux")],
                 reserved_bytes=cfg["output_cap_bytes"],consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry)
    authority_ref=save(packet,"reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=pins["budget"]["sha256"],
                             expected_launcher_sha256=pins["launcher"]["sha256"])
    save(packet,"reserve_receipt.json",receipt)
    require(receipt["kind"]=="concurrent_completed_scope_reserve_receipt_v1","Reviewed overlay reserve receipt required")
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=job,output_directory=output,reserved_new_bytes=cfg["output_cap_bytes"],
        reservation=receipt["value"],expires_at_utc=expiry,completed_overlay_policy_ref=policy_ref,
        initial_ledger_pins=receipt["initial_ledger_pins"],
        budget_components=dict(implementation=pins["budget"],launcher=pins["launcher"],consumer_adapter=pins["overlay"]))
    spec=dict(root_authorizer="root_agent",hypothesis_id="H041",job_id=job,execution_authorized=True,
        GPU_authorized=False,source_ref=pins["worker"],source_closure_refs=plan["source_closure_refs"],
        config_ref=config_ref,runtime_ref=pins["runtime"],storage_adapter_ref=pins["overlay"],
        marker_repair_authority_ref=pins["repair_authority"],completed_overlay_policy_ref=policy_ref,
        qualification_evidence_refs=plan["qualification_evidence_refs"],input_closure_ref=pins["inputs_"+stage],
        driver_ref=ref(SOURCE),storage=storage,output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry)
    admission_ref=save(packet,"admission.json",spec,m,storage)
    result=dict(stage=stage,admission_ref=admission_ref,driver_ref=ref(SOURCE),policy_ref=policy_ref,
                source_closure_refs=plan["source_closure_refs"],cap_bytes=cfg["output_cap_bytes"],execution_started=False)
    save(packet,"admitted.json",result,m,storage)
    return result


def live_work_observation_v1(path):
    """Single-pass owned scratch observation, not a stable inventory or OS quota."""
    total=count=0;stack=[path]
    while stack:
        folder=stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    count+=1;require(count<=50000,"Bounded private work directory required")
                    try:info=entry.stat(follow_symlinks=False)
                    except FileNotFoundError:continue
                    require(not stat.S_ISLNK(info.st_mode),"No private-work symlinks")
                    if stat.S_ISDIR(info.st_mode):stack.append(Path(entry.path))
                    else:
                        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1,"Regular private work files only")
                        total+=max(info.st_size,info.st_blocks*512)
        except FileNotFoundError:continue
    return total

def owned_group_absent_v1(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return True
    except OSError as error:
        if error.errno==errno.ESRCH:return True
        raise
    return False

def wait_owned_worker_v1(proc,terminate):
    """Always call direct wait; clean only the new session created for this worker."""
    errors=[]
    def send(signum):
        try:os.killpg(proc.pid,signum)
        except ProcessLookupError:pass
        except BaseException as error:errors.append(type(error).__name__+": "+str(error)[:1024])
    if terminate:send(signal.SIGTERM)
    while True:
        try:
            rc=proc.wait(timeout=30 if terminate else None)
            break
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL);terminate=True
        except BaseException as error:
            errors.append(type(error).__name__+": "+str(error)[:1024])
            send(signal.SIGTERM);terminate=True
    # A waited parent can leave descendants. No boundary scan/log disk write until ESRCH.
    absent=False;started=time.monotonic()
    try:
        while not (absent:=owned_group_absent_v1(proc.pid)):
            send(signal.SIGTERM if time.monotonic()-started<30 else signal.SIGKILL)
            if time.monotonic()-started>60:break
            time.sleep(.1)
    except BaseException as error:errors.append(type(error).__name__+": "+str(error)[:1024])
    return rc,absent,errors[-16:]

def launch_v5(stage,policy_ref):
    plan=preflight_v5(stage,policy_ref);require(plan["ready_for_root_admission"],plan["qualification_status"]+"; "+plan["policy_status"])
    _,pins=frozen();v3=helper(pins)
    packet,output=PACKETS[stage],OUTPUTS[stage]
    admitted=read(ref(packet/"admitted.json"))
    require(admitted["driver_ref"]==ref(SOURCE) and admitted["policy_ref"]==policy_ref,"Exact root driver/policy admission")
    spec=read(admitted["admission_ref"]);storage=spec["storage"]
    require(spec["driver_ref"]==ref(SOURCE) and spec["storage_adapter_ref"]==pins["overlay"]
            and storage["completed_overlay_policy_ref"]==policy_ref
            and spec["qualification_evidence_refs"]==plan["qualification_evidence_refs"]
            and spec["source_closure_refs"]==plan["source_closure_refs"]
            and read(spec["config_ref"])==dict(read(pins["config_"+stage]),execution_enabled=True)
            and spec["runtime_ref"]==pins["runtime"],"Exact admitted unchanged worker/config/runtime/qualification")
    m=overlay_guard_v5(policy_ref,pins)
    # Keep the reserve receipt's verified prefix. Unrelated later reserve events
    # are valid descendants; no refresh/rewrite of the old-base pins occurs.
    receipt=read(ref(packet/"reserve_receipt.json"))
    require(storage["initial_ledger_pins"]==receipt["initial_ledger_pins"]
            and storage["reservation"]==receipt["value"],"Exact own reserve-event prefix required")
    m.validate_storage_v9(spec,read(spec["config_ref"]),storage,source_ref=pins["worker"],
                          config_ref=spec["config_ref"],runtime_ref=pins["runtime"])
    worker_aux=packet/"worker_aux"
    require(storage["reservation"]["auxiliary_scopes"]==[str(worker_aux)] and not worker_aux.exists(),
            "Fresh exact reserved worker_aux required")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=32768)
    worker_aux.mkdir();work=worker_aux/"work";(work/"tmp").mkdir(parents=True);(work/"numba_cache").mkdir()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(ROOT/"src"),PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="1",
        TMPDIR=str(work/"tmp"),TMP=str(work/"tmp"),TEMP=str(work/"tmp"),
        NUMBA_CACHE_DIR=str(work/"numba_cache"),XDG_CACHE_HOME=str(work/"cache"),
        ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(policy_ref,sort_keys=True,separators=(",",":")))
    command=[str(PYTHON),"-B","-m","arggnn.american_c3_observe_v3","--execute",
        "--admission",admitted["admission_ref"]["path"],"--expected-admission-sha256",admitted["admission_ref"]["sha256"]]
    log_path=worker_aux/"launch.log";m.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+8192)
    started=time.monotonic();failure=None;proc=None;selector=None
    captured=bytearray();received=0;terminate_at=None;last_guard=last_work=started
    supervisor_errors=[];post_wait_errors=[];live_work_peak=0
    # No disk log or charged-scope metadata writes while worker boundaries may scan.
    # Compact launch metadata is in the root packet's charged metadata allowance.
    try:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,start_new_session=True)
        save(packet,"launch.json",dict(stage=stage,pid=proc.pid,process_group=proc.pid,command=command,
             driver_ref=ref(SOURCE),admission_ref=admitted["admission_ref"],policy_ref=policy_ref,
             worker_auxiliary_scope=str(worker_aux),actual_wait_pending=True,
             stdout_policy="bounded_RAM_until_direct_wait_and_owned_group_ESRCH"))
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        while selector.get_map() or proc.poll() is None:
            now=time.monotonic()
            if failure is None:
                if now-started>75600:failure="Registered full-stage operational walltime exceeded"
                elif datetime.now(timezone.utc)>=datetime.fromisoformat(spec["expires_at_utc"]):
                    failure="Root admission expired"
                if now-last_guard>=15:
                    # Validate exact reservation/policy/journal, without scanning mutable product scopes.
                    m.budget_check_v9(storage,boundary=False);last_guard=now
                if now-last_work>=5:
                    live_work_peak=max(live_work_peak,live_work_observation_v1(work));last_work=now
                    if live_work_peak>WORK_CAP:failure="Private temporary/cache work allowance exceeded"
                if failure is not None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    terminate_at=now
            if terminate_at is not None and proc.poll() is None and now-terminate_at>30:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            events=selector.select(timeout=1)
            for key,_ in events:
                block=os.read(key.fileobj.fileno(),65536)
                if not block:selector.unregister(key.fileobj);continue
                received+=len(block)
                remaining=LOG_CAP-len(captured);captured.extend(block[:remaining])
                if failure is None and len(block)>remaining:
                    failure="Retained launch log RAM cap exceeded"
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    terminate_at=time.monotonic()
            if not events and proc.poll() is not None:break
    except BaseException as error:
        failure=failure or "Supervisor exception: "+type(error).__name__
        supervisor_errors.append(dict(type=type(error).__name__,message=str(error)[:4096]))
    finally:
        if proc is not None:
            rc,group_absent,cleanup_errors=wait_owned_worker_v1(proc,failure is not None)
        else:rc=None;group_absent=False;cleanup_errors=[]
        if selector is not None:
            try:selector.close()
            except BaseException as error:post_wait_errors.append(type(error).__name__+": "+str(error)[:1024])
        if proc is not None and proc.stdout is not None:
            try:proc.stdout.close()
            except BaseException as error:post_wait_errors.append(type(error).__name__+": "+str(error)[:1024])
    if proc is None:
        record=dict(stage=stage,status="launch_failed_no_worker_created",actual_direct_worker_wait=False,
            supervisor_errors=supervisor_errors,driver_ref=ref(SOURCE),admission_ref=admitted["admission_ref"])
        save(packet,"launch_failure.json",record)
        raise RuntimeError("Worker launch failed before a child existed; no actual wait claimed")
    if not group_absent:failure=failure or "Owned worker group not proven ESRCH; no product boundary or log write allowed"
    log_ref=None;completed_ref=None;work_bytes=None
    if group_absent:
        try:
            # All worker/group writers are terminal. Strict original boundary guard is safe here.
            m.budget_check_v9(storage,boundary=True,required_write_bytes=len(captured)+8192)
            with log_path.open("xb") as log:
                log.write(captured);log.flush();os.fsync(log.fileno())
            log_ref=ref(log_path)
            work_bytes=v3.work_bytes(work)
            if work_bytes>WORK_CAP:failure=failure or "Terminal private work allowance exceeded"
            m.budget_check_v9(storage,boundary=True)
        except BaseException as error:
            failure=failure or "Post-wait storage/log audit failed"
            post_wait_errors.append(type(error).__name__+": "+str(error)[:4096])
        try:
            completed_path=local(output)/"completed.json"
            if completed_path.exists():completed_ref=ref(completed_path)
        except BaseException as error:
            failure=failure or "Post-wait completion metadata binding failed"
            post_wait_errors.append(type(error).__name__+": "+str(error)[:4096])
    if rc==0 and completed_ref is None:failure=failure or "Exit0 without inferred completion"
    if cleanup_errors:failure=failure or "Owned worker wait/cleanup required follow-up"
    wait=dict(stage=stage,pid=proc.pid,process_group=proc.pid,returncode=rc,operational_failure=failure,
        actual_direct_worker_wait=True,owned_worker_group_ESRCH=group_absent,
        supervisor_errors=supervisor_errors,cleanup_errors=cleanup_errors,post_wait_errors=post_wait_errors,
        driver_ref=admitted["driver_ref"],admission_ref=admitted["admission_ref"],policy_ref=policy_ref,
        completed_ref=completed_ref,log_ref=log_ref,captured_log_bytes=len(captured),
        stdout_bytes_received=received,captured_log_sha256=hashlib.sha256(captured).hexdigest(),
        retained_log_bytes=len(captured) if log_ref else 0,log_truncated=received>len(captured),
        worker_auxiliary_scope=str(worker_aux),elapsed_seconds=time.monotonic()-started,
        private_work_bytes=work_bytes,live_private_work_peak_observation_bytes=live_work_peak,
        live_work_observation_is_stable_inventory=False,scientific_worker_ref=pins["worker"],
        failed_v5_artifacts_preserved=True,finished_utc=datetime.now(timezone.utc).isoformat())
    # Root packet evidence is compact charged metadata, outside mutable product scopes.
    # Do not let a guard failure erase an actual wait already consumed from the OS.
    try:print(json.dumps(dict(actual_worker_wait_evidence=wait),sort_keys=True),flush=True)
    except BaseException as error:
        wait["evidence_stdout_error"]=type(error).__name__+": "+str(error)[:2048]
    try:save(packet,"actual_wait.json",wait)
    except BaseException as error:
        fallback=dict(wait=wait,evidence_write_error=type(error).__name__+": "+str(error)[:2048],
                      root_followup_required=True)
        save(packet,"actual_wait_fallback.json",fallback)
    return wait


def runtime_report_v5():
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(ROOT/"src"),PYTHONDONTWRITEBYTECODE="1")
    cmd=[str(PYTHON),"-B","-m","arggnn.american_c3_observe_v3","--runtime-report"]
    result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60,check=True)
    require(len(result.stdout)<=MAX_JSON,"Bounded metadata runtime report required")
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight",choices=STAGES);mode.add_argument("--admit",choices=STAGES)
    mode.add_argument("--launch",choices=STAGES);mode.add_argument("--source-report",action="store_true")
    parser.add_argument("--policy");parser.add_argument("--policy-sha")
    args=parser.parse_args()
    require(bool(args.policy)==bool(args.policy_sha),"Supply both exact --policy and --policy-sha")
    policy_ref=dict(path=local(args.policy).relative_to(ROOT).as_posix(),sha256=args.policy_sha) if args.policy else None
    if args.admit:result=admit_v5(args.admit,policy_ref)
    elif args.launch:result=launch_v5(args.launch,policy_ref)
    elif args.preflight:result=preflight_v5(args.preflight,policy_ref)
    else:
        closure_ref,pins=frozen()
        result=dict(status="source_only_full30mb_supervisor_repair_inactive",driver_ref=ref(SOURCE),
            frozen_refs_ref=closure_ref,overlay_ref=pins["overlay"],stages=list(STAGES),policy_ref=policy_ref,
            full_requires_current_short_success=True,unary_preprocessing_policy=PREPROCESSING_POLICY,
            root_admission_unperformed=True,guard_configured=False,inference_executed=False)
    print(json.dumps(result,indent=2,sort_keys=True))
    return (result["returncode"] or 1) if args.launch and (result["returncode"]!=0 or result["operational_failure"] is not None) else 0

if __name__=="__main__":raise SystemExit(main())

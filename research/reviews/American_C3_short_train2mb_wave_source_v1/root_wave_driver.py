"""Root-only independent2Mb American C3 TRAIN preparation, at most4 parents.

Import/default report reads source metadata only. No bulk admission, activation,
credit, scientific execution or GPU activity occurs without explicit root mode.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import argparse,errno,fcntl,hashlib,importlib.util,json,os,selectors,subprocess,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_short_train2mb_wave_source_v1"
HERE=ROOT/REL
SOURCE=REL+"/root_wave_driver.py"
CONDITION="C3_independent20x2Mb_TRAIN_1600haps_inferred_unary_v1"
STAGES=("simulate","infer")
UNARY_POLICY="direct_tskit_simplify_all_samples_keep_unary_false_no_site_filter_no_trimming_no_gap_split_v1"
PARENT_IDS=tuple("american_c3_short_train_%03d"%i for i in range(20))
LOG_CAP=16777216
WORK_CAP=134217728
sys.path[:0]=[str(ROOT),str(ROOT/"src")]


def require(ok,message):
    if not ok:raise ValueError(message)


def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(".." not in p.parts and p.resolve()==p,"Canonical nonsymlink project path required")
    p.relative_to(ROOT);return p


def ref(value):
    p=local(value);h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8388608),b""):h.update(b)
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=h.hexdigest())


def read(record):
    require(ref(record["path"])==record,"Exact metadata SHA required")
    raw=local(record["path"]).read_bytes();require(len(raw)<=65536,"Compact root metadata required")
    return json.loads(raw)


def save(folder,name,value,guard=None,storage=None):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Root operation/receipt metadata must fit64KiB")
    if guard is not None:guard.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw)+8192)
    p=folder/name
    with p.open("xb") as f:f.write(raw)
    return ref(p)


def frozen():
    r=ref(REL+"/frozen_refs.json");value=read(r)
    for item in value["refs"].values():require(ref(item["path"])==item,"Frozen source metadata changed")
    return r,value["refs"]


def helper(pins):
    s=importlib.util.spec_from_file_location("C3_wave_pinned_v5_helper",local(pins["v5_driver"]["path"]))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def registry_parent(parent,pins):
    require(parent in PARENT_IDS,"Only registered independent2Mb TRAIN IDs allowed")
    registry=read(pins["registry"])
    require(registry["scientific_condition_id"]==CONDITION and registry["new_independent_TRAIN_count"]==20,
            "Exact new prospective training condition required")
    return next(row for row in registry["parents"] if row["parent_id"]==parent)


def qualifier(pins):
    v5=helper(pins);old=read(pins["v5_frozen"])["refs"]
    ready,reason,evidence=v5.qualification_v5(old)
    require(ready,"Single actual v5 unary-policy qualifier required: "+reason)
    return evidence


def stage_identity(parent,stage,row):
    require(stage in STAGES,"Exact simulate/infer stage required")
    output=row["simulation_output_directory" if stage=="simulate" else "inference_output_directory"]
    job="American-C3-short2mb-"+parent+"-"+stage+"-v1"
    return HERE/"admitted"/parent/stage,output,job,"H041/"+job


def successful_stage(parent,stage,pins):
    row=registry_parent(parent,pins);packet,output,job,key=stage_identity(parent,stage,row)
    wait_ref=ref(packet/"actual_wait.json");wait=read(wait_ref)
    require(wait["parent_id"]==parent and wait["stage"]==stage and wait["returncode"]==0
            and wait["actual_direct_worker_wait"] is True and wait["worker_process_group_drained"] is True
            and wait["operational_failure"] is None,"Actual exit0 and complete worker-group drain required")
    completed_ref=wait["completed_ref"];require(completed_ref["path"]==output+"/completed.json","Exact producer completion path required")
    complete=read(completed_ref)
    if stage=="simulate":
        require(complete["status"]=="completed_C3_simulation_reconstruction" and complete["parent_id"]==parent
                and complete["sample_unit"]=="haploid" and 0<complete["sites"]<=100000,
                "Exact complete new simulation required")
    else:
        require(complete["status"]=="completed_current_C3_inference" and complete["tsinfer_num_threads"]==8
                and complete["unary_preprocessing_policy"]==UNARY_POLICY,
                "Actual complete declared unary-policy inference required")
    observation=read(complete["observation_ref"])
    require(observation["parent_id"]==parent and observation["split"]=="train"
            and observation["sequence_length_bp"]==2000000 and observation["sample_unit"]=="haploid"
            and observation["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
            and observation["class_order"]==["EUR","EAS","AFR"]
            and observation["query_ancestry_truth_in_inputs"] is False,
            "Exact original1600-haplotype2Mb TRAIN observation identity required")
    if stage=="infer":
        require(observation["sites_and_sample_identities_verified"] is True
                and observation["migrations_and_internal_population_labels_removed"] is True,
                "All original sites/sample identities must be verified")
    return dict(wait_ref=wait_ref,completion_ref=completed_ref,complete=complete,observation=observation)


@contextmanager
def root_lock(name,nonblocking=False):
    folder=HERE/"scheduler";folder.mkdir(exist_ok=True)
    fd=os.open(folder/name,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0))
        yield fd
    finally:
        os.close(fd)


@contextmanager
def parent_slot(parent):
    folder=HERE/"scheduler";folder.mkdir(exist_ok=True)
    for index in range(4):
        fd=os.open(folder/("parent_slot_%d.lock"%index),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:os.close(fd);continue
        try:yield dict(fd=fd,index=index,parent_id=parent);return
        finally:os.close(fd)
    raise ValueError("All four wave parent slots are occupied; root must wait for own workers")


def slot_valid(slot,parent):
    require(slot["parent_id"]==parent and slot["index"] in range(4),"Exact owned parent slot required")
    named=HERE/"scheduler"/("parent_slot_%d.lock"%slot["index"])
    a,b=os.fstat(slot["fd"]),named.stat()
    require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),"Inherited live parent-slot identity required")


def activate_config(parent,stage,row,pins):
    cfg_ref=row["simulation_config_ref" if stage=="simulate" else "inference_template_ref"]
    cfg=dict(read(cfg_ref),execution_enabled=True)
    dependencies={}
    if stage=="infer":
        prior=successful_stage(parent,"simulate",pins)
        cfg["observation_ref"]=prior["complete"]["observation_ref"]
        dependencies=dict(simulation_wait_ref=prior["wait_ref"],simulation_completion_ref=prior["completion_ref"])
    return cfg_ref,cfg,dependencies


def admit_stage_v1(parent,stage,policy_ref):
    require(Path.cwd().resolve()==ROOT,"Root must run in repository")
    frozen_ref,pins=frozen();row=registry_parent(parent,pins)
    require(policy_ref==pins["policy"],"Exact activated root overlay policy ref required")
    evidence=qualifier(pins);cfg_template,cfg,dependencies=activate_config(parent,stage,row,pins)
    packet,output,job,key=stage_identity(parent,stage,row)
    require(not packet.exists() and not local(output).exists(),"Fresh per-stage packet/output required")
    # The wave serializes its own operations. Root coordinates external credit/admission operations.
    with root_lock("admission.lock"):
        packet.mkdir(parents=True);local(output).parent.mkdir(parents=True,exist_ok=True)
        config_ref=save(packet,"config.json",cfg)
        worker=pins["simulation_worker" if stage=="simulate" else "inference_worker"]
        runtime=pins["simulation_runtime" if stage=="simulate" else "inference_runtime"]
        closure=[pins["common"],worker]
        if stage=="infer":closure.append(pins["simulation_worker"])
        v5=helper(pins);guard=v5.overlay_guard_v5(policy_ref,pins)
        with guard.budget.v4.mutex(ROOT):
            state=guard._state_locked(pins["budget"]["sha256"],pins["launcher"]["sha256"],fresh=True)
        require(len(state)==7,"Exact current credit overlay state API required")
        # No fifth open wave-parent allocation, even through separate root CLI calls.
        open_parents=set()
        for registered in read(pins["registry"])["parents"]:
            for which in STAGES:
                own_key=stage_identity(registered["parent_id"],which,registered)[3]
                if own_key in state[0] and own_key not in state[1]:open_parents.add(registered["parent_id"])
        require(parent in open_parents or len(open_parents)<4,"Root must credit/drain completed parents before a fifth allocation")
        consumers=[*closure,config_ref,runtime,ref(SOURCE),pins["v5_driver"],pins["v3_driver"],
            frozen_ref,pins["registry"],cfg_template,policy_ref,*state[3].values(),*evidence.values(),*dependencies.values()]
        consumers=list({(r["path"],r["sha256"]):r for r in consumers}.values())
        require(len(consumers)<=64 and all(Path(r["path"]).suffix in (".py",".sh",".json") for r in consumers),
                "Existing64 source/JSON direct consumer limit required")
        expiry=(datetime.now(timezone.utc)+timedelta(hours=22)).isoformat()
        request=dict(key=key,output=str(ROOT/output),auxiliary_scopes=[str(packet/"worker_aux")],
            reserved_bytes=cfg["output_cap_bytes"],consumer_bindings=consumers)
        auth=guard.operation_draft_v1("reserve",request,expected_before=guard._pins(state[2]),bindings=state[3])
        auth.update(active=True,expires_at_utc=expiry);authority_ref=save(packet,"reserve_authority.json",auth)
        receipt=guard.apply_reserve_v1(authority_ref,expected_source_sha256=pins["budget"]["sha256"],
            expected_launcher_sha256=pins["launcher"]["sha256"])
        save(packet,"reserve_receipt.json",receipt)
        storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
            accounting_mode=guard.MODE,job_id=job,output_directory=output,reserved_new_bytes=cfg["output_cap_bytes"],
            reservation=receipt["value"],expires_at_utc=expiry,completed_overlay_policy_ref=policy_ref,
            initial_ledger_pins=receipt["initial_ledger_pins"],
            budget_components=dict(implementation=pins["budget"],launcher=pins["launcher"],consumer_adapter=pins["overlay"]))
        spec=dict(root_authorizer="root_agent",hypothesis_id="H041",scientific_condition_id=CONDITION,
            job_id=job,execution_authorized=True,GPU_authorized=False,source_ref=worker,source_closure_refs=closure,
            config_ref=config_ref,runtime_ref=runtime,storage_adapter_ref=pins["overlay"],
            marker_repair_authority_ref=pins["repair_authority"],completed_overlay_policy_ref=policy_ref,
            qualification_evidence_refs=evidence,stage_dependencies=dependencies,driver_ref=ref(SOURCE),
            parent_id=parent,stage=stage,storage=storage,output_directory=output,
            created_at_utc=datetime.now(timezone.utc).isoformat(),expires_at_utc=expiry)
        admission_ref=save(packet,"admission.json",spec,guard,storage)
        admitted=dict(parent_id=parent,stage=stage,admission_ref=admission_ref,driver_ref=ref(SOURCE),policy_ref=policy_ref)
        save(packet,"admitted.json",admitted,guard,storage)
        return admitted


def group_drained(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return True
    return False


def _launch(parent,stage,policy_ref,slot):
    slot_valid(slot,parent)
    _,pins=frozen();row=registry_parent(parent,pins);qualifier(pins)
    packet,output,job,key=stage_identity(parent,stage,row)
    admitted=read(ref(packet/"admitted.json"))
    require(admitted["driver_ref"]==ref(SOURCE) and admitted["policy_ref"]==policy_ref==pins["policy"],"Exact source/policy admission")
    spec=read(admitted["admission_ref"]);storage=spec["storage"]
    template,cfg,dependencies=activate_config(parent,stage,row,pins)
    require(read(spec["config_ref"])==cfg and spec["stage_dependencies"]==dependencies,
            "Exact scientific config and prior actual simulation required")
    v5=helper(pins);guard=v5.overlay_guard_v5(policy_ref,pins)
    receipt=read(ref(packet/"reserve_receipt.json"))
    require(storage["reservation"]==receipt["value"] and storage["initial_ledger_pins"]==receipt["initial_ledger_pins"],
            "Own reserve-event prefix must remain unchanged")
    guard.validate_storage_v9(spec,cfg,storage,source_ref=spec["source_ref"],config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    auxiliary=packet/"worker_aux";require(not auxiliary.exists(),"Fresh worker auxiliary scope required")
    guard.budget_check_v9(storage,boundary=True,required_write_bytes=32768)
    auxiliary.mkdir();work=auxiliary/"work";(work/"tmp").mkdir(parents=True);(work/"numba_cache").mkdir()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(ROOT/"src"),PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="1",
        TMPDIR=str(work/"tmp"),TMP=str(work/"tmp"),TEMP=str(work/"tmp"),NUMBA_CACHE_DIR=str(work/"numba_cache"),
        XDG_CACHE_HOME=str(work/"cache"),ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(policy_ref,sort_keys=True))
    executable=ROOT/(".runtime/env/bin/python" if stage=="simulate" else ".runtime/envs/biological_tsinfer_v1/bin/python")
    module="arggnn.american_c3_simulate_v1" if stage=="simulate" else "arggnn.american_c3_observe_v3"
    command=[str(executable),"-B","-m",module,"--execute","--admission",admitted["admission_ref"]["path"],
             "--expected-admission-sha256",admitted["admission_ref"]["sha256"]]
    v3=v5.helper(read(pins["v5_frozen"])["refs"])
    log_path=auxiliary/"launch.log";started=time.monotonic();failure=None;retained=0;terminate_at=None
    guard.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+8192)
    with log_path.open("xb") as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(slot["fd"],))
        roster=[dict(pid=os.getpid(),role="root_parent_stage_supervisor"),
                dict(pid=proc.pid,role="direct_scientific_worker",process_group=proc.pid)]
        launch=dict(parent_id=parent,stage=stage,pid=proc.pid,worker_pid=proc.pid,worker_process_group=proc.pid,
            supervisor_pid=os.getpid(),owned_process_roster=roster,command=command,admission_ref=admitted["admission_ref"],
            parent_slot=slot["index"],slot_fd_inherited=True,actual_wait_pending=True,no_helper_launcher=True)
        save(packet,"launch.json",launch,guard,storage)
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        try:
            while selector.get_map():
                if failure is None:
                    if time.monotonic()-started>(3600 if stage=="simulate" else 10800):failure="Stage operational walltime exceeded"
                    elif v3.work_bytes(work)>WORK_CAP:failure="Private work allowance exceeded"
                    elif datetime.now(timezone.utc)>=datetime.fromisoformat(spec["expires_at_utc"]):failure="Root admission expired"
                    if failure is not None:proc.terminate();terminate_at=time.monotonic()
                if terminate_at is not None and proc.poll() is None and time.monotonic()-terminate_at>30:proc.kill()
                for ready,_ in selector.select(timeout=1):
                    data=os.read(ready.fileobj.fileno(),65536)
                    if not data:selector.unregister(ready.fileobj);continue
                    if failure is None and retained+len(data)>LOG_CAP:
                        failure="Retained log cap exceeded";proc.terminate();terminate_at=time.monotonic()
                    if failure is None:
                        guard.budget_check_v9(storage,boundary=True,required_write_bytes=len(data)+8192)
                        log.write(data);log.flush();retained+=len(data)
            rc=proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
        finally:selector.close();proc.stdout.close()
    drained=group_drained(proc.pid)
    if not drained:failure="Root-owned worker process group not fully drained; no completion credit"
    path=local(output)/"completed.json";completed_ref=ref(path) if path.exists() else None
    if rc==0 and completed_ref is None:failure="Exit0 without producer completion"
    wait=dict(parent_id=parent,stage=stage,pid=proc.pid,worker_pid=proc.pid,worker_process_group=proc.pid,
        supervisor_pid=os.getpid(),owned_process_roster=roster,returncode=rc,operational_failure=failure,
        worker_process_group_drained=drained,group_drain_check="killpg0_ESRCH_after_direct_wait",
        actual_direct_worker_wait=True,admission_ref=admitted["admission_ref"],driver_ref=ref(SOURCE),
        completed_ref=completed_ref,log_ref=ref(log_path),elapsed_seconds=time.monotonic()-started,
        parent_slot=slot["index"],retained_log_bytes=retained,finished_utc=datetime.now(timezone.utc).isoformat(),
        root_must_observe_supervisor_exit_before_credit=True)
    try:save(packet,"actual_wait.json",wait,guard,storage)
    except Exception as exc:
        save(packet,"actual_wait_guard_failure.json",dict(wait=wait,error_type=type(exc).__name__,error=str(exc)));raise
    require(rc==0 and failure is None,"Stage failed; preserve all partial artifacts and stop this parent")
    successful_stage(parent,stage,pins)
    return wait


def launch_stage_v1(parent,stage,policy_ref):
    with parent_slot(parent) as slot:return _launch(parent,stage,policy_ref,slot)


def prepare_parent_v1(parent,policy_ref):
    _,pins=frozen();row=registry_parent(parent,pins)
    with parent_slot(parent) as slot:
        stages={}
        for stage in STAGES:
            packet,output,job,key=stage_identity(parent,stage,row)
            if (packet/"actual_wait.json").exists():successful_stage(parent,stage,pins)
            else:
                if not (packet/"admitted.json").exists():admit_stage_v1(parent,stage,policy_ref)
                _launch(parent,stage,policy_ref,slot)
            stages[stage]=successful_stage(parent,stage,pins)
        parent_packet=HERE/"admitted"/parent
        result=dict(status="completed_C3_2Mb_TRAIN_parent_preparation",scientific_condition_id=CONDITION,
            parent_id=parent,split="train",sequence_length_bp=2000000,
            inferred_model_observation_ref=stages["infer"]["complete"]["observation_ref"],
            external_TRAIN_truth_ref=stages["simulate"]["complete"]["private_truth_ref"],
            stages={s:{k:stages[s][k] for k in ("wait_ref","completion_ref")} for s in STAGES},
            truth_kept_out_of_model_inputs=True,GPU_used=False,model_training_performed=False,
            source_registry_ref=pins["registry"],root_supervisor_pid=os.getpid(),
            root_must_wait_for_supervisor_and_complete_owned_roster_before_credit=True)
        save(parent_packet,"parent_preparation_completed.json",result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument("--admit",choices=STAGES);mode.add_argument("--launch",choices=STAGES)
    mode.add_argument("--run-parents",nargs="+");mode.add_argument("--source-report",action="store_true")
    p.add_argument("--parent");p.add_argument("--policy");p.add_argument("--policy-sha");a=p.parse_args()
    require(bool(a.policy)==bool(a.policy_sha),"Supply exact policy path and SHA together")
    policy=dict(path=local(a.policy).relative_to(ROOT).as_posix(),sha256=a.policy_sha) if a.policy else None
    if a.admit or a.launch:
        require(a.parent and policy,"Root parent and policy required")
        result=admit_stage_v1(a.parent,a.admit,policy) if a.admit else launch_stage_v1(a.parent,a.launch,policy)
    elif a.run_parents:
        require(policy and 1<=len(a.run_parents)<=4 and len(set(a.run_parents))==len(a.run_parents)
                and set(a.run_parents)<=set(PARENT_IDS),"Explicit batch of1–4 exact independent parents required")
        with ThreadPoolExecutor(max_workers=len(a.run_parents)) as pool:
            futures=[pool.submit(prepare_parent_v1,parent,policy) for parent in a.run_parents]
            result=[future.result() for future in futures]
    else:
        closure,pins=frozen()
        result=dict(status="source_only_inactive_20x2Mb_TRAIN_wave",driver_ref=ref(SOURCE),frozen_refs_ref=closure,
            registry_ref=pins["registry"],parent_ids=list(PARENT_IDS),maximum_concurrent_wave_parents=4,
            no_bulk_admission=True,requires_single_successful_v5_qualification=True,
            VAL_TEST_generation_authorized=False,scientific_execution_performed=False)
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())

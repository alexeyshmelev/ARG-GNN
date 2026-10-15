"""Root-only full30Mb C3 inference through reviewed completed-credit overlay.

Read-only modes never configure a guard. Root activates/credits/reserves.
No short inference is launched here: exact successful v3 qualification is required.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, hashlib, importlib.util, json, os, selectors, subprocess, sys, time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_observation_root_execution_20261015_v4"
HERE=ROOT/REL
SOURCE=REL+"/root_full_observation_driver.py"
V3="research/reviews/American_C3_observation_root_execution_20261015_v3"
OVERLAY="research/reviews/concurrent_completed_scope_credits_v2/completed_scope_credits_v2.py"
OVERLAY_SHA="f1b5141247244acbf74c60f2b0c2f93048736e9d417b2f7b634b609c41fe4ec3"
WORKER="src/arggnn/american_c3_observe_v2.py"
CLOSURE=["src/arggnn/american_c3_common_v1.py",WORKER,"src/arggnn/american_c3_simulate_v1.py"]
PYTHON=ROOT/".runtime/envs/biological_tsinfer_v1/bin/python"
OUTPUT="experiments/results/American_C3_observed_full30mb_20261015_v4"
PACKET=HERE/"admitted_full30mb_v4"
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


def qualification_v4(pins):
    wait_path=ROOT/V3/"admitted_qualification100kb_v3/actual_wait.json"
    if not wait_path.exists():return False,"Exact v3 short inference direct wait not yet available",{}
    wait_ref=ref(wait_path);wait=read(wait_ref)
    require(wait["driver_ref"]==pins["v3_driver"] and wait["stage"]=="qualification100kb",
            "Qualification must use exact frozen v3 root driver")
    if wait["returncode"]!=0 or wait.get("operational_failure") is not None:
        return False,"Exact v3 short inference did not succeed; preserve failed artifacts",{"actual_wait_ref":wait_ref}
    require(wait["actual_direct_worker_wait"] is True and wait.get("completed_ref"),
            "Actual successful v3 worker wait and completion required")
    admission=read(wait["admission_ref"]);cfg=read(pins["qualification_config"])
    expected=dict(cfg,execution_enabled=True)
    require(admission["source_ref"]==pins["worker"]
            and admission["source_closure_refs"]==[ref(path) for path in CLOSURE]
            and admission["GPU_authorized"] is False and admission["execution_authorized"] is True
            and read(admission["config_ref"])==expected
            and read(admission["runtime_ref"])==read(pins["runtime"]),
            "Exact unchanged v3 observed inference capability/config/runtime required")
    qoutput="experiments/results/American_C3_observed_qualification100kb_20261015_v3"
    require(admission["output_directory"]==qoutput
            and wait["completed_ref"]["path"]==qoutput+"/completed.json",
            "Exact v3 qualification output identity required")
    complete=read(wait["completed_ref"])
    require(complete["status"]=="completed_current_C3_inference" and complete["tsinfer_num_threads"]==8
            and complete["observation_ref"]["path"]==qoutput+"/observed/manifest.json",
            "Actual completed eight-thread inferred observation required")
    observed=read(complete["observation_ref"]);original=read(cfg["observation_ref"])
    require(observed["schema"]=="american_c3_observed_v1"
            and observed["parent_id"]=="american_c3_qualification_000"
            and observed["sequence_length_bp"]==100000
            and observed["counts_haploid"]=={"AFR":8,"EUR":8,"ASIA":8,"ADMIX":40}
            and observed["class_order"]==["EUR","EAS","AFR"]
            and observed["sites_and_sample_identities_verified"] is True
            and observed["query_ancestry_truth_in_inputs"] is False
            and observed["migrations_and_internal_population_labels_removed"] is True
            and observed["source_observation_ref"]==cfg["observation_ref"]
            and observed["inference_runtime_settings"]["tsinfer_num_threads"]==8
            and observed["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
            "Complete unchanged v3 inferred identity required")
    for key in ("genotypes_ref","positions_ref","alleles_ref"):
        require(observed[key]==original[key],"Qualification original observed input identity changed")
    roster=read(observed["roster_ref"]);prior=read(original["roster_ref"])
    require(roster["parent_id"]==original["parent_id"] and roster["class_order"]==original["class_order"]
            and len(roster["samples"])==64
            and len(roster["sample_index_to_inferred_node"])==64,"Exact inferred64-sample roster required")
    for index,(a,b) in enumerate(zip(prior["samples"],roster["samples"])):
        require({k:v for k,v in a.items() if k!="observed_node_id"}==
                {k:v for k,v in b.items() if k!="observed_node_id"}
                and b["sample_index"]==index
                and b["observed_node_id"]==roster["sample_index_to_inferred_node"][index],
                "Exact original-column to inferred-node identity required")
    evidence=dict(actual_wait_ref=wait_ref,admission_ref=wait["admission_ref"],
                  completion_ref=wait["completed_ref"],observation_ref=complete["observation_ref"],
                  roster_ref=observed["roster_ref"])
    return True,"Exact successful v3 inferred short qualification verified",evidence


def policy_metadata_v4(policy_ref,pins):
    if policy_ref is None:return False,"Root must supply exact active overlay policy ref"
    p=read(policy_ref)
    require(p["source_ref"]==pins["overlay"] and p["marker_repair_authority_ref"]==pins["repair_authority"]
            and p["implementation_ref"]==pins["budget"] and p["launcher_ref"]==pins["launcher"]
            and p["eligible_physical_GPU_ids"]==[4,5]
            and p["new_GPU_occupancy_on_2_3_authorized"] is False,
            "Exact reviewed overlay/base/repair/GPU45 policy metadata required")
    return p.get("active") is True, "Root overlay policy is active" if p.get("active") is True else "Root overlay policy is inactive"


def preflight_v4(policy_ref=None):
    frozen_ref,pins=frozen();v3=helper(pins)
    cfg=read(pins["config"]);inputs=read(pins["inputs"])
    require(cfg["execution_enabled"] is False and type(cfg.get("num_threads")) is int
            and cfg["num_threads"]==8 and cfg["observation_ref"]==inputs["observation_ref"],
            "Unchanged v3 full inactive config with explicit8threads required")
    m=read(cfg["observation_ref"]);expected=inputs["expected_identity"]
    require(m["parent_id"]=="american_c3_train_000" and m["sequence_length_bp"]==30000000
            and m["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
            and m["class_order"]==["EUR","EAS","AFR"] and m["sample_unit"]=="haploid"
            and m["query_ancestry_truth_in_inputs"] is False
            and expected["sites"]==226185,"Exact original observed30Mb/1600-haplotype source required")
    require(set(inputs["consumed_data_refs"])=={"genotypes_ref","positions_ref","alleles_ref","roster_ref"},
            "Only observed inputs allowed")
    for key,record in inputs["consumed_data_refs"].items():
        require(m[key]==record and ref(record["path"])==record,"Original observed input changed: "+key)
    g=v3.npy_header(m["genotypes_ref"]["path"]);p=v3.npy_header(m["positions_ref"]["path"])
    require(g["shape"]==(226185,1600) and g["descr"]=="|i1" and not g["fortran_order"]
            and p["shape"]==(226185,) and p["descr"]=="<f8" and not p["fortran_order"],
            "Exact original array metadata required; no array loads")
    serialized=cfg["max_sample_store_bytes"]+2*(cfg["max_tables_bytes"]+16777216)+16777216
    require(serialized+WORK_CAP+LOG_CAP+METADATA_ALLOWANCE<=cfg["output_cap_bytes"],
            "Original serialization/work/log plan must fit unchanged cap")
    require(not local(OUTPUT).exists(),"Fresh exclusive full v4 output required")
    qready,qreason,evidence=qualification_v4(pins)
    pready,preason=policy_metadata_v4(policy_ref,pins)
    return dict(status="read_only_full30Mb_preflight",ready_for_root_admission=qready and pready,
        qualification_ready=qready,qualification_status=qreason,qualification_evidence_refs=evidence,
        policy_metadata_ready=pready,policy_status=preason,policy_ref=policy_ref,
        frozen_refs_ref=frozen_ref,config_ref=pins["config"],runtime_ref=pins["runtime"],
        input_closure_ref=pins["inputs"],source_closure_refs=[ref(path) for path in CLOSURE],
        source_ref=pins["worker"],driver_ref=ref(SOURCE),output_directory=OUTPUT,
        output_cap_bytes=cfg["output_cap_bytes"],serialized_count_bound_bytes=serialized,
        planned_total_bytes=serialized+WORK_CAP+LOG_CAP+METADATA_ALLOWANCE,
        tsinfer_num_threads=8,guard_configured=False,admission_created=False,
        scientific_arrays_loaded=False,oracle_or_private_truth_payload_opened=False)


def overlay_guard_v4(policy_ref,pins):
    require(policy_ref is not None and pins["overlay"]==dict(path=OVERLAY,sha256=OVERLAY_SHA),
            "Exact independently reviewed overlay/policy required")
    spec=importlib.util.spec_from_file_location("C3_reviewed_completed_credit_overlay_v2",local(OVERLAY))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.configure_v2(pins["repair_authority"],policy_ref=policy_ref)
    return module


def admit_v4(policy_ref):
    require(Path.cwd().resolve()==ROOT,"Root must run from repository")
    plan=preflight_v4(policy_ref);require(plan["ready_for_root_admission"],plan["qualification_status"]+"; "+plan["policy_status"])
    _,pins=frozen();report=helper(pins).runtime_report()
    require(report["runtime"]==read(pins["runtime"])
            and report["source_closure_refs"]==plan["source_closure_refs"],"Existing inference runtime/worker changed")
    require(not PACKET.exists(),"Fresh root full-only packet required")
    PACKET.mkdir()
    cfg=dict(read(pins["config"]),execution_enabled=True)
    config_ref=save(PACKET,"config.json",cfg)
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=22)).isoformat()
    job="American-C3-observe-full30mb-20261015-v4"
    m=overlay_guard_v4(policy_ref,pins)
    with m.budget.v4.mutex(ROOT):
        state=m._state_locked(pins["budget"]["sha256"],pins["launcher"]["sha256"],fresh=True)
    require(len(state)==7,"Reviewed overlay seven-part state API required")
    view,bindings=state[2],state[3]
    consumers=[*plan["source_closure_refs"],config_ref,pins["runtime"],ref(SOURCE),pins["v3_driver"],
        plan["frozen_refs_ref"],plan["input_closure_ref"],pins["config"],policy_ref,
        *bindings.values(),*plan["qualification_evidence_refs"].values()]
    consumers=list({(r["path"],r["sha256"]):r for r in consumers}.values())
    require(len(consumers)<=64 and all(Path(r["path"]).suffix in (".py",".sh",".json") for r in consumers),
            "Existing64 direct source/JSON consumer limit required")
    request=dict(key="H041/"+job,output=str(ROOT/OUTPUT),auxiliary_scopes=[str(PACKET/"worker_aux")],
                 reserved_bytes=cfg["output_cap_bytes"],consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry)
    authority_ref=save(PACKET,"reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=pins["budget"]["sha256"],
                             expected_launcher_sha256=pins["launcher"]["sha256"])
    save(PACKET,"reserve_receipt.json",receipt)
    require(receipt["kind"]=="concurrent_completed_scope_reserve_receipt_v1","Reviewed overlay reserve receipt required")
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=job,output_directory=OUTPUT,reserved_new_bytes=cfg["output_cap_bytes"],
        reservation=receipt["value"],expires_at_utc=expiry,completed_overlay_policy_ref=policy_ref,
        initial_ledger_pins=receipt["initial_ledger_pins"],
        budget_components=dict(implementation=pins["budget"],launcher=pins["launcher"],consumer_adapter=pins["overlay"]))
    spec=dict(root_authorizer="root_agent",hypothesis_id="H041",job_id=job,execution_authorized=True,
        GPU_authorized=False,source_ref=pins["worker"],source_closure_refs=plan["source_closure_refs"],
        config_ref=config_ref,runtime_ref=pins["runtime"],storage_adapter_ref=pins["overlay"],
        marker_repair_authority_ref=pins["repair_authority"],completed_overlay_policy_ref=policy_ref,
        qualification_evidence_refs=plan["qualification_evidence_refs"],input_closure_ref=pins["inputs"],
        driver_ref=ref(SOURCE),storage=storage,output_directory=OUTPUT,created_at_utc=now.isoformat(),expires_at_utc=expiry)
    admission_ref=save(PACKET,"admission.json",spec,m,storage)
    result=dict(stage="full30mb",admission_ref=admission_ref,driver_ref=ref(SOURCE),policy_ref=policy_ref,
                source_closure_refs=plan["source_closure_refs"],cap_bytes=cfg["output_cap_bytes"],execution_started=False)
    save(PACKET,"admitted.json",result,m,storage)
    return result


def launch_v4(policy_ref):
    plan=preflight_v4(policy_ref);require(plan["ready_for_root_admission"],plan["qualification_status"]+"; "+plan["policy_status"])
    _,pins=frozen();v3=helper(pins)
    admitted=read(ref(PACKET/"admitted.json"))
    require(admitted["driver_ref"]==ref(SOURCE) and admitted["policy_ref"]==policy_ref,"Exact root driver/policy admission")
    spec=read(admitted["admission_ref"]);storage=spec["storage"]
    require(spec["driver_ref"]==ref(SOURCE) and spec["storage_adapter_ref"]==pins["overlay"]
            and storage["completed_overlay_policy_ref"]==policy_ref
            and spec["qualification_evidence_refs"]==plan["qualification_evidence_refs"]
            and spec["source_closure_refs"]==plan["source_closure_refs"]
            and read(spec["config_ref"])==dict(read(pins["config"]),execution_enabled=True)
            and spec["runtime_ref"]==pins["runtime"],"Exact admitted unchanged worker/config/runtime/qualification")
    m=overlay_guard_v4(policy_ref,pins)
    # Keep the reserve receipt's verified prefix. Unrelated later reserve events
    # are valid descendants; no refresh/rewrite of the old-base pins occurs.
    receipt=read(ref(PACKET/"reserve_receipt.json"))
    require(storage["initial_ledger_pins"]==receipt["initial_ledger_pins"]
            and storage["reservation"]==receipt["value"],"Exact own reserve-event prefix required")
    m.validate_storage_v9(spec,read(spec["config_ref"]),storage,source_ref=pins["worker"],
                          config_ref=spec["config_ref"],runtime_ref=pins["runtime"])
    worker_aux=PACKET/"worker_aux"
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
    command=[str(PYTHON),"-B","-m","arggnn.american_c3_observe_v2","--execute",
        "--admission",admitted["admission_ref"]["path"],"--expected-admission-sha256",admitted["admission_ref"]["sha256"]]
    log_path=worker_aux/"launch.log";m.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+8192)
    started=time.monotonic();failure=None;retained=0;terminate_at=None
    with log_path.open("xb") as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        save(PACKET,"launch.json",dict(stage="full30mb",pid=proc.pid,command=command,driver_ref=ref(SOURCE),
             admission_ref=admitted["admission_ref"],policy_ref=policy_ref,worker_auxiliary_scope=str(worker_aux),
             actual_wait_pending=True),m,storage)
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        try:
            while selector.get_map():
                if failure is None:
                    if time.monotonic()-started>75600:failure="Original full operational walltime exceeded"
                    elif v3.work_bytes(work)>WORK_CAP:failure="Private temporary/cache work allowance exceeded"
                    elif datetime.now(timezone.utc)>=datetime.fromisoformat(spec["expires_at_utc"]):failure="Root admission expired"
                    if failure is not None:proc.terminate();terminate_at=time.monotonic()
                if terminate_at is not None and proc.poll() is None and time.monotonic()-terminate_at>30:proc.kill()
                for key,_ in selector.select(timeout=1):
                    block=os.read(key.fileobj.fileno(),65536)
                    if not block:selector.unregister(key.fileobj);continue
                    if failure is None and retained+len(block)>LOG_CAP:
                        failure="Retained launch log cap exceeded";proc.terminate();terminate_at=time.monotonic()
                    if failure is None:
                        m.budget_check_v9(storage,boundary=True,required_write_bytes=len(block)+8192)
                        log.write(block);log.flush();retained+=len(block)
            rc=proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
        finally:selector.close();proc.stdout.close()
    completed_path=local(OUTPUT)/"completed.json";completed_ref=ref(completed_path) if completed_path.exists() else None
    if rc==0 and completed_ref is None:failure="Exit0 without inferred completion"
    wait=dict(stage="full30mb",pid=proc.pid,returncode=rc,operational_failure=failure,actual_direct_worker_wait=True,
        driver_ref=ref(SOURCE),admission_ref=admitted["admission_ref"],policy_ref=policy_ref,completed_ref=completed_ref,
        log_ref=ref(log_path),retained_log_bytes=retained,worker_auxiliary_scope=str(worker_aux),
        elapsed_seconds=time.monotonic()-started,private_work_bytes=v3.work_bytes(work),
        finished_utc=datetime.now(timezone.utc).isoformat())
    try:save(PACKET,"actual_wait.json",wait,m,storage)
    except Exception as exc:
        save(PACKET,"actual_wait_guard_failure.json",dict(wait=wait,error_type=type(exc).__name__,
             error=str(exc),root_followup_required=True));raise
    return wait


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight",action="store_true");mode.add_argument("--admit",action="store_true")
    mode.add_argument("--launch",action="store_true");mode.add_argument("--source-report",action="store_true")
    parser.add_argument("--policy");parser.add_argument("--policy-sha")
    args=parser.parse_args()
    require(bool(args.policy)==bool(args.policy_sha),"Supply both exact --policy and --policy-sha")
    policy_ref=dict(path=local(args.policy).relative_to(ROOT).as_posix(),sha256=args.policy_sha) if args.policy else None
    if args.admit:result=admit_v4(policy_ref)
    elif args.launch:result=launch_v4(policy_ref)
    elif args.preflight:result=preflight_v4(policy_ref)
    else:
        closure_ref,pins=frozen()
        result=dict(status="source_only_full30Mb_inactive",driver_ref=ref(SOURCE),frozen_refs_ref=closure_ref,
            overlay_ref=pins["overlay"],stage="full30mb",policy_ref=policy_ref,requires_v3_successful_qualification=True,
            root_activation_credits_and_admission_unperformed=True,guard_configured=False,inference_executed=False)
    print(json.dumps(result,indent=2,sort_keys=True))
    return (result["returncode"] or 1) if args.launch and (result["returncode"]!=0 or result["operational_failure"] is not None) else 0


if __name__=="__main__":raise SystemExit(main())

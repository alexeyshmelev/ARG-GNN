"""Inactive operational successor. No guard, Torch or project import until explicit guarded execution."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,os,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1"
SOURCE=HERE+"/evaluation_contract_v1.py"
DRIVER=HERE+"/root_driver_v1.py"
ENTRY=CPU_MAIN=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
PARENT="American_C3_short20_TRAIN_original5VAL"
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
EPOCHS=(5,10,15,20,25,30)
PHASES=tuple("predict_val_%03d"%i for i in range(5))+("matrix","score")
GPU_PHASES=()
GPU_MODULE=None
STATUSES={p:"completed_C3_full30Mb_CPU_evaluation_"+p for p in PHASES}
TRAIN_JOB="C3-GNN-efficient-resume-GPU0-TRAIN-v1"
TRAIN_OUTPUT="experiments/results/C3_GNN_efficient_resume_GPU0_train_v1"
TRAIN_INPUT_REF=dict(path="research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/root_resume_actual_v1/input.json",
 sha256="b420377394b1246a55dd2ce2a731f523692ef557d8ea074d2fd3137231008a25")
TRAIN_ENVELOPE_REF=dict(path="research/reviews/C3_module_route_repair_root_20261016_v1/C3_RESUME_TRAIN/envelope.json",
 sha256="e8561d14f8e25094fca3eef30714e88854ca1d31827ddaeeabd0053bd5bcc49d")
MIGRATION_REF=dict(path="research/reviews/C3_actual_full_state_handover_root_20261016_v1/migration_v2.root_sealed_manifest_v1.json",
 sha256="7cbd2387ea536bf49adb7af93477c229c6473b7a2f9e4f681583e8d97b466830")
TRAIN_DRIVER_REF=dict(path="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/root_driver_v1.py",
 sha256="1b30a9e68f4c31e65492f26af28e1d22d45ec96211d301758e9ef5a9864a6536")
BASE_DRIVER_REF=dict(path="research/reviews/C3_GNN_VAL002_plan_H020v3_source_20261016_v2/root_driver_v1.py",
 sha256="c44b17b69201ef27c6b55c82ac126de5e43e8e1daf4f617818f9c1cebbc77047")
GUARD_REF=dict(path="research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py",
 sha256="fab6bb67bc1f4be989df9bb16e7a3a17d24652ca8238617ca060a56cb8c89d43")
POLICY_REF=dict(path="research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/activation_policy.root_active_v1.json",
 sha256="72ff346c43f96ba8430411cfe6b2250480a3aa66fa8e3b8c904a625a0399932a")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",
 sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
RUNTIME_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json",
 sha256="e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602")
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CAPS=dict(schema="C3_full30Mb_CPU_evaluation_caps_v1",reservation_cap_bytes=33554432,
 phase_write_budget_bytes=16777216,outer_auxiliary_cap_bytes=4194304,outer_log_cap_bytes=262144,
 temp_allowance_bytes=8388608,root_packet_overhead_cap_bytes=524288,
 max_cpu_rss_bytes=137438953472,max_cuda_reserved_bytes=0,operational_deadline_seconds=86400,
 CPU_threads=32,dataloader_workers=0,prediction_cells_per_parent=6)
_ACTIVE_SPEC=None

HERE='research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1'
SOURCE=HERE+"/acceptance_contract_v1.py"
DRIVER=HERE+"/root_driver_v1.py"
ENTRY=CPU_MAIN=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
FROZEN=HERE+"/frozen_science.v1.json"
CORE_SOURCE=HERE+"/saved_state_core_v1.py"
ACCEPT_PHASE="validate_saved600"
EVAL_PHASES=PHASES
PHASES=(ACCEPT_PHASE,)+EVAL_PHASES
STATUSES=dict(STATUSES,validate_saved600="completed_C3_saved600_artifact_validation")
GUARD_REF={'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py', 'sha256': '60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c'}
POLICY_REF={'path': 'research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417'}
OLD_CONTRACT_REF={'path': 'research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1/evaluation_contract_v1.py', 'sha256': '19c71454c8abc30886ae97814dd00be3abd76762451f427160ce282cbbbf2173'}
OLD_WORKER_REF={'path': 'research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1/worker_entry_v1.py', 'sha256': 'a0226a0dbde5b8c0f34fd1ad45167a98f7b625cc501bb53791fa7fc44b5ebdce'}
def require(ok,message):
    if not ok:raise ValueError(message)

def local(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(p.is_relative_to(ROOT) and ".." not in p.parts and p.resolve()==p,"Canonical contained project path")
    return p

def ref(path):
    p=local(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=33554432,
                        "Bounded regular metadata/source")
    require(p.suffix in (".py",".json",".sh"),"This operational reader never hashes scientific payloads")
    raw=p.read_bytes()
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest())

def read(record,cap=33554432):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,
            "Exact authenticated metadata/source reference")
    p=local(record["path"]);require(p.suffix==".json" and p.stat().st_size<=cap,"JSON metadata only")
    return json.loads(p.read_bytes())

def save(directory,name,value):
    require(Path(name).name==name and name.endswith(".json"),"Flat fresh administrative JSON")
    p=local(directory)/name;raw=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Bounded genuine administrative receipt")
    existing=sum(x.lstat().st_size+8192 for x in local(directory).iterdir() if x.is_file())
    require(existing+len(raw)+8192<=CAPS["root_packet_overhead_cap_bytes"],
            "ROOT packet metadata stays within current-child per-event metadata charge")
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return ref(p)

def load(record,name):
    require(ref(record["path"])==record,"Authenticate source before deferred import")
    spec=importlib.util.spec_from_file_location(name,local(record["path"]))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def active(value):
    require(value.get("root_authorizer")=="root_agent" and value.get("active",value.get("status")=="active") is True
            and value.get("execution_authorized") is True,"Explicit separate ROOT execution authority")
    end=datetime.fromisoformat(value["expires_at_utc"].replace("Z","+00:00"))
    require(end.tzinfo is not None and end>datetime.now(timezone.utc),"Unexpired actual ROOT authority")

def original_guard():
    g=load(GUARD_REF,"_C3_eval_exact_current_Census_guard")
    g.configure_with_renewal_v1(REPAIR_REF,policy_ref=POLICY_REF)
    return g

def consumer_guard():return original_guard(),None

def _candidate_metadata_v1(training,migration):
    require(training.get("status")=="completed_C3_checkpointed_complete_parent_training"
            and training.get("schema")=="american_c3_checkpointed_complete_parent_training_v1"
            and training.get("complete_parent_updates")==600 and training.get("epochs")==30
            and training.get("focal_exposures")==6000000
            and training.get("trained_parent_ids")==["american_c3_short_train_%03d"%i for i in range(20)]
            and training.get("train_sequence_lengths_bp")==[2000000]*20
            and training.get("VAL_used_for_optimization") is False and training.get("TEST_deferred") is True
            and training.get("H043_extension") is None and training.get("purity_filter") is False,
            "Genuine original20x2Mb/600-update completion, never progress/checkpoint lower bounds")
    require(training["resume_ref"]==migration["migrated_boundary_ref"]
            and training["protocol_sha256"]==migration["successor_protocol_sha256"]
            and training["observation_seal_sha256"]==migration["observation_seal_sha256"]
            and training["target_join_sha256"]==migration["target_join_sha256"],
            "Exact restored-prefix continuation science/data/protocol lineage")
    candidates=training["checkpoint_candidates"]
    require([x["epoch"] for x in candidates]==list(EPOCHS)
            and [x["global_step"] for x in candidates]==[20*e for e in EPOCHS],"All six original frozen candidates")
    receipts={x["global_step"]:x["migrated_ref"] for x in migration["migrated_checkpoint_receipts"]}
    for x in candidates:
        r=x["checkpoint_ref"]
        require(set(r)=={"path","sha256"} and len(r["sha256"])==64,"Exact checkpoint ref, no source-time payload read")
        if x["global_step"]<=migration["actual_global_step"]:
            require(receipts.get(x["global_step"])==r,"Original scheduled prefix restored by authenticated migration")
        else:require(r["path"]==TRAIN_OUTPUT+"/checkpoint_epoch_%03d.pt"%x["epoch"],
                     "Frozen continuation checkpoint namespace")
    require(len({x["checkpoint_ref"]["path"] for x in candidates})==6,"Six distinct original candidate files")
    return training

def terminal0_v1(e,completion_ref):
    wait=read(e["actual_wait_ref"]);env=read(completion_ref)
    require(wait.get("driver_ref")==env.get("root_driver_ref")
            and wait.get("input_ref")==env.get("input_ref"),"Same completed producer/input/driver lineage")
    require(wait.get("completed_ref")==completion_ref and wait.get("returncode")==0
            and wait.get("actual_direct_launcher_wait",wait.get("actual_direct_worker_wait")) is True
            and wait.get("all_owned_processes_and_groups_drained",wait.get("owned_worker_group_ESRCH")) is True
            and wait.get("operational_failure") is None,"Same-product actual direct wait0 and owned drain")
    if e.get("outer_terminal_ref") is not None:
        terminal=read(e["outer_terminal_ref"])
        require(terminal.get("actual_terminal_consumed") is True and terminal.get("actual_exit_code")==0
                and terminal.get("completed_ref")==completion_ref
                and terminal.get("actual_wait_ref",terminal.get("direct_wait_ref"))==e["actual_wait_ref"],
                "Actual root-consumed stage terminal0")
    else:
        z=read(e["close_receipt_ref"]);req=z["value"]["request"];obs=read(req["observation_ref"])
        approval=read(obs["root_approval_ref"])
        require(req["completed_ref"]==completion_ref and req["direct_wait_ref"]==e["actual_wait_ref"]
                and approval.get("root_actually_consumed_supervisor_terminal_wait0") is True,
                "Historical original closed plan binds actual consumed root terminal0")
    return wait

def plan_ready_v1(e,parent):
    require(parent in VAL and e["parent_id"]==parent,"Exact original heldout parent")
    env=read(e["completion_ref"]);terminal0_v1(e,e["completion_ref"])
    result=read(env["result_ref"])
    require(result.get("status")=="completed_C3_observation_only_block_plan"
            and result.get("parent_id")==parent and result.get("targets_opened") is False
            and result.get("block_index_ref")==e["plan_ref"],"Actual completed truth-free plan")
    plan=read(e["plan_ref"]);cfg=plan["config"];prep=read(cfg["native_preparation_ref"]);obs=read(plan["observation_ref"])
    require(plan.get("schema")=="american_c3_whole_tree_block_index_v1" and plan["parent_id"]==parent
            and plan["sequence_length_bp"]==30000000 and plan["haplotypes"]==1600 and plan["focal_records"]==10000
            and plan["all_trees_and_samples_retained"] is True and plan["targets_opened"] is False
            and plan["class_order"]==["EUR","EAS","AFR"],"Complete original30Mb/1600/10000 plan")
    require(prep["status"]=="completed_native_C3_path_preparation" and prep["parent_id"]==parent
            and prep["focals_ref"]==plan["focal_registry_ref"] and prep["observation_ref"]==plan["observation_ref"]
            and prep["query_focal_count"]==10000 and prep["query_ancestry_truth_opened"] is False
            and obs["parent_id"]==parent and obs["split"]=="val" and obs["sequence_length_bp"]==30000000
            and obs["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
            and obs["query_ancestry_truth_in_inputs"] is False,"Same native focal/query/observed ancestry-free preparation")
    z=read(e["close_receipt_ref"])
    require(z["kind"]=="concurrent_completed_scope_close_receipt_v1"
            and z["value"]["future_writes_and_relaunch_revoked"] is True
            and z["value"]["request"]["completed_ref"]==e["completion_ref"]
            and z["value"]["request"]["direct_wait_ref"]==e["actual_wait_ref"],
            "Actually closed immutable observation plan, no stale or live rewrite")
    return plan

def make_config(req,caps):
    return dict(schema="C3_full30Mb_evaluation_config_v1",execution_enabled=True,phase=req["phase"],
                output_cap_bytes=caps["reservation_cap_bytes"],short_tmp_directory=req["packet_directory"]+"/worker_tmp")

def _evaluation_request_v1(record):
    req=read(record);active(req);phase=req["phase"];require(phase in PHASES,"Only original VAL prediction/matrix/score")
    require(req["schema"]=="C3_full30Mb_evaluation_ROOT_request_v1"
            and req["device"]=="cpu" and req["GPU_authorized"] is False
            and req["physical_gpu"] is None and req["runtime_ref"]==RUNTIME_REF
            and req["source_manifest_ref"]==ref(MANIFEST),"Separately admitted exact CPU runtime, no borrowed-controller submission")
    require(req["output_directory"].startswith("experiments/results/C3_full30Mb_eval_")
            and req["packet_directory"].startswith("research/reviews/C3_full30Mb_eval_ROOT_")
            and not local(req["packet_directory"]).is_relative_to(local(HERE)),"Fresh separate ROOT output/packet scopes")
    manifest_v1();caps=read(req["caps_ref"]);require(caps==caps_v1(phase),"Exact reviewed failstop caps")
    b,training=training_ready_v1(req["training_evidence_ref"])
    if phase.startswith("predict_val_"):
        plan_ready_v1(req["plan_evidence"],VAL[int(phase[-3:])])
        require("VAL_truth_capability" not in req and "external_truth_ref" not in req,"Prediction has no private truth capability")
    elif phase=="matrix":
        require(set(req["VAL_predictions"])==set(VAL) and "VAL_truth_capability" not in req,
                "All five completed six-candidate cells, no truth at barrier stage")
        for parent,e in req["VAL_predictions"].items():
            env=read(e["completion_ref"]);terminal0_v1(e,e["completion_ref"])
            require(env["stage"]=="predict_val_"+parent[-3:] and env["status"]==STATUSES[env["stage"]] and env.get("root_driver_ref")==ref(DRIVER),
                    "Same-source completed independent VAL prediction stage")
            six=read(env["result_ref"])
            require(six["parent_id"]==parent and six["training_completion_ref"]==b["training_completion_ref"]
                    and len(six["prediction_seal_refs"])==6,"All six frozen candidates per original parent")
    else:
        e=req["matrix"];env=read(e["completion_ref"]);terminal0_v1(e,e["completion_ref"])
        require(env["stage"]=="matrix" and env["status"]==STATUSES["matrix"] and env.get("root_driver_ref")==ref(DRIVER),"Actually terminal complete pretruth matrix")
        require(set(req["VAL_truth_capability"])==set(VAL) and set(req["plan_evidence_by_parent"])==set(VAL),
                "Original5VAL separate truth capabilities, TEST absent")
    return req,caps,read(RUNTIME_REF)

def validate_input_v1(record):
    spec=read(record);active(spec);req,caps,_=validate_request(spec["root_stage_request_ref"])
    require(spec["schema"]=="C3_full30Mb_evaluation_active_input_v1"
            and spec["phase"]==req["phase"] and spec["root_driver_ref"]==ref(DRIVER)
            and spec["main_source_ref"]==ref(ENTRY) and spec["GPU_authorized"] is False
            and spec["physical_gpu"] is None and spec["output_directory"]==req["output_directory"]
            and spec["packet_directory"]==req["packet_directory"] and spec["refs"]["runtime"]==RUNTIME_REF
            and spec["source_sha256"]==manifest_v1()["source_sha256"],"Actual reviewed CPU producer/source/input/runtime")
    storage=read(spec["refs"]["storage"])
    require(storage["accounting_mode"]=="native14_stopped_retention_v3"
            and storage["completed_overlay_policy_ref"]==POLICY_REF
            and storage["job_id"]==spec["job_id"] and storage["output_directory"]==spec["output_directory"]
            and storage["reserved_new_bytes"]==caps["reservation_cap_bytes"]
            and storage["reservation"]["output"]==str(local(spec["output_directory"]))
            and storage["reservation"]["auxiliary_scopes"]==[
                str(local(spec["packet_directory"]+"/worker_aux")),str(local(spec["packet_directory"]+"/worker_tmp"))]
            and len(storage["reservation"]["consumer_bindings"])<=64
            and all(r in storage["reservation"]["consumer_bindings"] for r in required_direct_refs_v1(spec,req)),
            "Fresh exact current-child per-stage charge and every required direct role")
    require(read(spec["refs"]["config"])==make_config(req,caps),"Exact operational-only config")
    return spec,req,caps

def set_active_spec_v1(record):
    global _ACTIVE_SPEC
    require(_ACTIVE_SPEC is None,"One ROOT supervisor invocation")
    _ACTIVE_SPEC,req,caps=validate_input_v1(record)
    return _ACTIVE_SPEC,req,caps

def phase_paths(phase):
    require(_ACTIVE_SPEC is not None and phase==_ACTIVE_SPEC["phase"],"Exact activated phase")
    return local(_ACTIVE_SPEC["packet_directory"]),_ACTIVE_SPEC["output_directory"]

def manifest_v1():
    m=read(ref(MANIFEST))
    require(m["schema"]=="C3_saved600_acceptance_evaluation_source_manifest_v1" and m["source_only"] is True,
            "Separate independently reviewed source-only successor")
    for path,sha in m["source_sha256"].items():require(ref(path)["sha256"]==sha,"Frozen effective source changed")
    return m
def caps_v1(phase):
    require(phase in PHASES,"Only saved-artifact validation and original evaluation stages")
    if phase==ACCEPT_PHASE:
        return dict(CAPS,phase=phase,CPU_threads=1,max_cpu_rss_bytes=2147483648,operational_deadline_seconds=900)
    return dict(CAPS,phase=phase)
def frozen_v1():
    f=read(ref(FROZEN))
    require(f["schema"]=="C3_saved600_exact_original_artifact_contract_v1"
            and f["original_history"]==ORIGINAL_HISTORY,"Exact failed producer and UNKNOWN original outer history")
    return f
ORIGINAL_HISTORY=dict(producer_cli_exit_code=1,producer_leader_wait_status_known=True,
    controller_owned_drain_observed=True,controller_state="failed",
    original_outer_74136_terminal_consumed=None,original_outer_74136_exit_code=None,
    original_worker_drain_manifest=None,original_completed_envelope=None,
    actual_parent_release_count=None,actual_terminal_exception_text=None,
    source_required_releases=620,resumed_expected_fresh_releases_source_derived=520,
    observed_failure_cause_asserted=False)
def history_values_v1(value):
    require(json.dumps(value,sort_keys=True,allow_nan=False)==json.dumps(ORIGINAL_HISTORY,sort_keys=True,allow_nan=False),
        "Preserve producer1, missing original drain/envelope and original outerUNKNOWN with exact JSON value types")
def review_values_v1(review,source_ref,manifest_ref):
    require(review.get("verdict") in ("PASS","SOURCE_PASS")
        and review.get("source_ref")==source_ref and review.get("source_manifest_ref")==manifest_ref
        and review.get("independent") is True,"Independent complete new-source review")
def acceptance_inputs_v1(req):
    f=frozen_v1();history_values_v1(req["original_history"])
    require(not local(TRAIN_OUTPUT+"/completed.json").exists()
        and not local(TRAIN_OUTPUT+"/observation_prefetch_all_worker_drains.json").exists(),
        "Preserve absent original envelope/drain manifest; unexpected new original writes block")
    training=read(f["training_completion_ref"]);migration=read(MIGRATION_REF)
    require(f["training_completion_ref"]==req["training_completion_ref"]
        and f["checkpoint_candidates"]==training["checkpoint_candidates"],"Immutable original completion and six exact checkpoint SHAs")
    _candidate_metadata_v1(training,migration)
    failure=read(f["original_failure_audit_ref"]);line=failure["C3_recorded_terminal_lineage"]
    require(line["state"]=="failed" and line["exit_code"]==1 and line["leader_wait_status_known"] is True
        and line["owned_drain_observed"] is True and line["leader_pid"]==3352021 and line["leader_start_ticks"]==481303226,
        "Actual frozen controller direct leader1 and owned drain, never success inferred from artifacts")
    snapshot=read(f["original_controller_snapshot_ref"])
    jobs=[j for j in snapshot["jobs"] if j.get("logical_job")==TRAIN_JOB]
    require(len(jobs)==1 and jobs[0]["state"]=="failed" and jobs[0]["exit_code"]==1
        and jobs[0]["leader_wait_status_known"] is True and jobs[0]["owned_drain_observed"] is True
        and jobs[0]["requested"]["input_ref"]==TRAIN_ENVELOPE_REF,"Exact immutable recorded producer failure")
    inp=read(TRAIN_INPUT_REF);envelope=read(TRAIN_ENVELOPE_REF);operational=read(envelope["request_ref"])
    require(operational["method_job"]["input_ref"]==TRAIN_INPUT_REF
        and operational["method_job"]["job_id"]==TRAIN_JOB,"Original operational route to exact scientific continuation")
    for path,sha in inp["source_sha256"].items():require(ref(path)["sha256"]==sha,"Original actual scientific/runtime source closure")
    for r in (training["observation_manifest_ref"],training["target_manifest_ref"]):read(r)
    return dict(training_completion_ref=f["training_completion_ref"]),training
def validate_old_owner_absence_v1():
    """Future dedicated CPU worker only, after active authority/storage/runtime validation."""
    rows=[]
    for pid in (3352021,3352975,3352976,3352977):
        try:os.kill(pid,0)
        except ProcessLookupError:rows.append(dict(pid=pid,ESRCH=True))
        else:raise ValueError("Exact old C3 PID exists, including possible reuse; ROOT must resolve before acceptance")
    try:os.killpg(3352021,0)
    except ProcessLookupError:pass
    else:raise ValueError("Exact old C3 producer group still exists")
    return dict(exact_old_PID_absence=rows,old_group=3352021,old_group_ESRCH=True,
        historical_finalizer_manifest_reconstructed=False,controller_shutdown_or_edit=False)
def root_validation_request_v1(req):
    require(req.get("schema")=="C3_saved600_ROOT_validation_request_v1" and req.get("phase")==ACCEPT_PHASE
        and req.get("purpose")=="validate_saved_artifacts_only" and "training_evidence_ref" not in req,
        "Dedicated separately authorized validation, no circular or premature acceptance")
    require(req.get("ROOT_accepts_original_producer1_and_outerUNKNOWN_for_artifact_validation_only") is True
        and req.get("future_original_C3_training_writes_cooperatively_revoked") is True
        and req.get("controller_preserved_for_borrowed_native_work") is True,
        "Explicit narrow ROOT failure/UNKNOWN acknowledgement and future producer-write revocation")
    require(all(k not in req for k in ("VAL_truth_capability","external_truth_ref","VAL_predictions","matrix","plan_evidence")),
        "Validator has no heldout truth, prediction or score capability")
    review_values_v1(read(req["source_review_ref"]),ref(SOURCE),ref(MANIFEST))
    return acceptance_inputs_v1(req)
def validate_request(record):
    req=read(record)
    if req.get("phase")!=ACCEPT_PHASE:return _evaluation_request_v1(record)
    active(req);manifest_v1()
    require(req["device"]=="cpu" and req["GPU_authorized"] is False and req["physical_gpu"] is None
        and req["runtime_ref"]==RUNTIME_REF and req["source_manifest_ref"]==ref(MANIFEST),"Exact separate CPU validator runtime")
    require(req["output_directory"].startswith("experiments/results/C3_full30Mb_eval_")
        and req["packet_directory"].startswith("research/reviews/C3_full30Mb_eval_ROOT_")
        and not local(req["packet_directory"]).is_relative_to(local(HERE)),"Fresh separately admitted guarded validator scopes")
    caps=read(req["caps_ref"]);require(caps==caps_v1(ACCEPT_PHASE),"Exact CPU-only saved-state validation failstop caps")
    root_validation_request_v1(req)
    return req,caps,read(RUNTIME_REF)
def acceptance_values_v1(b,env,result,wait,terminal,source_ref,driver_ref,manifest_ref,expected_training_ref,expected_candidates):
    history_values_v1(b["original_history"]);history_values_v1(result["original_history"])
    require(b.get("schema")=="C3_saved600_ROOT_artifact_acceptance_v1" and b.get("root_authorizer")=="root_agent"
        and b.get("accepted_saved_artifacts_only") is True
        and b.get("original_invocation_success_claimed") is False
        and b.get("original_outerUNKNOWN_explicitly_acknowledged") is True
        and b.get("future_original_C3_training_writes_cooperatively_revoked") is True
        and b.get("storage_close_or_credit_authorized") is False,"Explicit ROOT saved-artifact acceptance after independent validator evidence")
    require(b["training_completion_ref"]==expected_training_ref and b["source_manifest_ref"]==manifest_ref,
        "Same immutable training family and reviewed bridge")
    require(env.get("status")=="completed_C3_saved600_artifact_validation" and env.get("stage")==ACCEPT_PHASE
        and env.get("root_driver_ref")==driver_ref and env.get("training_completion_ref")==expected_training_ref
        and env.get("input_ref")==b["validator_input_ref"] and env.get("result_ref")==b["validation_result_ref"],
        "Exact dedicated validator completion/input/source joins")
    require(result.get("status")=="validated_C3_saved600_artifacts_after_failed_producer"
        and result.get("all_six_verified") is True and result.get("saved_effective_logical_updates")==600
        and type(result.get("optimizer_updates_performed"))is int and result.get("optimizer_updates_performed")==0
        and result.get("VAL_or_TEST_truth_opened") is False
        and result.get("root_acceptance_still_required") is True and result.get("original_history")==ORIGINAL_HISTORY
        and result.get("dedicated_validator_CPU_runtime_checked") is True and result.get("CUDA_initialized") is False
        and result.get("source_manifest_ref")==manifest_ref
        and [dict(epoch=x["epoch"],global_step=x["global_step"],checkpoint_ref=x["checkpoint_ref"]) for x in result["checkpoints"]]==expected_candidates
        and all(x["all_checks_passed"] is True and x["actual_AdamW_step_min"]==x["global_step"]==x["actual_AdamW_step_max"]
            and x["optimizer_parameter_states"]==38 and x["model_parameters"]==29571 for x in result["checkpoints"]),
        "All six genuinely validated original saved states, never progress/final-only substitutes")
    require(result["provenance"]["parents"]==20 and result["provenance"]["all_original_block_seals"] is True
        and result["original_producer_success_claimed"] is False
        and result["original_worker_drain_manifest_reconstructed"] is False
        and result["storage_close_or_credit_authorized"] is False,"Complete original TRAIN seal provenance and preserved failure boundary")
    require(type(wait.get("returncode"))is int and wait.get("returncode")==0 and wait.get("actual_direct_launcher_wait") is True
        and wait.get("all_owned_processes_and_groups_drained") is True and wait.get("operational_failure") is None
        and wait.get("driver_ref")==driver_ref and wait.get("input_ref")==b["validator_input_ref"]
        and wait.get("completed_ref")==b["validator_completion_ref"] and type(wait.get("pid"))is int,
        "Dedicated validator's genuine actual direct CPU wait0 and owned drain")
    require(terminal.get("actual_terminal_consumed") is True and type(terminal.get("actual_exit_code"))is int
        and terminal.get("actual_exit_code")==0 and terminal.get("completed_ref")==b["validator_completion_ref"]
        and terminal.get("actual_wait_ref")==b["actual_wait_ref"]
        and terminal.get("input_ref")==b["validator_input_ref"] and terminal.get("driver_ref")==driver_ref,
        "Dedicated validator's own consumed actual outer terminal0, distinct from original outerUNKNOWN")
def training_ready_v1(bundle_ref):
    b=read(bundle_ref);active(b);f=frozen_v1()
    review_values_v1(read(b["source_review_ref"]),ref(SOURCE),ref(MANIFEST))
    env=read(b["validator_completion_ref"]);result=read(b["validation_result_ref"])
    wait=read(b["actual_wait_ref"]);terminal=read(b["outer_terminal_ref"])
    acceptance_values_v1(b,env,result,wait,terminal,ref(SOURCE),ref(DRIVER),ref(MANIFEST),
        f["training_completion_ref"],f["checkpoint_candidates"])
    spec=read(b["validator_input_ref"]);req=read(spec["root_stage_request_ref"])
    root_validation_request_v1(req)
    require(req["device"]=="cpu" and req["GPU_authorized"] is False and req["physical_gpu"] is None
        and req["runtime_ref"]==RUNTIME_REF and req["source_manifest_ref"]==ref(MANIFEST)
        and read(req["caps_ref"])==caps_v1(ACCEPT_PHASE),"Historical exact dedicated validator request/runtime/caps")
    require(spec["phase"]==ACCEPT_PHASE and env["input_ref"]==b["validator_input_ref"]
        and spec["main_source_ref"]==ref(ENTRY) and spec["root_driver_ref"]==ref(DRIVER)
        and spec["source_sha256"]==manifest_v1()["source_sha256"] and spec["root_stage_request_ref"]==b["validator_request_ref"],
        "Actual guarded validation input and reviewed effective source closure")
    bundle,training=acceptance_inputs_v1(req)
    require(result["training_completion_ref"]==bundle["training_completion_ref"]
        and result["protocol_sha256"]==training["protocol_sha256"]
        and result["observation_seal_sha256"]==training["observation_seal_sha256"]
        and result["target_join_sha256"]==training["target_join_sha256"],"Actual validator result binds exact unchanged training lineage")
    return b,training
def required_direct_refs_v1(spec,req):
    m=manifest_v1()
    evidence=req["source_review_ref"] if req["phase"]==ACCEPT_PHASE else req["training_evidence_ref"]
    rows=[*m["current_guard_roles"].values(),ref(SOURCE),ref(DRIVER),ref(ENTRY),ref(MANIFEST),ref(CORE_SOURCE),
        ref(FROZEN),OLD_CONTRACT_REF,OLD_WORKER_REF,req["caps_ref"],spec["refs"]["config"],RUNTIME_REF,
        spec["root_stage_request_ref"],evidence,BASE_DRIVER_REF,*m["supervisor_dependency_refs"]]
    indexed={}
    for r in rows:
        require(Path(r["path"]).suffix in (".py",".sh",".json"),"Original direct binding types")
        require(r["path"] not in indexed or indexed[r["path"]]==r,"No conflicting ancestor/new direct role")
        indexed[r["path"]]=r
    require(len(indexed)<=64,"Every current ancestor and actual operational role fits original64 cap")
    return list(indexed.values())

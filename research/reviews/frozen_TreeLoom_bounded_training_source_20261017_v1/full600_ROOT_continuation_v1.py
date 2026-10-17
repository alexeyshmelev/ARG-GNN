"""Finite ROOT full600 continuation: metadata wait, original admission, one exec.

Default is inert. ROOT supplies immutable finite authority and exact future
paths. No scientific payload is read here. No source pass proves full600 fit.
The sealed worker constructs fresh seed70001/model/AdamW; no qualification state
or checkpoints are loaded. A missing or insufficient measured launch window
returns READY_DATA_NO_ADMISSION. There is no retry, resume, owner renewal or credit.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,math,os,re,stat,sys,time
from . import contract_v1 as c
HERE=c.HERE
SOURCE=HERE+"/full600_ROOT_continuation_v1.py"
MANIFEST=HERE+"/full600_ROOT_continuation_manifest.v1.json"
FULL_TEMPLATE=HERE+"/ROOT_full600_input.INACTIVE.v1.json"
BARRIER_TEMPLATE=HERE+"/ROOT_full600_observation_barrier.INACTIVE.v1.json"
REVIEW=dict(path="research/reviews/frozen_TreeLoom_bounded_training_independent_20261017_v1/review.qualification.v1.json",sha256="ed77e7885a9b7403d778dc483ac2074e1b3f4bd4794eaa996f7488a8fcca001f")
POLICY=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json",sha256="5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5")
SELECTOR=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json",sha256="fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")
ENDPOINT=dict(path="research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/empty_gpu0_controller_v1.py",sha256="dca4230a3d79540b947fa5488c6a49a885970db367b9955f78c343b8305b398f")
COORDINATOR=dict(path="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/worker_entry_v1.py",sha256="2c4a976949fe147238ba775fa6aee58bbb782a9cef2c76d65945a472eae6be73")
PARENTS=["american_c3_train_%03d"%i for i in range(20)]
ADMIN_CAP=384*1024
RESERVATION_BYTES=64*2**20
class Pending(Exception):pass
class ReadyOnly(Exception):pass

def _short_ref(r):
    c.require(type(r)is dict and set(r)=={"path","sha256"}and type(r["path"])is str
        and type(r["sha256"])is str and re.fullmatch("[0-9a-f]{64}",r["sha256"])is not None,
        "Exact compact immutable reference")
    c.local(r["path"]);return r

def _save(directory,name,value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    c.require(len(raw)<=65536,"Each administrative JSON <=64KiB")
    total=4096;count=0
    for p in directory.iterdir():
        s=p.lstat()
        c.require(stat.S_ISREG(s.st_mode)and not p.is_symlink()and s.st_nlink==1
            and p.suffix==".json"and s.st_size<=65536,"Flat own metadata only")
        total+=s.st_size+4096;count+=1
    c.require(count<32 and total+len(raw)+4096<=ADMIN_CAP and ADMIN_CAP+65536+4096<=524288,
        "Own records plus original event fit unchanged512KiB allowance")
    fd=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,"wb")as stream:
        c.require(stream.write(raw)==len(raw),"Complete exclusive administrative write")
        stream.flush();os.fsync(stream.fileno())
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    return c.ref(str((directory/name).relative_to(c.ROOT)))

def _future(path):
    p=c.local(path)
    if not p.exists():raise Pending(path)
    s=p.lstat()
    c.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and not p.is_symlink()and s.st_size<=65536,
        "Bounded single-link future compact ROOT JSON")
    raw=p.read_bytes();after=p.lstat()
    if len(raw)!=s.st_size or (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise Pending(path)
    try:value=json.loads(raw)
    except json.JSONDecodeError:raise Pending(path)
    return dict(path=str(p.relative_to(c.ROOT)),sha256=hashlib.sha256(raw).hexdigest()),value

def _request(record):
    q=c.read(record);c.active(q)
    c.require(q["schema"]=="ROOT_TreeLoom_full600_continuation_request_v1"
        and q["continuation_manifest_ref"]==c.ref(MANIFEST)
        and q["source_manifest_ref"]==c.ref(c.MANIFEST)and q["source_review_ref"]==REVIEW
        and q["storage_policy_selector_ref"]==SELECTOR and q["accounting_coordinator_source_ref"]==COORDINATOR
        and q["automatic_retry_or_resume"]is False
        and q["all_other_GPU0_borrower_routes_revoked"]is True
        and q["preserve_controller_until_owned_drain"]is True
        and type(q["automatic_single_parent_launch_authorized"])is bool,
        "Exact finite ROOT continuation, sealed science and current serialized accounting")
    c.require(type(q["instance"])is str and re.fullmatch("[a-z][a-z0-9_]{0,39}",q["instance"])is not None,
        "New bounded one-shot administrative instance")
    start=datetime.fromisoformat(q["created_at_utc"]);end=datetime.fromisoformat(q["expires_at_utc"])
    wait=datetime.fromisoformat(q["wait_deadline_at_utc"])
    c.require(start<=wait<=end and (end-start).total_seconds()<=14*86400
        and type(q["poll_seconds"])is int and 5<=q["poll_seconds"]<=60,
        "Finite overall ROOT authority and finite5..60s metadata polling interval")
    for key in("future_qualification_acceptance_path","future_all20_readiness_path",
        "future_all20_truth_handles_path","future_launch_authority_path"):
        p=c.local(q[key]);c.require(p.suffix==".json"and str(p.relative_to(c.ROOT)).startswith("research/reviews/"),
            "Explicit exact future compact review path")
    c.raw(q["qualification_input_ref"]);c.raw(q["accounting_batch_input_ref"])
    m=c.read(q["source_manifest_ref"]);review=c.read(REVIEW);own=c.read(q["continuation_manifest_ref"])
    c.require(own["source_ref"]==c.ref(SOURCE)and own["trainer_manifest_ref"]==q["source_manifest_ref"]
        and own["accounting_coordinator_source_ref"]==COORDINATOR,
        "This separate continuation source and exact current coordinator")
    c.require(review["independent"]is True and review["verdict"]=="SOURCE_PASS"and review["blockers"]==[]
        and review["candidate_manifest_ref"]==q["source_manifest_ref"],"Existing independent whole trainer SOURCE_PASS")
    for path,sha in m["source_sha256"].items():c.raw(dict(path=path,sha256=sha))
    return q,m

def _qualification(record,value,q):
    a=value
    c.require(a["schema"]=="ROOT_frozen_TreeLoom_actual_qualification_acceptance_v1"
        and a["root_authorizer"]=="root_agent"and a["actual_outer_terminal_consumed"]is True
        and type(a["actual_exit_code"])is int and a["actual_exit_code"]==0
        and a["all_owned_processes_drained"]is True and a["source_manifest_ref"]==q["source_manifest_ref"]
        and a["synthetic_all49_parity_passed"]is True
        and a["full30Mb_TRAIN000_disposable_optimizer_update_verified"]is True
        and a["qualification_weights_reused"]is False
        and a["qualification_input_ref"]==q["qualification_input_ref"],
        "Actual exact accepted qualification0 and fresh-state condition")
    qualified=c.read(a["qualification_input_ref"])
    done=c.read(a["completed_ref"]);wait=c.read(a["actual_parent_wait_ref"]);c.raw(a["actual_outer_terminal_ref"])
    parity_ref=a.get("synthetic_parity_ref")
    if parity_ref is None:parity_ref=c.ref(qualified["output_directory"]+"/synthetic_parity.json")
    parity=c.read(parity_ref)
    c.require(qualified["phase"]=="qualification"and qualified["source_manifest_ref"]==q["source_manifest_ref"]
        and done["input_ref"]==a["qualification_input_ref"]and done["phase"]=="qualification"
        and done["complete_parent_updates"]==1 and done["trainable_tensors"]==49
        and done["qualification_weights_are_not_final_training_weights"]is True
        and done["qualification_warmstart_or_optimizer_or_rng_reuse"]is False
        and done["VAL_truth_read"]is False and done["TEST_truth_read"]is False
        and type(done["elapsed_seconds"])in(int,float)and math.isfinite(done["elapsed_seconds"])and done["elapsed_seconds"]>0
        and wait["scientific_input_ref"]==a["qualification_input_ref"]
        and type(wait["actual_direct_returncode"])is int and wait["actual_direct_returncode"]==0
        and type(wait["actual_raw_wait"])is int and os.waitstatus_to_exitcode(wait["actual_raw_wait"])==0
        and wait["all_owned_processes_drained"]is True and wait["pipe_eof"]is True
        and wait["failure"]is None and wait["cleanup_errors"]==[]and wait["adopted_waits_truncated"]is False
        and all(x["returncode"]==0 for x in wait["adopted_waits"])
        and parity["passed"]is True
        and parity["all49_raw_and_clipped_gradients_and_AdamW_parameters_moments_compared"]is True
        and parity["nonzero_memory_gradient"]is True,
        "Real whole-parent/parity completion, raw/direct/adopted waits and drain, no score")
    return dict(acceptance_ref=record,acceptance=a,qualified_input=qualified,completed=done,parity_ref=parity_ref)

def _all20(readiness_ref,readiness,truth_ref,truth,qualification):
    c.require(readiness["schema"]=="TreeLoom_all20_training_readiness_v1"
        and readiness["all20_complete"]is True and readiness["query_truth_opened"]is False,
        "Actual all20 observation/nativefocal seal, no query truth")
    # Exact author row schema; no inferred names or discovery/glob over product directories.
    c.require(readiness["TRAIN_parent_ids"]==PARENTS and list(readiness["parents"])==PARENTS
        and truth["schema"]=="TreeLoom_all20_training_truth_handles_v1"
        and truth["TRAIN_parent_ids"]==PARENTS and list(truth["parents"])==PARENTS
        and truth["class_order"]==["EUR","EAS","AFR"]and truth["query_truth_opened"]is False,"Exact original20TRAIN order, no VAL/TEST or substituted parent")
    barrier=c.read(c.ref(BARRIER_TEMPLATE))
    barrier.update(root_authorizer="root_agent",active=True,actual_records_verified=True,source_only=False,
        qualification_acceptance_ref=qualification["acceptance_ref"],all20_readiness_ref=readiness_ref)
    parents={};handles={}
    for parent_id in PARENTS:
        row=readiness["parents"][parent_id];label=truth["parents"][parent_id]
        c.require(row["parent_id"]==label["parent_id"]==parent_id
            and row["observation_only"]is True and row["all_native_trees_nodes_sites_samples_retained"]is True,
            "Exact same-parent original all20 joins")
        c.raw(row["inference_close_accepted_ref"]);c.raw(row["native_close_accepted_ref"])
        observed_ref=_short_ref(row["observation_ref"])
        prepared_ref=_short_ref(row["native_preparation_ref"])
        focals_ref=_short_ref(row["focal_registry_ref"])
        observed=c.read(observed_ref);prepared=c.read(prepared_ref)
        c.require(observed["parent_id"]==prepared["parent_id"]==parent_id
            and observed["split"]=="train"and observed["sequence_length_bp"]==30000000
            and observed["query_ancestry_truth_in_inputs"]is False
            and prepared["observation_ref"]==observed_ref and prepared["focals_ref"]==focals_ref
            and prepared["query_ancestry_truth_opened"]is False,
            "Each real full30Mb sanitized observation and native10000focal preparation")
        identity=row["focal_identity_sha256"]
        c.require(type(identity)is str and re.fullmatch("[0-9a-f]{64}",identity)is not None,
            "Actual label-blind native focal identity metadata; no array read here")
        private=_short_ref(label["truth_ref"])
        c.require(private["path"].endswith(".npz")and "/truth/"in private["path"]
            and not any(s in private["path"].lower()for s in("val_","test_")),"TRAIN-only original truth handle; payload remains unopened")
        parents[parent_id]=dict(observation_ref=observed_ref,native_preparation_ref=prepared_ref,focal_registry_ref=focals_ref)
        handles[parent_id]=private
        barrier["parents"][parent_id]=dict(parents[parent_id],focal_identity_sha256=identity,
            observation_only=True,all_native_trees_nodes_sites_samples_retained=True)
    c.require(barrier["all20_observations_sealed_before_this_full_training_truth"]is True
        and barrier["this_full_training_truth_opened"]is False
        and barrier["prior_disposable_TRAIN000_qualification_genuine_truth_opened"]is True
        and barrier["qualification_weights_optimizer_or_rng_reused"]is False,"Honest new full-training truth barrier")
    return dict(readiness_ref=readiness_ref,truth_handles_ref=truth_ref,parents=parents,train_truth_refs=handles,barrier=barrier)

def _launch(record,launch,q,qualification,data):
    try:c.active(launch)
    except ValueError as exc:raise ReadyOnly("ROOT launch authority is not currently active: "+str(exc))
    c.require(launch["schema"]=="ROOT_TreeLoom_full600_launch_authority_v1"
        and launch["continuation_request_ref"]==q["_request_ref"]
        and launch["source_manifest_ref"]==q["source_manifest_ref"]
        and launch["qualification_acceptance_ref"]==qualification["acceptance_ref"]
        and launch["all20_readiness_ref"]==data["readiness_ref"]
        and launch["all20_truth_handles_ref"]==data["truth_handles_ref"]
        and launch["automatic_single_parent_launch_authorized"]is True
        and launch["all_other_GPU0_borrower_routes_revoked"]is True
        and launch["preserve_controller_until_owned_drain"]is True
        and launch["qualification_weights_optimizer_or_rng_reused"]is False,
        "Exact separately ROOT-approved full600 automatic launch and fresh-state authority")
    owner=c.read(launch["controller_capability_ref"]);ready=c.read(launch["controller_ready_ref"])
    try:c.active(owner)
    except ValueError as exc:raise ReadyOnly("Actual owner authority is not currently active; fresh ROOT owner required: "+str(exc))
    status=ready["actual_status"]
    c.require(ready["root_authorizer"]=="root_agent"and ready["actual_status_terminal_consumed"]is True
        and type(ready["actual_status_tool_exit_code"])is int and ready["actual_status_tool_exit_code"]==0
        and ready["controller_capability_ref"]==launch["controller_capability_ref"]
        and ready["controller_pid"]==status["controller"]["pid"]
        and ready["controller_start_ticks"]==status["controller"]["start_ticks"]
        and ready["source_ref"]==ENDPOINT and ready["owner_parent_still_live"]is True
        and status["capability_ref"]==launch["controller_capability_ref"]
        and status["jobs"]==[]and status["lifecycle"]=="running"and status["safety_error"]is None
        and status["ambiguous_descendants_or_lease_holders"]==[]
        and owner["source_refs"][ENDPOINT["path"]]==ENDPOINT and owner["jobs"]=={}
        and owner["physical_gpu"]==0 and owner["gpu_uuid"]==c.GPU0_UUID
        and owner["protected_envelope_bytes"]==0
        and owner["storage_policy_selector_ref"]==SELECTOR and owner["capacity_policy_ref"]==POLICY,
        "Genuine immutable ready actual current or fresh v3 owner; parent revalidates actual liveness/capacity")
    runtime=c.read(launch["runtime_ref"]);runtime_qualification=c.read(launch["runtime_qualification_ref"])
    affinity=runtime_qualification["cpu_affinity"]
    c.require(runtime["source_ref"]==c.ref(c.SOURCE)and runtime["CUDA_VISIBLE_DEVICES"]==c.GPU0_UUID
        and runtime["python"]==runtime_qualification["python"]
        and runtime_qualification["ROOT_qualified_existing_Torch_PyG_runtime"]is True
        and runtime_qualification["runtime_ref"]==launch["runtime_ref"]
        and type(affinity)is list and 8<=len(affinity)<=32
        and len(set(affinity))==len(affinity)and all(type(x)is int and x>=0 for x in affinity),
        "Actual ROOT-qualified existing scientific interpreter and explicit bounded CPU affinity")
    plan=launch["measured_runtime_resource_plan"];done=qualification["completed"];seconds=launch["max_seconds"]
    c.require(plan["schema"]=="ROOT_TreeLoom_full600_measured_runtime_resource_plan_v1"
        and plan["ROOT_approved"]is True and plan["qualification_completed_ref"]==qualification["acceptance"]["completed_ref"]
        and plan["actual_qualification_elapsed_seconds"]==done["elapsed_seconds"]
        and plan["actual_qualification_peak_cuda_reserved_bytes"]==done["peak_cuda_reserved_bytes"]
        and plan["actual_qualification_peak_cpu_RSS_bytes"]==done["peak_cpu_RSS_bytes"]
        and plan["all20_parent_dimensions_reviewed"]is True and plan["length_only_extrapolation"]is False
        and plan["persistent_preparation_or_tensor_cache_assumed"]is False
        and plan["full_optimizer_updates"]==600 and type(plan["basis"])is str and bool(plan["basis"].strip()),
        "Measured qualification and all20 dimensions inform explicit ROOT full600 plan, no unimplemented cache")
    estimate=plan["estimated_full600_work_seconds"]
    c.require(type(estimate)in(int,float)and math.isfinite(estimate)and estimate>0
        and type(seconds)is int and 300<seconds<=7*86400,"Finite explicit600update work estimate and original <=7day cap")
    start=datetime.now(timezone.utc);end=datetime.fromisoformat(launch["expires_at_utc"])
    root_end=datetime.fromisoformat(q["expires_at_utc"])
    if not (estimate+300<=seconds and start.timestamp()+seconds<=end.timestamp()
        and end<=root_end and end.timestamp()<=owner["expires_unix"]):
        raise ReadyOnly("Actual measured full600 work plus original300s margin does not fit ROOT launch/actual owner clock; no extension or admission")
    return dict(launch_ref=record,authority=launch,owner=owner,ready=ready,
        runtime=runtime,runtime_qualification=runtime_qualification,created_at_utc=start.isoformat(),
        expires_at_utc=end.isoformat(),drain_expires_unix=end.timestamp())

def _wait(q,directory):
    deadline=min(time.monotonic()+max(0.,datetime.fromisoformat(q["wait_deadline_at_utc"]).timestamp()-time.time()),
        time.monotonic()+14*86400)
    data=None;qualified=None;pending=[]
    print(json.dumps(dict(status="WAITING_FINITE_FULL600_METADATA",wait_deadline_at_utc=q["wait_deadline_at_utc"],
        automatic_launch=q["automatic_single_parent_launch_authorized"])),flush=True)
    while True:
        if time.monotonic()>=deadline:
            reason="Finite ROOT metadata deadline reached; pending="+",".join(pending)
            if data is not None:return qualified,data,None,reason
            raise Pending(reason)
        c.active(q)
        try:
            ar,a=_future(q["future_qualification_acceptance_path"]);qualified=_qualification(ar,a,q)
            rr,r=_future(q["future_all20_readiness_path"]);tr,t=_future(q["future_all20_truth_handles_path"])
            data=_all20(rr,r,tr,t,qualified)
            if not q["automatic_single_parent_launch_authorized"]:return qualified,data,None,"ROOT requested data readiness only"
            lr,l=_future(q["future_launch_authority_path"])
            try:launch=_launch(lr,l,q,qualified,data)
            except ReadyOnly as exc:return qualified,data,None,str(exc)
            return qualified,data,launch,None
        except Pending as exc:pending=[str(exc)]
        if time.monotonic()>=deadline:
            reason="Finite ROOT metadata deadline reached; pending="+",".join(pending)
            if data is not None:return qualified,data,None,reason
            raise Pending(reason)
        time.sleep(min(q["poll_seconds"],max(0.,deadline-time.monotonic())))

def _controls(launch,job):
    a=launch["authority"];ready=launch["ready"]
    common=dict(root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,
        created_at_utc=launch["created_at_utc"],expires_at_utc=launch["expires_at_utc"])
    binding=dict(common,schema="native_VAL000_downstream_GPU0_controller_binding_v1",
        source_review_passed=True,endpoint_source_ref=ENDPOINT,physical_gpu=0,GPU_UUID=c.GPU0_UUID,
        additional_controller_and_other_borrower_submissions_revoked=True,
        controller_preserved_until_downstream_drain=True,independent_source_review_ref=REVIEW,
        route="reviewed_same_protocol_successor",protected_job=None,protected_envelope_bytes=0,
        successor_capability_explicitly_ROOT_authorized=True,controller_pid=ready["controller_pid"],
        controller_start_ticks=ready["controller_start_ticks"],controller_capability_ref=a["controller_capability_ref"],
        original_live_ready_evidence_ref=a["controller_ready_ref"])
    attach=dict(common,schema="native_VAL000_downstream_direct_attachment_authority_v1",
        borrowed_physical_inode_attachment_authorized=True,
        additional_old_controller_submissions_revoked_while_downstream_attached=True,
        old_controller_preserved_until_downstream_drained=True,
        controller_pid=ready["controller_pid"],controller_start_ticks=ready["controller_start_ticks"],
        controller_capability_ref=a["controller_capability_ref"],controller_binding_ref=None,
        downstream_job_id=job,protected_envelope_bytes=0,downstream_max_cuda_reserved_bytes=64*2**30,
        downstream_non_Torch_margin_bytes=2*2**30,headroom_bytes=4*2**30)
    return binding,attach

def _remaining_fits(actual,plan):
    end=min(datetime.fromisoformat(actual["created_at_utc"]).timestamp()+actual["caps"]["max_seconds"],
        actual["drain_expires_unix"])
    return time.time()+plan["estimated_full600_work_seconds"]+300<=end

def _admit(q,manifest,directory,qualified,data,launch):
    c.active(q);c.active(launch["authority"])
    # Repeat the immutable measured-window/owner join immediately before admission.
    launch=_launch(launch["launch_ref"],c.read(launch["launch_ref"]),q,qualified,data)
    suffix="full600_"+q["instance"];job="TreeLoom-frozen-bounded-"+suffix
    output="experiments/results/TreeLoom_frozen_bounded_"+suffix
    auxiliary=".runtime/tmp/TreeLoom_frozen_bounded_"+suffix
    c.require(not c.local(output).exists()and not c.local(auxiliary).exists(),"Fresh absent full600 primary/AUX; no reuse")
    barrier_ref=_save(directory,"all20_observation_barrier.json",data["barrier"])
    bundle_ref=_save(directory,"ready_control_bundle.json",dict(schema="ROOT_TreeLoom_full600_exact_ready_control_bundle_v1",
        continuation_request_ref=q["_request_ref"],qualification_acceptance_ref=qualified["acceptance_ref"],
        all20_readiness_ref=data["readiness_ref"],all20_truth_handles_ref=data["truth_handles_ref"],
        full_training_observation_barrier_ref=barrier_ref,launch_authority_ref=launch["launch_ref"],
        exact_measured_plan=launch["authority"]["measured_runtime_resource_plan"],
        qualification_model_optimizer_rng_or_checkpoint_reused=False,all20_before_this_full_training_truth=True))
    binding,attachment=_controls(launch,job)
    binding_ref=_save(directory,"controller_binding.json",binding)
    attachment["controller_binding_ref"]=binding_ref
    attachment_ref=_save(directory,"direct_attachment_authority.json",attachment)
    actual=c.read(c.ref(FULL_TEMPLATE));a=launch["authority"]
    actual.update(root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,
        created_at_utc=launch["created_at_utc"],expires_at_utc=launch["expires_at_utc"],
        drain_expires_unix=launch["drain_expires_unix"],job_id=job,output_directory=output,
        worker_auxiliary_scope=auxiliary,ROOT_record_directory=auxiliary,
        source_manifest_ref=q["source_manifest_ref"],source_review_ref=REVIEW,
        runtime_ref=a["runtime_ref"],runtime_qualification_ref=a["runtime_qualification_ref"],
        cpu_affinity=launch["runtime_qualification"]["cpu_affinity"],storage_policy_selector_ref=SELECTOR,
        controller_binding_ref=binding_ref,direct_attachment_authority_ref=attachment_ref,
        qualification_acceptance_ref=qualified["acceptance_ref"],observation_barrier_ref=barrier_ref,
        parents=data["parents"],train_truth_refs=data["train_truth_refs"],
        ROOT_admission_request_ref=q["_request_ref"],ROOT_ready_control_bundle_ref=bundle_ref)
    actual["caps"]["max_seconds"]=a["max_seconds"]
    g=c.load(c.GUARD,"_TreeLoom_full600_ROOT_mutation_Census")
    c.require(g.load_selected_guard_v1(SELECTOR)is g and g.MODE=="Census_observation_two_scope_retention_v1"
        and g.MAX_JSON==65536 and g.META_CHARGE==524288,"Exact existing current Census mutation route")
    c.require(all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__ for n in("canonical","file_bytes","mutex")),
        "Fresh unwrapped ROOT mutation interpreter, never consumer memo")
    policy=c.read(POLICY);implementation=policy["implementation_ref"];launcher=policy["launcher_ref"]
    c.raw(implementation);c.raw(launcher)
    direct=manifest["executed_new_source_refs"]+[c.GUARD,q["source_manifest_ref"],actual["config_ref"],a["runtime_ref"],
        a["runtime_qualification_ref"],q["_request_ref"],REVIEW,binding_ref,attachment_ref,
        c.ref(SOURCE),q["continuation_manifest_ref"],COORDINATOR,q["accounting_batch_input_ref"],
        bundle_ref,qualified["acceptance_ref"]]
    # Single real control-bundle consolidation authenticates readiness/truth/launch
    # metadata by exact refs; never omit any of47 current guard roles or raw-lock bind.
    for r in direct:c.raw(r)
    coordinator=c.load(COORDINATOR,"_TreeLoom_full600_accounting_transaction")
    with coordinator.accounting_transaction(q["accounting_batch_input_ref"]):
        c.active(q);c.active(a)
        if not _remaining_fits(actual,a["measured_runtime_resource_plan"]):
            raise ReadyOnly("Measured full600 work no longer fits remaining immutable clock before capture; no reservation")
        reservation=dict(key="H056/"+job,output=str(c.local(output)),
            auxiliary_scopes=[str(c.local(auxiliary))],reserved_bytes=RESERVATION_BYTES)
        with g.budget.v4.mutex(c.ROOT):
            if not _remaining_fits(actual,a["measured_runtime_resource_plan"]):
                raise ReadyOnly("Measured full600 work no longer fits after waiting original mutex; no reservation")
            state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
            reservation["consumer_bindings"]=g.consumer_bindings_v1(list(state[3].values()),direct)
            g.budget.v4.scope_spec(c.ROOT,reservation)
            authority=g.operation_draft_v1("reserve",reservation,expected_before=g._pins(state[2]),bindings=state[3])
            authority.update(active=True,expires_at_utc=actual["expires_at_utc"])
            authority_ref=_save(directory,"reserve_authority.json",authority)
            event=dict(schema="Census_observation_exact_retention_event_v1",sequence=len(state[4])+1,
                previous_sha256=state[5][-1]if state[5]else None,bindings=state[3],operation="reserve",
                authority=authority_ref,value=dict(reservation,root_admission=authority_ref))
            event_bytes=len(g.budget.v4.encoded(event)+b"\n")
            c.require(event_bytes<=65536,"Original serialized reserve event remains within64KiB")
        if not _remaining_fits(actual,a["measured_runtime_resource_plan"]):
            raise ReadyOnly("Measured full600 work no longer fits before original apply; no reservation")
        receipt=g.apply_reserve_v1(authority_ref,expected_source_sha256=implementation["sha256"],
            expected_launcher_sha256=launcher["sha256"])
    receipt_ref=_save(directory,"reserve_receipt.json",receipt)
    c.require(receipt["value"]==dict(reservation,root_admission=authority_ref),"Genuine original one-shot reserve return")
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=g.MODE,job_id=job,output_directory=output,reserved_new_bytes=RESERVATION_BYTES,
        reservation=receipt["value"],created_at_utc=actual["created_at_utc"],expires_at_utc=actual["expires_at_utc"],
        budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD),
        completed_overlay_policy_ref=POLICY,storage_policy_selector_ref=SELECTOR,
        initial_ledger_pins=receipt["initial_ledger_pins"])
    actual["storage_ref"]=_save(directory,"storage.json",storage)
    input_ref=_save(directory,"input.json",actual)
    c.validate_input(input_ref) # stdlib source/metadata contract only, no consumer memo.
    c.require(not c.local(output).exists()and not c.local(auxiliary).exists(),"Parent alone creates full600 primary/AUX")
    argv=[str(c.ROOT/".runtime/env/bin/python"),"-B","-s","-m",HERE.replace("/",".")+".parent_v1",
        "--execute","--capability-ref",json.dumps(input_ref,sort_keys=True)]
    exec_allowed=_remaining_fits(actual,a["measured_runtime_resource_plan"])
    admitted_ref=_save(directory,"admitted.json",dict(schema="ROOT_TreeLoom_full600_admitted_v1",
        measured_remaining_window_sufficient_at_parent_exec=exec_allowed,
        input_ref=input_ref,reserve_receipt_ref=receipt_ref,ready_control_bundle_ref=bundle_ref,
        consumer_binding_count=len(receipt["value"]["consumer_bindings"]),baseline_role_count=len(state[3]),
        original_reserve_event_bytes=event_bytes,whole_primary_plus_AUX_reservation_bytes=RESERVATION_BYTES,
        original_apply_returned_successfully=True,separate_builder_outer_terminal_consumed=False,
        source_guard_or_owner_clock_extended=False,qualification_state_reused=False,
        execute_argv=argv,automatic_single_exec_explicitly_ROOT_authorized=True,
        all_scientific_execution_and_wait_evidence_still_pending=True))
    return dict(status="FULL600_ADMITTED_SINGLE_PARENT_EXEC_NEXT"if exec_allowed else"FULL600_ADMITTED_WINDOW_INSUFFICIENT_NO_LAUNCH",
        automatic_exec_allowed=exec_allowed,input_ref=input_ref,
        reserve_receipt_ref=receipt_ref,admitted_ref=admitted_ref,execute_argv=argv)

def continue_v1(request_ref):
    q,manifest=_request(request_ref);q["_request_ref"]=request_ref
    directory=c.local(HERE+"/ROOT_full600_records_"+q["instance"])
    c.require(not directory.exists(),"Fresh ROOT continuation consumes attempt before waiting, no replay")
    directory.mkdir(mode=0o700)
    _save(directory,"attempt.json",dict(schema="ROOT_TreeLoom_full600_continuation_attempt_v1",
        request_ref=request_ref,finite_wait_deadline=q["wait_deadline_at_utc"],
        automatic_retry=False,scientific_execution_started=False))
    try:qualified,data,launch,reason=_wait(q,directory)
    except Pending as exc:
        record=_save(directory,"pending_deadline.json",dict(status="PENDING_GENUINE_PREREQUISITES_DEADLINE",
            reason=str(exc)[:2000],admission=False,execution=False,no_automatic_retry=True))
        return dict(status="PENDING_GENUINE_PREREQUISITES_DEADLINE",record_ref=record),2
    if launch is None:
        ready_ref=_save(directory,"ready_data_no_admission.json",dict(status="READY_DATA_NO_ADMISSION",
            qualification_acceptance_ref=qualified["acceptance_ref"],all20_readiness_ref=data["readiness_ref"],
            all20_truth_handles_ref=data["truth_handles_ref"],reason=reason,admission=False,execution=False,
            owner_clock_extended=False,requires_new_explicit_finite_ROOT_launch_authority=True))
        return dict(status="READY_DATA_NO_ADMISSION",ready_ref=ready_ref,reason=reason),0
    try:result=_admit(q,manifest,directory,qualified,data,launch)
    except ReadyOnly as exc:
        record=_save(directory,"ready_data_no_admission.json",dict(status="READY_DATA_NO_ADMISSION",
            qualification_acceptance_ref=qualified["acceptance_ref"],all20_readiness_ref=data["readiness_ref"],
            all20_truth_handles_ref=data["truth_handles_ref"],reason=str(exc),admission=False,execution=False))
        return dict(status="READY_DATA_NO_ADMISSION",ready_ref=record,reason=str(exc)),0
    if result["automatic_exec_allowed"]is not True:return result,2
    # Mutation interpreter is replaced; parent gets a fresh read-only guard/memo
    # lifetime and remains the existing sole scientific supervisor and ROOT outer.
    print(json.dumps(result,sort_keys=True),flush=True)
    env=os.environ.copy();env.update(PYTHONPATH=str(c.ROOT)+":"+str(c.ROOT/"src")+":"+str(c.ROOT/"scripts"),
        PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    os.execve(result["execute_argv"][0],result["execute_argv"],env)
    raise RuntimeError("execve returned unexpectedly; consumed attempt must not retry")

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--continue-ref");a=p.parse_args(argv)
    if not a.continue_ref:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",wait=False,admission=False,execution=False)));return 0
    c.require(sys.platform.startswith("linux")and sys.dont_write_bytecode and sys.flags.no_user_site
        and sys.flags.optimize==0 and Path.cwd()==c.ROOT
        and not any(n in sys.modules for n in("torch","torch_geometric")),
        "Fresh ROOT Linux -B -s nonoptimized stdlib continuation interpreter")
    result,code=continue_v1(json.loads(a.continue_ref));print(json.dumps(result,sort_keys=True));return code
if __name__=="__main__":raise SystemExit(main())

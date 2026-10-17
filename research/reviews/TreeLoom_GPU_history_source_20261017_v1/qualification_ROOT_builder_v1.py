"""ROOT-only cold GPU-history/fast-preparation qualification admission; inert by default.

Use a fresh -B -s nonoptimized interpreter. Never install a consumer memo here.
Only ROOT may invoke --prepare-ref after inspecting this source. Any uncertain
attempt consumes its administrative namespace; no retries or automatic credit.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,os,re,stat,sys
from . import contract_v1 as c

BUILDER=c.HERE+"/qualification_ROOT_builder_v1.py"
TEMPLATE="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_input.INACTIVE.v1.json"
REFERENCE_MANIFEST=dict(path="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/source_manifest.v1.json",sha256="35760763a7f0608ed56cd62565ec555d4fcae7f622ea6bfb9dd5280257d20fa0")
REFERENCE_WORKER=dict(path="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/worker_v1.py",sha256="360b26b6ec131ab7e9f0b1f60bb20f2752307db30f6805136ea62ac870e94c76")
PREPARATION_REUSE_MANIFEST=dict(path="research/reviews/TreeLoom_preparation_reuse_source_20261017_v1/source_manifest.v1.json",sha256="bb5f6ef121dfbbe7f098653764fa70c83865de63b1aaa0930cc61fe7cdfacd41")
EXECUTION_EXTENSION="TreeLoom_GPU_history_v1"
COLD_PREPARATION="vectorized_original_recipes_no_cache_v1"
GPU_CAP_BYTES=24*2**30
RSS_CAP_BYTES=128*2**30
RESOURCE_CAP_POLICY="new_downward_failstops_from_reference64GiB_GPU384GiB_RSS_v1"
PREFIX="research/reviews/research_execution_ROOT_20261017_v34/"
def R(name,sha):return dict(path=PREFIX+name,sha256=sha)
RUNTIME=R("TreeLoom_qualification_actual_runtime.root_v1.json","1bc917b9a283490f58cbd964c000ade645547c8269f272b85b261120914ab848")
RUNTIME_QUALIFICATION=R("TreeLoom_qualification_actual_runtime_qualification.root_v1.json","398781fb9c6289af00227af2964be15f9bce31d3fa0edc4c1b7f0280fe8ff551")
OLD_OWNER_UNUSED=R("native_owner_v3_capability.root_active_v1.json","104d4f99d16327a73c96959cf7356f38fed8190e8b24145548e0fa6b14cb3a71")
OLD_READY_UNUSED=R("native_owner_v3_live_ready.root_v1.json","8b4351cf17f238747d8ba8b98a00419cf9faab2815f4461a09279211a015abbd")
TRUTH_HANDLE=R("TreeLoom_qualification_original_TRAIN000_truth_handle.root_v1.json","7c673afd3a2f6aa5fde4c76dd456900430c57dff17961995df0d0010b32bb8a1")
POLICY=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json",sha256="5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5")
SELECTOR=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json",sha256="fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")
REFERENCE_REVIEW=dict(path="research/reviews/frozen_TreeLoom_bounded_training_independent_20261017_v1/review.qualification.v1.json",sha256="ed77e7885a9b7403d778dc483ac2074e1b3f4bd4794eaa996f7488a8fcca001f")
ENDPOINT=dict(path="research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/empty_gpu0_controller_v1.py",sha256="dca4230a3d79540b947fa5488c6a49a885970db367b9955f78c343b8305b398f")
OWNER=READY=None
OWNER4_HERE='research/reviews/native_saved_MLM_successor_controller_ready_20261017_v4'
COORDINATOR=dict(path='research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/worker_entry_v1.py',sha256='2c4a976949fe147238ba775fa6aee58bbb782a9cef2c76d65945a472eae6be73')
FAILED_QUALIFICATION=dict(path='research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_records_root01_oct17/input.json',sha256='195799008a160489cd82930368b03e88c11d521f56fd003baef2ffc7f3cad538')
ADMIN_CAP=384*1024
EVENT_CAP=65536
EVENT_NODE=4096
WHOLE_RESERVATION=64*2**20

def _budget(directory,extra=0):
    total=4096;count=0
    for p in directory.iterdir():
        s=p.lstat()
        c.require(stat.S_ISREG(s.st_mode)and not p.is_symlink()and s.st_nlink==1
            and p.suffix==".json"and s.st_size<=65536,"Flat own bounded administrative JSON")
        total+=s.st_size+4096;count+=1
    c.require(count<32 and total+extra<=ADMIN_CAP,"Administrative records remain within384KiB")
    c.require(ADMIN_CAP+EVENT_CAP+EVENT_NODE<=524288,"Unchanged original512KiB event allowance")

def _save(directory,name,value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    c.require(len(raw)<=65536,"Each own compact ROOT JSON <=64KiB")
    _budget(directory,len(raw)+4096)
    path=directory/name
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,"wb")as stream:
        c.require(stream.write(raw)==len(raw),"Complete exclusive administrative write")
        stream.flush();os.fsync(stream.fileno())
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    return c.ref(str(path.relative_to(c.ROOT)))

def _interval(q,owner):
    c.active(q)
    start=datetime.fromisoformat(q["created_at_utc"]).timestamp()
    end=datetime.fromisoformat(q["expires_at_utc"]).timestamp()
    seconds=q["max_seconds"]
    c.require(type(seconds)is int and 300<seconds<=28800 and start+seconds<=end
        and end<=owner["expires_unix"],"Finite <=8h qualification entirely inside real owner authority")
    return end

def _request(request_ref):
    global OWNER,READY
    q=c.read(request_ref);c.active(q)
    c.require(q["schema"]=="ROOT_TreeLoom_GPU_history_cold_qualification_admission_request_v1"
        and q["source_manifest_ref"]==c.ref(c.MANIFEST)
        and q["reference_scientific_manifest_ref"]==REFERENCE_MANIFEST
        and q["reference_scientific_review_ref"]==REFERENCE_REVIEW
        and q["preparation_reuse_source_manifest_ref"]==PREPARATION_REUSE_MANIFEST
        and q["runtime_ref"]==RUNTIME and q["runtime_qualification_ref"]==RUNTIME_QUALIFICATION
        and q["train_truth_handle_ref"]==TRUTH_HANDLE and q["storage_policy_selector_ref"]==SELECTOR
        and q["accounting_coordinator_source_ref"]==COORDINATOR
        and q["owner4_service_manifest_ref"]==c.ref(OWNER4_HERE+"/source_manifest.v1.json")
        and q["execution_extension"]==EXECUTION_EXTENSION
        and q["history_holds_per_window"]==128 and type(q["history_holds_per_window"])is int
        and q["profile_interval_seconds"]==5 and type(q["profile_interval_seconds"])is int
        and q["cold_preparation"]==COLD_PREPARATION
        and q["resource_cap_policy"]==RESOURCE_CAP_POLICY
        and q["preparation_cache_plan_ref"]is None and q["preparation_cold_qualification"]is True
        and q["all_other_GPU0_borrower_routes_revoked"]is True
        and q["preserve_controller_until_owned_drain"]is True and q["automatic_retry_or_resume"]is False,
        "Exact fresh ROOT cold fast-preparation/GPU-history/profiler request and immutable original condition")
    OWNER=q["controller_capability_ref"];READY=q["controller_ready_ref"]
    owner=c.read(OWNER);ready=c.read(READY);c.active(owner);_interval(q,owner)
    c.require(type(q["instance"])is str and re.fullmatch("[a-z][a-z0-9_]{0,39}",q["instance"])is not None,"Fresh bounded ROOT instance")
    m=c.read(q["source_manifest_ref"]);review=c.read(q["source_review_ref"])
    c.require(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"
        and review.get("blockers")==[] and review.get("candidate_manifest_ref")==q["source_manifest_ref"],
        "Independent whole new source closure PASS including concrete cold ROOT builder")
    reviewed=review.get("reviewed_sources",[])
    for path in(BUILDER,c.SOURCE,c.HERE+"/parent_v1.py",c.HERE+"/contract_v1.py"):
        record=c.ref(path)
        c.require(record in reviewed and m["source_sha256"].get(path)==record["sha256"],
            "Actual reviewed executed builder/parent/worker/contract source refs, no stale old-parent qualification")
    for path,sha in m["source_sha256"].items():c.raw(dict(path=path,sha256=sha))
    original=c.read(REFERENCE_MANIFEST);original_review=c.read(REFERENCE_REVIEW)
    c.require(original_review["independent"]is True and original_review["verdict"]=="SOURCE_PASS"
        and original_review["blockers"]==[] and original_review["candidate_manifest_ref"]==REFERENCE_MANIFEST,
        "Original scientific-condition review remains distinct from new operational source review")
    for path,sha in original["source_sha256"].items():c.raw(dict(path=path,sha256=sha))
    # Immutable observed-preparation dependencies are authenticated, never activated or produced.
    previous=c.read(PREPARATION_REUSE_MANIFEST)
    for record in previous["own_refs"]:
        if record["path"].endswith(".py"):c.raw(record)
    om=c.read(q["owner4_service_manifest_ref"]);oreview=c.read(owner["source_review_ref"])
    c.require(owner["source_manifest_ref"]==q["owner4_service_manifest_ref"]
        and owner["source_refs"].get(ENDPOINT["path"])==ENDPOINT
        and owner["source_refs"].get(OWNER4_HERE+"/empty_gpu0_controller_v1.py")==c.ref(OWNER4_HERE+"/empty_gpu0_controller_v1.py")
        and oreview["independent"]is True and oreview["verdict"]=="SOURCE_PASS"and oreview["blockers"]==[]
        and oreview["candidate_manifest_ref"]==owner["source_manifest_ref"],
        "Independent reviewed owner4 service actually executes unchanged authenticated v3 protocol")
    for path,sha in om["source_sha256"].items():c.raw(dict(path=path,sha256=sha))
    h=c.read(owner["handover_ref"])
    c.require(h["failed_TreeLoom_qualification_input_ref"]==FAILED_QUALIFICATION
        and h["failed_TreeLoom_qualification_worker_started"]is False
        and h["failed_TreeLoom_qualification_CUDA_context_created"]is False
        and h["failed_TreeLoom_qualification_namespace_replay_authorized"]is False,
        "Actual failed old qualification remains distinct and cannot replay")
    c.raw(h["failed_TreeLoom_qualification_actual_outer_ref"])
    runtime=c.read(RUNTIME);qualification=c.read(RUNTIME_QUALIFICATION)
    c.require(runtime["source_ref"]==REFERENCE_WORKER and runtime["CUDA_VISIBLE_DEVICES"]==c.GPU0_UUID
        and runtime["python"]==qualification["python"]and qualification["runtime_ref"]==RUNTIME
        and qualification["ROOT_qualified_existing_Torch_PyG_runtime"]is True
        and qualification["CUDA_context_created"]is False and qualification["cpu_affinity"]==list(range(176,184)),
        "Exact existing qualified interpreter/CPU affinity; source rebinding adds no environment or CUDA probe")
    status=ready["actual_status"];identity=status["controller"]
    c.require(ready["root_authorizer"]=="root_agent"and ready["actual_status_terminal_consumed"]is True
        and type(ready["actual_status_tool_exit_code"])is int and ready["actual_status_tool_exit_code"]==0
        and ready["owner_parent_still_live"]is True and ready["controller_capability_ref"]==OWNER
        and ready["source_ref"]==c.ref(OWNER4_HERE+"/empty_gpu0_controller_v1.py")
        and type(ready["controller_pid"])is int and ready["controller_pid"]>0
        and type(ready["controller_start_ticks"])is int and ready["controller_start_ticks"]>0
        and ready["controller_pid"]==identity["pid"]and ready["controller_start_ticks"]==identity["start_ticks"]
        and status["capability_ref"]==OWNER and status["physical_gpu"]==0
        and status["lifecycle"]=="running"and status["jobs"]==[]and status["safety_error"]is None
        and status["ambiguous_descendants_or_lease_holders"]==[]
        and status["owner_service_revision"]==4 and status["periodic_registry_writes"]is False
        and status["periodic_accounting_transactions"]is False
        and owner["jobs"]=={}and owner["protected_envelope_bytes"]==0
        and owner["storage_policy_selector_ref"]==SELECTOR and owner["capacity_policy_ref"]==POLICY,
        "Genuine consumed owner4 status/ready; parent still revalidates live capacity and title")
    handle=c.read(TRUTH_HANDLE);truth=handle["private_truth_ref"]
    c.require(handle["root_authorizer"]=="root_agent"and handle["original_whole30Mb_parent"]is True
        and handle["truth_payload_opened"]is False and set(truth)=={"path","sha256"}
        and truth["path"]=="experiments/results/American_C3_full30mb_20261015_v1/truth/c3_pulse_ancestry.npz"
        and re.fullmatch("[0-9a-f]{64}",truth["sha256"])is not None,"Same genuine TRAIN000 truth handle only; no payload read")
    c.raw(q["accounting_batch_input_ref"])
    return q,owner,ready,m,qualification,truth

def _controls(q,owner,ready):
    common=dict(root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"])
    binding=dict(common,schema="native_VAL000_downstream_GPU0_controller_binding_v1",
        source_review_passed=True,endpoint_source_ref=ENDPOINT,physical_gpu=0,GPU_UUID=c.GPU0_UUID,
        additional_controller_and_other_borrower_submissions_revoked=True,
        controller_preserved_until_downstream_drain=True,independent_source_review_ref=owner["source_review_ref"],
        route="reviewed_same_protocol_successor",protected_job=None,protected_envelope_bytes=0,
        successor_capability_explicitly_ROOT_authorized=True,controller_pid=ready["controller_pid"],
        controller_start_ticks=ready["controller_start_ticks"],controller_capability_ref=OWNER,
        original_live_ready_evidence_ref=READY,owner4_service_manifest_ref=owner["source_manifest_ref"],owner4_service_source_ref=c.ref(OWNER4_HERE+"/empty_gpu0_controller_v1.py"))
    authority=dict(common,schema="native_VAL000_downstream_direct_attachment_authority_v1",
        borrowed_physical_inode_attachment_authorized=True,
        additional_old_controller_submissions_revoked_while_downstream_attached=True,
        old_controller_preserved_until_downstream_drained=True,
        controller_pid=ready["controller_pid"],controller_start_ticks=ready["controller_start_ticks"],
        controller_capability_ref=OWNER,controller_binding_ref=None,downstream_job_id=None,
        protected_envelope_bytes=0,downstream_max_cuda_reserved_bytes=GPU_CAP_BYTES,
        downstream_non_Torch_margin_bytes=2*2**30,headroom_bytes=4*2**30)
    return binding,authority

def _runtime_source_binding_v1(admin,qualification):
    """Exact existing runtime evidence with truthful new executed-source binding only."""
    runtime=copy.deepcopy(c.read(RUNTIME))
    runtime.update(source_ref=c.ref(c.SOURCE),original_runtime_ref=RUNTIME,
        original_runtime_qualification_ref=RUNTIME_QUALIFICATION,
        metadata_source_binding_only=True,new_runtime_probe_performed=False)
    runtime_ref=_save(admin,"runtime_source_binding.json",runtime)
    qualified=copy.deepcopy(qualification)
    qualified.update(runtime_ref=runtime_ref,original_runtime_ref=RUNTIME,
        original_runtime_qualification_ref=RUNTIME_QUALIFICATION,
        metadata_source_binding_only=True,new_runtime_probe_performed=False)
    qualification_ref=_save(admin,"runtime_qualification_source_binding.json",qualified)
    return runtime_ref,qualification_ref

def _actual_input_v1(q,qualification,truth,*,output,auxiliary,job,request_ref,binding_ref,
        attachment_ref,runtime_ref,runtime_qualification_ref,drain_expires_unix):
    """Pure assembly apart from reading the exact frozen scientific input template."""
    actual=c.read(c.ref(TEMPLATE))
    actual.update(root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        drain_expires_unix=drain_expires_unix,job_id=job,
        output_directory=output,worker_auxiliary_scope=auxiliary,ROOT_record_directory=auxiliary,
        source_manifest_ref=q["source_manifest_ref"],source_review_ref=q["source_review_ref"],
        runtime_ref=runtime_ref,runtime_qualification_ref=runtime_qualification_ref,
        cpu_affinity=qualification["cpu_affinity"],storage_policy_selector_ref=SELECTOR,
        controller_binding_ref=binding_ref,direct_attachment_authority_ref=attachment_ref,
        ROOT_admission_request_ref=request_ref,execution_extension=EXECUTION_EXTENSION,
        history_holds_per_window=128,profile_interval_seconds=5,cold_preparation=COLD_PREPARATION,
        preparation_cache_plan_ref=None,preparation_cold_qualification=True,
        reference_scientific_manifest_ref=REFERENCE_MANIFEST,reference_scientific_review_ref=REFERENCE_REVIEW,
        preparation_reuse_source_manifest_ref=PREPARATION_REUSE_MANIFEST,
        resource_cap_policy=RESOURCE_CAP_POLICY,
        reference_original_resource_caps=dict(max_cuda_reserved_bytes=64*2**30,max_cpu_rss_bytes=384*2**30),
        new_downward_resource_failstops=True,full_work_or_focal_count_reduction_authorized=False)
    actual["caps"]["max_seconds"]=q["max_seconds"]
    actual["caps"]["max_cuda_reserved_bytes"]=GPU_CAP_BYTES
    actual["caps"]["max_cpu_rss_bytes"]=RSS_CAP_BYTES
    actual["train_truth_refs"]={"american_c3_train_000":truth}
    c.require(actual["phase"]=="qualification"and list(actual["parents"])==["american_c3_train_000"],
        "Disposable complete original30Mb TRAIN000/exact10000focals qualification, no cache or full training")
    return actual

def _direct_bindings_v1(q,actual,request_ref,binding_ref,attachment_ref,runtime_ref,runtime_qualification_ref):
    # Full new source closure is authenticated by the real new manifest/review.
    # Original scientific condition and immutable observed-preparation provenance stay explicit.
    return [c.ref(c.SOURCE),c.GUARD,q["source_manifest_ref"],q["source_review_ref"],actual["config_ref"],
        runtime_ref,runtime_qualification_ref,request_ref,binding_ref,attachment_ref,
        c.ref(BUILDER),OWNER,TRUTH_HANDLE,COORDINATOR,q["accounting_batch_input_ref"],
        REFERENCE_MANIFEST,REFERENCE_REVIEW,PREPARATION_REUSE_MANIFEST]

def prepare_v1(request_ref):
    q,owner,ready,manifest,qualification,truth=_request(request_ref)
    instance=q["instance"];suffix="qualification_GPU_history_"+instance
    admin=c.local(c.HERE+"/ROOT_qualification_owner4_records_"+instance)
    output="experiments/results/TreeLoom_frozen_bounded_"+suffix
    auxiliary=".runtime/tmp/TreeLoom_frozen_bounded_"+suffix
    c.require(not admin.exists()and not c.local(output).exists()and not c.local(auxiliary).exists(),
        "Fresh admin attempt and absent primary/AUX; consumed namespace cannot retry")
    # Consume the immutable attempt before importing the actual guard or reading state.
    admin.mkdir(mode=0o700)
    _save(admin,"attempt.json",dict(schema="ROOT_TreeLoom_qualification_attempt_v1",request_ref=request_ref,
        no_retry=True,scientific_execution_started=False))
    binding,authority=_controls(q,owner,ready)
    binding_ref=_save(admin,"controller_binding.json",binding)
    job="TreeLoom-frozen-bounded-"+suffix
    authority.update(controller_binding_ref=binding_ref,downstream_job_id=job)
    attachment_ref=_save(admin,"direct_attachment_authority.json",authority)
    runtime_ref,runtime_qualification_ref=_runtime_source_binding_v1(admin,qualification)
    actual=_actual_input_v1(q,qualification,truth,output=output,auxiliary=auxiliary,
        job=job,request_ref=request_ref,binding_ref=binding_ref,attachment_ref=attachment_ref,
        runtime_ref=runtime_ref,runtime_qualification_ref=runtime_qualification_ref,
        drain_expires_unix=_interval(q,owner))
    # ROOT mutation singleton: original selector configured before any validation;
    # no contract.guard_v1, consumer memo, scientific import, process/GPU probe or launch.
    coordinator=c.load(COORDINATOR,"_TreeLoom_owner4_ROOT_accounting_transaction")
    entered=False
    try:
        with coordinator.accounting_transaction(q["accounting_batch_input_ref"]):
            entered=True;c.active(q)
            g=c.load(c.GUARD,"_TreeLoom_ROOT_mutation_Census")
            c.require(g.load_selected_guard_v1(SELECTOR)is g and g.MODE=="Census_observation_two_scope_retention_v1"
                and g.MAX_JSON==EVENT_CAP and g.META_CHARGE==524288,"Exact unchanged current Census mutation route")
            c.require(all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__ for n in("canonical","file_bytes","mutex")),
                "Fresh unwrapped ROOT mutation interpreter, never consumer memo")
            policy=c.read(POLICY);implementation=policy["implementation_ref"];launcher=policy["launcher_ref"]
            c.raw(implementation);c.raw(launcher)
            reservation=dict(key="H056/"+job,output=str(c.local(output)),
                auxiliary_scopes=[str(c.local(auxiliary))],reserved_bytes=WHOLE_RESERVATION)
            direct=_direct_bindings_v1(q,actual,request_ref,binding_ref,attachment_ref,
                runtime_ref,runtime_qualification_ref)
            for record in direct:c.raw(record)
            with g.budget.v4.mutex(c.ROOT):
                state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
                reservation["consumer_bindings"]=g.consumer_bindings_v1(list(state[3].values()),direct)
                g.budget.v4.scope_spec(c.ROOT,reservation)
                auth=g.operation_draft_v1("reserve",reservation,expected_before=g._pins(state[2]),bindings=state[3])
                auth.update(active=True,expires_at_utc=q["expires_at_utc"])
                auth_ref=_save(admin,"reserve_authority.json",auth)
                event=dict(schema="Census_observation_exact_retention_event_v1",sequence=len(state[4])+1,
                    previous_sha256=state[5][-1]if state[5]else None,bindings=state[3],
                    operation="reserve",authority=auth_ref,value=dict(reservation,root_admission=auth_ref))
                event_bytes=len(g.budget.v4.encoded(event)+b"\n")
                c.require(event_bytes<=EVENT_CAP and len(reservation["consumer_bindings"])<=64,
                    "Complete actual baseline/direct union, original full scope and serialized event preflight")
            receipt=g.apply_reserve_v1(auth_ref,expected_source_sha256=implementation["sha256"],
                expected_launcher_sha256=launcher["sha256"])
    except ValueError as exc:
        if not entered and str(exc)=="Finite shared ROOT accounting coordinator wait":
            r=_save(admin,"ready_no_admission.json",dict(status="READY_NO_ADMISSION_COORDINATOR_TIMEOUT",admission_started=False,automatic_retry=False))
            return dict(status="READY_NO_ADMISSION_COORDINATOR_TIMEOUT",record_ref=r)
        raise
    receipt_ref=_save(admin,"reserve_receipt.json",receipt)
    own=receipt["value"]
    c.require(own==dict(reservation,root_admission=auth_ref),"Genuine unchanged original reserve result")
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=g.MODE,job_id=job,output_directory=output,reserved_new_bytes=WHOLE_RESERVATION,
        reservation=own,created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD),
        completed_overlay_policy_ref=POLICY,storage_policy_selector_ref=SELECTOR,
        initial_ledger_pins=receipt["initial_ledger_pins"])
    actual["storage_ref"]=_save(admin,"storage.json",storage)
    input_ref=_save(admin,"input.json",actual)
    c.validate_input(input_ref) # stdlib source/input contract only; never c.guard_v1.
    c.require(not c.local(output).exists()and not c.local(auxiliary).exists(),
        "Admission creates no scientific primary/AUX; parent owns exclusive creation")
    argv=[".runtime/env/bin/python","-B","-s","-m",c.HERE.replace("/",".")+".parent_v1",
        "--execute","--capability-ref",json.dumps(input_ref,sort_keys=True)]
    admitted_ref=_save(admin,"admitted.json",dict(schema="ROOT_TreeLoom_qualification_admitted_v1",
        request_ref=request_ref,input_ref=input_ref,reserve_receipt_ref=receipt_ref,
        whole_primary_plus_AUX_reservation_bytes=WHOLE_RESERVATION,
        consumer_binding_count=len(own["consumer_bindings"]),baseline_role_count=len(state[3]),
        baseline_unique_count=len({json.dumps(v,sort_keys=True)for v in state[3].values()}),
        original_serialized_reserve_event_bytes=event_bytes,
        execute_argv=argv,execute_environment={"PYTHONPATH":".:src:scripts"},
        execution_extension=EXECUTION_EXTENSION,cold_preparation=COLD_PREPARATION,
        history_holds_per_window=128,profile_interval_seconds=5,
        max_cuda_reserved_bytes=GPU_CAP_BYTES,max_cpu_rss_bytes=RSS_CAP_BYTES,
        resource_cap_policy=RESOURCE_CAP_POLICY,new_downward_resource_failstops=True,
        reference_original_resource_caps=dict(max_cuda_reserved_bytes=64*2**30,max_cpu_rss_bytes=384*2**30),
        preparation_cache_plan_ref=None,actual_new_parent_and_new_worker_bound=True,
        scientific_execution_started=False,measured_fit=False,automatic_launch=False,
        genuine_ROOT_builder_outer0_required_before_single_parent_launch=True,
        ROOT_administrative_metadata_cap_bytes=ADMIN_CAP,original_event_JSON_cap_bytes=EVENT_CAP,
        unchanged_original_event_allowance_bytes=524288))
    return dict(status="QUALIFICATION_ADMITTED_ONLY",input_ref=input_ref,reserve_receipt_ref=receipt_ref,
        admitted_ref=admitted_ref,execute_argv=argv,execute_environment={"PYTHONPATH":".:src:scripts"},
        scientific_execution_started=False,automatic_launch=False,automatic_close_or_credit=False)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--prepare-ref")
    args=parser.parse_args(argv)
    if not args.prepare_ref:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
    c.require(sys.platform.startswith("linux")and sys.dont_write_bytecode and sys.flags.no_user_site
        and sys.flags.optimize==0 and Path.cwd()==c.ROOT
        and not any(n in sys.modules for n in("torch","torch_geometric")),
        "Fresh ROOT Linux -B -s nonoptimized stdlib administrative interpreter")
    print(json.dumps(prepare_v1(json.loads(args.prepare_ref)),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

"""Execution-only ROOT source bridge over immutable fully charged native v2 roots.
The original guard authenticates its original bound input/source. A separate exact
ROOT amendment authenticates this new adapter. No reserve/close/credit entrypoint.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,os,stat,sys,time
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2.common_v1 import *
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import common_v1 as original
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v11"
SOURCE=HERE+"/amendment_contract_v1.py"
ACTUAL_DRIVER=HERE+"/execution_only_v1.py"
DIRECT=HERE+"/direct_gpu0_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
ORIGINAL_INPUT={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/input.json","sha256":"9550f5b0a33b82b329e483796683d400e7053cbd9bb84319c9b6e00d883436e5"}
PRIOR_WAIT={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/actual_wait.json","sha256":"612ad2070a9a49283e972c2ff151e6383e1b700ba11a0a5aef54c8ee84e07a4c"}
OUTER_TERMINAL={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_preparation_20261016_v2/actual_execution_tool_terminal.root_v1.json","sha256":"014782455fb6963a3aa0d073c8cf9ed730daade57b0b9553643a91d546c5b4a9"}
ORIGINAL_SOURCE_HANDOFF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/source_handoff.v2.json","sha256":"6b597779e144710f64b46f47b5aef09022c4d7e36cbb41162e1f68cbb389ffb0"}
CONTROLLER_ENDPOINT_SOURCE_REF={"path":"research/reviews/C3_native_worker_module_route_fix_source_20261016_v3/gpu0_declared_pool_v1.py","sha256":"e98dc3f5b0c1aa969f9f640bc1318f538686ac9fdf2543be60fe9733ad419818"}
CONTROLLER_CAPABILITY_REF={"path":"research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json","sha256":"4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870"}
LATEST_METADATA_WAIT={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/same_charge_execution_amendment_v1/actual_wait.json","sha256":"a7341fb3d54e74bfed4cd66db15b7c12d901df193bbbfadc7b8c5b3d9cbed50d"}
LATEST_METADATA_OUTER={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/actual_same_charge_v2_tool_terminal.root_v1.json","sha256":"13a4359908f2a7ebd432f027fae625a26fc9675d5008b2b68ca9cad19122de3a"}
LATEST_ROSTER_WAIT={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/same_charge_metadata_bridge_v3/actual_wait.json","sha256":"6d3c18831e9386b0d0dd605b73e53fc16087040dd05ff75be96406738807c43a"}
LATEST_ROSTER_OUTER={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/actual_same_charge_metadata_bridge_v3_tool_terminal.root_v1.json","sha256":"d68eac7f677bceb1ac4229c6755046be3ce5a1aaf93d3eabea2509294c2ba0f0"}
OWNED_ROSTER_CAP=4096
MINIMAL_WAIT_CAP_BYTES=65536
TERMINAL_RECEIPT_CAP_BYTES=8*2**20
OWNED_ROSTER_POLICY_REF={'path': 'research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4/owned_roster_policy.INACTIVE.v4.json', 'sha256': 'ea1ddb59ab4248b8ba95dbbe77d25db30636cc318a4fdaebde1e1ae6e5a7344f'}
CHILD="same_charge_full_state_resume_v11"
AMENDMENT_REF=None
ORIGINAL_STATE=None
RESUME_STATE=None
V4_STDOUT_PROJECTION_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_owned_roster_v4_actual_stdout_required_fields.root_v1.json', 'sha256': 'af114c2f0bcaef24c202ca9c0fafcd4b3a575fa68fd6575c9b4911a34c79cfdb'}
V4_OUTER_PROJECTION_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_owned_roster_v4_consumed_outer_projection.root_v1.json', 'sha256': '705dfe54bfd2afe77e39c096a256729e09317224e231749ad194e416a4451e4e'}
V4_LAUNCH_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/same_charge_owned_roster_v4/launch.json', 'sha256': '2f86420dfec54338e08862750b59c9cc080d736efab214bf19034c33174e9564'}
V4_CUDA_OBSERVATION_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_owned_roster_v4_CUDA_observation.root_v1.json', 'sha256': '95a7e00cc13990433905cc83b0e97484603169cb6c0029f9599324bf995b7f84'}
V4_PRIOR_TARGET_REGISTRY_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_owned_roster_v4_preparation.root_actual_v1.json', 'sha256': '394e9ca41f209d1d53a401bbba98fd2367bacf61ee56e4eefd7a0b4e44970fa7'}
RESUME_CHECKPOINT_REF={'path': 'experiments/results/native_VAL000_full30mb_borrowed32_GPU0_v2/same_charge_owned_roster_v4/checkpoints/ep0-ba6500-rank0.pt', 'sha256': '17e266f77b9a91f6a72fcf50a4bb6551e794da86fdbe8ca382da7fccb180bce0'}
ROOT_PRESERVED_METADATA_REF={'path': 'research/reviews/native_v4_preserved_checkpoint_resume_root_metadata_20261016_v1/preserved_checkpoint_metadata.root_v2.json', 'sha256': '15a68643d5d940ee3b6304bc691e8aede1ac223d68b9a61581633805c0f3c7da'}
V4_MANIFEST_REF={'path': 'research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4/source_manifest.v1.json', 'sha256': '4fc7f605f36d18b82ed155c24b0095a83b4b8b3a9649d8d65f81b23bc8cfae37'}
RESUME_POLICY_REF={'path': 'research/reviews/native_VAL000_full_state_resume_source_20261016_v11/resume_policy.INACTIVE.v5.json', 'sha256': '142f457431ed58d8f8e6b133ca2b92b2b0b1bc2a356aeb9d459fbc2a502d846c'}
ORIGINAL_CAPS=copy.deepcopy(CAPS)
CAPS=dict(CAPS,reservation_bytes=5*2**30,scientific_output_bytes=CAPS["scientific_output_bytes"]-2*2**30)
DRIVER=ACTUAL_DRIVER
FRESH_GUARD_REF={"path":"research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py","sha256":"60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c"}
FRESH_POLICY_REF={"path":"research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json","sha256":"72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417"}
FRESH_STATE=None
ACCOUNTING_CHOICES_V11=(5*2**30,17*2**28)
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
LIVENESS_COMPONENT_MANIFEST_REF={'path': 'research/reviews/native_VAL000_terminal_controller_liveness_component_20261016_v11/source_manifest.v11.json', 'sha256': 'b629111f9edcef050e2276399e5bc29a613dc49ded874428d968aa077a634081'}
LIVENESS_COMPONENT_AUTHORITY_AST_SHA256='a481daf147c78599566d9ba7c0b75a5294bfbfc65b3b204c48be2947d9c681a0'
_COMPONENT_CHECKED_V11=False


def set_amendment_v1(record):
    global AMENDMENT_REF
    require(AMENDMENT_REF is None or AMENDMENT_REF==record,"One exact ROOT execution amendment per interpreter")
    AMENDMENT_REF=record

def current_amendment_ref_v1():
    require(AMENDMENT_REF is not None,"Actual ROOT amendment installed")
    return AMENDMENT_REF

def active_amendment_v1():
    require(AMENDMENT_REF is not None,"Actual separate ROOT amendment ref required")
    a=read(AMENDMENT_REF);active(a)
    require(a.get("template_only") is not True,"Inactive templates never authorize execution")
    require(a.get("schema")=="native32_verified22000_fresh5GiB_ROOT_amendment_v11"
        and a.get("root_authorizer")=="root_agent" and a.get("status")=="active"
        and a.get("execution_authorized") is True and a.get("independent_source_review_passed") is True
        and a.get("adapter_manifest_ref")==ref(MANIFEST) and a.get("original_input_ref")==ORIGINAL_INPUT
        and a.get("prior_actual_wait_ref")==PRIOR_WAIT and a.get("prior_outer_terminal_ref")==OUTER_TERMINAL
        and a.get("latest_metadata_failure_wait_ref")==LATEST_METADATA_WAIT
        and a.get("latest_metadata_failure_outer_terminal_ref")==LATEST_METADATA_OUTER
        and a.get("latest_roster_failure_wait_ref")==LATEST_ROSTER_WAIT
        and a.get("latest_roster_failure_outer_terminal_ref")==LATEST_ROSTER_OUTER
        and a.get("owned_roster_policy_ref")==OWNED_ROSTER_POLICY_REF
        and a.get("cumulative_owned_identity_cap")==OWNED_ROSTER_CAP
        and a.get("terminal_receipt_cap_bytes")==TERMINAL_RECEIPT_CAP_BYTES
        and a.get("v3_metadata_bridge_future_execution_revoked") is True
        and a.get("metadata_bridge_only_missing_operational_field") is True
        and a.get("metadata_bridge_resets_or_rewrites_preparation") is False
        and a.get("v2_same_charge_future_execution_revoked") is True
        and a.get("one_exact_descendant_attempt_authorized") is True
        and a.get("original_native_v2_future_execution_revoked") is True
        and a.get("original_reservation_authenticates_new_adapter") is False
        and a.get("separate_ROOT_amendment_authenticates_new_executed_sources") is True
        and a.get("new_reservation_or_close_or_credit_authorized") is False
        and a.get("deadline_reset_or_automatic_retry_authorized") is False
        and a.get("failed_artifacts_and_full_original7GiB_charge_retained") is True
        and a.get("fresh_child_name")==CHILD and a.get("max_seconds")==172800
        and a.get("expires_at_utc")==original.ORIGINAL_ABSOLUTE_EXPIRY,"Exact explicit ROOT same-charge source bridge")
    roster_policy=read(OWNED_ROSTER_POLICY_REF)
    require(roster_policy["schema"]=="native_VAL000_finite_cumulative_owned_roster_policy_v4"
        and roster_policy["cumulative_identity_ceiling"]==OWNED_ROSTER_CAP==4096
        and roster_policy["adopted_wait_ceiling"]==roster_policy["cleanup_signal_retention_ceiling"]==OWNED_ROSTER_CAP
        and roster_policy["group_census_retention_ceiling"]==roster_policy["complete_ownership_evidence_ceiling"]==OWNED_ROSTER_CAP
        and roster_policy["minimal_wait_receipt_ceiling_bytes"]==MINIMAL_WAIT_CAP_BYTES
        and roster_policy["full_terminal_receipt_ceiling_bytes"]==TERMINAL_RECEIPT_CAP_BYTES==8*2**20
        and roster_policy["inherited_save_ceiling_bytes"]==8*2**20
        and roster_policy["cleanup_seconds"]==CAPS["cleanup_seconds"]==30
        and roster_policy["complete_roster_required_for_complete_drain"] is True
        and roster_policy["overflow_or_truncation_authorizes_success"] is False,
        "Exact finite4096 identity/8MiB terminal policy, original30s cleanup and fully charged AUX")
    manifest=read(a["adapter_manifest_ref"])
    expected=[ref(HERE+"/"+name) for name in ("__init__.py","amendment_contract_v1.py","direct_gpu0_v1.py","execution_only_v1.py","corpus_metadata_bridge_v1.py","resume_scientific_bridge_v5.py","alias_namespace_v7.py","fresh_entry_v9.py")]
    require(manifest["new_executed_source_refs"]==expected
        and manifest["original_source_handoff_ref"]==ORIGINAL_SOURCE_HANDOFF
        and manifest["original_source_manifest_ref"]==original.ref(original.MANIFEST),
        "Every actual new executed source and unchanged original source closure authenticated separately")
    for r in manifest["new_executed_source_refs"]:raw(r)
    require(manifest["owned_roster_policy_ref"]==OWNED_ROSTER_POLICY_REF,"Exact separately authenticated operational roster policy")
    require(manifest["installed_controller_endpoint_source_ref"]==CONTROLLER_ENDPOINT_SOURCE_REF,
        "Exact actual installed Declared0Controller endpoint source bound by amendment")
    raw(CONTROLLER_ENDPOINT_SOURCE_REF)
    controller_cap=read(CONTROLLER_CAPABILITY_REF)
    require(controller_cap["source_refs"].get(CONTROLLER_ENDPOINT_SOURCE_REF["path"])==CONTROLLER_ENDPOINT_SOURCE_REF,
        "Endpoint implementation belongs to exact original controller capability")
    review=read(a["independent_source_review_ref"])
    candidate=review.get("candidate_manifest_ref",review.get("candidate_source_manifest_ref",review.get("source_manifest_ref")))
    require(candidate==ref(MANIFEST) and review.get("verdict")=="PASS_SOURCE_ONLY","Independent exact-manifest source review required")
    for record in manifest["source_authentication_refs"]:raw(record)
    original_s=read(ORIGINAL_INPUT);q=read(original_s["root_stage_request_ref"])
    anchor=datetime.fromisoformat(q["created_at_utc"]).timestamp()
    require(a.get("original48h_window_anchor_utc")==q["created_at_utc"]
        and time.time()<min(anchor+172800,datetime.fromisoformat(original.ORIGINAL_ABSOLUTE_EXPIRY).timestamp()),
        "Original48hour window anchored before first attempt, never reset")
    validate_alias_authority_v7(a)
    validate_v9_failure_lineage_v10(a)
    terminal_snapshot_authority_v11(a)
    validate_liveness_component_v11(manifest)
    return a,original_s,q

def work_deadline_monotonic_v1():
    _,_,q=active_amendment_v1()
    end=min(datetime.fromisoformat(q["created_at_utc"]).timestamp()+172800,
        datetime.fromisoformat(original.ORIGINAL_ABSOLUTE_EXPIRY).timestamp())
    return time.monotonic()+end-time.time()-CAPS["work_cutoff_margin_seconds"]

def validate_input_v1(record):
    global ORIGINAL_STATE
    require(record==ORIGINAL_INPUT,"Original immutable input, with explicit separate execution amendment")
    a,original_s,original_q=active_amendment_v1()
    s,q,cfg,storage,protocol=original.validate_input_v1(record)
    require(s==original_s and q==original_q and storage["reserved_new_bytes"]==7*2**30,
        "Original immutable admitted input/request/storage and full7GiB charge")
    wait=read(PRIOR_WAIT);outer=read(OUTER_TERMINAL)
    require(wait.get("returncode")==-15 and wait.get("actual_OS_wait_status")==15
        and wait.get("actual_direct_worker_wait") is True and wait.get("all_owned_processes_and_groups_drained") is True
        and wait.get("completed_ref") is None and wait.get("no_future_stage_authorized") is True
        and outer.get("actual_exit_code")==241 and outer.get("actual_terminal_consumed") is True
        and outer.get("actual_tool_session_id")==26943 and outer.get("actual_wait_ref")==PRIOR_WAIT
        and outer.get("actual_direct_worker_returncode")==-15
        and outer.get("all_owned_processes_and_groups_drained") is True
        and outer.get("full_original7GiB_charge_retained") is True
        and outer.get("native_CUDA_context_observed_MiB")==418 and outer.get("optimizer_updates")=="UNVERIFIED",
        "Genuine prior failed outer terminal, native418MiB context and actual worker drain, never success")
    latest_wait=read(LATEST_METADATA_WAIT);latest_outer=read(LATEST_METADATA_OUTER)
    require(latest_wait.get("returncode")==1 and latest_wait.get("actual_OS_wait_status")==256
        and latest_wait.get("actual_direct_worker_wait") is True
        and latest_wait.get("all_owned_processes_and_groups_drained") is True
        and latest_wait.get("completed_ref") is None and latest_wait.get("no_future_stage_authorized") is True
        and latest_outer.get("actual_exit_code")==1 and latest_outer.get("actual_terminal_consumed") is True
        and latest_outer.get("actual_tool_session_id")==62408
        and latest_outer.get("actual_wait_ref")==LATEST_METADATA_WAIT
        and latest_outer.get("actual_direct_worker_returncode")==1
        and latest_outer.get("all_owned_processes_and_groups_drained") is True
        and latest_outer.get("full_original7GiB_charge_retained") is True
        and latest_outer.get("native_CUDA_context_observed_MiB")==0
        and latest_outer.get("optimizer_updates")=="UNVERIFIED",
        "Genuine latest metadata failure/drain, no observed CUDA or verified optimizer, never scientific success")
    roster_wait=read(LATEST_ROSTER_WAIT);roster_outer=read(LATEST_ROSTER_OUTER)
    require(roster_wait.get("returncode")==-15 and roster_wait.get("actual_OS_wait_status")==15
        and roster_wait.get("actual_direct_worker_wait") is True
        and roster_wait.get("all_owned_processes_and_groups_drained") is True
        and roster_wait.get("completed_ref") is None and roster_wait.get("no_future_stage_authorized") is True
        and roster_wait.get("operational_failure")=="ValueError: Bounded fixed native32 owned process identities"
        and roster_wait["ownership"].get("complete_owned_roster_known") is False
        and roster_wait["ownership"].get("owned_roster_truncated") is True
        and roster_outer.get("actual_exit_code")==241 and roster_outer.get("actual_terminal_consumed") is True
        and roster_outer.get("actual_tool_session_id")==93626 and roster_outer.get("actual_wait_ref")==LATEST_ROSTER_WAIT
        and roster_outer.get("actual_direct_worker_returncode")==-15
        and roster_outer.get("all_owned_processes_and_groups_drained") is True
        and roster_outer.get("complete_owned_roster_known") is False and roster_outer.get("owned_roster_truncated") is True
        and roster_outer.get("native_CUDA_context_observed_MiB")==1316
        and roster_outer.get("full_original7GiB_charge_retained") is True
        and roster_outer.get("optimizer_updates")=="UNVERIFIED",
        "Genuine93626 failure/declaration of owned-group drain and truncated roster, never verified training")
    require(a["original_storage_ref"]==s["refs"]["storage"] and a["original_root_request_ref"]==s["root_stage_request_ref"]
        and a["original_author_scientific_config_ref"]==s["refs"]["config"]
        and a["capacity_policy_ref"]==q["capacity_policy_ref"]==POLICY
        and a["capacity_guard_ref"]==q["capacity_guard_ref"]==GUARD
        and a["original_caps"]==ORIGINAL_CAPS,"Original science/config/guard/caps/policy untouched")
    validate_resume_prerequisites_v5(a)
    ORIGINAL_STATE=(copy.deepcopy(s),copy.deepcopy(q),copy.deepcopy(storage))
    return fresh_continuation_input_v11(a,s,q,dict(cfg,max_checkpoint_bytes=2*2**30),storage,protocol)


class OriginalGuardWithExplicitSourceBridge:
    def __init__(self,real):self.real=real
    def validate_storage_v9(self,s,cfg,storage,*,source_ref,config_ref,runtime_ref):
        active_amendment_v1();require(ORIGINAL_STATE is not None,"Exact original state established")
        old,_,old_storage=ORIGINAL_STATE
        require(storage==old_storage and source_ref==original.ref(original.DRIVER)
            and config_ref==old["refs"]["config"] and runtime_ref==old["refs"]["runtime"]
            and s["actual_execution_amendment_ref"]==AMENDMENT_REF
            and s["output_directory"]==old["output_directory"]+"/"+CHILD
            and s["worker_auxiliary_scope"]==old["worker_auxiliary_scope"]+"/"+CHILD,
            "Original guard validates original roots/source only. Separate ROOT bridge validates new adapter/view")
        receipt=self.real.validate_storage_v9(old,cfg,old_storage,source_ref=source_ref,config_ref=config_ref,runtime_ref=runtime_ref)
        active_amendment_v1()  # A contended original guard may have consumed the amended lifetime.
        return receipt
    def budget_check_v9(self,storage,**kwargs):
        active_amendment_v1();require(ORIGINAL_STATE is not None and storage==ORIGINAL_STATE[2],"Only original unchanged fully charged storage")
        receipt=self.real.budget_check_v9(storage,**kwargs)
        active_amendment_v1()  # Original guard knows absolute expiry, not the earlier anchored end.
        return receipt

def guard_v1(q,*,consumer=True):
    require(consumer is True and ORIGINAL_STATE is not None and FRESH_STATE is not None,
        "Only separately admitted fresh continuation with fully retained original scope")
    return FreshContinuationGuardV11(original.guard_v1(ORIGINAL_STATE[1],consumer=True),
        fresh_guard_runtime_v11(consumer=True))

def charged_census_v1(spec,*,required_write_bytes=0):
    require(ORIGINAL_STATE is not None and RESUME_STATE is not None and FRESH_STATE is not None,
        "Both exact separately charged scopes and22000 retained input required")
    old=ORIGINAL_STATE[0]
    old_output=original._sizes(local(old["output_directory"]),allow_checkpoint_alias=True)
    old_aux=original._sizes(local(old["worker_auxiliary_scope"]))
    old_checkpoint=sum(n for parts,n,kind in old_output if "checkpoints" in parts)
    old_regular=[(parts,n-4096) for parts,n,kind in old_output if "checkpoints" in parts and kind=="regular" and parts[-1].endswith(".pt")]
    require(len(old_regular)<=3 and all(n<=3*2**30 for parts,n in old_regular)
        and old_checkpoint<=6*2**30+65536,"Original retained6GiB checkpoint section stays unchanged")
    old_totals={old["output_directory"]:sum(n for _,n,_ in old_output),old["worker_auxiliary_scope"]:sum(n for _,n,_ in old_aux)}
    require(old_totals[old["output_directory"]]<=ORIGINAL_CAPS["scientific_output_bytes"]
        and old_totals[old["worker_auxiliary_scope"]]<=ORIGINAL_CAPS["auxiliary_bytes"]
        and sum(old_totals.values())<=ORIGINAL_CAPS["reservation_bytes"]==7*2**30,
        "Entire original7GiB output/AUX remains fully measured and retained")
    output=original._sizes(local(spec["output_directory"]),allow_checkpoint_alias=True)
    auxiliary=original._sizes(local(spec["worker_auxiliary_scope"]))
    regular=[(parts,n-4096) for parts,n,kind in output if "checkpoints" in parts and kind=="regular" and parts[-1].endswith(".pt")]
    checkpoint=sum(n for parts,n,kind in output if "checkpoints" in parts)
    require(all(parts==("checkpoints","latest-rank0.pt") for parts,n,kind in output if kind=="alias")
        and len(regular)<=2 and all(parts[0]=="checkpoints" and n<=2*2**30 for parts,n in regular)
        and checkpoint<=4*2**30+65536,"Only unchanged two new2GiB author checkpoint products in fresh scope")
    totals={spec["output_directory"]:sum(n for _,n,_ in output),spec["worker_auxiliary_scope"]:sum(n for _,n,_ in auxiliary)}
    prospective=2*(2*2**30+4096)+256*1024
    require(totals[spec["output_directory"]]-checkpoint+prospective+required_write_bytes<=CAPS["scientific_output_bytes"]
        and totals[spec["output_directory"]]-checkpoint+prospective+CAPS["auxiliary_bytes"]+CAPS["directory_and_accounting_allowance_bytes"]<=CAPS["reservation_bytes"],
        "Both full new2GiB products plus unchanged AUX/directory/admin envelope fit fresh5GiB before writes")
    require(totals[spec["output_directory"]]+required_write_bytes<=CAPS["scientific_output_bytes"]
        and totals[spec["worker_auxiliary_scope"]]+required_write_bytes<=CAPS["auxiliary_bytes"]
        and sum(totals.values())+required_write_bytes<=CAPS["reservation_bytes"]==5*2**30,
        "Entire fresh continuation output+AUX uses its own5GiB charge")
    selected=FRESH_STATE[2]["reserved_new_bytes"]
    require(selected in ACCOUNTING_CHOICES_V11
        and sum(totals.values())+required_write_bytes<=selected
        and totals[spec["output_directory"]]-checkpoint+prospective+totals[spec["worker_auxiliary_scope"]]+required_write_bytes<=selected,
        "Actual whole fresh scope and both unchanged prospective2GiB products fit selected accounting failstop")
    # Old guard is read/check only. Neither allocation is retired, rebased or credited.
    original.guard_v1(ORIGINAL_STATE[1],consumer=True).budget_check_v9(ORIGINAL_STATE[2],boundary=True,required_write_bytes=0)
    return dict(old_totals,**totals)

def compact_v1(spec,name,value,g,storage,*,cap=2**20):
    active_amendment_v1();directory=local(spec["worker_auxiliary_scope"])
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(data)<=cap,"Bounded operational evidence")
    charged_census_v1(spec,required_write_bytes=len(data)+8192)
    g.budget_check_v9(storage,boundary=True,required_write_bytes=len(data)+8192)
    active_amendment_v1()  # Immediately before the new write, after census and guard.
    return save(directory,name,value)


def validate_historical6500_resume_prerequisites_v11(a):
    global RESUME_STATE
    require(a.get("resume_policy_ref")==RESUME_POLICY_REF and read(RESUME_POLICY_REF)["original_whole_scope_bytes"]==7*2**30
        and a.get("failed_v4_and_all_prior_launch_capabilities_revoked") is True
        and a.get("original_checkpoint_and_all_failed_artifacts_preserved") is True,
        "Explicit narrowly reviewed same-charge full-state resume, no deletion or automatic retry")
    require(a["failed_v4_stdout_observation_ref"]==V4_STDOUT_PROJECTION_REF
        and a["failed_v4_outer_terminal_ref"]==V4_OUTER_PROJECTION_REF,"Exact genuine ROOT factual projections, no reconstructed stdout array")
    failed=read(a["failed_v4_stdout_observation_ref"]);outer=read(a["failed_v4_outer_terminal_ref"])
    require(failed.get("adapter_manifest_ref")==V4_MANIFEST_REF
        and failed.get("returncode")==-15 and failed.get("actual_OS_wait_status")==15
        and failed.get("actual_direct_worker_wait") is True and failed.get("all_owned_processes_and_groups_drained") is True
        and failed.get("complete_owned_roster_known") is True and failed.get("completed_ref") is None
        and failed.get("operational_failure")=="ValueError: Actual external+all unfilled C3/native32 envelopes+4GiB headroom fitGPU0"
        and outer.get("actual_tool_session_id")==24814 and outer.get("actual_exit_code")==0
        and outer.get("actual_terminal_consumed") is True and outer.get("actual_direct_worker_returncode")==-15
        and outer.get("all_owned_processes_and_groups_drained") is True
        and outer.get("actual_stdout_observation_ref")==a["failed_v4_stdout_observation_ref"],
        "Preserve genuine v4worker-15 and consumed outer0 as failure, never producer0")
    require(a["ROOT_resume_qualification_ref"]==V8_QUALIFICATION_REF and a.get("old_ROOT_rows_are_historical_not_current_entry_evidence") is True and a.get("new_actual_entry_observations_mandatory") is True,"Historical original qualification is not fresh entry evidence")
    qualified=read(V8_QUALIFICATION_REF)
    require(qualified.get("schema")=="ROOT_native_VAL000_full_state_resume_qualification_v5"
        and qualified.get("root_authorizer")=="root_agent" and qualified.get("status")=="qualified"
        and qualified.get("template_only") is not True
        and qualified.get("checkpoint_ref")==a["resume_checkpoint_ref"]
        and qualified.get("checkpoint_full_SHA_via_own_readonly_FD_verified") is True
        and qualified.get("genuine_full_state_model_optimizer_scheduler_RNG_timestamp_verified") is True
        and qualified.get("whole_original7GiB_and6GiB_checkpoint_capacity_qualified") is True
        and qualified.get("current_v4_complete64_identity_and_group_drain_verified") is True
        and qualified.get("fresh_targeted_known_old_root_PID_start_and_group_absence_verified") is True
        and qualified.get("all_cooperative_old_capabilities_permanently_revoked") is True
        and qualified.get("historical_v3_descendant_coverage")=="UNKNOWN_TRUNCATED64"
        and qualified.get("historical_v3_unknown_descendants_absent") is None
        and qualified.get("private_FD_no_Composer_TMP_symlink_or_copy_route_reviewed") is True
        and qualified.get("actual_source_manifest_ref")==V8_MANIFEST_REF,"ROOT actual post-drain full-state/FD/capacity prerequisite, not source fit")
    require(qualified["actual_ROOT_verifier_receipt_ref"]==ROOT_PRESERVED_METADATA_REF,"Exact genuinely verified original6500 full-state metadata receipt")
    evidence=read(ROOT_PRESERVED_METADATA_REF)
    require(evidence["checkpoint_ref"]==RESUME_CHECKPOINT_REF and evidence["actual_verified_optimizer_updates"]==6500
        and evidence["full_byte_read_and_sha256"] is True and evidence["pre_post_file_and_FD_identity_equal"] is True
        and evidence["full_budget_complete"] is False,"Actual original6500 is partial full-state progress, never full227990")
    raw(qualified["whole_parent_capacity_observation_ref"])
    validate_known_owner_scope_v6(a,qualified,failed)
    checkpoint=qualified["checkpoint_ref"];identity=qualified["checkpoint_identity"]
    require(checkpoint==RESUME_CHECKPOINT_REF and local(checkpoint["path"])==local("experiments/results/native_VAL000_full30mb_borrowed32_GPU0_v2/same_charge_owned_roster_v4/checkpoints/ep0-ba6500-rank0.pt")
        and type(checkpoint["sha256"]) is str and len(checkpoint["sha256"])==64
        and identity["bytes"]==2017931910 and identity["link_count"]==1
        and type(qualified["checkpoint_charge_bytes"]) is int and identity["bytes"]<=qualified["checkpoint_charge_bytes"]<=2*2**30,
        "Exact original immutable retained checkpoint path/identity and future actual full SHA")
    progress=qualified["actual_saved_progress"]
    require(type(progress["epoch"]) is int and 0<=progress["epoch"]<5
        and type(progress["batch"]) is int and 0<progress["batch"]<227990
        and progress["optimizer_unique_steps"]==[progress["batch"]]==[6500]
        and type(progress["batch_in_epoch"]) is int and 0<=progress["batch_in_epoch"]<=progress["batch"],
        "Actual qualified partial full-state checkpoint, never final science")
    require(time.time()<datetime.fromisoformat("2026-10-16T23:31:32.476768+00:00").timestamp(),
        "Original strict fresh-entry cutoff retained")
    RESUME_STATE=copy.deepcopy(qualified)
    return qualified


def validate_known_owner_scope_v6(a,qualified,failed):
    global HISTORICAL_ROOT_SCOPE_V9
    record=a["ROOT_known_owner_observation_bundle_ref"]
    require(record==V8_KNOWN_BUNDLE_REF,"Exact historical v8 ROOT known-owner bundle, never relabelled current")
    require(qualified["ROOT_known_owner_observation_bundle_ref"]==record,"One exact genuine ROOT known-owner observation bundle")
    bundle=read(record)
    require(bundle.get("schema")=="ROOT_native_VAL000_known_owner_observation_bundle_v6"
        and bundle.get("root_authorizer")=="root_agent" and bundle.get("template_only") is not True and bundle.get("source_manifest_ref")==V8_MANIFEST_REF
        and bundle.get("failed_v4_stdout_observation_ref")==a["failed_v4_stdout_observation_ref"]
        and bundle.get("failed_v4_outer_terminal_ref")==a["failed_v4_outer_terminal_ref"]
        and bundle.get("current_v4_complete_roster_identity_count")==64
        and bundle.get("current_v4_complete_identity_and_group_drain_verified") is True
        and bundle.get("historical_v3_descendant_coverage")=="UNKNOWN_TRUNCATED64"
        and bundle.get("historical_v3_unknown_descendants_absent") is None
        and bundle.get("historical_v3_full_ownership_reconstructed") is False,
        "Known-owner scope only, v4complete64 and permanently UNKNOWN truncated v3 history")
    require(failed.get("known_identity_count")==64 and failed.get("adopted_wait_count")==41
        and failed.get("complete_owned_roster_known") is True and failed.get("adopted_waits_truncated") is False
        and failed.get("all_owned_processes_and_groups_drained") is True
        and failed.get("full_original_stdout_or_identity_array_not_reconstructed") is True,
        "Actual recorded currentv4 complete64 drain, without inventing unavailable full identity arrays")
    registry=read(bundle["known_identity_registry_ref"])
    pids=read(bundle["fresh_known_PID_start_absence_ref"])
    groups=read(bundle["fresh_known_process_group_absence_ref"])
    revoke=read(bundle["permanent_cooperative_revocation_ref"])
    require(all(value.get("template_only") is not True for value in (registry,pids,groups,revoke)),"Inactive observation templates never establish facts")
    expected_sources=[PRIOR_WAIT,LATEST_METADATA_WAIT,LATEST_ROSTER_WAIT,V4_STDOUT_PROJECTION_REF,V4_LAUNCH_REF,V4_CUDA_OBSERVATION_REF,V4_PRIOR_TARGET_REGISTRY_REF]
    require(registry.get("root_authorizer")=="root_agent"
        and registry.get("known_old_root_source_refs")==expected_sources
        and registry.get("all_available_known_old_root_identities_from_listed_sources_and_ROOT_launchers_included") is True
        and registry.get("unknown_historical_descendants_claimed_listed") is False,
        "Exact source-bound finite known-owner registry, no inferred historical completeness")
    identities=registry["identities"];pgids=registry["process_groups"]
    require(type(identities) is list and 1<=len(identities)<=4096
        and all(set(row)=={"pid","start_ticks"} and type(row["pid"]) is int and row["pid"]>=2
            and (row["start_ticks"] is None or type(row["start_ticks"]) is int and row["start_ticks"]>0) for row in identities)
        and len({(r["pid"],r["start_ticks"]) for r in identities})==len(identities)
        and type(pgids) is list and 1<=len(pgids)<=4096 and all(type(g) is int and g>=2 for g in pgids)
        and len(set(pgids))==len(pgids),"Complete bounded known PID/start and group registry only")
    known_pairs=set();known_pids=set();known_groups=set()
    extra=registry.get("additional_ROOT_launcher_source_refs",[])
    require(type(extra) is list and len(extra)<=64,"Explicit bounded extra ROOT launcher metadata only")
    for source_ref in expected_sources+extra:
        source=read(source_ref);own=source.get("ownership",{})
        for name in ("owned_identities","retained_owned_identities"):
            for row in own.get(name,[]):
                known_pairs.add((row["pid"],row["start"]));known_pids.add(row["pid"])
        for row in source.get("adopted_waits",[]):known_pids.add(row["pid"])
        if type(source.get("pid")) is int:known_pids.add(source["pid"])
        if type(source.get("parent_pid")) is int:known_pids.add(source["parent_pid"])
        for key in ("worker_identity_before","worker_identity_after"):
            row=source.get(key)
            if isinstance(row,dict) and type(row.get("pid")) is int and type(row.get("start_ticks")) is int:
                known_pairs.add((row["pid"],row["start_ticks"]));known_pids.add(row["pid"])
        for row in source.get("known_old_targeted_absence",[]):
            if type(row.get("pid")) is int:known_pids.add(row["pid"])
            if type(row.get("pgid")) is int:known_groups.add(row["pgid"])
        if source_ref==V4_LAUNCH_REF:known_groups.add(source["pid"])
        for row in source.get("group_observations",[]):known_groups.add(row["pgid"])
    require(known_pairs<={(row["pid"],row["start_ticks"]) for row in identities}
        and known_pids<={row["pid"] for row in identities} and known_groups<=set(pgids),
        "Every recorded known old-source identity/PID/group included, no historical completeness inferred")
    require(pids.get("root_authorizer")==groups.get("root_authorizer")=="root_agent"
        and pids.get("known_identity_registry_ref")==groups.get("known_identity_registry_ref")==bundle["known_identity_registry_ref"]
        and pids.get("unknown_history_absence_asserted") is False and groups.get("unknown_history_absence_asserted") is False,
        "Fresh targeted observations reference exact known registry without unknown-history assertion")
    rows=pids["observations"];grows=groups["observations"];historical_end=datetime.fromisoformat(read(V8_TERMINAL_REF)["recorded_utc"])
    require(len(rows)==len(identities) and len(grows)==len(pgids),"No known observation omitted or truncated")
    for identity,row in zip(identities,rows):
        require(row["pid"]==identity["pid"] and row["start_ticks"]==identity["start_ticks"]
            and row["known_identity_absent"] is True and row["method"]=="targeted PID/start or numeric ESRCH"
            and (identity["start_ticks"] is not None or row["numeric_PID_absent"] is True)
            and datetime.fromisoformat(row["checked_utc"]).tzinfo is not None and datetime.fromisoformat(row["checked_utc"])<=historical_end,
            "Authenticated historical ROOT PID/start facts with original timestamps, no current-freshness claim")
    for pgid,row in zip(pgids,grows):
        require(row["pgid"]==pgid and row["absent"] is True and row["method"]=="targeted os.killpg(pgid,0) ESRCH"
            and datetime.fromisoformat(row["checked_utc"]).tzinfo is not None and datetime.fromisoformat(row["checked_utc"])<=historical_end,
            "Authenticated historical ROOT group facts with original timestamps, no current-freshness claim")
    require(revoke.get("schema")=="ROOT_native_VAL000_permanent_cooperative_old_capability_revocation_v6"
        and revoke.get("root_authorizer")=="root_agent" and revoke.get("source_manifest_ref")==V8_MANIFEST_REF
        and revoke.get("all_source_referenced_old_launch_capabilities_permanently_revoked") is True
        and revoke.get("old_original_resume_entrypoints_and_automatic_retries_revoked") is True
        and revoke.get("new_exception_is_exact_reviewed_v8_ROOT_amendment_only") is True
        and revoke.get("historical_files_or_failures_mutated") is False
        and revoke.get("unknown_old_processes_absent_asserted") is False,
        "Permanent cooperative old-capability revocation, exact reviewed new exception, unknown history remains unknown")
    HISTORICAL_ROOT_SCOPE_V9=dict(bundle_ref=record,registry_ref=bundle["known_identity_registry_ref"],registry=copy.deepcopy(registry),current_entry_absence_established=False)
    return dict(bundle_ref=record,known_identities=len(identities),known_groups=len(pgids),
        historical_v3_descendant_coverage="UNKNOWN_TRUNCATED64",historical_unknown_absence=None)


ROOT_V6_AMENDMENT_REF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_same_charge_full_state_resume_v6.root_active_v1.json","sha256":"4150fe809de142bfdc8a79f799c0fe5ac5fa48c5c7973672330de0b7170673de"}
V6_PREPARATION_FAILURE_FACTS=[
    dict(actual_tool_session_id=19149,actual_exit_code=1,native_launch_occurred=False,
        stage="original_strict_storage_boundary_before_native_launch",
        error="canonical project path without symlinks required",
        evidence_provenance="ROOT_actual_tool_stdout_no_filesystem_receipt_was_written"),
    dict(actual_tool_session_id=78681,actual_exit_code=1,native_launch_occurred=False,
        stage="ROOT_later_attempted_failure_receipt_before_write",
        error="Planned writes require exact owned-scope boundary",
        evidence_provenance="ROOT_actual_tool_stdout_no_filesystem_receipt_was_written")]
ALIAS_SOURCE=HERE+"/alias_namespace_v7.py"
ALIAS_ORIGINAL_BEFORE="ROOT_full_state_resume_v9_original_Composer_alias.before_unlink.json"
ALIAS_ORIGINAL_AFTER="ROOT_full_state_resume_v9_original_Composer_alias.strict_after.json"
ALIAS_TERMINAL_BEFORE="ROOT_full_state_resume_v9_terminal_Composer_alias.before_unlink.json"
ALIAS_METADATA_CAP=65536

def validate_alias_authority_v7(a):
    require(a.get("preserved_actual_v6_amendment_ref")==ROOT_V6_AMENDMENT_REF
        and a.get("actual_v6_future_execution_permanently_revoked") is True
        and a.get("prior_v6_preparation_failure_facts")==V6_PREPARATION_FAILURE_FACTS
        and a.get("exact_Composer_alias_metadata_only_unlink_authorized") is True
        and a.get("exact_precanonical_metadata_receipt_exception_authorized") is True
        and a.get("precanonical_metadata_max_bytes")==ALIAS_METADATA_CAP
        and a.get("canonical_or_global_guard_change_authorized") is False
        and a.get("regular_checkpoint_copy_or_delete_authorized") is False,
        "Exact narrow ROOT v7 alias metadata/unlink authority, preserved failed v6 and unchanged strict guard")
    require(a.get("preserved_actual_v8_amendment_ref")==V8_AMENDMENT_REF
        and a.get("v8_future_launch_write_capabilities_permanently_revoked") is True
        and a.get("genuine_v8_failed_terminal_ref")==V8_TERMINAL_REF
        and a.get("known_target_readonly_entry_probes_authorized") is True
        and a.get("entry_probe_scope")=="exact_finite_authenticated_known_registry_including_v8"
        and a.get("entry_fresh_seconds")==120 and a.get("entry_probe_max_receipt_bytes")==2*2**20
        and a.get("startup_original_alias_unlink_authorized") is False,
        "Exact ROOT v9 fresh-entry-only probe/metadata authority and immutable v8 failure lineage")
    validate_v8_failure_lineage_v9()
    old=read(ROOT_V6_AMENDMENT_REF)
    require(old.get("schema")=="native32_full_state_resume_same_charge_ROOT_amendment_v6"
        and old.get("root_authorizer")=="root_agent" and old.get("status")=="active",
        "Exact immutable actual v6 authority retained as historical evidence only")
    return True

def alias_module_v7():
    return load(ref(ALIAS_SOURCE),"_native_VAL000_exact_alias_namespace_v7")

def alias_readonly_guard_v7(spec,g,storage,*,projected_metadata_bytes=0):
    require(type(projected_metadata_bytes) is int and 0<=projected_metadata_bytes<=ALIAS_METADATA_CAP+8192,
        "Only bounded exact pre-canonical metadata projection")
    # This unchanged original call is read-only. It does not authorize a write.
    g.budget_check_v9(storage,boundary=False,required_write_bytes=0)
    # The separately explicit ROOT authority authorizes only the exact bounded
    # metadata receipt after a full original no-follow charge/cap projection.
    charged_census_v1(spec,required_write_bytes=projected_metadata_bytes)
    active_amendment_v1()
    return True

def alias_strict_boundary_v7(spec,g,storage):
    charged_census_v1(spec)
    g.budget_check_v9(storage,boundary=True,required_write_bytes=0)
    active_amendment_v1()
    return dict(original_unchanged_budget_check_boundary_True_passed=True,
        original_whole_scope_no_follow_charge_caps_passed=True,
        storage_close_or_credit_authorized=False)

def alias_durable_metadata_v7(spec,g,storage,directory,name,value):
    require(name==ALIAS_TERMINAL_BEFORE,
        "v9 authorizes only this fresh child terminal alias metadata, no startup unlink/receipt replay")
    expected=local(spec["original_charged_auxiliary_scope"]) if name==ALIAS_ORIGINAL_BEFORE \
        else local(spec["worker_auxiliary_scope"])
    require(directory==expected and directory.is_dir(),"Exact ROOT-owned original or own terminal AUX")
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(0<len(data)<=ALIAS_METADATA_CAP,"Exact64KiB metadata exception, no checkpoint payload")
    alias_readonly_guard_v7(spec,g,storage,projected_metadata_bytes=len(data)+8192)
    active_amendment_v1()
    target=directory/name
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    dfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(dfd)
    finally:os.close(dfd)
    result=ref(str(target.relative_to(ROOT)))
    alias_module_v7().verify_receipt_v7(ROOT,result,value)
    return result

def alias_original_context_v7():
    return dict(kind="fixed_original_stopped_Composer_alias",
        actual_execution_amendment_ref=current_amendment_ref_v1(),adapter_manifest_ref=ref(MANIFEST),
        original_resume_checkpoint_ref=RESUME_CHECKPOINT_REF,
        ROOT_full_state_verifier_receipt_ref=ROOT_PRESERVED_METADATA_REF,
        factual_v4_current_complete64_drain_ref=V4_STDOUT_PROJECTION_REF,
        historical_v3_descendant_coverage="UNKNOWN_TRUNCATED64",
        historical_v3_unknown_descendants_absent=None,
        preserved_actual_v6_amendment_ref=ROOT_V6_AMENDMENT_REF,
        prior_v6_preparation_failure_facts=V6_PREPARATION_FAILURE_FACTS,
        boundary_False_check_is_read_only_not_write_authority=True,
        separate_exact_ROOT_precanonical_metadata_exception=True)

def normalize_original_alias_v7(spec,q,g,storage):
    require(ORIGINAL_STATE is not None and RESUME_STATE is not None,"Authenticated historical full-state prerequisite first")
    validate_v8_failure_lineage_v9()
    module=alias_module_v7()
    before=read(V8_ALIAS_BEFORE_REF);after=read(V8_ALIAS_AFTER_REF)
    module.verify_receipt_v7(ROOT,V8_ALIAS_BEFORE_REF,before)
    directory=local(RESUME_CHECKPOINT_REF["path"]).parent;alias=directory/"latest-rank0.pt"
    require(before["schema"]=="native_VAL000_durable_ROOT_Composer_alias_before_unlink_v7"
        and before["context"]["actual_execution_amendment_ref"]==V8_AMENDMENT_REF
        and before["context"]["adapter_manifest_ref"]==V8_MANIFEST_REF
        and before["context"]["historical_v3_descendant_coverage"]=="UNKNOWN_TRUNCATED64"
        and before["context"]["historical_v3_unknown_descendants_absent"] is None
        and before["target_ref"]==RESUME_CHECKPOINT_REF and before["target_identity"]==RESUME_STATE["historical6500_checkpoint_identity"]
        and before["alias"]==str(alias.relative_to(ROOT)) and before["target_text"]=="ep0-ba6500-rank0.pt"
        and before["actual_complete_owned_writer_drain_verified"] is True
        and before["regular_checkpoint_payload_read_or_copied"] is False,
        "Exact immutable successful v8 original-alias before metadata and original6500 identity")
    require(after["schema"]=="native_VAL000_exact_Composer_alias_removed_v7"
        and after["before_unlink_metadata_ref"]==V8_ALIAS_BEFORE_REF
        and after["alias"]==before["alias"] and after["alias_absent"] is True
        and after["all_regular_checkpoint_metadata_unchanged"] is True
        and after["regular_checkpoint_payload_read_copied_or_deleted"] is False
        and after["strict_original_boundary_after_unlink"]["original_unchanged_budget_check_boundary_True_passed"] is True
        and after["strict_original_boundary_after_unlink"]["original_whole_scope_no_follow_charge_caps_passed"] is True,
        "Exact immutable successful v8 strict-after receipt, no normalization replay")
    try:alias.lstat()
    except FileNotFoundError:pass
    else:raise ValueError("Original alias reappeared, v9 has no startup unlink authority")
    require(module.identity(local(RESUME_CHECKPOINT_REF["path"]).lstat())==RESUME_STATE["historical6500_checkpoint_identity"]
        and module.regular_snapshot(directory)==before["regular_checkpoint_metadata_before"],
        "Every retained original6500 regular metadata unchanged, never copied or deleted")
    return dict(schema="native_VAL000_v9_authenticated_v8_original_alias_absence",
        v8_failed_terminal_ref=V8_TERMINAL_REF,before_unlink_metadata_ref=V8_ALIAS_BEFORE_REF,
        strict_after_metadata_ref=V8_ALIAS_AFTER_REF,alias_absent=True,unlink_replayed=False,
        original_regular_metadata_unchanged=True,strict_original_boundary=alias_strict_boundary_v7(spec,g,storage))


def normalize_resumed_alias_v7(spec,q,g,storage,*,prove_drained,minimal_wait_facts):
    module=alias_module_v7()
    out=local(spec["output_directory"]);directory=out/"checkpoints"
    # This callable is supplied only by the actual post-finally supervisor. Its
    # drained boolean already requires complete4096 roster, wait, EOF, identities
    # and targeted group absence. No historical unknown is upgraded.
    require(prove_drained(),"Genuine complete current owned drain before any terminal alias operation")
    require(FRESH_STATE is not None and spec["output_directory"]==FRESH_STATE[0]["output_directory"]
        and spec["worker_auxiliary_scope"]==FRESH_STATE[0]["worker_auxiliary_scope"],
        "Only this exact separately admitted fresh continuation alias after actual drain")
    try:s=directory.lstat()
    except FileNotFoundError:
        return dict(alias_present=False,strict_original_boundary=alias_strict_boundary_v7(spec,g,storage))
    require(stat.S_ISDIR(s.st_mode),"Own real terminal checkpoint directory")
    alias=directory/"latest-rank0.pt"
    try:alias.lstat()
    except FileNotFoundError:
        return dict(alias_present=False,strict_original_boundary=alias_strict_boundary_v7(spec,g,storage))
    return module.normalize_latest_v7(ROOT,directory,prove_drained=prove_drained,
        preserve=lambda value:alias_durable_metadata_v7(spec,g,storage,
            local(spec["worker_auxiliary_scope"]),ALIAS_TERMINAL_BEFORE,value),
        revalidate_pre_unlink=lambda:alias_readonly_guard_v7(spec,g,storage),
        strict_boundary=lambda:alias_strict_boundary_v7(spec,g,storage),
        max_bytes=2*2**30,max_regular_files=2,
        context=dict(kind="only_this_actual_resumed_child_after_complete_owned_drain",
            actual_execution_amendment_ref=current_amendment_ref_v1(),adapter_manifest_ref=ref(MANIFEST),
            actual_minimal_wait_facts=minimal_wait_facts,
            regular_checkpoint_content_SHA_not_claimed=True,
            separate_exact_ROOT_precanonical_metadata_exception=True))


V8_AMENDMENT_REF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_same_charge_full_state_resume_v8.root_active_v1.json","sha256":"129a2e0f34668d3bffa8bb4b1214881876c59c814937d31dff1dd0a791da1bdb"}
V8_MANIFEST_REF={"path":"research/reviews/native_VAL000_full_state_resume_source_20261016_v8/source_manifest.v1.json","sha256":"168918ab804797a3cd65562c737da3e615475b4c18a3ad9df3e3d774aad0c065"}
V8_QUALIFICATION_REF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_full_state_resume_v8_qualification.root_v1.json","sha256":"0099de8018b18280e1e4b92f67c96062771d69ef1cdeace2b31134f89359c04b"}
V8_KNOWN_BUNDLE_REF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_full_state_resume_v8_known_owner_bundle.root_v1.json","sha256":"a1a4982bf234470f02152aab03a69362976af40768fed339bc8d14cb08865e45"}
V8_TERMINAL_REF={"path":"research/reviews/native_v8_actual_terminal_root_20261016_v1/terminal.root_v1.json","sha256":"6431f72c236aaa2268b4960d51cba9b47e2ceb531dde9b495edbdbc96628364a"}
V8_WAIT_REF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/same_charge_full_state_resume_v8/actual_wait.json","sha256":"9d50458c8730f420fb6d2ca87641778e9590fee9f49d1a3b7eab46958b7c570e"}
V8_LAUNCH_REF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/same_charge_full_state_resume_v8/launch.json","sha256":"02c9928047c0d31c4e6c638919795fd8409fa20d142cf48f451ee47a1cd2289a"}
V8_ALIAS_BEFORE_REF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_full_state_resume_v8_original_Composer_alias.before_unlink.json","sha256":"995ac216401e6e52b72b49690cdfa292fa3c000d835b6d0aa3e71fbaddf7ab6f"}
V8_ALIAS_AFTER_REF={"path":"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_full_state_resume_v8_original_Composer_alias.strict_after.json","sha256":"82f6e374eb4b20cd0dca51e820b23b5eb78a0fee3c7ec4f108be7c455b72e159"}
HISTORICAL_ROOT_SCOPE_V9=None
ENTRY_REGISTRY_CACHE_V9=None

def validate_v8_failure_lineage_v9():
    terminal=read(V8_TERMINAL_REF);wait=read(V8_WAIT_REF);launch=read(V8_LAUNCH_REF)
    require(terminal.get("schema")=="ROOT_native_resume_v8_actual_consumed_terminal_v1"
        and terminal.get("actual_tool_session_id")==74968 and terminal.get("actual_exit_code")==0
        and terminal.get("actual_terminal_consumed") is True and terminal.get("actual_direct_worker_returncode")==1
        and terminal.get("actual_OS_wait_status")==256 and terminal.get("all_owned_processes_and_groups_drained") is True
        and terminal.get("complete_owned_roster_known") is True and terminal.get("known_identity_count")==1
        and terminal.get("actual_wait_ref")==V8_WAIT_REF and terminal.get("actual_launch_ref")==V8_LAUNCH_REF
        and terminal.get("actual_amendment_ref")==V8_AMENDMENT_REF
        and terminal.get("known_worker_identity")==dict(pid=3660358,start=484268347)
        and terminal.get("actual_supervisor_pid")==3659453 and terminal.get("actual_supervisor_start_ticks")=="UNKNOWN"
        and terminal.get("native_CUDA_context_observed_MiB")==0 and terminal.get("native_model_optimizer_restore_reached") is False
        and terminal.get("new_optimizer_updates")==0 and terminal.get("scientific_completion_asserted") is False
        and terminal.get("original_alias_before_metadata_ref")==V8_ALIAS_BEFORE_REF
        and terminal.get("original_alias_strict_after_ref")==V8_ALIAS_AFTER_REF,
        "Genuine consumed v8 outer0 is failed direct1/noCUDA/no restore, never scientific completion")
    require(wait.get("returncode")==1 and wait.get("actual_OS_wait_status")==256
        and wait.get("actual_direct_worker_wait") is True and wait.get("all_owned_processes_and_groups_drained") is True
        and wait.get("complete_owned_roster_known") is True and wait.get("completed_ref") is None
        and wait.get("adapter_manifest_ref")==V8_MANIFEST_REF and wait.get("actual_execution_amendment_ref")==V8_AMENDMENT_REF
        and wait["ownership"]["owned_identities"]==[dict(pid=3660358,start=484268347)]
        and wait["ownership"]["known_identity_count"]==1 and wait["ownership"]["owned_roster_truncated"] is False
        and launch["pid"]==3660358 and launch["parent_pid"]==3659453
        and launch["ownership"]["owned_identities"]==[dict(pid=3660358,start=484268347)],
        "Exact preserved v8 one-identity launch/wait/drain, unknown supervisor start remains unknown")
    return terminal,wait,launch

def entry_known_registry_v9():
    global ENTRY_REGISTRY_CACHE_V9
    require(HISTORICAL_ROOT_SCOPE_V9 is not None,"Authenticated historical ROOT scope before new entry probes")
    if ENTRY_REGISTRY_CACHE_V9 is not None:
        require(ENTRY_REGISTRY_CACHE_V9["actual_execution_amendment_ref"]==current_amendment_ref_v1(),"One exact v9 registry generation")
        return copy.deepcopy(ENTRY_REGISTRY_CACHE_V9)
    a,_,_=active_amendment_v1()
    terminal,wait,launch=validate_v8_failure_lineage_v9()
    old=HISTORICAL_ROOT_SCOPE_V9["registry"];pairs=[dict(row) for row in old["identities"]];groups=list(old["process_groups"])
    def add(pid,start):
        row=dict(pid=pid,start_ticks=start)
        if row not in pairs:pairs.append(row)
    add(3660358,484268347);add(3659453,None)
    for row in wait["group_observations"]:
        require(type(row["pgid"]) is int and row["absent"] is True,"Only actual recorded v8 groups")
        if row["pgid"] not in groups:groups.append(row["pgid"])
    extra=a.get("additional_known_old_ROOT_launcher_source_refs",[])
    require(type(extra) is list and len(extra)<=64,"Bounded explicit extra old ROOT launcher metadata only")
    for r in extra:
        source=read(r)
        for key in ("pid","parent_pid","supervisor_pid"):
            if type(source.get(key)) is int:add(source[key],None)
        for row in source.get("ownership",{}).get("owned_identities",[]):add(row["pid"],row["start"])
        for row in source.get("group_observations",[]):
            if row["pgid"] not in groups:groups.append(row["pgid"])
    require(1<=len(pairs)<=4096 and 1<=len(groups)<=4096
        and all(type(row["pid"]) is int and row["pid"]>=2 and (row["start_ticks"] is None or type(row["start_ticks"]) is int and row["start_ticks"]>0) for row in pairs)
        and all(type(value) is int and value>=2 for value in groups),"Exact finite known entry registry")
    ENTRY_REGISTRY_CACHE_V9=dict(schema="native_VAL000_exact_known_entry_registry_v9",
        actual_execution_amendment_ref=current_amendment_ref_v1(),
        historical_ROOT_bundle_ref=V8_KNOWN_BUNDLE_REF,historical_registry_ref=HISTORICAL_ROOT_SCOPE_V9["registry_ref"],
        new_v8_source_refs=[V8_TERMINAL_REF,V8_WAIT_REF,V8_LAUNCH_REF],additional_known_old_ROOT_launcher_source_refs=extra,
        identities=pairs,process_groups=groups,v8_supervisor_start_ticks=None,
        historical_v3_descendant_coverage="UNKNOWN_TRUNCATED64",unknown_historical_descendants_claimed_listed=False,
        current_entry_absence_established_by_registry_alone=False)
    return copy.deepcopy(ENTRY_REGISTRY_CACHE_V9)


def make_entry_tick_v9(spec,external_tick=None):
    a,_,q=active_amendment_v1()
    original_end=min(datetime.fromisoformat(a["original48h_window_anchor_utc"]).timestamp()+172800,
        datetime.fromisoformat(original.ORIGINAL_ABSOLUTE_EXPIRY).timestamp(),
        datetime.fromisoformat(spec["expires_at_utc"]).timestamp())-CAPS["work_cutoff_margin_seconds"]
    strict_entry=datetime.fromisoformat("2026-10-16T23:31:32.476768+00:00").timestamp()
    def tick():
        if external_tick is not None:external_tick()
        require(time.time()<min(original_end,strict_entry),
            "Actual original anchored work/entry cutoff, no reset during entry probes")
    tick()
    return tick


V9_ACTUAL_TERMINAL_REF={'path': 'research/reviews/native_v9_ROOT_actual_preparation_20261016_v1/execution_terminal.root_v1.json', 'sha256': '94fa7a222b63f446812f1a551fc1428a7ebbf2ca99411606c68abeb1db7b587f'}
V9_ACTUAL_AMENDMENT_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_same_charge_full_state_resume_v9.root_active_v1.json', 'sha256': 'bc7a2db633373791e18bb83ffe403aa5e45277b34b1daaa4f2afee2855d00eb3'}
V9_CONTROLLER_AUDIT_REF={'path': 'research/reviews/native_v9_controller_entry_failure_audit_20261016_v1/audit.v1.json', 'sha256': 'b25487fa5480ac92faf33fd51d5a33fcc451c51a19e41a1e217eb5d0a6b6178d'}

def validate_v9_failure_lineage_v10(a):
    require(a.get("preserved_actual_v9_amendment_ref")==V9_ACTUAL_AMENDMENT_REF
        and a.get("genuine_v9_controller_entry_failed_terminal_ref")==V9_ACTUAL_TERMINAL_REF
        and a.get("sealed_v9_controller_failure_audit_ref")==V9_CONTROLLER_AUDIT_REF
        and a.get("v9_future_launch_write_capabilities_permanently_revoked") is True
        and a.get("exact_recorded_failed_and_drained_C3_exit1_attachment_authorized") is True
        and a.get("failed_C3_relabel_or_outer_success_claim_authorized") is False,
        "Separate ROOT v10 exact failed-C3 operational authority and immutable v9 failure")
    old=read(V9_ACTUAL_AMENDMENT_REF)
    require(old.get("schema")=="native32_full_state_resume_same_charge_ROOT_amendment_v9"
        and old.get("adapter_manifest_ref")==ref("research/reviews/native_VAL000_full_state_resume_source_20261016_v9/source_manifest.v1.json")
        and old.get("original_input_ref")==ORIGINAL_INPUT
        and old.get("fresh_child_name")=="same_charge_full_state_resume_v9",
        "Exact consumed v9 source and namespace remain preserved")
    terminal=read(V9_ACTUAL_TERMINAL_REF)
    require(terminal.get("schema")=="ROOT_actual_native_v9_execution_subprocess_terminal_v1"
        and terminal.get("actual_returncode")==1 and terminal.get("amendment_ref")==V9_ACTUAL_AMENDMENT_REF
        and terminal.get("input_ref")==ORIGINAL_INPUT and terminal.get("no_automatic_retry") is True
        and terminal.get("stderr")=="","Genuine failed v9 execution wrapper, no outer-success relabel")
    packets=[json.loads(line) for line in terminal["stdout"].splitlines()]
    require(len(packets)==4,"Exact actual bounded v9 failure stdout packets")
    minimal=packets[0]["native32_actual_minimal_wait_facts"]
    cleanup=packets[1]["native32_bounded_cleanup_observation"];result=packets[-1]
    message="ValueError: Only original C3 may run/drain/finish; no native/other queued or live work"
    for value in (minimal,cleanup):
        require(value.get("actual_execution_amendment_ref")==V9_ACTUAL_AMENDMENT_REF
            and value.get("actual_direct_worker_wait") is False
            and value.get("actual_OS_wait_status") is None and value.get("returncode") is None
            and value.get("pid") is None and value.get("operational_failure")==message
            and value.get("complete_owned_roster_known") is False
            and value.get("all_owned_processes_and_groups_drained") is False,
            "v9 no-Popen NULL identity/wait and incomplete old drain remain unknown")
    require(cleanup.get("gate_released") is False and cleanup.get("borrowed_physical_inode") is False
        and cleanup.get("ownership",{}).get("owned_tree_constructed") is False
        and cleanup.get("owned_identity_absence") is None
        and result.get("status")=="FAILED_UNRESOLVED_CLEANUP" and result.get("returncode")==1
        and result.get("actual_direct_worker_returncode") is None
        and result.get("actual_direct_worker_wait") is False and result.get("actual_wait_ref") is None
        and result.get("completed_ref") is None and result.get("all_owned_processes_and_groups_drained") is False
        and result.get("no_future_stage_authorized") is True,
        "No v9 worker, borrowed lease, gate, restore, update or complete-drain claim invented")
    audit=read(V9_CONTROLLER_AUDIT_REF)
    require(audit.get("status")=="SEALED_TARGETED_METADATA_AND_SOURCE_AUDIT"
        and audit.get("v9_actual_failed_attempt",{}).get("terminal_ref")==V9_ACTUAL_TERMINAL_REF
        and audit["v9_actual_failed_attempt"].get("amendment_ref")==V9_ACTUAL_AMENDMENT_REF
        and audit.get("exact_v9_rejection",{}).get("failing_state")=="failed"
        and audit["C3_recorded_terminal_lineage"].get("outer_ROOT_handle_74136_terminal_consumed") is None,
        "Exact sealed audit binds recorded C3 failure without claiming separate outer74136 terminal")
    return terminal,audit


V10_AMENDMENT_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_same_charge_full_state_resume_v10.root_active_v1.json', 'sha256': 'bc7db02872d605247d5ee19d9d083337a7c148d109e04facd74a50c178e902c3'}
V10_WAIT_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/same_charge_full_state_resume_v10/actual_wait.json', 'sha256': 'bed797470aa161f34c53dab16e6f782224b008fbbaa47c93e65383aee59e094b'}
V10_LAUNCH_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/same_charge_full_state_resume_v10/launch.json', 'sha256': '5ce0054231adbf53f85695ea7976c519f84f4897c522b7c81adb7ebc53c455df'}
V10_ATTACHMENT_REF={'path': 'research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/same_charge_full_state_resume_v10/attachment.json', 'sha256': '8a91775dc59fc7bc3ff5254acaa0efb861bf4806b3491bbdd27cc936196358d9'}
V10_SUBPROCESS_TERMINAL_REF={'path': 'research/reviews/native_v10_ROOT_actual_preparation_20261016_v1/execution_terminal.root_v1.json', 'sha256': '6cdc69d71291273f5e69cc192d11fc4f9804a4d4557ef058bc50b178f2ead454'}
V10_MANIFEST_REF={'path': 'research/reviews/native_VAL000_full_state_resume_source_20261016_v10/source_manifest.v1.json', 'sha256': '9ab34741ee948a2ddc9fe986abb0bd72533ead0a788c79270fe3064307193719'}

def terminal_snapshot_authority_v11(a):
    require(a.get("terminal_failed_C3_historical_snapshot_authorized") is True
        and a.get("all_old_controller_submissions_permanently_revoked") is True
        and a.get("v10_future_launch_write_capabilities_permanently_revoked") is True
        and a.get("preserved_actual_v10_amendment_ref")==V10_AMENDMENT_REF
        and a.get("genuine_v10_direct_wait_ref")==V10_WAIT_REF
        and a.get("genuine_v10_subprocess_terminal_ref")==V10_SUBPROCESS_TERMINAL_REF,
        "Explicit ROOT terminal scheduling snapshot, permanent submission revocation and preserved failed v10")
    directive=read(a["ROOT_directive_ref"])
    require(directive.get("root_authorizer")=="root_agent" and directive.get("template_only") is False
        and directive.get("source_manifest_ref")==ref(MANIFEST)
        and directive.get("controller_capability_ref")==CONTROLLER_CAPABILITY_REF
        and directive.get("controller_pid")==3349405 and directive.get("controller_start_ticks")==481281638
        and directive.get("terminal_failed_C3_historical_snapshot_authorized") is True
        and directive.get("all_old_controller_submissions_permanently_revoked") is True
        and directive.get("v10_future_launch_write_capabilities_permanently_revoked") is True,
        "Actual exact-manifest ROOT directive preserves controller and permanently revokes submissions")
    old=read(V10_AMENDMENT_REF);wait=read(V10_WAIT_REF);terminal=read(V10_SUBPROCESS_TERMINAL_REF)
    require(old.get("adapter_manifest_ref")==V10_MANIFEST_REF
        and old.get("original_input_ref")==ORIGINAL_INPUT
        and old.get("fresh_child_name")=="same_charge_full_state_resume_v10",
        "Original v10 sources/admission and failed namespace stay immutable")
    require(wait.get("actual_execution_amendment_ref")==V10_AMENDMENT_REF
        and wait.get("input_ref")==ORIGINAL_INPUT and wait.get("actual_direct_worker_wait") is True
        and wait.get("returncode")==-15 and wait.get("actual_OS_wait_status")==15
        and wait.get("complete_owned_roster_known") is True
        and wait.get("all_owned_processes_and_groups_drained") is True and wait.get("pipe_EOF_observed") is True
        and wait.get("completed_ref") is None and wait.get("outer_tool_terminal_consumed") is False
        and wait.get("operational_failure")=="ValueError: No fresh authenticated controller reply before bounded deadline",
        "Actual v10 direct -15/drain and original unconsumed-outer field remain failure facts")
    require(terminal.get("schema")=="ROOT_actual_native_v10_execution_subprocess_terminal_v1"
        and terminal.get("actual_returncode")==241 and terminal.get("amendment_ref")==V10_AMENDMENT_REF
        and terminal.get("input_ref")==ORIGINAL_INPUT and terminal.get("no_automatic_retry") is True,
        "Actual subprocess241 remains distinct from outer tool10747")
    require(a.get("genuine_v10_consumed_outer_terminal_ref")==V10_OUTER_TERMINAL_REF,"Exact genuinely consumed outer10747 failure receipt")
    outer=read(V10_OUTER_TERMINAL_REF)
    require(outer.get("root_authorizer")=="root_agent" and outer.get("actual_tool_session_id")==10747
        and outer.get("actual_terminal_consumed") is True and type(outer.get("actual_exit_code")) is int
        and outer.get("actual_wait_ref")==V10_WAIT_REF
        and outer.get("execution_subprocess_terminal_ref")==V10_SUBPROCESS_TERMINAL_REF
        and outer.get("actual_direct_worker_returncode")==-15
        and outer.get("all_owned_processes_and_groups_drained") is True
        and outer.get("completed_ref") is None and outer.get("native_v10_attempt_status")=="FAILED",
        "ROOT must actually consume outer10747 and preserve its actual exit separately, never infer success")
    extra=a.get("additional_known_old_ROOT_launcher_source_refs",[])
    require(V10_WAIT_REF in extra and V10_LAUNCH_REF in extra,
        "Every actual recorded v10 owned PID/start/group and supervisor joins unchanged fresh entry probes")
    return True

V10_OUTER_TERMINAL_REF={'path': 'research/reviews/native_v10_ROOT_actual_preparation_20261016_v1/actual_tool_terminal.root_v1.json', 'sha256': '9551bff5fb65dee4a9347e7088e033b386e195996d2dfd92b866e6a166199f74'}
V11_SAVED22000_AUDIT_REF={'path': 'research/reviews/native_v10_stopped_checkpoint_progress_root_20261016_v2/checkpoint_progress.root_v2.json', 'sha256': 'b988f895b540591a45dfb6c8b1f920df61a1960c1d55f3f580fce507c8ffb85a'}
V11_RESUME_CHECKPOINT_REF={'path': 'experiments/results/native_VAL000_full30mb_borrowed32_GPU0_v2/same_charge_full_state_resume_v10/checkpoints/ep0-ba22000-rank0.pt', 'sha256': 'a2f646f32e9d7c73e0d659893115b8db6f6c2345d3857eaeb2bbbebf6e16300c'}
V11_SAVED22000_VERIFIER_REF={'path': 'research/reviews/native_v10_checkpoint_progress_root_20261016_v1/verify_saved_progress_v1.py', 'sha256': 'b9d4182cf5312ae829b97cd31ec7f43c3c3c21403067ec47a242c490178164d9'}

def validate_resume_prerequisites_v5(a):
    global RESUME_STATE
    # Preserve the old6500 qualification as immutable history only. It is never
    # the current load path. Actual future restore still reads/hashes only22000.
    historical=dict(a,ROOT_resume_qualification_ref=V8_QUALIFICATION_REF,
        resume_checkpoint_ref=RESUME_CHECKPOINT_REF)
    qualified=validate_historical6500_resume_prerequisites_v11(historical)
    require(a.get("ROOT_resume_qualification_ref")==V11_SAVED22000_AUDIT_REF
        and a.get("resume_checkpoint_ref")==V11_RESUME_CHECKPOINT_REF
        and a.get("resume_exact_verified22000_full_state_authorized") is True,
        "Only exact ROOT-audited22000 full-state continuation; no6500 replay or unverified later updates")
    audit=read(V11_SAVED22000_AUDIT_REF);counters=audit["actual_saved_optimizer_counters"];stamp=audit["Composer_timestamp"]
    require(audit.get("schema")=="ROOT_live_native_saved_checkpoint_progress_v1"
        and audit.get("status")=="VERIFIED_SAVED_PROGRESS" and audit.get("checkpoint_ref")==V11_RESUME_CHECKPOINT_REF
        and audit.get("verifier_ref")==V11_SAVED22000_VERIFIER_REF
        and audit.get("checkpoint_bytes")==2017943366 and audit.get("full_byte_hash_and_FD_mmap") is True
        and audit.get("pre_post_named_and_FD_identity_equal") is True and audit.get("checkpoint_modified") is False
        and audit.get("scheduler_callbacks_RNG_present") is True and audit.get("model_parameter_entries")==78
        and audit.get("native_v10_current_worker_stopped") is True and audit.get("native_v10_direct_wait_returncode")==-15
        and audit.get("training_or_model_forward_executed") is False and audit.get("CUDA_context_created") is False
        and audit.get("actual_saved_optimizer_updates")==22000 and audit.get("additional_saved_updates_beyond6500")==15500
        and audit.get("full_budget_complete") is False and audit.get("full_update_budget")==227990
        and audit.get("live_updates_after_saved_checkpoint_unknown") is True
        and counters.get("parameter_state_count")==77 and counters.get("optimizer_names")==["DecoupledAdamW"]
        and counters.get("unique_step_values")==[22000] and counters.get("minimum_step")==counters.get("maximum_step")==22000
        and stamp.get("epoch")==0 and stamp.get("batch")==stamp.get("batch_in_epoch")==22000,
        "Exact verified saved22000 model/77AdamW/scheduler/callback/RNG/timestamp state, partial227990 budget only")
    raw(V11_SAVED22000_VERIFIER_REF)
    path=local(V11_RESUME_CHECKPOINT_REF["path"]);st=path.lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size==audit["checkpoint_bytes"]<=2*2**30,
        "Current named22000 checkpoint is regular single-link original-sized state")
    identity=dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,
        mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,link_count=st.st_nlink)
    progress=dict(epoch=0,batch=22000,batch_in_epoch=22000,optimizer_unique_steps=[22000])
    qualified.update(checkpoint_ref=V11_RESUME_CHECKPOINT_REF,checkpoint_identity=identity,
        checkpoint_charge_bytes=2*2**30,actual_saved_progress=progress,
        actual_ROOT_verifier_receipt_ref=V11_SAVED22000_AUDIT_REF,
        historical6500_checkpoint_ref=RESUME_CHECKPOINT_REF,historical6500_ROOT_metadata_ref=ROOT_PRESERVED_METADATA_REF,
        historical6500_checkpoint_identity=copy.deepcopy(qualified["checkpoint_identity"]),
        verified_saved_updates=22000,earned_updates_beyond6500=15500,post_save_extra_updates="UNKNOWN",
        current_restore_revalidates_full_SHA_and_all_original_state=True)
    RESUME_STATE=copy.deepcopy(qualified)
    return qualified


def fresh_guard_runtime_v11(*,consumer):
    require(type(consumer) is bool,"Explicit fresh current-child mutation/read role")
    load(RENEWAL,"_native11_preserved_renewal").install_renewal_v1(CERT)
    real=load(FRESH_GUARD_REF,"_native11_actual_native14v3_guard")
    real.configure_v2(REPAIR,policy_ref=FRESH_POLICY_REF)
    require(all(getattr(real.budget.v4,n).__module__==real.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Original unwrapped native14v3 guard")
    if consumer:
        namespace=load(NAMESPACE_MEMO,"_native11_fresh_consumer_memo")
        real._native11_readonly_memo=namespace.install_consumer_v1(real.budget.v4,real.budget,
            consumer_role="read_only_consumer",expected_source_path=ROOT/V4["path"],expected_source_sha256=V4["sha256"])
    return real

def fresh_required_roles_v11(request_ref,request,closure_ref):
    rows=[ref(MANIFEST),*[ref(HERE+"/"+name) for name in ("__init__.py","amendment_contract_v1.py","direct_gpu0_v1.py",
        "execution_only_v1.py","corpus_metadata_bridge_v1.py","resume_scientific_bridge_v5.py","alias_namespace_v7.py","fresh_entry_v9.py")],
        ref(HERE+"/prepare_continuation_root_v11.py"),request_ref,closure_ref,request["ROOT_directive_ref"],
        request["independent_source_review_ref"],ORIGINAL_INPUT,request["config_ref"],request["runtime_ref"],
        FRESH_GUARD_REF,FRESH_POLICY_REF,V11_SAVED22000_AUDIT_REF,V10_OUTER_TERMINAL_REF,
        request["direct_attachment_authority_ref"],read(ORIGINAL_INPUT)["refs"]["storage"]]
    unique={}
    for row in rows:
        raw(row);require(row["path"] not in unique or unique[row["path"]]==row,"Every exact source role, no hash conflict")
        unique[row["path"]]=row
    return list(unique.values())

def fresh_continuation_input_v11(a,old,oldq,cfg,old_storage,protocol):
    global FRESH_STATE
    require(a.get("fresh5GiB_continuation_execution_authorized") is True
        and a.get("original7GiB_scope_is_retained_readonly") is True
        and a.get("fresh_continuation_policy_ref")==FRESH_POLICY_REF
        and a.get("fresh_continuation_guard_ref")==FRESH_GUARD_REF,
        "Explicit separately charged fresh continuation; original scope never expanded or reused")
    record=a["fresh_continuation_input_ref"];s=read(record);active(s)
    req=read(s["root_stage_request_ref"]);active(req)
    require(req.get("schema")=="native_VAL000_verified22000_fresh5GiB_ROOT_request_v11"
        and req.get("root_authorizer")=="root_agent" and req.get("execution_authorized") is True
        and req["source_manifest_ref"]==ref(MANIFEST)
        and req["original_input_ref"]==ORIGINAL_INPUT and req["checkpoint_audit_ref"]==V11_SAVED22000_AUDIT_REF
        and req["resume_checkpoint_ref"]==V11_RESUME_CHECKPOINT_REF
        and req["config_ref"]==old["refs"]["config"] and req["runtime_ref"]==old["refs"]["runtime"]
        and req["ROOT_directive_ref"]==a["ROOT_directive_ref"]
        and req["independent_source_review_ref"]==a["independent_source_review_ref"]
        and req["expires_at_utc"]==s["expires_at_utc"]==a["expires_at_utc"]==old["expires_at_utc"]
        and req["fresh_caps"]==CAPS and req["original_caps"]==ORIGINAL_CAPS,
        "Actual exact-checkpoint source/review/runtime/original-deadline fresh admission")
    packet=req["packet_directory"];output=req["output_directory"]
    require(packet.startswith("research/reviews/native_VAL000_fresh_continuation_ROOT_")
        and output.startswith("experiments/results/native_VAL000_verified22000_continuation_")
        and record["path"]==packet+"/input.json"
        and not local(output).is_relative_to(local(old["output_directory"]))
        and not local(packet).is_relative_to(local(old["worker_auxiliary_scope"])),
        "Separate canonical fresh continuation packet/output, never inside or overlapping original charged roots")
    storage=read(s["refs"]["storage"]);closure=read(s["consumer_manifest_ref"])
    require(s.get("schema")=="native_VAL000_fresh5GiB_continuation_input_v11"
        and s.get("root_authorizer")=="root_agent" and s.get("execution_authorized") is True
        and s.get("GPU_authorized") is True and s.get("phase")=="mlm"
        and s["output_directory"]==output==a["fresh_science_directory"]
        and s["worker_auxiliary_scope"]==packet+"/worker_aux"==a["fresh_admin_directory"]
        and s["job_id"]==req["job_id"]==storage["job_id"]
        and s["refs"]["config"]==req["config_ref"] and s["refs"]["runtime"]==req["runtime_ref"]
        and s["created_at_utc"]==req["created_at_utc"]
        and s["source_manifest_ref"]==ref(MANIFEST)
        and closure==dict(source_manifest_ref=ref(MANIFEST),root_request_ref=s["root_stage_request_ref"],
            original_input_ref=ORIGINAL_INPUT,checkpoint_audit_ref=V11_SAVED22000_AUDIT_REF,
            config_ref=req["config_ref"],runtime_ref=req["runtime_ref"]),
        "Actual immutable separately admitted source/input/config/checkpoint closure")
    selected=req["accounting_reservation_bytes"]
    require(type(selected) is int and selected in ACCOUNTING_CHOICES_V11
        and a.get("fresh_accounting_reservation_bytes")==selected
        and req.get("reservation_only_downward_failstop_authorized") is (selected<CAPS["reservation_bytes"])
        and a.get("reservation_only_downward_failstop_authorized") is (selected<CAPS["reservation_bytes"]),
        "Explicit selected5GiB or17/4GiB accounting failstop; original5GiB envelope never reduced")
    require(storage["reserved_new_bytes"]==storage["reservation"]["reserved_bytes"]==selected
        and storage["output_directory"]==output
        and storage["reservation"]["output"]==str(local(output))
        and storage["reservation"]["auxiliary_scopes"]==[str(local(s["worker_auxiliary_scope"]))]
        and storage["completed_overlay_policy_ref"]==FRESH_POLICY_REF
        and storage["budget_components"]["consumer_adapter"]==FRESH_GUARD_REF,
        "Entire fresh output+AUX under its own actual explicitly selected current native14v3 reservation")
    needed=fresh_required_roles_v11(s["root_stage_request_ref"],req,s["consumer_manifest_ref"])
    require(len(storage["reservation"]["consumer_bindings"])<=64
        and all(r in storage["reservation"]["consumer_bindings"] for r in needed),
        "Every exact fresh phase/source/runtime/old-input role stays bound under original64 limit")
    FRESH_STATE=(copy.deepcopy(s),copy.deepcopy(req),copy.deepcopy(storage))
    view=dict(s,actual_execution_amendment_ref=AMENDMENT_REF,
        original_charged_output_directory=old["output_directory"],original_charged_auxiliary_scope=old["worker_auxiliary_scope"],
        actual_executed_adapter_manifest_ref=ref(MANIFEST))
    qview=dict(oldq,fresh_storage_job_id=s["job_id"],caps=copy.deepcopy(CAPS),capacity_policy_ref=FRESH_POLICY_REF,
        capacity_guard_ref=FRESH_GUARD_REF,actual_execution_amendment_ref=AMENDMENT_REF)
    return view,qview,cfg,storage,protocol

class FreshContinuationGuardV11:
    def __init__(self,retained,fresh):self.retained=retained;self.fresh=fresh
    def validate_storage_v9(self,s,cfg,storage,*,source_ref,config_ref,runtime_ref):
        active_amendment_v1()
        old,_,old_storage=ORIGINAL_STATE
        require(FRESH_STATE is not None and storage==FRESH_STATE[2] and s["output_directory"]==FRESH_STATE[0]["output_directory"]
            and s["worker_auxiliary_scope"]==FRESH_STATE[0]["worker_auxiliary_scope"]
            and source_ref==ref(ACTUAL_DRIVER) and config_ref==old["refs"]["config"] and runtime_ref==old["refs"]["runtime"],
            "Actual fresh execution source and unchanged scientific config/runtime; old charge authenticates no new writer")
        self.retained.validate_storage_v9(old,dict(output_cap_bytes=ORIGINAL_CAPS["reservation_bytes"]),old_storage,
            source_ref=original.ref(original.DRIVER),config_ref=config_ref,runtime_ref=runtime_ref)
        result=self.fresh.validate_storage_v9(s,cfg,storage,source_ref=source_ref,config_ref=config_ref,runtime_ref=runtime_ref)
        active_amendment_v1();return result
    def budget_check_v9(self,storage,**kwargs):
        active_amendment_v1()
        require(FRESH_STATE is not None and storage==FRESH_STATE[2],"Only actual fresh storage receives prospective new writes")
        self.retained.budget_check_v9(ORIGINAL_STATE[2],boundary=True,required_write_bytes=0)
        result=self.fresh.budget_check_v9(storage,**kwargs)
        active_amendment_v1();return result


def validate_liveness_component_v11(manifest):
    global _COMPONENT_CHECKED_V11
    require(manifest.get("liveness_component_manifest_ref")==LIVENESS_COMPONENT_MANIFEST_REF,
        "Final integration pins independently reviewed sealed liveness component")
    component=read(LIVENESS_COMPONENT_MANIFEST_REF)
    require(component["direct_source_ref"]["sha256"]==ref(DIRECT)["sha256"]
        and component["terminal_authority_AST_sha256"]==LIVENESS_COMPONENT_AUTHORITY_AST_SHA256,
        "Exact frozen direct source and terminal-snapshot authority predicate")
    if not _COMPONENT_CHECKED_V11:
        tree=ast.parse(local(SOURCE).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="terminal_snapshot_authority_v11")
        require(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            ==LIVENESS_COMPONENT_AUTHORITY_AST_SHA256,"Exact authority AST remains unchanged after storage integration")
        _COMPONENT_CHECKED_V11=True

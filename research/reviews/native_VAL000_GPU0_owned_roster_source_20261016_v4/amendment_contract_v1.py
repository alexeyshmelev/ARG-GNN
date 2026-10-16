"""Execution-only ROOT source bridge over immutable fully charged native v2 roots.
The original guard authenticates its original bound input/source. A separate exact
ROOT amendment authenticates this new adapter. No reserve/close/credit entrypoint.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,stat,sys,time
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2.common_v1 import *
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import common_v1 as original
HERE="research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4"
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
CHILD="same_charge_owned_roster_v4"
AMENDMENT_REF=None
ORIGINAL_STATE=None

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
    require(a.get("schema")=="native32_execution_only_same_charge_ROOT_amendment_v1"
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
    expected=[ref(HERE+"/"+name) for name in ("__init__.py","amendment_contract_v1.py","direct_gpu0_v1.py","execution_only_v1.py","corpus_metadata_bridge_v1.py")]
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
    raw(a["independent_source_review_ref"])
    original_s=read(ORIGINAL_INPUT);q=read(original_s["root_stage_request_ref"])
    anchor=datetime.fromisoformat(q["created_at_utc"]).timestamp()
    require(a.get("original48h_window_anchor_utc")==q["created_at_utc"]
        and time.time()<min(anchor+172800,datetime.fromisoformat(original.ORIGINAL_ABSOLUTE_EXPIRY).timestamp()),
        "Original48hour window anchored before first attempt, never reset")
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
        and a["original_caps"]==CAPS,"Original science/config/guard/caps/policy untouched")
    science=s["output_directory"]+"/"+CHILD;admin=s["worker_auxiliary_scope"]+"/"+CHILD
    require(a["fresh_science_directory"]==science and a["fresh_admin_directory"]==admin,
        "Exact distinct validated fresh descendant science/admin paths")
    for value,parent in [(science,s["output_directory"]),(admin,s["worker_auxiliary_scope"])]:
        require(local(value).parent==local(parent) and local(parent).is_dir(),"Fresh children strictly inside actual original charged roots")
    ORIGINAL_STATE=(copy.deepcopy(s),copy.deepcopy(q),copy.deepcopy(storage))
    view=dict(s,output_directory=science,worker_auxiliary_scope=admin,actual_execution_amendment_ref=AMENDMENT_REF,
        original_charged_output_directory=s["output_directory"],original_charged_auxiliary_scope=s["worker_auxiliary_scope"],
        actual_executed_adapter_manifest_ref=ref(MANIFEST))
    qview=dict(q,actual_execution_amendment_ref=AMENDMENT_REF)
    return view,qview,cfg,storage,protocol

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
    require(consumer is True and ORIGINAL_STATE is not None,"Execution-only read/check role, no mutation/admission interface")
    return OriginalGuardWithExplicitSourceBridge(original.guard_v1(ORIGINAL_STATE[1],consumer=True))

def charged_census_v1(spec,*,required_write_bytes=0):
    require(ORIGINAL_STATE is not None,"Whole original charged roots required")
    # Includes original failed bytes and the fresh children. No subtraction or credit.
    return original.charged_census_v1(ORIGINAL_STATE[0],required_write_bytes=required_write_bytes)

def compact_v1(spec,name,value,g,storage,*,cap=2**20):
    active_amendment_v1();directory=local(spec["worker_auxiliary_scope"])
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(data)<=cap,"Bounded operational evidence")
    charged_census_v1(spec,required_write_bytes=len(data)+8192)
    g.budget_check_v9(storage,boundary=True,required_write_bytes=len(data)+8192)
    active_amendment_v1()  # Immediately before the new write, after census and guard.
    return save(directory,name,value)

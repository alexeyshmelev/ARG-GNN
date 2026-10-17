"""Inactive bounded original C3 remaining-cell contract; all runtime operations require ROOT authority."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,stat,sys,time
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import common_v1 as original
ROOT=original.ROOT
HERE="research/reviews/C3_prepared_feature_future_integration_source_20261017_v3"
DRIVER=HERE+"/execution_v1.py";DIRECT=DRIVER;WORKER=HERE+"/worker_v1.py";MANIFEST=HERE+"/source_manifest.v1.json";BUILDER=HERE+"/contract_v1.py"
PARENT="American_C3_short20_TRAIN_original5VAL"
RUNTIME_REF={'path': 'research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json', 'sha256': 'e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602'}
VAL=tuple("american_c3_val_%03d"%i for i in range(5));EPOCHS=(5,10,15,20,25,30)
GPU_PHASES=tuple("predict_val_%03d"%i for i in (0,1,2));PHASES=GPU_PHASES+("matrix",)
BASE_CAPS=dict(CPU_threads=32,dataloader_workers=0,reservation_bytes=32*2**20,log_bytes=256*2**10,max_seconds=86400,max_cpu_rss_bytes=128*2**30,headroom_bytes=4*2**30,cleanup_seconds=30,work_cutoff_margin_seconds=300,terminal_evidence_margin_seconds=150,max_checkpoint_bytes=3*2**30,max_checkpoint_files=2)
CAPS=dict(BASE_CAPS,max_cuda_reserved_bytes=0)
OWNED_LIMITS_V8=dict(pending=1024,discoveries=2**28,groups=65536,adopted_waits=2**28,history_rows=16)
MINIMAL_WAIT_CAP_BYTES=65536;TERMINAL_RECEIPT_CAP_BYTES=8*2**20
GPU0_UUID=original.GPU0_UUID;BASE_POOL=original.BASE_POOL;GPU0_GUARD=original.GPU0_GUARD;PIDFD=original.PIDFD;OWNED=original.OWNED;NATIVE_PYTHON_REF=dict(path=".runtime/env/bin/python3.11",runtime_metadata_ref=RUNTIME_REF)
require=original.require;local=original.local;read=original.read;raw=original.raw;ref=original.ref;load=original.load;save=original.save
PINS=None;ROOT_INPUT_REF=None;STATE=None;GUARD=None;POLICY=None;SELECTOR_REF=None;_GUARD=None;_MEMO=None
def pins_v1():
 global PINS
 if PINS is None:PINS=read(ref(HERE+"/pins.v1.json"))
 return PINS
def active(v):
 now=datetime.now(timezone.utc)
 require(v.get("root_authorizer")=="root_agent"and v.get("status",v.get("active"))in("active",True)and v.get("template_only")is not True and datetime.fromisoformat(v["created_at_utc"])<=now<datetime.fromisoformat(v["expires_at_utc"]),"Exact finite active ROOT metadata")
def configure_selector_v1(q):
 global GUARD,POLICY,SELECTOR_REF,_GUARD
 r=q["storage_policy_selector_ref"];p=pins_v1()["pins"]
 require(r==p["CENSUS_SELECTOR"]and q["Census_verification_ref"]==p["CENSUS_VERIFICATION_OUTER"],"Explicit exact already verified Census selector and consumed verification receipt")
 verified=read(q["Census_verification_ref"]);inner=read(p["CENSUS_VERIFICATION"])
 require(verified.get("schema")=="ROOT_Census_observation_fresh_child_verification_actual_outer_consumed_v1"and verified.get("selector_ref")==r and verified.get("verification_ref")==p["CENSUS_VERIFICATION"]and verified.get("actual_terminal_consumed")is True and type(verified.get("actual_outer_exit_code"))is int and verified["actual_outer_exit_code"]==0 and type(verified.get("tool_session_id"))is int and verified["tool_session_id"]>0 and inner.get("schema")=="ROOT_Census_observation_fresh_child_verification_v1"and inner.get("selector_ref")==r and inner.get("actual_direct_guard_roles")==47 and inner.get("all_parent_pins_and_46roles_preserved")is True,"Genuine consumed ROOT verification0 and full47 roles, no inferred activation")
 p=pins_v1()["pins"];v=read(r)
 require(v.get("active")is True and v.get("root_authorizer")=="root_agent"and v.get("source_ref")==v.get("selector_source_ref")==p["CENSUS_GUARD"]and v.get("accounting_mode")=="Census_observation_two_scope_retention_v1"and verified.get("policy_ref")==v["policy_ref"],"Exact reviewed two-Census selector/policy; no discovery or fallback")
 if _GUARD is not None:
  require(SELECTOR_REF==r and POLICY==v["policy_ref"]and GUARD==p["CENSUS_GUARD"],"Immutable selector/policy for interpreter");return _GUARD
 g=load(p["CENSUS_GUARD"],"_C3_remaining_exact_Census")
 require(g.load_selected_guard_v1(r)is g,"Actual selected child singleton")
 SELECTOR_REF=copy.deepcopy(r);POLICY=copy.deepcopy(v["policy_ref"]);GUARD=copy.deepcopy(p["CENSUS_GUARD"]);_GUARD=g;return g
def selected_guard_v1(*,consumer):
 global _MEMO
 require(type(consumer)is bool and _GUARD is not None,"Explicit selected guard role")
 if not consumer:
  require(_MEMO is None and all(getattr(_GUARD.budget.v4,n).__module__==_GUARD.budget.v4.__name__ for n in("canonical","file_bytes","mutex")),"Fresh unwrapped ROOT mutation interpreter");return _GUARD
 if _MEMO is None:
  memo=load(original.NAMESPACE_MEMO,"_C3_remaining_read_only_memo")
  _MEMO=memo.install_consumer_v1(_GUARD.budget.v4,_GUARD.budget,consumer_role="read_only_consumer",expected_source_path=ROOT/original.V4["path"],expected_source_sha256=original.V4["sha256"])
 return _GUARD
def review_v1(r):
 v=read(r)
 require(v.get("independent")is True and v.get("verdict")in("SOURCE_PASS","PASS_SOURCE_ONLY","PASS")and v.get("candidate_manifest_ref",v.get("source_manifest_ref"))==ref(MANIFEST)and v.get("blockers")==[],"Exact complete independent source review")
 return v
def manifest_v1():
 m=read(ref(MANIFEST))
 for r in m["executed_source_refs"]+m["source_authentication_refs"]:raw(r)
 return m
def training_v1():
 p=pins_v1()["pins"];legacy=load(p["SAVED_CONTRACT"],"_C3_remaining_original_saved600_contract")
 bundle,training=legacy.training_ready_v1(p["ARTIFACT_ACCEPTANCE"])
 frozen=read(p["FROZEN"])
 require(training["checkpoint_candidates"]==frozen["checkpoint_candidates"]and [x["epoch"]for x in training["checkpoint_candidates"]]==list(EPOCHS),"All6 original saved600 checkpoints remain frozen")
 return bundle,training
def original_terminal_values_v1(e,parent):
 require(set(e)=={"input_ref","launch_ref","wait_ref","outer_ref","permanent_revocation_ref"},"Exact original CPU disposition evidence")
 expected=(pins_v1()["old_CPU_inputs"]|pins_v1()["completed_inputs"])[parent]
 require(e["input_ref"]==expected,"Exact original already-admitted CPU input")
 inp=read(expected);launch=read(e["launch_ref"]);wait=read(e["wait_ref"]);outer=read(e["outer_ref"]);rev=read(e["permanent_revocation_ref"])
 phase="predict_val_"+parent[-3:];driver=inp["root_driver_ref"]
 require(launch.get("input_ref")==wait.get("input_ref")==expected and launch.get("driver_ref")==wait.get("driver_ref")==driver and launch.get("phase")==wait.get("phase")==phase and launch.get("actual_launch")is True and launch.get("pid")==wait.get("pid")and wait.get("launch_ref")==e["launch_ref"]and type(wait.get("returncode"))is int and wait.get("actual_direct_launcher_wait")is True and wait.get("all_owned_processes_and_groups_drained")is True,"Genuine original direct terminal and complete drain; nonzero stays nonzero")
 require(type(outer.get("actual_exit_code"))is int and outer.get("actual_terminal_consumed")is True and outer.get("actual_wait_ref")==e["wait_ref"]and outer.get("actual_direct_worker_returncode")==wait["returncode"]and type(outer.get("actual_tool_session_id"))is int and outer["actual_tool_session_id"]>0,"Actual consumed original terminal, preserving its own exit and direct status separately")
 require(rev.get("root_authorizer")=="root_agent"and rev.get("input_ref")==expected and rev.get("future_writes_and_relaunch_permanently_revoked")is True and rev.get("preserve_all_files_and_original_charge")is True and rev.get("old_wait_ref")==e["wait_ref"]and rev.get("old_outer_ref")==e["outer_ref"]and datetime.fromisoformat(rev["revoked_at_utc"])>=datetime.fromisoformat(wait["finished_utc"]),"Explicit permanent original-writer revocation without old success or credit")
 return inp,wait,outer,rev
def election_values_v1(record):return election_value_v1(read(record))
def election_value_v1(e):
 active(e)
 require(e.get("schema")=="C3_ROOT_poststop_retained_cells_acceptance_v1"and e.get("saved600_acceptance_ref")==pins_v1()["pins"]["ARTIFACT_ACCEPTANCE"]and e.get("inventory_sealed_after_all_original_terminals_and_drains")is True and e.get("private_truth_opened")is False and e.get("original_nonzero_waits_preserved")is True and set(e["parents"])==set(VAL),"Separate truthful ROOT post-stop artifact acceptance, original5VAL and no truth")
 _,t=training_v1();candidates={x["epoch"]:x["checkpoint_ref"]for x in t["checkpoint_candidates"]}
 retained={};pending={}
 for parent,row in e["parents"].items():
  inp,wait,outer,rev=original_terminal_values_v1(row["original_disposition"],parent)
  require(datetime.fromisoformat(row["inventory_sealed_at_utc"])>=datetime.fromisoformat(rev["revoked_at_utc"]),"Inventory sealed only after permanent revocation")
  refs=row["retained_prediction_seal_refs"];cells={}
  for r in refs:
   seal=read(r);epoch=seal["epoch"]
   require(epoch in candidates and epoch not in cells and seal.get("parent_id")==parent and seal.get("status")=="sealed_C3_original_heldout_predictions"and seal.get("split")=="val"and seal.get("records")==10000 and seal.get("checkpoint_ref")==candidates[epoch]and seal.get("training_completion_ref")==tref_v1()and seal.get("private_truth_opened")is False and seal.get("probabilities_modified")is False and seal["prediction_ref"]["path"].startswith(inp["output_directory"]+"/")and r["path"].startswith(inp["output_directory"]+"/"),"Retained exact original immutable cell metadata/provenance; payload is not opened here")
   cells[epoch]=r
  missing=sorted(set(EPOCHS)-set(cells))
  require((parent in pins_v1()["old_CPU_inputs"]and set(missing)<=set((25,30))and set((5,10,15,20))<=set(cells))or(parent in pins_v1()["completed_inputs"]and not missing),"Only post-stop missing25/30 may run; every answered cell reused")
  require(row["pending_epochs"]==missing,"ROOT sealed inventory determines exact pending subset")
  retained[parent]=cells;pending[parent]=missing
 return e,retained,pending
def draft_election_v1(request_ref):
 q=read(request_ref);active(q)
 require(q.get("schema")=="C3_ROOT_draft_poststop_cell_election_request_v1"and q.get("source_manifest_ref")==ref(MANIFEST),"Exact reviewed metadata-only election producer")
 review_v1(q["source_review_ref"]);manifest_v1()
 inventory=read(q["inventory_ref"])
 require(inventory.get("schema")=="C3_ROOT_poststop_original_prediction_inventory_v1"and inventory.get("root_authorizer")=="root_agent"and inventory.get("private_truth_opened")is False and set(inventory["parents"])==set(VAL),"Actual ROOT post-stop sealed all-parent inventory")
 rows=copy.deepcopy(inventory["parents"])
 for parent,row in rows.items():
  epochs=[read(r)["epoch"]for r in row["retained_prediction_seal_refs"]]
  require(len(epochs)==len(set(epochs)),"No duplicate chosen old cell")
  row["pending_epochs"]=sorted(set(EPOCHS)-set(epochs))
 value=dict(schema="C3_ROOT_poststop_retained_cells_acceptance_v1",root_authorizer="root_agent",status="active",created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],saved600_acceptance_ref=pins_v1()["pins"]["ARTIFACT_ACCEPTANCE"],source_manifest_ref=ref(MANIFEST),source_review_ref=q["source_review_ref"],poststop_inventory_ref=q["inventory_ref"],inventory_sealed_after_all_original_terminals_and_drains=True,private_truth_opened=False,original_nonzero_waits_preserved=True,parents=rows,accepts_saved_cells_only=True,original_invocation_success_claimed=False,storage_close_or_credit_authorized=False)
 election_value_v1(value)
 return value
def tref_v1():return read(pins_v1()["pins"]["FROZEN"])["training_completion_ref"]
def peer_values_v1(record):
 e=read(record);active(e);p=pins_v1()["pins"]
 require(e.get("schema")=="C3_ROOT_exact_live_H058_peer_v1" and e.get("H058_input_ref")==p["H058_INPUT"]
  and e.get("H058_request_ref")==p["H058_REQUEST"] and e.get("controller_terminal_qualification_ref")==p["H058_TERMINAL"]
  and e.get("controller_capability_ref")==p["CONTROLLER_CAP"] and e.get("H058_full_cap_plus_margin_bytes")==30*2**30
  and e.get("C3_reserved_bytes")==26*2**30 and e.get("headroom_bytes")==4*2**30
  and e.get("H058_source_or_registry_mutation_authorized")is False
  and e.get("terminal_peer_absence_preserves_full30GiB")is True,"Exact current v9 peer and unchanged complete reservations")
 launch_ref=e["H058_launch_ref"]
 require(type(launch_ref)is dict and set(launch_ref)=={"path","sha256"}
  and launch_ref==p["H058_LAUNCH"],"Exact immutable actual current v9 launch ref, never an old or invented peer")
 qualification=read(e["peer_source_qualification_ref"]);active(qualification)
 require(qualification.get("schema")=="C3_ROOT_exact_H058_v9_peer_source_and_launch_qualification_v1"
  and qualification.get("H058_input_ref")==p["H058_INPUT"] and qualification.get("H058_request_ref")==p["H058_REQUEST"]
  and qualification.get("H058_launch_ref")==launch_ref
  and qualification.get("H058_direct_source_ref")==p["H058_DIRECT"]
  and qualification.get("H058_common_source_ref")==p["H058_COMMON"]
  and qualification.get("H058_whole_source_review_ref")==p["H058_REVIEW"]
  and qualification.get("H058_source_manifest_ref")==p["H058_MANIFEST"]
  and qualification.get("controller_terminal_qualification_ref")==p["H058_TERMINAL"]
  and qualification.get("controller_capability_ref")==p["CONTROLLER_CAP"]
  and qualification.get("all_original_controller_capacity_native_absence_title_checks_preserved")is True
  and qualification.get("original_C3_26_H058_30_EVAL_10_headroom_4_GiB_envelopes_preserved")is True
  and qualification.get("no_controller_peer_or_original_job_mutation_authorized")is True,
  "Separate exact ROOT current-peer qualification before preparation")
 review=read(p["H058_REVIEW"])
 require(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS" and review.get("blockers")==[]
  and review.get("source_ref")==p["H058_DRIVER"] and review.get("source_manifest_ref")==p["H058_MANIFEST"],
  "Genuine current H058v9 whole source pass; historical v7 review is not current authority")
 raw(p["H058_DIRECT"]);raw(p["H058_COMMON"])
 inp=read(p["H058_INPUT"]);launch=read(launch_ref);q=read(p["H058_REQUEST"])
 require(inp["root_stage_request_ref"]==p["H058_REQUEST"] and launch["input_ref"]==p["H058_INPUT"]
  and launch.get("schema")=="H058_fixed_full_parent_actual_child_launch_v1"
  and launch["borrowed_physical_inode"]is True and launch.get("controller_inheritance_claimed")is False
  and q["caps"]["max_cuda_reserved_bytes"]==28*2**30 and q["caps"]["C3_fixed_envelope_bytes"]==26*2**30
  and q["controller_terminal_qualification_ref"]==p["H058_TERMINAL"]
  and q["source_review_ref"]==p["H058_REVIEW"] and q["adapter_manifest_ref"]==p["H058_MANIFEST"],
  "Actual admitted current v9 input/launch/source/caps lineage")
 ownership=launch["ownership"]
 require("owned_identities"not in ownership and ownership.get("owned_tree_constructed")is True
  and ownership.get("complete_owned_roster_known")is True and ownership.get("owned_roster_truncated")is False
  and ownership.get("historical_roster_summarized")is True,
  "Actual v9 summarized ownership schema; legacy injected format is not accepted")
 rows=ownership["current_owned_identities"]
 attachment=read(p["H058_ATTACHMENT"])
 require(launch.get("attachment_ref")==p["H058_ATTACHMENT"]
  and attachment.get("schema")=="H058_borrowed_GPU0_attachment_v1"
  and attachment.get("borrowed_physical_inode")is True
  and attachment.get("H058_logical_lock_owned_by_supervisor")is True
  and attachment.get("physical_exclusive_lock_owner")=="unchanged_original_controller"
  and attachment.get("job_id")==q["job_id"]
  and attachment.get("direct_attachment_authority_ref")==q["direct_attachment_authority_ref"]
  and attachment.get("physical_gpu")==0 and attachment.get("gpu_uuid")==GPU0_UUID,
  "Exact actual current launch/borrowed attachment/admitted request metadata lineage")
 require(len([r for r in rows if r["pid"]==launch["pid"]])==1
  and e["worker_identity"]==next(r for r in rows if r["pid"]==launch["pid"])
  and qualification["worker_identity"]==e["worker_identity"],
  "Exact current peer PID/start from genuine launch, never guessed")
 return e,q
def caps_v1(phase):return dict(BASE_CAPS,max_cuda_reserved_bytes=8*2**30 if phase in GPU_PHASES else 0)
def config_v1(phase):return dict(schema="C3_prepared_feature_operational_config_v7",output_cap_bytes=32*2**20,phase=phase,caps=caps_v1(phase),model_checkpoint_recurrence_and_scorer_unchanged=True,preprocessing_implementation_changed=True,FP32_support_preprocessing_adaptation="accepted_v3_compensated_FP32")
def owned_drain_values_v2(wait,expected_driver=None):
 if expected_driver is None:expected_driver=ref(DRIVER)
 require(wait.get("schema")=="C3_remaining_cells_actual_wait_v2"and wait.get("owned_roster_policy_ref")==expected_driver and wait.get("owned_roster_policy")==OWNED_LIMITS_V8,"Exact executed bounded-live ownership policy")
 own=wait["ownership"];groups=wait["group_absence_summary"];adopted=wait["adopted_waits"]
 require(own.get("complete_owned_roster_known")is True and own.get("owned_roster_truncated")is False
  and own.get("current_owned_identities")==[]and own.get("limits")==OWNED_LIMITS_V8
  and own.get("retired_exact_identities_confirmed_absent_or_replaced")is True
  and wait.get("owned_identity_absence")is True and wait.get("group_observations_summarized")is True,"Complete bounded current roster and exact identity retirement")
 discovered=own["discovered_identity_history"]["count"];retired=own["retired_identity_history"]["count"];count=own["group_count"]
 require(type(discovered)is int and 1<=discovered<=OWNED_LIMITS_V8["discoveries"]
  and type(retired)is int and retired==discovered
  and sum(own["retirement_reasons"].values())==retired
  and type(count)is int and 1<=count<=OWNED_LIMITS_V8["groups"],"Every discovered exact identity retired; every group retained")
 require(groups.get("all_owned_groups_observed_absent")is True and groups.get("scan_finished_within_deadline")is True and groups.get("errors")==[]
  and all(type(groups.get(k))is int for k in("expected_group_count","observed_group_count","absent_group_count"))
  and groups.get("expected_group_count")==groups.get("observed_group_count")==groups.get("absent_group_count")==count
  and groups["history"]["count"]==count,"All retained groups genuinely observed ESRCH within original cleanup")
 require(type(adopted["count"])is int and 0<=adopted["count"]<=OWNED_LIMITS_V8["adopted_waits"]
  and adopted["nonzero_returncodes"]==0,"No omitted nonzero adopted OS wait")
 histories=[own["discovered_identity_history"],own["retired_identity_history"],own["reused_numeric_PID_history"],groups["history"],adopted]
 for history in histories:
  digest=history["ordered_chain_sha256"]
  require(len(history["first"])<=16 and len(history["last"])<=16 and history["full_historical_roster_serialized"]is False
   and type(digest)is str and len(digest)==64 and all(c in "0123456789abcdef"for c in digest),"Bounded first/last/count/digest ownership evidence")
 require(wait["group_observations"]==groups["history"]["first"]+groups["history"]["last"]
  and wait["group_observations"]and all(r.get("absent")is True for r in wait["group_observations"]),"Retained group samples bind full group absence summary")
 return wait

def new_stage_values_v1(e, parent, election, pending):
    require(set(e) == {'input_ref', 'completion_ref', 'wait_ref', 'outer_ref'}, 'Exact new successful remaining stage evidence')
    s = read(e['input_ref'])
    q = read(s['root_stage_request_ref'])
    prior_v5_failure_resolution_values_v3(q,historical=True)
    done = read(e['completion_ref'])
    wait = read(e['wait_ref'])
    outer = read(e['outer_ref'])
    require(parent in VAL[:3] and pending==[30],"Only fresh full-parent epoch30 VAL000/001/002 stages")
    require(e["input_ref"]!=pins_v1()["preserved_v5_VAL000_input"],"Stopped failed v5 input is historical, never a successful new stage")
    expected_manifest=ref(MANIFEST);expected_driver=ref(DRIVER)
    source_review=read(q["source_review_ref"])
    require(source_review.get("independent")is True and source_review.get("verdict")=="SOURCE_PASS"
      and source_review.get("blockers")==[]
      and source_review.get("candidate_manifest_ref",source_review.get("source_manifest_ref"))==expected_manifest,
      "Independent exact per-parent producer source review")
    owned_drain_values_v2(wait,expected_driver)
    require(s['source_manifest_ref'] == expected_manifest and q['source_manifest_ref'] == expected_manifest and (s['phase'] == q['phase'] == 'predict_val_' + parent[-3:]) and (q['cell_election_ref'] == election) and (q['parent_id'] == parent) and (q['pending_epochs'] == pending), 'Exact same source/election/parent/pending actual new producer')
    require(done['status'] == 'completed_C3_remaining_cells' and done['parent_id'] == PARENT and (done['original_VAL_parent_id'] == parent) and (done['child_input_ref'] == e['input_ref']) and (done['actual_execution_source_manifest_ref'] == expected_manifest) and (done['cell_election_ref'] == election) and (done['epochs'] == pending) and (done['training_completion_ref'] == tref_v1()) and (done['query_ancestry_truth_opened'] is False), 'Genuine own exact completed new cells')
    require(type(wait.get('returncode')) is int and wait['returncode'] == 0 and (wait['actual_direct_worker_wait'] is True) and (wait['actual_OS_wait_status'] == 0) and (wait['all_owned_processes_and_groups_drained'] is True) and (wait['complete_owned_roster_known'] is True) and (wait['completed_ref'] == e['completion_ref']) and (wait['input_ref'] == e['input_ref']) and (wait['adapter_manifest_ref'] == expected_manifest) and (wait['operational_failure'] is None) and wait['group_observations'] and all((x['absent'] is True for x in wait['group_observations'])), 'New GPU cells require actual OSwait0 and complete owned drain')
    require(outer['root_authorizer'] == 'root_agent' and type(outer['actual_exit_code']) is int and (outer['actual_exit_code'] == 0) and (outer['actual_terminal_consumed'] is True) and (outer['actual_wait_ref'] == e['wait_ref']) and (outer['actual_direct_worker_returncode'] == 0) and (type(outer['actual_tool_session_id']) is int) and (outer['actual_tool_session_id'] > 0), 'Genuine consumed new outer0')
    result = read(done['result_ref'])
    require(result['status'] == 'sealed_C3_original_remaining_cells' and result['parent_id'] == parent and (result['epochs'] == pending) and (result['training_completion_ref'] == tref_v1()) and (result['cell_election_ref'] == election) and (result['private_truth_opened'] is False) and (len(result['prediction_seal_refs']) == len(pending)), 'Only actually completed selected new cells')
    rows = {}
    for r in result['prediction_seal_refs']:
        seal = read(r)
        epoch = seal['epoch']
        require(epoch in pending and epoch not in rows and (seal['parent_id'] == parent) and (seal['training_completion_ref'] == tref_v1()) and (seal['private_truth_opened'] is False) and r['path'].startswith(s['output_directory'] + '/') and seal['prediction_ref']['path'].startswith(s['output_directory'] + '/'), 'Own immutable new seal provenance')
        rows[epoch] = r
    require(set(rows) == set(pending), 'Exact complete requested subset')
    return rows

def registration_values_v5(q,request_ref):
 r=q["registration_ref"];require(type(r)is dict and set(r)=={"path","sha256"}
  and r["path"].startswith("research/reviews/C3_prepared_feature_v7_ROOT_preparation_")
  and request_ref["path"].startswith("research/reviews/C3_prepared_feature_v7_ROOT_preparation_"),
  "Fresh v5 registration and request namespace")
 v=read(r);active(v);p=pins_v1()["pins"]
 require(v.get("schema")=="C3_prepared_feature_v7_ROOT_registration_v1"
  and all(v.get(k)==q.get(k) for k in ("source_manifest_ref","source_review_ref","phase","job_id","output_directory","worker_auxiliary_scope","cell_election_ref","peer_launch_capsule_ref","prior_v5_stop_ref","prior_v5_known_targets_ref","prior_v5_resolution_ref"))
  and v.get("C3_v3_permanent_revocation_ref")==p["C3_V3_REVOCATION"]
  and v.get("fresh_v7_admission_and_one_shot_caller_required")is True
  and v.get("old_v3_charge_preserved_bytes")==32*2**20
  and v.get("old_v3_Python_supervisor_wait")=="UNKNOWN"
  and v.get("old_v3_authority_or_admission_reuse_authorized")is False
  and v.get("automatic_retry_or_credit_authorized")is False,
  "Exact fresh v5 registration; revoked v3 is historical only and retains its charge")
 rev=read(p["C3_V3_REVOCATION"])
 require(rev.get("all_C3_remaining_v3_future_execution_resume_retry_write_routes_permanently_cooperatively_revoked")is True
  and rev.get("original_python_supervisor_direct_wait")=="UNKNOWN"
  and rev.get("does_not_release_any_old_storage_charge")is True,
  "ROOT permanent revocation and unknown original Python wait preserved")
 return v
def prior_v5_failure_resolution_values_v3(q,*,historical=False):
 p=pins_v1()
 require(q.get("prior_v5_stop_ref")==p["prior_v5_stop"],"Exact genuine stopped failed v5 record")
 stop=read(p["prior_v5_stop"])
 require(stop.get("schema")=="ROOT_C3_v5_actual_stop_outer1_unresolved_cleanup_v1"
  and stop.get("root_authorizer")=="root_agent"and stop.get("actual_terminal_consumed")is True
  and type(stop.get("actual_outer_exit_code"))is int and stop["actual_outer_exit_code"]==1
  and stop.get("actual_tool_session_id")==87405 and stop.get("actual_direct_worker_wait")is True
  and stop.get("actual_direct_worker_returncode")==-15 and stop.get("actual_worker_OS_wait_status")==15
  and stop.get("actual_saved_wait_ref")is None
  and stop.get("original_all_owned_processes_and_groups_drained")is False
  and stop.get("full_historical_descendant_drain_not_claimed")is True
  and stop.get("owned_output_files")==[]
  and stop.get("all_files_and_full_original_charge_preserved")is True
  and stop.get("storage_close_or_credit_authorized")is False
  and stop.get("new_prediction_or_retry_authorized")is False,
  "Genuine outer1 and worker SIGTERM wait, old full drain remains unproven and full charge preserved")
 for key in ("prior_v5_known_targets_ref","prior_v5_resolution_ref"):
  row=q.get(key)
  require(type(row)is dict and set(row)=={"path","sha256"}
   and row["path"].startswith("research/reviews/C3_prepared_feature_v7_ROOT_preparation_"),
   "Fresh exact ROOT prior-v5 proof/resolution namespace")
 proof=read(q["prior_v5_known_targets_ref"])
 require(proof.get("schema")=="ROOT_C3_v5_known_targets_poststop_observation_v1"
  and proof.get("root_authorizer")=="root_agent"and proof.get("actual_observation")is True
  and proof.get("stopped_failure_ref")==p["prior_v5_stop"]
  and proof.get("old_v5_input_ref")==p["preserved_v5_VAL000_input"]
  and proof.get("known_root_observations")==[
   dict(pid=66065,expected_start=491497125,present=False),
   dict(pid=70270,expected_start=491547684,present=False),
   dict(pid=66030,expected_start=None,present=False)]
  and all(type(row.get("pid"))is int and row.get("present")is False for row in proof["known_root_observations"])
  and proof.get("known_positive_group_observations")==[dict(pgid=66065,absent=True),dict(pgid=70270,absent=True)]
  and all(type(row.get("pgid"))is int and row.get("absent")is True for row in proof["known_positive_group_observations"])
  and proof.get("only_listed_known_targets_observed")is True
  and proof.get("historical_descendant_drain_proven")is False
  and proof.get("storage_close_or_credit_authorized")is False,
  "Fresh known-root and positive-group absence proof is not full historical descendant drain")
 proof_time=datetime.fromisoformat(proof["utc"]);request_time=datetime.fromisoformat(q["created_at_utc"])
 require(datetime.fromisoformat(stop["utc"])<=proof_time<=request_time
  and 0<=(request_time-proof_time).total_seconds()<=300,
  "Known-target proof after actual stop and within300seconds before new ROOT request")
 resolution=read(q["prior_v5_resolution_ref"])
 require(type(historical)is bool,"Explicit completed-stage history role")
 if historical:
  require(resolution.get("root_authorizer")=="root_agent"and resolution.get("status",resolution.get("active"))in("active",True)
   and resolution.get("template_only")is not True,"Genuine finite original ROOT resolution retained as completed-stage history")
 else:active(resolution)
 require(resolution.get("schema")=="C3_prepared_feature_v7_ROOT_prior_v5_resolution_v1"
  and resolution.get("stopped_failure_ref")==p["prior_v5_stop"]
  and resolution.get("known_targets_ref")==q["prior_v5_known_targets_ref"]
  and resolution.get("old_v5_input_ref")==p["preserved_v5_VAL000_input"]
  and resolution.get("old_v5_source_manifest_ref")==p["preserved_v5_manifest"]
  and resolution.get("old_v5_execution_ref")==p["preserved_v5_execution"]
  and all(resolution.get(k)==q.get(k)for k in("source_manifest_ref","source_review_ref","phase","job_id",
    "output_directory","worker_auxiliary_scope","cell_election_ref"))
  and resolution.get("all_old_v5_future_execution_resume_retry_write_routes_revoked")is True
  and resolution.get("old_v5_historical_descendant_drain")=="UNPROVEN"
  and resolution.get("residual_historical_uncertainty_accepted_for_new_phase_only")is True
  and resolution.get("does_not_certify_old_drain_or_cleanup_success")is True
  and resolution.get("new_exact_phase_adoption_authorized")is True
  and resolution.get("all_old_files_and_full_original_charge_preserved")is True
  and resolution.get("storage_close_or_credit_authorized")is False
  and resolution.get("old_authority_admission_or_output_reuse_authorized")is False
  and resolution.get("state_or_output_migration_authorized")is False
  and resolution.get("automatic_retry_authorized")is False
  and datetime.fromisoformat(resolution["created_at_utc"])<=request_time
  and datetime.fromisoformat(q["expires_at_utc"])<=datetime.fromisoformat(resolution["expires_at_utc"]),
  "Separate finite ROOT exact-phase resolution preserves uncertainty, revokes old route and never grants credit or migration")
 return stop,proof,resolution

def accepted_feature_component_values_v2():
 p=pins_v1()
 review=read(p["feature_review"])
 manifest=read(p["feature_manifest"])
 require(review.get("schema")=="C3_CPU_prepared_features_independent_review_v3"
  and review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"
  and review.get("blockers")==[]
  and review.get("candidate_ref")==p["feature_component"]
  and review.get("candidate_manifest_ref")==p["feature_manifest"],
  "Exact genuine accepted v3 candidate and manifest review, separate from whole integration review")
 require(manifest.get("schema")=="C3_inactive_stable_FP32_feature_candidate_manifest_v3"
  and manifest.get("component_ref")==p["feature_component"],
  "Exact accepted v3 candidate manifest component identity")
 return review,manifest

def validate_request_v1(record):
 q=read(record);active(q);registration_values_v5(q,record)
 require(0<(datetime.fromisoformat(q["expires_at_utc"])-datetime.fromisoformat(q["created_at_utc"])).total_seconds()<=86400,"At most original24h scientific interval")
 require(type(q["job_id"])is str and q["job_id"].startswith("C3prepared-v7-") and q["output_directory"].startswith("experiments/results/C3_prepared_feature_GPU0_v7_")and q["worker_auxiliary_scope"].startswith("research/reviews/C3_prepared_feature_GPU0_ROOT_v7_")and q["worker_auxiliary_scope"].endswith("/worker_aux"),"Fresh own output/packet roots outside every old producer")
 require(q.get("schema")=="C3_remaining_cells_ROOT_request_v1"and q.get("execution_authorized")is True and q.get("source_manifest_ref")==ref(MANIFEST)and q.get("phase")in PHASES and "truth"not in q and "VAL_truth_capability"not in q and "TEST"not in q,"Exact inactive-until-ROOT original remaining-cell or matrix authority")
 review_v1(q["source_review_ref"]);manifest_v1()
 prior_v5_failure_resolution_values_v3(q)
 configure_selector_v1(q)
 accepted_feature_component_values_v2()

 require(q["capacity_guard_ref"]==GUARD and q["capacity_policy_ref"]==POLICY and q["caps"]==caps_v1(q["phase"]),"Exact selected Census and original fixed finite8GiB evaluator/CPU matrix caps")
 e,retained,pending=election_values_v1(q["cell_election_ref"])
 require(sum(len(v)for v in retained.values())==27
  and all(pending[parent]==([30]if parent in VAL[:3]else[])for parent in VAL),
  "Exact preserved27retained plus only3unfinished epoch30cells")
 require(q.get("original_CPU_jobs_or_live_H058_mutation_authorized")is False and q.get("automatic_restart_or_fallback")is False,"No live job/controller changes or automatic work")
 if q["phase"]in GPU_PHASES:
  parent=VAL[int(q["phase"][-3:])]
  require(q["parent_id"]==parent and q["pending_epochs"]==pending[parent]and q["pending_epochs"]and q["plan_evidence"]["parent_id"]==parent,"Only exact post-stop pending cells for this original parent")
  legacy=load(pins_v1()["pins"]["SAVED_CONTRACT"],"_C3_remaining_original_plan_contract")
  require(q["plan_evidence"]==read(read(pins_v1()["old_CPU_inputs"][parent])["root_stage_request_ref"])["plan_evidence"],"Exact original admitted same-parent plan evidence and input support")
  legacy.plan_ready_v1(q["plan_evidence"],parent)
  peer_values_v1(q["peer_launch_capsule_ref"])
 else:
  require(q["pending_epochs"]==[]and q["parent_id"]==PARENT,"CPU-only cellwise matrix")
  needed={parent for parent in VAL if pending[parent]}
  require(set(q["new_stage_evidence"])==needed,"Complete actual new-stage wait/drain/outer0 evidence before matrix admission")
  for parent in needed:new_stage_values_v1(q["new_stage_evidence"][parent],parent,q["cell_election_ref"],pending[parent])
 return q,retained,pending
def set_amendment_v1(r):
 global ROOT_INPUT_REF
 require(ROOT_INPUT_REF is None or ROOT_INPUT_REF==r,"One exact current stage per interpreter");ROOT_INPUT_REF=r
def current_amendment_ref_v1():require(ROOT_INPUT_REF is not None,"Actual input installed");return ROOT_INPUT_REF
def active_amendment_v1():
 s=read(current_amendment_ref_v1())
 if STATE is None:q,_,_=validate_request_v1(s["root_stage_request_ref"])
 else:
  require(s==STATE[0],"Immutable input for current interpreter");q=read(s["root_stage_request_ref"]);require(q==STATE[2],"Immutable ROOT request for current interpreter")
 active(s);active(q)
 require(s["source_manifest_ref"]==ref(MANIFEST)and s["storage_policy_selector_ref"]==SELECTOR_REF and s["created_at_utc"]==q["created_at_utc"]and s["expires_at_utc"]==q["expires_at_utc"],"Exact current input/request/source/clock")
 return q,s,s
def validate_input_v1(r):
 global STATE
 require(r==ROOT_INPUT_REF,"Exact current input");q,s,_=active_amendment_v1()
 require(s.get("schema")=="C3_remaining_cells_actual_input_v1"and s.get("execution_authorized")is True and s["phase"]==q["phase"]and s["job_id"]==q["job_id"]and s["output_directory"]==q["output_directory"]and s["worker_auxiliary_scope"]==q["worker_auxiliary_scope"]and s["refs"]["runtime"]==RUNTIME_REF,"Exact admitted output/AUX/runtime")
 storage=read(s["refs"]["storage"]);cfg=read(s["refs"]["config"])
 require(cfg==config_v1(q["phase"])and storage["accounting_mode"]=="Census_observation_two_scope_retention_v1"and storage["storage_policy_selector_ref"]==SELECTOR_REF and storage["completed_overlay_policy_ref"]==POLICY and storage["job_id"]==s["job_id"]and storage["output_directory"]==s["output_directory"]and storage["reservation"]["output"]==str(local(s["output_directory"]))and storage["reserved_new_bytes"]==storage["reservation"]["reserved_bytes"]==32*2**20 and storage["reservation"]["auxiliary_scopes"]==[str(local(s["worker_auxiliary_scope"]))],"Actual fresh separate whole32MiB scope and exact Census charge")
 CAPS.clear();CAPS.update(q["caps"]);STATE=(copy.deepcopy(s),copy.deepcopy(storage),copy.deepcopy(q),copy.deepcopy(s))
 return s,q,cfg,storage,cfg
def guard_v1(q):configure_selector_v1(q);return selected_guard_v1(consumer=True)
def work_deadline_monotonic_v1():
 q,s,_=active_amendment_v1();end=min(datetime.fromisoformat(q["created_at_utc"]).timestamp()+CAPS["max_seconds"],datetime.fromisoformat(q["expires_at_utc"]).timestamp())
 return time.monotonic()+end-time.time()-CAPS["work_cutoff_margin_seconds"]
def require_work_window_v2(s):require(time.monotonic()<work_deadline_monotonic_v1(),"No fresh work within cleanup/evidence reserve")
def charged_census_v1(s,*,required_write_bytes=0):
 sizes=original._sizes(local(s["output_directory"]))+original._sizes(local(s["worker_auxiliary_scope"]))
 total=sum(n for _,n,_ in sizes);require(total+required_write_bytes<=32*2**20,"Whole scientific output+AUX/TMP/cache/log32MiB failstop");return {s["output_directory"]:total}
def compact_v1(s,name,value,g,storage,*,cap=2**20):
 active_amendment_v1();data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 require(len(data)<=cap,"Bounded operational metadata")
 charged_census_v1(s,required_write_bytes=len(data)+8192);g.budget_check_v9(storage,boundary=True,required_write_bytes=len(data)+8192)
 return save(local(s["worker_auxiliary_scope"]),name,value)
def worker_env_paths_v1(s):return original.worker_env_paths_v1(s)
def prepare_master_parents_v1(s,g,storage):
 require(local(s["output_directory"]).parent.is_dir()and local(s["worker_auxiliary_scope"]).parent.is_dir(),"Existing canonical fresh stage parents")
def validate_child_completion_v1(v,s,q):
 require(v.get("parent_id")==PARENT and v.get("child_input_ref")==ROOT_INPUT_REF and v.get("actual_execution_source_manifest_ref")==ref(MANIFEST)and v.get("query_ancestry_truth_opened")is False,"Exact source/input truth-free child")
 require(v.get("status")==("completed_C3_remaining_cells"if s["phase"]in GPU_PHASES else"completed_C3_remaining_cellwise_matrix"),"Genuine exact completed stage")
 if s["phase"]in GPU_PHASES:require(v["epochs"]==q["pending_epochs"],"Only requested pending cells completed")

ROOT_RECORD_CAP=454656
def record_budget_v1(directory,extra=0,*,worker_auxiliary_scope=None):
 total=4096;count=0
 require(type(extra)is int and extra>=0,"Nonnegative actual ROOT metadata write allowance")
 if worker_auxiliary_scope is not None:
  require(worker_auxiliary_scope.parent==directory and worker_auxiliary_scope.name=="worker_aux",
   "Only exact own separately charged worker_aux child directory")
 if directory.exists():
  for p in directory.iterdir():
   if worker_auxiliary_scope is not None and p==worker_auxiliary_scope:
    st=p.lstat()
    require(stat.S_ISDIR(st.st_mode)and not p.is_symlink(),"Exact nonsymlink charged AUX directory node")
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
     opened=os.fstat(fd);named=p.lstat()
     require(stat.S_ISDIR(opened.st_mode)and stat.S_ISDIR(named.st_mode)
      and (st.st_dev,st.st_ino)==(opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino),
      "Stable canonical nofollow identity for only charged AUX directory node")
    finally:os.close(fd)
    total+=4096;count+=1
   else:
    require(p.is_file()and not p.is_symlink()and p.suffix==".json"and p.lstat().st_nlink==1 and p.stat().st_size<=65536,"Own bounded flat ROOT metadata only")
    total+=p.stat().st_size+4096;count+=1
 require(total+extra<=ROOT_RECORD_CAP and count<64,"Combined ROOT records plus65536event+4096node within524288 allowance")
def root_save_v1(directory,name,v):
 data=(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 require(len(data)<=65536,"Bounded actual ROOT JSON");record_budget_v1(directory,len(data)+8192)
 return save(directory,name,v)
def direct_refs_v1(q,request_ref,config_ref):
 p=pins_v1()["pins"];m=read(ref(MANIFEST))
 # Five whole operational files (including inert __init__) plus the complete closure
 # anchor, review and actual request/config. Fixed library/history refs stay exact.
 rows=list(m["executed_source_refs"])+[ref(MANIFEST),q["source_review_ref"],request_ref,config_ref,RUNTIME_REF,p["CENSUS_GUARD"],GPU0_GUARD,BASE_POOL,OWNED,PIDFD,original.NAMESPACE_MEMO,p["ARTIFACT_ACCEPTANCE"],p["CONTROLLER_CAP"],p["H058_INPUT"],p["H058_TERMINAL"],p["H058_DIRECT"],q["cell_election_ref"]]
 if q["phase"]in GPU_PHASES:rows.extend([q["direct_attachment_authority_ref"],q["peer_launch_capsule_ref"]])
 return rows
def prepare_v1(request_ref):
 q,_,_=validate_request_v1(request_ref);g=selected_guard_v1(consumer=False)
 require(g.MAX_JSON==65536 and g.META_CHARGE==524288,"Exact combined metadata bound against pinned child")
 directory=local(q["worker_auxiliary_scope"]).parent
 require(not directory.exists()and not local(q["output_directory"]).exists(),"Fresh separately admitted packet/output")
 policy=g._json(POLICY);implementation=policy["implementation_ref"];launcher=policy["launcher_ref"]
 raw(implementation);raw(launcher);directory.mkdir()
 config_ref=root_save_v1(directory,"config.json",config_v1(q["phase"]))
 reservation=dict(key="H056/"+q["job_id"],output=str(local(q["output_directory"])),auxiliary_scopes=[str(local(q["worker_auxiliary_scope"]))],reserved_bytes=32*2**20)
 with g.budget.v4.mutex(ROOT):
  state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
  reservation["consumer_bindings"]=g.consumer_bindings_v1(list(state[3].values()),direct_refs_v1(q,request_ref,config_ref))
  authority=g.operation_draft_v1("reserve",reservation,expected_before=g._pins(state[2]),bindings=state[3])
  authority.update(active=True,expires_at_utc=q["expires_at_utc"]);authority_ref=root_save_v1(directory,"reserve_authority.json",authority)
 receipt=g.apply_reserve_v1(authority_ref,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
 receipt_ref=root_save_v1(directory,"reserve_receipt.json",receipt)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,job_id=q["job_id"],output_directory=q["output_directory"],reserved_new_bytes=32*2**20,reservation=receipt["value"],created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=GUARD),completed_overlay_policy_ref=POLICY,storage_policy_selector_ref=SELECTOR_REF,initial_ledger_pins=receipt["initial_ledger_pins"])
 storage_ref=root_save_v1(directory,"storage.json",storage)
 spec=dict(schema="C3_remaining_cells_actual_input_v1",root_authorizer="root_agent",status="active",execution_authorized=True,created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],phase=q["phase"],job_id=q["job_id"],output_directory=q["output_directory"],worker_auxiliary_scope=q["worker_auxiliary_scope"],source_manifest_ref=ref(MANIFEST),storage_policy_selector_ref=SELECTOR_REF,root_stage_request_ref=request_ref,refs=dict(config=config_ref,runtime=RUNTIME_REF,storage=storage_ref))
 input_ref=root_save_v1(directory,"input.json",spec);set_amendment_v1(input_ref);s,a,cfg,st,_=validate_input_v1(input_ref)
 g.validate_storage_v9(s,dict(output_cap_bytes=32*2**20),st,source_ref=ref(DRIVER),config_ref=config_ref,runtime_ref=RUNTIME_REF)
 admitted_ref=root_save_v1(directory,"admitted.json",dict(input_ref=input_ref,reserve_receipt_ref=receipt_ref,source_manifest_ref=ref(MANIFEST),scientific_execution_started=False))
 argv=[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",HERE.replace("/",".")+".execution_v1","--supervise-once","--input-ref",json.dumps(input_ref),"--amendment-ref",json.dumps(input_ref)]
 return dict(status="ADMITTED_NOT_EXECUTED",input_ref=input_ref,admitted_ref=admitted_ref,execute_argv=argv,automatic_execution=False)
def main(argv=None):
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group();actions.add_argument("--prepare-ref");actions.add_argument("--draft-election-ref");args=parser.parse_args(argv)
 if args.prepare_ref is None and args.draft_election_ref is None:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
 require(sys.platform.startswith("linux")and sys.dont_write_bytecode and "torch"not in sys.modules and "torch_geometric"not in sys.modules,"Fresh isolated stdlib ROOT preparation")
 print(json.dumps(draft_election_v1(json.loads(args.draft_election_ref))if args.draft_election_ref is not None else prepare_v1(json.loads(args.prepare_ref)),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

CLASSES=("EUR","EAS","AFR")
def _complete_plan_metadata_v6(plan,*,parent_id,length_bp):
    cfg=plan["config"];blocks=plan["blocks"]
    require(plan["schema"]=="american_c3_whole_tree_block_index_v1"
        and plan["parent_id"]==parent_id and plan["sequence_length_bp"]==length_bp
        and plan["class_order"]==list(CLASSES) and plan["haplotypes"]==1600
        and plan["original_reference_counts"]==[200,200,200]
        and plan["effective_reference_counts"]==[200,200,200]
        and plan["focal_records"]==10000 and plan["all_trees_and_samples_retained"] is True
        and plan["targets_opened"] is False and blocks and plan["trees"]>0
        and cfg["class_order"]==list(CLASSES) and cfg["expected_haplotypes"]==1600
        and cfg["input_reference_class_mask"] in (None,[True,True,True])
        and cfg["input_reference_sample_indices"] is None and cfg["allow_missing"] is False
        and cfg["donor_age_split"]==350.,"Exact complete original C3 baseline plan/mask/cardinality")
    cursor=0
    for b in blocks:
        require(type(b["start_tree"]) is int and type(b["stop_tree"]) is int
            and b["start_tree"]==cursor and b["stop_tree"]>cursor and b["nodes"]>0 and b["sites"]>=0
            and b["stop_tree"]-b["start_tree"]<=cfg["max_trees_per_block"]
            and b["nodes"]<=cfg["max_nodes_per_block"] and b["sites"]<=cfg["max_sites_per_block"],
            "Every original whole tree must occur in one contiguous bounded block")
        cursor=b["stop_tree"]
    require(cursor==plan["trees"] and sum(b["sites"] for b in blocks)==plan["sites"],
        "Complete original tree/site plan coverage")
def validate_all_original_plan_joins_v6(chosen):
 plans={}
 for parent in VAL:
  record=pins_v1()["original_plan_refs"][parent];plan=read(record)
  _complete_plan_metadata_v6(plan,parent_id=parent,length_bp=30000000)
  plans[parent]=record
  require(set(chosen[parent])==set(EPOCHS),"All six immutable states per original parent")
  for epoch in EPOCHS:
   seal=read(chosen[parent][epoch])
   require(seal["parent_id"]==parent and seal["epoch"]==epoch
    and seal["block_index_ref"]==record and seal["observation_ref"]==plan["observation_ref"]
    and seal["focal_registry_ref"]==plan["focal_registry_ref"]
    and seal["records"]==10000 and seal["private_truth_opened"]is False,
    "Every retained/new seal joins its exact original plan, observations and focals")
 return plans

def per_parent_source_policy_v6():
 return {parent:dict(source_manifest_ref=ref(MANIFEST),execution_ref=ref(DRIVER),
  stopped_failed_v5_input_historical_only=pins_v1()["preserved_v5_VAL000_input"],
  prior_v5_stop_ref=pins_v1()["prior_v5_stop"])for parent in VAL[:3]}

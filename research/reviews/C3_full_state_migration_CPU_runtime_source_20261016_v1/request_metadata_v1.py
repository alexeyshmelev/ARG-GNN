"""Deferred root metadata only. No checkpoint payload read or execution on import."""
from pathlib import Path
import hashlib,json,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_full_state_migration_CPU_runtime_source_20261016_v1"
SOURCE=HERE+"/request_metadata_v1.py"
SCOPE_CAP=8*2**20
FIXED_NONCHECKPOINT_ALLOWANCE=2**20
COPY_GROWTH_ALLOWANCE=128*2**10
MAX_INPUTS=6

def need(ok,message):
    if not ok:raise ValueError(message)

def exact_ref(r):
    need(isinstance(r,dict) and set(r)=={"path","sha256"} and type(r["path"]) is str
        and not Path(r["path"]).is_absolute() and ".." not in Path(r["path"]).parts
        and type(r["sha256"]) is str and len(r["sha256"])==64
        and all(c in "0123456789abcdef" for c in r["sha256"]),"Exact contained path/SHA")
    return r

def ref(path):
    p=ROOT/path
    need(p.resolve()==p and not p.is_symlink() and p.is_file() and p.stat().st_size<=2**20,"Small canonical source/metadata")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(r):
    exact_ref(r);need(ref(r["path"])==r,"Exact source/metadata bytes")
    return json.loads((ROOT/r["path"]).read_text())

def checkpoint_plan_v1(intent,actual_lengths):
    """Pure metadata arithmetic. Runtime restats every input before migration."""
    scheduled=intent["old_scheduled_checkpoints"];boundary=exact_ref(intent["old_boundary_ref"])
    need(isinstance(scheduled,list) and len(scheduled)<=5 and isinstance(actual_lengths,list),"At most5 prior scheduled candidates")
    expected=[]
    for item in scheduled:
        need(set(item)=={"epoch","global_step","checkpoint_ref"}
            and type(item["epoch"]) is int and 1<=item["epoch"]<=25
            and type(item["global_step"]) is int and item["global_step"]==20*item["epoch"],"Actual scheduled checkpoint metadata")
        expected.append(dict(name="checkpoint_epoch_%03d.pt"%item["epoch"],source_ref=exact_ref(item["checkpoint_ref"])))
    if boundary not in [x["source_ref"] for x in expected]:
        expected.append(dict(name="resume_complete_parent_boundary.pt",source_ref=boundary))
    need(1<=len(expected)<=MAX_INPUTS and len({x["name"] for x in expected})==len(expected)
        and len({x["source_ref"]["path"] for x in expected})==len(expected),"Finite unique actual checkpoint output plan")
    sizes={}
    for row in actual_lengths:
        need(set(row)=={"checkpoint_ref","actual_bytes"} and type(row["actual_bytes"]) is int
            and 0<row["actual_bytes"]<=intent["max_checkpoint_bytes"],"Actual positive source byte count")
        r=exact_ref(row["checkpoint_ref"]);key=(r["path"],r["sha256"])
        need(key not in sizes,"No duplicate actual source length")
        sizes[key]=row["actual_bytes"]
    need(set(sizes)=={(x["source_ref"]["path"],x["source_ref"]["sha256"]) for x in expected},"Every and only actual distinct checkpoint length")
    for row in expected:
        size=sizes[(row["source_ref"]["path"],row["source_ref"]["sha256"])]
        row["actual_source_bytes"]=size
        row["maximum_migrated_serialized_bytes"]=((size+COPY_GROWTH_ALLOWANCE+4095)//4096)*4096
        row["maximum_guarded_write_charge_bytes"]=row["maximum_migrated_serialized_bytes"]+8192
    total=FIXED_NONCHECKPOINT_ALLOWANCE+sum(x["maximum_guarded_write_charge_bytes"] for x in expected)
    need(total<=SCOPE_CAP,"Actual count-derived complete scope exceeds8MiB, no implicit cap expansion")
    return dict(schema="C3_CPU_migration_count_derived_output_plan_v1",copies=expected,
        input_count=len(expected),fixed_noncheckpoint_allowance_bytes=FIXED_NONCHECKPOINT_ALLOWANCE,
        per_copy_growth_allowance_bytes=COPY_GROWTH_ALLOWANCE,planned_complete_scope_bound_bytes=total,
        reserved_new_bytes=SCOPE_CAP,input_read_cap_is_not_output_reservation=True)

def intent_metadata_refs_v1(intent):
    """Metadata closure only, checkpoints stay authenticated by the original validator."""
    rows=[intent[k] for k in ("old_input_ref","old_terminal_observation_ref","new_integration_ref",
        "new_runtime_ref","bounds_ref","qualification_ref")]
    terminal=read(intent["old_terminal_observation_ref"])
    need(terminal.get("actual_source_job_terminal_consumed") is True
        and type(terminal.get("actual_source_exit_code")) is int
        and terminal.get("all_old_owned_processes_drained") is True
        and terminal.get("all_old_future_write_relaunch_capabilities_revoked") is True,
        "Genuine post-terminal C3 wait/drain/revocation metadata, no live migration")
    rows+=terminal.get("original_attempt_elapsed_provenance_refs",[])
    rows+=terminal.get("actual_cursor_provenance_refs",[])
    if intent["rollback_authorization_ref"] is not None:
        rows.append(intent["rollback_authorization_ref"])
        approval=read(intent["rollback_authorization_ref"])
        rows+=approval.get("later_update_count_provenance_refs",[])
        rows+=approval.get("durable_checkpoint_selection_provenance_refs",[])
    unique={}
    for row in rows:
        exact_ref(row);need(ref(row["path"])==row,"Exact existing source/evidence metadata")
        need(row["path"] not in unique or unique[row["path"]]==row,"Unambiguous metadata lineage")
        unique[row["path"]]=row
    return list(unique.values())

def make_root_request_v1(static_contract_ref,intent_ref,actual_checkpoint_lengths,*,root_execute_authorized):
    """Root invokes AFTER actual old terminal/drain. Returns JSON, writes nothing."""
    need(root_execute_authorized is True,"Explicit root CPU execution authorization")
    c=read(static_contract_ref);intent=read(intent_ref)
    need(c["execution_authorized"] is False and intent["schema"]=="C3_actual_complete_boundary_migration_intent_v1"
        and intent["active"] is True and intent["root_authorizer"]=="root_agent"
        and intent["migration_output_directory"]==c["output_directory"],"Actual active original migration intent and fixed fresh output")
    plan=checkpoint_plan_v1(intent,actual_checkpoint_lengths)
    return dict(schema="C3_full_state_migration_CPU_root_request_v1",active=True,root_authorizer="root_agent",
        execution_authorized=True,GPU_authorized=False,static_contract_ref=static_contract_ref,
        actual_runtime_source_ref=ref(HERE+"/ordinary_migration_cpu_v1.py"),
        migration_intent_ref=intent_ref,actual_checkpoint_lengths=actual_checkpoint_lengths,
        checkpoint_byte_plan=plan,intent_metadata_refs=intent_metadata_refs_v1(intent),
        actual_migration_or_terminal_or_success_claimed=False)

def validate_root_request_v1(request,c):
    need(request.get("schema")=="C3_full_state_migration_CPU_root_request_v1" and request.get("active") is True
        and request.get("root_authorizer")=="root_agent" and request.get("execution_authorized") is True
        and request.get("GPU_authorized") is False and request.get("actual_migration_or_terminal_or_success_claimed") is False,
        "Explicit root CPU request, no manufactured migration result")
    intent=read(request["migration_intent_ref"])
    need(intent["migration_output_directory"]==c["output_directory"]
        and request["checkpoint_byte_plan"]==checkpoint_plan_v1(intent,request["actual_checkpoint_lengths"])
        and request["intent_metadata_refs"]==intent_metadata_refs_v1(intent),"Exact post-terminal intent/count/source metadata")
    return intent

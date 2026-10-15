"""Pure read-only actual-evidence/synthetic-gate fixtures. No live actions."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json
HERE=Path(__file__).resolve().parent
SOURCE=HERE/"migration_gate_successor_v2.py"
tree=ast.parse(SOURCE.read_text())
ns=dict(Path=Path,datetime=datetime,timezone=timezone,ast=ast,hashlib=hashlib,json=json)
nodes=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef)
    and n.name in ("require","stamp","authenticate_failure_flow_v2","_identity","normalize_root_capture_v2","validate_gate_v2")]
mod=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,"<pure_gate_source_only>","exec"),ns)
root=ns["ROOT"]
refs=[ns[k] for k in ("FLOW_REF","ROOT_CAPTURE_REF","ACTUAL_MINIMAL_REF","ACTUAL_TOOL_REF","ACTUAL_DRAIN_REF","ACTUAL_REVOCATION_REF")]
data={}
for record in refs:
    raw=(root/record["path"]).read_bytes();assert hashlib.sha256(raw).hexdigest()==record["sha256"]
    data[record["path"]]=json.loads(raw)
capture=data[ns["ROOT_CAPTURE_REF"]["path"]];input_ref=capture["exact_original_input_ref"]
raw=(root/input_ref["path"]).read_bytes();assert hashlib.sha256(raw).hexdigest()==input_ref["sha256"];data[input_ref["path"]]=json.loads(raw)
gate_ref=dict(path="pure_fixture/in_memory_gate.json",sha256="a"*64)
gate=dict(schema="GPU45_C3cache_H050_known_roster_terminal_handover_gate_v2",status="active",root_authorizer="root_agent",
    physical_gpu=4,native_GPU5_preserved=True,successor_evidence_not_original_full_receipt=True,original_full_actual_wait_ref=None,
    exact_final_exception_observed=False,old_driver_full_ancestry_completion_claimed=False,failure_flow_is_source_explanation_only=True,
    source_ref=dict(path=ns["SOURCE"],sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()),original_driver_ref=ns["DRIVER_REF"],
    failure_flow_authentication_ref=ns["FLOW_REF"],original_C3_input_ref=input_ref,original_C3_minimal_wait_ref=ns["ACTUAL_MINIMAL_REF"],
    actual_tool_terminal_ref=ns["ACTUAL_TOOL_REF"],fresh_owned_roster_ref=ns["ROOT_CAPTURE_REF"],
    fresh_owned_drain_ref=ns["ACTUAL_DRAIN_REF"],original_capability_revocation_ref=ns["ACTUAL_REVOCATION_REF"],
    created_utc=datetime.now(timezone.utc).isoformat())
data[gate_ref["path"]]=gate;baseline=copy.deepcopy(data)
ns["read"]=lambda record,maximum=None:copy.deepcopy(data[record["path"]])
results=[]
def passed(name,ok):assert ok,name;results.append(dict(case=name,passed=True))
def reset():data.clear();data.update(copy.deepcopy(baseline))
def rejects(name,mutate):
    reset();mutate()
    try:ns["validate_gate_v2"](gate_ref)
    except (ValueError,KeyError):passed(name,True)
    else:raise AssertionError(name)
passed("real_immutable_evidence_validates_only_in_memory_labelled_gate",ns["validate_gate_v2"](gate_ref)["physical_gpu"]==4)
rejects("minimal_wait_alone_never_certifies_drain",lambda:data[ns["ACTUAL_DRAIN_REF"]["path"]].update(identities=[]))
rejects("nonterminal_tool_result_rejected",lambda:data[ns["ACTUAL_TOOL_REF"]["path"]].update(actual_exec_exit_code=None))
rejects("wrong_actual_tool_session_rejected",lambda:data[ns["ACTUAL_TOOL_REF"]["path"]].update(actual_exec_session_id=99))
rejects("unknown_final_exception_not_invented",lambda:data[gate_ref["path"]].update(exact_final_exception_observed=True))
rejects("source_flow_not_claimed_as_actual_traceback",lambda:data[gate_ref["path"]].update(failure_flow_is_source_explanation_only=False))
rejects("full_ancestry_completion_not_inferred",lambda:data[gate_ref["path"]].update(old_driver_full_ancestry_completion_claimed=True))
rejects("minimal_wait_drain_claim_rejected",lambda:data[ns["ACTUAL_MINIMAL_REF"]["path"]].update(drain_state="certified"))
rejects("wrong_failure_flow_source_rejected",lambda:data[gate_ref["path"]].update(failure_flow_authentication_ref=dict(path="wrong",sha256="b"*64)))
rejects("missing_old_full_receipt_not_fabricated",lambda:data[gate_ref["path"]].update(original_full_actual_wait_ref=dict(path="full",sha256="b"*64)))
rejects("native_GPU5_preserved_required",lambda:data[gate_ref["path"]].update(native_GPU5_preserved=False))
rejects("incomplete_pre_cancel_capture_rejected",lambda:data[ns["ROOT_CAPTURE_REF"]["path"]].update(two_stable_identity_snapshots=False))
rejects("changed_known_start_rejected",lambda:data[ns["ROOT_CAPTURE_REF"]["path"]]["known_seeds"][0].update(start_ticks=1))
rejects("foreign_parent_tree_rejected",lambda:data[ns["ROOT_CAPTURE_REF"]["path"]]["identities"][-1].update(PPID=1))
rejects("missing_actual_owned_group_observation_rejected",lambda:data[ns["ACTUAL_DRAIN_REF"]["path"]]["groups"].pop())
rejects("present_owned_identity_rejected",lambda:data[ns["ACTUAL_DRAIN_REF"]["path"]]["identities"][0].update(present=True))
rejects("group_without_ESRCH_rejected",lambda:data[ns["ACTUAL_DRAIN_REF"]["path"]]["groups"][0].update(ESRCH=False))
rejects("root_revocation_required",lambda:data[ns["ACTUAL_REVOCATION_REF"]["path"]].update(all_original_C3_launch_resume_write_capabilities_revoked=False))
rejects("old_allocation_credit_not_created",lambda:data[ns["ACTUAL_REVOCATION_REF"]["path"]].update(old_original_storage_scope_fully_charged=False))
print(json.dumps(dict(schema="C3_known_roster_migration_gate_source_fixtures_v2",source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    cases=results,cases_passed=len(results),active_gate_files_created=0,
    actual_probes_signals_guard_calls_jobs_accounting_or_new_terminal_claims=0),indent=2,sort_keys=True))

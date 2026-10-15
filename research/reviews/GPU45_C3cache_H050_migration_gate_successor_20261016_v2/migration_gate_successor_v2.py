"""Labelled original-C3 minimal-wait handover gate, metadata/source only.

Never invents original full launch/wait, observes a process/group, sends a signal,
calls a guard, creates credit, or alters science. Root must supply genuine tool/
minimal-wait/fresh owned identity capture/drain/revocation evidence after terminal.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU45_C3cache_H050_migration_gate_successor_20261016_v2"
SOURCE=HERE+"/migration_gate_successor_v2.py"
DRIVER_REF=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/root_driver_v1.py",
 sha256="7724766b65ad4463af9c2f8f108ce063a31564c2cd07a044e944fe65b6dc6e46")
FLOW_REF={'path': 'research/reviews/GPU45_C3cache_H050_migration_gate_successor_20261016_v2/failure_flow_source_authentication_v2.json', 'sha256': '6ee1d9dd8df3c70c5891879917af7ae88cf6eccfc9fa9e7faa12a8a2286fec48'}
ROOT_CAPTURE_REF=dict(path="research/reviews/C3_original_handover_root_20261016_v1/pre_cancellation_owned_roster.root_v1.json",sha256="694f71cae0f9fc70b13aa294dd78d1c2ebd0e59a737be659ba8700cc51eaa32d")
CAPTURE_SOURCE_REF=dict(path="research/reviews/C3_original_handover_root_20261016_v1/capture_owned_tree_v1.py",sha256="d1ed73eb950c62b4ebcb11c61891ed9343c8f94187dd1b62749b0011f5365b67")
ACTUAL_MINIMAL_REF=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/admitted_train_v1/actual_wait_minimal.json",sha256="99b957e2570abf908e15da1bc27cb0ed6c8cf39fa26e00cf5f5b5fc34ed47f8f")
ACTUAL_TOOL_REF=dict(path="research/reviews/C3_original_handover_root_20261016_v1/original_tool_terminal_actual_v1.json",sha256="869fbcda23422f6aa230d89eba8b82486af914d120a55e61e6c343df5bbb46ba")
ACTUAL_DRAIN_REF=dict(path="research/reviews/C3_original_handover_root_20261016_v1/fresh_original_owned_identity_group_absence.root_v1.json",sha256="813707926ddc3535f9281801f87a64323f8f1796ac39fef321609dd4901c2604")
ACTUAL_REVOCATION_REF=dict(path="research/reviews/C3_original_handover_root_20261016_v1/original_C3_capabilities_revoked.root_v1.json",sha256="3bf61f4001893b7f0693de030020fe726758360590105d0206cb3cb4fb459f37")
OLD_REL="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
OLD_PARENT="American_C3_short20_TRAIN_original5VAL"
OLD_JOB="American-C3-GNN-prefetch-isolated980-train-v1"


def require(ok,message):
    if not ok:raise ValueError(message)


def read(record,maximum=4*2**20):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact successor evidence ref")
    p=ROOT/record["path"]
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),
        "Canonical regular successor evidence")
    with p.open("rb") as stream:raw=stream.read(maximum+1)
    require(len(raw)<=maximum and hashlib.sha256(raw).hexdigest()==record["sha256"],"Successor evidence bytes changed")
    return json.loads(raw)


def stamp(value):
    t=datetime.fromisoformat(value.replace("Z","+00:00"))
    require(t.tzinfo is not None and t<=datetime.now(timezone.utc),"Genuine aware past evidence time")
    return t


def authenticate_failure_flow_v2(flow_ref):
    require(flow_ref==FLOW_REF,"Exact reviewed expired-final-guard failure-flow authentication")
    flow=read(flow_ref);p=ROOT/DRIVER_REF["path"];raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==DRIVER_REF["sha256"] and flow["original_driver_ref"]==DRIVER_REF,
        "Exact unchanged original C3 root driver")
    launch=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=="launch")
    require(hashlib.sha256(ast.dump(launch,include_attributes=False).encode()).hexdigest()==flow["launch_ast_sha256"],
        "Exact authenticated original wait/drain/final-guard control flow")
    require(flow["minimal_wait_after_actual_proc_wait"] is True
        and flow["full_launch_wait_written_only_after_post_drain_guard"] is True
        and flow["post_drain_guard_line"]==251 and flow["minimal_wait_line"]==216,
        "Reviewed source establishes only minimal wait then own drain before failed final guard")
    return flow


def _identity(row):
    fields=("pid","uid","start_ticks","ppid","pgid","sid")
    require(isinstance(row,dict) and all(type(row.get(k)) is int for k in fields)
        and row["pid"]>1 and row["uid"]>=0 and row["start_ticks"]>0
        and all(row[k]>0 for k in ("ppid","pgid","sid")),"Exact fresh PID/UID/start/parent/group/session identity")
    return (row["pid"],row["uid"],row["start_ticks"])


def normalize_root_capture_v2(record,input_ref):
    """Read the actual separately labelled pre-cancellation capture, unchanged."""
    require(record==ROOT_CAPTURE_REF,"Exact actual root pre-cancellation capture ref")
    raw=read(record)
    require(raw.get("schema")=="C3_original_pre_cancellation_owned_roster_root_v1"
        and raw.get("root_authorizer")=="root_agent" and raw.get("exact_original_input_ref")==input_ref
        and raw.get("exact_original_driver_ref")==DRIVER_REF and raw.get("source_ref")==CAPTURE_SOURCE_REF
        and raw.get("physical_gpu")==4 and raw.get("native_GPU5_preserved") is True,
        "Exact genuine root-owned C3 capture identity/source")
    capture=ROOT/CAPTURE_SOURCE_REF["path"]
    require(hashlib.sha256(capture.read_bytes()).hexdigest()==CAPTURE_SOURCE_REF["sha256"],
        "Exact actual own-only capture source bytes")
    require(raw.get("owned_only") is True and raw.get("complete_current_owned_task_children_capture") is True
        and raw.get("two_stable_identity_snapshots") is True and raw.get("unrelated_host_inventory_performed") is False
        and raw.get("drain_claimed") is False and raw.get("signal_sent") is False,
        "Actual pre-signal stable own-only capture; no drain inferred")
    identities=[dict(pid=r["pid"],uid=r["UID"],start_ticks=r["start_ticks"],ppid=r["PPID"],pgid=r["PGID"],sid=r["SID"])
        for r in raw["identities"]]
    for row in identities:_identity(row)
    table={r["pid"]:r for r in identities}
    require(len(table)==len(identities) and raw["root_pid"] in table
        and table[raw["root_pid"]]["start_ticks"]==raw["root_start_ticks"]
        and table[raw["root_pid"]]["uid"]==raw["root_UID"]
        and raw["launcher_pid"] in table and table[raw["launcher_pid"]]["start_ticks"]==raw["launcher_start_ticks"]
        and raw["model_worker_pid"] in table and table[raw["model_worker_pid"]]["start_ticks"]==raw["model_worker_start_ticks"],
        "Actual known root/launcher/model starts all matched")
    require(all(s["pid"] in table and table[s["pid"]]["start_ticks"]==s["start_ticks"] for s in raw["known_seeds"]),
        "Every actual known owned seed identity preserved")
    expected_edges=sorted([r["ppid"],r["pid"]] for r in identities if r["pid"]!=raw["root_pid"])
    require(sorted(raw["owned_edges"])==expected_edges
        and raw["owned_process_groups"]==sorted({r["pgid"] for r in identities}),"Exact captured own parent edges and derived groups")
    return dict(schema="C3_original_fresh_owned_identity_capture_v2",root_authorizer="root_agent",
        original_C3_input_ref=input_ref,original_driver_ref=DRIVER_REF,
        captured_while_supervisor_and_launcher_live=True,all_current_owned_descendants_and_recorded_identities_listed=True,
        coverage="fresh_current_owned_tree_and_known_recorded_identities",captured_utc=raw["observed_utc"],
        identities=identities,supervisor_identity=table[raw["root_pid"]],groups=raw["owned_process_groups"],
        actual_original_exec_session_id=raw["original_exec_session_id"])


def validate_gate_v2(record,gpu=4):
    gate=read(record)
    require(gate.get("schema")=="GPU45_C3cache_H050_known_roster_terminal_handover_gate_v2"
        and gate.get("status")=="active" and gate.get("root_authorizer")=="root_agent"
        and gate.get("physical_gpu")==gpu==4 and gate.get("native_GPU5_preserved") is True
        and gate.get("successor_evidence_not_original_full_receipt") is True
        and gate.get("original_full_actual_wait_ref") is None
        and gate.get("exact_final_exception_observed") is False
        and gate.get("old_driver_full_ancestry_completion_claimed") is False
        and gate.get("failure_flow_is_source_explanation_only") is True,
        "Explicit labelled known-roster transition only; no original full wait/unknown exception inferred")
    require(gate.get("source_ref",{}).get("path")==SOURCE and gate.get("original_driver_ref")==DRIVER_REF,
        "Exact new/old source identities")
    require(hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()==gate["source_ref"]["sha256"],"Exact reviewed successor source")
    authenticate_failure_flow_v2(gate["failure_flow_authentication_ref"])
    input_ref=gate["original_C3_input_ref"];minimal_ref=gate["original_C3_minimal_wait_ref"]
    require(input_ref["path"]==OLD_REL+"/admitted_train_v1/input.json"
        and minimal_ref==ACTUAL_MINIMAL_REF,"Exact genuine original minimal wait")
    spec=read(input_ref);minimal=read(minimal_ref)
    require(spec.get("root_driver_ref")==DRIVER_REF and spec.get("physical_gpu")==4
        and spec.get("phase")=="train" and spec.get("parent_id")==OLD_PARENT and spec.get("job_id")==OLD_JOB
        and spec.get("main_source_ref",{}).get("path")=="src/arggnn/american_c3_gnn_prefetch_isolated980_exec_v2.py",
        "Exact original isolated-v2 C3 admission")
    require(minimal.get("input_ref")==input_ref and minimal.get("driver_ref")==DRIVER_REF
        and minimal.get("actual_direct_launcher_wait") is True and type(minimal.get("pid")) is int
        and type(minimal.get("returncode")) is int and minimal.get("phase")=="train"
        and minimal.get("parent_id")==OLD_PARENT and minimal.get("drain_state")=="not_yet_certified",
        "Original minimal wait authenticates only actual direct launcher wait")
    minimal_time=stamp(minimal["finished_utc"])
    roster=normalize_root_capture_v2(gate["fresh_owned_roster_ref"],input_ref)
    captured=stamp(roster["captured_utc"]);require(captured<=minimal_time,"Fresh capture precedes genuine direct wait")
    identities=roster["identities"];keys=[_identity(r) for r in identities]
    table={r["pid"]:r for r in identities};supervisor=roster["supervisor_identity"]
    require(2<=len(table)==len(identities)<=512 and supervisor["pgid"]==supervisor["sid"]==supervisor["pid"],
        "Exact bounded dedicated-session own tree")
    require(minimal["pid"] in table and table[minimal["pid"]]["ppid"]==supervisor["pid"],
        "Actual minimal launcher belongs to captured original root")
    reached={supervisor["pid"]}
    for unused in range(len(table)):
        reached.update(p for p,r in table.items() if r["ppid"] in reached and r["uid"]==supervisor["uid"])
    require(reached==set(table),"Every captured identity is in the own parent tree")
    require(gate["actual_tool_terminal_ref"]==ACTUAL_TOOL_REF,"Exact real root-consumed tool terminal")
    tool=read(ACTUAL_TOOL_REF)
    require(tool.get("schema")=="C3_original_tool_terminal_actual_root_v1" and tool.get("root_authorizer")=="root_agent"
        and tool.get("root_consumed_original_supervisor_terminal") is True
        and tool.get("original_input_ref")==input_ref and tool.get("old_driver_ref")==DRIVER_REF
        and tool.get("actual_console_direct_launcher_wait_ref")==minimal_ref
        and tool.get("actual_exec_session_id")==roster["actual_original_exec_session_id"]
        and type(tool.get("actual_exec_exit_code")) is int and tool["actual_exec_exit_code"]!=0
        and isinstance(tool.get("actual_tool_chunk_id"),str) and tool["actual_tool_chunk_id"]
        and tool.get("actual_tool_output_kind")=="genuine_direct_launcher_wait_event",
        "Genuine exact original tool terminal/session/chunk and console minimal-wait ref")
    require(tool.get("full_original_actual_wait_exists") is False and tool.get("full_original_launch_receipt_exists") is False
        and tool.get("ancestry_completion_claimed") is False
        and tool.get("original_optimizer_updates") is None and tool.get("original_optimizer_count_unknown_not_zero") is True,
        "Missing full receipts, unknown ancestry/counts stay explicitly unknown")
    tool_time=stamp(tool["recorded_utc"]);require(tool_time>=minimal_time,"Actual tool terminal follows direct wait")
    require(gate["fresh_owned_drain_ref"]==ACTUAL_DRAIN_REF,"Exact real post-terminal known-owned observation")
    drain=read(ACTUAL_DRAIN_REF)
    require(drain.get("schema")=="C3_original_post_terminal_known_owned_absence_root_v1"
        and drain.get("root_authorizer")=="root_agent"
        and drain.get("original_pre_cancellation_roster_ref")==gate["fresh_owned_roster_ref"]
        and drain.get("actual_original_tool_terminal_ref")==ACTUAL_TOOL_REF
        and drain.get("actual_original_minimal_direct_wait_ref")==minimal_ref
        and drain.get("exact_current_owned_roster_scope_only") is True
        and drain.get("historical_global_absence_claimed") is False
        and drain.get("unrelated_host_inventory_performed") is False and drain.get("native_GPU5_preserved") is True,
        "Fresh targeted known own scope only, never global/historical absence")
    observed=stamp(drain["observed_utc"])
    require(0<=(observed-tool_time).total_seconds()<=300,"Known own observation genuinely fresh at transition")
    observed_rows=drain["identities"]
    require(isinstance(observed_rows,list) and len(observed_rows)==len(keys)
        and len({r.get("pid") for r in observed_rows})==len(keys)
        and {r.get("pid") for r in observed_rows}==set(table),"Every actual fresh owned identity observed once")
    for row in observed_rows:
        expected=table[row["pid"]]
        require(row.get("original_start_ticks")==expected["start_ticks"]
            and row.get("original_identity_absent") is True and row.get("present") is False
            and row.get("PID_reused") is False,"Actual targeted current missing original identity")
    groups=sorted({r["pgid"] for r in identities});observed_groups=drain["groups"]
    require(drain.get("all_recorded_original_identities_absent") is True
        and drain.get("all_recorded_original_groups_ESRCH") is True
        and isinstance(observed_groups,list) and len(observed_groups)==len(groups)
        and sorted(r.get("PGID") for r in observed_groups)==groups
        and all(r.get("ESRCH") is True for r in observed_groups),"All and only captured original groups freshly ESRCH")
    require(gate["original_capability_revocation_ref"]==ACTUAL_REVOCATION_REF,"Exact root old-capability revocation")
    revocation=read(ACTUAL_REVOCATION_REF)
    require(revocation.get("schema")=="C3_original_root_future_capability_revocation_v1"
        and revocation.get("root_authorizer")=="root_agent" and revocation.get("active") is True
        and revocation.get("original_input_ref")==input_ref and revocation.get("original_driver_ref")==DRIVER_REF
        and revocation.get("actual_direct_launcher_wait_ref")==minimal_ref
        and revocation.get("actual_tool_terminal_ref")==ACTUAL_TOOL_REF
        and revocation.get("fresh_known_owned_absence_ref")==ACTUAL_DRAIN_REF
        and revocation.get("all_original_C3_launch_resume_write_capabilities_revoked") is True
        and revocation.get("no_old_allocation_completion_or_credit_claimed") is True
        and revocation.get("old_original_storage_scope_fully_charged") is True
        and revocation.get("old_files_preserved") is True and revocation.get("native_GPU5_preserved") is True,
        "Root revokes old launch/resume/write; full old charge/files preserved, no success/credit")
    created=stamp(gate["created_utc"]);revoked=stamp(revocation["created_utc"])
    require(tool_time<=revoked<=created and observed<=created,"Genuine terminal/transition observations precede explicit gate seal")
    return gate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--validate",action="store_true")
    parser.add_argument("--record");parser.add_argument("--sha256");a=parser.parse_args(argv)
    if not a.validate:
        print(json.dumps(dict(status="source_only_root_gate_activation_required",signals_probes_guards_or_jobs=0)));return 0
    require(a.record and a.sha256,"Exact genuine successor gate ref")
    validate_gate_v2(dict(path=a.record,sha256=a.sha256));print(json.dumps(dict(status="validated_known_roster_successor_only")));return 0

if __name__=="__main__":raise SystemExit(main())

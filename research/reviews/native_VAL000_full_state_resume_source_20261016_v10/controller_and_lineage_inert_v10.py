"""Bounded actual-public-controller and v9-lineage tests with fake resource endpoints."""
from pathlib import Path
from unittest.mock import patch
import ast,copy,importlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v10"
PKG=HERE.replace("/",".")
def run():
    assert sys.dont_write_bytecode and sys.flags.no_user_site
    c=importlib.import_module(PKG+".amendment_contract_v1")
    d=importlib.import_module(PKG+".direct_gpu0_v1")
    audit=c.read(c.V9_CONTROLLER_AUDIT_REF)
    snapshot=c.read(audit["registry_observation"]["immutable_exact_snapshot_ref"])
    original_input=c.read(c.ORIGINAL_INPUT);q=c.read(original_input["root_stage_request_ref"])
    cap=c.read(d.CURRENT_CAP)
    names=[j["logical_job"] for j in snapshot["jobs"]]
    assert names.count(d.C3_JOB)==1
    baseline=next(j for j in snapshot["jobs"] if j["logical_job"]==d.C3_JOB)
    assert baseline["state"]=="failed" and baseline["requested"]==cap["jobs"][d.C3_JOB]
    results=[]
    cases=[
      ("exact_recorded_failed_and_drained_exit1",None,True),
      ("running_original_C3",lambda s,r:r.update(state="running"),True),
      ("draining_original_C3",lambda s,r:r.update(state="draining"),True),
      ("succeeded_original_C3",lambda s,r:r.update(state="succeeded",exit_code=0,owned_drain_observed=True),True),
      ("failed_missing_leader_wait",lambda s,r:r.pop("leader_wait_status_known"),False),
      ("failed_false_leader_wait",lambda s,r:r.update(leader_wait_status_known=False),False),
      ("failed_missing_drain",lambda s,r:r.pop("owned_drain_observed"),False),
      ("failed_false_drain",lambda s,r:r.update(owned_drain_observed=False),False),
      ("failed_unknown_exit",lambda s,r:r.update(exit_code=None),False),
      ("failed_exit0_not_relabelled",lambda s,r:r.update(exit_code=0),False),
      ("failed_exit2",lambda s,r:r.update(exit_code=2),False),
      ("failed_boolean_exit",lambda s,r:r.update(exit_code=True),False),
      ("failed_unrecorded_PID",lambda s,r:r.update(pid=r["pid"]+1),False),
      ("failed_unrecorded_start",lambda s,r:r.update(leader_start_ticks=r["leader_start_ticks"]+1),False),
      ("failed_unrecorded_finish",lambda s,r:r.update(finished_unix=r["finished_unix"]+1),False),
      ("failed_unrecorded_descendant_wait",lambda s,r:r.update(descendant_waits=[]),False),
      ("failed_changed_requested_spec",lambda s,r:r["requested"].update(cuda_cap_bytes=1),False),
      ("duplicate_C3",lambda s,r:s["jobs"].append(copy.deepcopy(r)),False),
      ("other_queued",lambda s,r:s["jobs"][0].update(state="queued"),False),
      ("other_launching",lambda s,r:s["jobs"][0].update(state="launching"),False),
      ("other_running",lambda s,r:s["jobs"][0].update(state="running"),False),
      ("other_draining",lambda s,r:s["jobs"][0].update(state="draining"),False),
      ("cancelled_C3",lambda s,r:r.update(state="cancelled"),False),
      ("registry_safety_error",lambda s,r:s.update(safety_error="SYNTHETIC_ERROR"),False),
      ("registry_ambiguous",lambda s,r:s.update(ambiguous_descendants_or_lease_holders=[800001]),False),
      ("wrong_registry_controller",lambda s,r:s["controller"].update(pid=800001),False),
      ("controller_not_running",lambda s,r:s.update(lifecycle="closed"),False)]
    for name,mutate,allowed in cases:
        value=copy.deepcopy(snapshot)
        row=next(j for j in value["jobs"] if j["logical_job"]==d.C3_JOB)
        if mutate:mutate(value,row)
        before=copy.deepcopy(value);calls=[]
        def fake_identity(pid):
            calls.append(dict(endpoint="fake_proc_identity",pid=pid))
            assert pid==d.CONTROLLER_PID
            return dict(uid=1000,start_ticks=d.CONTROLLER_START,state="S")
        fake_base=types.SimpleNamespace(proc_identity=fake_identity)
        fake_os=types.SimpleNamespace(getuid=lambda:1000)
        fake_time=types.SimpleNamespace(time=lambda:value["updated_unix"]+1.)
        accepted=False
        with patch.object(d,"active_amendment_v1",lambda:None),patch.object(d,"active",lambda authority:None),\
            patch.object(d,"_registry_v1",lambda q0,b0,c0:value),patch.object(d,"os",fake_os),patch.object(d,"time",fake_time):
            try:
                result=d.controller_v1(q,fake_base)
                accepted=True
                assert result[2] is row
            except ValueError:pass
        assert accepted is allowed,(name,accepted,allowed)
        assert value==before,"Actual public controller must not mutate/relabel any registry row"
        results.append(dict(name=name,status="PASS_ACCEPTED_INERT" if accepted else "PASS_REJECTED_INERT"))
    a=dict(preserved_actual_v9_amendment_ref=c.V9_ACTUAL_AMENDMENT_REF,
        genuine_v9_controller_entry_failed_terminal_ref=c.V9_ACTUAL_TERMINAL_REF,
        sealed_v9_controller_failure_audit_ref=c.V9_CONTROLLER_AUDIT_REF,
        v9_future_launch_write_capabilities_permanently_revoked=True,
        exact_recorded_failed_and_drained_C3_exit1_attachment_authorized=True,
        failed_C3_relabel_or_outer_success_claim_authorized=False)
    terminal,actual_audit=c.validate_v9_failure_lineage_v10(a)
    assert terminal["actual_returncode"]==1 and actual_audit==audit
    lineage=[dict(name="actual_public_v9_failure_and_audit_authentication",status="PASS_SOURCE_METADATA_ONLY")]
    for key in ("v9_future_launch_write_capabilities_permanently_revoked",
        "exact_recorded_failed_and_drained_C3_exit1_attachment_authorized"):
        bad=dict(a);bad[key]=False
        try:c.validate_v9_failure_lineage_v10(bad)
        except ValueError:pass
        else:raise AssertionError(key)
        lineage.append(dict(name=key+"_missing",status="PASS_REJECTED"))
    saved_read=c.read
    def changed_terminal(kind):
        value=copy.deepcopy(terminal)
        if kind=="wrapper_success":value["actual_returncode"]=0
        else:
            packets=[json.loads(line) for line in value["stdout"].splitlines()]
            if kind=="invent_worker_PID":packets[1]["native32_bounded_cleanup_observation"]["pid"]=800001
            if kind=="invent_gate":packets[1]["native32_bounded_cleanup_observation"]["gate_released"]=True
            if kind=="invent_old_drain":packets[1]["native32_bounded_cleanup_observation"]["all_owned_processes_and_groups_drained"]=True
            if kind=="invent_wait":packets[0]["native32_actual_minimal_wait_facts"]["actual_direct_worker_wait"]=True
            value["stdout"]="\n".join(json.dumps(v) for v in packets)+"\n"
        return value
    for name in ("wrapper_success","invent_worker_PID","invent_gate","invent_old_drain","invent_wait","invent_C3_outer74136_terminal"):
        def fake_read(record):
            if name=="invent_C3_outer74136_terminal" and record==c.V9_CONTROLLER_AUDIT_REF:
                v=copy.deepcopy(audit);v["C3_recorded_terminal_lineage"]["outer_ROOT_handle_74136_terminal_consumed"]=True;return v
            if name!="invent_C3_outer74136_terminal" and record==c.V9_ACTUAL_TERMINAL_REF:return changed_terminal(name)
            return saved_read(record)
        with patch.object(c,"read",fake_read):
            try:c.validate_v9_failure_lineage_v10(a)
            except ValueError:pass
            else:raise AssertionError(name)
        lineage.append(dict(name=name,status="PASS_REJECTED_EXPLICITLY_SYNTHETIC_METADATA"))
    old=ast.parse((ROOT/"research/reviews/native_VAL000_full_state_resume_source_20261016_v9/direct_gpu0_v1.py").read_text())
    new=ast.parse((ROOT/HERE/"direct_gpu0_v1.py").read_text())
    for name in ("capacity_v1","attach_v1","check_attachment_v1","install_before_torch_v1","_fresh_status_v1","_registry_v1"):
        get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(get(old),include_attributes=False)==ast.dump(get(new),include_attributes=False),name
    fresh_old=(ROOT/"research/reviews/native_VAL000_full_state_resume_source_20261016_v9/fresh_entry_v9.py").read_text()
    fresh_new=(ROOT/HERE/"fresh_entry_v9.py").read_text()
    assert fresh_new==fresh_old.replace("native_VAL000_full_state_resume_source_20261016_v9","native_VAL000_full_state_resume_source_20261016_v10")
    assert not any(n in sys.modules for n in ("torch","torch_geometric","composer"))
    return dict(schema="native_VAL000_v10_actual_public_controller_and_v9_lineage_inert_fixture",
        status="PASS_SOFTWARE_ONLY",controller_cases=results,lineage_cases=lineage,
        actual_public_controller_function=d.controller_v1.__module__+".controller_v1",
        actual_public_controller_code_used_with_explicit_fake_process_registry_clock_authority_endpoints=True,
        actual_metadata_read_helpers_not_stubbed_in_positive=True,
        real_capacity_attach_lease_status_and_worker_functions_never_called=True,
        original_capacity_C326GiB_and_all_lease_source_bodies_unchanged=True,
        fresh120second_mechanism_code_only_import_route_changed=True,
        failed_row_never_mutated_or_relabelled=True,
        no_actual_guard_authority_probe_accounting_checkpoint_tensor_or_scientific_execution=True)
if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit inert-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))

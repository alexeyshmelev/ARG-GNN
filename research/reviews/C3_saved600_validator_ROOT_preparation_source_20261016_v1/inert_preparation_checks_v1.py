"""Inert saved600 ROOT-preparation schema/role tests. No guard or admission runs."""
from pathlib import Path
import ast,copy,importlib,importlib.util,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_saved600_validator_ROOT_preparation_source_20261016_v1"
PACKAGE="research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1"
def run():
    assert sys.dont_write_bytecode and sys.flags.no_user_site
    prep=importlib.import_module(HERE.replace("/",".")+".prepare_root_v1")
    manifest=json.loads(prep.source_bytes_v1(prep.MANIFEST_REF))
    loader=importlib.util.spec_from_file_location("_saved600_inert_contract",ROOT/PACKAGE/"acceptance_contract_v1.py")
    c=importlib.util.module_from_spec(loader);loader.loader.exec_module(c)
    caps=c.caps_v1(c.ACCEPT_PHASE)
    expected=json.loads((ROOT/PACKAGE/"caps_validate_saved600.INACTIVE.v1.json").read_text())
    assert caps==expected and caps["reservation_cap_bytes"]==33554432
    assert caps["operational_deadline_seconds"]==900 and caps["max_cpu_rss_bytes"]==2147483648
    assert caps["CPU_threads"]==1 and caps["max_cuda_reserved_bytes"]==caps["dataloader_workers"]==0
    req=json.loads((ROOT/PACKAGE/"ROOT_validation_request.INACTIVE.template.v1.json").read_text())
    assert req["original_history"]==c.ORIGINAL_HISTORY
    req.update(packet_directory="SYNTHETIC_ONLY_PACKET",phase=c.ACCEPT_PHASE)
    config=c.make_config(req,caps)
    assert config==dict(schema="C3_full30Mb_evaluation_config_v1",execution_enabled=True,
        phase="validate_saved600",output_cap_bytes=33554432,short_tmp_directory="SYNTHETIC_ONLY_PACKET/worker_tmp")
    roles_fn=types.FunctionType(c.required_direct_refs_v1.__code__,
        dict(c.required_direct_refs_v1.__globals__,manifest_v1=lambda:manifest))
    assert roles_fn.__code__ is c.required_direct_refs_v1.__code__
    req.update(source_review_ref=c.ref(PACKAGE+"/source_review.INACTIVE.template.v1.json"),
        caps_ref=c.ref(PACKAGE+"/caps_validate_saved600.INACTIVE.v1.json"))
    spec=dict(refs=dict(config=c.ref(PACKAGE+"/input.INACTIVE.template.v1.json")),
        root_stage_request_ref=c.ref(PACKAGE+"/ROOT_validation_request.INACTIVE.template.v1.json"))
    current=manifest["current_guard_roles"]
    source=(ROOT/c.GUARD_REF["path"]).read_text()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="consumer_bindings_v1")
    ns=dict(copy=copy,require=prep.require,_raw=lambda record:None)
    exec(compile(ast.Module(body=[function],type_ignores=[]),"<actual_native14_pure_binding_function_with_fake_raw>","exec"),ns)
    guard=types.SimpleNamespace(consumer_bindings_v1=ns["consumer_bindings_v1"])
    cc=types.SimpleNamespace(required_direct_refs_v1=roles_fn)
    extras=[c.ref(PACKAGE+"/ROOT_artifact_acceptance.INACTIVE.template.v1.json"),
        c.ref(PACKAGE+"/ROOT_evaluation_request.INACTIVE.template.v1.json")]
    required=roles_fn(spec,req)
    union=prep.role_union_v1(cc,guard,dict(bindings=current),spec,req,extras)
    assert all(row in union for row in [*current.values(),*required,*extras])
    assert len(union)<=64
    checks=[dict(name="actual_required_direct_function_and_native14_binding_body_with_fake_raw",
        status="PASS_SOURCE_ONLY",unique_roles=len(union),all_declared_current_roles_retained=True)]
    rows=[dict(path="SYNTHETIC_"+str(i)+".json",sha256="0"*64) for i in range(65)]
    plain=types.SimpleNamespace(required_direct_refs_v1=lambda spec,req:[])
    assert len(prep.role_union_v1(plain,guard,dict(bindings={str(i):r for i,r in enumerate(rows[:64])}),{}, {}, []))==64
    try:prep.role_union_v1(plain,guard,dict(bindings={str(i):r for i,r in enumerate(rows)}),{}, {}, [])
    except ValueError:pass
    else:raise AssertionError("65roles accepted")
    checks.append(dict(name="exact64accepted65rejected",status="PASS"))
    bad=dict(rows[0],sha256="1"*64)
    try:prep.role_union_v1(plain,guard,dict(bindings=dict(first=rows[0])),{}, {}, [bad])
    except ValueError:pass
    else:raise AssertionError("Conflicting same-path SHA accepted")
    checks.append(dict(name="conflicting_role_hash_rejected",status="PASS"))
    receipt=dict(value=dict(key="SYNTHETIC_VALIDATOR_KEY",output="SYNTHETIC_OUTPUT",
        consumer_bindings=rows[:1],reserved_bytes=33554432),
        initial_ledger_pins=dict(base_pins=dict(SYNTHETIC=True),completed_overlay_prefix=dict(events=123,tail_sha256="2"*64)))
    unchanged=copy.deepcopy(receipt)
    fake_guard=types.SimpleNamespace(MODE="native14_stopped_retention_v3",
        budget=types.SimpleNamespace(SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py"))
    storage=prep.storage_v1(c,fake_guard,"SYNTHETIC_JOB","SYNTHETIC_OUTPUT",receipt,"SYNTHETIC_EXPIRY")
    assert storage["reservation"]==receipt["value"] and storage["initial_ledger_pins"]==receipt["initial_ledger_pins"]
    assert storage["reservation"] is not receipt["value"] and storage["initial_ledger_pins"] is not receipt["initial_ledger_pins"]
    assert receipt==unchanged and storage["reserved_new_bytes"]==33554432
    assert storage["accounting_mode"]=="native14_stopped_retention_v3" and storage["completed_overlay_policy_ref"]==c.POLICY_REF
    checks.append(dict(name="actual_receipt_prefix_and_reservation_copied_without_rebase",status="PASS"))
    assert c.ORIGINAL_HISTORY["producer_cli_exit_code"]==1
    assert all(c.ORIGINAL_HISTORY[key] is None for key in ("original_outer_74136_terminal_consumed",
        "original_outer_74136_exit_code","original_worker_drain_manifest","original_completed_envelope",
        "actual_parent_release_count","actual_terminal_exception_text"))
    assert not any(name in sys.modules for name in ("torch","torch_geometric","numpy"))
    return dict(schema="saved600_ROOT_prep_inert_schema_role_storage_checks_v1",status="PASS_SOFTWARE_ONLY",
        checks=checks,original_history=c.ORIGINAL_HISTORY,
        exact_validator_caps=caps,actual_source_helpers_with_explicit_synthetic_metadata_only=True,
        actual_guard_configure_inspect_apply_reserve_runtime_probe_or_authority_execution=False,
        script_ref=c.ref(HERE+"/prepare_root_v1.py"))
if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit inert-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))

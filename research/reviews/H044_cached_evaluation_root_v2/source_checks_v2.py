"""Static source checks only. Never import workers, contexts, guards or Torch."""
from pathlib import Path
import ast,hashlib,json,datetime
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H044_cached_evaluation_root_v2"

def node(source,name):
    return next(x for x in ast.parse(source).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
def segment(source,name):
    n=node(source,name);return "\n".join(source.splitlines()[n.lineno-1:n.end_lineno])+"\n"
def assigned(source):
    return {n.targets[0].id:n.value for n in ast.parse(source).body
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def main():
    old_driver=(ROOT/"research/reviews/H044_marker_resolution_root_v1/root_driver_v4.py").read_text()
    old_context=(ROOT/"research/reviews/H044_marker_resolution_root_v1/context_factory_v2.py").read_text()
    driver=(HERE/"root_driver_v2.py").read_text();context=(HERE/"context_factory_v2.py").read_text()
    entry=(ROOT/"src/arggnn/evaluate_marker_resolution_cached_v2.py").read_text()
    for path,source in [(HERE/"root_driver_v2.py",driver),(HERE/"context_factory_v2.py",context),
        (ROOT/"src/arggnn/evaluate_marker_resolution_cached_v2.py",entry)]:
        ast.parse(source);compile(source,str(path),"exec",dont_inherit=True)
    for name in ("_stat","_tree","_signal_owned","_absent_group","_RAMLog","_subreaper","_children"):
        assert ast.dump(node(old_driver,name),include_attributes=False)==ast.dump(node(driver,name),include_attributes=False)
    marker='    _subreaper();require(not _children(),"Fresh root supervisor without unrelated child processes")'
    assert segment(old_driver,"launch").split(marker,1)[1]==segment(driver,"launch").split(marker,1)[1]
    original_constants=assigned(old_context);new_constants=assigned(context)
    for name in ("GUARD_REF","POLICY_REF","LAUNCHER_REF","REPAIR_REF","AUTH_REF","MEMO_REF","V4_REF"):
        assert ast.dump(original_constants[name],include_attributes=False)==ast.dump(new_constants[name],include_attributes=False)
    for name in ("CORE","BASE"):
        assert ast.dump(assigned(old_driver)[name],include_attributes=False)==ast.dump(assigned(driver)[name],include_attributes=False)
    for name in ("module_from_ref_v1","completed_preparation_v1"):
        assert ast.dump(node(old_context,name),include_attributes=False)==ast.dump(node(context,name),include_attributes=False)
    start='    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title'
    end='    preparation=completed_preparation_v1(reader,spec,cfg)'
    assert old_context.split(start,1)[1].split(end,1)[0]==context.split(start,1)[1].split(end,1)[0]
    admission=segment(driver,"admit")
    assert 'phase=="evaluate"' in admission
    assert admission.index("c.validate_original_train_v2(")<admission.index("packet.mkdir()")<admission.index("m=load_guard(c)")
    for operation in ("m._state_locked(CORE,BASE,fresh=True)",'m.operation_draft_v1("reserve"',
        "m.apply_reserve_v1(","expected_source_sha256=CORE,expected_launcher_sha256=BASE"):
        assert operation in admission
    build=segment(context,"build_context_v2")
    assert 'spec.get("phase")=="evaluate"' in build and 'spec.get("fit_authorized") is False' in build
    assert build.index("validate_original_train_v2(")<build.index("storage_guard=module_from_ref_v1(")
    assert 'if spec["phase"]=="prepare"' not in build and 'ids=worker.VAL' in build
    old_train_refs=[ast.literal_eval(new_constants[n]) for n in ("ORIGINAL_CONFIG_REF","ORIGINAL_CONTEXT_REF","ORIGINAL_DRIVER_REF","ORIGINAL_WORKER_REF","ORIGINAL_CORE_REF")]
    for r in old_train_refs:assert digest(r["path"])==r["sha256"]
    entry_constants=assigned(entry)
    for name in ("PRODUCER_REF","CACHE_REF"):
        r=ast.literal_eval(entry_constants[name]);assert digest(r["path"])==r["sha256"]
    assert ast.literal_eval(new_constants["WORKER_SOURCE"])=="src/arggnn/evaluate_marker_resolution_cached_v2.py"
    launcher=(ROOT/"scripts/launch_reserved_gpu_full_storage_v1.py").read_text()
    assert 'args.module.startswith("arggnn.")' in launcher and 'module_path = "src/" + args.module.replace(".", "/") + ".py"' in launcher
    cfg_ref=ast.literal_eval(new_constants["ORIGINAL_CONFIG_REF"]);cfg=json.loads((ROOT/cfg_ref["path"]).read_text())
    frozen=json.loads((ROOT/cfg["source_manifest_ref"]["path"]).read_text())
    manifest=json.loads((HERE/"source_manifest_v2.json").read_text())
    assert manifest["execution_authorized"] is False and manifest["frozen_source_manifest_ref"]==cfg["source_manifest_ref"]
    assert all(manifest["source_sha256"][p]==d for p,d in frozen["source_sha256"].items())
    for path,sha in manifest["source_sha256"].items():assert path.endswith(".py") and digest(path)==sha
    request=json.loads((HERE/"evaluate_request_inactive_v2.json").read_text())
    assert request["status"]!="active" and request["physical_gpu"] is None
    assert request["train_actual_wait_ref"] is None and request["train_completion_ref"] is None
    assert request["reserved_bytes"]==100000000 and request["max_cpu_rss_bytes"]==256000000000
    for key in ("source_manifest_ref","memory_envelope_ref","cache_policy_ref","config_ref"):
        r=request[key];assert digest(r["path"])==r["sha256"]
    memory=json.loads((HERE/"memory_envelope_v2.json").read_text())
    assert memory["producer_live_cached_parents"]==1 and memory["per_parent_cached_feature_upper_bytes"]==2634240000
    assert memory["named_planning_total_with_margin_bytes"]==sum(memory["named_allowances_bytes"].values())<=request["max_cpu_rss_bytes"]
    producer=(ROOT/ast.literal_eval(new_constants["PRODUCER_SOURCE"])).read_text()
    assert 'ram.verify();reader.truth_gate=True' in producer and 'cache.release(pid)' in producer
    result=dict(schema="H044_cached_evaluation_root_source_checks_v2",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parse_compile=True,original_supervision_and_direct_wait_drain_unchanged=True,
        original_guard_launcher_policy_authority_pins_unchanged=True,original_runtime_qualification_unchanged=True,
        original_preparation_verification_unchanged=True,TRAIN_terminal_gate_before_admission_and_runtime=True,
        evaluation_only=True,canonical_arggnn_entry_and_launcher_restriction_preserved=True,
        original_scientific_config_immutable=True,complete_original_source_closure_preserved=True,
        all64_source_hashes_verified=True,request_inactive_with_null_TRAIN_terminals=True,
        reservation_bytes=100000000,CPU_planning_cap_bytes=256000000000,
        named_CPU_with_margin_bytes=memory["named_planning_total_with_margin_bytes"],
        source_and_metadata_only=True,project_modules_or_Torch_imported=False,
        model_data_GPU_runtime_guards_or_accounting_executed=False,
        original_TRAIN_terminal_records_read=False,packages_or_Git=False)
    (HERE/"source_checks_receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result));return 0

if __name__=="__main__":raise SystemExit(main())

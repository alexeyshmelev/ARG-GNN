"""Source-only independent H044 successor review. No project/science/guard imports."""
from pathlib import Path
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/H044_cached_evaluation_independent_review_20261015_v1"
NEW="research/reviews/H044_cached_evaluation_root_v2"
OLD="research/reviews/H044_marker_resolution_root_v1"
PRODUCER="research/reviews/H044_cached_evaluation_source_v2/evaluate_cached_v2.py"
CACHE="src/arggnn/marker_resolution_input_cache_v2.py"
ENTRY="src/arggnn/evaluate_marker_resolution_cached_v2.py"
TRAINER="src/arggnn/train_marker_resolution_readout_v1.py"
def raw(p):return (ROOT/p).read_bytes()
def text(p):return raw(p).decode()
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def ref(p):return dict(path=p,sha256=sha(p))
def data(p):return json.loads(raw(p))
def tree(p):return ast.parse(text(p),p)
def funcs(p):return {n.name:n for n in tree(p).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
def literal(p,k):
    for n in tree(p).body:
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==k for x in n.targets):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="dict":
                return {a.arg:ast.literal_eval(a.value) for a in n.value.keywords}
            return ast.literal_eval(n.value)
    raise KeyError(k)
def message(n):
    if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and n.value.args and isinstance(n.value.args[-1],ast.Constant):return n.value.args[-1].value

def call_index(body,name):
    for i,n in enumerate(body):
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id==name:return i
    raise AssertionError(name)

checks={}
manifest=data(NEW+"/source_manifest_v2.json");frozen=data(manifest["frozen_source_manifest_ref"]["path"])
checks["actual_frozen_manifest_ref"]=ref(manifest["frozen_source_manifest_ref"]["path"])==manifest["frozen_source_manifest_ref"]
actual_sources={p:sha(p) for p in manifest["source_sha256"]}
checks["all64_actual_source_hashes_match"]=len(actual_sources)==64 and actual_sources==manifest["source_sha256"]
checks["every_original_manifest_mapping_preserved"]=all(actual_sources.get(p)==s for p,s in frozen["source_sha256"].items())
request=data(NEW+"/evaluate_request_inactive_v2.json");cfg=data(request["config_ref"]["path"])
checks["original_scientific_config_exact_and_manifest_unmodified"]=ref(request["config_ref"]["path"])==request["config_ref"]==manifest["original_scientific_config_ref"] and cfg["source_manifest_ref"]==manifest["frozen_source_manifest_ref"]
checks["request_and_metadata_refs_exact"]=all(ref(request[k]["path"])==request[k] for k in ("source_manifest_ref","cache_policy_ref","memory_envelope_ref"))
checks["preview_is_inactive_without_GPU_or_train_terminal_authority"]=request["status"]=="inactive_waiting_original_TRAIN30pass960updates_and_actual_direct_wait0" and request["physical_gpu"] is None and request["train_completion_ref"] is None and request["train_actual_wait_ref"] is None
checks["100MB_effective_reservation_preserves_original_config_cap"]=request["reserved_bytes"]==100000000 and cfg["output_cap_bytes"]==200000000
newd,oldd=funcs(NEW+"/root_driver_v2.py"),funcs(OLD+"/root_driver_v4.py")
for name in ("require","ref","read","save","unique","context","load_guard","_stat","_tree","_signal_owned","_absent_group","_RAMLog","_subreaper","_children"):
    checks["original_driver_function_unchanged:"+name]=dump(newd[name])==dump(oldd[name])
nt=newd["launch"].body[call_index(newd["launch"].body,"_subreaper"):]
ot=oldd["launch"].body[call_index(oldd["launch"].body,"_subreaper"):]
checks["whole_original_launch_supervision_direct_wait_drain_and_terminal_tail_unchanged"]=dump(ast.Module(body=nt,type_ignores=[]))==dump(ast.Module(body=ot,type_ignores=[]))
newc,oldc=funcs(NEW+"/context_factory_v2.py"),funcs(OLD+"/context_factory_v2.py")
for name in ("require","actual","module_from_ref_v1","completed_preparation_v1"):
    checks["original_context_helper_unchanged:"+name]=dump(newc[name])==dump(oldc[name])
for name in ("GUARD_REF","POLICY_REF","LAUNCHER_REF","REPAIR_REF","AUTH_REF","MEMO_REF","V4_REF"):
    checks["original_context_authority_constant_unchanged:"+name]=literal(NEW+"/context_factory_v2.py",name)==literal(OLD+"/context_factory_v2.py",name)
# Compare the exact original GPU runtime/title/qualification/limits segment.
def qualification_segment(fn):
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.ImportFrom) and any(a.name=="train_marker_resolution_readout_v1" for a in n.names))
    end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="preparation" for t in n.targets))
    return ast.Module(body=fn.body[start:end],type_ignores=[])
checks["original_GPU_protocol_lease_title_runtime_qualification_memory_guard_segment_unchanged"]=dump(qualification_segment(newc["build_context_v2"]))==dump(qualification_segment(oldc["build_context_v1"]))
# Inspect source ordering rather than executing admission/context/guards.
admit=text(NEW+"/root_driver_v2.py");context=text(NEW+"/context_factory_v2.py")
checks["original_TRAIN_gate_before_packet_guard_and_reserve"]=admit.index("c.validate_original_train_v2(")<admit.index("packet.mkdir()")<admit.index("m=load_guard(c)")<admit.index("m.operation_draft_v1(")
checks["context_TRAIN_gate_before_guard_install_and_runtime"]=context.index("    validate_original_train_v2(reader,")<context.index("    storage_guard=module_from_ref_v1(")<context.index("    memo_handle=memo.install(")<context.index("    device,qualified_runtime,qualified_lease=configure_fit(")

class RemoveInspectedCacheAdditions(ast.NodeTransformer):
    def visit_ImportFrom(self,n):
        if n.module=="arggnn.mutation_routed_full_panel_primary_v1":n.module="mutation_routed_full_panel_primary_v1";n.level=1
        return n
    def visit_Assign(self,n):
        if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="bind_cached_bundle_v2":return None
        if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=="cache" and n.value.func.attr=="get":return None
        if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=="result" for t in n.targets):return None
        return self.generic_visit(n)
    def visit_Expr(self,n):
        if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=="cache" and n.value.func.attr=="release":return None
        return self.generic_visit(n)
    def visit_Delete(self,n):
        n.targets=[t for t in n.targets if not isinstance(t,ast.Name) or t.id!="blocks"]
        return n
new_eval=copy.deepcopy(funcs(PRODUCER)["run_evaluate_cached_v2"])
start=next(i for i,n in enumerate(new_eval.body) if isinstance(n,ast.ImportFrom) and n.module=="arggnn.mutation_routed_full_panel_primary_v1")
new_eval.body=new_eval.body[start:];new_eval.name="run_evaluate_v1"
new_eval=RemoveInspectedCacheAdditions().visit(new_eval)
checks["entire_evaluation_AST_matches_original_after_inspected_cache_adapter_and_receipt_only"]=dump(new_eval)==dump(funcs(TRAINER)["run_evaluate_v1"])
checks["cache_releases_parent_after_all12_predictions"]=text(PRODUCER).index("        del bundle,device_flags,cpu_bundle,flags_cpu,blocks")<text(PRODUCER).index("        cache.release(pid)")<text(PRODUCER).index("    ram.verify();reader.truth_gate=True")
checks["canonical_entry_inert_and_actual_arggnn_module"]=literal(ENTRY,"SOURCE")==ENTRY and "from arggnn" not in text(ENTRY) and "spec.get(\"evaluation_authorized\") is True and spec.get(\"fit_authorized\") is False" in text(ENTRY)
checks["cache_default_imports_no_Torch_project_or_guard"]=all(not isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.Import) and all(a.name in ("hashlib","json") for a in n.names) or isinstance(n,ast.ImportFrom) and n.module=="dataclasses" for n in tree(CACHE).body)
policy=data(request["cache_policy_ref"]["path"]);memory=data(request["memory_envelope_ref"]["path"])
feature_pair=2*20000*686*4*6*4;weights=12*31428*4
checks["independent_cache_and_checkpoint_bytes"]=feature_pair==memory["per_parent_cached_feature_upper_bytes"]==2634240000 and 8*feature_pair==policy["limits"]["maximum_feature_bytes"]==21073920000 and weights==memory["named_allowances_bytes"]["original_twelve_checkpoint_weight_bytes"]==1508544
checks["named_old_CPU_caps_match_original_config"]=all(v==cfg[k] for k,v in memory["named_allowances_bytes"].items() if k in cfg)
checks["independent_memory_sum_and_cap"]=sum(memory["named_allowances_bytes"].values())==memory["named_planning_total_with_margin_bytes"]==146991685056 and memory["CPU_planning_cap_bytes"]==request["max_cpu_rss_bytes"]==256000000000
checks["all96_FP32_prediction_bytes_match_original"]=96*20000*686*4*4==cfg["full_grid_prediction_upper_bytes"]==21073920000 and cfg["full_grid_prediction_upper_bytes"]<=cfg["RAM_prediction_cap_bytes"]
fixture_path="research/reviews/H044_cached_evaluation_root_checks_20261015_v1/result.json";fixture=data(fixture_path)
checks["tiny_fixture_only_reports_feature_cache_checks_for_actual_sources"]=fixture["status"]=="passed" and all(fixture["checks"].values()) and fixture["GPU_used"] is False and fixture["production_loader_or_model_run"] is False and fixture["real_data_or_checkpoints_read"] is False and fixture["source_refs"]=={p:sha(p) for p in fixture["source_refs"]}
for p in (NEW+"/root_driver_v2.py",NEW+"/context_factory_v2.py",ENTRY,PRODUCER,CACHE):compile(text(p),p,"exec")
checks["all_new_modules_compile_as_text_without_execution"]=True
failed=[k for k,v in checks.items() if not v]
result=dict(schema="H044_cached_evaluation_independent_static_source_audit_v1",checks=checks,all_passed=not failed,
    source_refs=[ref(p) for p in (NEW+"/root_driver_v2.py",NEW+"/context_factory_v2.py",ENTRY,PRODUCER,CACHE,TRAINER,"src/arggnn/marker_resolution_readout_v1.py",OLD+"/root_driver_v4.py",OLD+"/context_factory_v2.py")],
    manifest_ref=ref(NEW+"/source_manifest_v2.json"),all64_actual_source_sha256=actual_sources,
    original_manifest_ref=manifest["frozen_source_manifest_ref"],original_scientific_config_ref=request["config_ref"],
    inactive_request_ref=ref(NEW+"/evaluate_request_inactive_v2.json"),cache_policy_ref=request["cache_policy_ref"],memory_envelope_ref=request["memory_envelope_ref"],fixture_receipt_ref=ref(fixture_path),fixture_checks=fixture["checks"],
    arithmetic=dict(per_parent_cache_bytes=feature_pair,all8_cache_hard_limit_bytes=8*feature_pair,checkpoint_weight_bytes=weights,named_host_allowances_with_margin_bytes=sum(memory["named_allowances_bytes"].values()),host_cap_bytes=memory["CPU_planning_cap_bytes"],all96_prediction_bytes=cfg["full_grid_prediction_upper_bytes"],effective_output_aux_reservation_bytes=request["reserved_bytes"],unchanged_original_config_output_cap_bytes=cfg["output_cap_bytes"]),
    forbidden_actions_performed=False,project_Torch_or_guard_imports=False,scientific_data_model_checkpoint_runtime_reads_or_execution=False,admission_accounting_process_GPU_package_or_Git_actions=False,
    limits="Static source and compact metadata only. Existing tiny CPU fixture receipt was read, not rerun. Actual complete-model parity, GPU runtime qualification, peak RSS, throughput, live GPU availability, genuine terminal evidence and storage admission were not established.")
with (OUT/"source_audit_receipt.json").open("x") as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write("\n")
print(json.dumps(dict(checks=len(checks),all_passed=not failed,failed=failed,receipt_ref=ref((OUT/"source_audit_receipt.json").relative_to(ROOT).as_posix()))))
assert not failed,failed

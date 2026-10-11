"""Reproduce the exact v2-to-v3 bounded transport delta without runtime state."""
from pathlib import Path
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REVIEW=ROOT/"research/reviews/budgeted_shared_pool_v3_author"
FIT2={"source":{"path":"src/arggnn/census_biological_fit_v2.py","sha256":"c21884ad90a2c940b49968353545edc52d345c0c7bc10fd87c492c54d4b38c34"},"config":{"path":"configs/census_biological_fit_v2.json","sha256":"a512816bbab61e80052e7466613d2de4c861b77ef61d0d35cd50fbf81905c621"},"shared_launch_allowed":True,"compatibility_limit":"eligibility_only_worker_scientific_review_GPU_profile_launcher_watchdog_and_fit_capabilities_required_no_profile_only_CLI"}
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def additional(node):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ADDITIONAL_WORKERS" for t in node.targets)
def normalized(text):
    tree=ast.parse(text.replace("budgeted_shared_pool_v3","budgeted_shared_pool_v2"))
    tree.body=[n for n in tree.body if not additional(n)]
    return ast.dump(tree,include_attributes=False)
def main():
    old=(ROOT/"src/arggnn/budgeted_shared_pool_v2.py").read_text()
    actual=(ROOT/"src/arggnn/budgeted_shared_pool_v3.py").read_text()
    expected=old.replace("budgeted_shared_pool_v2","budgeted_shared_pool_v3")
    assignment=next(n for n in ast.parse(expected).body if additional(n))
    contracts=ast.literal_eval(assignment.value)
    original_contracts=copy.deepcopy(contracts)
    contracts["arggnn.census_biological_fit_v2"]=FIT2
    lines=expected.splitlines(keepends=True)
    lines[assignment.lineno-1:assignment.end_lineno]=["ADDITIONAL_WORKERS = "+repr(contracts)+"\n"]
    expected="".join(lines)
    oldcfg=json.loads((ROOT/"configs/budgeted_shared_pool_v2.json").read_text())
    cfg=json.loads((ROOT/"configs/budgeted_shared_pool_v3.json").read_text())
    expectedcfg=copy.deepcopy(oldcfg)
    expectedcfg["kind"]="budgeted_shared_pool_v3_policy"
    expectedcfg["allowed_modules"]=sorted([*expectedcfg["allowed_modules"],"arggnn.census_biological_fit_v2"])
    expectedcfg["additional_worker_contracts"]["arggnn.census_biological_fit_v2"]=FIT2
    pins=json.loads((REVIEW/"predecessor_and_worker_pins.json").read_text())["artifact_sha256"]
    assert all(sha(p)==h for p,h in pins.items())
    assert actual==expected and cfg==expectedcfg and normalized(old)==normalized(actual)
    assert all(cfg["additional_worker_contracts"][m]==v for m,v in original_contracts.items())
    assert cfg["additional_worker_contracts"]["arggnn.native_argformer_tsinfer_fit_v5"]["shared_launch_allowed"] is False
    result={"kind":"budgeted_shared_pool_v3_exact_delta_proof","status":"PASS","v2_implementation":{"path":"src/arggnn/budgeted_shared_pool_v2.py","sha256":sha("src/arggnn/budgeted_shared_pool_v2.py")},"v2_config":{"path":"configs/budgeted_shared_pool_v2.json","sha256":sha("configs/budgeted_shared_pool_v2.json")},"implementation":{"path":"src/arggnn/budgeted_shared_pool_v3.py","sha256":sha("src/arggnn/budgeted_shared_pool_v3.py")},"config":{"path":"configs/budgeted_shared_pool_v3.json","sha256":sha("configs/budgeted_shared_pool_v3.json")},"finite_added_module":"arggnn.census_biological_fit_v2","worker_contract":FIT2,"whole_source_bytes_equal_after_only_version_bindings_and_one_worker_contract":actual==expected,"whole_config_equal_after_only_version_binding_and_one_module_contract":cfg==expectedcfg,"complete_AST_equal_after_removing_only_ADDITIONAL_WORKERS_and_normalizing_version_strings":normalized(old)==normalized(actual),"normalized_complete_AST_sha256":hashlib.sha256(normalized(actual).encode()).hexdigest(),"old_worker_contracts_unchanged":True,"native5_shared_launch_blocked":True,"original_Controller_owned_mutex_FD_device_cap_FIFO_memory_storage_argv_plan_review_process_supervision_unchanged":True,"original_v1_v2_and_scientific_sources_unchanged":True,"actual_resource_or_scientific_worker_execution_performed":False}
    (REVIEW/"exact_delta_proof.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
if __name__=="__main__":print(json.dumps(main(),indent=2))

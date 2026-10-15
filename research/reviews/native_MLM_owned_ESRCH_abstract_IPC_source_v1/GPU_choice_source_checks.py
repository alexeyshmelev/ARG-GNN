"""Incremental metadata-only GPU4/5 selector checks; no second real fork."""
from pathlib import Path
import ast,hashlib,json,os,sys
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import abstract_unix_bootstrap_v1 as b
passed=[]
def check(n,v):
    if not v:raise AssertionError(n)
    passed.append(n)
def rejects(**kw):
    try:b.validate_root_gpu_choice_v1(**kw)
    except ValueError:return True
    return False
def body(path):
    tree=ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="install_abstract_unix_v1"),include_attributes=False)
check("real_fork_qualified_IPC_function_AST_unchanged",body(HERE/"abstract_unix_bootstrap.before_gpu_selection.py")==body(HERE/"abstract_unix_bootstrap_v1.py"))
fixture_uuids={4:"GPU-00000004-0000-0000-0000-000000000004",5:"GPU-00000005-0000-0000-0000-000000000005"}
for device,uuid in fixture_uuids.items():
    for visible in [str(device),uuid]:
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":visible}):
            r=b.validate_root_gpu_choice_v1(physical_gpu=device,gpu_uuid=uuid)
            check(f"GPU{device}_{'UUID' if visible==uuid else 'ordinal'}_binding",r["physical_gpu"]==device and r["root_authorized_gpu_uuid"]==uuid and not r["GPU_probe_performed"])
    with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":uuid}),patch.object(b,"install_abstract_unix_v1",return_value={"namespace":"mock-attempt"}) as install:
        r=b.bootstrap_native_execution_v1("mock-attempt",physical_gpu=device,gpu_uuid=uuid)
        check(f"GPU{device}_wrapper_calls_unchanged_IPC",install.call_args.args==("mock-attempt",) and r["operational_device"]["physical_gpu"]==device)
with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"4"}):
    check("all_excluded_devices_and_noninteger_ids_rejected",all(rejects(physical_gpu=x,gpu_uuid=fixture_uuids[4]) for x in [0,1,2,3,6,-1,True,4.0]))
    check("malformed_or_missing_UUID_rejected",all(rejects(physical_gpu=4,gpu_uuid=x) for x in ["",None,"GPU4","4"]))
with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"5"}):
    check("mismatched_visible_ordinal_rejected",rejects(physical_gpu=4,gpu_uuid=fixture_uuids[4]))
with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"4,5"}):
    check("multiple_visible_devices_rejected",rejects(physical_gpu=4,gpu_uuid=fixture_uuids[4]))
base=json.loads((HERE/"base_source_checks_result.json").read_text())
report={"schema":"native_operational_GPU_choice_source_checks_v1","status":"PASS",
 "checks_passed":len(passed),"checks":passed,"all_UUIDs_in_this_test_are_explicit_mock_fixtures":True,
 "GPU_probe_or_CUDA_import":False,"additional_fork_children":0,
 "qualified_IPC_function_unchanged":True,
 "source_sha256":hashlib.sha256((HERE/"abstract_unix_bootstrap_v1.py").read_bytes()).hexdigest()}
(HERE/"GPU_choice_source_checks_result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
combined=dict(base)
combined.update(schema="owned_ESRCH_abstract_IPC_GPU45_source_qualification_v1",
 checks_passed=base["checks_passed"]+len(passed),checks=base["checks"]+passed,
 base_real_fork_report_ref={"path":str(HERE/"base_source_checks_result.json"),"sha256":hashlib.sha256((HERE/"base_source_checks_result.json").read_bytes()).hexdigest()},
 real_fork_qualified_IPC_source_ref={"path":str(HERE/"abstract_unix_bootstrap.before_gpu_selection.py"),"sha256":hashlib.sha256((HERE/"abstract_unix_bootstrap.before_gpu_selection.py").read_bytes()).hexdigest()},
 incremental_GPU_choice_report=report)
combined["source_refs"]["abstract_unix_bootstrap_v1.py"]=report["source_sha256"]
(HERE/"source_checks_result.json").write_text(json.dumps(combined,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":"PASS","combined_checks":combined["checks_passed"],"new_metadata_only_checks":len(passed),"total_real_fork_children":1}))

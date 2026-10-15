"""Narrow direct source qualification. No live lease/inventory/process/socket/guard calls."""
from pathlib import Path
import ast,contextlib,hashlib,importlib.util,io,json,sys,traceback,datetime
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/GPU0_shared_execution_source_20261016_v1"
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
results=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    results.append(dict(case=name,passed=True))
def module(name):
    p=HERE/(name+".py");sp=importlib.util.spec_from_file_location("_GPU0_qualify_"+name,p)
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m

def main():
    for p in HERE.glob("*.py"):ast.parse(p.read_text())
    pool=module("gpu0_shared_pool_v1");runtime=module("gpu0_runtime_adapter_v1")
    guard=module("gpu0_native_guard_v1");bootstrap=module("gpu0_bootstrap_renewal_v1")
    context=module("gpu0_h050_context_v1")
    check("standard_library_imports_only_no_Torch", "torch" not in sys.modules and "torch_geometric" not in sys.modules)
    row=dict(index=0,uuid=pool.GPU0_UUID,used_mib=6,total_mib=81920,utilization=0,compute=[])
    check("private_original_inventory_validator_accepts_only_required0",set(pool.base.checked_inventory(lambda:[row]))=={0})
    failed=False
    try:pool.base.checked_inventory(lambda:[dict(row,index=4)])
    except ValueError:failed=True
    check("missingGPU0_rejected",failed)
    old=dict(pid=101,uid=1000,start_ticks=111,state="R")
    job=dict(logical_job="H050",state="draining",process_identities=[old],requested=dict(peak_mib=30720))
    app=dict(pid=101,name="[Not Found]",used_mib=1024)
    check("exited_historical_NVML_name_is_external_not_title_failure",pool.stale_compute_classification_v1(app,{}, {},[job])==(None,True))
    check("reused_historical_PID_is_external",pool.stale_compute_classification_v1(app,{}, {101:dict(old,start_ticks=222)},[job])==(None,True))
    failed=False
    try:pool.stale_compute_classification_v1(app,{}, {101:old},[job])
    except ValueError:failed=True
    check("live_owned_identity_missing_assignment_fails_closed",failed)
    ctl=object.__new__(pool.Shared0Controller)
    ctl.inventory=lambda:[dict(row,used_mib=1024,compute=[app])]
    ctl.metadata=dict(uuid=pool.GPU0_UUID);ctl.jobs={"H050":job};ctl.pid_jobs={};ctl.processes={}
    ctl.device=row;ctl.headroom=4096;ctl.cap=dict(maximum_concurrent_jobs=2);ctl.ambiguous=[]
    memory=ctl._memory()
    check("stale_bytes_still_charged_and_do_not_block_owned_drain",memory["unattributed_mib"]==1024 and memory["owned_compute_jobs"]==[] and len(memory["stale_historical_compute"])==1)
    check("original_shared_pair_memory_arithmetic", (28+2+40+2+4)*1024+6<=81920)
    # Qualify only source transformations. Do not import scientific modules/guard APIs.
    for path,name,before,after in [
        (runtime.H050_COMMON,"validate_spec_v1",'spec.get("physical_gpu") in (4,5)','spec.get("physical_gpu")==0'),
        (runtime.H050_CONTEXT,"build_context_v1",'spec["physical_gpu"] in (4,5)','spec["physical_gpu"]==0'),
        (runtime.C3_CONTRACT,"validate_request",'req["physical_gpu"] in (4,5)','req["physical_gpu"]==0')]:
        candidate=runtime.function_source_v1(path,name,[(before,after)])
        original=ast.get_source_segment(runtime.exact(path),next(n for n in ast.parse(runtime.exact(path)).body if isinstance(n,ast.FunctionDef) and n.name==name))
        check("single_operational_eligibility_delta_"+name, candidate.replace(after,before)==original)
    source=(HERE/"gpu0_shared_pool_v1.py").read_text()
    check("new_bootstrap_actual_argv_and_bound_checker",pool.legacy.RENEWAL_BOOTSTRAP_REF["path"].endswith("/gpu0_bootstrap_renewal_v1.py") and pool.ref(pool.legacy.RENEWAL_BOOTSTRAP_REF["path"])==pool.legacy.RENEWAL_BOOTSTRAP_REF and "CHECKER_REF" in pool.Shared0Controller._check.__code__.co_names)
    check("server_reply_deadline_reset_and300s_client_bound",'state["deadline"] = time.monotonic() + 30' in pool.runtext and "settimeout(300)" in source)
    check("GPU0_bootstrap_module_matches_spawn",bootstrap.MODULE=="research.reviews.GPU0_shared_execution_source_20261016_v1.gpu0_shared_pool_v1")
    # Compare original native helper ASTs, except the single eligibility message.
    init=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=="Shared0Controller")
    init=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    assignment=next(n for n in ast.walk(init) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=="socket_path" for t in n.targets))
    check("GPU0_controller_and_client_share_exact_endpoint",isinstance(assignment.value,ast.Call) and isinstance(assignment.value.func,ast.Name) and assignment.value.func.id=="socket_name" and pool.main.__globals__["socket_name"] is pool.socket_name and pool.socket_name(dict(pool="fixture",plan_ref=dict(path="synthetic",sha256="a"*64))).startswith("\0arggnn_gpu0_fixture_"))
    original=(ROOT/"src/arggnn/native_argformer_exec.py").read_text();new=(HERE/"gpu0_native_guard_v1.py").read_text()
    for name in ("canonical_uuid","project_path","apply_and_validate_title"):
        a=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name==name)
        check("unchanged_original_native_helper_"+name,ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
    initial=json.loads((HERE/"preserved_sources_v1.json").read_text())
    check("frozen_sources_unchanged",all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in initial.items()))
    result=dict(schema="GPU0_narrow_direct_source_validation_v1",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=results,cases_passed=len(results),
        no_live_inventory_or_process_or_GPU_or_lock_or_socket_probe=True,no_budget_guard_or_admission_called=True,no_scientific_module_import_or_data_truth_checkpoint_read=True,
        actual_runtime_route_tested=False,validation_scope="AST/imports and contained pure synthetic ownership/memory inputs only",
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")})
    (HERE/"source_validation.actual_v1.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NARROW_SOURCE_CASES_PASS",len(results))
if __name__=="__main__":
    try:main()
    except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)

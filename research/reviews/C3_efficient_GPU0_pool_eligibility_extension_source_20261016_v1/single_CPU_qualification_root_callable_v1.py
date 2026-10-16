"""Root-deferred single CPU FP32 qualification. No CLI or import-time execution.
Not an admission or result certificate. Root must consume the actual terminal.
"""
from pathlib import Path
import hashlib,importlib.util,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
FIXTURE_REF=dict(path="research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/single_FP32_qualification_deferred_v1.py",
 sha256="3657bdf6726dee11bfa187d7e6f65ba156fc470159507ae27e1e043a228b01d7")
_RAN=False

def run_single_CPU_qualification_v1(*,resource_check):
    global _RAN
    if _RAN or not callable(resource_check):raise ValueError("One root-guarded qualification invocation only")
    resource_check("before_single_new_FP32_qualification_source_load")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Root CPU-only qualification environment required")
    p=ROOT/FIXTURE_REF["path"]
    if p.resolve()!=p or not p.is_file() or p.is_symlink() or p.stat().st_size>1048576:
        raise ValueError("Canonical bounded exact deferred fixture source")
    if hashlib.sha256(p.read_bytes()).hexdigest()!=FIXTURE_REF["sha256"]:
        raise ValueError("Exact already-sealed new query/cache fixture")
    _RAN=True
    spec=importlib.util.spec_from_file_location("_C3_efficient_single_CPU_qualification_fixture",p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run_once_v1(resource_check=resource_check,device="cpu")
    resource_check("after_single_new_FP32_qualification")
    if result.get("passed") is not True or result.get("optimizer_steps")!=1 or result.get("precision")!="fp32":
        raise ValueError("One genuine ordinaryFP32 gradient/AdamW comparison required")
    return dict(schema="C3_efficient_single_CPU_qualification_callable_result_v1",fixture_ref=FIXTURE_REF,
                fixture_result=result,root_terminal_consumption_not_asserted=True,
                admission_or_execution_authority_granted=False)

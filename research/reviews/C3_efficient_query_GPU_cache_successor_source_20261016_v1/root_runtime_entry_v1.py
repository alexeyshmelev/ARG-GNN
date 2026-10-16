"""Root-invoked future C3 sharedGPU0 runtime. No CLI or launch authority.
Private original GPU0 wrappers, same operational guards, one new TRAIN hook.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1"
SOURCE=HERE+"/root_runtime_entry_v1.py"
BASE_REF=dict(path="research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_runtime_adapter_v1.py",
 sha256="957ec724bb345a76932d5ea43b0df527bc9d12b068815db91cf17da1b0c85896")
INTEGRATION_REF={'path': 'research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/integration_api_v1.py', 'sha256': 'bd7bf2151084ce64dacc1bd59b9d0e5b587b97ff2c891cc3a5772a29c294a68b'}
STEP_REF={'path': 'research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/efficient_step_v1.py', 'sha256': '0df05b1957affc222ca1c11d9f34cc3277f5a3098834fb36fedee64bb242d2e8'}
CACHE_REF={'path': 'research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/single_parent_device_cache_v1.py', 'sha256': 'd19879458474883e125b7efc7cbd649839f25c3db22ecb876ba3f7f5ab9b49b6'}
GUARD="research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py"

def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    require(isinstance(path,str) and not Path(path).is_absolute() and ".." not in Path(path).parts,
            "Exact project-relative source/metadata")
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
            and p.stat().st_size<=1048576,"Canonical bounded exact source/metadata")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(record,label):
    require(ref(record["path"])==record,"Exact future runtime source")
    spec=importlib.util.spec_from_file_location(label,ROOT/record["path"])
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def authenticate_binding_v1(j,spec):
    require(j["worker_kind"]=="C3" and j["runtime_entry_ref"]==ref(SOURCE)
            and ref(j["input_ref"]["path"])==j["input_ref"]
            and ref(j["storage_ref"]["path"])==j["storage_ref"],
            "Exact admitted future C3 entry/input/storage")
    require(spec["job_id"]==j["job_id"] and spec["phase"]=="train"
            and spec["refs"]["storage"]==j["storage_ref"]
            and spec["operational_GPU0_runtime_ref"]==ref(SOURCE)
            and spec["operational_GPU0_guard_ref"]==ref(GUARD)
            and spec["physical_gpu"]==0 and spec["execution_authorized"] is True
            and spec["GPU_authorized"] is True and spec["root_authorizer"]=="root_agent"
            and spec["max_cpu_rss_bytes"]==128*2**30 and spec["max_cuda_reserved_bytes"]==40*2**30,
            "Fresh explicit source family, original whole-process caps and actual GPU0 guard")
    # Stdlib-only module and bounded JSON authentication before numerical imports.
    integration=load(INTEGRATION_REF,"_C3_efficient_future_integration")
    bounds_ref=spec["efficient_training_caps_ref"]
    bounds=integration.read_bounds_v1(bounds_ref)
    require(j["cuda_cap_bytes"]==bounds["memory_caps"]["max_successor_cuda_reserved_bytes"]
            and 0<j["cuda_cap_bytes"]<=spec["max_cuda_reserved_bytes"],
            "Actual logical scheduling cap equals additional abort cap, original40GiB guard retained")
    required=[ref(SOURCE),INTEGRATION_REF,STEP_REF,CACHE_REF,bounds_ref,bounds["qualification_ref"],BASE_REF,ref(GUARD)]
    require(all(ref(r["path"])==r and spec["source_sha256"].get(r["path"])==r["sha256"] for r in required),
            "Every new source/caps/actual qualification and original wrapper/guard is directly input-bound")
    storage=json.loads((ROOT/j["storage_ref"]["path"]).read_text())
    reservation=storage["reservation"];bindings=reservation["consumer_bindings"]
    require(storage["kind"]=="project_budget_v9_storage_record" and storage["active"] is True
            and storage["accounting_mode"]=="prospective_980gb_v1" and storage["consumer_uses_v9_guard"] is True
            and storage["job_id"]==j["job_id"] and storage["output_directory"]==spec["output_directory"]
            and reservation["output"]==str(ROOT/spec["output_directory"])
            and isinstance(bindings,list) and 0<len(bindings)<=64
            and len({x["path"] for x in bindings})==len(bindings)
            and all(r in bindings for r in required),
            "Original guarded actual charge, at most64 direct consumers, exact new source/caps closure")
    return integration,bounds_ref

def invoke_job_v1(j):
    require(ref(j["input_ref"]["path"])==j["input_ref"],"Exact future input")
    spec=json.loads((ROOT/j["input_ref"]["path"]).read_text())
    integration,bounds_ref=authenticate_binding_v1(j,spec)
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    require(validate_lease()["physical_gpu"]==0,"Original inherited GPU0 lease installed before numerical imports")
    apply_and_validate_title()
    base=load(BASE_REF,"_C3_original_GPU0_runtime_for_efficient_family")
    text=(ROOT/BASE_REF["path"]).read_text()
    n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="run_c3_v1")
    source=ast.get_source_segment(text,n)
    needle="contract.make_config=make_config"
    require(source.count(needle)==1,"One private operational hooks insertion")
    source=source.replace(needle,needle+"\n    original_hooks=contract.hooks\n    contract.hooks=lambda: efficient_integration.HooksProxyV1(original_hooks(),efficient_bounds_ref)")
    namespace=dict(base.__dict__,efficient_integration=integration,efficient_bounds_ref=bounds_ref)
    exec(compile(source,"<original_GPU0_C3_private_contract_with_efficient_TRAIN_hook>","exec"),namespace)
    return namespace["run_c3_v1"](j)

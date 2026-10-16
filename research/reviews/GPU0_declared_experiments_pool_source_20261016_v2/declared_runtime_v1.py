"""Exact finite envelope dispatcher; no scientific scope/clock mutation."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_declared_experiments_pool_source_20261016_v2"
SOURCE=HERE+"/declared_runtime_v1.py"
CONTRACT_REF={"path":HERE+"/finite_jobs_v1.py","sha256":"53fb27b18c72582c86c227fb2b4687ac6a18a3d8b0df003b14f4b520bfeda4b4"}
CATALOGUE_REF={"path":"research/reviews/GPU0_declared_experiments_pool_source_20261016_v1/finite_catalogue.v1.json","sha256":"ee6f2da59c80295296495029ab90041e76a0b56283ca46b9e1eeecb9506310c3"}

def require(ok,message):
    if not ok:raise ValueError(message)

def load(record,label):
    p=ROOT/record["path"]
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
        and p.stat().st_size<=32*2**20 and hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"],
        "Canonical exact finite source before import")
    spec=importlib.util.spec_from_file_location(label,p);m=importlib.util.module_from_spec(spec)
    sys.modules[label]=m;spec.loader.exec_module(m);return m

def invoke_job_v1(j):
    c=load(CONTRACT_REF,"_GPU0_declared_runtime_finite_contract")
    require(c.ref(CATALOGUE_REF["path"])==CATALOGUE_REF and c.ref(SOURCE)==j["runtime_entry_ref"],
        "Exact sealed finite contract/catalogue/dispatcher")
    envelope,request,method=c.envelope_parts_v1(j["input_ref"])
    require(c.job_from_envelope_v1(j["input_ref"])==j and envelope["operational_storage_ref"]==j["storage_ref"],
        "Exact outer job; original nested scientific method remains unchanged")
    catalogue=c.catalogue_v1();slot=request["slot"]
    require(slot in catalogue["slots"],"No arbitrary module or phase")
    spec=c.read(method["input_ref"]);contract=catalogue["slots"][slot]
    require(all(method[k]==contract[k] for k in ("worker_kind","module","module_ref"))
        and spec["phase"]==contract["phase"] and spec["job_id"]==method["job_id"]
        and spec["refs"]["storage"]==method["storage_ref"] and spec["physical_gpu"]==0,
        "Unchanged exact scientific slot/input/storage")
    require(contract.get("known_runtime_ref")==method["runtime_entry_ref"],
        "Only exact reviewed runtime; no arbitrary unauthenticated callable")
    require(method["runtime_entry_ref"] in request["runtime_source_refs"]
        and method["module_ref"] in request["runtime_source_refs"]
        and all(r in request["runtime_source_refs"] for r in contract.get("required_runtime_source_refs",[])),
        "Exact per-slot runtime/science closure")
    for r in request["runtime_source_refs"]:require(c.ref(r["path"])==r,"Actual runtime/source bytes")
    require(time.time()<c.expiry(envelope["expires_at_utc"])<=c.expiry(spec["expires_at_utc"]),
        "Unchanged method absolute expiry; no fresh scientific clock")
    require(os.environ.get("ARGGNN_PHYSICAL_GPU")=="0"
        and os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==method["job_id"]
        and os.environ.get("ARGGNN_JOB_NAME","").startswith("gpu0_shared_v1:"),
        "Inherited canonical physical0 logical lease lineage")
    # The original runtime validates its complete original scientific guard,
    # storage, caps and literal args. Do not pre-guard/re-reserve that scope.
    runtime=load(method["runtime_entry_ref"],"_GPU0_declared_original_method_runtime")
    return runtime.invoke_job_v1(method)

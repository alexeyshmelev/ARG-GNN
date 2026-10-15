"""Optional native third-job assembly. No state/guard/probe or admission APIs."""
from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_shared_admission_draft_20261016_v1"
SOURCE=HERE+"/native_assembly_v1.py"
CORE_REF=dict(path=HERE+"/root_admission_draft_v1.py",sha256="48bfab0e57aab5f97f761a16a62ec96a01bf19af8b59e352050f27822f9146c4")
NATIVE="research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/"

def core():
    p=ROOT/CORE_REF["path"]
    if hashlib.sha256(p.read_bytes()).hexdigest()!=CORE_REF["sha256"]:raise ValueError("Frozen bound admission core changed")
    sp=importlib.util.spec_from_file_location("_GPU0_optional_native_original_core",p)
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m

def declared_job(record,a,admitted_ref):
    c=core();c.require(a.get("optional_native_shared_job_authorized_after_separate_actual_admission") is True
        and a.get("native_40GiB_operational_ceiling_authorized") is True,"Root optional native authority")
    admitted=c.read(admitted_ref);ir=admitted["input_ref"];spec=c.read(ir);caps_ref=admitted["caps_ref"];caps=c.read(caps_ref)
    storage_ref=spec["refs"]["storage"];storage=c.read(storage_ref);runtime=spec["operational_GPU0_runtime_ref"];science=spec["main_source_ref"]
    c.require(spec["physical_gpu"]==0 and spec["max_cuda_reserved_bytes"]==caps["max_cuda_reserved_bytes"]==40*2**30
        and spec["max_cpu_rss_bytes"]==caps["max_cpu_rss_bytes"] and spec["job_id"]==storage["job_id"]
        and spec["output_directory"]==storage["output_directory"] and admitted.get("storage")==storage,"Authentic own native storage/caps/input")
    c.require(runtime["path"]==NATIVE+"shared_native_runtime_entry_v1.py" and science["path"].startswith(NATIVE)
        and runtime in storage["reservation"]["consumer_bindings"] and science in storage["reservation"]["consumer_bindings"],"Actual native runtime/science direct")
    c.require(spec["operational_pool_source_ref"]==c.ref(c.POOL)
        and spec["operational_GPU0_guard_ref"]==c.ref(c.GUARD0) and spec["operational_GPU0_policy_ref"]==a["GPU0_policy_ref"],"Same authentic actual shared route")
    closure_ref=spec["transitive_source_closure_ref"];closure=c.read(closure_ref)
    c.require(closure_ref in storage["reservation"]["consumer_bindings"]
        and closure.get("source_sha256")==spec.get("source_sha256")
        and all(closure["source_sha256"].get(r["path"])==r["sha256"] for r in c.pool_required_refs(a)),"Complete exact original native transitive pool closure before controller admission")
    deadline=caps["operational_deadline_seconds"]
    c.require(type(deadline) is int and 0<deadline<=259200
        and (deadline<=86400 or a.get("native_72h_operational_deadline_authorized") is True),"Explicit native original deadline")
    phase=spec["phase"];c.require(phase in ("mlm","fit","embeddings"),"Original native scientific stage only")
    j=dict(worker_kind="native_ARGformer",job_id=spec["job_id"],module="native_ARGformer_VAL000_"+phase,module_ref=science,
        runtime_entry_ref=runtime,input_ref=ir,storage_ref=storage_ref,cuda_caps_ref=caps_ref,
        cuda_cap_bytes=40*2**30,cpu_rss_cap_bytes=caps["max_cpu_rss_bytes"],operational_deadline_seconds=deadline,
        margin_mib=2048,peak_mib=43008,log_limit_bytes=2**20,env={},
        args=["--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]])
    # The native reservation already authenticates its full own source map.
    # Controller directly binds only the actual native entries and this assembly.
    return j,[c.ref(SOURCE),runtime,science]

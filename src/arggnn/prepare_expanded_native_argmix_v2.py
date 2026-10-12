"""Root CPU supervisor: native H038 ARGMix 32 TRAIN then 8 VAL, followed by assembly."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/prepare_expanded_native_argmix_v2.py"
ADAPTER="src/arggnn/expanded_native_argmix_v1.py"
ADAPTER_SHA="00c8e57423d38b484bda3cfeb126c188add63c383e399df874a1a2e55e4bca23"
STORAGE="src/arggnn/project_storage_terminal_retired_scopes_v1.py"
MODE="journal_terminal_retired_scopes_credit_v1"
STORAGE_SHA="bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab"
NATIVE_PREFIX=ROOT/".runtime/official_argmix_abi210_v1/cu129/env"
PARENTS=tuple("ap_expanded_%s_%03d"%(split,i) for split,n in (("train",32),("val",8)) for i in range(n))
PRODUCER="src/arggnn/prepare_expanded_full_panel_v1.py"
ORIGINAL_PRODUCER="src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py"
SOURCE_PATHS=(SOURCE,ADAPTER,"src/arggnn/official_argmix_biological.py",
    "src/arggnn/full_panel_native_baseline_contracts_v1.py",
    "src/arggnn/mutation_routed_full_panel_observations_v1.py",
    "src/arggnn/mutation_routed_parent_loader_v1.py",
    STORAGE,"src/arggnn/project_storage_terminal_success_scopes_v1.py","src/arggnn/project_storage_terminal_scopes_v1.py",
    "src/arggnn/project_storage_cpu_closed_scopes_v1.py","src/arggnn/project_storage_closed_scopes_v1.py",
    "scripts/project_budget_v9.py","scripts/launch_reserved_gpu_v6.py",
    "src/arggnn/project_budget_v9_consumer.py",PRODUCER,ORIGINAL_PRODUCER)
SCHEMA="expanded_native_argmix_preparation_v1_root_admission"


def path_v1(value):
    path=Path(value)
    if path.is_absolute():path=path.relative_to(ROOT)
    if not path.parts or ".." in path.parts:raise ValueError("Canonical project-local path required")
    result=ROOT/path
    for count in range(1,len(path.parts)+1):
        if ROOT.joinpath(*path.parts[:count]).is_symlink():raise ValueError("Project symlink forbidden")
    return result


def ref_v1(value):
    path=path_v1(value);digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=digest.hexdigest())


def json_v1(record,expected=None):
    if (not isinstance(record,dict) or set(record)!={"path","sha256"}
            or expected is not None and record["path"]!=expected
            or not record["path"].endswith(".json") or "/truth/" in record["path"]):
        raise ValueError("Bound nontruth JSON metadata role required")
    path=path_v1(record["path"]);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record["sha256"]:raise ValueError("Metadata pin changed")
    return json.loads(raw)


def utc_v1():return datetime.now(timezone.utc).isoformat()
def digest_v1(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def admit_v1(input_ref):
    spec=json_v1(input_ref)
    expiry=datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    if (spec.get("schema")!=SCHEMA or spec.get("root_authorizer")!="root_agent"
            or spec.get("execution_authorized") is not True or spec.get("status")!="active"
            or spec.get("GPU_authorized") is not False or spec.get("fit_authorized") is not False
            or spec.get("outer_included") is not False
            or spec.get("TRAIN_and_VAL_truth_preparation_authorized") is not True
            or spec.get("parent_order")!=list(PARENTS)
            or spec.get("workers")!=1 or os.environ.get("CUDA_VISIBLE_DEVICES")!=""
            or expiry.tzinfo is None or expiry<=datetime.now(timezone.utc)):
        raise ValueError("Active exact CPU native-preparation root admission required")
    for name in ("output_cap_bytes","max_cpu_rss_bytes","readiness_timeout_seconds"):
        if type(spec.get(name)) is not int or spec[name]<=0:raise ValueError("Positive root resource bounds required")
    if not .5<=spec.get("poll_seconds",2.)<=30:raise ValueError("Bounded readiness polling required")
    sources=spec.get("source_sha256")
    if not isinstance(sources,dict) or set(sources)!=set(SOURCE_PATHS):raise ValueError("Exact supervisor/scientific/storage source closure required")
    for name,sha in sources.items():
        if ref_v1(name)!={"path":name,"sha256":sha}:raise ValueError("Source pin changed: "+name)
    if (sources[ADAPTER]!=ADAPTER_SHA or spec["source_ref"]!=ref_v1(SOURCE)
            or spec["storage_adapter_ref"]!={"path":STORAGE,"sha256":STORAGE_SHA}
            or spec["storage_adapter_ref"]!=ref_v1(STORAGE)):
        raise ValueError("Frozen scientific adapter and root-pinned successor storage required")
    cfg=json_v1(spec["config_ref"])
    generation=json_v1(spec["generation_ref"],"configs/expanded_full_panel_v1.json")
    registration=json_v1(spec["registration_ref"],"research/reviews/expanded_full_panel_v1_root/registration.json")
    if (cfg.get("execution_guard_ready") is not True
            or cfg["generation_ref"]!=spec["generation_ref"] or cfg["registration_ref"]!=spec["registration_ref"]
            or registration["parent_registry"]!=generation["parent_registry"]
            or generation["output_root"]!="data/expanded_full_panel_v1"):
        raise ValueError("Frozen H038 generation/native config binding required")
    runtime=json_v1(spec["runtime_ref"]);json_v1(spec["build_ref"]);json_v1(spec["CPU_library_admission_ref"])
    if (runtime.get("device")!="cpu" or runtime.get("CUDA_VISIBLE_DEVICES")!=""
            or runtime.get("native_prefix")!=str(NATIVE_PREFIX.relative_to(ROOT))
            or runtime.get("job_id")!=spec["job_id"] or runtime.get("source_ref")!=spec["source_ref"]):
        raise ValueError("Root-bound native CPU runtime required")
    for name in ("maximum_native_trees","maximum_native_sites"):
        bounds=spec[name]
        if not isinstance(bounds,dict) or set(bounds)!=set(PARENTS) or any(type(v) is not int or v<=0 for v in bounds.values()):
            raise ValueError("Every parent needs an admitted storage shape bound")
    storage=spec["storage"]
    if (storage["accounting_mode"]!=MODE or storage["output_directory"]!=spec["output_directory"]
            or storage["reserved_new_bytes"]!=spec["output_cap_bytes"]):
        raise ValueError("Successor terminal accounting output/cap binding required")
    needed=[spec[name] for name in ("source_ref","config_ref","generation_ref","registration_ref",
        "runtime_ref","build_ref","CPU_library_admission_ref","storage_adapter_ref")]
    needed.extend({"path":name,"sha256":sha} for name,sha in sources.items())
    if any(record not in storage["reservation"]["consumer_bindings"] for record in needed):
        raise ValueError("Every native/supervisor/source/runtime binding must be reserved")
    module=importlib.import_module("arggnn.project_storage_terminal_retired_scopes_v1")
    if module.MODE!=MODE:raise ValueError("Successor accounting mode drift")
    return spec,cfg,generation,module


def guard_v1(spec,module,*,first=False):
    return module.budget_check_v9(spec["storage"],first=first,boundary=True)


def make_guard_v1(spec,module):
    expiry=datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    def guard(planned=0):
        if datetime.now(timezone.utc)>=expiry:raise ValueError("Native preparation authority expired")
        return module.budget_check_v9(spec["storage"],boundary=True,required_write_bytes=int(planned))
    return guard


def save_v1(path,value,guard):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    guard(len(raw)+8192)
    with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return ref_v1(path)


def ready_v1(parent,spec,generation):
    base="data/expanded_full_panel_v1/"+parent
    completed=path_v1(base+"/expanded_parent_completed.json")
    if not completed.exists():return None
    value=json_v1(ref_v1(completed))
    split="train" if "_train_" in parent else "val"
    if (value.get("schema")!="expanded_full_panel_v1_parent_completion"
            or value.get("parent_id")!=parent or value.get("split")!=split
            or value.get("generation_completed") is not True or value.get("Relate_completed") is not True
            or value.get("source_ref")!=ref_v1(PRODUCER) or value.get("original_producer_ref")!=ref_v1(ORIGINAL_PRODUCER)
            or value.get("config_content_sha256")!=digest_v1(generation)
            or value.get("parent_registry_sha256")!=digest_v1(generation["parent_registry"])
            or value.get("outer_truth_evaluated") is not False):
        raise ValueError("Complete original same-parent generation/Relate lineage required")
    entry_ref=value["observation_entry_ref"]
    if entry_ref["path"]!=base+"/full_panel_observed_entry.json":raise ValueError("Original observed entry path required")
    inference=json_v1(value["inference_ref"],base+"/relate/inference.json")["post_five_iterations"]
    if (inference["trees"]>spec["maximum_native_trees"][parent]
            or inference["sites"]>spec["maximum_native_sites"][parent]):
        raise ValueError("Complete parent exceeds admitted storage shape bound; increase reservation without truncation")
    generated=json_v1(value["generation_ref"],base+"/generation.json")
    if generated["parent_id"]!=parent:raise ValueError("Generation parent identity differs")
    binding=dict(split=split,generation_receipt=value["generation_ref"],truth_npz={
        "path":base+"/truth/full_truth_intervals.npz",
        "sha256":generated["files"]["truth/full_truth_intervals.npz"]["sha256"]})
    return entry_ref,binding


def worker_v1(input_ref,parent):
    spec,cfg,generation,module=admit_v1(input_ref);guard=make_guard_v1(spec,module)
    if Path(sys.prefix).resolve()!=NATIVE_PREFIX:raise ValueError("Worker requires existing qualified native interpreter")
    from . import expanded_native_argmix_v1 as adapter
    cfg=adapter.load_config_v1(spec["config_ref"])
    modules=adapter.admit_native_v1(spec["build_ref"],spec["CPU_library_admission_ref"],cfg)
    modules[0].set_num_threads(1)
    if modules[0].cuda.is_initialized():raise ValueError("CPU native preparation created CUDA context")
    output=path_v1(spec["output_directory"])
    if parent=="assemble":
        products=json_v1(ref_v1(output/"parent_products.json"))["parents"]
        result=adapter.assemble_development_v1(products,output/"assembled",cfg,guard=guard)
        save_v1(output/"assembled_product.json",dict(native_prepared_ref=result,GPU_used=False,fit_performed=False),guard)
    else:
        if parent not in PARENTS:raise ValueError("No outer/unknown parent worker")
        ready=ready_v1(parent,spec,generation)
        if ready is None:raise ValueError("Worker parent is not completed")
        directory=output/"parents"/parent;guard(32768);directory.mkdir()
        frame=adapter.prepare_observed_parent_v1(parent,ready[0],directory/"observed",cfg,guard=guard)
        product=adapter.prepare_native_parent_v1(frame,ready[1],directory/"native",cfg,guard=guard)
        save_v1(directory/"product.json",dict(parent_id=parent,native_parent_ref=product,GPU_used=False),guard)
    if modules[0].cuda.is_initialized():raise ValueError("CPU preparation unexpectedly created CUDA context")


def owned_rss_v1(pid):
    """Only this spawned worker and its explicitly listed descendants; no global scan."""
    todo=[pid];seen=set();total=0
    while todo:
        current=todo.pop()
        if current in seen:continue
        seen.add(current)
        try:
            for line in Path("/proc/%d/status"%current).read_text().splitlines():
                if line.startswith("VmRSS:"):total+=int(line.split()[1])*1024
            children=Path("/proc/%d/task/%d/children"%(current,current)).read_text().split()
            todo.extend(int(value) for value in children)
        except FileNotFoundError:pass
    return total,sorted(seen)


def native_env_v1(output):
    env=os.environ.copy()
    for name in ("PYTHONHOME","LIBTORCH_BYPASS_VERSION_CHECK","LIBTORCH","LIBTORCH_INCLUDE",
                 "LIBTORCH_LIB","LIBTORCH_STATIC","LIBTORCH_CXX11_ABI","TORCH_CUDA_VERSION"):
        env.pop(name,None)
    env.update(CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(ROOT/"src")+":"+str(ROOT),PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",CONDA_PREFIX=str(NATIVE_PREFIX),WANDB_MODE="disabled",
        LIBTORCH_USE_PYTORCH="1",HF_HUB_OFFLINE="1",HDF5_DIR=str(NATIVE_PREFIX),
        LD_LIBRARY_PATH=str(NATIVE_PREFIX/"lib")+":"+str(NATIVE_PREFIX/"lib/python3.12/site-packages/torch/lib"),
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",RAYON_NUM_THREADS="1")
    env["PATH"]=str(NATIVE_PREFIX/"bin")+os.pathsep+env.get("PATH","")
    sandbox=output/"sandbox"
    for key in ("TMPDIR","TMP","TEMP"):env[key]=str(sandbox/"tmp")
    for key in ("XDG_CACHE_HOME","HF_HOME","TORCH_HOME","MPLCONFIGDIR","NUMBA_CACHE_DIR",
                "TRITON_CACHE_DIR","CUDA_CACHE_PATH","TORCHINDUCTOR_CACHE_DIR"):env[key]=str(sandbox/"cache")
    for key in ("XDG_CONFIG_HOME","WANDB_CONFIG_DIR"):env[key]=str(sandbox/"config")
    for key in ("XDG_DATA_HOME","WANDB_DIR","WANDB_CACHE_DIR"):env[key]=str(sandbox/"data")
    return env


def run_child_v1(parent,input_ref,spec,module,output):
    guard=make_guard_v1(spec,module)
    command=[str(NATIVE_PREFIX/"bin/python"),"-B","-m","arggnn.prepare_expanded_native_argmix_v2",
        "--execute","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"],
        "--worker",parent]
    guard(1_000_000);peak=0;observed=set();process=None
    with (output/"logs"/(parent+".log")).open("xb") as log:
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=native_env_v1(output),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save_v1(output/"logs"/(parent+".launch.json"),dict(parent_id=parent,pid=process.pid,
                process_group=process.pid,command=command,input_ref=input_ref,source_ref=spec["source_ref"],
                CUDA_VISIBLE_DEVICES="",GPU_used=False,utc=utc_v1()),guard)
            while process.poll() is None:
                rss,pids=owned_rss_v1(process.pid);peak=max(peak,rss);observed.update(pids)
                if rss>spec["max_cpu_rss_bytes"]:raise MemoryError("Owned native preparation process tree exceeds RSS admission")
                guard(0);time.sleep(2.)
            if process.returncode:raise RuntimeError("Native CPU worker failed: "+parent+"; see own log")
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    guard(0)
    return dict(parent_id=parent,command=command,returncode=process.returncode,
        peak_owned_process_tree_rss_bytes=peak,observed_owned_pids=sorted(observed))


def run_v1(input_ref):
    spec,cfg,generation,module=admit_v1(input_ref);guard=make_guard_v1(spec,module)
    module.validate_storage_v9(spec,dict(cfg,output_cap_bytes=spec["output_cap_bytes"]),spec["storage"],
        source_ref=spec["source_ref"],config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    output=path_v1(spec["output_directory"]);guard(131072);output.mkdir(parents=False,exist_ok=False)
    for directory in ("parents","logs","sandbox","sandbox/tmp","sandbox/cache","sandbox/config","sandbox/data"):
        (output/directory).mkdir()
    save_v1(output/"started.json",dict(schema=SCHEMA+"_started",input_ref=input_ref,source_ref=spec["source_ref"],
        utc=utc_v1(),parent_order=list(PARENTS),GPU_used=False,fit_performed=False),guard)
    def terminate_owned(*_):raise InterruptedError("Root native supervisor stop requested")
    signal.signal(signal.SIGTERM,terminate_owned)
    products={};commands=[];deadline=time.monotonic()+spec["readiness_timeout_seconds"]
    try:
        for parent in PARENTS:
            while ready_v1(parent,spec,generation) is None:
                if time.monotonic()>=deadline:raise TimeoutError("Original registered parent readiness expired: "+parent)
                guard(0);time.sleep(spec.get("poll_seconds",2.))
            command=run_child_v1(parent,input_ref,spec,module,output);commands.append(command)
            row=json_v1(ref_v1(output/"parents"/parent/"product.json"));products[parent]=row["native_parent_ref"]
            save_v1(output/("progress_%02d.json"%len(products)),dict(completed_parents=list(products),
                latest_parent=parent,worker=command,GPU_used=False,fit_performed=False),guard)
            print(json.dumps(dict(event="native_parent_prepared",parent_id=parent,completed=len(products))),flush=True)
        save_v1(output/"parent_products.json",dict(parents=products,parent_order=list(PARENTS)),guard)
        commands.append(run_child_v1("assemble",input_ref,spec,module,output))
        assembled=json_v1(ref_v1(output/"assembled_product.json"))
        result=dict(schema=SCHEMA+"_completed",input_ref=input_ref,source_ref=spec["source_ref"],
            native_prepared_ref=assembled["native_prepared_ref"],parent_products=products,commands=commands,
            GPU_used=False,fit_performed=False,TRAIN_parents=32,VAL_parents=8,outer_parents=0,completed_utc=utc_v1())
        save_v1(output/"completed.json",result,guard);return result
    except BaseException as error:
        failure=dict(schema=SCHEMA+"_failed",input_ref=input_ref,source_ref=spec["source_ref"],
            completed_parent_products=products,commands=commands,error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc(),GPU_used=False,automatic_replay=False,utc=utc_v1())
        try:save_v1(output/"failed.json",failure,guard)
        except Exception:print(json.dumps(failure),file=sys.stderr,flush=True)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256");parser.add_argument("--worker",default=None)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_admitted_or_executed",source_ref=ref_v1(SOURCE),
            frozen_adapter_ref={"path":ADAPTER,"sha256":ADAPTER_SHA},storage_adapter_path=STORAGE,
            accounting_mode=MODE,TRAIN=32,VAL=8,workers=1,GPU_used=False),indent=2));return
    if not args.input or not args.expected_input_sha256:raise ValueError("Pinned root input capability required")
    input_ref={"path":args.input,"sha256":args.expected_input_sha256}
    if args.worker is not None:worker_v1(input_ref,args.worker)
    else:run_v1(input_ref)


if __name__=="__main__":main()

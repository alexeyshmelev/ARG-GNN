"""Explicit pool-native adapter; standard library and no import-time actions.
Original root/direct/native wait and owned-drain machinery performs execution.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1"
SOURCE=HERE+"/shared_native_runtime_entry_v1.py"
GUARD_REF={"path":"research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py","sha256":"3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69"}
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),"Exact canonical own source/metadata path")
    require(p.stat().st_size<=4*2**20,"Bounded source/metadata read; no scientific payload")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(ref(record["path"])==record,"Exact admitted native metadata")
    return json.loads((ROOT/record["path"]).read_bytes())
def load(record,label):
    require(ref(record["path"])==record,"Exact native operational source")
    spec=importlib.util.spec_from_file_location(label,ROOT/record["path"])
    m=importlib.util.module_from_spec(spec);sys.modules[label]=m;spec.loader.exec_module(m);return m
def invoke_job_v1(j):
    require(j["worker_kind"]=="native_ARGformer" and j["runtime_entry_ref"]==ref(SOURCE),"Exact explicit shared-native runtime")
    spec=read(j["input_ref"]);storage=read(j["storage_ref"]);caps=read(j["cuda_caps_ref"])
    phase=spec["phase"]
    require(phase in ("mlm","fit","embeddings") and j["module"]=="native_ARGformer_VAL000_"+phase,
        "One exact native scientific phase; no additional engineering run")
    stage=HERE+("/mlm_v1" if phase=="mlm" else "")
    entry=stage+("/entry_mlm980_v1.py" if phase=="mlm" else "/entry_postMLM_v1.py")
    require(j["module_ref"]==ref(entry) and spec["main_source_ref"]==j["module_ref"]
        and j["args"]==["--execute","--input",j["input_ref"]["path"],"--expected-input-sha256",j["input_ref"]["sha256"]],
        "Exact native entry and literal original native input arguments")
    require(spec["job_id"]==storage["job_id"]==j["job_id"] and spec["refs"]["storage"]==j["storage_ref"]
        and spec["caps_ref"]==j["cuda_caps_ref"] and spec["output_directory"]==storage["output_directory"]
        and spec["max_cuda_reserved_bytes"]==caps["max_cuda_reserved_bytes"]==j["cuda_cap_bytes"]==40*2**30
        and spec["max_cpu_rss_bytes"]==caps["max_cpu_rss_bytes"]==j["cpu_rss_cap_bytes"]
        and j["operational_deadline_seconds"]==caps["operational_deadline_seconds"]
        and spec["operational_GPU0_runtime_ref"]==ref(SOURCE) and spec["operational_GPU0_guard_ref"]==GUARD_REF,
        "Separately charged native storage, exact operational caps and original input")
    require(os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==j["job_id"]
        and os.environ.get("ARGGNN_JOB_NAME","").startswith("gpu0_shared_v1:")
        and os.getpid()==os.getpgrp()==os.getsid(0),"Pool worker owns its dedicated session/logical identity")
    guard=load(GUARD_REF,"_shared_native_pool_guard")
    guard.install_before_torch_v1();lease=guard.validate_lease();guard.apply_and_validate_title()
    require(lease["physical_gpu"]==0 and lease["launcher_metadata"]["job"]==os.environ["ARGGNN_JOB_NAME"],"Canonical inherited pool lease")
    sys.path[:0]=[str(ROOT/stage),str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    driver=load(ref(stage+"/root_native_driver_v1.py"),"_shared_native_actual_root_driver")
    os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(driver.c.POLICY_REF,separators=(",",":"))
    packet,_=driver.c.phase_paths(phase);admitted=driver.c.read(driver.c.ref(packet/"admitted.json"))
    require(admitted["input_ref"]==j["input_ref"],"Exact already separately admitted native packet")
    # No second physical lock and no lease metadata mutation: original root
    # supervisor passes the inherited FD to the narrow shared native launcher.
    return driver.launch(phase)

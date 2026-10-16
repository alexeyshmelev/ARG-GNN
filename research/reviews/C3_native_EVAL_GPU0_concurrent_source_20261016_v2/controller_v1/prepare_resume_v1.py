"""Native plus original-EVAL future ROOT metadata utilities. No import-time state/guard/probe."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,math,re,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
MIB=2**20
GPU0_UUID='GPU-ff06ca1f-403a-f2d0-b584-71e177452667'
OLD_CAP_REF={'path': 'research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json', 'sha256': '4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870'}
FINITE_POOL_REF={'path': 'research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2/controller_v1/gpu0_declared_pool_v1.py', 'sha256': '1155deddfd4195c37682cffa6c43164bf848cc7f304f4a4281826aabeff56332'}
FINITE_CONTRACT_REF={'path': 'research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2/controller_v1/finite_jobs_v1.py', 'sha256': 'f8deeb1b5c1c2a0baaa7e3fe155b3c48c9fbbe186e68b750e447a8dcf0b6b35d'}
FINITE_LAUNCHER_REF={'path': 'research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2/controller_v1/launch_declared_gpu0_v1.py', 'sha256': '26d4236b13924f230398c0578195a85eec37295458c479ab1231d054b906a787'}
def need(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    p=ROOT/path
    need(type(path) is str and not Path(path).is_absolute() and ".." not in Path(path).parts
        and p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=32*MIB,
        "Canonical bounded source/JSON metadata only")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(r):
    need(isinstance(r,dict) and set(r)=={"path","sha256"} and ref(r["path"])==r
        and Path(r["path"]).suffix==".json","Exact existing JSON evidence, no scientific payload")
    return json.loads((ROOT/r["path"]).read_text())

def load(r,label):
    need(ref(r["path"])==r,"Pinned source before inert metadata import")
    sp=importlib.util.spec_from_file_location(label,ROOT/r["path"])
    m=importlib.util.module_from_spec(sp);sys.modules[label]=m;sp.loader.exec_module(m)
    return m

def unique(rows,*,limit=64):
    result={}
    for r in rows:
        need(set(r)=={"path","sha256"} and Path(r["path"]).suffix in (".py",".sh",".json")
            and ref(r["path"])==r,"Exact actual source or JSON ref")
        need(r["path"] not in result or result[r["path"]]==r,"No conflicting bindings")
        result[r["path"]]=r
    need(limit is None or len(result)<=limit,"Actual full required union exceeds original64direct consumers")
    return list(result.values())

def native_method_v2(input_ref):
    spec=read(input_ref);caps=read(spec['caps_ref']);storage=read(spec['refs']['storage'])
    need(spec['job_id']=='American-C3-native-val000-MLM-monitor-corrected-GPU0-mlm-v3'
        and spec['expires_at_utc']=='2026-10-18T23:41:32.476768+00:00'
        and storage['reserved_new_bytes']==7*2**30,'Exact fresh native method and original expiry')
    return dict(worker_kind='native_ARGformer',job_id=spec['job_id'],module='native_ARGformer_VAL000_mlm',
        module_ref=spec['main_source_ref'],runtime_entry_ref=spec['operational_GPU0_runtime_ref'],input_ref=input_ref,
        storage_ref=spec['refs']['storage'],cuda_caps_ref=spec['caps_ref'],cuda_cap_bytes=caps['max_cuda_reserved_bytes'],
        cpu_rss_cap_bytes=caps['max_cpu_rss_bytes'],operational_deadline_seconds=caps['operational_deadline_seconds'],
        margin_mib=2048,peak_mib=43008,log_limit_bytes=1048576,env={},
        args=['--execute','--input',input_ref['path'],'--expected-input-sha256',input_ref['sha256']])

EVAL_CONTRACT_REF={'path': 'research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v1/eval_v1/contract_eval_v1.py', 'sha256': '406c8c39005bd9f5024d3ce0092bad7de62e11f09c9a9f1d6f2ba3345866af84'}
def eval_method_v1(input_ref):
    return load(EVAL_CONTRACT_REF,'_actual_original_C3_EVAL_method').method_v1(input_ref)

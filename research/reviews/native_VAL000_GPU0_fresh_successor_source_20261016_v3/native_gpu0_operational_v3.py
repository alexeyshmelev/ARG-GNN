# Source-only canonical GPU0 monitor and exact native subprocess route.
from pathlib import Path
import argparse,ctypes,hashlib,json,os,platform
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE='research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/native_gpu0_operational_v3.py'
HERE='research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3'
GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
class ProcessInfo(ctypes.Structure):
    _fields_=[("pid",ctypes.c_uint),("usedGpuMemory",ctypes.c_ulonglong),
        ("gpuInstanceId",ctypes.c_uint),("computeInstanceId",ctypes.c_uint)]
def require(ok,message):
    if not ok:raise ValueError(message)
def reference(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
class OwnedGpuMemory:
    def __init__(self,uuid,physical_gpu):
        require(platform.system()=="Linux" and platform.machine()=="x86_64"
            and ctypes.sizeof(ProcessInfo)==24,"Reviewed Linux x86_64 NVML ABI")
        require(type(physical_gpu) is int and physical_gpu==0 and uuid==GPU0_UUID,"Only canonical physical GPU0 metadata-query device")
        self.owner_pid=os.getpid();self.uuid=uuid;self.physical_gpu=physical_gpu
        self.lib=ctypes.CDLL("libnvidia-ml.so.1")
        def bind(name,args):
            f=getattr(self.lib,name);f.argtypes=args;f.restype=ctypes.c_int;return f
        self.init=bind("nvmlInit_v2",[])
        self.by_uuid=bind("nvmlDeviceGetHandleByUUID",[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)])
        self.get_uuid=bind("nvmlDeviceGetUUID",[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint])
        self.get_index=bind("nvmlDeviceGetIndex",[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)])
        self.processes=bind("nvmlDeviceGetComputeRunningProcesses_v3",
            [ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ProcessInfo)])
        self._ok(self.init(),"NVML init")
        self.handle=ctypes.c_void_p()
        self._ok(self.by_uuid(uuid.encode(),ctypes.byref(self.handle)),"NVML exact UUID handle")
        buf=ctypes.create_string_buffer(96)
        self._ok(self.get_uuid(self.handle,buf,ctypes.c_uint(len(buf))),"NVML actual UUID")
        index=ctypes.c_uint()
        self._ok(self.get_index(self.handle,ctypes.byref(index)),"NVML physical index")
        require(buf.value.decode()==uuid and index.value==physical_gpu,"Actual NVML UUID/index equals held physical lease")
    @staticmethod
    def _ok(status,label):
        require(status==0,label+" failed NVML status="+str(status))
    def query(self):
        require(os.getpid()==self.owner_pid,"Do not reuse NVML handle across another process/fork")
        for _ in range(4):
            count=ctypes.c_uint(0);code=self.processes(self.handle,ctypes.byref(count),None)
            if code==0 and count.value==0:return []
            require(code in (0,7) and count.value<=4096,"Bounded NVML process count/API response")
            capacity=max(8,count.value*2+5);array=(ProcessInfo*capacity)();count=ctypes.c_uint(capacity)
            code=self.processes(self.handle,ctypes.byref(count),array)
            if code==7:continue
            self._ok(code,"NVML fresh compute-process memory")
            require(count.value<=capacity,"NVML output count exceeds allocated ABI array")
            return [dict(pid=int(array[i].pid),used_memory_bytes=int(array[i].usedGpuMemory)) for i in range(count.value)]
        raise ValueError("GPU process list changed too often; no stale memory result")
    def owned_bytes(self,roster):
        used=0
        for row in self.query():
            pid=row["pid"]
            if pid not in roster:continue
            try:
                statrow=Path("/proc/%d/stat"%pid).read_text().rsplit(")",1)[1].split()
                start=int(statrow[19])
                title=Path("/proc/%d/cmdline"%pid).read_bytes().split(bytes([0]))[0].decode().strip()
            except FileNotFoundError:continue
            require(roster[pid] is not None and start==roster[pid],"GPU PID/start must match current owned RSS roster")
            require(title=="AI Auto Research Pipeline Test","Exact actual owned GPU process title")
            require(0<=row["used_memory_bytes"]<2**64-1,"Actual per-process GPU memory available")
            used+=row["used_memory_bytes"]
        return used

# Explicit private routing only; no import-time action.
import ast,importlib.util,sys
from datetime import datetime,timezone
CAPABILITY_ENV="ARGGNN_NATIVE_GPU0_MONITOR_CAPABILITY_REF"
ORIGINAL_REFS={'monitor': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/gpu_monitor_v8.py', 'sha256': 'f3e0028e072204790957a2ea2327b0e3af28ef245a8570fd2a0eb1c3ade0f9d4'}, 'entry': {'path': 'research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/mlm_v1/entry_mlm980_v1.py', 'sha256': 'a9df966d7bfa84646fd4b7697462a12b38a8e93ee8e1cb16c2930dce687357f7'}, 'child': {'path': 'research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/mlm_v1/native_child_v1.py', 'sha256': '2e1ddea7f51a79a738546d7e2fcacf23200d1fa07861b64d67503b225a56d5ec'}, 'supervisor': {'path': 'research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/mlm_v1/supervisor_v1.py', 'sha256': 'e501fa2524e979a19a6a5351af0963ea65ad4991303732f28ac25ca08f652b11'}, 'launcher': {'path': 'research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/mlm_v1/launcher_mlm980_v1.py', 'sha256': 'ea991200582c2da38d208c306557c13213e54009ebef1991f0e6132ac58e85fc'}, 'driver': {'path': 'research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/mlm_v1/root_native_driver_v1.py', 'sha256': 'c63ff0b8cef4d35b2d22dcc3f10fb405be3c0455eaad8c180b99a9412ff33269'}, 'context': {'path': 'research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/mlm_v1/context_factory_v1.py', 'sha256': '13ee1c57c64a99aea790594a36b2fa25c796e46db2255f1fbbf87c49f4ebd2a2'}}
ORIGINAL_INPUT_REF={'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/input.json', 'sha256': '52242fe04f8f944dccbc81bf12940ad311d0d80f300e3cdd099be5e1748bc5d2'}
ORIGINAL_STORAGE_REF={'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/storage.json', 'sha256': '49d4097f8ceb367f1b8bd7a99649b0ae90db1045515953a0059b2e42bbf3eb81'}
ORIGINAL_CAPS_REF={'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/caps_mlm.root_active_v1.json', 'sha256': '2f9763d36a58e45f6d0f06b78c2bead18041e2fc88aaff084845414ae99d1b7e'}
FAILURE_WAIT_REF={'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/actual_wait.json', 'sha256': '87988a8996409579340809c5091c242f6a6f154c22673e9b000cf6eca4b459af'}
IPC_ADAPTER_REF={'path': 'research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/gpu0_abstract_ipc_adapter_v3.py', 'sha256': 'c69176b701999a8980ed17b187068a8b0682a99a6afa67dfe1f07dd974b43e16'}
NATIVE_JOB="American-C3-native-val000-MLM-monitor-corrected-GPU0-mlm-v3"

def _ref(path):
    require(isinstance(path,str) and not Path(path).is_absolute() and ".." not in Path(path).parts,"Contained operational source/metadata")
    p=ROOT/path
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=1048576,"Canonical bounded source/metadata")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def _read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and _ref(record["path"])==record,"Exact source/metadata SHA")
    return json.loads((ROOT/record["path"]).read_bytes())

def _authority_v1(record=None):
    record=json.loads(os.environ.get(CAPABILITY_ENV,"null")) if record is None else record
    require(isinstance(record,dict),"Exact future ROOT capability required, otherwise source-only")
    cap=_read(record);old=_read(ORIGINAL_INPUT_REF);now=datetime.now(timezone.utc).timestamp()
    require(cap.get("schema")=="GPU0_declared_logical_pool_capability_v1" and cap.get("status")=="active"
        and cap.get("root_authorizer")=="root_agent" and type(cap.get("physical_gpu")) is int and cap["physical_gpu"]==0
        and cap.get("gpu_uuid")==GPU0_UUID and cap.get("maximum_distinct_project_GPUs")==1
        and now<cap["expires_unix"] and now<datetime.fromisoformat(old["expires_at_utc"]).timestamp(),
        "Existing GPU0-only pool contract and original absolute native expiry")
    require(all(row in cap.get("source_refs",{}).values() for row in [_ref(SOURCE),IPC_ADAPTER_REF,*ORIGINAL_REFS.values()]),
        "Actual consolidated monitor/runtime/routes and retained role/source closure")
    envelope=_read(cap["jobs"][NATIVE_JOB]["input_ref"]);request=_read(envelope["request_ref"]);method=request["method_job"]
    spec=_read(method["input_ref"]);storage=_read(method["storage_ref"]);caps=_read(method["cuda_caps_ref"]);old_caps=_read(ORIGINAL_CAPS_REF)
    require(method["job_id"]==spec["job_id"]==storage["job_id"]==NATIVE_JOB
        and method["runtime_entry_ref"]==_ref(SOURCE) and method["module_ref"]==ORIGINAL_REFS["entry"]
        and spec["main_source_ref"]==method["module_ref"] and spec["refs"]["storage"]==method["storage_ref"]
        and spec["caps_ref"]==method["cuda_caps_ref"] and storage["reserved_new_bytes"]==7*2**30
        and storage["budget_components"]["consumer_adapter"]==_read(spec["completed_overlay_policy_ref"])["source_ref"]
        and storage["completed_overlay_policy_ref"]==spec["completed_overlay_policy_ref"]
        and storage["reservation"]["key"]=="H041/"+NATIVE_JOB,
        "Genuinely fresh separately charged7GiB same-science input, not failed scope reuse")
    require(spec["output_directory"]=="experiments/results/American_C3_native_val000_full30mb_MLM_monitor_corrected_GPU0_v3"
        and spec["root_auxiliary_directory"]==HERE+"/mlm_v1/admitted_mlm_v3/worker_aux"
        and spec["expires_at_utc"]==old["expires_at_utc"]==storage["expires_at_utc"],
        "Fresh exact output/aux namespace and unchanged absolute expiry")
    for key in ("parent_id","split","phase","physical_gpu","execution_gpu_uuid","max_cpu_rss_bytes","max_cuda_reserved_bytes",
                "engineering_TRAIN_only","heldout_parent_label_preserved","author_source_manifest_ref"):
        require(spec[key]==old[key],"Original scientific identity/limits unchanged: "+key)
    for key in ("preparation","preparation_config","preparation_close","counts","runtime"):
        require(spec["refs"][key]==old["refs"][key],"Original native corpus/counts/runtime input unchanged: "+key)
    require(set(caps)==set(old_caps) and all(caps[k]==old_caps[k] for k in caps if k!='driver_source_ref')
        and caps["driver_source_ref"]==ORIGINAL_REFS["driver"],"All original numeric/native settings and resource caps preserved")
    require(_read(FAILURE_WAIT_REF)["returncode"]==1,"Original failure retained without success substitution")
    require(Path.cwd().resolve()==ROOT and os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==NATIVE_JOB
        and os.environ.get("ARGGNN_JOB_NAME","").startswith("gpu0_shared_v1:"),"Exact inherited pool logical lease lineage")
    return record

def _load_exact(record,label):
    require(_ref(record["path"])==record and label not in sys.modules,"Exact original source and fresh private alias")
    loaded=importlib.util.spec_from_file_location(label,ROOT/record["path"])
    module=importlib.util.module_from_spec(loaded);sys.modules[label]=module;loaded.loader.exec_module(module)
    return module

def _install_monitor_v1():
    require(Path(__file__).resolve()==ROOT/SOURCE,"Exact combined monitor/route source")
    current=sys.modules[__name__];prior=sys.modules.get("gpu_monitor_v8")
    require(prior is None or prior is current,"Stale gpu_monitor_v8 alias must not survive bootstrap")
    sys.modules["gpu_monitor_v8"]=current
    require(sys.modules["gpu_monitor_v8"].OwnedGpuMemory is OwnedGpuMemory,"Exact GPU0 monitor alias")

def _replace_function_v1(module,record,name,before,after):
    require(Path(module.__file__).resolve()==ROOT/record["path"] and _ref(record["path"])==record,"Exact private original module")
    text=(ROOT/record["path"]).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    body=ast.get_source_segment(text,node);require(body.count(before)==1,"One exact original subprocess route: "+name)
    module.__dict__["_NATIVE_GPU0_MONITOR_ROUTE_SOURCE"]=SOURCE
    exec(compile(body.replace(before,after,1),"<"+SOURCE+":"+name+":route_only>","exec"),module.__dict__)

def patch_private_driver_v1(driver,capability_ref):
    record=_authority_v1(capability_ref)
    require(not getattr(driver,"_gpu0_monitor_route_installed_v1",False),"No duplicate driver routing")
    _replace_function_v1(driver,ORIGINAL_REFS["driver"],"launch",'str(ROOT/c.launcher_ref_v1()["path"])',
        'str(ROOT/_NATIVE_GPU0_MONITOR_ROUTE_SOURCE),"--route-role","launcher","--"')
    driver._gpu0_monitor_route_installed_v1=True;os.environ[CAPABILITY_ENV]=json.dumps(record,separators=(",",":"))
    return driver

def invoke_job_v1(j):
    capability_ref=_authority_v1()
    spec=_read(j["input_ref"]);storage=_read(j["storage_ref"]);caps=_read(j["cuda_caps_ref"])
    require(j["worker_kind"]=="native_ARGformer" and j["runtime_entry_ref"]==_ref(SOURCE)
        and j["module"]=="native_ARGformer_VAL000_mlm" and j["module_ref"]==ORIGINAL_REFS["entry"]
        and j["job_id"]==NATIVE_JOB and spec["phase"]=="mlm"
        and j["args"]==["--execute","--input",j["input_ref"]["path"],"--expected-input-sha256",j["input_ref"]["sha256"]],
        "Exact native method job and literal scientific input arguments")
    require(spec["refs"]["storage"]==j["storage_ref"] and spec["caps_ref"]==j["cuda_caps_ref"]
        and spec["output_directory"]==storage["output_directory"]
        and spec["max_cuda_reserved_bytes"]==caps["max_cuda_reserved_bytes"]==j["cuda_cap_bytes"]==40*2**30
        and spec["max_cpu_rss_bytes"]==caps["max_cpu_rss_bytes"]==j["cpu_rss_cap_bytes"]
        and caps["operational_deadline_seconds"]==j["operational_deadline_seconds"]==259200
        and spec["operational_GPU0_runtime_ref"]==_ref(SOURCE),"Exact original native resource budget and actual consolidated runtime")
    require(os.getpid()==os.getpgrp()==os.getsid(0),"Pool method worker owns its dedicated session")
    guard_ref=dict(path='research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py',sha256='3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69')
    guard=_load_exact(guard_ref,"_fresh_native_GPU0_inherited_guard")
    guard.install_before_torch_v1();lease=guard.validate_lease();guard.apply_and_validate_title()
    require(lease["physical_gpu"]==0 and lease["visible_gpu_uuid"]==GPU0_UUID
        and lease["launcher_metadata"]["job"]==os.environ["ARGGNN_JOB_NAME"],"Canonical inherited GPU0 lease")
    os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(spec["completed_overlay_policy_ref"],separators=(",",":"))
    os.environ['ARGGNN_NATIVE_CAPACITY_GUARD_REF']=json.dumps(storage['budget_components']['consumer_adapter'],separators=(',',':'))
    _path_v1();driver=_load_exact(ORIGINAL_REFS["driver"],"_native_GPU0_successor_private_driver")
    patch_private_driver_v1(driver,capability_ref)
    packet,_=driver.c.phase_paths("mlm");admitted=driver.c.read(driver.c.ref(packet/"admitted.json"))
    require(admitted["input_ref"]==j["input_ref"] and driver.c.POLICY_REF==spec["completed_overlay_policy_ref"],
        "Exact actual fresh admission and reviewed new capacity policy")
    return driver.launch("mlm")

def _path_v1():
    stage=ORIGINAL_REFS["entry"]["path"].rsplit("/",1)[0]
    sys.path[:0]=[str(ROOT/stage),str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),
        str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12"),
        str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT)]

def _launcher_v1(args):
    _path_v1();_install_monitor_v1();module=_load_exact(ORIGINAL_REFS["launcher"],"_native_GPU0_monitor_private_launcher")
    _replace_function_v1(module,ORIGINAL_REFS["launcher"],"main",
        'command = [str(ROOT / ".runtime/env/bin/python"), "-B", str(ROOT/args.module), *args.module_args]',
        'command = [str(ROOT / ".runtime/env/bin/python"), "-B", str(ROOT/_NATIVE_GPU0_MONITOR_ROUTE_SOURCE),"--route-role","supervisor","--", *args.module_args]')
    previous=sys.argv
    try:
        sys.argv=[str(ROOT/ORIGINAL_REFS["launcher"]["path"]),*args];return module.main()
    finally:sys.argv=previous

def _supervisor_v1(args):
    _path_v1();_install_monitor_v1();module=_load_exact(ORIGINAL_REFS["supervisor"],"supervisor_v1")
    require(module.SOURCE==ORIGINAL_REFS["entry"]["path"] and module.CHILD==ORIGINAL_REFS["child"]["path"],"Frozen entry/child declarations")
    _replace_function_v1(module,ORIGINAL_REFS["supervisor"],"_run_child",'str(ROOT/CHILD),"--child-input"',
        'str(ROOT/_NATIVE_GPU0_MONITOR_ROUTE_SOURCE),"--route-role","child","--","--child-input"')
    return _load_exact(ORIGINAL_REFS["entry"],"_native_GPU0_monitor_original_entry").main(args)

def _child_v1(args):
    _path_v1();_install_monitor_v1()
    ipc=_load_exact(IPC_ADAPTER_REF,"_native_GPU0_exact_private_IPC_adapter_v3")
    ipc.install_private_module_before_native_child_v3()
    return _load_exact(ORIGINAL_REFS["child"],"_native_GPU0_monitor_original_child").main(args)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--route-role",choices=("launcher","supervisor","child"))
    parser.add_argument("args",nargs=argparse.REMAINDER);args=parser.parse_args(argv)
    if args.route_role is None:
        print(json.dumps(dict(status="source_only_native_GPU0_monitor_route",source_ref=_ref(SOURCE),execution=False)));return 0
    _authority_v1();values=args.args[1:] if args.args[:1]==["--"] else args.args
    return {"launcher":_launcher_v1,"supervisor":_supervisor_v1,"child":_child_v1}[args.route_role](values)

if __name__=="__main__":raise SystemExit(main())

# Source-only canonical GPU0 monitor and exact native subprocess route.
from pathlib import Path
import argparse,ctypes,hashlib,json,os,platform
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE='research/reviews/native_VAL000_GPU0_monitor_operational_source_20261016_v1/native_gpu0_monitor_route_v1.py'
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
ORIGINAL_REFS={'monitor': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/gpu_monitor_v8.py', 'sha256': 'f3e0028e072204790957a2ea2327b0e3af28ef245a8570fd2a0eb1c3ade0f9d4'}, 'entry': {'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/entry_mlm980_v1.py', 'sha256': '87f3eb67cfd5cb667c03742abaff57821297676952b99fcd76e5c2c39a85be5b'}, 'child': {'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/native_child_v1.py', 'sha256': '67a0c13c9a78f7aea25631456c2ed66985f62a60e81f5b9e0534af8b9d043ac1'}, 'supervisor': {'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/supervisor_v1.py', 'sha256': '170f0a04b8b52c0c67af9ed1b8b182bdfce1ff56b4dc846b86aafc8f2de3ea33'}, 'launcher': {'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/launcher_mlm980_v1.py', 'sha256': '94db776b80e5edbce675920760d4fb63552a2e236e4a9e5ed19d4ac56b8dfcfc'}, 'driver': {'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/root_native_driver_v1.py', 'sha256': '740d2545f4e71af8be3390dd9c3bb9024a59a8b21e08d38ecad8902c381e9e28'}, 'context': {'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/context_factory_v1.py', 'sha256': '5933615a1915bc2dbe8365f7ff1071490f7dabb92a48daf5283d3ff082d7dfc3'}, 'runtime': {'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/shared_native_runtime_entry_v1.py', 'sha256': '51d888173f720f6e37298002d16252d71d8f5f64b301351fbeb61095c5bd5286'}}
ORIGINAL_INPUT_REF={'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/input.json', 'sha256': '52242fe04f8f944dccbc81bf12940ad311d0d80f300e3cdd099be5e1748bc5d2'}
ORIGINAL_STORAGE_REF={'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/storage.json', 'sha256': '49d4097f8ceb367f1b8bd7a99649b0ae90db1045515953a0059b2e42bbf3eb81'}
ORIGINAL_CAPS_REF={'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/caps_mlm.root_active_v1.json', 'sha256': '2f9763d36a58e45f6d0f06b78c2bead18041e2fc88aaff084845414ae99d1b7e'}
FAILURE_WAIT_REF={'path': 'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/actual_wait.json', 'sha256': '87988a8996409579340809c5091c242f6a6f154c22673e9b000cf6eca4b459af'}
NATIVE_JOB="American-C3-native-val000-MLM-prediction-route-sharedGPU0-mlm-v1"

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
    require(isinstance(record,dict),"Actual future root capability reference required; otherwise source-only")
    cap=_read(record);spec=_read(ORIGINAL_INPUT_REF);storage=_read(ORIGINAL_STORAGE_REF);now=datetime.now(timezone.utc).timestamp()
    require(cap.get("schema")=="GPU0_declared_logical_pool_capability_v1" and cap.get("status")=="active"
        and cap.get("root_authorizer")=="root_agent" and type(cap.get("physical_gpu")) is int and cap["physical_gpu"]==0
        and cap.get("gpu_uuid")==GPU0_UUID and cap.get("maximum_distinct_project_GPUs")==1
        and now<cap["expires_unix"] and now<datetime.fromisoformat(spec["expires_at_utc"]).timestamp(),
        "Unexpired existing GPU0 root capability and unchanged original native expiry")
    require(all(row in cap.get("source_refs",{}).values() for row in [_ref(SOURCE),*ORIGINAL_REFS.values()]),
        "Actual combined monitor/route and retained original source roles bound in root closure")
    envelope=_read(cap["jobs"][NATIVE_JOB]["input_ref"]);request=_read(envelope["request_ref"]);method=request["method_job"]
    require(method["input_ref"]==ORIGINAL_INPUT_REF and method["storage_ref"]==ORIGINAL_STORAGE_REF
        and method["cuda_caps_ref"]==ORIGINAL_CAPS_REF and method["job_id"]==NATIVE_JOB
        and spec["refs"]["storage"]==ORIGINAL_STORAGE_REF and spec["caps_ref"]==ORIGINAL_CAPS_REF
        and storage["reserved_new_bytes"]==7*2**30 and storage["job_id"]==NATIVE_JOB,
        "Only original native MLM input/caps/7GiB charge")
    require(Path.cwd().resolve()==ROOT and os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==NATIVE_JOB
        and os.environ.get("ARGGNN_JOB_NAME","").startswith("gpu0_shared_v1:"),"Exact existing native logical lease lineage")
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

def wrap_original_runtime_v1(runtime,capability_ref):
    _authority_v1(capability_ref)
    require(Path(runtime.__file__).resolve()==ROOT/ORIGINAL_REFS["runtime"]["path"]
        and _ref(ORIGINAL_REFS["runtime"]["path"])==ORIGINAL_REFS["runtime"],"Exact original native pool runtime")
    require(not getattr(runtime,"_gpu0_monitor_loader_installed_v1",False),"No duplicate private runtime loader")
    original_load=runtime.load
    def load(record,label):
        module=original_load(record,label)
        return patch_private_driver_v1(module,capability_ref) if record==ORIGINAL_REFS["driver"] else module
    runtime.load=load;runtime._gpu0_monitor_loader_installed_v1=True
    return runtime

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
    return _load_exact(ORIGINAL_REFS["child"],"_native_GPU0_monitor_original_child").main(args)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--route-role",choices=("launcher","supervisor","child"))
    parser.add_argument("args",nargs=argparse.REMAINDER);args=parser.parse_args(argv)
    if args.route_role is None:
        print(json.dumps(dict(status="source_only_native_GPU0_monitor_route",source_ref=_ref(SOURCE),execution=False)));return 0
    _authority_v1();values=args.args[1:] if args.args[:1]==["--"] else args.args
    return {"launcher":_launcher_v1,"supervisor":_supervisor_v1,"child":_child_v1}[args.route_role](values)

if __name__=="__main__":raise SystemExit(main())

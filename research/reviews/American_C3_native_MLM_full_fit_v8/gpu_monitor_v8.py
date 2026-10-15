"""In-process readonly NVML monitor. CLI preflight is ROOT-only, default inert."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,platform
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/American_C3_native_MLM_full_fit_v8/gpu_monitor_v8.py"
PROVENANCE="research/reviews/American_C3_native_MLM_full_fit_v8/nvml_abi_provenance.json"
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
        require(physical_gpu in (4,5),"Eligible physical metadata-query device only")
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
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root",action="store_true",required=True)
    parser.add_argument("--physical-gpu",type=int,choices=(4,5),required=True)
    parser.add_argument("--gpu-uuid",required=True)
    parser.add_argument("--own-pid",type=int,action="append",default=[])
    args=parser.parse_args(argv)
    # Exactly one SMI snapshot, solely in this explicit ROOT-only preflight.
    import subprocess,xml.etree.ElementTree as ET
    monitor=OwnedGpuMemory(args.gpu_uuid,args.physical_gpu)
    nvml={r["pid"]:r["used_memory_bytes"] for r in monitor.query() if r["pid"] in args.own_pid}
    result=subprocess.run(["nvidia-smi","--id="+str(args.physical_gpu),"-q","-x"],
        check=True,capture_output=True,text=True,timeout=15)
    require(len(result.stdout.encode())<=1048576,"Bounded one SMI XML snapshot")
    gpu=ET.fromstring(result.stdout).find("gpu")
    require(gpu is not None and gpu.findtext("uuid")==args.gpu_uuid,"One SMI physical UUID equals NVML UUID")
    smi={}
    for proc in gpu.findall("./processes/process_info"):
        pid=int(proc.findtext("pid"))
        if pid not in args.own_pid:continue
        raw=proc.findtext("used_memory","")
        require(raw.endswith(" MiB"),"Actual SMI process memory in MiB")
        smi[pid]=int(raw[:-4])*1048576
    details=[dict(pid=pid,nvml_bytes=nvml.get(pid),smi_bytes=smi.get(pid),
        difference_bytes=(nvml[pid]-smi[pid]) if pid in nvml and pid in smi else None) for pid in sorted(set(nvml)|set(smi))]
    print(json.dumps(dict(schema="native_MLM_readonly_NVML_SMI_probe_v8",
        status="completed_readonly_NVML_SMI_comparison",source_ref=reference(SOURCE),
        abi_provenance_ref=reference(PROVENANCE),physical_gpu=args.physical_gpu,GPU_UUID=args.gpu_uuid,
        own_process_memory=details,non_owned_process_details_recorded=False,
        CUDA_context_created=False,GPU_allocations_or_training=False,accounting_or_reservation=False),indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())

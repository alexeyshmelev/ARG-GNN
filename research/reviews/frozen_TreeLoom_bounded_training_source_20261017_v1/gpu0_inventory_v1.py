"""In-process read-only physicalGPU0 inventory; no CUDA or subprocess calls.
Uses the previously source-reviewed Linux x86_64 NVML process-v3 ABI.
"""
from pathlib import Path
import ctypes,os,platform
UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
MIB=2**20
class ProcessInfo(ctypes.Structure):
    _fields_=[("pid",ctypes.c_uint),("usedGpuMemory",ctypes.c_ulonglong),
              ("gpuInstanceId",ctypes.c_uint),("computeInstanceId",ctypes.c_uint)]
class MemoryInfo(ctypes.Structure):
    _fields_=[("total",ctypes.c_ulonglong),("free",ctypes.c_ulonglong),("used",ctypes.c_ulonglong)]
class Utilization(ctypes.Structure):
    _fields_=[("gpu",ctypes.c_uint),("memory",ctypes.c_uint)]
def need(ok,label):
    if not ok:raise ValueError(label)
class GPU0InventoryV1:
    def __init__(self):
        need(platform.system()=="Linux"and platform.machine()=="x86_64"and ctypes.sizeof(ProcessInfo)==24
             and ctypes.sizeof(MemoryInfo)==24,"Reviewed Linuxx86_64 NVML ABI")
        self.pid=os.getpid();self.lib=ctypes.CDLL("libnvidia-ml.so.1")
        def bind(name,args):
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=ctypes.c_int;return fn
        self.init=bind("nvmlInit_v2",[])
        self.by_uuid=bind("nvmlDeviceGetHandleByUUID",[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)])
        self.uuid=bind("nvmlDeviceGetUUID",[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint])
        self.index=bind("nvmlDeviceGetIndex",[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)])
        self.processes=bind("nvmlDeviceGetComputeRunningProcesses_v3",[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ProcessInfo)])
        self.memory=bind("nvmlDeviceGetMemoryInfo",[ctypes.c_void_p,ctypes.POINTER(MemoryInfo)])
        self.util=bind("nvmlDeviceGetUtilizationRates",[ctypes.c_void_p,ctypes.POINTER(Utilization)])
        self.name=bind("nvmlSystemGetProcessName",[ctypes.c_uint,ctypes.c_char_p,ctypes.c_uint])
        self.ok(self.init(),"init");self.handle=ctypes.c_void_p()
        self.ok(self.by_uuid(UUID.encode(),ctypes.byref(self.handle)),"exactGPU0UUIDhandle")
    @staticmethod
    def ok(code,label):need(code==0,"Fresh NVML "+label+" status="+str(code))
    def query(self):
        need(os.getpid()==self.pid,"Never reuse a native handle after fork")
        text=ctypes.create_string_buffer(96);index=ctypes.c_uint()
        self.ok(self.uuid(self.handle,text,len(text)),"UUID");self.ok(self.index(self.handle,ctypes.byref(index)),"index")
        need(text.value.decode()==UUID and index.value==0,"Only held exact physicalGPU0UUID")
        memory=MemoryInfo();util=Utilization()
        self.ok(self.memory(self.handle,ctypes.byref(memory)),"memory");self.ok(self.util(self.handle,ctypes.byref(util)),"utilization")
        need(0<memory.total<2**64-1 and memory.used<=memory.total and memory.free<=memory.total
             and util.gpu<=100,"Actual bounded GPU0 memory/utilization fields")
        result=None
        for _ in range(4):
            count=ctypes.c_uint(0);code=self.processes(self.handle,ctypes.byref(count),None)
            if code==0 and count.value==0:result=[];break
            need(code in(0,7)and count.value<=4096,"Bounded fresh NVML compute-process count")
            capacity=max(8,count.value*2+5);array=(ProcessInfo*capacity)();count=ctypes.c_uint(capacity)
            code=self.processes(self.handle,ctypes.byref(count),array)
            if code==7:continue
            self.ok(code,"processes");need(count.value<=capacity,"NVML output within allocated ABI array")
            result=[]
            for at in range(count.value):
                pid=int(array[at].pid);used=int(array[at].usedGpuMemory)
                need(pid>=2 and 0<=used<2**64-1,"Actual available process memory; no unavailable sentinel")
                name=ctypes.create_string_buffer(4096);status=self.name(pid,name,len(name))
                shown=name.value.decode("utf-8",errors="replace")if status==0 else"<NVML process name unavailable status=%d>"%status
                result.append(dict(pid=pid,name=shown,used_mib=(used+MIB-1)//MIB))
            break
        need(result is not None,"No stale sample after four NVML process-count races")
        need(len({r["pid"]for r in result})==len(result),"One current compute record per PID")
        return [dict(index=0,uuid=UUID,used_mib=(int(memory.used)+MIB-1)//MIB,
            total_mib=int(memory.total)//MIB,utilization=int(util.gpu),compute=result,
            inventory_source="in_process_readonly_NVML_v3_no_helpers")]
_INSTANCE=None
def inventory_v1():
    global _INSTANCE
    if _INSTANCE is None:_INSTANCE=GPU0InventoryV1()
    return _INSTANCE.query()

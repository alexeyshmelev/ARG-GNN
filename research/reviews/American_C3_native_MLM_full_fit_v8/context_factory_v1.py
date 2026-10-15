"""Concrete existing-guard context for original C3 full TRAIN engineering.

No import-time guard, runtime, process, hardware or data actions. Consumer memo2
is installed only by parent/CPU factories. Native children never load the
storage adapter or construct a writer while their scope is active.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,resource,stat,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_full_fit_v8"
def contract():
    path=ROOT/REL/"root_native_contract_v1.py"
    ms=importlib.util.spec_from_file_location("_C3_native_factory_contract",path)
    c=importlib.util.module_from_spec(ms);ms.loader.exec_module(c);return c
def _proc_reader(c,spec):
    path=c.local(c.PROC_HELPER);c.require(c.ref(path)["sha256"]==spec["source_sha256"][c.PROC_HELPER],"Exact existing owned-process reader")
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="owned_group_RSS_v1")
    ns=dict(Path=Path,os=os,c=c)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+":unchanged_owned_RSS_reader","exec"),ns)
    return ns["owned_group_RSS_v1"]
def _validate(c,record,spec):
    c.active(spec);req,caps,runtime=c.validate_request(spec["root_stage_request_ref"])
    packet,output=c.phase_paths(spec["phase"]);admitted=c.read(c.ref(packet/"admitted.json"))
    c.require(admitted["input_ref"]==record and admitted["driver_ref"]==c.ref(c.DRIVER)
        and spec["root_context_module_ref"]==c.ref(c.CONTEXT)
        and spec["parent_id"]==c.PARENT and spec["output_directory"]==output
        and spec["caps_ref"]==req["caps_ref"] and spec["upstream_wait_refs"]==req["upstream_wait_refs"]
        and c.read(spec["refs"]["config"])==c.make_config(req,caps)
        and all(spec["refs"].get(k)==v for k,v in req["refs"].items()),
        "Exact root specific request/active input/caps/output/upstream metadata")
    c.require(json.loads(os.environ.get("ARGGNN_COMPLETED_CREDITS_POLICY_REF","null"))==c.POLICY_REF,
        "Exact current policy inherited by all consumers")
    for path,sha in spec["source_sha256"].items():c.require(c.ref(path)["sha256"]==sha,"Actual complete source closure")
    return req,caps,runtime
def _lease(c,spec,runtime,CPU):
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    if CPU:
        c.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Explicit CPU context hides CUDA");return None
    lease=validate_lease();apply_and_validate_title()
    c.require(lease["physical_gpu"]==spec["physical_gpu"]==runtime["target_physical_gpu"]
        and lease["physical_gpu"] in (4,5) and lease["visible_gpu_uuid"]==runtime["target_GPU_UUID"]
        and (os.environ.get("ARGGNN_LOGICAL_JOB_NAME") or os.environ.get("ARGGNN_JOB_NAME"))==spec["job_id"],
        "Actual original held GPU45 UUID/logical job/runtime")
    return lease
def _fast_guard(c,spec,caps,lease):
    from gpu_monitor_v8 import OwnedGpuMemory
    gpu=OwnedGpuMemory(lease["visible_gpu_uuid"],lease["physical_gpu"]) if lease is not None else None
    reader=_proc_reader(c,spec);known={};started=time.monotonic()
    def check(event,**owned):
        from arggnn.census_tsinfer_fit_v1 import require_logical_job_v1
        c.active(spec)
        c.require(time.monotonic()-started<=caps["operational_deadline_seconds"],"Actual stage operational deadline")
        if lease is not None:
            from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
            c.require(validate_lease()==lease,"Inherited lease unchanged");apply_and_validate_title()
            require_logical_job_v1(lease,spec)
        pid=owned.get("child_pid",os.getpid());roster=known.setdefault(pid,{pid:None})
        rss=reader(pid,roster)
        if pid!=os.getpid():rss+=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        c.require(rss<=caps["max_cpu_rss_bytes"],"Actual owned parent/native-descendant CPU RSS bound")
        if lease is not None:
            used=gpu.owned_bytes(roster)
            c.require(used<=caps["max_cuda_reserved_bytes"],"Fresh owned GPU-process memory cap")
        return dict(event=event,owned_RSS_bytes=rss,no_storage_journal_or_scope_scan=True)
    return check
class _PrimitiveJSONWriter:
    """Existing v9 writer with explicit primitive-JSON serialization adaptation.

    Frozen v9 JSON serialization requires uninitialized census np/torch globals.
    Keep that numerical module untouched and use its identical guarded file I/O
    for primitive JSON. Torch/NPZ serialization delegates to the original writer.
    """
    def __init__(self,writer,guard):self.writer=writer;self.guard=guard;self.path=writer.path
    def save(self,name,value,kind="json"):
        if kind!="json":return self.writer.save(name,value,kind=kind)
        m=self.guard;writer=self.writer
        m.require(isinstance(name,str) and Path(name).name==name and name not in ("",".",".."),"Flat exclusive output name")
        def primitive(v):
            if v is None or type(v) in (str,int,float,bool):return
            if isinstance(v,(list,tuple)):
                for x in v:primitive(x)
                return
            if isinstance(v,dict) and all(type(k) is str for k in v):
                for x in v.values():primitive(x)
                return
            raise ValueError("Native supervisor JSON must contain primitive metadata only")
        primitive(value);data=m.budget.v4.encoded(value)+b"\n"
        planned=len(data)+8192
        m.require(writer.used+planned<=writer.storage["reserved_new_bytes"],"Owned writer counter exceeds cap")
        m.budget_check_v9(writer.storage,boundary=True,required_write_bytes=planned)
        m.require(planned<=os.statvfs(writer.path).f_bavail*os.statvfs(writer.path).f_frsize,"Actual filesystem free bytes required")
        path=writer.path/name;fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno());identity=os.fstat(stream.fileno())
        named=path.lstat()
        m.require(stat.S_ISREG(named.st_mode) and named.st_nlink==1
            and (identity.st_dev,identity.st_ino)==(named.st_dev,named.st_ino)
            and named.st_size==len(data),"Created output pathname changed during save")
        writer.used+=planned
        return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
class _SealCompletionEnvelope:
    """Explicit root completion status, preserving the full original seal receipt."""
    def __init__(self,writer):self.writer=writer;self.path=writer.path
    def save(self,name,value,kind="json"):
        if name=="completed.json" and kind=="json":
            if value.get("status")!="sealed_native_C3_top1":raise ValueError("Original native seal receipt required")
            value=dict(value,status="completed_C3_native_original_top1_seal",
                original_producer_completion=dict(value),original_producer_status=value["status"])
        return self.writer.save(name,value,kind=kind)
def _parent_context(record,spec,CPU):
    c=contract();req,caps,runtime=_validate(c,record,spec);lease=_lease(c,spec,runtime,CPU)
    from arggnn.census_tsinfer_fit_v1 import ConsumerReaderV1
    from arggnn.american_c3_common_v1 import runtime_report
    expected=c.read(runtime["shared_cpu_runtime_ref"])
    expected=dict(expected,CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    c.require(runtime_report()==expected,"Actual qualified shared supervisor/CPU runtime and separately leased visibility")
    guard,memo=c.consumer_guard();storage=c.read(spec["refs"]["storage"]);cfg=c.read(spec["refs"]["config"])
    c.require(storage["completed_overlay_policy_ref"]==c.POLICY_REF
        and storage["reserved_new_bytes"]==caps["reservation_cap_bytes"],"Actual same-stage current policy/cap")
    guard.validate_storage_v9(spec,dict(output_cap_bytes=caps["reservation_cap_bytes"]),storage,
        source_ref=spec["main_source_ref"],config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
    writer=_PrimitiveJSONWriter(guard.PilotOutputV9(spec["output_directory"],storage,ConsumerReaderV1()),guard)
    if spec["phase"]=="prediction_seal":writer=_SealCompletionEnvelope(writer)
    def boundary(event,required_write_bytes=0):
        c.active(spec)
        return guard.budget_check_v9(storage,boundary=True,required_write_bytes=required_write_bytes)
    return dict(root_authorizer="root_agent",admitted=True,authority_validated_by_existing_guard=True,
        upstream_inputs_complete_and_drained=True,fast_guard_scans_or_writes_scope=False,
        fast_guard=_fast_guard(c,spec,caps,lease),boundary_guard=boundary,writer=writer,
        consumer_memo2=memo,device="cpu" if CPU else "cuda")
def build_context_v1(record,spec):return _parent_context(record,spec,False)
def build_cpu_context_v1(record,spec):return _parent_context(record,spec,True)
def _actual_context_uuid(c,lease,torch):
    """Reuse the current-context driver method in existing biological_candidate.

    Torch2.4 lacks a UUID property. This I/O-only adaptation retains evidence in
    the returned runtime receipt rather than writing a qualification directory.
    """
    import ctypes,uuid
    torch.cuda.set_device(0)
    keep_context=torch.empty(1,device=torch.device("cuda",0));torch.cuda.synchronize()
    c.require(torch.cuda.current_device()==0,"Exact single logical CUDA0 context")
    drivers=set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)==6 and Path(parts[5]).name.startswith("libcuda.so."):
            path=Path(parts[5])
            c.require(path.is_absolute() and path.is_file(),"Actual own loaded CUDA driver file")
            drivers.add(path.resolve())
    c.require(len(drivers)==1,"Exactly one own loaded CUDA driver")
    driver_path=drivers.pop();driver=ctypes.CDLL(str(driver_path));device=ctypes.c_int(-1)
    lookup=driver.cuCtxGetDevice;lookup.argtypes=[ctypes.POINTER(ctypes.c_int)];lookup.restype=ctypes.c_int
    c.require(lookup(ctypes.byref(device))==0,"Actual driver context device lookup")
    class CudaUUID(ctypes.Structure):
        _fields_=[("bytes",ctypes.c_ubyte*16)]
    identity=CudaUUID();name="cuDeviceGetUuid_v2" if hasattr(driver,"cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    lookup=getattr(driver,name);lookup.argtypes=[ctypes.POINTER(CudaUUID),ctypes.c_int];lookup.restype=ctypes.c_int
    c.require(lookup(ctypes.byref(identity),device.value)==0,"Actual driver context UUID lookup")
    actual=str(uuid.UUID(bytes=bytes(identity.bytes)))
    c.require(actual==lease["canonical_uuid"],"Actual CUDA context UUID equals held original GPU45 lease")
    del keep_context
    return dict(actual_context_uuid=actual,driver_library=str(driver_path),uuid_function=name,
        context_device=device.value,method_source_ref=c.ref("src/arggnn/biological_candidate.py"))
def build_native_child_context_v1(record,spec):
    c=contract();req,caps,runtime=_validate(c,record,spec);lease=_lease(c,spec,runtime,False)
    fast=_fast_guard(c,spec,caps,lease);verified={}
    def validate_runtime(runtime_ref,value):
        ms=importlib.util.spec_from_file_location("_C3_actual_native_runtime",c.local(c.RUNTIME_HELPER))
        helper=importlib.util.module_from_spec(ms);ms.loader.exec_module(helper)
        receipt,after=helper.validate_native_runtime_v1(runtime_ref,value);verified.update(after=after,receipt=receipt)
        return receipt
    def configure_device(value,actual_lease):
        c.require(actual_lease==lease and "after" in verified,"Native runtime validated before device use")
        fast("before_actual_native_device")
        import torch
        c.require(torch.cuda.device_count()==1,"One exact visible leased device")
        device=torch.device("cuda",0);props=torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(min(1.,caps["max_cuda_reserved_bytes"]/props.total_memory),device)
        identity=_actual_context_uuid(c,lease,torch)
        verified["after"]()
        verified["receipt"].update(actual_device_identity=identity,
            actual_total_memory_bytes=props.total_memory,max_cuda_reserved_bytes=caps["max_cuda_reserved_bytes"])
        fast("after_actual_native_runtime_imports")
    # Deliberately no storage guard import/install or writer construction here.
    return dict(root_authorizer="root_agent",admitted=True,authority_validated_by_existing_guard=True,
        upstream_inputs_complete_and_drained=True,fast_guard_scans_or_writes_scope=False,
        fast_guard=fast,validate_native_runtime=validate_runtime,configure_native_device=configure_device,
        verify_runtime_imports=lambda:verified["after"](),actual_device_receipt=lambda:verified["receipt"])

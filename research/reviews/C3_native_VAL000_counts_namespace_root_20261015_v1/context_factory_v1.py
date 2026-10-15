"""Counts-only existing consumer context; current980 and original primitive writer."""
from pathlib import Path
import hashlib,importlib.util,json,os,resource,stat,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL='research/reviews/C3_native_VAL000_counts_namespace_root_20261015_v1'
def contract():
    path=ROOT/REL/"root_native_contract_v1.py"
    ms=importlib.util.spec_from_file_location("_C3_native_factory_contract",path)
    c=importlib.util.module_from_spec(ms);ms.loader.exec_module(c);return c

def _validate(c,record,spec):
    c.active(spec);req,caps,runtime=c.validate_request(spec["root_stage_request_ref"])
    packet,output=c.phase_paths(spec["phase"]);admitted=c.read(c.ref(packet/"admitted.json"))
    c.require(admitted["input_ref"]==record and admitted["driver_ref"]==c.ref(c.DRIVER)
        and spec["root_context_module_ref"]==c.ref(c.CONTEXT)
        and spec["parent_id"]==c.PARENT and spec["output_directory"]==output
        and spec["caps_ref"]==req["caps_ref"] and spec["upstream_wait_refs"]==req["upstream_wait_refs"]
        and spec["split"]=="val" and spec["original_heldout_VAL000_counts_only"] is True
        and spec["engineering_TRAIN_only"] is False
        and spec["consumer_memo_ref"]==c.MEMO_REF and spec["consumer_namespace_memo_ref"]==c.NAMESPACE_MEMO_REF
        and c.read(spec["refs"]["config"])==c.make_config(req,caps)
        and all(spec["refs"].get(k)==v for k,v in req["refs"].items()),
        "Exact root specific request/active input/caps/output/upstream metadata")
    c.require(json.loads(os.environ.get("ARGGNN_COMPLETED_CREDITS_POLICY_REF","null"))==c.POLICY_REF,
        "Exact current policy inherited by all consumers")
    for path,sha in spec["source_sha256"].items():c.require(c.ref(path)["sha256"]==sha,"Actual complete source closure")
    return req,caps,runtime

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
    def boundary(event,required_write_bytes=0):
        c.active(spec)
        return guard.budget_check_v9(storage,boundary=True,required_write_bytes=required_write_bytes)
    return dict(root_authorizer="root_agent",admitted=True,authority_validated_by_existing_guard=True,
        upstream_inputs_complete_and_drained=True,fast_guard_scans_or_writes_scope=False,
        fast_guard=_fast_guard(c,spec,caps,lease),boundary_guard=boundary,writer=writer,
        consumer_memo2=memo,device="cpu" if CPU else "cuda")

def build_cpu_context_v1(record,spec):return _parent_context(record,spec,True)


def _lease(c,spec,runtime,CPU):
    c.require(CPU and spec["phase"]=="counts" and spec["GPU_authorized"] is False
        and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Counts-only CPU context, no lease/CUDA")
    return None

def _fast_guard(c,spec,caps,lease):
    c.require(lease is None,"Counts has no GPU lease")
    started=time.monotonic()
    def check(event,**owned):
        c.active(spec)
        c.require(time.monotonic()-started<=caps["operational_deadline_seconds"],"Counts operational deadline")
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        c.require(rss<=caps["max_cpu_rss_bytes"],"Actual CPU count-process peak RSS bound")
        return dict(event=event,own_peak_RSS_bytes=rss,no_storage_journal_or_scope_scan=True)
    return check

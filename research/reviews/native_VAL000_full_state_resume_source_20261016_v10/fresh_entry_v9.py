"""Finite read-only known-owner entry probes, durable own receipts, exact late hook.

No import-time actions. No scientific/model/corpus import. The historical ROOT
metadata is authenticated elsewhere and is never relabelled currently fresh.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy,json,os,stat,sys,time,types
from research.reviews.native_VAL000_full_state_resume_source_20261016_v10 import amendment_contract_v1 as c

SOURCE=c.HERE+"/fresh_entry_v9.py"
MAX_ENTRY_BYTES=2*2**20
FRESH_SECONDS=120
MAX_PROBE_SECONDS=90
WORKER_EVENT="before_first_cuda_tensor"
ENTRY_REFS={}
ENTRY_NAMES={"supervisor":"entry_supervisor_known_owner.v9.json","worker":"entry_worker_known_owner.v9.json"}

def require_fresh_v9(value,*,now=None):
    c.require(value.get("schema")=="native_VAL000_actual_fresh_known_owner_entry_v9"
        and value.get("status")=="ACTUAL_ENTRY_PROBES_PASSED"
        and value.get("all_known_observations_complete") is True
        and value.get("historical_v3_descendant_coverage")=="UNKNOWN_TRUNCATED64"
        and value.get("historical_v3_unknown_descendants_absent") is None
        and value.get("unknown_historical_ownership_reconstructed") is False,
        "Only actual new complete known-scope entry probes, never old ROOT snapshots")
    now=datetime.now(timezone.utc) if now is None else now
    registry=value["known_registry"]
    rows=value["PID_observations"];groups=value["group_observations"]
    c.require(len(rows)==len(registry["identities"]) and len(groups)==len(registry["process_groups"]),
        "No known entry observation omitted or truncated")
    for expected,row in zip(registry["identities"],rows):
        c.require(row["pid"]==expected["pid"] and row["start_ticks"]==expected["start_ticks"]
            and row["known_identity_absent"] is True and type(row["numeric_PID_absent"]) is bool
            and (expected["start_ticks"] is not None or row["numeric_PID_absent"] is True)
            and row["method"]=="actual targeted PID/start or numeric ESRCH"
            and 0<=(now-datetime.fromisoformat(row["checked_utc"])).total_seconds()<=FRESH_SECONDS,
            "Genuine newly observed known PID/start absence within original120seconds")
    for expected,row in zip(registry["process_groups"],groups):
        c.require(row["pgid"]==expected and row["absent"] is True
            and row["method"]=="actual targeted os.killpg(pgid,0) ESRCH"
            and 0<=(now-datetime.fromisoformat(row["checked_utc"])).total_seconds()<=FRESH_SECONDS,
            "Genuine newly observed known group absence within original120seconds")
    return True

def probe_PID_v9(row,*,kill0=None,read_stat=None,now=None):
    # Exact finite target only. Zero is read-only, no signals are delivered.
    kill0=(lambda pid:os.kill(pid,0)) if kill0 is None else kill0
    def actual_stat(pid):
        p=Path("/proc")/str(pid)/"stat"
        with p.open("r") as stream:data=stream.read(8193)
        c.require(len(data)<=8192,"Bounded exact target proc stat")
        ending=data.rfind(")");fields=data[ending+2:].split()
        c.require(ending>0 and int(data[:data.find("(")].strip())==pid and len(fields)>=20,
            "Exact targeted proc identity metadata")
        return int(fields[19])
    read_stat=actual_stat if read_stat is None else read_stat
    now=(lambda:datetime.now(timezone.utc)) if now is None else now
    pid=row["pid"];start=row["start_ticks"]
    c.require(type(pid) is int and pid>=2 and (start is None or type(start) is int and start>0),
        "Exact known finite PID/start descriptor")
    observed_start=None
    try:kill0(pid)
    except ProcessLookupError:numeric_absent=True
    else:
        numeric_absent=False
        c.require(start is not None,"Unknown saved start requires actual numeric PID ESRCH")
        # If a numeric PID disappears/races during this one targeted read, fail
        # closed. No inferred identity, broad scan or automatic probe retry.
        observed_start=read_stat(pid)
        c.require(type(observed_start) is int and observed_start>0 and observed_start!=start,
            "Known saved PID/start identity still present or observation unresolved")
    return dict(pid=pid,start_ticks=start,known_identity_absent=True,numeric_PID_absent=numeric_absent,
        observed_different_start_ticks=observed_start,
        method="actual targeted PID/start or numeric ESRCH",checked_utc=now().isoformat())

def probe_group_v9(pgid,*,killpg0=None,now=None):
    killpg0=(lambda value:os.killpg(value,0)) if killpg0 is None else killpg0
    now=(lambda:datetime.now(timezone.utc)) if now is None else now
    c.require(type(pgid) is int and pgid>=2,"Exact finite known group")
    try:killpg0(pgid)
    except ProcessLookupError:pass
    else:raise ValueError("Known old process group still exists or is reused")
    return dict(pgid=pgid,absent=True,method="actual targeted os.killpg(pgid,0) ESRCH",checked_utc=now().isoformat())

def collect_v9(registry,*,stage,context,tick,pid_probe=probe_PID_v9,group_probe=probe_group_v9,
        wall_now=None,monotonic=None):
    wall_now=(lambda:datetime.now(timezone.utc)) if wall_now is None else wall_now
    monotonic=time.monotonic if monotonic is None else monotonic
    c.require(stage in ENTRY_NAMES and 1<=len(registry["identities"])<=4096
        and 1<=len(registry["process_groups"])<=4096,"Exact bounded entry kind and known scope")
    started=monotonic()
    value=dict(schema="native_VAL000_actual_fresh_known_owner_entry_v9",status="ENTRY_PROBES_INCOMPLETE",
        entry_stage=stage,context=context,known_registry=copy.deepcopy(registry),
        PID_observations=[],group_observations=[],created_utc=wall_now().isoformat(),
        historical_v3_descendant_coverage="UNKNOWN_TRUNCATED64",historical_v3_unknown_descendants_absent=None,
        unknown_historical_ownership_reconstructed=False,entry_only_not_continuous_freshness_claim=True,
        previous_ROOT_rows_rewritten_or_relabelled_fresh=False,all_known_observations_complete=False,
        max_receipt_bytes=MAX_ENTRY_BYTES,original_fresh_seconds=FRESH_SECONDS)
    def bounded_tick():
        tick();c.require(monotonic()-started<MAX_PROBE_SECONDS,"Finite90second exact entry probe bound")
    try:
        for row in registry["identities"]:
            bounded_tick();value["PID_observations"].append(pid_probe(row))
        for group in registry["process_groups"]:
            bounded_tick();value["group_observations"].append(group_probe(group))
        bounded_tick()
        value.update(status="ACTUAL_ENTRY_PROBES_PASSED",all_known_observations_complete=True,
            completed_utc=wall_now().isoformat(),elapsed_probe_seconds=monotonic()-started)
        require_fresh_v9(value,now=wall_now())
        c.require(len((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())<=MAX_ENTRY_BYTES,
            "Full exact finite entry receipt fits bounded own AUX")
        return value
    except BaseException as exc:
        value.update(status="ENTRY_PROBES_FAILED",all_known_observations_complete=False,
            error=type(exc).__name__+": "+str(exc)[:2000],
            completed_utc=wall_now().isoformat(),no_CUDA_model_or_training_entry_authorized=True)
        raise EntryProbeFailure(value) from exc

class EntryProbeFailure(RuntimeError):
    def __init__(self,value):
        self.value=value
        super().__init__(value["error"])

def capture_entry_v9(spec,q,g,storage,*,stage,tick):
    c.require(stage in ENTRY_NAMES and stage not in ENTRY_REFS,"One exact immutable entry evidence receipt per stage, no retry")
    c.active_amendment_v1()
    registry=c.entry_known_registry_v9()
    directory=c.local(spec["worker_auxiliary_scope"]);name=ENTRY_NAMES[stage]
    c.require(directory.is_dir() and not (directory/name).exists(),"Fresh exact charged own entry metadata name")
    # All potentially long source/storage checks precede the actual observations.
    c.charged_census_v1(spec,required_write_bytes=MAX_ENTRY_BYTES+8192)
    g.budget_check_v9(storage,boundary=True,required_write_bytes=MAX_ENTRY_BYTES+8192)
    c.active_amendment_v1();tick()
    context=dict(actual_execution_amendment_ref=c.current_amendment_ref_v1(),adapter_manifest_ref=c.ref(c.MANIFEST),
        source_ref=c.ref(SOURCE),historical_ROOT_known_owner_bundle_ref=c.V8_KNOWN_BUNDLE_REF,
        v8_genuine_failed_terminal_ref=c.V8_TERMINAL_REF,
        current_entry_PID=os.getpid(),before_any_CUDA_model_or_training=True,
        long_source_storage_setup_complete_before_actual_entry_probes=True)
    value=None;failure=None
    try:value=collect_v9(registry,stage=stage,context=context,tick=tick)
    except EntryProbeFailure as exc:value=exc.value;failure=exc
    # A checked complete write envelope exists before the probes. Do not run a
    # long guard between fresh observations and their immutable durable receipt.
    tick()
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    c.require(len(data)<=MAX_ENTRY_BYTES,"Bounded own actual-or-failed entry metadata")
    record=None
    try:
        record=c.save(directory,name,value)
        dfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    except BaseException as exc:
        print(json.dumps(dict(native_v9_entry_receipt_failed=True,entry_stage=stage,actual_entry_receipt_ref=None,
            error=type(exc).__name__+": "+str(exc)[:2000],actual_or_unknown_probe_status=value["status"],
            no_CUDA_model_or_training_entry_authorized=True)),flush=True)
        raise
    if failure is not None:
        print(json.dumps(dict(native_v9_entry_probe_failure_receipt_ref=record,
            no_CUDA_model_or_training_entry_authorized=True)),flush=True)
        raise failure
    require_fresh_v9(value)
    ENTRY_REFS[stage]=record
    require_entry_ref_v9(record,spec,stage=stage,tick=tick)
    return record

def require_entry_ref_v9(record,spec,*,stage,tick):
    tick()
    c.require(ENTRY_REFS.get(stage)==record and record["path"]==spec["worker_auxiliary_scope"]+"/"+ENTRY_NAMES[stage],
        "Actual newly minted own entry ref only, no old snapshot substitute")
    value=c.read(record)
    c.require(value["entry_stage"]==stage and value["context"]["actual_execution_amendment_ref"]==c.current_amendment_ref_v1()
        and value["context"]["adapter_manifest_ref"]==c.ref(c.MANIFEST)
        and value["known_registry"]==c.entry_known_registry_v9(),
        "Exact current source/amendment and full known registry for actual entry")
    require_fresh_v9(value);tick()
    return value

class ExactNativeEntryGuardV9:
    def __init__(self,native_guard,native_run,*,capture,verify):
        self.native_guard=native_guard;self.capture=capture;self.verify=verify;self.worker_entry_ref=None
        candidates=[v for v in native_run.__code__.co_consts if isinstance(v,types.CodeType) and v.co_name=="fast_guard"]
        c.require(len(candidates)==1,"One exact unchanged native nested fast_guard code object")
        self.expected_fast_code=candidates[0]
    def __getattr__(self,name):return getattr(self.native_guard,name)
    def validate_lease(self,*args,**kwargs):
        lease=self.native_guard.validate_lease(*args,**kwargs)
        frame=sys._getframe(1)
        event=frame.f_locals.get("event") if frame.f_code.co_name=="fast_guard" else None
        if event==WORKER_EVENT:
            c.require(frame.f_code is self.expected_fast_code and self.worker_entry_ref is None,
                "Only exact original nested before-first-CUDA event, no replay")
            self.worker_entry_ref=self.capture()
            self.verify(self.worker_entry_ref)
            again=self.native_guard.validate_lease(*args,**kwargs)
            c.require(again==lease,"Unchanged real lease after actual entry probes/receipt")
            self.verify(self.worker_entry_ref)
            return again
        return lease

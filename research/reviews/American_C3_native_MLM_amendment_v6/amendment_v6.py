"""One never-spawned allocation: execution authority only; no accounting mutations.

Default imports are standard-library only. Authority, absence and lifetime
checks are explicit. Immutable old code does not consult this authority.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,importlib.util,json,os,stat,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_amendment_v6"
SOURCE=REL+"/amendment_v6.py"
ENTRY=REL+"/entry_v6.py"
LAUNCHER=REL+"/launcher_v6.py"
DRIVER=REL+"/root_native_driver_v1.py"
KEY="H041/American-C3-native-MLM-full30mb-qualify_mlm-v2"
JOB="American-C3-native-MLM-full30mb-qualify_mlm-v2"
OUTPUT="experiments/results/American_C3_native_MLM_full30mb_qualify_mlm_v4"
AUX=ROOT/"research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/worker_aux"
TMP=ROOT/".runtime/tmp/c3mlm4/q"
ATTEMPT=AUX/"amended_v6_attempt1"
LOCK=AUX/"amendment_v6.activation"
GPU_UUID="GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71"
CAP=15032385536
CREATED="2026-10-15T10:01:33.907094+00:00"
EXPIRES="2026-10-16T10:01:33.907094+00:00"
ENV_REF="ARGGNN_MLM_AMENDMENT_V6_REF"
ENV_FD="ARGGNN_MLM_AMENDMENT_V6_FD"
FIXED_REFS={'ORIGINAL_STORAGE_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/storage.json', 'sha256': '87caac31ff4beb5b6d8c327e4a120a4901b8b2b33206c52aaa20398d26bf2963'}, 'ORIGINAL_INPUT_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/input.json', 'sha256': 'a20fc6cd03cda560de3ff0f4f8ae326e40ee33581e8144c5c836dab4c0e98a14'}, 'ORIGINAL_ADMITTED_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/admitted.json', 'sha256': '58e58ca2a8f44671986ab909ab21cc17cb91b7795c8b21fd469c10c73c87ceca'}, 'ORIGINAL_CONFIG_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/config.json', 'sha256': 'efbfba8670f4758a906d696e1f442e6651fcc6a9930494d82a0a0d433121a876'}, 'ORIGINAL_RUNTIME_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/runtime_root_GPU4_v1.json', 'sha256': 'd9f4f45fb3fa5e1eb9878b5d00385c90cd4d80a3080e65f4438baaacf1a28263'}, 'ORIGINAL_CAPS_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/qualify_mlm_caps_root_v1.json', 'sha256': '84c4514dc98be5e1f2367632f2473ba0fb225aa00f6150dd82325dc54d8d8533'}, 'ORIGINAL_REQUEST_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/qualify_mlm_request_root_v1.json', 'sha256': '2b65c8adcedea79c7b1a663105f581e8458a4001aaa96083d3f6fb5d091104d3'}, 'ORIGINAL_FAILURE_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/pre_spawn_failure_root_v1.json', 'sha256': '23b084bec04059a803a4fd762cbead661f92e23f6d17016edad877ed6e01f80e'}, 'ORIGINAL_STDOUT_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/root_supervisor_stdout.log', 'sha256': '5e687e1027b95374ea4f1aa1b89aa4b0527e9e189dc1adb5f1670254e73a8c6a'}, 'ORIGINAL_START_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/admitted_qualify_mlm_v1/root_supervisor_start.json', 'sha256': 'b94eec793aba4080a4d7271fdd813c8598cc57dbe8af7e4e84a45d40e1a1ac5c'}, 'PROBE_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v5/actual_pidfd_self_probes_root_v1.json', 'sha256': '72f8fa2585e432aee18a2e9608dece5371a732fff8064e23a9f88df406922779'}, 'BACKEND_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py', 'sha256': '61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca'}}
globals().update(FIXED_REFS)
def require(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(p!=ROOT and ".." not in p.parts and p.resolve()==p and p.is_relative_to(ROOT),"Canonical project path")
    return p
def ref(value):
    p=local(value);h=hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda:stream.read(8388608),b""):h.update(block)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())
def read(record,cap=1048576):
    require(type(record) is dict and set(record)=={"path","sha256"},"Exact metadata ref")
    p=local(record["path"]);s=p.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size<=cap and ref(p)==record,"Exact bounded regular metadata")
    return json.loads(p.read_text())
def save(path,value):
    p=local(path);raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Bounded root execution metadata")
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(p)
def check_expiry(spec,*,now=None):
    now=datetime.now(timezone.utc) if now is None else now
    require(spec["created_at_utc"]==CREATED and spec["expires_at_utc"]==EXPIRES,"Original absolute expiry; no renewal")
    require(datetime.fromisoformat(CREATED)<=now<datetime.fromisoformat(EXPIRES),"Original authority expired")
def authority_ref():
    raw=os.environ.get(ENV_REF)
    require(raw is not None,"Explicit root amendment authority required")
    return json.loads(raw)
def lifetime_fd():
    value=os.environ.get(ENV_FD,"")
    require(value.isdigit() and int(value)>=3,"Inherited amendment lifetime FD required")
    return int(value)
def _stat(pid):
    try:
        fields=Path("/proc/%d/stat"%pid).read_text().rsplit(")",1)[1].split()
        return dict(pid=pid,state=fields[0],ppid=int(fields[1]),start=int(fields[19]))
    except FileNotFoundError:return None
def same_identity(actual,expected):
    return actual is not None and actual["pid"]==expected["pid"] and actual["start"]==expected["start"] and actual["state"] not in ("Z","X")
def descends_from(pid,owner,read_stat=_stat):
    seen=set()
    for _ in range(128):
        current=read_stat(pid)
        if not current or pid in seen:return False
        if pid==owner["pid"]:return same_identity(current,owner)
        seen.add(pid);pid=current["ppid"]
    return False
def check_lifetime():
    check_expiry(dict(created_at_utc=CREATED,expires_at_utc=EXPIRES))
    fd=lifetime_fd();descriptor=os.fstat(fd);named=LOCK.lstat()
    require(stat.S_ISREG(descriptor.st_mode) and descriptor.st_nlink==1
        and (descriptor.st_dev,descriptor.st_ino)==(named.st_dev,named.st_ino),"Exact inherited attempt-lock descriptor")
    raw=os.pread(fd,65537,0);require(len(raw)<=65536,"Bounded attempt token")
    token=json.loads(raw)
    require(token["schema"]=="native_MLM_execution_attempt_token_v6" and token["authority_ref"]==authority_ref()
        and token["attempt_directory"]==str(ATTEMPT.relative_to(ROOT)),"Exact active one-use attempt token")
    require(descends_from(os.getpid(),token["owner"]),"Actual root-owned attempt descendant")
    # An independent open must see the owner's exclusive lock. Do not acquire/release
    # flock on the inherited open-file-description, which would disturb its owner.
    probe=os.open(LOCK,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise ValueError("Lifetime exclusion lock is not held")
    finally:os.close(probe)
    return token
def fixed_authority_fields():
    return dict(schema="native_MLM_execution_amendment_authority_v6",root_authorizer="root_agent",
        key=KEY,job_id=JOB,original_refs=copy.deepcopy(FIXED_REFS),output_directory=OUTPUT,
        auxiliary_scopes=[str(AUX),str(TMP)],reserved_bytes=CAP,physical_gpu=4,GPU_UUID=GPU_UUID,
        phase="qualify_mlm",created_at_utc=CREATED,expires_at_utc=EXPIRES,
        attempt_directory=str(ATTEMPT.relative_to(ROOT)),accounting_mutation_authorized=False,
        source_or_reservation_mutation_authorized=False,scientific_scope_changed=False,
        full_original_reservation_remains_charged=True,automatic_replay=False)
def validate_authority(record,*,allow_expired=False):
    auth=read(record);fixed=fixed_authority_fields()
    require(all(auth.get(k)==v for k,v in fixed.items()),"Only exact never-spawned original allocation amendment")
    require(set(auth)==set(fixed)|{"active","source_closure_ref","input_ref","root_directive_ref","old_writer_absence_ref"} and auth["active"] is True,"Exact active root execution authority")
    if not allow_expired:check_expiry(auth)
    for item in FIXED_REFS.values():require(ref(item["path"])==item,"Original source/evidence changed")
    probe=read(PROBE_REF)
    require(probe["status"]=="actual_self_probes_passed_in_both_interpreters" and probe["source_ref"]==BACKEND_REF
        and all(probe[k]["returncode"]==0 and probe[k]["result"]["status"]=="passed_own_pidfd_signal0_and_CLOEXEC" for k in ("project","native")),"Actual original pidfd self-probes")
    failure=read(ORIGINAL_FAILURE_REF)
    require(failure["status"]=="outer_launch_pre_spawn_pidfd_API_failure"
        and failure["actual_direct_worker_wait_available"] is False and failure["output_directory_created"] is False
        and failure["qualifier_optimizer_steps"]==0 and failure["full_original_reservation_remains_charged"] is True,"Original failed pre-spawn case only")
    storage=read(ORIGINAL_STORAGE_REF)
    require(storage["reservation"]["key"]==KEY and storage["reserved_new_bytes"]==CAP
        and storage["reservation"]["output"]==str(ROOT/OUTPUT)
        and storage["reservation"]["auxiliary_scopes"]==[str(AUX),str(TMP)],"Original immutable storage allocation")
    directive=read(auth["root_directive_ref"])
    require(directive==dict(schema="native_MLM_execution_amendment_root_directive_v6",root_authorizer="root_agent",
        active=True,key=KEY,source_closure_ref=auth["source_closure_ref"],input_ref=auth["input_ref"],
        amendment_source_ref=ref(SOURCE),original_storage_ref=ORIGINAL_STORAGE_REF,
        all_original_launch_resume_write_capabilities_revoked=True,
        cooperative_root_writer_model_acknowledged=True,immutable_old_code_mechanical_rejection_claimed=False,
        no_accounting_or_original_source_mutation=True,reserved_bytes=CAP,expires_at_utc=EXPIRES),"Explicit exact root directive/revocation")
    closure=read(auth["source_closure_ref"])
    require(closure["schema"]=="native_MLM_execution_source_closure_v6"
        and closure["original_storage_ref"]==ORIGINAL_STORAGE_REF
        and closure["actual_backend_ref"]==BACKEND_REF,"Separate actual execution source closure")
    require(set(closure)=={"schema","original_storage_ref","actual_backend_ref","source_sha256","v5_provenance_source_refs"},"Exact closure fields")
    for path,sha in closure["source_sha256"].items():require(ref(path)["sha256"]==sha,"Actual complete execution source changed")
    for item in closure["v5_provenance_source_refs"]:require(ref(item["path"])==item,"Original v5 provenance source changed")
    original_sources=read(ORIGINAL_INPUT_REF)["source_sha256"]
    require(all(closure["source_sha256"].get(path)==sha for path,sha in original_sources.items()),"Preserve complete original immutable source closure")

    require(all(path in closure["source_sha256"] for path in required_execution_sources()),"Complete versioned execution path closure")
    spec=read(auth["input_ref"]);validate_input_fields(auth,spec)
    return auth,spec,storage
def required_execution_sources():
    names=("amendment_v6.py","entry_v6.py","launcher_v6.py","root_native_driver_v1.py","root_native_contract_v1.py",
        "consumer_contract_v1.py","context_factory_v1.py","supervisor_v1.py","native_child_v1.py",
        "qualifier_v1.py","terminal_bounds_v1.py","corpus_auth_v2.py","owned_lifecycle_v2.py","native_runtime_v1.py")
    return tuple(REL+"/"+name for name in names)+(BACKEND_REF["path"],"src/arggnn/resources.py")
def validate_input_fields(auth,spec):
    old=read(ORIGINAL_INPUT_REF);expected=copy.deepcopy(old)
    expected.update(main_source_ref=ref(ENTRY),actual_launcher_ref=ref(LAUNCHER),
        root_context_module_ref=ref(REL+"/context_factory_v1.py"),source_sha256=read(auth["source_closure_ref"])["source_sha256"],
        transitive_source_closure_ref=auth["source_closure_ref"],input_binding_mode="root_execution_amendment_v6")
    require(spec==expected,"Only actual execution provenance differs from immutable original input")
def validate_execution(record,spec,source):
    auth,actual,storage=validate_authority(authority_ref())
    require(record==auth["input_ref"] and spec==actual and source==ENTRY,"Actual amendment input and producer")
    check_lifetime()
    return storage
def allocation_boundary(guard,storage,*,required_write_bytes=0):
    require(storage==read(ORIGINAL_STORAGE_REF),"Original storage object only; no synthetic bindings")
    check_lifetime()
    return guard.budget_check_v9(storage,boundary=True,required_write_bytes=required_write_bytes)
class AllocationBridge:
    """Execution checks plus untouched allocation checker. No new accounting."""
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def budget_check_v9(self,storage,**kw):
        require(storage==read(ORIGINAL_STORAGE_REF),"Exact original storage")
        check_lifetime()
        return self.original.budget_check_v9(storage,**kw)
def observe_old_absence(auth):
    cert=read(auth["old_writer_absence_ref"])
    require(cert["schema"]=="native_MLM_old_writer_absence_root_v6" and cert["root_authorizer"]=="root_agent"
        and cert["active"] is True and cert["key"]==KEY and cert["all_old_writers_and_capabilities_listed"] is True
        and cert["all_old_launch_resume_write_capabilities_revoked"] is True,"Complete root old-writer certificate")
    require(cert["original_outer_OS_wait"]=="unknown"
        and cert["original_scientific_worker_wait_ref"] is None,"Preserve unknown historical outer wait and absent scientific worker wait")
    started=read(ORIGINAL_START_REF)
    require(started["pid"]==2744659 and started["start_ticks"]==473263284
        and started["actual_wait_observed"] is False,"Exact actual original supervisor start; no reconstructed exit")
    rows=cert["observations"];require(1<=len(rows)<=256 and len({r["pid"] for r in rows})==len(rows)
        and 2744659 in {r["pid"] for r in rows},"Complete bounded actual old PID roster")
    now=datetime.now(timezone.utc)
    for row in rows:
        require(row["absent"] is True and row["method"]=="targeted os.kill(pid,0) ESRCH"
            and type(row["pid"]) is int and row["pid"]>=2,"Honest original root absence record")
        stamp=datetime.fromisoformat(row["checked_utc"])
        require(stamp.tzinfo and 0<=(now-stamp).total_seconds()<=300,"Fresh root old-writer observation")
        try:os.kill(row["pid"],0)
        except ProcessLookupError:pass
        else:raise ValueError("Old writer still present/reused; no amendment")
def activate(record,guard):
    auth,spec,storage=validate_authority(record)
    require(os.getpid()==os.getpgrp()==os.getsid(0),"Root dedicated setsid session")
    # This is an actual action gate, used only by ROOT after explicit activation.
    with guard.budget.v4.mutex(ROOT):
        auth,spec,storage=validate_authority(record)
        guard.check_storage_locked_v1(storage,boundary=True,required_write_bytes=134217728)
        observe_old_absence(auth)
        require(not (ROOT/OUTPUT).exists() and not TMP.exists(),"No prior scientific output/tmp")
        require(AUX.is_dir() and not any(AUX.iterdir()),"Original auxiliary scope remains empty; no retry")
        fd=os.open(LOCK,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        # Creation consumes the one-use capability even if subsequent activation fails.
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            owner=_stat(os.getpid());require(owner is not None,"Real root owner identity")
            token=dict(schema="native_MLM_execution_attempt_token_v6",authority_ref=record,
                owner=owner,attempt_directory=str(ATTEMPT.relative_to(ROOT)))
            data=(json.dumps(token,sort_keys=True)+"\n").encode();os.write(fd,data);os.fsync(fd)
            ATTEMPT.mkdir()
            original_admitted=read(ORIGINAL_ADMITTED_REF)
            admitted=dict(schema="native_MLM_amended_execution_packet_v6",storage=storage,
                input_ref=auth["input_ref"],driver_ref=ref(DRIVER),execution_amendment_ref=record,
                original_admitted_ref=ORIGINAL_ADMITTED_REF,
                original_allocation_consumer_binding_count=original_admitted["consumer_binding_count"],
                reserve_receipt_ref=original_admitted["reserve_receipt_ref"],
                caps_ref=ORIGINAL_CAPS_REF,config_ref=ORIGINAL_CONFIG_REF,runtime_ref=ORIGINAL_RUNTIME_REF,
                root_stage_request_ref=ORIGINAL_REQUEST_REF,phase="qualify_mlm",parent_id="american_c3_train_000")
            save(ATTEMPT/"admitted.json",admitted)
            os.environ[ENV_REF]=json.dumps(record,separators=(",",":"));os.environ[ENV_FD]=str(fd)
            return fd
        except BaseException:os.close(fd);raise

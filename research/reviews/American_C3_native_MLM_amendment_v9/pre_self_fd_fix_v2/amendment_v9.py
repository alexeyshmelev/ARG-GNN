"""Root execution-only amendment of spawned failed v8; no new reservation."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,fcntl,hashlib,importlib.util,json,os,stat,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_amendment_v9"
SOURCE=REL+"/amendment_v9.py"
ENTRY=REL+"/entry_v9.py"
LAUNCHER=REL+"/launcher_v9.py"
DRIVER=REL+"/root_native_driver_v1.py"
KEY="H041/American-C3-native-MLM-full30mb-mlm-fit-v8"
JOB="American-C3-native-MLM-full30mb-mlm-fit-v8"
OUTPUT="experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8"
AUX=ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/worker_aux"
TMP=ROOT/".runtime/tmp/c3full8/f"
ATTEMPT=AUX/"amended_v9_attempt1"
LOCK=AUX/"amendment_v9.activation"
GPU_UUID="GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71"
CAP=15032385536
ENV_REF="ARGGNN_MLM_AMENDMENT_V9_REF"
ENV_FD="ARGGNN_MLM_AMENDMENT_V9_FD"
CREATED='2026-10-15T11:47:27.217781+00:00'
EXPIRES='2026-10-18T11:47:27.217781+00:00'
FIXED_REFS={'ORIGINAL_ADMITTED_REF': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/admitted.json', 'sha256': 'f8e848ddaf8feb4fbbeb276d07004514c8433e92aca46edece6c84a60ef2e4c3'}, 'ORIGINAL_CAPS_REF': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/mlm_caps.root_active_v1.json', 'sha256': 'a9d9b29cd1fe944aa73a098cd6eda11b614613b2437b0b6c7f39bdbb94b45337'}, 'ORIGINAL_CONFIG_REF': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/config.json', 'sha256': 'de7563f2180a7f41fcdd50ae6c5eb687dd33e9d70943295152b1f4225714c36d'}, 'ORIGINAL_DRIVER_REF': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/root_native_driver_v1.py', 'sha256': '0af05152d32d58f5e1485b4d0ba33ebdc6aa099d92609cef5661407d756a7230'}, 'ORIGINAL_INPUT_REF': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/input.json', 'sha256': 'a5aac7f1ba2f3bcbb08b6614d4c37f4ec005d64607750c3bd30fa8d07419e6a0'}, 'ORIGINAL_REQUEST_REF': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/mlm_request.root_active_v1.json', 'sha256': '5d8ea663efe93e197533aa9640de3f685301da144b539f5cc6e10ba4887e3f46'}, 'ORIGINAL_RESERVE_REF': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/reserve_receipt.json', 'sha256': 'c8ecfcc493205719b154352e9785fab40a9421087b7d6f416cf1964c8a608db9'}, 'ORIGINAL_RUNTIME_REF': {'path': 'research/reviews/American_C3_native_MLM_full_source_v4/runtime_root_GPU4_v1.json', 'sha256': 'd9f4f45fb3fa5e1eb9878b5d00385c90cd4d80a3080e65f4438baaacf1a28263'}, 'ORIGINAL_STORAGE_REF': {'path': 'research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/storage.json', 'sha256': 'ff3867cafa84ba18d7b298a2eaa0cc78dea61207248057d949f04db9421a3ab9'}}
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
    require(token["schema"]=="native_MLM_execution_attempt_token_v9" and token["authority_ref"]==authority_ref()
        and token["attempt_directory"]==str(ATTEMPT.relative_to(ROOT)),"Exact active one-use attempt token")
    require(descends_from(os.getpid(),token["owner"]),"Actual root-owned attempt descendant")
    # An independent open must see the owner's exclusive lock. Do not acquire/release
    # flock on the inherited open-file-description, which would disturb its owner.
    probe=os.open(LOCK,os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise ValueError("Lifetime exclusion lock is not held")
    finally:os.close(probe)
    return token

def check_expiry(spec=None,*,now=None):
    now=datetime.now(timezone.utc) if now is None else now
    if spec is not None:
        require(spec["created_at_utc"]==CREATED and spec["expires_at_utc"]==EXPIRES,"Original absolute admission timestamps")
    require(datetime.fromisoformat(CREATED)<=now<datetime.fromisoformat(EXPIRES),"Original fullfit expiry; no renewal")
def required_execution_sources():
    return [REL+"/"+name for name in ("amendment_v9.py","entry_v9.py","launcher_v9.py","root_native_contract_v1.py",
        "root_native_driver_v1.py","context_factory_v1.py","consumer_contract_v1.py","supervisor_v1.py","native_child_v1.py",
        "native_runtime_v1.py","owned_lifecycle_v2.py","terminal_bounds_v1.py","corpus_auth_v2.py","terminal_capture_v9.py","terminal_namespace_v9.py","root_archive_v9.py","current_writer_proof_v9.py")]
def validate_authority(record,*,allow_expired=False):
    auth=read(record)
    require(auth["schema"]=="native_MLM_fullfit_execution_authority_v9" and auth["root_authorizer"]=="root_agent"
        and auth["active"] is True and auth["key"]==KEY and auth["job_id"]==JOB
        and auth["original_refs"]==FIXED_REFS and auth["reserved_bytes"]==CAP
        and auth["created_at_utc"]==CREATED and auth["expires_at_utc"]==EXPIRES
        and auth["physical_gpu"]==4 and auth["GPU_UUID"]==GPU_UUID
        and auth["accounting_or_original_binding_mutation_authorized"] is False
        and auth["scientific_scope_changed"] is False and auth["all_v8_launch_resume_write_capabilities_cooperatively_revoked"] is True
        and auth["immutable_old_code_mechanical_rejection_claimed"] is False
        and auth["dead_owned_socket_cleanup_after_preserved_metadata_authorized"] is True
        and auth["checkpoint_latest_regular_copy_after_drain_authorized"] is True,
        "Explicit actual root same-charge execution-only authority")
    if not allow_expired:check_expiry(auth)
    for row in FIXED_REFS.values():require(ref(row["path"])==row,"Immutable original allocation/input/science sources")
    original=read(ORIGINAL_INPUT_REF);storage=read(ORIGINAL_STORAGE_REF)
    require(storage["reservation"]["key"]==KEY and storage["reserved_new_bytes"]==CAP
        and storage["reservation"]["output"]==str(ROOT/OUTPUT)
        and storage["reservation"]["auxiliary_scopes"]==[str(AUX),str(TMP)]
        and storage["expires_at_utc"]==EXPIRES,"Byte-exact original charge/scopes/expiry")
    outer=read(auth["actual_old_outer_wait_ref"])
    require(outer["actual_direct_outer_supervisor_wait"] is True and outer["pid"]==2817234
        and outer["returncode"]==1,"Genuine failed v8 root outer wait1, never native reconstruction")
    absence=read(auth["old_writer_absence_ref"]);archive=read(auth["archive_receipt_ref"])
    from current_writer_proof_v9 import validate_static_certificate
    validate_static_certificate(absence)  # Static binding only while live v9 legitimately reuses this job/scopes.
    require(archive["root_authorizer"]=="root_agent" and archive["active"] is True
        and archive["all_scientific_and_regular_files_preserved"] is True
        and archive["all_socket_metadata_preserved_before_narrow_cleanup"] is True
        and archive["archive_regular_namespace_confirmed"] is True
        and archive["same_filesystem_rename"] is True
        and archive["historical_ids_complete"] is False and archive["historical_descendant_ids_lost"] is True
        and archive["retained_archive_and_metadata_bytes"]<=5242880
        and archive["old_writer_absence_ref"]==auth["old_writer_absence_ref"],
        "Root regular archive <=5MiB, preserving failed payload and dead IPC metadata")
    closure=read(auth["source_closure_ref"])
    require(closure["schema"]=="native_MLM_fullfit_execution_source_closure_v9"
        and closure["original_storage_ref"]==ORIGINAL_STORAGE_REF
        and all(closure["source_sha256"].get(p)==sha for p,sha in original["source_sha256"].items())
        and all(p in closure["source_sha256"] for p in required_execution_sources()),
        "All original31 allocation pins plus truthful complete actual v9 execution closure")
    for path,sha in closure["source_sha256"].items():require(ref(path)["sha256"]==sha,"Actual v9 source closure bytes")
    spec=read(auth["input_ref"]);expected=copy.deepcopy(original)
    expected.update(main_source_ref=ref(ENTRY),actual_launcher_ref=ref(LAUNCHER),
        root_context_module_ref=ref(REL+"/context_factory_v1.py"),source_sha256=closure["source_sha256"],
        transitive_source_closure_ref=auth["source_closure_ref"],input_binding_mode="actual_root_execution_amendment_v9")
    require(spec==expected,"Only execution provenance changes; exact original science/storage/expiry")
    return auth,spec,storage
def validate_execution(record,spec,source):
    auth,actual,storage=validate_authority(authority_ref())
    require(record==auth["input_ref"] and spec==actual and source==ENTRY,"Actual new producer is amendment-bound, not originally bound")
    check_lifetime();return storage
def allocation_boundary(guard,storage,*,required_write_bytes=0):
    require(storage==read(ORIGINAL_STORAGE_REF),"Exact original immutable allocation")
    check_lifetime();return guard.budget_check_v9(storage,boundary=True,required_write_bytes=required_write_bytes)
class AllocationBridge:
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def budget_check_v9(self,storage,**kw):
        require(storage==read(ORIGINAL_STORAGE_REF),"Original storage only")
        check_lifetime();return self.original.budget_check_v9(storage,**kw)
def observe_old_absence(auth):
    from current_writer_proof_v9 import validate_static_certificate,prove_current_absence
    validate_static_certificate(read(auth["old_writer_absence_ref"]))
    return prove_current_absence()  # Actual roots/groups+complete current metadata scan, never a callback.

def preserve_prestart_proof(auth,name):
    proof=observe_old_absence(auth)
    archive=read(auth["archive_receipt_ref"])
    directory=local(archive["archive_root"])
    save(directory/name,proof)  # <=64KiB administrative allowance, before risky original boundary checks.
    return proof
def capture(role):
    check_lifetime()
    from terminal_capture_v9 import Capture
    admitted=read(ref(ATTEMPT/"admitted.json"));manifest=read(admitted["terminal_capture_manifest_ref"])
    require(manifest["directory"]==str(ATTEMPT/"terminal_capture"),"Exact precharged capture scope")
    return Capture(role,manifest)
def admin_save(name,value):
    """Narrow root-authorized bounded terminal/admin evidence; no scope census."""
    require(Path(name).name==name and name.endswith(".json"),"Flat unique administrative evidence")
    return save(ATTEMPT/name,value)
def activate(record,guard):
    auth,spec,storage=validate_authority(record)
    require(os.getpid()==os.getpgrp()==os.getsid(0),"Dedicated root setsid owner")
    preserve_prestart_proof(auth,"preactivation_current_writer_proof_v9_v2.json")
    require(not (ROOT/OUTPUT).exists() and not TMP.exists(),"Old output/tmp preserved in root archive before fresh reuse")
    require(not LOCK.exists() and not ATTEMPT.exists(),"One-use v9 attempt, never replay")
    with guard.budget.v4.mutex(ROOT):
        guard.check_storage_locked_v1(storage,boundary=True,required_write_bytes=134217728)
        preserve_prestart_proof(auth,"preactivation_locked_current_writer_proof_v9_v2.json")
        fd=os.open(LOCK,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            ATTEMPT.mkdir()
            from terminal_capture_v9 import prepare
            manifest=prepare(ATTEMPT/"terminal_capture")
            manifest_ref=save(ATTEMPT/"terminal_capture_manifest.json",manifest)
            old=read(ORIGINAL_ADMITTED_REF)
            save(ATTEMPT/"admitted.json",dict(schema="native_MLM_execution_attempt_admitted_v9",
                input_ref=auth["input_ref"],driver_ref=ref(DRIVER),storage=storage,
                reserve_receipt_ref=old["reserve_receipt_ref"],terminal_capture_manifest_ref=manifest_ref,
                execution_amendment_ref=record,original_admitted_ref=ORIGINAL_ADMITTED_REF))
            token=dict(schema="native_MLM_execution_attempt_token_v9",authority_ref=record,
                owner=_stat(os.getpid()),attempt_directory=str(ATTEMPT.relative_to(ROOT)))
            raw=json.dumps(token,sort_keys=True).encode();os.write(fd,raw);os.fsync(fd)
            os.environ[ENV_REF]=json.dumps(record,separators=(",",":"));os.environ[ENV_FD]=str(fd)
            return fd
        except BaseException:os.close(fd);raise

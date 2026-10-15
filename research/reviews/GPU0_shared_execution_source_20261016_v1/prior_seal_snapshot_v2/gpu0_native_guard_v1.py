"""GPU0-only private lease/title adapter. Standard library, no probes on import.
Original native helper bodies retained except eligibility/error text.
No CUDA, inventory, admission or installation occurs on import.
"""
from __future__ import annotations
import fcntl,hashlib,json,os,stat,sys,types,uuid
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py"
PROCESS_TITLE="AI Auto Research Pipeline Test"
ALLOWED_GPUS=(0,)
GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
POLICY_REF={'path': 'research/reviews/GPU0_only_user_transition_root_20261016_v1/GPU0_only_policy.root_active_v1.json', 'sha256': 'bba84cecf416b2d284126bd31b72cff26ef9c46ee24dae375721d3f57240ed24'}
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def project_path(value):
    lexical = Path(value)
    if ".." in lexical.parts:
        raise ValueError("Output traversal is forbidden")
    path = lexical if lexical.is_absolute() else ROOT / lexical
    if path != path.resolve() or not path.is_relative_to(ROOT):
        raise ValueError("Canonical in-project path without symlink traversal required")
    return path

def canonical_uuid(value):
    if not isinstance(value, str):
        raise RuntimeError("GPU UUID must be a string")
    bare = value[4:] if value.startswith("GPU-") else value
    parsed = str(uuid.UUID(bare))
    if bare != parsed:
        raise RuntimeError("Canonical lowercase GPU UUID required")
    return parsed

def validate_lease(environ=None):
    """Validate the actual held inherited descriptor, without CUDA inspection."""
    environ = os.environ if environ is None else environ
    physical = environ.get("ARGGNN_PHYSICAL_GPU")
    if physical not in {str(gpu) for gpu in ALLOWED_GPUS}:
        raise RuntimeError("Only physical GPU0 is eligible")
    expected = ROOT / ".runtime/locks" / ("gpu_" + physical + ".lock")
    try:
        lease = Path(environ["ARGGNN_GPU_LEASE"])
        raw_fd = environ["ARGGNN_GPU_LEASE_FD"]
        if not raw_fd.isdecimal() or str(int(raw_fd)) != raw_fd:
            raise ValueError("Canonical descriptor number required")
        fd = int(raw_fd)
        descriptor = os.fstat(fd)
        disk = lease.lstat()
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError("Valid inherited GPU lease required") from error
    if (lease != expected or lease.resolve() != expected or
            not stat.S_ISREG(disk.st_mode) or not stat.S_ISREG(descriptor.st_mode) or
            (disk.st_dev, disk.st_ino) != (descriptor.st_dev, descriptor.st_ino) or
            Path("/proc/self/fd/" + str(fd)).resolve() != expected):
        raise RuntimeError("Lease descriptor, inode or canonical path differs")
    contents = os.pread(fd, 65536, 0)
    if len(contents) != descriptor.st_size:
        raise RuntimeError("Lease metadata is oversized or changed")
    metadata = json.loads(contents)
    gpu_uuid = metadata.get("uuid")
    canonical_uuid(gpu_uuid)
    if (not gpu_uuid.startswith("GPU-") or metadata.get("gpu") != int(physical) or
            environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or
            not environ.get("ARGGNN_JOB_NAME") or
            metadata.get("job") != environ.get("ARGGNN_JOB_NAME") or
            type(metadata.get("launcher_pid")) is not int or metadata["launcher_pid"] <= 0):
        raise RuntimeError("Lease UUID, physical index or job metadata differs")
    with expected.open("r+") as independent:
        try:
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(independent, fcntl.LOCK_UN)
            raise RuntimeError("Scheduler reservation is not held")
    return {"physical_gpu": int(physical), "visible_gpu_uuid": gpu_uuid,
            "canonical_uuid": canonical_uuid(gpu_uuid), "lease_path": str(expected),
            "lease_fd": fd, "lease_device": descriptor.st_dev,
            "lease_inode": descriptor.st_ino, "lease_metadata_sha256":
            hashlib.sha256(contents).hexdigest(), "launcher_metadata": metadata}

def apply_and_validate_title():
    import setproctitle
    setproctitle.setproctitle(PROCESS_TITLE)
    observed = Path("/proc/self/cmdline").read_bytes().split(b"\0")[0].decode().strip()
    if setproctitle.getproctitle() != PROCESS_TITLE or observed != PROCESS_TITLE:
        raise RuntimeError("Required actual process title missing")
    return observed

_original_validate_lease=validate_lease
def validate_policy_v1():
    p=ROOT/POLICY_REF["path"]
    if p.resolve()!=p or not p.is_file() or p.is_symlink() or sha256_file(p)!=POLICY_REF["sha256"]:
        raise RuntimeError("Exact canonical active GPU0 policy changed")
    policy=json.loads(p.read_bytes())
    if not (policy.get("schema")=="root_latest_human_GPU_policy_v1"
        and policy.get("status")=="active" and policy.get("root_authorizer")=="root_agent"
        and policy.get("allowed_physical_GPUs")==[0]
        and policy.get("maximum_distinct_project_GPUs")==1
        and policy.get("all_nonzero_GPUs_excluded") is True
        and policy.get("parallel_jobs_on_GPU0_allowed") is True
        and policy.get("preserve_external_jobs") is True and policy.get("GPU0_uuid")==GPU0_UUID):
        raise RuntimeError("Current human GPU0-only policy required")
    return policy

def validate_lease(environ=None):
    validate_policy_v1()
    e=os.environ if environ is None else environ
    if e.get("ARGGNN_PHYSICAL_GPU")!="0" or e.get("CUDA_VISIBLE_DEVICES")!=GPU0_UUID:
        raise RuntimeError("Only policy-bound GPU0 UUID visibility is allowed")
    result=_original_validate_lease(e)
    if result["physical_gpu"]!=0 or result["visible_gpu_uuid"]!=GPU0_UUID:
        raise RuntimeError("GPU0 inherited lease UUID differs")
    return result

def install_before_torch_v1():
    """Explicit interpreter-local canonical guard substitution, before imports.
    No installed file is changed. Actual adapter ref must be in the input closure.
    """
    if "torch" in sys.modules or "torch_geometric" in sys.modules:
        raise RuntimeError("Install GPU0 guard before scientific/Torch imports")
    validate_policy_v1()
    old=sys.modules.get("arggnn.native_argformer_exec")
    if old is not None and getattr(old,"GPU0_ADAPTER_SOURCE",None)!=SOURCE:
        raise RuntimeError("Canonical native guard already imported before GPU0 adapter")
    m=types.ModuleType("arggnn.native_argformer_exec",__doc__)
    m.__file__=str(ROOT/SOURCE);m.__package__="arggnn"
    for name in ("ROOT","PROCESS_TITLE","ALLOWED_GPUS","sha256_file","project_path","canonical_uuid","validate_lease","apply_and_validate_title"):
        setattr(m,name,globals()[name])
    m.GPU0_ADAPTER_SOURCE=SOURCE
    sys.modules[m.__name__]=m
    import arggnn
    arggnn.native_argformer_exec=m
    return m

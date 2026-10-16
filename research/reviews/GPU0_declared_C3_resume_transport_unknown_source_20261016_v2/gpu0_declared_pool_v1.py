"""Finite future GPU0 facade; inactive without exact root capability.

Privately reuses reviewed memory/ownership/wait/drain/lease bodies. Imports do
not inspect processes, CUDA, inventory, sockets, locks or storage accounting.
"""
from __future__ import annotations
from pathlib import Path
import ast,hashlib,importlib.util,json,os,resource,sys,time,types

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_declared_C3_resume_transport_unknown_source_20261016_v2"
SOURCE=HERE+"/gpu0_declared_pool_v1.py"
LAUNCHER=HERE+"/launch_declared_gpu0_v1.py"
DISPATCHER=HERE+"/declared_runtime_v1.py"
BOOTSTRAP=HERE+"/bootstrap_declared_v1.py"
CONTRACT_REF={"path":HERE+"/finite_jobs_v1.py","sha256":"e13b9f4a1d92d7720f87cbbdf9ab5ed5ad4523f81e48a1a5898393fbe02256f2"}
CATALOGUE_REF={"path":HERE+"/finite_catalogue.v1.json","sha256":"8226afd683102b52f0e08e11b0d1bb2f966b36d63d2e97943cd962525b95a34d"}
REVIEWED="research/reviews/GPU0_shared_execution_source_20261016_v1"
REVIEWED_POOL_REF={"path":REVIEWED+"/gpu0_shared_pool_v1.py","sha256":"28bb690f33c2384d04f88dbacff1f910694c398c81874f9b9751336f4fb6429c"}
GUARD_REF={"path":REVIEWED+"/gpu0_native_guard_v1.py","sha256":"3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69"}
GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    require(isinstance(path,str) and not Path(path).is_absolute() and ".." not in Path(path).parts,"Contained source/metadata")
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
        and p.stat().st_size<=32*2**20,"Canonical bounded source/metadata")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def load(record,label):
    require(ref(record["path"])==record,"Exact source before private import")
    spec=importlib.util.spec_from_file_location(label,ROOT/record["path"])
    module=importlib.util.module_from_spec(spec);sys.modules[label]=module;spec.loader.exec_module(module)
    return module

contract=load(CONTRACT_REF,"_GPU0_declared_finite_contract_v1")
require(ref(CATALOGUE_REF["path"])==CATALOGUE_REF,"Exact finite scientific catalogue")
reviewed=load(REVIEWED_POOL_REF,"_GPU0_declared_private_reviewed_pool_v1")
legacy=reviewed.legacy
base=reviewed.base
bound=legacy.bound
CHECKER_REF=reviewed.CHECKER_REF
MEMO_REF={"path":"research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py","sha256":"c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364"}
V4_REF={"path":"src/arggnn/project_budget_v4.py","sha256":"a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47"}
# Only this newly loaded private legacy namespace is adapted.
legacy.SOURCE=SOURCE
legacy.LAUNCHER=LAUNCHER
legacy.RENEWAL_BOOTSTRAP_REF={"path":BOOTSTRAP,"sha256":"47f10e780d736bad18c12a8cb721d9882802538200215430cb1efd403a1b938c"}

def socket_name(cap):
    token=hashlib.sha256(base.encoded(cap["plan_ref"])).hexdigest()[:16]
    return "\0arggnn_gpu0_declared_"+cap["pool"]+"_"+token

def guard_module():
    return load(GUARD_REF,"_GPU0_declared_exact_native_guard")

def common_required_v1(cap,g):
    paths=[SOURCE,LAUNCHER,DISPATCHER,BOOTSTRAP,
        REVIEWED+"/launch_gpu0_shared_v1.py",REVIEWED+"/gpu0_runtime_adapter_v1.py",
        REVIEWED+"/gpu0_bootstrap_renewal_v1.py",reviewed.DOWN,reviewed.OLD,
        legacy.BASE]
    rows=[ref(p) for p in paths]
    rows += [CONTRACT_REF,CATALOGUE_REF,REVIEWED_POOL_REF,GUARD_REF,CHECKER_REF,MEMO_REF,V4_REF,
        dict(path=legacy.GUARD,sha256=legacy.GUARD_SHA),legacy.POLICY,legacy.REPAIR,
        legacy.RENEWAL_SOURCE_REF,cap["marker_repair_renewal_ref"],g.POLICY_REF,cap["stop_evidence_ref"]]
    rows += contract.resume_transition_refs_v1(cap["prior_controller_completion_ref"])
    rows += contract.transport_transition_refs_v1(cap["prior_controller_completion_ref"])
    return [r for i,r in enumerate(rows) if r not in rows[:i]]

def load_cap(record,allow_drain=False):
    cap=json.loads(bound(record))
    require(Path(__file__).resolve()==ROOT/SOURCE,"Exact newly reviewed additive route")
    require(cap.get("schema")=="GPU0_declared_logical_pool_capability_v1"
        and cap.get("status")=="active" and cap.get("root_authorizer")=="root_agent"
        and type(cap.get("physical_gpu")) is int and cap["physical_gpu"]==0
        and cap.get("gpu_uuid")==GPU0_UUID and cap.get("maximum_distinct_project_GPUs")==1
        and cap.get("logical_job_sharing_authorized") is True and cap.get("external_job_sharing_authorized") is True
        and cap.get("legacy_exclusive_coexistence_allowed") is False,"Explicit fresh GPU0-only root capability")
    count=cap.get("maximum_concurrent_jobs")
    require(type(count) is int and 1<=count<=3
        and (count<3 or cap.get("three_logical_jobs_authorized") is True),
        "Root-authorized maximum3 logical jobs; actual unchanged memory gate still decides")
    require(isinstance(cap.get("pool"),str) and base.NAME.fullmatch(cap["pool"]) and len(cap["pool"])<=32
        and cap.get("headroom_mib")==4096,"Safe name and unchanged4GiB headroom")
    # Require genuine old outer-controller wait/closed/drain before any future hook.
    contract.validate_previous_controller_v1(cap["prior_controller_completion_ref"])
    g=guard_module();policy=g.validate_policy_v1()
    require(cap.get("GPU0_policy_ref")==g.POLICY_REF and cap.get("stop_evidence_ref")==policy["stop_evidence_ref"],
        "Unchanged active human policy and original GPU0 transition")
    bound(cap["stop_evidence_ref"])
    certificate=legacy.install_marker_renewal(cap["marker_repair_renewal_ref"])
    renewal_end=contract.expiry(certificate["renewed_expires_at_utc"])
    require(type(cap["expires_unix"]) in (int,float) and type(cap["drain_expires_unix"]) in (int,float)
        and time.time()<(cap["drain_expires_unix"] if allow_drain else cap["expires_unix"])
        and cap["drain_expires_unix"]<renewal_end,"Finite authority and complete owned drain within existing renewal")
    require(cap.get("owned_termination_escalation_authorized") is True
        and cap.get("owned_drain_after_expiry_authorized") is True,"Original owned-only drain/escalation authority")
    base.integer(cap["termination_grace_seconds"],"TERM grace",1)
    require(cap["termination_grace_seconds"]<=120
        and cap["drain_expires_unix"]>=cap["expires_unix"]+cap["termination_grace_seconds"]+5,
        "Unchanged bounded TERM grace and drain")
    sources=cap["source_refs"]
    require(isinstance(sources,dict) and len(sources)>0,"Finite authenticated source/metadata closure")
    for path,r in sources.items():
        require(path==r["path"],"Closure key/path");bound(r)
    required=common_required_v1(cap,g)
    require(all(r in sources.values() for r in required),"Actual facade plus original reviewed source/guard/policy closure")
    plan=json.loads(bound(cap["plan_ref"]))
    fields=("pool","physical_gpu","gpu_uuid","maximum_distinct_project_GPUs","maximum_concurrent_jobs",
        "three_logical_jobs_authorized","logical_job_sharing_authorized","external_job_sharing_authorized",
        "legacy_exclusive_coexistence_allowed","headroom_mib","termination_grace_seconds",
        "owned_termination_escalation_authorized","owned_drain_after_expiry_authorized",
        "expires_unix","drain_expires_unix","jobs","source_refs","marker_repair_renewal_ref","GPU0_policy_ref",
        "stop_evidence_ref","finite_job_map_ref","prior_controller_completion_ref","controller_job_id",
        "native_40GiB_operational_ceiling_authorized","native_72h_operational_deadline_authorized",
        "efficient_C3_full_state_resume_authorized")
    require(plan.get("schema")=="GPU0_declared_logical_pool_plan_v1" and plan.get("status")=="active"
        and plan.get("root_authorizer")=="root_agent" and all(plan[k]==cap[k] for k in fields),
        "Exact root plan for every operational authority, source and finite job")
    admin=json.loads(bound(cap["controller_storage_ref"]))
    require(admin["job_id"]==cap["controller_job_id"] and base.NAME.fullmatch(admin["job_id"]),
        "Separate safe root controller scope identity")
    require(type(admin["reserved_new_bytes"]) is int and 32*2**20<=admin["reserved_new_bytes"]<=64*2**20
        and admin["reservation"]["auxiliary_scopes"]==[],"Unchanged isolated32to64MiB controller checker scope")
    require(isinstance(cap["jobs"],dict) and 1<=len(cap["jobs"])<=min(base.MAX_JOBS,13),
        "Finite source-bound catalogue only")
    contract.validate_jobs_v1(cap,required,admin,legacy.ENV_KEYS)
    return cap

legacy.load_cap=load_cap

def _clone_globals_v1(function):
    namespace=dict(function.__globals__,SOURCE=SOURCE,LAUNCHER=LAUNCHER,load_cap=load_cap,
        RENEWAL_BOOTSTRAP_REF=legacy.RENEWAL_BOOTSTRAP_REF,socket_name=socket_name)
    value=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    value.__kwdefaults__=function.__kwdefaults__
    return value

class Declared0Controller(reviewed.Shared0Controller):
    def __init__(self,cap_ref):
        super().__init__(cap_ref)
        # Native original route requires this exact shared lease prefix.
        self.pool_job="gpu0_shared_v1:"+self.pool+":gpu0"
        self.socket_path=socket_name(self.cap)
        require(len(os.fsencode(self.socket_path))<=107,"Bounded distinct declared endpoint")
    def status(self):
        value=super().status();value["schema"]="GPU0_declared_logical_pool_registry_v1"
        if getattr(self,"transport_unknown_start_recheck",None) is not None:
            value["prior_controller_original_external_tool_exit_status"]="UNKNOWN"
            value["prior_controller_original_external_tool_exit_code"]=None
            value["prior_controller_outer_wait0_claimed"]=False
            value["prior_transport_unknown_start_recheck"]=self.transport_unknown_start_recheck
        return value

# Code objects are identical to the reviewed implementations; globals only.
def _verify_transport_lease_on_acquire_v1(app,lease):
    expected=getattr(app,"transport_unknown_expected_lease",None)
    if expected is not None:contract.transport_helper_v1().verify_acquired_lease_v1(lease,expected)

# Same reviewed GPU0 start body plus one authoritative pinned-inode gate before
# any lease metadata overwrite. No guard call is inserted under scheduler flock.
_start_source=(ROOT/reviewed.DOWN).read_text()
_old_forbidden="""for idx in (2,3):
                        with base.lock_file(ROOT,f"gpu_{idx}.lock") as h:
                            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            except BlockingIOError:raise RuntimeError("Forbidden legacy GPU2/3 lease remains active")
                    """
_raw_start=reviewed._method(_start_source,"Shared45Controller","start",[
    (_old_forbidden,""),
    ("self.lease=lease","self.lease=lease\n                    _verify_transport_lease_on_acquire_v1(self,lease)")])
_raw_start.__globals__["_verify_transport_lease_on_acquire_v1"]=_verify_transport_lease_on_acquire_v1
_START_WITH_PINNED_INODE_V1=_clone_globals_v1(_raw_start)

def _start_with_transport_unknown_v1(self):
    prior=contract.read(self.cap["prior_controller_completion_ref"])
    if prior.get("schema")==contract.UNKNOWN_SCHEMA:
        self._check(0)
        started=time.monotonic()
        def fast_resource(event):
            require(time.monotonic()-started<120 and time.time()<self.cap["expires_unix"],
                "Finite120s actual closed-handover startup preflight")
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2*2**30,
                "Bounded2GiB CPU-only handover metadata preflight")
        observation,expected=contract.recheck_transport_unknown_before_start_v1(
            self.cap["prior_controller_completion_ref"],resource_check=fast_resource)
        self.transport_unknown_start_recheck=observation
        self.transport_unknown_expected_lease=expected
        fast_resource("after_actual_closed_handover_preflight")
        self._check(0)
    return _START_WITH_PINNED_INODE_V1(self)

Declared0Controller.start=_start_with_transport_unknown_v1
Declared0Controller._check=_clone_globals_v1(reviewed.Shared0Controller._check)
# _memory/run and every ownership, wait, signal, drain, lock and log body inherit.
# One exact literal module route changes inside the frozen queue body.
old=(ROOT/reviewed.OLD).read_text()
cls=next(n for n in ast.parse(old).body if isinstance(n,ast.ClassDef) and n.name=="Shared45Controller")
node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_launch_queue")
launchtext=ast.get_source_segment(old,node)
before='"arggnn.gpu45_c3cache_h050_pool_v1"'
require(launchtext.count(before)==1,"Single frozen worker module literal")
launchtext=launchtext.replace(before,'"research.reviews.GPU0_declared_C3_resume_transport_unknown_source_20261016_v2.gpu0_declared_pool_v1"')
# Preserve the original queue body, adding only individual expired-queue rejection.
queue_anchor='            if job["state"]!="queued":continue'
queue_expiry=queue_anchor+'\n            original_end=contract.job_expiry_v2(job["requested"])\n            if time.time()>=original_end:\n                job.update(state="cancelled",finished_unix=time.time(),\n                    error="Original per-job input/envelope expired before launch",\n                    expiry_rejected_before_start=True,original_expires_unix=original_end)\n                self._persist()\n                continue'
require(launchtext.count(queue_anchor)==1,"One original queued-state boundary")
launchtext=launchtext.replace(queue_anchor,queue_expiry,1)
# Recheck after slow guard/log preparation; original no-child failure path
# rejects a newly expired job without Popen or aborting unrelated running work.
spawn_anchor='                child=subprocess.Popen('
spawn_gate='                require(contract.job_expiry_v2(j)>time.time(),"Original per-job input/envelope expired at final Popen boundary")\n'+spawn_anchor
require(launchtext.count(spawn_anchor)==1,"One original Popen boundary")
launchtext=launchtext.replace(spawn_anchor,spawn_gate,1)
namespace=dict(vars(legacy),contract=contract)
exec(compile(launchtext,"<reviewed_queue_plus_per_job_expiry_only>","exec"),namespace)
Declared0Controller._launch_queue=namespace["_launch_queue"]

# Preserve the original outer RSS/deadline/global-expiry checks and failure/drain
# handling. A short active job keeps its own absolute expiry under a longer pool.
owned_node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_owned_resource_check")
ownedtext=ast.get_source_segment(old,owned_node)
owned_anchor='        for name,job in active:'
owned_expiry=owned_anchor+'\n            if contract.job_expiry_v2(job["requested"])<=time.time():\n                violations.append(name+": original per-job absolute input/envelope expiry exceeded")'
require(ownedtext.count(owned_anchor)==1,"One original active-job resource boundary")
ownedtext=ownedtext.replace(owned_anchor,owned_expiry,1)
owned_namespace=dict(vars(legacy),contract=contract)
exec(compile(ownedtext,"<original_owned_checks_plus_per_job_expiry_only>","exec"),owned_namespace)
Declared0Controller._owned_resource_check=owned_namespace["_owned_resource_check"]

def worker(cap_ref,name):
    cap=load_cap(cap_ref);require(name in cap["jobs"],"Exact predeclared finite logical job")
    j=cap["jobs"][name];g=guard_module();g.install_before_torch_v1()
    lease=g.validate_lease();g.apply_and_validate_title()
    require(lease["physical_gpu"]==0 and lease["visible_gpu_uuid"]==GPU0_UUID
        and os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==name
        and lease["launcher_metadata"]["job"]=="gpu0_shared_v1:"+cap["pool"]+":gpu0",
        "Canonical inherited lifetime GPU0 FD/title and exact logical identity")
    # Only the distinct small outer scope is checked here. The exact method
    # runtime remains solely responsible for its original scientific guard.
    # The outer envelope has few boundaries and an explicit count-derived
    # small reservation (for example4MiB), so use the unchanged original full
    # guard rather than the32MiB-minimum controller checker. No outer memo
    # wraps shared budget functions before a scientific method's own checker.
    legacy.guarded_check(legacy.current_guard(),json.loads(bound(j["storage_ref"])))
    runtime=load(j["runtime_entry_ref"],"_GPU0_declared_exact_dispatcher")
    return runtime.invoke_job_v1(j)

# Frozen CLI preserves actual wait/cancel/shutdown return semantics.
node=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=="main")
maintext=ast.get_source_segment(old,node)
maintext=maintext.replace("Shared45Controller(","Declared0Controller(").replace("base.request(","controller_request(").replace("physical_GPUs=[4,5]","physical_GPUs=[0]")
namespace=dict(vars(legacy),Declared0Controller=Declared0Controller,worker=worker,
    controller_request=reviewed.controller_request,socket_name=socket_name)
exec(compile(maintext,"<frozen_CLI_declared_route_only>","exec"),namespace)
main=namespace["main"]
if __name__=="__main__":raise SystemExit(main())

"""Source-only GPU4/5 logical sharing successor. No CUDA or accounting mutation on import.

A single controller owns the existing EXCLUSIVE physical device lease. Its
logical jobs inherit that descriptor. No legacy exclusive worker can coexist
on that device. Frozen worker storage, lease, title and Torch caps remain intact.
"""
from __future__ import annotations
import datetime as dt
import argparse, fcntl, hashlib, importlib, importlib.util, json, math, os
from pathlib import Path
import signal, stat, subprocess, sys, threading, time
from arggnn import gpu_pool_v1 as base

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/gpu45_c3cache_h050_pool_v1.py"
LAUNCHER="scripts/launch_reserved_gpu45_c3cache_h050_pool_v1.py"
BASE="src/arggnn/gpu_pool_v1.py"
BASE_SHA="ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1"
GUARD="research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py"
GUARD_SHA="fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56"
POLICY={"path":"research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json","sha256":"eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018"}
REPAIR={"path":"research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json","sha256":"f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"}
MODULE_PINS={
 "arggnn.american_c3_gnn_prefetch_ramcache_isolated980_exec_v3":dict(
  path="src/arggnn/american_c3_gnn_prefetch_ramcache_isolated980_exec_v3.py",
  sha256="56185555115269b9fa27aaca778f6faecf42d3af54e1d6fb217cd7f34deddbe0",cuda_bytes=40*2**30,cpu_bytes=128*2**30),
 "arggnn.H050_direct_readout_cut_memory_exec_v1":dict(
  path="src/arggnn/H050_direct_readout_cut_memory_exec_v1.py",
  sha256="f32a1deb64858549f77cd0ad7b7de7cf4c0cbfaaa975a7be32b631c89bce8a56",cuda_bytes=28*2**30,cpu_bytes=64*2**30)}
MODULES=frozenset(MODULE_PINS)
MIB=1024**2
TITLE="AI Auto Research Pipeline Test"
ACTIVE=frozenset({"launching","running","draining"})
ENV_KEYS=frozenset({"TMPDIR","TMP","TEMP","XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME","HF_HOME","TORCH_HOME","TRITON_CACHE_DIR","CUDA_CACHE_PATH","TORCHINDUCTOR_CACHE_DIR","NUMBA_CACHE_DIR","MPLCONFIGDIR","WANDB_DIR","WANDB_CACHE_DIR","WANDB_CONFIG_DIR","HF_DATASETS_CACHE","HUGGINGFACE_HUB_CACHE","CUDA_CACHE_MAXSIZE","HF_DATASETS_OFFLINE","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","WANDB_MODE","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"})

def require(ok,message):
    if not ok: raise ValueError(message)

def ref(path):
    p=base.canonical_path(ROOT,path)
    require(p.is_file(),"Regular canonical bound file required")
    return {"path":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}

def bound(record,maximum=32*1024**2):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact path/SHA256 reference")
    p=base.canonical_path(ROOT,record["path"])
    with p.open("rb") as h: data=h.read(maximum+1)
    require(len(data)<=maximum and ref(record["path"])==record,"Bound bytes changed")
    return data

def validate_migration_gate(record,gpu):
    """Consume genuine old-run terminal evidence; never probe or revoke old jobs."""
    gate=json.loads(bound(record))
    require(gate.get("schema")=="GPU45_C3cache_H050_migration_gate_v1"
        and gate.get("status")=="active" and gate.get("root_authorizer")=="root_agent"
        and gate.get("physical_gpu")==gpu==4
        and gate.get("all_original_C3_launch_resume_write_capabilities_revoked") is True
        and gate.get("root_consumed_original_supervisor_terminal") is True
        and gate.get("native_GPU5_preserved") is True,"Fresh root-owned GPU4 migration gate only after original terminal/drain/revocation")
    old=json.loads(bound(gate["original_C3_input_ref"]))
    launch=json.loads(bound(gate["original_C3_launch_ref"]))
    wait=json.loads(bound(gate["original_C3_actual_wait_ref"]))
    require(old.get("physical_gpu")==gpu and old.get("phase")=="train"
        and old.get("parent_id")=="American_C3_short20_TRAIN_original5VAL"
        and old.get("main_source_ref",{}).get("path")!=MODULE_PINS[
            "arggnn.american_c3_gnn_prefetch_ramcache_isolated980_exec_v3"]["path"],
        "Gate refers to the original C3 training admission, never a future successor")
    require(launch.get("actual_launch") is True and launch.get("input_ref")==gate["original_C3_input_ref"]
        and wait.get("input_ref")==gate["original_C3_input_ref"]
        and wait.get("launch_ref")==gate["original_C3_launch_ref"]
        and type(launch.get("pid")) is int and launch["pid"]==wait.get("pid")
        and type(wait.get("returncode")) is int and wait.get("actual_direct_launcher_wait") is True
        and wait.get("phase")=="train" and wait.get("all_owned_processes_and_groups_drained") is True,
        "Exact actual original launch/direct wait/owned drain required; success and credit are not inferred")
    return gate


class OwnedProcReader:
    """Discover this controller's tree only, without same-UID or foreign FD scans.

    A verified Linux child subreaper anchors escaped-session orphans. Actual Popen
    PID/start identities seed known roots, task/children edges discover descendants,
    and newly adopted unknown children remain controller_descendant ambiguity.
    Stable scans do not certify external absence or perform any wait or signal.
    Trusted exact worker sources must not delegate work/lease FDs to foreign jobs.
    """
    def __init__(self,identity=None,children=None,uid=None):
        self.identity=identity or self._identity
        self.children=children or self._children
        self.uid=os.getuid() if uid is None else uid
        self.controller_children=[]
        self.uninspected_unrelated=[]  # compatibility field; no unrelated reads

    @staticmethod
    def _identity(pid):
        # One own stat read binds RSS to this exact PID/start sample.
        directory=Path("/proc")/str(pid)
        raw=(directory/"stat").read_text()
        fields=raw[raw.rfind(")")+2:].split()
        return dict(pid=pid,uid=directory.stat().st_uid,state=fields[0],
            ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),
            start_ticks=int(fields[19]),rss_bytes=int(fields[21])*os.sysconf("SC_PAGE_SIZE"))

    @staticmethod
    def _children(pid):
        result=set()
        with os.scandir(Path("/proc")/str(pid)/"task") as tasks:
            for count,task in enumerate(tasks):
                require(count<4096,"Owned task inventory limit exceeded")
                try:
                    text=(Path(task.path)/"children").read_text()
                except FileNotFoundError:
                    continue  # A thread may finish while its task entry is read.
                for token in text.split():
                    require(token.isdecimal() and int(token)>0,"Positive owned child PID")
                    result.add(int(token))
                    require(len(result)<=8192,"Owned child inventory limit exceeded")
        return result

    def __call__(self,lease,controller_pid,known=None):
        known=known or {}
        require(len(known)<=8192,"Bounded recorded owned process identities")
        self.uninspected_unrelated=[]
        def identity(pid,required=False):
            try:return self.identity(pid)
            except (FileNotFoundError,ProcessLookupError):
                if required:raise RuntimeError("Own controller identity disappeared")
                return None
            except PermissionError:
                raise RuntimeError(f"Cannot inspect owned PID {pid} identity")
        def children(pid,required=False):
            try:return set(self.children(pid))
            except (FileNotFoundError,ProcessLookupError):
                if required:raise RuntimeError("Own controller child list disappeared")
                return set()  # Keep the prior process row conservatively this pass.
            except PermissionError:
                raise RuntimeError(f"Cannot inspect owned PID {pid} task children")
        controller=identity(controller_pid,True)
        require(controller["uid"]==self.uid,"Controller UID changed")
        controller_start=controller["start_ticks"]
        previous=None
        for unused in range(8):
            table={controller_pid:dict(controller,lease_fd_present=False,controller_descendant=False)}
            direct=children(controller_pid,True)
            seeds={pid:(None,controller_pid) for pid in direct}
            for pid,saved in known.items():
                require(type(saved.get("start_ticks")) is int,"Actual Popen/recorded start identity required")
                seeds.setdefault(pid,(saved["start_ticks"],None))
            queue=list(seeds.items());visited=set()
            while queue:
                pid,(expected_start,parent_pid)=queue.pop()
                if pid==controller_pid or pid in visited:continue
                visited.add(pid)
                require(len(visited)<=8192,"Owned process inventory limit exceeded")
                item=identity(pid)
                if item is None:continue
                owned=(expected_start is None or item["start_ticks"]==expected_start)
                # A PID reused after a saved descendant exited is observed only
                # to invalidate the old identity. Its task tree is never opened.
                if not owned:
                    table[pid]=dict(item,lease_fd_present=False,controller_descendant=False)
                    continue
                require(item["uid"]==self.uid,"Own descendant UID changed")
                if parent_pid is not None and item["ppid"] not in (parent_pid,controller_pid):
                    raise RuntimeError("Owned task child ancestry changed during discovery")
                table[pid]=dict(item,lease_fd_present=False,controller_descendant=True)
                descendants=children(pid)
                after=identity(pid)
                if after is None:continue
                if after["start_ticks"]!=item["start_ticks"]:
                    # Never attach children read across a reused identity.
                    raise RuntimeError("Owned identity changed during child discovery")
                require(after["uid"]==self.uid,"Own descendant UID changed during discovery")
                for child in descendants:
                    require(type(child) is int and child>0,"Positive owned child identity")
                    queue.append((child,(None,pid)))
            after_controller=identity(controller_pid,True)
            require(after_controller["uid"]==self.uid and after_controller["start_ticks"]==controller_start,
                "Controller identity changed during owned discovery")
            final_direct=children(controller_pid,True)
            signature=(tuple(sorted(final_direct)),tuple(sorted(
                (p,r["start_ticks"],r["ppid"],r["controller_descendant"]) for p,r in table.items())))
            if direct==final_direct and final_direct.issubset(table) and signature==previous:
                self.controller_children=sorted(final_direct)
                return table
            previous=signature
        raise RuntimeError("Owned subreaper tree did not stabilise")

def load_cap(record,allow_drain=False):
    cap=json.loads(bound(record))
    require(Path(__file__).resolve()==ROOT/SOURCE,"Install exact reviewed additive source before execution")
    require(cap.get("schema")=="GPU45_shared_logical_pool_capability_v1"
        and cap.get("status")=="active" and cap.get("root_authorizer")=="root_agent"
        and cap.get("logical_job_sharing_authorized") is True
        and cap.get("legacy_exclusive_coexistence_allowed") is False
        and cap.get("physical_gpu") in (4,5)
        and cap.get("maximum_distinct_project_GPUs")==2
        and cap.get("external_job_sharing_authorized") is True,"Active exact root capability required")
    require(cap.get("owned_termination_escalation_authorized") is True,"Explicit owned PIDFD escalation authority")
    base.integer(cap.get("termination_grace_seconds"),"bounded owned TERM grace",1)
    require(cap["termination_grace_seconds"]<=120 and cap["drain_expires_unix"]>=cap["expires_unix"]+cap["termination_grace_seconds"]+5,"TERM grace fits bounded drain window")
    require(base.NAME.fullmatch(cap["pool"]) and len(cap["pool"])<=32,"Safe pool identity")
    base.canonical_gpu_uuid(cap["gpu_uuid"])
    require(cap["drain_expires_unix"]>=cap["expires_unix"] and cap.get("owned_drain_after_expiry_authorized") is True,"Bounded owned drain authority")
    require(time.time()<(cap["drain_expires_unix"] if allow_drain else cap["expires_unix"]),"Pool capability expired")
    require(cap["source_refs"][SOURCE]==ref(SOURCE) and cap["source_refs"][LAUNCHER]==ref(LAUNCHER)
        and cap["source_refs"][BASE]=={"path":BASE,"sha256":BASE_SHA}
        and ref(BASE)==cap["source_refs"][BASE],"Exact successor and frozen controller sources")
    require(cap["maximum_concurrent_jobs"]==2 and cap["headroom_mib"]==4096,"Exact two-job ceiling and4GiB device headroom")
    validate_migration_gate(cap["migration_gate_ref"],cap["physical_gpu"])
    require({j["module"] for j in cap["jobs"].values()}==MODULES and len(cap["jobs"])==2,"Exactly one cache-C3 and one canonical H050 worker")
    base.integer(cap["headroom_mib"],"device headroom",1)
    for r in cap["source_refs"].values():bound(r)
    plan=json.loads(bound(cap["plan_ref"]))
    require(plan["schema"]=="GPU45_shared_logical_pool_plan_v1"
        and all(plan[k]==cap[k] for k in ["pool","physical_gpu","gpu_uuid","maximum_concurrent_jobs","headroom_mib","termination_grace_seconds","jobs"]),"Exact pre-admission pool plan")
    admin=json.loads(bound(cap["controller_storage_ref"]))
    required=[ref(SOURCE),ref(LAUNCHER),ref(BASE),cap["plan_ref"],{"path":GUARD,"sha256":GUARD_SHA},POLICY,REPAIR]
    require(all(r in admin["reservation"]["consumer_bindings"] for r in required),"Controller source/plan/current-guard closure reserved")
    require(isinstance(cap["jobs"],dict) and 1<=len(cap["jobs"])<=base.MAX_JOBS,"Finite declared jobs")
    for name,j in cap["jobs"].items():
        require(base.NAME.fullmatch(name) and j["job_id"]==name and j["module"] in MODULES,"Exact logical worker")
        require(isinstance(j["args"],list) and all(isinstance(x,str) and "\0" not in x for x in j["args"]),"Literal worker arguments")
        require(set(j["env"]).issubset(ENV_KEYS) and all(isinstance(x,str) for x in j["env"].values()),"Only declared worker cache/thread environment")
        require(j["args"]==["--execute","--input",j["input_ref"]["path"],"--expected-input-sha256",j["input_ref"]["sha256"]],"Exact existing inert-dispatcher execution route")
        spec=json.loads(bound(j["input_ref"]));caps=json.loads(bound(j["cuda_caps_ref"]))
        require(spec["job_id"]==name and spec.get("status")=="active" and spec.get("physical_gpu")==cap["physical_gpu"],"Original root input job/device identity")
        require(caps["max_cuda_reserved_bytes"]==j["cuda_cap_bytes"],"Unchanged original worker Torch cap")
        pin=MODULE_PINS[j["module"]]
        expiry=dt.datetime.fromisoformat(spec["expires_at_utc"])
        require(expiry.tzinfo is not None and expiry.timestamp()<=cap["expires_unix"]+0.001,
            "Worker input authority cannot outlive pool execution authority")
        require(j["cpu_rss_cap_bytes"]==caps["max_cpu_rss_bytes"]==spec["max_cpu_rss_bytes"]==pin["cpu_bytes"]
            and j["operational_deadline_seconds"]==86400,"Original outer owned RSS and24h deadline limits")
        require(j["module_ref"]==dict(path=pin["path"],sha256=pin["sha256"])
            and j["cuda_cap_bytes"]==pin["cuda_bytes"] and j["margin_mib"]==2048,"Exact reviewed shim/cap plus2GiB margin")
        if j["module"].endswith("exec_v3"):
            require(spec.get("schema")=="C3_GNN_prefetch_isolated980_active_input_v1" and spec.get("phase")=="train"
                and spec.get("execution_gpu_uuid")==cap["gpu_uuid"],"Cache-C3 exact TRAIN and UUID")
        else:
            require(spec.get("schema")=="H050_direct_readout_root_input_v1" and spec.get("hypothesis_id")=="H050"
                and spec.get("arm")=="joint_four_arm" and spec.get("phase")=="train","Canonical H050 exact four-arm TRAIN")
        require(spec.get("operational_pool_source_ref")==ref(SOURCE)
            and spec.get("operational_launcher_ref")==ref(LAUNCHER)
            and spec.get("operational_pool_base_ref")==ref(BASE)
            and spec.get("operational_canonical_module_ref")==j["module_ref"],"Honest actual shared route in fresh input")
        base.integer(j["cuda_cap_bytes"],"original CUDA cap",1)
        base.integer(j["margin_mib"],"per-job non-Torch/driver margin",1)
        base.integer(j["log_limit_bytes"],"bounded logical job log",1)
        require(j["peak_mib"]==math.ceil(j["cuda_cap_bytes"]/MIB)+j["margin_mib"],"Separate cap plus margin reservation")
        base.integer(j["peak_mib"],"logical reservation",1)
        require(j["module_ref"]==ref("src/"+j["module"].replace(".","/")+".py"),"Actual original canonical worker source")
        storage=json.loads(bound(j["storage_ref"]))
        require(storage["job_id"]==name and spec["refs"]["storage"]==j["storage_ref"]
            and spec["max_cuda_reserved_bytes"]==j["cuda_cap_bytes"],"Original per-job storage/cap identity")
        require(all(r in storage["reservation"]["consumer_bindings"] for r in [ref(SOURCE),ref(LAUNCHER),ref(BASE),j["module_ref"]]),"Future per-job admission must bind actual shared launcher and base")
        require(j["module_ref"] in cap["source_refs"].values(),"Canonical worker included in bound closure")
    return cap

def socket_name(cap):
    # Abstract AF_UNIX creates no special filesystem node in a guarded scope.
    token=hashlib.sha256(base.encoded(cap["plan_ref"])).hexdigest()[:16]
    return "\0arggnn_gpu45_"+cap["pool"]+"_"+token

def current_guard():
    require(ref(GUARD)=={"path":GUARD,"sha256":GUARD_SHA},"Frozen active storage guard")
    bound(POLICY);bound(REPAIR)
    spec=importlib.util.spec_from_file_location("_arggnn_gpu45_current980",ROOT/GUARD)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.configure_v2(REPAIR,policy_ref=POLICY)
    return module

def guarded_check(guard,storage,required=0):
    # Direct original guard at each mutation. No replacement, memo wrapper or credit.
    return guard.budget_check_v9(storage,boundary=True,required_write_bytes=required)

class Shared45Controller(base.Controller):
    """Reuse base socket loop, lease validation, ownership and PIDFD signalling."""
    def __init__(self,cap_ref):
        self.cap_ref=cap_ref;self.cap=load_cap(cap_ref);cap=self.cap
        self.guard=current_guard();self.admin=json.loads(bound(cap["controller_storage_ref"]))
        require(self.admin["job_id"]==cap["controller_job_id"],"Separate controller storage job")
        self.root=ROOT;self.pool=cap["pool"];self.gpu=cap["physical_gpu"]
        self.pool_job="gpu45_shared_v1:"+self.pool+":gpu"+str(self.gpu)
        self.headroom=cap["headroom_mib"];self.inventory=base.real_inventory
        self.proc_reader=OwnedProcReader();self.python=str(ROOT/".runtime/env/bin/python")
        self.modules=MODULES
        self.directory=base.canonical_path(ROOT,self.admin["output_directory"])
        self.base=self.directory.parent;self.registry=self.directory/"registry.json"
        self.socket_path=socket_name(cap)
        require(len(os.fsencode(self.socket_path))<=107,"Bound abstract Unix endpoint size")
        self.lock=threading.RLock();self.lease=None;self.listener=None
        self.jobs={};self.children={};self.logs={};self.processes={};self.pid_jobs={}
        self.ambiguous=[];self.lifecycle="new";self.terminate_requested=False
        self.safety_error=None;self.memory_observation=None;self.shutdown_requested=False
        self.socket_inode=None;self.device=None;self.storage_account=None
        self.controller_identity=base.proc_identity(os.getpid());self.next_poll=0.
        self.orphan_waits=[]

    def _check(self,required=0):
        require(time.time()<self.cap["drain_expires_unix"],"Bounded owned drain authority expired")
        require(ref(SOURCE)==self.cap["source_refs"][SOURCE] and ref(BASE)==self.cap["source_refs"][BASE],"Admitted controller sources unchanged")
        self.storage_account=guarded_check(self.guard,self.admin,required)
        return self.storage_account

    def start(self):
        import socket
        with self.lock:
            require(self.lifecycle=="new","New controller required")
            base.enable_child_subreaper();self._check(64*1024)
            require(self.base.is_dir() and not self.directory.exists(),"Fresh controller output and existing parent")
            self.directory.mkdir(mode=0o700);self._check(16*1024)
            (self.directory/"jobs").mkdir(mode=0o700)
            self._check(8192)
            self.listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            self.listener.bind(self.socket_path)
            self.listener.listen(32);self.listener.setblocking(False)
            try:
                # Same EXCLUSIVE inode as every legacy launcher. Never LOCK_UN it.
                # Refuse any lingering forbidden device2/3 lease, without adopting it.
                with base.lock_file(ROOT,"scheduler.lock") as scheduler:
                    fcntl.flock(scheduler,fcntl.LOCK_EX)
                    for idx in (2,3):
                        with base.lock_file(ROOT,f"gpu_{idx}.lock") as h:
                            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            except BlockingIOError:raise RuntimeError("Forbidden legacy GPU2/3 lease remains active")
                    lease=base.lock_file(ROOT,f"gpu_{self.gpu}.lock")
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BaseException:lease.close();raise
                    self.lease=lease
                    rows=base.checked_inventory(self.inventory);r=rows[self.gpu]
                    require(r["uuid"]==self.cap["gpu_uuid"],"Root-bound actual physical UUID")
                    require(r["used_mib"]+self.headroom<r["total_mib"],"Actual free memory exceeds device reserve")
                    self.device=r;self.metadata=dict(launcher_pid=os.getpid(),gpu=self.gpu,uuid=r["uuid"],job=self.pool_job,started_unix=time.time(),storage_bytes=self.admin["reserved_new_bytes"],logical_sharing=True)
                    raw=base.encoded(self.metadata);self._check(len(raw)+8192)
                    os.ftruncate(lease.fileno(),len(raw));os.pwrite(lease.fileno(),raw,0);os.fsync(lease.fileno())
                    self.lease_sha=hashlib.sha256(raw).hexdigest()
                self.lifecycle="running";self._persist();return self
            except BaseException:
                self.lifecycle="startup_failed";self._release_handles();raise

    def status(self):
        # Base status needs only these fields, not the retired StorageLedger.
        s=super().status();s.update(schema="GPU45_shared_logical_pool_registry_v1",capability_ref=self.cap_ref,owned_orphan_waits=self.orphan_waits)
        return s

    def _persist(self):
        data=base.encoded(self.status())+b"\n"
        self._check(2*len(data)+16384);base.atomic_json(self.registry,self.status())

    def _process_scan(self):
        super()._process_scan()
        # Only this controller's subreaped tree is a supervision domain.
        # Foreign jobs are charged by NVML and never inspected, adopted or signalled.
        # Keep unknown owned zombies as ambiguity rather than declaring a drain.
        for pid,p in self.processes.items():
            if p.get("controller_descendant") and pid not in self.pid_jobs:
                matches=[name for name,j in self.jobs.items() for k in j["process_identities"]
                         if k["pid"]==pid and k["start_ticks"]==p["start_ticks"]]
                if len(set(matches))!=1 and pid not in self.ambiguous:self.ambiguous.append(pid)
        self._owned_resource_check()

    def _owned_resource_check(self):
        """Detect original outer limits without interrupting actual drain waits."""
        active=[(name,j) for name,j in self.jobs.items() if j["state"] in ACTIVE]
        if not active:return
        violations=[]
        if time.time()>=self.cap["expires_unix"]:
            violations.append("Pool execution authority expired; owned drain required")
        controller=self.processes.get(os.getpid())
        require(controller is not None and type(controller.get("rss_bytes")) is int,
            "Current owned controller RSS required")
        unknown=sum(p["rss_bytes"] for pid,p in self.processes.items()
            if p.get("controller_descendant") and pid not in self.pid_jobs and p["state"] not in {"Z","X"})
        for name,job in active:
            rss=controller["rss_bytes"]+unknown+sum(self.processes[pid]["rss_bytes"]
                for pid,assigned in self.pid_jobs.items() if assigned==name)
            job["observed_owned_RSS_bytes"]=rss
            job["peak_owned_RSS_bytes"]=max(job.get("peak_owned_RSS_bytes",0),rss)
            job["unattributed_owned_RSS_bytes"]=unknown
            if rss>job["requested"]["cpu_rss_cap_bytes"]:
                violations.append(name+": original owned process-tree RSS cap exceeded")
            require(type(job.get("started_monotonic")) in (int,float),"Actual monotonic owned launch time")
            if time.monotonic()-job["started_monotonic"]>=job["requested"]["operational_deadline_seconds"]:
                violations.append(name+": original86400s monotonic operational deadline exceeded")
        # Resource/expiry violations request exact-owned TERM/KILL but must not
        # throw out of discovery: repeated draining ticks must reach waitpid and
        # Popen.poll even while the violated deadline remains in the past.
        if violations:self._fail_owned_supervision("; ".join(violations))


    def _memory(self,incoming=0):
        rows=base.checked_inventory(self.inventory);r=rows[self.gpu]
        require(r["uuid"]==self.metadata["uuid"] and r["total_mib"]==self.device["total_mib"],"Actual physical UUID/total changed")
        active=[j for j in self.jobs.values() if j["state"] in ACTIVE]
        owned={};external=[]
        for a in r["compute"]:
            name=self.pid_jobs.get(a["pid"])
            if name is None:
                # Retain a draining job's reservation while NVML still lists a
                # historical owned PID, even after /proc no longer has it.
                possible={j["logical_job"] for j in active for p in j["process_identities"] if p["pid"]==a["pid"]}
                require(len(possible)<=1,"Historical compute PID belongs to multiple jobs")
                if possible:name=next(iter(possible))
            if name is None:external.append(dict(a))
            else:
                require(a["name"]==TITLE,"Owned CUDA PID lacks exact process title")
                owned[name]=owned.get(name,0)+a["used_mib"]
        reserved=sum(j["requested"]["peak_mib"] for j in active)
        # All external/driver usage is charged. Pending jobs keep their full peak.
        unattributed=max(self.device["used_mib"],r["used_mib"]-sum(owned.values()),0)
        projected=reserved+incoming+unattributed+self.headroom
        # A second independent check preserves all still-unallocated active peaks.
        unfilled=sum(max(0,j["requested"]["peak_mib"]-owned.get(j["logical_job"],0)) for j in active)
        actual_free_required=unfilled+incoming+self.headroom
        overruns=[n for n,u in owned.items() if u>self.jobs[n]["requested"]["peak_mib"]]
        reasons=[]
        if self.ambiguous:reasons.append("ambiguous_owned_processes")
        if overruns:reasons.append("owned_memory_cap_plus_margin_overrun")
        if projected>r["total_mib"] or r["total_mib"]-r["used_mib"]<actual_free_required:reasons.append("actual_free_memory_or_reserved_headroom")
        if incoming and len(active)>=self.cap["maximum_concurrent_jobs"]:reasons.append("logical_concurrency_limit")
        self.memory_observation=dict(observed=r,active_peak_mib=reserved,incoming_peak_mib=incoming,unattributed_mib=unattributed,projected_mib=projected,actual_free_required_mib=actual_free_required,external_compute=external,owned_compute_jobs=sorted(owned),overrun_jobs=overruns,blocked_by=reasons)
        return self.memory_observation

    def submit_declared(self,name):
        with self.lock:
            require(self.lifecycle=="running" and name in self.cap["jobs"] and name not in self.jobs,"Fresh predeclared logical job")
            self.tick(False);require(not self.safety_error,"Supervision error blocks new submission")
            declared=self.cap["jobs"][name]
            # Revalidate original input, source, cap and storage before queueing.
            load_cap(self.cap_ref)
            storage=json.loads(bound(declared["storage_ref"]));guarded_check(self.guard,storage)
            m=self._memory(declared["peak_mib"])
            self.jobs[name]=dict(logical_job=name,requested=declared,state="queued",submitted_unix=time.time(),admission=m,pid=None,leader_start_ticks=None,process_identities=[],exit_code=None,log_bytes=0,log_dropped_bytes=0,descendant_waits=[])
            self._persist();self._launch_queue();self._persist();return dict(self.jobs[name])

    def _launch_queue(self):
        if self.lifecycle!="running" or self.safety_error:return
        for name,job in self.jobs.items():
            if job["state"]!="queued":continue
            self._process_scan();m=self._memory(job["requested"]["peak_mib"]);job["admission"]=m
            if m["blocked_by"]:break
            load_cap(self.cap_ref);j=job["requested"]
            guarded_check(self.guard,json.loads(bound(j["storage_ref"])))
            self._lease_check();job.update(state="launching",started_unix=time.time(),started_monotonic=time.monotonic());self._persist()
            env=os.environ.copy();env.pop("PYTHONHOME",None);env.update(j["env"])
            env.update(CUDA_VISIBLE_DEVICES=self.metadata["uuid"],ARGGNN_PHYSICAL_GPU=str(self.gpu),ARGGNN_JOB_NAME=self.pool_job,ARGGNN_LOGICAL_JOB_NAME=name,ARGGNN_GPU_LEASE=str(ROOT/f".runtime/locks/gpu_{self.gpu}.lock"),ARGGNN_GPU_LEASE_FD=str(self.lease.fileno()),PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/"src")+os.pathsep+str(ROOT/"scripts"),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(POLICY))
            storage=json.loads(bound(j["storage_ref"]))
            env["ARGGNN_BUDGET_V9_SOURCE_SHA256"]=storage["budget_components"]["implementation"]["sha256"]
            env["ARGGNN_BUDGET_V9_LAUNCHER_SHA256"]=storage["budget_components"]["launcher"]["sha256"]
            argv=[self.python,"-B","-m","arggnn.gpu45_c3cache_h050_pool_v1","worker","--execute","--capability",self.cap_ref["path"],"--sha256",self.cap_ref["sha256"],"--job",name]
            log_path=self.directory/"jobs"/(name+".log");log=None;child=None
            try:
                self._check(j["log_limit_bytes"]+8192);log=log_path.open("xb")
                child=subprocess.Popen(argv,cwd=ROOT,env=env,shell=False,start_new_session=True,close_fds=True,pass_fds=(self.lease.fileno(),),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                os.set_blocking(child.stdout.fileno(),False);self.children[name]=child;self.logs[name]=log
                job.update(state="running",pid=child.pid,log_path=str(log_path),argv=argv)
                # Capture our direct Popen leader before owned-tree discovery.
                # A failed own scan retains the leader and reservation for drain.
                leader=base.proc_identity(child.pid)
                require(leader["uid"]==os.getuid() and leader["pid"]==child.pid
                    and leader["pgid"]==child.pid and leader["sid"]==child.pid,
                    "Exact owned leader UID/start/session before supervision")
                job["leader_start_ticks"]=leader["start_ticks"]
                job["process_identities"]=[dict(leader)]
                self.processes[child.pid]=dict(leader);self.pid_jobs[child.pid]=name
                self._process_scan()
            except Exception as error:
                if child is not None:
                    self.children[name]=child;self.logs[name]=log;job.update(state="draining",pid=child.pid,error=str(error))
                    self._fail_owned_supervision("Post-spawn supervision failed: "+str(error))
                    self._service_termination();break
                if log is not None:log.close()
                job.update(state="failed",error=str(error),finished_unix=time.time())

    def _drain_logs(self):
        for name,child in list(self.children.items()):
            stream=child.stdout
            if stream is None or stream.closed:continue
            job=self.jobs[name]
            for _ in range(16):
                try:block=os.read(stream.fileno(),65536)
                except BlockingIOError:break
                if not block:stream.close();break
                keep=block[:max(0,job["requested"]["log_limit_bytes"]-job["log_bytes"])]
                if keep:
                    self._check(len(keep)+8192);self.logs[name].write(keep);self.logs[name].flush();job["log_bytes"]+=len(keep)
                job["log_dropped_bytes"]+=len(block)-len(keep)

    def _reap_owned_adopted(self):
        leaders={c.pid for c in self.children.values()}
        for pid,p in list(self.processes.items()):
            if pid in leaders or p["ppid"]!=os.getpid() or p["state"] not in {"Z","X"}:continue
            names={name for name,j in self.jobs.items() for k in j["process_identities"] if k["pid"]==pid and k["start_ticks"]==p["start_ticks"]}
            require(len(names)==1,"Unattributed adopted child prevents complete wait claim")
            waited,status=os.waitpid(pid,os.WNOHANG)
            if waited:
                row=dict(pid=pid,start_ticks=p["start_ticks"],raw_wait_status=status,returncode=os.waitstatus_to_exitcode(status),wait_status_known=True)
                self.orphan_waits.append(row);self.jobs[next(iter(names))]["descendant_waits"].append(row)

    def _fail_owned_supervision(self,message):
        self.safety_error=message;self.lifecycle="draining";self.terminate_requested=True
        for job in self.jobs.values():
            if job["state"]=="queued":
                job.update(state="cancelled",finished_unix=time.time(),error=message)
            elif job["state"] in ACTIVE:
                job.update(state="draining",resource_abort=message)

    def _service_termination(self):
        # This path never needs a host scan, memory inventory or storage write.
        # Failure to supervise blocks new work but cannot suppress a known
        # owned cancellation. Unknown/opaque identities are never adopted.
        if time.time()>=self.cap["drain_expires_unix"]:
            self.termination_authority_expired=True
            return
        for name,job in list(self.jobs.items()):
            if name not in self.children:continue
            if self.terminate_requested or job.get("cancel_requested") or job.get("resource_abort"):
                try:self._signal_job(name)
                except Exception as error:
                    job["termination_service_error"]=f"{type(error).__name__}: {error}"[:2048]

    def tick(self,launch_queued=True):
        with self.lock:
            if self.lifecycle not in {"running","draining"}:return
            # Explicit cancel/shutdown TERM is attempted before slow/failing
            # scans. Repeated ticks provide bounded owned SIGKILL escalation.
            self._service_termination()
            try:
                if time.time()>=self.cap["expires_unix"] and self.lifecycle=="running":
                    self.lifecycle="draining"
                    for job in self.jobs.values():
                        if job["state"]=="queued":job.update(state="cancelled",finished_unix=time.time(),error="Pool submission authority expired; owned drain continues")
                self._lease_check();self._drain_logs();self._process_scan();self._reap_owned_adopted();memory=self._memory()
                self.safety_error=None
                if memory["overrun_jobs"]:
                    self._fail_owned_supervision("Owned GPU reservation overrun; drain required")
                for name,child in list(self.children.items()):
                    job=self.jobs[name];exit_code=child.poll()
                    if exit_code is not None:
                        job.update(exit_code=exit_code,state="draining",leader_wait_status_known=True)
                        self._process_scan();self._reap_owned_adopted();memory=self._memory()
                        live=name in self.pid_jobs.values()
                        own_compute=name in memory["owned_compute_jobs"]
                        if not live and not own_compute and not self.ambiguous:
                            descendant_ok=all(r["returncode"]==0 for r in job["descendant_waits"])
                            self._drain_logs();job.update(state="succeeded" if exit_code==0 and descendant_ok and not job.get("resource_abort") and not job.get("cancel_requested") else "failed",finished_unix=time.time(),owned_drain_observed=True)
                            if child.stdout is not None:child.stdout.close()
                            self.logs.pop(name).close();del self.children[name]
                if launch_queued:self._launch_queue()
                if self.lifecycle=="draining" and not self.children and not self.ambiguous:self.lifecycle="closed"
            except Exception as error:
                self._fail_owned_supervision(f"{type(error).__name__}: {error}")
            finally:
                # Includes scan/guard errors. Do not depend on persistence to
                # send a previously authorised, exact owned PIDFD signal.
                self._service_termination()
            try:self._persist()
            except Exception as error:
                self._fail_owned_supervision("Controller persistence failed: "+f"{type(error).__name__}: {error}")
                self._service_termination()

    def cancel_declared(self,name):
        with self.lock:
            require(name in self.jobs,"Known logical job")
            job=self.jobs[name]
            if job["state"]=="queued":
                job.update(state="cancelled",finished_unix=time.time(),error="Explicit logical cancellation")
            elif job["state"] in ACTIVE:job["cancel_requested"]=True
            self.tick(False);return dict(job)

    def _signal_job(self,name):
        require(self.cap.get("owned_termination_escalation_authorized") is True,"Explicit owned termination authority")
        job=self.jobs[name]
        # Only source-owned identities already captured for this logical job,
        # including the direct Popen leader and known descendants. No title,
        # unrelated PID, process-group broadcast or new ancestry inference.
        identities={(p["pid"],p["start_ticks"],p["uid"]):dict(p)
            for p in job["process_identities"]}
        for pid,assigned in list(self.pid_jobs.items()):
            if assigned==name and pid in self.processes:
                p=self.processes[pid];identities[(pid,p["start_ticks"],p["uid"])]=dict(p)
        attempts=job.setdefault("owned_termination",{})
        for (pid,start,uid),saved in sorted(identities.items()):
            if time.time()>=self.cap["drain_expires_unix"]:
                self.termination_authority_expired=True;return
            if uid!=os.getuid():continue
            key=str(pid)+":"+str(start);prior=attempts.get(key,{})
            elapsed=time.monotonic()-prior.get("TERM_sent_monotonic",time.monotonic())
            if prior.get("KILL_sent_monotonic") is not None:continue
            if prior.get("TERM_sent_monotonic") is not None:
                if elapsed<self.cap["termination_grace_seconds"]:continue
                signum=signal.SIGKILL
            else:signum=signal.SIGTERM
            try:
                fd=base.open_pidfd(pid)
                try:
                    now=base.proc_identity(pid)
                    if (now["uid"],now["start_ticks"])!=(uid,start):
                        prior["identity_mismatch_observed"]=True;attempts[key]=prior;continue
                    base.send_pidfd(fd,signum)
                    label="KILL" if signum==signal.SIGKILL else "TERM"
                    prior.update(pid=pid,start_ticks=start,uid=uid)
                    prior[label+"_sent_monotonic"]=time.monotonic()
                    prior[label+"_sent_unix"]=time.time()
                    attempts[key]=prior
                finally:os.close(fd)
            except (FileNotFoundError,ProcessLookupError):
                # This is not a wait/drain receipt and releases nothing.
                prior["target_missing_observed"]=True;attempts[key]=prior
            except Exception as error:
                prior["last_signal_error"]=f"{type(error).__name__}: {error}"[:2048]
                attempts[key]=prior

    def _signal_own(self):
        # Preserve base shutdown semantics through the same exact-owned path.
        for name in list(self.children):self._signal_job(name)


    def dispatch(self,r):
        if isinstance(r,dict) and set(r)=={"v","op","job"} and type(r["v"]) is int and r["v"]==1:
            if r["op"]=="submit":return self.submit_declared(r["job"])
            if r["op"]=="cancel":return self.cancel_declared(r["job"])
        return super().dispatch(r)

    def _release_handles(self):
        # Abstract endpoint disappears when its owned listener closes. No unlink.
        if self.listener is not None:self.listener.close();self.listener=None
        if self.lease is not None:self.lease.close();self.lease=None
        # Never LOCK_UN: inherited worker descriptors retain the physical lease.

def worker(cap_ref,name):
    cap=load_cap(cap_ref);require(name in cap["jobs"],"Declared original worker")
    j=cap["jobs"][name]
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    lease=validate_lease();apply_and_validate_title()
    require(lease["physical_gpu"]==cap["physical_gpu"] and lease["visible_gpu_uuid"]==cap["gpu_uuid"]
        and os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==name
        and lease["launcher_metadata"]["job"]=="gpu45_shared_v1:"+cap["pool"]+":gpu"+str(cap["physical_gpu"]),"Inherited pool lease and exact logical identity before scientific imports")
    guard=current_guard();guarded_check(guard,json.loads(bound(j["storage_ref"])))
    # Original scientific dispatchers retain original guards and per-process cap.
    module=importlib.import_module(j["module"])
    return module.main(j["args"])

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("start","submit","run","cancel","status","wait","shutdown","worker"),nargs="?")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--capability");parser.add_argument("--sha256");parser.add_argument("--job")
    parser.add_argument("--terminate",action="store_true")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_activated",source=SOURCE,physical_GPUs=[4,5],legacy_exclusive_coexistence=False,external_job_sharing_authorized_only_by_fresh_root_capability=True)));return 0
    require(args.capability and args.sha256 and args.command,"Pinned active capability and explicit operation")
    cap_ref={"path":args.capability,"sha256":args.sha256};cap=load_cap(cap_ref,allow_drain=args.command in {"status","wait","shutdown","cancel"})
    if args.command=="worker":return worker(cap_ref,args.job)
    socket_path=socket_name(cap)
    if args.command=="start":
        controller=Shared45Controller(cap_ref).start()
        def stop(signum,frame):controller.shutdown_requested=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        print(json.dumps(controller.status(),sort_keys=True),flush=True);controller.run();return 0
    if args.command=="cancel":
        answer=base.request(socket_path,dict(v=1,op="cancel",job=args.job));print(json.dumps(answer,sort_keys=True),flush=True);return 0
    interrupted=[False]
    if args.command=="run":
        def cancel_signal(signum,frame):interrupted[0]=True
        signal.signal(signal.SIGTERM,cancel_signal);signal.signal(signal.SIGINT,cancel_signal)
    if args.command in {"submit","run"}:
        require(args.job in cap["jobs"],"Declared job")
        answer=base.request(socket_path,dict(v=1,op="submit",job=args.job));print(json.dumps(answer,sort_keys=True),flush=True)
        if args.command=="submit":return 0
    if args.command=="shutdown":
        answer=base.request(socket_path,dict(v=1,op="shutdown",terminate=args.terminate));print(json.dumps(answer,sort_keys=True),flush=True);return 0
    cancellation_sent=False
    while True:
        if interrupted[0] and not cancellation_sent:
            base.request(socket_path,dict(v=1,op="cancel",job=args.job));cancellation_sent=True
        answer=base.request(socket_path,dict(v=1,op="status"))
        if args.command=="status":print(json.dumps(answer,sort_keys=True));return 0
        jobs={j["logical_job"]:j for j in answer["jobs"]};require(args.job in jobs,"Submitted job")
        job=jobs[args.job]
        if job["state"] in base.TERMINAL:
            print(json.dumps(job,sort_keys=True),flush=True)
            return job["exit_code"] if job["state"]=="succeeded" or type(job.get("exit_code")) is int and job["exit_code"]!=0 else 2
        time.sleep(1)

if __name__=="__main__":
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}",file=sys.stderr);raise SystemExit(2)

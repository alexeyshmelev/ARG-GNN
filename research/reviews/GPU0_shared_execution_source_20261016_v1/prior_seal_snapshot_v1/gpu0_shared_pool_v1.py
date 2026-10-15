"""GPU0-only execution facade over the frozen reviewed logical-pool lifecycle.

No inventory, process, lock, socket, guard or CUDA probe occurs on import.
All live operations require fresh root-bound capability and separate charges.
"""
from __future__ import annotations
from pathlib import Path
import ast,hashlib,importlib.util,json,math,os,sys,time,types

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_shared_execution_source_20261016_v1"
SOURCE=HERE+"/gpu0_shared_pool_v1.py"
LAUNCHER=HERE+"/launch_gpu0_shared_v1.py"
RUNTIME=HERE+"/gpu0_runtime_adapter_v1.py"
GUARD0=HERE+"/gpu0_native_guard_v1.py"
OLD="src/arggnn/gpu45_c3cache_h050_pool_v1.py"
OLD_SHA="d3279c6208737371031330f4bef9325199ba20c86f0f7d8e930c134b128d29fb"
GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(name):
    p=ROOT/name
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),"Canonical bound source")
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def load(record,label):
    require(ref(record["path"])==record,"Exact source bytes")
    sp=importlib.util.spec_from_file_location(label,ROOT/record["path"])
    m=importlib.util.module_from_spec(sp);sys.modules[label]=m;sp.loader.exec_module(m);return m

require(ref("src/arggnn/gpu_pool_v1.py")==dict(path="src/arggnn/gpu_pool_v1.py",sha256="ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1"),"Pinned canonical base before import")
legacy=load(dict(path=OLD,sha256=OLD_SHA),"_GPU0_private_frozen_shared_lifecycle")
# Interpreter-local facade, never mutate the imported canonical base module.
base=types.SimpleNamespace(**vars(legacy.base))
source=(ROOT/legacy.BASE).read_text()
tree=ast.parse(source)
checked=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="checked_inventory")
ns=dict(vars(legacy.base),ALLOWED_GPUS=(0,))
exec(compile(ast.Module(body=[checked],type_ignores=[]),"<frozen_inventory_GPU0_eligibility_only>","exec"),ns)
base.checked_inventory=ns["checked_inventory"]
legacy.base=base;legacy.SOURCE=SOURCE;legacy.LAUNCHER=LAUNCHER
legacy.RENEWAL_BOOTSTRAP_REF={'path': 'research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_bootstrap_renewal_v1.py', 'sha256': '58fc5e27f5cdc02f70463314068160299981778cf0196d1e77c0577ca27cfcc1'}
CHECKER_REF={'path': 'research/reviews/GPU45_TRAIN_controller_execution_override_20261016_v1/controller_metadata_scope_check_v1.py', 'sha256': '2621353de0dc56f328f7f308ba17a341425ef2fc02d54739b6ccfc7d1a6ce2cf'}
bound=legacy.bound
MIB=2**20
ACTIVE=legacy.ACTIVE
TITLE=legacy.TITLE

def guard_module():
    return load(dict(path=GUARD0,sha256="3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69"),"_GPU0_fresh_guard_for_pool")

def load_cap(record,allow_drain=False):
    cap=json.loads(bound(record));g=guard_module();policy=g.validate_policy_v1()
    require(Path(__file__).resolve()==ROOT/SOURCE,"Exact additive route path")
    require(cap.get("schema")=="GPU0_shared_logical_pool_capability_v1" and cap.get("status")=="active"
        and cap.get("root_authorizer")=="root_agent" and type(cap.get("physical_gpu")) is int
        and cap["physical_gpu"]==0 and cap.get("gpu_uuid")==GPU0_UUID
        and cap.get("maximum_distinct_project_GPUs")==1 and cap.get("maximum_concurrent_jobs")==2
        and cap.get("logical_job_sharing_authorized") is True and cap.get("external_job_sharing_authorized") is True
        and cap.get("legacy_exclusive_coexistence_allowed") is False,"Fresh exact GPU0 root capability")
    require(cap.get("GPU0_policy_ref")==g.POLICY_REF and cap.get("stop_evidence_ref")==policy["stop_evidence_ref"],"Actual policy/transition lineage")
    bound(cap["stop_evidence_ref"])
    certificate=legacy.install_marker_renewal(cap["marker_repair_renewal_ref"])
    expiry=legacy.dt.datetime.fromisoformat(certificate["renewed_expires_at_utc"].replace("Z","+00:00")).timestamp()
    require(time.time()<(cap["drain_expires_unix"] if allow_drain else cap["expires_unix"])
        and cap["drain_expires_unix"]<expiry,"Bounded pool authority inside renewal")
    require(cap.get("owned_termination_escalation_authorized") is True and cap.get("owned_drain_after_expiry_authorized") is True,"Owned-only drain/escalation authority")
    base.integer(cap["termination_grace_seconds"],"TERM grace",1)
    require(cap["termination_grace_seconds"]<=120 and cap["drain_expires_unix"]>=cap["expires_unix"]+cap["termination_grace_seconds"]+5,"Original bounded TERM/drain limits")
    require(base.NAME.fullmatch(cap["pool"]) and len(cap["pool"])<=32 and cap.get("headroom_mib")==4096,"Safe name and unchanged4GiB headroom")
    sources=cap["source_refs"]
    required=[ref(SOURCE),ref(LAUNCHER),ref(RUNTIME),ref(GUARD0),ref(OLD),ref(legacy.BASE),
        g.POLICY_REF,cap["stop_evidence_ref"],ref(DOWN),legacy.RENEWAL_SOURCE_REF,legacy.RENEWAL_BOOTSTRAP_REF,cap["marker_repair_renewal_ref"]]
    require(ref(legacy.BASE)==dict(path=legacy.BASE,sha256=legacy.BASE_SHA),"Frozen base lifecycle")
    for path,r in sources.items():require(path==r["path"],"Closure key/path");bound(r)
    require(all(r in sources.values() for r in required+[CHECKER_REF]),"Actual new/frozen lifecycle/guard/policy/checker closure")
    plan=json.loads(bound(cap["plan_ref"]))
    fields=("pool","physical_gpu","gpu_uuid","maximum_concurrent_jobs","headroom_mib","termination_grace_seconds","jobs","marker_repair_renewal_ref","GPU0_policy_ref","stop_evidence_ref")
    require(plan.get("schema")=="GPU0_shared_logical_pool_plan_v1" and all(plan[k]==cap[k] for k in fields),"Exact pre-admission plan")
    admin=json.loads(bound(cap["controller_storage_ref"]))
    require(admin["job_id"]==cap["controller_job_id"],"Separate controller storage identity")
    require(all(r in admin["reservation"]["consumer_bindings"] for r in required+[CHECKER_REF,cap["plan_ref"],dict(path=legacy.GUARD,sha256=legacy.GUARD_SHA),legacy.POLICY,legacy.REPAIR]),"Controller unchanged guard closure")
    require(isinstance(cap["jobs"],dict) and 1<=len(cap["jobs"])<=3,"Only finite declared H050/C3/native jobs")
    ids={admin["job_id"]};outputs={admin["output_directory"]};kinds=set()
    for name,j in cap["jobs"].items():
        kind=j["worker_kind"]
        require(kind in ("H050","C3","native_ARGformer") and kind not in kinds and base.NAME.fullmatch(name) and j["job_id"]==name,"One job per declared kind")
        kinds.add(kind)
        caps=json.loads(bound(j["cuda_caps_ref"]));spec=json.loads(bound(j["input_ref"]));storage=json.loads(bound(j["storage_ref"]))
        require(j["runtime_entry_ref"] in sources.values() and j["module_ref"] in sources.values(),"Exact actual runtime/science entry refs")
        bound(j["runtime_entry_ref"]);bound(j["module_ref"])
        if kind in ("H050","C3"):
            module="arggnn.H050_direct_readout_cut_memory_exec_v1" if kind=="H050" else "arggnn.american_c3_gnn_prefetch_ramcache_isolated980_exec_v3"
            pin=legacy.MODULE_PINS[module]
            require(j["module"]==module and j["module_ref"]==dict(path=pin["path"],sha256=pin["sha256"])
                and j["runtime_entry_ref"]==ref(RUNTIME),"Frozen H050/C3 algorithm entry plus actual GPU0 adapter")
            gpu_cap=pin["cuda_bytes"];cpu_cap=pin["cpu_bytes"]
            require(spec.get("phase")=="train" and spec.get("physical_gpu")==0,"Only fresh GPU0 TRAIN")
            require(j["args"]==["--execute","--input",j["input_ref"]["path"],"--expected-input-sha256",j["input_ref"]["sha256"]],"Exact original literal dispatcher args")
        else:
            require(cap.get("native_40GiB_operational_ceiling_authorized") is True
                and j["runtime_entry_ref"]["path"].startswith("research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/"),"Explicit root-reviewed native40GiB operational extension")
            gpu_cap=40*2**30;cpu_cap=j["cpu_rss_cap_bytes"]
            require(type(cpu_cap) is int and cpu_cap>0,"Original native RSS cap supplied/authenticated in native input")
        require(j["cuda_cap_bytes"]==caps["max_cuda_reserved_bytes"]==spec["max_cuda_reserved_bytes"]==gpu_cap
            and j["cpu_rss_cap_bytes"]==caps["max_cpu_rss_bytes"]==spec["max_cpu_rss_bytes"]==cpu_cap
            and j["margin_mib"]==2048,"Unchanged declared per-job CUDA/RSS cap and margin")
        deadline=j["operational_deadline_seconds"]
        require(type(deadline) is int and ((kind!="native_ARGformer" and deadline==86400) or (kind=="native_ARGformer" and 0<deadline<=259200 and (deadline<=86400 or cap.get("native_72h_operational_deadline_authorized") is True))),"Original H050/C3 24h or explicitly reviewed native<=72h deadline")
        require(j["peak_mib"]==math.ceil(gpu_cap/MIB)+2048,"Separate logical memory reservation")
        base.integer(j["log_limit_bytes"],"bounded logical log",1)
        require(set(j["env"]).issubset(legacy.ENV_KEYS) and all(isinstance(v,str) for v in j["env"].values()),"Only original private cache/thread environment")
        end=legacy.dt.datetime.fromisoformat(spec["expires_at_utc"])
        require(end.tzinfo is not None and time.time()<end.timestamp()<=cap["expires_unix"]+.001,"Worker authority within pool")
        require(spec.get("status")=="active" and spec.get("root_authorizer")=="root_agent" and spec["job_id"]==name
            and spec.get("operational_pool_source_ref")==ref(SOURCE) and spec.get("operational_launcher_ref")==ref(LAUNCHER)
            and spec.get("operational_pool_base_ref")==ref(legacy.BASE) and spec.get("operational_GPU0_runtime_ref")==j["runtime_entry_ref"]
            and spec.get("operational_GPU0_guard_ref")==ref(GUARD0) and spec.get("operational_GPU0_policy_ref")==g.POLICY_REF
            and spec.get("marker_repair_renewal_ref")==cap["marker_repair_renewal_ref"],"Honest actual new operational lineage")
        require(storage["job_id"]==name and spec["refs"]["storage"]==j["storage_ref"] and name not in ids
            and storage["output_directory"]==spec["output_directory"] and storage["output_directory"] not in outputs,"Separate per-job charge/output")
        ids.add(name);outputs.add(storage["output_directory"])
        bindings=storage["reservation"]["consumer_bindings"]
        if kind=="native_ARGformer":
            transitive=spec["transitive_source_closure_ref"]
            require(transitive in bindings and j["runtime_entry_ref"] in bindings and j["module_ref"] in bindings,
                "Native original direct transitive closure plus actual entry/science bindings")
            source_closure=json.loads(bound(transitive))
            require(source_closure.get("source_sha256")==spec.get("source_sha256")
                and all(source_closure["source_sha256"].get(r["path"])==r["sha256"] for r in required),
                "Exact native existing transitive path/SHA closure for every pool requirement,64 direct limit preserved")
        else:
            require(all(r in bindings for r in required+[j["runtime_entry_ref"],j["module_ref"]]),"Actual per-job unchanged guard source closure")
    return cap

legacy.load_cap=load_cap

def controller_request(socket_path,value):
    import socket
    data=base.encoded(value)+b"\n";require(len(data)<=base.MAX_REQUEST,"Bounded request")
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(300);client.connect(str(socket_path));client.sendall(data);reply=bytearray()
        while b"\n" not in reply:
            block=client.recv(65536);require(bool(block),"Missing complete actual controller reply")
            reply.extend(block);require(len(reply)<=base.MAX_REPLY,"Bounded response")
    answer=json.loads(bytes(reply).split(b"\n",1)[0]);require(answer.get("ok"),answer.get("error","Rejected"))
    return answer["result"]

def stale_compute_classification_v1(app,pid_jobs,processes,jobs):
    """Pure ownership rule. Exited/reused historical PID bytes remain external."""
    name=pid_jobs.get(app["pid"])
    if name is not None:return name,False
    saved=[p for j in jobs for p in j["process_identities"] if p["pid"]==app["pid"]]
    current=processes.get(app["pid"])
    matching=[p for p in saved if current and current["uid"]==p["uid"] and current["start_ticks"]==p["start_ticks"] and current["state"] not in {"Z","X"}]
    require(not matching,"Live owned compute PID lost logical attribution")
    return None,bool(saved)

class Shared0Controller(legacy.Shared45Controller):
    def __init__(self,cap_ref):
        super().__init__(cap_ref)
        self.pool_job="gpu0_shared_v1:"+self.pool+":gpu0"
        self.modules=frozenset(j["module"] for j in self.cap["jobs"].values())
        token=hashlib.sha256(base.encoded(self.cap["plan_ref"])).hexdigest()[:16]
        self.socket_path="\0arggnn_gpu0_"+self.pool+"_"+token

    def _check(self,required=0):
        require(time.time()<self.cap["drain_expires_unix"],"Bounded owned drain expiry")
        require(ref(SOURCE)==self.cap["source_refs"][SOURCE] and ref(OLD)==dict(path=OLD,sha256=OLD_SHA)
            and ref(legacy.BASE)==dict(path=legacy.BASE,sha256=legacy.BASE_SHA)
            and ref(CHECKER_REF["path"])==CHECKER_REF,"Actual new/frozen controller sources unchanged")
        guard_module().validate_policy_v1()
        if not hasattr(self,"_isolated_controller_metadata"):
            checker=load(CHECKER_REF,"_GPU0_reviewed_isolated_metadata_checker")
            self._isolated_controller_metadata=checker.ControllerMetadataScopeV1(self.guard,self.admin)
        self.storage_account=self._isolated_controller_metadata.check(required)
        return self.storage_account

    def status(self):
        value=super().status();value["schema"]="GPU0_shared_logical_pool_registry_v1"
        return value

    def _memory(self,incoming=0):
        r=base.checked_inventory(self.inventory)[0]
        require(r["uuid"]==self.metadata["uuid"]==GPU0_UUID,"Stable actual GPU0 UUID")
        active=[j for j in self.jobs.values() if j["state"] in ACTIVE]
        owned={};external=[];stale=[]
        for a in r["compute"]:
            name,is_stale=stale_compute_classification_v1(a,self.pid_jobs,self.processes,active)
            if name is None:
                external.append(dict(a))
                if is_stale:stale.append(dict(a,reason="owned_identity_exited_or_reused"))
            else:
                require(a["name"]==TITLE,"Live owned CUDA PID exact title")
                owned[name]=owned.get(name,0)+a["used_mib"]
        reserved=sum(j["requested"]["peak_mib"] for j in active)
        unattributed=max(self.device["used_mib"],r["used_mib"]-sum(owned.values()),0)
        projected=reserved+incoming+unattributed+self.headroom
        unfilled=sum(max(0,j["requested"]["peak_mib"]-owned.get(j["logical_job"],0)) for j in active)
        actual_free_required=unfilled+incoming+self.headroom
        overruns=[n for n,u in owned.items() if u>self.jobs[n]["requested"]["peak_mib"]]
        reasons=[]
        if self.ambiguous:reasons.append("ambiguous_owned_processes")
        if overruns:reasons.append("owned_memory_cap_plus_margin_overrun")
        if projected>r["total_mib"] or r["total_mib"]-r["used_mib"]<actual_free_required:reasons.append("actual_free_memory_or_reserved_headroom")
        if incoming and len(active)>=self.cap["maximum_concurrent_jobs"]:reasons.append("logical_concurrency_limit")
        self.memory_observation=dict(observed=r,active_peak_mib=reserved,incoming_peak_mib=incoming,unattributed_mib=unattributed,projected_mib=projected,actual_free_required_mib=actual_free_required,external_compute=external,stale_historical_compute=stale,owned_compute_jobs=sorted(owned),overrun_jobs=overruns,blocked_by=reasons)
        return self.memory_observation

# Exact narrow inherited method adaptations. Scientific code is not involved.
def _method(source,cls,name,replacements):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==cls)
    node=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
    text=ast.get_source_segment(source,node)
    for before,after in replacements:
        require(text.count(before)==1,"Single reviewed operational adaptation: "+name);text=text.replace(before,after)
    namespace=dict(vars(legacy))
    exec(compile(text,"<GPU0_operational_"+name+">","exec"),namespace)
    return namespace[name]

DOWN="research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py"
# This reviewed upstream method already moves the guard check before scheduler flock.
require(ref(DOWN)==dict(path=DOWN,sha256='07d930aba589c16be8bd8c97a7948e519e108def97bc7538bb061bf634c6a5ce'),"Frozen reviewed startup fix source")
down=(ROOT/DOWN).read_text()
start=_method(down,"Shared45Controller","start",[
    ('for idx in (2,3):\n                        with base.lock_file(ROOT,f"gpu_{idx}.lock") as h:\n                            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)\n                            except BlockingIOError:raise RuntimeError("Forbidden legacy GPU2/3 lease remains active")\n                    ',"")])
Shared0Controller.start=start
old=(ROOT/OLD).read_text()
Shared0Controller._launch_queue=_method(old,"Shared45Controller","_launch_queue",[
    ('"arggnn.gpu45_c3cache_h050_pool_v1"','"research.reviews.GPU0_shared_execution_source_20261016_v1.gpu0_shared_pool_v1"')])
# Preserve original peer-credential selector/service loop and reset only reply clock.
baseclass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Controller")
run=next(n for n in baseclass.body if isinstance(n,ast.FunctionDef) and n.name=="run")
runtext=ast.get_source_segment(source,run)
before='state["output"] = encoded(answer) + b"\\n"'
require(runtext.count(before)==1,"One exact post-dispatch output assignment")
runtext=runtext.replace(before,before+'\n'+' '*32+'state["deadline"] = time.monotonic() + 30')
run_ns=dict(vars(legacy.base));exec(compile(runtext,"<original_server_reply_clock_reset_only>","exec"),run_ns)
Shared0Controller.run=run_ns["run"]

def worker(cap_ref,name):
    cap=load_cap(cap_ref);require(name in cap["jobs"],"Declared logical job")
    j=cap["jobs"][name];g=guard_module();g.install_before_torch_v1()
    lease=g.validate_lease();g.apply_and_validate_title()
    require(lease["physical_gpu"]==0 and lease["visible_gpu_uuid"]==GPU0_UUID
        and os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==name
        and lease["launcher_metadata"]["job"]=="gpu0_shared_v1:"+cap["pool"]+":gpu0","Exact inherited pool lease/logical job/title")
    legacy.guarded_check(legacy.current_guard(),json.loads(bound(j["storage_ref"])))
    runtime=load(j["runtime_entry_ref"],"_GPU0_actual_logical_runtime")
    return runtime.invoke_job_v1(j)

# Keep original CLI/cancel/actual-wait semantics, with the new controller/worker.
mainnode=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=="main")
maintext=ast.get_source_segment(old,mainnode).replace("Shared45Controller(","Shared0Controller(").replace("base.request(","controller_request(").replace("physical_GPUs=[4,5]","physical_GPUs=[0]")
main_ns=dict(vars(legacy),Shared0Controller=Shared0Controller,worker=worker,controller_request=controller_request)
exec(compile(maintext,"<frozen_CLI_GPU0_route_only>","exec"),main_ns)
main=main_ns["main"]
if __name__=="__main__":raise SystemExit(main())

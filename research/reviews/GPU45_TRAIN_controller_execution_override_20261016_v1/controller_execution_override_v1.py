"""Additive TRAIN controller execution; original admitted workers stay byte-identical."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,importlib.util,json,signal,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU45_TRAIN_controller_execution_override_20261016_v1"
SOURCE=HERE+"/controller_execution_override_v1.py"
CHECKER=HERE+"/controller_metadata_scope_check_v1.py"
ORIGINAL_REF=dict(path="src/arggnn/gpu45_c3cache_h050_pool_v1.py",sha256="d3279c6208737371031330f4bef9325199ba20c86f0f7d8e930c134b128d29fb")
BASE_REF=dict(path="src/arggnn/gpu_pool_v1.py",sha256="ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1")

def require(ok,message):
    if not ok:raise ValueError(message)

def path(name):
    if isinstance(name,Path) and name.is_absolute():name=str(name.relative_to(ROOT))
    require(isinstance(name,str) and not Path(name).is_absolute() and ".." not in Path(name).parts,"Contained relative metadata path")
    p=ROOT/name;require(p.resolve()==p and p.is_relative_to(ROOT) and not p.is_symlink(),"Canonical metadata");return p

def ref(name):
    p=path(name);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record):
    require(ref(record["path"])==record,"Exact immutable metadata/source SHA")
    raw=path(record["path"]).read_bytes();require(len(raw)<=33554432,"Bounded metadata");return json.loads(raw)

def save(folder,name,value):
    p=folder/name;require(not p.exists() and p.parent.resolve().is_relative_to(ROOT),"Fresh root metadata only")
    p.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n");return ref(p)

def load(record,label):
    require(ref(record["path"])==record,"Exact reviewed source bytes")
    sp=importlib.util.spec_from_file_location(label,ROOT/record["path"]);m=importlib.util.module_from_spec(sp);sys.modules[label]=m;sp.loader.exec_module(m);return m

def original():
    require(ref(BASE_REF["path"])==BASE_REF,"Original controller base unchanged")
    return load(ORIGINAL_REF,"_GPU45_original_TRAIN_pool_for_execution_override_v1")

def adapted_start_source_v1():
    require(ref(ORIGINAL_REF["path"])==ORIGINAL_REF,"Frozen original TRAIN controller")
    s=path(ORIGINAL_REF["path"]).read_text();cls=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=="Shared45Controller")
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="start")
    text=ast.get_source_segment(s,node)
    before='                with base.lock_file(ROOT,"scheduler.lock") as scheduler:'
    after='''                # Complete the original prospective own-write guard BEFORE the
                # scheduler flock; that guard acquires this same lock by design.
                self._check(64*1024)
                with base.lock_file(ROOT,"scheduler.lock") as scheduler:'''
    require(text.count(before)==1,"Exact outer scheduler-lock boundary");text=text.replace(before,after)
    before='self._check(len(raw)+8192)'
    after='require(len(raw)+8192<=64*1024,"Lease metadata fits pre-lock prospective64KiB allowance")'
    require(text.count(before)==1,"Exact nested guard deadlock boundary");text=text.replace(before,after)
    compile(text,"<original_TRAIN_start_guard_before_scheduler_lock>","exec")
    return text

def validate_override_cap_v1(record,p):
    cap=p.load_cap(record)
    require(cap.get("controller_execution_override_ref")==ref(SOURCE)
        and cap.get("controller_metadata_scope_checker_ref")==ref(CHECKER)
        and cap.get("original_canonical_pool_and_workers_unchanged") is True,
        "Explicit exact root controller-execution override")
    storage=read(cap["controller_storage_ref"])
    for r in (ref(SOURCE),ref(CHECKER),cap["never_ready_controller_actual_termination_ref"]):
        require(r in cap["source_refs"].values() and r in storage["reservation"]["consumer_bindings"],
            "Actual fresh controller reserve directly binds execution override/checker/termination evidence")
    require(cap["source_refs"].get(ORIGINAL_REF["path"])==ORIGINAL_REF,"Original scientific worker route preserved")
    return cap

def validate_prior_never_ready_termination_v1(record,prior_capability_ref):
    v=read(record)
    require(v.get("schema")=="GPU45_never_ready_controller_actual_termination_v1"
        and v.get("root_authorizer")=="root_agent" and v.get("prior_capability_ref")==prior_capability_ref
        and v.get("never_ready") is True and v.get("no_jobs_submitted") is True
        and v.get("exact_controller_identity_revalidated_before_signal") is True
        and v.get("never_ready_owned_controller_only") is True
        and v.get("actual_tool_terminal_consumed") is True
        and type(v.get("actual_tool_exit_code")) is int
        and isinstance(v.get("tool_session_id"),(str,int)) and str(v["tool_session_id"])
        and isinstance(v.get("terminal_chunk_id"),str) and v["terminal_chunk_id"],
        "Root's actual never-ready controller identity/termination and consumed tool terminal")
    i=v.get("controller_identity")
    require(isinstance(i,dict) and all(type(i.get(k)) is int for k in ("pid","uid","start_ticks"))
        and i["pid"]>0 and i["uid"]>=0 and i["start_ticks"]>0,"Exact captured own controller identity")
    return v

def admit_v1(request_ref):
    req=read(request_ref)
    require(req.get("schema")=="GPU45_TRAIN_controller_execution_override_admission_v1" and req.get("status")=="active"
        and req.get("root_authorizer")=="root_agent","Explicit fresh controller-only reserve authority")
    prior_ref=req["prior_capability_ref"];termination_ref=req["never_ready_controller_actual_termination_ref"]
    validate_prior_never_ready_termination_v1(termination_ref,prior_ref)
    p=original();prior=p.load_cap(prior_ref);old_storage=read(prior["controller_storage_ref"]);old_plan=read(prior["plan_ref"])
    # No per-worker re-admission, config mutation, namespace reuse or source rewrite.
    rb=req["reserved_bytes"];require(type(rb) is int and rb==32*2**20,"Exactly32MiB new controller-only reservation")
    packet=ROOT/HERE/"admitted_controller_v1";output=HERE+"/controller_output_v1"
    require(not packet.exists() and not path(output).exists(),"Fresh controller metadata/output; old partial stays charged")
    packet.mkdir();plan=dict(old_plan,controller_execution_override_ref=ref(SOURCE),controller_metadata_scope_checker_ref=ref(CHECKER))
    require(plan["jobs"]==prior["jobs"],"Exact unchanged admitted C3/H050 jobs")
    plan_ref=save(packet,"pool_plan.root_active.json",plan)
    guard=p.current_guard()
    # Original guard/root reserve APIs; no memo is installed in admission process.
    h050_common=load(dict(path="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bridge_common_v1.py",
        sha256="5ec43d65bde777d18c071be2cdf4ee5c14ef2784bfe54ac27582f187ee93552b"),"_GPU45_override_original_guard_pins_v1")
    with guard.budget.v4.mutex(ROOT):state=guard._state_locked(h050_common.CORE,h050_common.BASE,fresh=True)
    view,bindings=state[2],state[3]
    consumers={r["path"]:r for r in [*bindings.values(),*old_storage["reservation"]["consumer_bindings"],
        ref(SOURCE),ref(CHECKER),request_ref,termination_ref,prior_ref,plan_ref]}
    require(len(consumers)<=64,"Original64 direct metadata limit")
    controller_job="GPU4-C3cache-H050-TRAIN-controller-execution-override-v1"
    expiry=datetime.fromtimestamp(prior["drain_expires_unix"],timezone.utc).isoformat()
    reserve=dict(key="H041/"+controller_job,output=str(ROOT/output),auxiliary_scopes=[],reserved_bytes=rb,consumer_bindings=list(consumers.values()))
    auth=guard.operation_draft_v1("reserve",reserve,expected_before=guard._pins(view),bindings=bindings)
    auth.update(active=True,expires_at_utc=expiry)
    receipt=guard.apply_reserve_v1(save(packet,"reserve_authority.json",auth),expected_source_sha256=h050_common.CORE,expected_launcher_sha256=h050_common.BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,
        job_id=controller_job,output_directory=output,reserved_new_bytes=rb,reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=expiry,completed_overlay_policy_ref=p.POLICY,
        budget_components=dict(implementation=ref(guard.budget.SOURCE_REL),launcher=ref(guard.budget.LAUNCHER_REL),consumer_adapter=dict(path=p.GUARD,sha256=p.GUARD_SHA)))
    sr=save(packet,"storage.json",storage)
    sources=dict(prior["source_refs"])
    for r in (ref(SOURCE),ref(CHECKER),termination_ref):sources[r["path"]]=r
    cap=dict(prior,plan_ref=plan_ref,controller_storage_ref=sr,controller_job_id=controller_job,source_refs=sources,
        controller_execution_override_ref=ref(SOURCE),controller_metadata_scope_checker_ref=ref(CHECKER),
        never_ready_controller_actual_termination_ref=termination_ref,original_canonical_pool_and_workers_unchanged=True,
        prior_never_ready_capability_ref=prior_ref)
    require(cap["jobs"]==prior["jobs"] and cap["pool"]==prior["pool"] and cap["physical_gpu"]==prior["physical_gpu"],
        "Same original logical workers and physical lease identity")
    cr=save(packet,"pool_capability.root_active.json",cap);validate_override_cap_v1(cr,p)
    result=dict(capability_ref=cr,controller_storage_ref=sr,plan_ref=plan_ref,consumer_binding_count=len(consumers),launched=False,
        old_partial_controller_charged_unchanged=True,worker_admissions_unchanged=True)
    save(packet,"admitted.json",result);print(json.dumps(result),flush=True);return result

def start_v1(cap_ref):
    p=original();validate_override_cap_v1(cap_ref,p)
    checker=load(ref(CHECKER),"_GPU45_TRAIN_override_metadata_scope_v1")
    class Controller(p.Shared45Controller):
        def _check(self,required=0):
            require(time.time()<self.cap["drain_expires_unix"],"Original bounded owned drain expiry")
            require(ref(SOURCE)==self.cap["controller_execution_override_ref"] and ref(CHECKER)==self.cap["controller_metadata_scope_checker_ref"]
                and ref(ORIGINAL_REF["path"])==ORIGINAL_REF and ref(BASE_REF["path"])==BASE_REF,"All actual controller execution source bytes unchanged")
            if not hasattr(self,"_isolated_controller_metadata"):
                self._isolated_controller_metadata=checker.ControllerMetadataScopeV1(self.guard,self.admin)
            self.storage_account=self._isolated_controller_metadata.check(required)
            return self.storage_account
    namespace=dict(p.__dict__)
    exec(compile(adapted_start_source_v1(),"<unchanged_original_TRAIN_start_except_guard_lock_order>","exec"),namespace)
    Controller.start=namespace["start"]
    controller=Controller(cap_ref).start()
    def stop(signum,frame):controller.shutdown_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps(controller.status(),sort_keys=True),flush=True);controller.run();return 0

def main(argv=None):
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument("action",choices=("describe","admit","start"),nargs="?",default="describe")
    pa.add_argument("--execute",action="store_true");pa.add_argument("--record");pa.add_argument("--sha256");a=pa.parse_args(argv)
    if a.action=="describe":print(json.dumps(dict(status="source_only",actual_execution=False,original_workers_unchanged=True)));return 0
    require(a.execute and a.record and a.sha256,"Explicit root operation and pinned metadata")
    record=dict(path=a.record,sha256=a.sha256)
    if a.action=="admit":admit_v1(record);return 0
    return start_v1(record)
if __name__=="__main__":raise SystemExit(main())

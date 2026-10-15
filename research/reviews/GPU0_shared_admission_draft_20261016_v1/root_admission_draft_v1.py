"""Root-only GPU0 fresh admission helper. Default describe; never launches jobs.

bind_metadata is pure phase-metadata preparation: no guard/state/imported science.
admit/assemble are explicit future root operations using unchanged original APIs.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,ast,copy,hashlib,importlib.util,json,sys,time,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_shared_admission_draft_20261016_v1"
SOURCE=HERE+"/root_admission_draft_v1.py"
EXEC="research/reviews/GPU0_shared_execution_source_20261016_v1"
POOL=EXEC+"/gpu0_shared_pool_v1.py";LAUNCHER=EXEC+"/launch_gpu0_shared_v1.py"
RUNTIME=EXEC+"/gpu0_runtime_adapter_v1.py";CONTEXT=EXEC+"/gpu0_h050_context_v1.py";GUARD0=EXEC+"/gpu0_native_guard_v1.py"
BASE="src/arggnn/gpu_pool_v1.py";OLD_POOL="src/arggnn/gpu45_c3cache_h050_pool_v1.py"
DOWN="research/reviews/GPU45_shared_downstream_lineage_source_20261016_v1/pool.staged.py"
CHECKER="research/reviews/GPU45_TRAIN_controller_execution_override_20261016_v1/controller_metadata_scope_check_v1.py"
RENEWAL="research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py"
BOOTSTRAP=EXEC+"/gpu0_bootstrap_renewal_v1.py"
ORIGINAL_ROUTE=dict(path="research/reviews/GPU45_C3cache_H050_migration_gate_successor_20261016_v2/root_shared_admission_v2.py",sha256="b22a18f7e4126b6435fdb335bfa2b972cb4e6e29d611ed4548196dc35eb32662")
H050_DRIVER=dict(path="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/root_driver_v1.py",sha256="f8399361305ecb3b27172695d6e6e5c1827da330d4fa3d4565bda2df1ed6d6ac")
C3_CONTRACT=dict(path="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/root_contract_v1.py",sha256="8143dcd241c23f9bcf3379644c275f66febc8cf8dea01727e542d88651d734a0")
MODULES=dict(H050="arggnn.H050_direct_readout_cut_memory_exec_v1",C3="arggnn.american_c3_gnn_prefetch_ramcache_isolated980_exec_v3")
REPAIRS={
"research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/tracking_factory_successor_v1.py":"e9fd0155e42412b2f55af45d48e7f5c01bcfd0814406c94bbf69b59d60d6ff13",
"research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/private_owned_cache_v1.py":"f2c1ba189436d5280b461d93822f4e75e54eb5205f1b95f490e4388d5b946dbf",
"research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/private_owned_tensors_v1.py":"3a647e5274c41b1859ddd66a50b43c7d7e596244f9f4c0ffccf06da9a8045e83"}

def require(ok,message):
    if not ok:raise ValueError(message)

def path(name):
    p=Path(name);p=p if p.is_absolute() else ROOT/p
    require(".." not in p.parts and p.resolve()==p and p.is_relative_to(ROOT) and not p.is_symlink(),"Canonical contained metadata path")
    return p

def ref(name):
    p=path(name);require(p.is_file(),"Regular source/metadata file")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,"Exact bounded JSON ref")
    raw=path(record["path"]).read_bytes();require(len(raw)<=32*2**20,"Bounded metadata")
    return json.loads(raw)

def save(folder,name,value):
    p=folder/name;require(not p.exists(),"No metadata replay/overwrite")
    with p.open("x") as h:json.dump(value,h,indent=2,sort_keys=True);h.write("\n")
    return ref(p)

def load(record,label):
    require(ref(record["path"])==record,"Exact frozen source")
    sp=importlib.util.spec_from_file_location(label,path(record["path"]))
    m=importlib.util.module_from_spec(sp);sys.modules[label]=m;sp.loader.exec_module(m);return m

def unique(rows):
    indexed={}
    for r in rows:
        require(r["path"] not in indexed or indexed[r["path"]]==r,"No conflicting path hashes")
        indexed[r["path"]]=r
    return list(indexed.values())

def activation(record,*,fresh=False):
    a=read(record);policy=read(a["GPU0_policy_ref"])
    require(a.get("schema")=="GPU0_fresh_H050_C3_root_activation_v1" and a.get("status")=="active"
        and a.get("root_authorizer")=="root_agent" and a.get("physical_gpu")==0
        and a.get("gpu_uuid")==policy.get("GPU0_uuid") and policy.get("allowed_physical_GPUs")==[0]
        and a["stop_evidence_ref"]==policy["stop_evidence_ref"],"Exact active GPU0 root activation")
    require(time.time()<a["expires_unix"]<a["drain_expires_unix"] and a["drain_expires_unix"]>=a["expires_unix"]+600,"Fresh bounded root execution/drain")
    read(a["stop_evidence_ref"]);certificate=read(a["marker_repair_renewal_ref"])
    require(a["drain_expires_unix"]<datetime.fromisoformat(certificate["renewed_expires_at_utc"].replace("Z","+00:00")).timestamp(),"Existing renewal covers complete drain")
    require(set(a["jobs"])=={"H050","C3"},"Fresh H050+C3 only")
    seen=set()
    for kind,j in a["jobs"].items():
        old=read(j["prior_input_ref"])
        require(j["job_id"] not in seen and j["job_id"]!=old["job_id"] and j["output_directory"]!=old["output_directory"]
            and isinstance(j["repeat_reason"],str) and len(j["repeat_reason"])>20 and j.get("prior_status"),"Explicit unfinished/interrupted repeat lineage, new identity/output")
        seen.add(j["job_id"])
        if fresh:require(not path(j["packet_directory"]).exists() and not path(j["output_directory"]).exists(),"Fresh worker packet/output, old artifacts preserved")
    require(a["controller"]["job_id"] not in seen and a["controller"]["reserved_bytes"]==32*2**20,"Separate32MiB controller identity")
    return a

def pool_required_refs(a):
    return unique([ref(n) for n in (POOL,LAUNCHER,RUNTIME,GUARD0,OLD_POOL,BASE,DOWN,RENEWAL,BOOTSTRAP)]+
        [a["GPU0_policy_ref"],a["stop_evidence_ref"],a["marker_repair_renewal_ref"]])

def operational_refs(a,kind):
    rows=pool_required_refs(a)
    if kind=="H050":rows=[r for r in rows if r["path"]!=DOWN]+[ref(CONTEXT)]
    if kind=="C3":rows += [dict(path=p,sha256=h) for p,h in REPAIRS.items()]
    return unique(rows+[ref(SOURCE),ref("src/"+MODULES[kind].replace(".","/")+".py")])

def operational_spec(record,a,kind):
    j=a["jobs"][kind]
    return dict(operational_launch_mode="GPU0_fresh_shared_interrupted_repeat_v1",
        operational_shared_route_ref=record,operational_pool_source_ref=ref(POOL),
        operational_launcher_ref=ref(LAUNCHER),operational_pool_base_ref=ref(BASE),
        operational_canonical_module_ref=ref("src/"+MODULES[kind].replace(".","/")+".py"),
        operational_root_entry_ref=ref(SOURCE),operational_GPU0_runtime_ref=ref(RUNTIME),
        operational_GPU0_guard_ref=ref(GUARD0),operational_GPU0_policy_ref=a["GPU0_policy_ref"],
        marker_repair_renewal_ref=a["marker_repair_renewal_ref"],
        interrupted_repeat_lineage=dict(activation_ref=record,prior_input_ref=j["prior_input_ref"],prior_status=j["prior_status"],
            repeat_reason=j["repeat_reason"],root_transition_stop_ref=a["stop_evidence_ref"],
            original_science_preserved=True,new_storage_charge_required=True,resume_or_prior_completion_not_asserted=True))

def binding_ref():
    return ref(HERE+"/bound_metadata_v1/binding.json")

def bind_metadata(record):
    """Pure JSON/source preparation before any original state/guard call."""
    a=activation(record,fresh=True);folder=path(HERE+"/bound_metadata_v1")
    require(not folder.exists(),"Fresh metadata binding directory")
    # Authenticate source bytes and all repair refs, without importing science.
    for r in pool_required_refs(a)+[ref(CONTEXT),ref(SOURCE)]+[dict(path=p,sha256=h) for p,h in REPAIRS.items()]:
        require(ref(r["path"])==r,"Actual reviewed source closure")
    folder.mkdir()
    h=a["jobs"]["H050"];old=read(h["prior_input_ref"]);cfg=read(h["prior_config_ref"])
    require(old["refs"]["config"]==h["prior_config_ref"] and cfg["max_cuda_reserved_bytes"]==28*2**30
        and cfg["max_cpu_rss_bytes"]==64*2**30,"Same H050 prior config/caps")
    manifest=read(cfg["source_manifest_ref"]);sources=dict(manifest["source_sha256"])
    for r in unique(pool_required_refs(a)+operational_refs(a,"H050")):sources[r["path"]]=r["sha256"]
    metadata=dict(manifest.get("metadata_refs",{}));metadata["GPU0_activation"]=record
    manifest_ref=save(folder,"H050_source_manifest.json",dict(manifest,source_sha256=sources,metadata_refs=metadata,
        original_manifest_ref=cfg["source_manifest_ref"],interrupted_repeat_lineage=operational_spec(record,a,"H050")["interrupted_repeat_lineage"]))
    cfg=copy.deepcopy(cfg);cfg.update(source_manifest_ref=manifest_ref,root_context_module_ref=ref(CONTEXT),**operational_spec(record,a,"H050"))
    config_ref=save(folder,"H050_config.json",cfg)
    req=read(h["base_admission_request_ref"]);req.update(config_ref=config_ref,physical_gpu=0,reserved_bytes=h["reserved_bytes"])
    h_request=save(folder,"H050_request.json",req)
    c=a["jobs"]["C3"];creq=read(c["base_worker_request_ref"])
    require(creq["phase"]=="train","Original C3 TRAIN")
    creq=copy.deepcopy(creq);creq.update(physical_gpu=0,GPU_UUID=a["gpu_uuid"],operational_short_tmp_directory=c["operational_short_tmp_directory"])
    require(creq["operational_short_tmp_directory"].startswith(".runtime/tmp/gpu0_shared/") and not path(creq["operational_short_tmp_directory"]).exists(),"Fresh private C3 TMP")
    c_request=save(folder,"C3_request.json",creq)
    result=dict(schema="GPU0_fresh_shared_metadata_binding_v1",activation_ref=record,H050_request_ref=h_request,
        H050_config_ref=config_ref,C3_request_ref=c_request,admitted=False,launched=False,
        original_warmcheckpoint_proxy_preserved=cfg["scientific_input_closure_ref"])
    out=save(folder,"binding.json",result);print(json.dumps(dict(binding_ref=out,**result)),flush=True);return result

def route(record):
    a=activation(record)
    renewal=load(ref(RENEWAL),"_GPU0_admission_original_renewal");renewal.install_renewal_v1(a["marker_repair_renewal_ref"])
    renewal.validate_certificate_v1(a["marker_repair_renewal_ref"])
    p=load(ref(POOL),"_GPU0_actual_execution_source_for_admission")
    require(p.guard_module().POLICY_REF==a["GPU0_policy_ref"],"Same exact GPU0 guard/policy")
    return a,p

def _function(record,name):
    source=path(record["path"]).read_text();require(ref(record["path"])==record,"Frozen original function source")
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,node)

def _replace(text,changes):
    for old,new in changes:
        require(text.count(old)==1,"One exact original admission adaptation: "+old[:72]);text=text.replace(old,new)
    compile(text,"<GPU0_original_admission_operational_changes_only>","exec");return text

def h050_admit_source_v1():
    return _replace(_function(H050_DRIVER,"admit"),[
        ('gpu in (4,5)','gpu==0'),
        ('packet=ROOT/HERE/("admitted_"+phase)','packet=ROOT/JOB_SPEC["packet_directory"]'),
        ('output="experiments/results/H050_GPU%d_direct_readout_cut_memory_%s_seed101_v1"%(gpu,phase)','output=JOB_SPEC["output_directory"]'),
        ('job="H050-direct-readout-%s-seed101-v1-GPU%d"%(phase,gpu)','job=JOB_SPEC["job_id"]'),
        ('expiry=(now+timedelta(hours=24)).isoformat()','expiry=min(now+timedelta(hours=24),datetime.fromtimestamp(ACTIVATION["expires_unix"],timezone.utc)).isoformat()'),
        ('validate_spec_v1(reader,spec,cfg)','spec.update(OPERATIONAL_SPEC)\n    validate_spec_v1(reader,spec,cfg)'),
        ('direct_required_refs_v1(cfg,spec)+list(budget_bindings.values())','direct_required_refs_v1(cfg,spec)+EXTRA_REQUIRED+list(budget_bindings.values())')])

def admit_h050(record):
    a,p=route(record);b=read(binding_ref());require(b["activation_ref"]==record,"Exact bound activation")
    runtime=load(ref(RUNTIME),"_GPU0_private_H050_metadata_for_admission");c=runtime.private_h050_common_v1()
    ns=dict(c.__dict__,datetime=datetime,timedelta=timedelta,timezone=timezone,json=json,ACTIVATION=a,
        JOB_SPEC=a["jobs"]["H050"],OPERATIONAL_SPEC=operational_spec(record,a,"H050"),EXTRA_REQUIRED=operational_refs(a,"H050"))
    exec(compile(h050_admit_source_v1(),"<original_H050_admit_GPU0_fresh_operational_namespace>","exec"),ns)
    return ns["admit"](b["H050_request_ref"])

def c3_contract():
    c=load(C3_CONTRACT,"_GPU0_original_C3_contract_for_admission")
    text=_replace(_function(C3_CONTRACT,"validate_request"),[('req["physical_gpu"] in (4,5)','req["physical_gpu"]==0')])
    exec(compile(text,"<original_C3_request_GPU0_eligibility_only>","exec"),c.__dict__)
    original=c.make_config
    def make_config(req,caps):
        cfg=original(req,caps);cfg["short_tmp_directory"]=req["operational_short_tmp_directory"];return cfg
    c.make_config=make_config;return c

def c3_admit_source_v1():
    return _replace(_function(ORIGINAL_ROUTE,"admit_c3"),[
        ('req["physical_gpu"]==4','req["physical_gpu"]==0'),
        ('packet=ROOT/HERE/"admitted_C3_shared_train_v1"','packet=ROOT/JOB_SPEC["packet_directory"]'),
        ('output="experiments/results/C3_GNN_cachev3_shared_train_v1"','output=JOB_SPEC["output_directory"]'),
        ('physical_gpu=4,execution_gpu_uuid','physical_gpu=0,execution_gpu_uuid')])

def admit_c3(record):
    a,p=route(record);b=read(binding_ref());require(b["activation_ref"]==record,"Exact bound activation")
    request=dict(worker_request_ref=b["C3_request_ref"],shared_route_ref=record)
    original=load(ORIGINAL_ROUTE,"_GPU0_original_shared_admission_namespace")
    ns=dict(original.__dict__,HERE=HERE,SOURCE=SOURCE,POOL=POOL,LAUNCHER=LAUNCHER,JOB_SPEC=a["jobs"]["C3"],
        JOBS={MODULES["C3"]:a["jobs"]["C3"]["job_id"]},
        prepare_admission=lambda record0,module:(request,a,p),c3_contract=c3_contract,
        operational_refs=lambda:operational_refs(a,"C3"),renewal_refs=lambda renewal:[],
        operational_spec=lambda record0,module:operational_spec(record,a,"C3"))
    exec(compile(c3_admit_source_v1(),"<original_shared_v2_C3_admit_GPU0_fresh_namespace>","exec"),ns)
    return ns["admit_c3"](record)

def assemble(record,H050_admitted_ref,C3_admitted_ref,native_admitted_ref=None):
    a,p=route(record);jobs={}
    for kind,ar in (("H050",H050_admitted_ref),("C3",C3_admitted_ref)):
        answer=read(ar);ir=answer["input_ref"];spec=read(ir);j=a["jobs"][kind]
        require(spec["job_id"]==j["job_id"] and spec["physical_gpu"]==0 and spec["output_directory"]==j["output_directory"]
            and spec["operational_shared_route_ref"]==record,"Exact fresh admitted worker")
        module=MODULES[kind];pin=p.legacy.MODULE_PINS[module];caps=spec["caps_ref"] if kind=="C3" else spec["refs"]["config"]
        jobs[j["job_id"]]=dict(worker_kind=kind,job_id=j["job_id"],module=module,module_ref=ref(pin["path"]),runtime_entry_ref=ref(RUNTIME),
            input_ref=ir,cuda_caps_ref=caps,storage_ref=spec["refs"]["storage"],cuda_cap_bytes=pin["cuda_bytes"],cpu_rss_cap_bytes=pin["cpu_bytes"],
            operational_deadline_seconds=86400,margin_mib=2048,peak_mib=pin["cuda_bytes"]//2**20+2048,log_limit_bytes=2**20,env={},
            args=["--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]])
    native_sources=[]
    if native_admitted_ref is not None:
        native=load(ref(HERE+"/native_assembly_v1.py"),"_GPU0_optional_native_assembly")
        native_job,native_sources=native.declared_job(record,a,native_admitted_ref)
        require(native_job["job_id"] not in jobs,"Distinct native logical job")
        jobs[native_job["job_id"]]=native_job
    j=a["controller"];packet=path(j["packet_directory"]);require(not packet.exists() and not path(j["output_directory"]).exists(),"Fresh controller packet/output");packet.mkdir()
    plan=dict(schema="GPU0_shared_logical_pool_plan_v1",pool=a["pool"],physical_gpu=0,gpu_uuid=a["gpu_uuid"],maximum_concurrent_jobs=2,headroom_mib=4096,
        termination_grace_seconds=30,jobs=jobs,marker_repair_renewal_ref=a["marker_repair_renewal_ref"],GPU0_policy_ref=a["GPU0_policy_ref"],stop_evidence_ref=a["stop_evidence_ref"])
    plan_ref=save(packet,"pool_plan.root_active.json",plan)
    guard=p.legacy.current_guard()
    original=load(ORIGINAL_ROUTE,"_GPU0_original_admission_guard_pins")
    c=load(ref("research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bridge_common_v1.py"),"_GPU0_original_common_guard_pins")
    with guard.budget.v4.mutex(ROOT):state=guard._state_locked(c.CORE,c.BASE,fresh=True)
    view,bindings=state[2],state[3]
    required=unique(pool_required_refs(a)+[ref(SOURCE),ref(CHECKER),ref(CONTEXT)]+[ref(x["path"]) for x in p.legacy.MODULE_PINS.values()]+native_sources)
    consumers=unique(list(bindings.values())+required+[plan_ref,record,dict(path=p.legacy.GUARD,sha256=p.legacy.GUARD_SHA),p.legacy.POLICY,p.legacy.REPAIR])
    require(len(consumers)<=64,"Original64 direct controller binding ceiling")
    expiry=datetime.fromtimestamp(a["drain_expires_unix"],timezone.utc).isoformat()
    reserve=dict(key="H041/"+j["job_id"],output=str(path(j["output_directory"])),auxiliary_scopes=[],reserved_bytes=j["reserved_bytes"],consumer_bindings=consumers)
    authority=guard.operation_draft_v1("reserve",reserve,expected_before=guard._pins(view),bindings=bindings);authority.update(active=True,expires_at_utc=expiry)
    receipt=guard.apply_reserve_v1(save(packet,"reserve_authority.json",authority),expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,job_id=j["job_id"],
        output_directory=j["output_directory"],reserved_new_bytes=j["reserved_bytes"],reservation=receipt["value"],initial_ledger_pins=receipt["initial_ledger_pins"],
        expires_at_utc=expiry,completed_overlay_policy_ref=p.legacy.POLICY,budget_components=dict(implementation=ref(guard.budget.SOURCE_REL),launcher=ref(guard.budget.LAUNCHER_REL),consumer_adapter=dict(path=p.legacy.GUARD,sha256=p.legacy.GUARD_SHA)))
    sr=save(packet,"storage.json",storage)
    cap=dict(plan,schema="GPU0_shared_logical_pool_capability_v1",status="active",root_authorizer="root_agent",logical_job_sharing_authorized=True,
        external_job_sharing_authorized=True,legacy_exclusive_coexistence_allowed=False,maximum_distinct_project_GPUs=1,
        source_refs={x["path"]:x for x in required},plan_ref=plan_ref,controller_storage_ref=sr,controller_job_id=j["job_id"],
        expires_unix=a["expires_unix"],drain_expires_unix=a["drain_expires_unix"],owned_drain_after_expiry_authorized=True,owned_termination_escalation_authorized=True,
        native_40GiB_operational_ceiling_authorized=bool(native_admitted_ref and a.get("native_40GiB_operational_ceiling_authorized")),native_72h_operational_deadline_authorized=bool(native_admitted_ref and a.get("native_72h_operational_deadline_authorized")))
    cr=save(packet,"pool_capability.root_active.json",cap);p.load_cap(cr)
    result=dict(capability_ref=cr,plan_ref=plan_ref,controller_storage_ref=sr,consumer_binding_count=len(consumers),launched=False)
    save(packet,"admitted.json",result);print(json.dumps(result),flush=True);return result

def main(argv=None):
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument("action",choices=("describe","bind","admit-h050","admit-c3","assemble"),nargs="?",default="describe")
    pa.add_argument("--execute",action="store_true");pa.add_argument("--record");pa.add_argument("--sha256")
    pa.add_argument("--native-admitted");pa.add_argument("--native-sha256");pa.add_argument("--h050-admitted");pa.add_argument("--h050-sha256");pa.add_argument("--c3-admitted");pa.add_argument("--c3-sha256");a=pa.parse_args(argv)
    if a.action=="describe":print(json.dumps(dict(status="source_only_default_inert",no_admission_or_jobs_called=True)));return 0
    require(a.execute and a.record and a.sha256,"Explicit root action and exact activation")
    r=dict(path=a.record,sha256=a.sha256)
    if a.action=="bind":bind_metadata(r)
    elif a.action=="admit-h050":admit_h050(r)
    elif a.action=="admit-c3":admit_c3(r)
    else:
        require(all((a.h050_admitted,a.h050_sha256,a.c3_admitted,a.c3_sha256)),"Both exact admitted JSON refs")
        assemble(r,dict(path=a.h050_admitted,sha256=a.h050_sha256),dict(path=a.c3_admitted,sha256=a.c3_sha256),dict(path=a.native_admitted,sha256=a.native_sha256) if a.native_admitted and a.native_sha256 else None)
    return 0
if __name__=="__main__":raise SystemExit(main())

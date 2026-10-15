"""Future root-only shared admission route. Default inert; no jobs or probes.

Uses original validators/science and original guard APIs. Source-only development
must never call execute modes. New source refs are direct-bound before reserve.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,ast,copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU45_C3cache_H050_pool_successor_20261016_v1"
SOURCE=HERE+"/root_shared_admission_v1.py"
POOL="src/arggnn/gpu45_c3cache_h050_pool_v1.py"
LAUNCHER="scripts/launch_reserved_gpu45_c3cache_h050_pool_v1.py"
BASE="src/arggnn/gpu_pool_v1.py"
C3_MODULE="arggnn.american_c3_gnn_prefetch_ramcache_isolated980_exec_v3"
H050_MODULE="arggnn.H050_direct_readout_cut_memory_exec_v1"
JOBS={C3_MODULE:"C3-GNN-cachev3-shared-TRAIN-v1",H050_MODULE:"H050-direct-readout-shared-TRAIN-v1"}
C3_DRIVER="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/root_driver_v1.py"
C3_CONTRACT="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/root_contract_v1.py"
H050_DRIVER="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/root_driver_v1.py"
EXCLUSIVE_CONFIG=dict(path="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bound_root_v1/config_train_evaluate.canonical.root_active.json",
 sha256="533393f13b475c61e3f9091499eb00eec471d58ce776a090ac29ecbc88855c81")
EXCLUSIVE_REQUEST=dict(path="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bound_root_v1/admission_request.train.canonical.root_active.json",
 sha256="e9ed00fb875fefd72b77a4ebb0a62a1b2ada4ffac17179d53833f603f2b97f49")
MODE="GPU4_C3cachev3_H050_canonical_shared_owned_subreaper_v1"


def require(ok,message):
    if not ok:raise ValueError(message)


def path(name):
    p=ROOT/name
    require(not p.is_symlink() and p.resolve()==p and p.is_relative_to(ROOT),"Canonical contained path")
    return p


def ref(name):
    p=path(name)
    require(p.is_file(),"Bound regular file")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,"Exact path/SHA256 ref")
    data=path(record["path"]).read_bytes();require(len(data)<=32*2**20,"Bounded source metadata")
    return json.loads(data)


def save(folder,name,value):
    p=folder/name
    require(not p.exists(),"No metadata replay or overwrite")
    with p.open("x") as h:json.dump(value,h,indent=2,sort_keys=True);h.write("\n")
    return ref(str(p.relative_to(ROOT)))


def load(name,label,expected=None):
    if expected:require(ref(name)["sha256"]==expected,"Frozen source hash")
    sp=importlib.util.spec_from_file_location(label,path(name))
    m=importlib.util.module_from_spec(sp);sys.modules[label]=m;sp.loader.exec_module(m);return m


def pool_source():
    # Loading the reviewed source itself has no guard, CUDA, proc scan or mutation.
    return load(POOL,"_GPU45_C3cache_H050_shared_source_v1")


def operational_refs():
    return [ref(POOL),ref(LAUNCHER),ref(BASE),ref(SOURCE)]


def route(record):
    r=read(record);p=pool_source()
    require(r.get("schema")=="GPU45_C3cache_H050_shared_route_v1"
        and r.get("status")=="active" and r.get("root_authorizer")=="root_agent"
        and r.get("pool")=="gpu4_c3cache_h050_v1" and r.get("physical_gpu")==4
        and r.get("external_job_sharing_authorized") is True
        and r.get("maximum_distinct_project_GPUs")==2 and r.get("headroom_mib")==4096
        and r.get("margin_mib")==2048 and r.get("maximum_concurrent_jobs")==2
        and r.get("operational_source_refs")==operational_refs(),"Exact active shared route/source/caps")
    p.base.canonical_gpu_uuid(r["gpu_uuid"])
    require(ref(BASE)["sha256"]==p.BASE_SHA,"Frozen base source")
    require(Path(p.__file__).resolve()==ROOT/POOL,"Exact source must be installed")
    for pin in p.MODULE_PINS.values():require(ref(pin["path"])==dict(path=pin["path"],sha256=pin["sha256"]),"Exact canonical shims")
    p.validate_migration_gate(r["migration_gate_ref"],4)
    require(type(r.get("expires_unix")) in (int,float) and r["expires_unix"]>__import__("time").time(),"Fresh route authority")
    return r,p


def operational_spec(record,module):
    return dict(operational_launch_mode=MODE,operational_shared_route_ref=record,
        operational_pool_source_ref=ref(POOL),operational_launcher_ref=ref(LAUNCHER),
        operational_pool_base_ref=ref(BASE),operational_canonical_module_ref=ref("src/"+module.replace(".","/")+".py"),
        operational_root_entry_ref=ref(SOURCE))


def bind_h050(record):
    r,p=route(record)
    from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import bridge_common_v1 as c
    reader=c.MetadataReaderV1();cfg=reader.decode_json(EXCLUSIVE_CONFIG)
    c.validate_config_v1(cfg,reader);original=c.exact_sources_v1(reader,cfg)
    c.require(cfg["max_cuda_reserved_bytes"]==28*2**30 and cfg["max_cpu_rss_bytes"]==64*2**30,"Reviewed H050 caps")
    # Copy to new shared files. The exclusive canonical config/request are untouched.
    cfg=copy.deepcopy(cfg);sources=dict(original)
    for x in operational_refs()+[ref(p.MODULE_PINS[H050_MODULE]["path"])]:
        c.require(x["path"] not in sources or sources[x["path"]]==x["sha256"],"No conflicting operational refs")
        sources[x["path"]]=x["sha256"]
    folder=ROOT/HERE/"h050_shared_metadata_v1"
    require(not folder.exists(),"Fresh separate shared H050 metadata")
    folder.mkdir()
    manifest=reader.decode_json(cfg["source_manifest_ref"])
    manifest_ref=save(folder,"source_manifest.shared.root_bound.json",dict(manifest,
        source_sha256=sources,original_exclusive_manifest_ref=cfg["source_manifest_ref"],
        operational_launch_mode=MODE,actual_pool_source_ref=ref(POOL),actual_operational_launcher_ref=ref(LAUNCHER)))
    cfg.update(source_manifest_ref=manifest_ref,**operational_spec(record,H050_MODULE))
    config_ref=save(folder,"config_train.shared.root_active.json",cfg)
    c.validate_config_v1(cfg,reader);c.exact_sources_v1(reader,cfg)
    req=reader.decode_json(EXCLUSIVE_REQUEST);req.update(config_ref=config_ref,physical_gpu=4)
    request_ref=save(folder,"admission_request.train.shared.root_active.json",req)
    result=dict(config_ref=config_ref,train_request_ref=request_ref,exclusive_config_preserved=EXCLUSIVE_CONFIG,
        exclusive_request_preserved=EXCLUSIVE_REQUEST,shared_route_ref=record,admitted=False,launched=False)
    save(folder,"metadata_binding.shared.json",result)
    print(json.dumps(result),flush=True);return result


def prepare_admission(record,module):
    request=read(record)
    require(request.get("schema")=="GPU45_C3cache_H050_worker_admission_v1"
        and request.get("status")=="active" and request.get("root_authorizer")=="root_agent"
        and request.get("module")==module,"Explicit fresh shared worker admission")
    r,p=route(request["shared_route_ref"])
    return request,r,p


def c3_contract():
    return load(C3_CONTRACT,"_C3cache_shared_original_contract_v1",
        "8143dcd241c23f9bcf3379644c275f66febc8cf8dea01727e542d88651d734a0")


def admit_c3(record):
    request,r,p=prepare_admission(record,C3_MODULE);request_ref=request["worker_request_ref"]
    c=c3_contract()
    req,caps,runtime=c.validate_request(request_ref);phase=req["phase"]
    require(phase=="train" and req["physical_gpu"]==4 and req["GPU_UUID"]==r["gpu_uuid"]
        and caps["max_cuda_reserved_bytes"]==40*2**30,"Exact cache-C3 TRAIN cap/device")
    packet=ROOT/HERE/"admitted_C3_shared_train_v1"
    output="experiments/results/C3_GNN_cachev3_shared_train_v1"
    c.require(not packet.exists() and not c.local(output).exists(),"Fresh shared GNN packet/output")
    config=c.make_config(req,caps);tmp=c.local(config["short_tmp_directory"])
    c.require(not tmp.exists(),"Fresh original separately reserved TMP")
    sources={x:c.ref(x)["sha256"] for x in c.source_paths()}
    for x in operational_refs()+[request["shared_route_ref"]]:
        if x["path"].endswith(".py"):sources[x["path"]]=x["sha256"]
    packet.mkdir();config_ref=c.save(packet,"config.json",config)
    closure_ref=c.save(packet,"source_and_input_closure.json",dict(schema="C3_GNN_prefetch_ramcache_isolated980_transitive_closure_v1",
        source_sha256=sources,root_stage_request_ref=request_ref,bindings=req["bindings"],
        scientific_train_source_ref=c.TRAIN_SOURCE_REF,hook_ref=c.HOOK_REF,native_MLM_dependency=False,
        operational_shared_route_ref=request["shared_route_ref"]))
    guard=c.original_guard()
    with guard.budget.v4.mutex(ROOT):state=guard._state_locked(c.CORE,c.BASE,fresh=True)
    view,bindings=state[2],state[3];main=c.verify_shim()
    mandatory=[main,c.ref(c.SOURCE),c.ref(c.DRIVER),c.ref(c.ENTRY),request_ref,req["caps_ref"],
        config_ref,c.RUNTIME_REF,c.BASE_CONFIG_REF,closure_ref,c.GUARD_REF,c.POLICY_REF,c.REPAIR_REF,
        c.MEMO_REF,c.V4_REF,c.LAUNCHER_REF,c.GPU_AUTH_REF,
        c.ref("scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"),
        c.ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        c.ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        c.ref("research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py"),
        c.ref("src/arggnn/project_budget_v9_consumer.py"),req["scientific_training_config_ref"],
        *operational_refs(),record,request["shared_route_ref"]]
    # Original source-only unique helper, no root driver or guard modification.
    indexed={}
    for x in [*bindings.values(),*mandatory]:
        require(Path(x["path"]).suffix in (".py",".sh",".json"),"Only exact source/JSON direct bindings")
        require(x["path"] not in indexed or indexed[x["path"]]==x,"No conflicting direct refs")
        indexed[x["path"]]=x
    consumers=list(indexed.values());require(len(consumers)<=64,"64 direct binding limit")
    now=datetime.now(timezone.utc);expiry=min(now+timedelta(seconds=caps["operational_deadline_seconds"]),datetime.fromtimestamp(r["expires_unix"],timezone.utc))
    job=JOBS[C3_MODULE];aux=packet/"worker_aux"
    reserve_request=dict(key="H041/"+job,output=str(ROOT/output),auxiliary_scopes=[str(aux),str(tmp)],
        reserved_bytes=caps["reservation_cap_bytes"],consumer_bindings=consumers)
    authority=guard.operation_draft_v1("reserve",reserve_request,expected_before=guard._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry.isoformat())
    authority_ref=c.save(packet,"reserve_authority.json",authority)
    reserved=guard.apply_reserve_v1(authority_ref,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    reserve_ref=c.save(packet,"reserve_receipt.json",reserved)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,
        job_id=job,output_directory=output,reserved_new_bytes=caps["reservation_cap_bytes"],reservation=reserved["value"],
        initial_ledger_pins=reserved["initial_ledger_pins"],expires_at_utc=expiry.isoformat(),completed_overlay_policy_ref=c.POLICY_REF,
        budget_components=dict(implementation=c.ref(guard.budget.SOURCE_REL),launcher=c.ref(guard.budget.LAUNCHER_REL),consumer_adapter=c.GUARD_REF))
    refs=dict(config=config_ref,runtime=c.RUNTIME_REF,storage=c.save(packet,"storage.json",storage))
    spec=dict(schema="C3_GNN_prefetch_isolated980_active_input_v1",status="active",root_authorizer="root_agent",
        execution_authorized=True,GPU_authorized=True,parent_id=c.PARENT,phase=phase,job_id=job,
        output_directory=output,root_stage_request_ref=request_ref,caps_ref=req["caps_ref"],root_driver_ref=c.ref(c.DRIVER),
        main_source_ref=main,actual_launcher_ref=ref(LAUNCHER),source_sha256=sources,transitive_source_closure_ref=closure_ref,refs=refs,
        physical_gpu=4,execution_gpu_uuid=r["gpu_uuid"],root_auxiliary_directory=str(aux.relative_to(ROOT)),
        created_at_utc=now.isoformat(),expires_at_utc=expiry.isoformat(),max_cpu_rss_bytes=caps["max_cpu_rss_bytes"],
        max_cuda_reserved_bytes=caps["max_cuda_reserved_bytes"],completed_overlay_policy_ref=c.POLICY_REF,
        **operational_spec(request["shared_route_ref"],C3_MODULE))
    input_ref=c.save(packet,"input.json",spec)
    answer=dict(input_ref=input_ref,driver_ref=ref(SOURCE),original_worker_driver_ref=c.ref(c.DRIVER),
        reserve_receipt_ref=reserve_ref,storage=storage,config_ref=config_ref,runtime_ref=c.RUNTIME_REF,caps_ref=req["caps_ref"],
        shared_route_ref=request["shared_route_ref"],consumer_binding_count=len(consumers),execution_started=False)
    c.save(packet,"admitted.json",answer);print(json.dumps(answer),flush=True);return answer


def admit_h050(record):
    request,r,p=prepare_admission(record,H050_MODULE)
    from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import bridge_common_v1 as c
    reader=c.MetadataReaderV1();req=reader.decode_json(request["worker_request_ref"]);cfg=reader.decode_json(req["config_ref"])
    require(req["phase"]=="train" and req["physical_gpu"]==4
        and cfg.get("operational_shared_route_ref")==request["shared_route_ref"]
        and all(cfg.get(k)==v for k,v in operational_spec(request["shared_route_ref"],H050_MODULE).items()),
        "Separate shared H050 config uses exact actual pool route")
    c.validate_config_v1(cfg,reader);c.exact_sources_v1(reader,cfg)
    require(cfg["max_cuda_reserved_bytes"]==28*2**30 and cfg["max_cpu_rss_bytes"]==64*2**30,"Reviewed H050 caps")
    # Reuse original admission AST with only operational namespace/closure additions.
    # Original validation and every guard API/counter implementation remain intact.
    src=path(H050_DRIVER).read_text()
    require(hashlib.sha256(src.encode()).hexdigest()=="f8399361305ecb3b27172695d6e6e5c1827da330d4fa3d4565bda2df1ed6d6ac","Frozen H050 admit source")
    node=next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=="admit")
    text=ast.get_source_segment(src,node)
    replacements={
        'packet=ROOT/HERE/("admitted_"+phase)':'packet=ROOT/SHARED_HERE/"admitted_H050_shared_train_v1"',
        'output="experiments/results/H050_GPU%d_direct_readout_cut_memory_%s_seed101_v1"%(gpu,phase)':'output="experiments/results/H050_shared_direct_readout_train_seed101_v1"',
        'job="H050-direct-readout-%s-seed101-v1-GPU%d"%(phase,gpu)':'job=SHARED_JOB',
        'expiry=(now+timedelta(hours=24)).isoformat()':'expiry=min(now+timedelta(hours=24),datetime.fromtimestamp(SHARED_EXPIRES_UNIX,timezone.utc)).isoformat()',
        'validate_spec_v1(reader,spec,cfg)':'spec.update(SHARED_OPERATIONAL_SPEC)\n    validate_spec_v1(reader,spec,cfg)',
        'direct_required_refs_v1(cfg,spec)+list(budget_bindings.values())':'direct_required_refs_v1(cfg,spec)+SHARED_REQUIRED_REFS+list(budget_bindings.values())'}
    for before,after in replacements.items():
        require(text.count(before)==1,"Exact single original admission adaptation boundary")
        text=text.replace(before,after)
    namespace={k:getattr(c,k) for k in dir(c) if not k.startswith("__")}
    namespace.update(datetime=datetime,timedelta=timedelta,timezone=timezone,json=json,
        SHARED_HERE=HERE,SHARED_JOB=JOBS[H050_MODULE],SHARED_EXPIRES_UNIX=r["expires_unix"],
        SHARED_OPERATIONAL_SPEC=operational_spec(request["shared_route_ref"],H050_MODULE),
        SHARED_REQUIRED_REFS=operational_refs()+[ref(p.MODULE_PINS[H050_MODULE]["path"]),record,request["shared_route_ref"]])
    # Exec is restricted to this frozen source function, never untrusted metadata.
    exec(compile(text,"<H050_frozen_shared_namespace_admit_v1>","exec"),namespace)
    answer=namespace["admit"](request["worker_request_ref"])
    return answer


def assemble(record):
    """Fresh controller reserve + final exact plan/capability after both admissions."""
    req=read(record)
    require(req.get("schema")=="GPU45_C3cache_H050_controller_admission_v1"
        and req.get("status")=="active" and req.get("root_authorizer")=="root_agent","Root controller admission")
    r,p=route(req["shared_route_ref"]);jobs={}
    for module,key in ((C3_MODULE,"C3_admitted_ref"),(H050_MODULE,"H050_admitted_ref")):
        admitted=read(req[key]);input_ref=admitted["input_ref"];spec=read(input_ref);name=JOBS[module]
        require(spec["job_id"]==name and spec.get("operational_shared_route_ref")==req["shared_route_ref"]
            and spec["physical_gpu"]==4,"Exact fresh shared input/route")
        caps_ref=spec["caps_ref"] if module==C3_MODULE else spec["refs"]["config"]
        pin=p.MODULE_PINS[module];storage=read(spec["refs"]["storage"])
        require(all(x in storage["reservation"]["consumer_bindings"] for x in operational_refs()+[ref(pin["path"])]),
            "Actual worker reservation binds actual pool/launcher/base/adapter/shim")
        require(spec["max_cuda_reserved_bytes"]==read(caps_ref)["max_cuda_reserved_bytes"]==pin["cuda_bytes"],"Exact per-worker caps")
        jobs[name]=dict(job_id=name,module=module,module_ref=ref(pin["path"]),input_ref=input_ref,cuda_caps_ref=caps_ref,
            storage_ref=spec["refs"]["storage"],cuda_cap_bytes=pin["cuda_bytes"],cpu_rss_cap_bytes=pin["cpu_bytes"],operational_deadline_seconds=86400,margin_mib=2048,
            peak_mib=pin["cuda_bytes"]//2**20+2048,log_limit_bytes=2**20,env={},
            args=["--execute","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]])
    packet=ROOT/HERE/"admitted_controller_v1";require(not packet.exists(),"Fresh controller admission metadata")
    output=HERE+"/controller_output_v1";require(not path(output).exists(),"Fresh controller output")
    reserve_bytes=req.get("reserved_bytes")
    require(type(reserve_bytes) is int and 32*2**20<=reserve_bytes<=64*2**20,"Explicit32to64MiB complete controller reservation")
    packet.mkdir()
    plan=dict(schema="GPU45_shared_logical_pool_plan_v1",pool=r["pool"],physical_gpu=4,gpu_uuid=r["gpu_uuid"],
        maximum_concurrent_jobs=2,headroom_mib=4096,termination_grace_seconds=30,jobs=jobs,
        shared_route_ref=req["shared_route_ref"],migration_gate_ref=r["migration_gate_ref"],nominal_two_job_plus_headroom_GiB=76)
    plan_ref=save(packet,"pool_plan.root_active.json",plan)
    guard=p.current_guard()
    from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import bridge_common_v1 as c
    with guard.budget.v4.mutex(ROOT):state=guard._state_locked(c.CORE,c.BASE,fresh=True)
    view,bindings=state[2],state[3]
    # Guard expected pins below are exact original CORE/BASE, obtained from H050's
    # unchanged metadata boundary, never modified or inferred from module globals.
    consumers=c.unique(list(bindings.values())+operational_refs()+[plan_ref,record,req["shared_route_ref"],r["migration_gate_ref"],
        dict(path=p.GUARD,sha256=p.GUARD_SHA),p.POLICY,p.REPAIR]+[ref(x["path"]) for x in p.MODULE_PINS.values()])
    now=datetime.now(timezone.utc);expiry=datetime.fromtimestamp(r["expires_unix"]+600,timezone.utc)
    controller_job="GPU4-C3cache-H050-shared-controller-v1"
    reserve=dict(key="H041/"+controller_job,output=str(ROOT/output),auxiliary_scopes=[],
        reserved_bytes=reserve_bytes,consumer_bindings=consumers)
    authority=guard.operation_draft_v1("reserve",reserve,expected_before=guard._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry.isoformat())
    receipt=guard.apply_reserve_v1(save(packet,"reserve_authority.json",authority),expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,
        job_id=controller_job,output_directory=output,reserved_new_bytes=reserve_bytes,reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=expiry.isoformat(),completed_overlay_policy_ref=p.POLICY,
        budget_components=dict(implementation=ref(guard.budget.SOURCE_REL),launcher=ref(guard.budget.LAUNCHER_REL),
            consumer_adapter=dict(path=p.GUARD,sha256=p.GUARD_SHA)))
    storage_ref=save(packet,"storage.json",storage)
    refs={x["path"]:x for x in operational_refs()+[ref(x["path"]) for x in p.MODULE_PINS.values()]}
    cap=dict(plan,schema="GPU45_shared_logical_pool_capability_v1",status="active",root_authorizer="root_agent",
        logical_job_sharing_authorized=True,external_job_sharing_authorized=True,legacy_exclusive_coexistence_allowed=False,
        maximum_distinct_project_GPUs=2,source_refs=refs,plan_ref=plan_ref,controller_storage_ref=storage_ref,
        controller_job_id=controller_job,expires_unix=r["expires_unix"],drain_expires_unix=r["expires_unix"]+600,
        owned_drain_after_expiry_authorized=True,owned_termination_escalation_authorized=True)
    cap_ref=save(packet,"pool_capability.root_active.json",cap)
    p.load_cap(cap_ref)
    answer=dict(capability_ref=cap_ref,plan_ref=plan_ref,controller_storage_ref=storage_ref,launched=False)
    save(packet,"admitted.json",answer);print(json.dumps(answer),flush=True);return answer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("describe","bind-h050","admit-c3","admit-h050","assemble"),nargs="?",default="describe")
    parser.add_argument("--execute",action="store_true");parser.add_argument("--record");parser.add_argument("--sha256")
    a=parser.parse_args(argv)
    if a.action=="describe":
        print(json.dumps(dict(status="source_only_uninstalled_unadmitted",actual_guard_calls=0,
            migration="original_C3_genuine_terminal_and_owned_drain_then_root_revocation",native_GPU5="preserve",
            caps_GiB=dict(C3=40,H050=28,per_job_margin=2,device_headroom=4),nominal_total_GiB=76)));return 0
    require(a.execute and a.record and a.sha256,"Explicit future root action with pinned request")
    record=dict(path=a.record,sha256=a.sha256)
    {"bind-h050":bind_h050,"admit-c3":admit_c3,"admit-h050":admit_h050,"assemble":assemble}[a.action](record)
    return 0

if __name__=="__main__":raise SystemExit(main())

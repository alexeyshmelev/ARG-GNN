"""H050 admission plus unchanged isolated original launch supervisor. Default inert."""
import argparse,importlib.util,json
from datetime import datetime,timedelta,timezone
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1.bridge_common_v1 import *
SOURCE=DRIVER


def plan_request_v1(request):
    cfg=read(request["config_ref"])
    return count_plan_v1(read(cfg["count_manifest_ref"]),request["phase"],read(cfg["preparation_completion_ref"]),cfg)


def admit(request_ref):
    reader=MetadataReaderV1();r=reader.decode_json(request_ref);phase=r.get("phase")
    require(r.get("schema")=="H050_root_admission_request_v1" and r.get("status")=="active"
        and r.get("root_authorizer")=="root_agent" and phase in ("train","evaluate"),"Explicit active root H050 admission")
    cfg=reader.decode_json(r["config_ref"]);validate_config_v1(cfg,reader);sources=exact_sources_v1(reader,cfg)
    gpu=r.get("physical_gpu");require(type(gpu) is int and gpu in (4,5),"Root-selected actual GPU4/5 only")
    old=reader.decode_json(cfg["original_H038_input_ref"])
    require(set(old["observation_entry_refs"])==set(worker.TRAIN+worker.VAL)
        and old["observation_cache_source_refs"]==cfg["observation_cache_source_refs"],"Unchanged original40 observed parent capabilities")
    prep_spec=preparation_spec_v1(cfg,sources,old)
    marker=framework.completed_marker_preparation_v1(reader,prep_spec)
    authenticate_actual_cut_preparation_v1(reader,prep_spec,cfg)
    require(cfg["cut_preparation_completion_ref"]["path"]==cfg["expected_future_preparation_output"]+"/completed.json","Genuine completed v5 recipe output")
    counts=reader.decode_json(cfg["count_manifest_ref"])
    require(all(counts["parents"][pid]["observation_entry_ref"]==old["observation_entry_refs"][pid] for pid in worker.TRAIN+worker.VAL),"Count manifest binds original entries")
    require(type(r.get("reserved_bytes")) is int and r["reserved_bytes"]>0,"Explicit positive whole-scope reservation bytes")
    plan=count_plan_v1(counts,phase,marker,cfg);validate_plan_caps_v1(plan,cfg,r["reserved_bytes"])
    terminal=terminal_v1(cfg);prep_refs=framework.declared_cut_preparation_source_refs_v2(cfg)
    lineage=None
    if phase=="evaluate":
        require(all(is_ref_v1(r.get(k)) for k in TRAIN_FIELDS),"Genuine TRAIN completion/launch/wait refs")
        terminal.update({k:r[k] for k in TRAIN_FIELDS})
        lineage=validate_train_lineage_v1(reader,dict(phase=phase,**terminal),cfg)
    else:require(all(r.get(k) is None for k in TRAIN_FIELDS),"Fresh TRAIN without prior-run evidence")
    packet=ROOT/HERE/("admitted_"+phase)
    output="experiments/results/H050_GPU%d_direct_readout_cut_memory_%s_seed101_v1"%(gpu,phase)
    auxiliary=str(packet.relative_to(ROOT)/"worker_aux")
    require(not packet.exists() and not (ROOT/output).exists() and not (ROOT/auxiliary).exists(),"Fresh exclusive H050 admission/output/auxiliary; no replay")
    packet.mkdir();now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=24)).isoformat()
    job="H050-direct-readout-%s-seed101-v1-GPU%d"%(phase,gpu)
    runtime=save(packet,"runtime.json",dict(status="active",device="cuda",physical_gpu=gpu,job_id=job,
        output_directory=output,process_title="AI Auto Research Pipeline Test",created_at_utc=now.isoformat(),
        expires_at_utc=expiry,completed_overlay_policy_ref=POLICY_REF))
    refs={k:old["refs"][k] for k in ("registration","generation_config","observation_cache")}
    refs.update(config=r["config_ref"],runtime=runtime)
    targets=reader.decode_json(old["refs"]["target_bindings"]);ids=worker.TRAIN if phase=="train" else worker.VAL
    refs["target_bindings"]=save(packet,"phase_target_bindings.json",dict(targets,
        parents={pid:targets["parents"][pid] for pid in ids},TRAIN_included=phase=="train",
        VAL_included=phase=="evaluate",outer_included=False,phase=phase))
    manifest=save(packet,"consumer_manifest.json",dict(schema="H050_exact_transitive_consumer_closure_v1",
        hypothesis_id="H050",phase=phase,source_sha256=sources,refs=dict(refs),
        observation_entry_refs=old["observation_entry_refs"],observation_cache_source_refs=old["observation_cache_source_refs"],
        train_io_lineage=lineage,**prep_refs,**terminal))
    spec=dict(schema="H050_direct_readout_root_input_v1",hypothesis_id="H050",arm="joint_four_arm",
        root_authorizer="root_agent",status="active",phase=phase,seed=101,device="cuda",physical_gpu=gpu,
        fit_authorized=True,pyg_authorized=True,job_id=job,output_directory=output,worker_auxiliary_scope=auxiliary,
        created_at_utc=now.isoformat(),expires_at_utc=expiry,refs=refs,source_sha256=sources,
        root_context_module_ref=ref(CONTEXT),root_driver_ref=ref(DRIVER),canonical_dispatcher_ref=ref(worker.SOURCE),
        producer_ref=ref(worker.SOURCE),main_source_ref=ref(worker.SOURCE),worker_cli_ref=ref(worker.SOURCE),
        consumer_manifest_ref=manifest,observation_entry_refs=old["observation_entry_refs"],
        observation_cache_source_refs=old["observation_cache_source_refs"],max_cpu_rss_bytes=cfg["max_cpu_rss_bytes"],
        max_cuda_reserved_bytes=cfg["max_cuda_reserved_bytes"],GPU45_authorization_ref=AUTH_REF,
        completed_overlay_policy_ref=POLICY_REF,storage_adapter_ref=GUARD_REF,storage_launcher_ref=LAUNCHER_REF,
        marker_repair_authority_ref=REPAIR_REF,io_source_manifest_ref=cfg["source_manifest_ref"],
        train_io_lineage=lineage,**prep_refs,**terminal)
    validate_spec_v1(reader,spec,cfg)
    # Unmodified prospective guard API and canonical v4 mutex, with original pins.
    m,_=load_guard_v1(consumer=False)
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    view,budget_bindings=state[2],state[3]
    consumers=unique(direct_required_refs_v1(cfg,spec)+list(budget_bindings.values()))
    require(len(consumers)<=64,"Exact transitive closure at most64 original direct bindings")
    reserve=dict(key="H050/"+job,output=str(ROOT/output),auxiliary_scopes=[str(ROOT/auxiliary)],
        reserved_bytes=r["reserved_bytes"],consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",reserve,expected_before=m._pins(view),bindings=budget_bindings)
    authority.update(active=True,expires_at_utc=expiry)
    receipt=m.apply_reserve_v1(save(packet,"reserve_authority.json",authority),expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=job,output_directory=output,reserved_new_bytes=r["reserved_bytes"],
        reservation=receipt["value"],expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],
        completed_overlay_policy_ref=POLICY_REF,budget_components=dict(implementation=ref(m.budget.SOURCE_REL),
            launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD_REF))
    refs["storage"]=save(packet,"storage.json",storage)
    validate_spec_v1(reader,spec,cfg,storage)
    input_ref=save(packet,"input.json",spec)
    admitted=dict(schema="H050_root_admitted_v1",input_ref=input_ref,driver_ref=ref(DRIVER),request_ref=request_ref,
        consumer_binding_count=len(consumers),phase=phase,physical_gpu=gpu,auxiliary_scope=auxiliary,
        scientific_execution_started=False,output_forecast=plan,passes=30,updates_per_arm=960,
        updates_total=3840,checkpoint_files=24,VAL_products=192,
        original_launch_receipt_format_labels=list(LEGACY_RECEIPT_SCHEMAS))
    admitted_ref=save(packet,"admitted.json",admitted)
    print(json.dumps(dict(admitted_ref=admitted_ref,admitted=admitted)),flush=True);return admitted


def isolated_original_supervisor_v1():
    """Authenticate then bind a private module; launch code and frozen files stay exact."""
    require(ref(REVIEWED_DRIVER_REF["path"])==REVIEWED_DRIVER_REF,"Exact reviewed original supervisor bytes")
    loader=importlib.util.spec_from_file_location("_H050_isolated_original_supervisor_v1",ROOT/REVIEWED_DRIVER_REF["path"])
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import context_factory_v1
    module.SOURCE=DRIVER
    module.CANONICAL_DISPATCHER_REF=ref(worker.SOURCE)
    module.context=lambda:context_factory_v1
    require(module.launch.__code__.co_filename==str(ROOT/REVIEWED_DRIVER_REF["path"]),"Launch executes original compiled function body")
    return module


def launch(admitted_ref):
    reader=MetadataReaderV1();admitted=reader.decode_json(admitted_ref)
    require(admitted.get("schema")=="H050_root_admitted_v1" and admitted.get("driver_ref")==ref(DRIVER)
        and admitted.get("updates_total")==3840 and admitted.get("checkpoint_files")==24
        and admitted.get("VAL_products")==192,"Actual H050 admitted four-arm24checkpoint192seal record")
    spec=reader.decode_json(admitted["input_ref"]);cfg=reader.decode_json(spec["refs"]["config"])
    require(admitted_ref["path"]==HERE+"/admitted_"+spec["phase"]+"/admitted.json"
        and admitted["input_ref"]["path"]==HERE+"/admitted_"+spec["phase"]+"/input.json"
        and admitted["phase"]==spec["phase"] and admitted["physical_gpu"]==spec["physical_gpu"],"Fresh same H050 admission packet")
    validate_spec_v1(reader,spec,cfg,reader.decode_json(spec["refs"]["storage"]))
    from arggnn import census_tsinfer_fit_v1 as fit
    fit._check_active_v1(spec)
    return isolated_original_supervisor_v1().launch(admitted_ref)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("describe","plan","admit","launch"),nargs="?",default="describe")
    parser.add_argument("--execute",action="store_true");parser.add_argument("--record");parser.add_argument("--sha256")
    args=parser.parse_args(argv)
    if args.action=="describe":
        print(json.dumps(dict(status="source_only_no_admission_or_execution",hypothesis_id="H050",arms=4,
            checkpoints=24,updates=3840,pretruth_seals=192,legacy_operational_receipt_schemas=list(LEGACY_RECEIPT_SCHEMAS))));return 0
    require(args.record and args.sha256,"Exact root metadata record");record=dict(path=args.record,sha256=args.sha256)
    if args.action=="plan":print(json.dumps(plan_request_v1(read(record)),indent=2));return 0
    require(args.execute,"Explicit root execute flag for actual admission/launch")
    if args.action=="admit":admit(record);return 0
    return launch(record)

if __name__=="__main__":raise SystemExit(main())

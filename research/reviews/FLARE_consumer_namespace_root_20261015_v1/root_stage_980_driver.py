"""Root-only980GB IO successor for unchanged FLARE convert/native/seal; no scoring mode.

Source import is inert. Root --admit binds actual completed target/predecessor
metadata and reserves through original unmemoized prospective980GB. Root
--launch monitors actual workers with polling soft bounds, not kernel quotas.
"""
from pathlib import Path
import argparse,hashlib,importlib,json,os,selectors,signal,subprocess,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
SOURCE="research/reviews/FLARE_consumer_namespace_root_20261015_v1/root_stage_980_driver.py"
PACKET="research/reviews/FLARE_scientific_root_stage_source_v1"
BUILD_DRIVER_REF=dict(path="research/reviews/FLARE_existing_JDK_source_v1/root_build_driver.py",
    sha256="d7afaf5c6c6976a5c4804fe41d320f8609fd346d3f4ed0bca29235001d7fe561")
CONTEXT_REF=dict(path="research/reviews/FLARE_consumer_namespace_root_20261015_v1/root_context_980_v1.py",
    sha256="0230fceb637d2d8ed19c026f17f9120fbc4a8f6259856373671b5530f7305881")
GUARD_REF=dict(path="research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py",
    sha256="fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56")
POLICY_REF=dict(path="research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json",
    sha256="eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",
    sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",
    sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
NAMESPACE_MEMO_REF={'path': 'research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py', 'sha256': '17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f'}
V4_REF=dict(path="src/arggnn/project_budget_v4.py",
    sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
TEMPLATES={
    "convert":dict(path=PACKET+"/convert_train000.active_template.json",sha256="a43e1e2e07c1216c9a53e2272f8392588d4b02425bbe75c9b9ef403dd46012bc"),
    "native":dict(path=PACKET+"/native_train000.active_template.json",sha256="6e649312d0b4ecc0eb7f3b36cc82f12caf2030a0284fc8638536d5b2bc12410b"),
    "seal":dict(path=PACKET+"/seal_train000.active_template.json",sha256="80709b90106126bf63b499a375583a267ec74c18bef93b603a750de38a8a020e")}
BUILD_INPUT_REF=dict(path=PACKET+"/actual_completed_build_input.json",sha256="8349f04040a7293ddbf208f270858fc5687cd5cb7ff96b66e1933f6eae0b58b3")
CAPS=dict(convert=950000000,native=9000000000,seal=32000000)
COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1"
VERSION="flare version 0.6.0 [04ad73fd0 31-Oct-2025]"
STAGE_SOURCE=dict(convert="src/arggnn/american_c3_flare_convert_v1.py",
    native="src/arggnn/american_c3_flare_run_v2.py",seal="src/arggnn/american_c3_flare_evaluate_v1.py")
ROLES=dict(train="TRAIN_full30Mb_inference_canary_development_not_heldout",
    val="registered_heldout_full30Mb_comparison",test="registered_heldout_full30Mb_comparison")
b=None

def base():
    path=ROOT/BUILD_DRIVER_REF["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=BUILD_DRIVER_REF["sha256"]:
        raise ValueError("Frozen build-driver helper source")
    for p in (str(ROOT),str(ROOT/"src")):
        if p not in sys.path:sys.path.insert(0,p)
    return importlib.import_module("research.reviews.FLARE_existing_JDK_source_v1.root_build_driver")

def configured_guard_v1(*,consumer=False):
    b.need(all(b.ref(r["path"])==r for r in (GUARD_REF,POLICY_REF,REPAIR_REF)),"Exact actual active980GB root guard/policy/repair")
    m=importlib.import_module("research.reviews.prospective_980GB_capacity_source_v1.prospective_980gb_v1")
    m.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    b.need(m.MODE=="prospective_980gb_v1" and all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Original unmemoized reserve under actual980GB guard")
    if consumer:
        b.need(all(b.ref(r["path"])==r for r in (MEMO_REF,V4_REF,NAMESPACE_MEMO_REF)),
               "Exact pinned consumer memo2/original-v4 sources")
        namespace_memo=importlib.import_module("research.reviews.current_mutex_contention_20261015_v1.consumer_namespace_memo_v1")
        b.need(Path(namespace_memo.__file__).resolve()==ROOT/NAMESPACE_MEMO_REF["path"],"Exact consumer namespace implementation")
        m._flare_consumer_memo2=namespace_memo.install_consumer_v1(m.budget.v4,m.budget,
            consumer_role="read_only_consumer",expected_source_path=ROOT/V4_REF["path"],
            expected_source_sha256=V4_REF["sha256"])
    return m

def metadata(record):
    b.need(isinstance(record,dict) and set(record)=={"path","sha256"}
        and Path(record["path"]).suffix==".json"
        and not any(p in ("truth","oracle") for p in Path(record["path"]).parts),"Observation/completion JSON metadata only")
    return b.read(record)

def argument_ref(args,name):
    value=getattr(args,name);digest=getattr(args,"expected_"+name+"_sha256")
    b.need(value and digest,"Actual completed "+name+" path and SHA required")
    record=dict(path=b.local(value).relative_to(ROOT).as_posix(),sha256=digest)
    metadata(record);return record

def target_v1(record):
    """Read metadata only; focal/genotype/tree payloads remain unopened here."""
    target=metadata(record)
    b.need(target["schema"]=="FLARE_root_scientific_target_v1" and target["external_truth_ref"] is None,"Explicit pre-truth target")
    refs=[target[k] for k in ("observation_ref","observation_completion_ref","focal_completion_ref","count_manifest_ref","simulation_config_ref")]
    obs,done,focal,count,sim=[metadata(r) for r in refs]
    b.need(done["status"]=="completed_current_C3_inference" and done["observation_ref"]==target["observation_ref"],
        "Actual completed inferred observation")
    b.need(obs["schema"]=="american_c3_observed_v1" and obs["query_ancestry_truth_in_inputs"] is False
        and obs["sites_and_sample_identities_verified"] is True
        and obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
        and obs["parent_id"]==target["parent_id"] and obs["split"] in ROLES
        and target["scientific_role"]==ROLES[obs["split"]]
        and (obs["split"]!="train" or obs["parent_id"]=="american_c3_train_000"),"Exact actual TRAIN000 development or independently completed heldout target")
    b.need(focal["status"].startswith("completed_") and focal["observation_ref"]==target["observation_ref"]
        and focal["parent_id"]==obs["parent_id"] and focal["focals_ref"]==target["focal_registry_ref"]
        and focal["query_ancestry_truth_opened"] is False and type(focal["query_focal_count"]) is int
        and focal["query_focal_count"]==10000,"Completed same-observation original10000 focal registry; no payload read")
    b.need(count["schema"]=="American_C3_full30Mb_observation_count_root_v1" and count["root_authorizer"]=="root_agent"
        and count["source_observation_ref"]==target["observation_ref"] and count["truth_opened"] is False
        and type(count["num_sites"]) is int and count["num_sites"]>0
        and count["num_samples"]==1600 and count["sequence_length"]==30000000,"Actual root count for this exact completed observation")
    b.need(sim["schema"]=="american_c3_simulation_reconstruction_v1" and sim["parent_id"]==obs["parent_id"]
        and sim["split"]==obs["split"] and sim["sequence_length_bp"]==obs["sequence_length_bp"]==30000000
        and sim["recombination_rate"]==1e-8 and sim["sample_unit"]=="haploid"
        and sim["sample_counts"]==obs["counts_haploid"]==dict(ADMIX=1000,AFR=200,ASIA=200,EUR=200)
        and obs["class_order"]==["EUR","EAS","AFR"],"Recorded simulation map and same reference/query sampling")
    b.need(b.ref("research/reviews/FLARE_existing_JDK_source_v1/resource_v1.py")==dict(
        path="research/reviews/FLARE_existing_JDK_source_v1/resource_v1.py",
        sha256="65a3f340a68efc01c9b2ede06c85b95d0284d0d49d16c19cd61b42cf417e9c1c"),"Frozen pure formula source")
    resource=importlib.import_module("research.reviews.FLARE_existing_JDK_source_v1.resource_v1")
    plan=resource.plan_v1(count["num_sites"],600,1000,focal["query_focal_count"])
    b.need(plan["conversion_original_formula_bytes"]+b.AUX_ALLOWANCE<=CAPS["convert"]
        and plan["native_original_formula_with1GiB_tmp_bytes"]+b.AUX_ALLOWANCE<=CAPS["native"]
        and plan["seal_original_formula_without_optional_tract_TSV_bytes"]+b.AUX_ALLOWANCE<=CAPS["seal"],
        "Full actual-site plans plus outer receipts fit exact stage caps; no assumed filter/compression saving")
    return target,obs,count,plan,refs

def conversion_v1(record,target,count,options,projection):
    value=metadata(record)
    b.need(value["status"]=="completed_C3_FLARE_observation_conversion" and value["author_Table1_exact_reproduction"] is False
        and value["observation_ref"]==target["observation_ref"] and value["focal_registry_ref"]==target["focal_registry_ref"]
        and value["simulation_config_ref"]==target["simulation_config_ref"] and value["parent_id"]==target["parent_id"]
        and value["input_sites"]==count["num_sites"] and value["reference_haplotypes"]==600 and value["query_haplotypes"]==1000
        and 0<value["native_expected_markers"]<=count["num_sites"] and value["common_focal_records"]==10000
        and value["unsupported_sites_dropped"]==0 and value["query_ancestry_truth_opened"] is False
        and value["common_focal_registry_unchanged"] is True and value["native_options"]==options
        and value["projection"]==projection and value["native_source_commit"]==COMMIT,
        "Actual unchanged full-site conversion from same observation/focals")
    return value

def config_v1(stage,target,count,plan,args):
    cfg=metadata(TEMPLATES[stage]);cfg.pop("template_only_requires_root_admission")
    cfg.update(count_manifest_ref=target["count_manifest_ref"],focal_registry_ref=target["focal_registry_ref"],
        scientific_role=target["scientific_role"],max_sites_bound=count["num_sites"],external_truth_ref=None)
    extra=[]
    if stage=="convert":
        cfg.update(observation_ref=target["observation_ref"],simulation_config_ref=target["simulation_config_ref"],
            max_sites=count["num_sites"],max_haplotypes=1600)
    elif stage=="native":
        converted=argument_ref(args,"conversion");conversion_v1(converted,target,count,cfg["native_options"],cfg["projection"])
        build_input=metadata(BUILD_INPUT_REF);build_ref=build_input["build_completion_ref"]
        if args.build_completion or args.expected_build_completion_sha256:
            b.need(argument_ref(args,"build_completion")==build_ref,"Use the already successful actual root build")
        build=metadata(build_ref)
        outer_ref=build_input["root_outer_wait_ref"];outer=metadata(outer_ref)
        minimal_ref=build_input["root_actual_direct_wait_ref"];minimal=metadata(minimal_ref)
        b.need(outer["actual_wait_ref"]==minimal_ref and outer["returncode"]==0 and outer["error"] is None
            and outer["observed_owned_groups_ESRCH"] is True and minimal["actual_direct_wait"] is True
            and minimal["returncode"]==0 and minimal["root_build_driver_ref"]==BUILD_DRIVER_REF,
            "Already completed root build actual outer/direct wait0")
        b.need(build["schema"]=="FLARE_actual_source_build_completed_v1"
            and build["status"]=="completed_verified_official_source_build"
            and build["source_ref"]==b.ref("research/reviews/FLARE_existing_JDK_source_v1/build_v1.py"),
            "Actual successful unchanged installed-JDK source build")
        binary_ref=build["binary_provenance_ref"];binary=metadata(binary_ref)
        java_ref=build["java_qualification_ref"];java=metadata(java_ref)
        record_ref=build["build_record_ref"];record=metadata(record_ref)
        b.need(binary["schema"]=="root_qualified_FLARE_binary_v1" and binary["root_authorizer"]=="root_agent"
            and binary["native_sources_modified"] is False and binary["source_commit_build_verified"] is True
            and binary["jar_ref"]==build["jar_ref"] and binary["java_qualification_ref"]==java_ref
            and record["official96Java_verified_before_after_and_inside_JAR"] is True
            and record["native_sources_modified"] is False and record["compiler_release_target"]==11
            and binary["source_commit"]==record["source_commit"]==COMMIT and binary["reported_version"]==VERSION
            and java["schema"]=="root_qualified_installed_JDK_for_FLARE_v1"
            and java["installed_JDK_qualification_ref"] in metadata(b.MANIFEST_REF)["required_metadata_refs"]
            and len(record["tool_waits"])==4 and all(r["returncode"]==0 and r["actual_direct_wait"] is True
                and r["owned_group_ESRCH"] is True for r in record["tool_waits"]),
            "Genuine unmodified official source build/provenance and real tool waits")
        installed_ref=java["installed_JDK_qualification_ref"];installed=metadata(installed_ref)
        qualified=next(r for r in installed["records"] if r["tool"]=="java")
        b.need(java["java_executable"]==qualified["executable"]
            and java["java_executable_sha256"]==qualified["executable_sha256"]
            and java["java_version_output_sha256"]==qualified["version_output_sha256"],
            "Exact existing root-qualified installed Java")
        cfg.update(conversion_ref=converted,binary_provenance_ref=binary_ref,jar_ref=build["jar_ref"],
            java_executable=java["java_executable"],java_executable_sha256=java["java_executable_sha256"],
            java_version_output_sha256=java["java_version_output_sha256"],
            max_native_vcf_bytes=plan["native_VCF_original_formula_bytes"])
        extra=[converted,BUILD_INPUT_REF,build_ref,binary_ref,java_ref,record_ref,installed_ref,outer_ref,minimal_ref]
    else:
        native_ref=argument_ref(args,"native_run");native=metadata(native_ref)
        converted=native["conversion_ref"]
        original=metadata(TEMPLATES["native"])
        conversion_v1(converted,target,count,original["native_options"],cfg["projection"])
        wait_ref=native["native_actual_wait_ref"];wait=metadata(wait_ref)
        b.need(native["status"]=="completed_C3_native_FLARE" and native["author_Table1_exact_reproduction"] is False
            and native["parent_id"]==target["parent_id"] and native["focal_registry_ref"]==target["focal_registry_ref"]
            and native["query_ancestry_truth_opened"] is False and native["native_outputs_modified"] is False
            and native["native_options"]==original["native_options"] and native["projection"]==cfg["projection"]
            and native["source_commit"]==COMMIT and native["native_version"]==VERSION
            and set(native["native_output_refs"])=={".anc.vcf.gz",".model",".global.anc.gz",".log"}
            and wait["actual_direct_native_wait"] is True and wait["returncode"]==0
            and wait["owned_process_group_ESRCH"] is True and wait["operational_failure"] is None,
            "Actual successful native products and direct wait required before sealing")
        cfg.update(native_run_ref=native_ref,max_markers=count["num_sites"],max_focal_records=10000,
            write_derived_marker_tracts=False,max_tract_records=0)
        extra=[native_ref,converted,wait_ref,native["binary_provenance_ref"],TEMPLATES["native"]]
    return cfg,extra

def admit(args):
    b.need(args.stage in CAPS and args.packet and args.output and args.job_id and args.expires_at_utc,"Explicit root stage and fresh scopes")
    b.expiry(args.expires_at_utc)
    frozen=metadata(b.MANIFEST_REF)
    for p,h in frozen["source_sha256"].items():b.need(b.ref(p)==dict(path=p,sha256=h),"Every frozen actual scientific/helper source")
    b.need(b.ref(CONTEXT_REF["path"])==CONTEXT_REF,"Actual separately pinned980GB IO context")
    runtime=b.actual_runtime()
    target_ref=argument_ref(args,"target");target,obs,count,plan,target_refs=target_v1(target_ref)
    cfg,extra=config_v1(args.stage,target,count,plan,args)
    packet,out=b.local(args.packet),b.local(args.output);aux=packet/"worker_aux"
    b.need(not packet.exists() and not out.exists() and not out.is_relative_to(packet)
        and not packet.is_relative_to(out),"Fresh disjoint stage output and metadata packet")
    packet.mkdir(parents=True)
    registration=b.put(packet/"registration.json",dict(schema="FLARE_root_target_registration_v1",root_authorizer="root_agent",
        observation_ref=target["observation_ref"],scientific_role=target["scientific_role"],
        focal_registry_ref=target["focal_registry_ref"],parent_id=target["parent_id"],split=obs["split"],
        target_ref=target_ref,count_manifest_ref=target["count_manifest_ref"],
        observation_completion_ref=target["observation_completion_ref"],focal_completion_ref=target["focal_completion_ref"],
        query_ancestry_truth_opened=False,score_authorized=False,created_utc=b.utc()))
    cfg["registration_ref"]=registration
    cfg_ref=b.put(packet/"config.json",cfg);runtime_ref=b.put(packet/"runtime.json",runtime)
    own=b.ref(SOURCE);m=configured_guard_v1()
    with m.budget.v4.mutex(ROOT):state=m._state_locked(b.CORE,b.BASE,fresh=True)
    resource_refs=b.unique(list(state[3].values()))
    code=dict(frozen["source_sha256"])
    for r in b.unique([own,BUILD_DRIVER_REF,CONTEXT_REF,GUARD_REF,MEMO_REF,V4_REF,NAMESPACE_MEMO_REF,*resource_refs]):
        if Path(r["path"]).suffix==".py":code[r["path"]]=r["sha256"]
    code=dict(sorted(code.items()))
    # Old TRAIN count is only a source-plan basis; the active target gets its actual count.
    meta=b.unique([r for r in frozen["required_metadata_refs"] if r!=frozen["observation_count_manifest_ref"]]
        +[b.MANIFEST_REF,BUILD_DRIVER_REF,TEMPLATES[args.stage],target_ref,*target_refs,registration,*extra,b.RUNTIME_BASIS_REF,
          POLICY_REF,REPAIR_REF,*[r for r in resource_refs if Path(r["path"]).suffix!=".py"]])
    manifest=dict(frozen,source_sha256=code,required_metadata_refs=meta,observation_count_manifest_ref=target["count_manifest_ref"],
        root_stage_driver_ref=own,scientific_target_ref=target_ref,scientific_role=target["scientific_role"])
    manifest_ref=b.put(packet/"source_manifest.json",manifest)
    closure=[dict(path=p,sha256=h) for p,h in code.items()]
    helpers=b.unique([r for r in closure if any(s in r["path"] for s in ("project_budget","failed_scope_capacity","marker_repair","completed_scope_credits"))]+resource_refs+[MEMO_REF,V4_REF,NAMESPACE_MEMO_REF])
    direct=b.unique([own,BUILD_DRIVER_REF,CONTEXT_REF,b.ref(STAGE_SOURCE[args.stage]),cfg_ref,runtime_ref,manifest_ref,
        GUARD_REF,POLICY_REF,REPAIR_REF,*helpers,*closure,*meta])
    consumers=b.unique([*direct,*state[3].values()])
    request=dict(key=args.job_id,output=str(out),auxiliary_scopes=[str(aux)],reserved_bytes=CAPS[args.stage],consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=args.expires_at_utc)
    authority_ref=b.put(packet/"reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=b.CORE,expected_launcher_sha256=b.BASE)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,
        job_id=args.job_id,output_directory=out.relative_to(ROOT).as_posix(),reserved_new_bytes=CAPS[args.stage],
        reservation=receipt["value"],expires_at_utc=args.expires_at_utc,initial_ledger_pins=receipt["initial_ledger_pins"],
        completed_overlay_policy_ref=POLICY_REF,budget_components=dict(implementation=b.ref(m.budget.SOURCE_REL),
        launcher=b.ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD_REF))
    storage_ref=b.put(packet/"storage.json",storage)
    spec=dict(schema="FLARE_existing_JDK_current980_root_admission_v1",root_authorizer="root_agent",execution_authorized=True,
        GPU_authorized=False,stage=args.stage,source_ref=CONTEXT_REF,stage_source_ref=b.ref(STAGE_SOURCE[args.stage]),
        root_stage_driver_ref=own,root_build_driver_ref=BUILD_DRIVER_REF,target_ref=target_ref,
        config_ref=cfg_ref,runtime_ref=runtime_ref,source_manifest_ref=manifest_ref,source_closure_refs=closure,
        storage_ref=storage_ref,job_id=args.job_id,output_directory=storage["output_directory"],
        auxiliary_directory=aux.relative_to(ROOT).as_posix(),expires_at_utc=args.expires_at_utc,
        storage_adapter_ref=GUARD_REF,completed_overlay_policy_ref=POLICY_REF,
        marker_repair_authority_ref=REPAIR_REF,mandatory_helper_refs=helpers,
        consumer_memo_ref=MEMO_REF,original_v4_ref=V4_REF,consumer_namespace_memo_ref=NAMESPACE_MEMO_REF,root_host_resource_enforcement_active=True)
    result=b.put(packet/"admission.json",spec);b.put(packet/"reserve_receipt.json",receipt)
    return dict(status="admitted_"+args.stage+"_not_launched",admission_ref=result,scientific_role=target["scientific_role"],
        reserved_bytes=CAPS[args.stage],consumer_binding_count=len(consumers),score_status="pending_no_truth_opened")

def load_admission(args):
    record=argument_ref(args,"admission");spec=metadata(record)
    b.need(spec["stage"] in CAPS and spec["root_authorizer"]=="root_agent" and spec["execution_authorized"] is True
        and spec["GPU_authorized"] is False and spec["source_ref"]==CONTEXT_REF
        and spec["root_stage_driver_ref"]==b.ref(SOURCE) and spec["root_build_driver_ref"]==BUILD_DRIVER_REF
        and spec.get("consumer_namespace_memo_ref")==NAMESPACE_MEMO_REF,
        "Exact unchanged CPU scientific-stage authority")
    b.expiry(spec["expires_at_utc"]);cfg=metadata(spec["config_ref"]);storage=metadata(spec["storage_ref"])
    manifest=metadata(spec["source_manifest_ref"])
    b.need(spec["source_closure_refs"]==[dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()]
        and storage["reservation"]["auxiliary_scopes"]==[str(b.local(spec["auxiliary_directory"]))],"Exact closure and auxiliary scope")
    refs=b.unique([spec["root_stage_driver_ref"],BUILD_DRIVER_REF,CONTEXT_REF,spec["stage_source_ref"],spec["config_ref"],
        spec["runtime_ref"],spec["source_manifest_ref"],GUARD_REF,POLICY_REF,REPAIR_REF,*spec["mandatory_helper_refs"],
        *spec["source_closure_refs"],*manifest["required_metadata_refs"]])
    b.need(len(storage["reservation"]["consumer_bindings"])<=64,"At most64 actual direct bindings")
    for r in refs:b.need(b.ref(r["path"])==r and r in storage["reservation"]["consumer_bindings"],"Every actual source/runtime/target/predecessor metadata bound")
    target,_,count,plan,_=target_v1(spec["target_ref"])
    b.need(cfg["execution_enabled"] is True and cfg["external_truth_ref"] is None
        and cfg["scientific_role"]==target["scientific_role"] and cfg["count_manifest_ref"]==target["count_manifest_ref"]
        and cfg["focal_registry_ref"]==target["focal_registry_ref"] and cfg["max_sites_bound"]==count["num_sites"]
        and cfg["output_cap_bytes"]==storage["reserved_new_bytes"]==CAPS[spec["stage"]]
        and cfg["root_CPU_threads"]==4 and cfg["root_monitor_RSS_bytes"]==(24 if spec["stage"]=="native" else 4)*1024**3
        and cfg["root_monitor_deadline_seconds"]==(14400 if spec["stage"]=="native" else 3600)
        and storage["accounting_mode"]=="prospective_980gb_v1" and storage["completed_overlay_policy_ref"]==POLICY_REF,
        "Exact admitted active target/resource config and980GB route")
    b.actual_runtime()
    return record,spec,cfg,storage

def worker(args):
    b.need(args.monitor_parent_pid==os.getppid() and args.monitor_read_fd is not None,"Actual outer Popen monitor parent")
    with os.fdopen(args.monitor_read_fd,"rb") as f:ticket=f.read(65)
    b.need(ticket==(args.expected_admission_sha256+"\n").encode(),"Parent seeds owned PID/RSS before release")
    record,spec,_,_=load_admission(args)
    def stop(signum,frame):raise b.WorkerStop("Outer monitor requested stage termination")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    m=configured_guard_v1(consumer=True);context=importlib.import_module("research.reviews.FLARE_consumer_namespace_root_20261015_v1.root_context_980_v1")
    result=context.execute_v1(record,adapter=m)
    print(json.dumps(dict(stage_completion_ref=result)),flush=True);return 0

def launch(args):
    record,spec,cfg,storage=load_admission(args);m=configured_guard_v1(consumer=True)
    m.validate_storage_v9(spec,cfg,storage,source_ref=CONTEXT_REF,config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    aux=b.local(spec["auxiliary_directory"])
    b.need(not aux.exists() and not b.local(spec["output_directory"]).exists(),"Single launch into fresh exclusive scopes")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=b.AUX_ALLOWANCE);aux.mkdir()
    env=os.environ.copy();env.update(b.ENV)
    for name in ("JAVA_TOOL_OPTIONS","_JAVA_OPTIONS","JDK_JAVA_OPTIONS"):env.pop(name,None)
    runtime=metadata(spec["runtime_ref"]);rfd,wfd=os.pipe()
    argv=[runtime["executable"],"-B","-m","research.reviews.FLARE_consumer_namespace_root_20261015_v1.root_stage_980_driver","--worker","--admission",record["path"],
        "--expected-admission-sha256",record["sha256"],"--monitor-parent-pid",str(os.getpid()),"--monitor-read-fd",str(rfd)]
    started=b.utc();clock=time.monotonic();owned={};groups={};peak=races=discarded=0;log=bytearray();error=None
    proc=None;selector=None;minimal_ref=None;rc=None;term_at=None;killed=False;seed_ref=None;released=False
    def retain(chunk):
        nonlocal discarded
        keep=min(len(chunk),b.LOG_LIMIT-len(log));log.extend(chunk[:keep]);discarded+=len(chunk)-keep
    try:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(rfd,))
        row=b.process_info(proc.pid)
        b.need(row is not None and row["pgid"]==proc.pid and row["session"]==proc.pid,"Seed actual owned PID identity before child release")
        owned[proc.pid]=row["start_ticks"];groups[proc.pid]={proc.pid:row["start_ticks"]}
        initial,misses=b.observe(owned,groups);peak=initial;races+=misses
        b.need(initial<=cfg["root_monitor_RSS_bytes"],"Initial aggregate worker/monitor RSS before child compute")
        seed_ref=b.put(aux/"launch_monitor_seed.json",dict(schema="FLARE980_precompute_owned_monitor_seed_v1",
            worker_pid=proc.pid,worker_start_ticks=row["start_ticks"],worker_pgid=row["pgid"],worker_session=row["session"],
            supervisor_pid=os.getpid(),initial_aggregate_RSS_bytes=initial,actual_owned_roster_seeded=True,
            child_compute_released=False,admission_ref=record,checked_utc=b.utc()))
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        os.write(wfd,(record["sha256"]+"\n").encode());released=True
        os.close(rfd);rfd=None;os.close(wfd);wfd=None
        while proc.poll() is None:
            now=time.monotonic()
            try:
                rss,misses=b.observe(owned,groups);peak=max(peak,rss);races+=misses
                b.need(rss<=cfg["root_monitor_RSS_bytes"],"Aggregate owned-tree plus monitor RSS soft bound exceeded")
                b.need(now-clock<=cfg["root_monitor_deadline_seconds"],"Outer stage wall deadline exceeded")
                b.expiry(spec["expires_at_utc"])
            except BaseException as exc:
                if error is None:error=dict(type=type(exc).__name__,message=str(exc)[:2048])
            if error is not None and term_at is None:proc.terminate();term_at=now
            if term_at is not None and now-term_at>=40 and not killed:
                b.signal_owned(groups,signal.SIGKILL);proc.kill();killed=True
            for key,_ in selector.select(.25):
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
    except BaseException as exc:
        error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:
            try:proc.terminate()
            except ProcessLookupError:pass
    finally:
        for fd in (rfd,wfd):
            if fd is not None:
                try:os.close(fd)
                except OSError as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        if proc is not None:
            try:rc=proc.wait(timeout=40 if error else 5)
            except subprocess.TimeoutExpired:
                try:b.signal_owned(groups,signal.SIGKILL)
                except BaseException as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
                try:proc.kill()
                except ProcessLookupError:pass
                rc=proc.wait()
            # First postwait disk write, before group checks or risky guard/log writes.
            minimal_ref=b.put(aux/"actual_wait_minimal.json",dict(schema="FLARE_scientific_stage_actual_direct_wait_minimal_v1",
                actual_direct_wait=True,returncode=rc,pid=proc.pid,worker_pgid=proc.pid,admission_ref=record,
                root_stage_driver_ref=spec["root_stage_driver_ref"],started_utc=started,waited_utc=b.utc(),
                elapsed_seconds=time.monotonic()-clock),limit=8192)
    if proc is None:
        b.put(aux/"spawn_failed.json",dict(actual_direct_wait=False,error=error));raise RuntimeError("Popen failed; no actual wait exists")
    live=b.absent_groups(groups)
    if live:
        b.signal_owned(groups,signal.SIGKILL);until=time.monotonic()+10
        while live and time.monotonic()<until:time.sleep(.1);live=b.absent_groups(groups)
    if selector is not None:
        for _ in range(16):
            ready=selector.select(0)
            if not ready:break
            for key,_ in ready:
                chunk=os.read(key.fd,65536)
                if chunk:retain(chunk)
                else:selector.unregister(key.fileobj)
        selector.close()
    proc.stdout.close()
    b.need(not live,"Observed owned group unresolved; preserve minimal wait and perform no guard/log writes")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=len(log)+65536)
    logpath=aux/"worker_stdout.txt"
    fd=os.open(logpath,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(log);f.flush();os.fsync(f.fileno())
    terminal=b.put(aux/"outer_wait.json",dict(schema="FLARE_scientific_stage_outer_monitor_wait_v1",actual_wait_ref=minimal_ref,
        returncode=rc,error=error,admission_ref=record,argv=argv,launch_monitor_seed_ref=seed_ref,child_compute_released=released,
        observed_owned_groups_ESRCH=True,observed_owned_groups=sorted(groups),
        observed_process_identities=owned,observed_peak_aggregate_RSS_bytes=peak,RSS_limit_bytes=cfg["root_monitor_RSS_bytes"],
        RSS_bound_is_polling_soft_bound_not_kernel_quota=True,aggregate_includes_outer_monitor=True,poll_seconds=.25,
        proc_exit_or_task_read_races=races,unobserved_between_poll_processes_possible=True,
        native_inference_group_separately_supervised_by_frozen_runner=True,version_probe_uses_original_subprocess_run=True,
        wall_deadline_seconds=cfg["root_monitor_deadline_seconds"],
        termination_grace_seconds=40,elapsed_seconds=time.monotonic()-clock,retained_stdout_ref=b.ref(logpath),discarded_stdout_bytes=discarded))
    m.budget_check_v9(storage,boundary=True)
    b.need(rc==0 and error is None,"Actual stage worker failed; preserve all partial artifacts and real wait")
    completion=b.local(spec["output_directory"])/"completed.json";b.need(completion.exists(),"Actual unchanged stage completion")
    return dict(status="completed_"+spec["stage"]+"_worker",completion_ref=b.ref(completion),actual_wait_ref=minimal_ref,
        outer_wait_ref=terminal,scientific_role=cfg["scientific_role"],score_status="pending_no_truth_opened",automatic_completed_scope_credit=False)

def main():
    global b
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    for name in ("admit","launch","worker"):mode.add_argument("--"+name,action="store_true")
    p.add_argument("--stage",choices=tuple(CAPS))
    for name in ("packet","output","job-id","expires-at-utc","target","expected-target-sha256","conversion",
        "expected-conversion-sha256","build-completion","expected-build-completion-sha256","native-run",
        "expected-native-run-sha256","admission","expected-admission-sha256"):p.add_argument("--"+name)
    p.add_argument("--monitor-parent-pid",type=int);p.add_argument("--monitor-read-fd",type=int)
    args=p.parse_args();b=base()
    result=admit(args) if args.admit else launch(args) if args.launch else worker(args)
    if not args.worker:print(json.dumps(result,sort_keys=True),flush=True)
    return result if args.worker else 0

if __name__=="__main__":raise SystemExit(main())

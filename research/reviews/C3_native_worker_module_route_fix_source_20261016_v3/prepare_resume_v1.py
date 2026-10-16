"""Deferred root metadata/admission materials only. Import is stdlib and inert.
No guard, reserve, launch, process probe, scientific tensor or checkpoint reader.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy,hashlib,importlib.util,json,math,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_native_worker_module_route_fix_source_20261016_v3"
SOURCE=HERE+"/prepare_resume_v1.py"
OLD_INPUT_REF={'path': 'research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_C3_GPU0_train_v1/input.json', 'sha256': 'ed223e3bc46a47a07e999a0c4599ee8a150c3fc237c9ac8880b131bfc2dae0dc'}
OLD_CAP_REF={'path': 'research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_controller_v1/pool_capability.root_active.json', 'sha256': 'c88db41d22769af8cf7a02ce09134a0c0ac8ddc95006875d60b6d61a4c10e852'}
GPU0_ADMISSION_REF={'path': 'research/reviews/GPU0_shared_admission_draft_20261016_v1/root_admission_draft_v1.py', 'sha256': '48bfab0e57aab5f97f761a16a62ec96a01bf19af8b59e352050f27822f9146c4'}
INTEGRATION_REF={'path': 'research/reviews/C3_efficient_resume_migration_source_20261016_v1/resume_integration_v1.py', 'sha256': '988c3ede4bf96bbd24917f7a724e879edcaf6bb891f9b2c3f057ad6bb2706f6f'}
RESUME_RUNTIME_REF={'path': 'research/reviews/C3_efficient_resume_migration_source_20261016_v1/root_resume_runtime_v1.py', 'sha256': '41367588489a2b2451156f881b2b824824955493b8172f85817d9a0b79387cb6'}
VALIDATOR_REF={'path': 'research/reviews/C3_efficient_resume_migration_source_20261016_v1/checkpoint_migration_validator_v1.py', 'sha256': '31414598d0f69533864cf600a9a6f0c24ef5f0d81f9ca131fb5f9612f761fafd'}
STEP_REF={'path': 'research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/efficient_step_v1.py', 'sha256': '0df05b1957affc222ca1c11d9f34cc3277f5a3098834fb36fedee64bb242d2e8'}
CACHE_REF={'path': 'research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/single_parent_device_cache_v1.py', 'sha256': 'd19879458474883e125b7efc7cbd649839f25c3db22ecb876ba3f7f5ab9b49b6'}
FINITE_POOL_REF={'path': 'research/reviews/C3_native_worker_module_route_fix_source_20261016_v3/gpu0_declared_pool_v1.py', 'sha256': 'e98dc3f5b0c1aa969f9f640bc1318f538686ac9fdf2543be60fe9733ad419818'}
FINITE_CONTRACT_REF={'path': 'research/reviews/C3_native_worker_module_route_fix_source_20261016_v3/finite_jobs_v1.py', 'sha256': 'e8fd72a31fb7df623053b666f1ba145c66f8e61d7e778e8869ef4966c3d91581'}
FINITE_LAUNCHER_REF={'path': 'research/reviews/C3_native_worker_module_route_fix_source_20261016_v3/launch_declared_gpu0_v1.py', 'sha256': 'e5acb649ad1e449a3f5ce52b2a0257aff52ebf84bf7927168d5d03af516e4c24'}
MIB=2**20
GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"

def need(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    p=ROOT/path
    need(type(path) is str and not Path(path).is_absolute() and ".." not in Path(path).parts
        and p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=32*MIB,
        "Canonical bounded source/JSON metadata only")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(r):
    need(isinstance(r,dict) and set(r)=={"path","sha256"} and ref(r["path"])==r
        and Path(r["path"]).suffix==".json","Exact existing JSON evidence, no scientific payload")
    return json.loads((ROOT/r["path"]).read_text())

def load(r,label):
    need(ref(r["path"])==r,"Pinned source before inert metadata import")
    sp=importlib.util.spec_from_file_location(label,ROOT/r["path"])
    m=importlib.util.module_from_spec(sp);sys.modules[label]=m;sp.loader.exec_module(m)
    return m

def unique(rows,*,limit=64):
    result={}
    for r in rows:
        need(set(r)=={"path","sha256"} and Path(r["path"]).suffix in (".py",".sh",".json")
            and ref(r["path"])==r,"Exact actual source or JSON ref")
        need(r["path"] not in result or result[r["path"]]==r,"No conflicting bindings")
        result[r["path"]]=r
    need(limit is None or len(result)<=limit,"Actual full required union exceeds original64direct consumers")
    return list(result.values())

def original_contract_v1():
    # Existing reviewed helper changes only private GPU0 eligibility/TMP handling.
    return load(GPU0_ADMISSION_REF,"_C3_metadata_existing_GPU0_admission").c3_contract()

def scope_names_v1(names):
    need(set(names)=={"job_id","packet_directory","output_directory","short_tmp_directory"},"Exact explicit fresh names")
    need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",names["job_id"]) is not None,"Safe logical job id")
    for k in ("packet_directory","output_directory","short_tmp_directory"):
        x=Path(names[k]);need(not x.is_absolute() and ".." not in x.parts and x.as_posix()==names[k],"Contained fresh namespace")
    need(names["packet_directory"].startswith(HERE+"/") and names["output_directory"].startswith("experiments/results/")
        and names["short_tmp_directory"].startswith(".runtime/tmp/gpu0_shared/"),"One owned fresh scientific namespace")
    return dict(names)

def count_budget_v1(*,retained_scheduled_checkpoints=1):
    # Output planning only. Original unchanged guard remains the actual fail-stop.
    telemetry=600*(8192+8192)
    need(type(retained_scheduled_checkpoints) is int and 1<=retained_scheduled_checkpoints<=5,
        "At least one genuine retained scheduled candidate is required for this32MiB recipe")
    maximum_new_checkpoints=6-retained_scheduled_checkpoints+1  # All remaining scheduled files plus one failure boundary.
    checkpoints=maximum_new_checkpoints*(1572864+8192)
    initialization=8*MIB
    control=512*1024
    total=telemetry+checkpoints+initialization+control
    need(total<=28*MIB,"Conservative named products fit prospective phase allowance")
    return dict(schema="C3_resume_count_derived_32MiB_scientific_plan_v1",
        reserved_new_bytes=32*MIB,phase_write_budget_bytes=28*MIB,
        temp_allowance_bytes=MIB,outer_auxiliary_cap_bytes=MIB,root_packet_overhead_cap_bytes=2*MIB,
        maximum_progress_records=600,maximum_serialized_progress_bytes=8192,
        guarded_overhead_per_output_bytes=8192,maximum_new_checkpoint_files=maximum_new_checkpoints,
        genuine_original_scheduled_candidates_retained=retained_scheduled_checkpoints,
        per_checkpoint_serialized_planning_allowance_bytes=1572864,
        runtime_initialization_planning_bytes=initialization,other_phase_control_planning_bytes=control,
        planned_phase_charge_bytes=total,checkpoint_sizes_are_prospective_not_measured_bounds=True,
        actual_original_guard_enforces_complete32MiB_scope=True)

def prepare_resume_v1(old_input_ref,root_sealed_migration_manifest_ref,names):
    need(old_input_ref==OLD_INPUT_REF,"Exact original live-at-registration input lineage")
    old=read(old_input_ref);names=scope_names_v1(names)
    integration=load(INTEGRATION_REF,"_C3_metadata_original_resume_integration")
    raw=read(root_sealed_migration_manifest_ref)
    manifest,intent=integration.read_resume_manifest_v1(root_sealed_migration_manifest_ref,raw["bounds_ref"])
    need(intent["old_input_ref"]==old_input_ref and intent["old_job_id"]==old["job_id"]
        and manifest["runtime_source_ref"]==RESUME_RUNTIME_REF
        and names["job_id"]!=old["job_id"] and names["output_directory"]!=old["output_directory"],
        "Genuine consumed migration and fresh full-state continuation identity")
    for other in (old["output_directory"],intent["migration_output_directory"]):
        a,b=ROOT/names["output_directory"],ROOT/other
        need(a!=b and not a.is_relative_to(b) and not b.is_relative_to(a),"Fresh training output outside old and migrated outputs")
    req=read(old["root_stage_request_ref"]);caps=read(old["caps_ref"])
    budget=count_budget_v1(retained_scheduled_checkpoints=len(manifest["original_scheduled_checkpoints"]))
    caps=copy.deepcopy(caps)
    for key in ("phase_write_budget_bytes","temp_allowance_bytes","outer_auxiliary_cap_bytes","root_packet_overhead_cap_bytes"):
        caps[key]=budget[key]
    caps.update(reservation_cap_bytes=32*MIB,outer_log_cap_bytes=256*1024)
    need(caps["max_cpu_rss_bytes"]==old["max_cpu_rss_bytes"]==128*2**30
        and caps["max_cuda_reserved_bytes"]==old["max_cuda_reserved_bytes"]==40*2**30
        and caps["operational_deadline_seconds"]==86400,"Original scientific CUDA/RSS/deadline preserved")
    c=original_contract_v1()
    c.validate_caps(req,caps,activated=True)  # Only existing source/plan JSON checks.
    scientific=read(req["scientific_training_config_ref"])
    need(scientific["epochs"]==30 and scientific["parents_per_epoch"]==20 and scientific["total_updates"]==600
        and scientific["focals_per_parent"]==10000 and scientific["model_kwargs"]["expected_haplotypes"]==1600,
        "Exact20parents/1600whole-tree/10000focals/600effective-update science")
    sources=dict(old["source_sha256"])
    needed=[RESUME_RUNTIME_REF,INTEGRATION_REF,VALIDATOR_REF,STEP_REF,CACHE_REF,
        manifest["bounds_ref"],manifest["qualification_ref"],root_sealed_migration_manifest_ref,
        manifest["migration_intent_ref"],GPU0_ADMISSION_REF,FINITE_POOL_REF,FINITE_LAUNCHER_REF,ref(SOURCE)]
    if intent["rollback_authorization_ref"] is not None:needed.append(intent["rollback_authorization_ref"])
    for r in needed:need(ref(r["path"])==r,"Actual complete new source/evidence family");sources[r["path"]]=r["sha256"]
    return dict(schema="C3_resume_scientific_metadata_materials_v1",names=names,old_input_ref=old_input_ref,
        old_input=old,old_request=req,caps=caps,budget=budget,source_sha256=sources,
        migration_manifest_ref=root_sealed_migration_manifest_ref,manifest=manifest,intent=intent,
        required_resume_refs=needed,actual_admission=False,actual_execution=False,
        original_RNG_AdamW_history_and_prefix_preserved_by_original_resume_runtime=True)

def worker_request_v1(materials,actual_caps_ref):
    need(read(actual_caps_ref)==materials["caps"],"Actually saved exact new operational caps")
    req=copy.deepcopy(materials["old_request"])
    req.update(caps_ref=actual_caps_ref,operational_short_tmp_directory=materials["names"]["short_tmp_directory"])
    need(req["physical_gpu"]==0 and req["GPU_UUID"]==GPU0_UUID,"GPU0-only actual original request")
    return req

def operational_config_v1(materials,actual_worker_request_ref):
    req=read(actual_worker_request_ref)
    need(req==worker_request_v1(materials,req["caps_ref"]),"Exact copied original bindings and new caps/TMP")
    return original_contract_v1().make_config(req,materials["caps"])

def admission_authorization_v1(materials,*,created_at_utc,expires_at_utc):
    start=datetime.fromisoformat(created_at_utc);end=datetime.fromisoformat(expires_at_utc)
    need(start.tzinfo is not None and end.tzinfo is not None and 0<(end-start).total_seconds()<=86400,
        "Fresh actual root admission lifetime, no old native expiry change")
    return dict(schema="C3_fresh_resumed_scientific_admission_authorization_v1",active=True,root_authorizer="root_agent",
        fresh_scientific_reservation_authorized=True,old_input_ref=materials["old_input_ref"],
        root_sealed_migration_manifest_ref=materials["migration_manifest_ref"],names=materials["names"],
        created_at_utc=created_at_utc,expires_at_utc=expires_at_utc,reserved_new_bytes=32*MIB,
        original_charge_or_file_or_expiry_mutation_authorized=False,native_input_or_charge_or_expiry_mutation_authorized=False)

def source_closure_v1(materials,actual_worker_request_ref,actual_admission_authorization_ref):
    req=read(actual_worker_request_ref);authorization=read(actual_admission_authorization_ref)
    need(authorization["names"]==materials["names"] and authorization["root_sealed_migration_manifest_ref"]==materials["migration_manifest_ref"]
        and authorization["fresh_scientific_reservation_authorized"] is True,"Actual fresh root admission authorization")
    return dict(schema="C3_GNN_prefetch_ramcache_isolated980_transitive_closure_v1",
        source_sha256=materials["source_sha256"],root_stage_request_ref=actual_worker_request_ref,
        bindings=req["bindings"],scientific_train_source_ref=req["scientific_train_source_ref"],
        hook_ref=req["hook_ref"],native_MLM_dependency=False,operational_shared_route_ref=actual_admission_authorization_ref)

def direct_scientific_consumers_v1(materials,actual_refs,*,actual_guard_bindings):
    need(set(actual_refs)=={"caps","request","config","runtime","closure","admission_authorization"},"Actual saved metadata refs only")
    old=materials["old_input"];storage=read(old["refs"]["storage"])
    replaced={
        old["caps_ref"]["path"]:actual_refs["caps"],
        old["root_stage_request_ref"]["path"]:actual_refs["request"],
        old["refs"]["config"]["path"]:actual_refs["config"],
        old["refs"]["runtime"]["path"]:actual_refs["runtime"],
        old["transitive_source_closure_ref"]["path"]:actual_refs["closure"],
        old["actual_launcher_ref"]["path"]:FINITE_LAUNCHER_REF,
        old["operational_pool_source_ref"]["path"]:FINITE_POOL_REF,
        old["operational_root_entry_ref"]["path"]:ref(SOURCE),
        old["operational_shared_route_ref"]["path"]:actual_refs["admission_authorization"],
    }
    # Preserve every actual old required binding, rebasing only changed metadata
    # and honest operational route fields. Re-add the original runtime/adapter
    # because the unchanged worker still directly consults them.
    rows=[replaced.get(r["path"],r) for r in storage["reservation"]["consumer_bindings"]]
    rows += [old["refs"]["runtime"],old["operational_GPU0_runtime_ref"],
        *materials["required_resume_refs"],*actual_refs.values(),*actual_guard_bindings]
    return unique(rows)

def finish_scientific_input_v1(materials,actual_refs,actual_storage_ref,*,actual_guard_bindings):
    authorization=read(actual_refs["admission_authorization"]);old=materials["old_input"]
    storage=read(actual_storage_ref);names=materials["names"]
    expected=direct_scientific_consumers_v1(materials,actual_refs,actual_guard_bindings=actual_guard_bindings)
    need(storage["kind"]=="project_budget_v9_storage_record" and storage["accounting_mode"]=="prospective_980gb_v1"
        and storage["active"] is True and storage["consumer_uses_v9_guard"] is True
        and storage["job_id"]==names["job_id"] and storage["output_directory"]==names["output_directory"]
        and storage["reserved_new_bytes"]==32*MIB and storage["reservation"]["consumer_bindings"]==expected
        and storage["reservation"]["output"]==str(ROOT/names["output_directory"])
        and set(storage["reservation"]["auxiliary_scopes"])=={
            str(ROOT/names["packet_directory"]/"worker_aux"),str(ROOT/names["short_tmp_directory"])}
        and storage["expires_at_utc"]==authorization["expires_at_utc"],"Genuine fresh exact original-guard32MiB reservation")
    need(read(actual_refs["config"])==operational_config_v1(materials,actual_refs["request"])
        and read(actual_refs["runtime"])==read(old["refs"]["runtime"])
        and read(actual_refs["closure"])==source_closure_v1(materials,actual_refs["request"],actual_refs["admission_authorization"]),
        "Actual exact saved config/runtime/source closure")
    spec=copy.deepcopy(old)
    spec.update(job_id=names["job_id"],output_directory=names["output_directory"],
        root_stage_request_ref=actual_refs["request"],caps_ref=actual_refs["caps"],
        transitive_source_closure_ref=actual_refs["closure"],source_sha256=materials["source_sha256"],
        root_auxiliary_directory=names["packet_directory"]+"/worker_aux",
        created_at_utc=authorization["created_at_utc"],expires_at_utc=authorization["expires_at_utc"],
        efficient_training_caps_ref=materials["manifest"]["bounds_ref"],
        efficient_resume_manifest_ref=materials["migration_manifest_ref"],
        operational_GPU0_runtime_ref=RESUME_RUNTIME_REF,
        actual_launcher_ref=FINITE_LAUNCHER_REF,operational_launcher_ref=FINITE_LAUNCHER_REF,
        operational_pool_source_ref=FINITE_POOL_REF,operational_root_entry_ref=ref(SOURCE),
        operational_shared_route_ref=actual_refs["admission_authorization"],
        operational_launch_mode="GPU0_finite_full_state_C3_resume_v1",
        refs=dict(config=actual_refs["config"],runtime=actual_refs["runtime"],storage=actual_storage_ref),
        resumed_full_state_lineage=dict(old_input_ref=materials["old_input_ref"],migration_manifest_ref=materials["migration_manifest_ref"],
            fresh_weights_or_optimizer_reset=False,effective_total_updates=600,retained_prefix_updates=materials["manifest"]["actual_global_step"]))
    # All original scientific sources/bindings/config remain present. Only
    # explicitly listed operational/input lineage fields change in this copy.
    need(spec["max_cpu_rss_bytes"]==128*2**30 and spec["max_cuda_reserved_bytes"]==40*2**30
        and spec["physical_gpu"]==0,"Unchanged whole-process scientific caps")
    return spec

def c3_method_v1(actual_input_ref):
    spec=read(actual_input_ref);slot=load(FINITE_CONTRACT_REF,"_C3_actual_finite_contract").catalogue_v1()["slots"]["C3_RESUME_TRAIN"]
    return dict(worker_kind="C3",module=slot["module"],module_ref=slot["module_ref"],runtime_entry_ref=RESUME_RUNTIME_REF,
        input_ref=actual_input_ref,storage_ref=spec["refs"]["storage"],cuda_caps_ref=spec["caps_ref"],
        job_id=spec["job_id"],cuda_cap_bytes=24*2**30,cpu_rss_cap_bytes=128*2**30,
        operational_deadline_seconds=86400,log_limit_bytes=256*1024,margin_mib=2048,peak_mib=26624,
        env={},args=["--execute","--input",actual_input_ref["path"],"--expected-input-sha256",actual_input_ref["sha256"]])

def native_method_unchanged_v1():
    prior=read(OLD_CAP_REF)
    rows=[x for x in prior["jobs"].values() if x["worker_kind"]=="native_ARGformer"]
    need(len(rows)==1,"One original never-started native job")
    return copy.deepcopy(rows[0])

"""Finite source-bound job contracts; metadata only, no guard/probe/admission."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,math,os,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_native_worker_module_route_fix_source_20261016_v3"
CATALOGUE=REL+"/finite_catalogue.v1.json"
SOURCE=REL+"/finite_jobs_v1.py"
POOL=REL+"/gpu0_declared_pool_v1.py"
LAUNCHER=REL+"/launch_declared_gpu0_v1.py"
DISPATCHER=POOL
BOOTSTRAP=LAUNCHER
MIB=2**20
NAME=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    require(isinstance(path,str) and not Path(path).is_absolute() and ".." not in Path(path).parts,"Contained relative source/metadata path")
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),
        "Canonical bound source/metadata")
    require(p.stat().st_size<=32*2**20,"Bounded source/metadata, no scientific payload")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and isinstance(record["sha256"],str)
        and len(record["sha256"])==64 and all(c in "0123456789abcdef" for c in record["sha256"]),"Exact path/SHA256 reference")
    require(ref(record["path"])==record,"Exact authenticated source/metadata bytes")
    return json.loads((ROOT/record["path"]).read_bytes())

def expiry(value):
    parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    require(parsed.tzinfo is not None,"Absolute timezone-aware authority")
    return parsed.timestamp()

def direct_bindings(storage,required):
    rows=storage["reservation"]["consumer_bindings"]
    require(isinstance(rows,list) and len(rows)<=64 and all(r in rows for r in required),
        "Original64 direct-binding ceiling; every actual facade/runtime/science ref bound")
    for row in rows:ref(row["path"]);require(ref(row["path"])==row,"Exact direct binding")
    return rows

def catalogue_v1():return json.loads((ROOT/CATALOGUE).read_bytes())

# One narrow additional metadata contract. All sources remain stdlib-only.
RESUME_RUNTIME_REF={"path":"research/reviews/C3_efficient_resume_migration_source_20261016_v1/root_resume_runtime_v1.py","sha256":"41367588489a2b2451156f881b2b824824955493b8172f85817d9a0b79387cb6"}
RESUME_INTEGRATION_REF={"path":"research/reviews/C3_efficient_resume_migration_source_20261016_v1/resume_integration_v1.py","sha256":"988c3ede4bf96bbd24917f7a724e879edcaf6bb891f9b2c3f057ad6bb2706f6f"}
RESUME_VALIDATOR_REF={"path":"research/reviews/C3_efficient_resume_migration_source_20261016_v1/checkpoint_migration_validator_v1.py","sha256":"31414598d0f69533864cf600a9a6f0c24ef5f0d81f9ca131fb5f9612f761fafd"}
STRICT_RESUME_CUDA_BYTES=24*2**30
ORIGINAL_C3_CUDA_BYTES=40*2**30
ORIGINAL_C3_RSS_BYTES=128*2**30

def load_resume_metadata_v1():
    require(ref(RESUME_INTEGRATION_REF["path"])==RESUME_INTEGRATION_REF,"Exact reviewed full-state resume integration")
    spec=importlib.util.spec_from_file_location("_finite_C3_resume_metadata_only",ROOT/RESUME_INTEGRATION_REF["path"])
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def resume_manifest_parts_v1(record):
    raw=read(record)
    integration=load_resume_metadata_v1()
    manifest,intent=integration.read_resume_manifest_v1(record,raw["bounds_ref"])
    bounds=integration.read_bounds_v1(raw["bounds_ref"])
    require(manifest["runtime_source_ref"]==RESUME_RUNTIME_REF
        and manifest["integration_source_ref"]==RESUME_INTEGRATION_REF
        and manifest["migration_validator_ref"]==RESUME_VALIDATOR_REF
        and bounds["memory_caps"]["max_successor_cuda_reserved_bytes"]==STRICT_RESUME_CUDA_BYTES,
        "Only exact reviewed full-state migration and strict24GiB efficient C3 ceiling")
    terminal=read(intent["old_terminal_observation_ref"])
    return integration,manifest,intent,bounds,terminal

def resume_transition_refs_v1(transition_ref):
    proof=read(transition_ref)
    record=proof.get("C3_resume_manifest_ref")
    if record is None:return []
    integration,manifest,intent,bounds,terminal=resume_manifest_parts_v1(record)
    rows=[record,manifest["migration_intent_ref"],manifest["bounds_ref"],manifest["qualification_ref"],
        intent["old_terminal_observation_ref"],intent["old_input_ref"],
        RESUME_RUNTIME_REF,RESUME_INTEGRATION_REF,RESUME_VALIDATOR_REF,
        integration.STEP_REF,integration.CACHE_REF]
    if intent["rollback_authorization_ref"] is not None:rows.append(intent["rollback_authorization_ref"])
    return [r for i,r in enumerate(rows) if r not in rows[:i]]

def validate_interrupted_C3_row_v1(proof,row,requested):
    """Actual failed/interrupted C3 remains failed, never relabelled success."""
    record=proof.get("C3_resume_manifest_ref")
    require(record is not None,"Explicit genuine full-state migration proof for interrupted original C3")
    _,manifest,intent,bounds,terminal=resume_manifest_parts_v1(record)
    require(requested["worker_kind"]=="C3" and requested["input_ref"]==intent["old_input_ref"]
        and requested["job_id"]==intent["old_job_id"]==row["logical_job"]
        and row["requested"]==requested and row["state"]=="failed"
        and type(row.get("exit_code")) is int and row["exit_code"]!=0
        and row.get("leader_wait_status_known") is True and row.get("owned_drain_observed") is True,
        "Exact genuine nonzero old C3 terminal and owned drain, preserved as interrupted")
    require(terminal.get("schema")=="C3_actual_stopped_boundary_transition_evidence_v1"
        and terminal.get("root_authorizer")=="root_agent"
        and terminal.get("old_job_id")==requested["job_id"]
        and terminal.get("old_input_ref")==requested["input_ref"]
        and terminal.get("actual_source_job_terminal_consumed") is True
        and terminal.get("actual_source_exit_code")==row["exit_code"]
        and terminal.get("all_old_owned_processes_drained") is True
        and terminal.get("all_old_future_write_relaunch_capabilities_revoked") is True
        and terminal.get("actual_boundary_ref")==manifest["old_boundary_ref"]
        and terminal.get("actual_boundary_update_count")==manifest["actual_global_step"]
        and any(p.get("pid")==terminal.get("actual_worker_pid")
            and p.get("start_ticks")==terminal.get("actual_worker_start_ticks")
            for p in row["process_identities"]),
        "Same actual original worker, source terminal, retained boundary and explicit old-writer revocation")
    require(0<manifest["actual_global_step"]<600 and manifest["remaining_updates"]==600-manifest["actual_global_step"]
        and manifest["total_planned_updates"]==600 and manifest["optimizer_updates_performed"]==0,
        "Strict complete-state retained prefix plus remaining suffix to600, no restart")

def validate_C3_resume_method_v1(cap,proof,final_rows,method,spec,caps,science,request,sources,admin):
    require(cap.get("efficient_C3_full_state_resume_authorized") is True
        and proof.get("C3_resume_manifest_ref")==spec.get("efficient_resume_manifest_ref"),
        "Explicit finite full-state C3 resume slot and same migration as old-controller transition")
    integration,manifest,intent,bounds,terminal=resume_manifest_parts_v1(spec["efficient_resume_manifest_ref"])
    old=read(intent["old_input_ref"])
    row=final_rows[intent["old_job_id"]]
    validate_interrupted_C3_row_v1(proof,row,row["requested"])
    require(method["runtime_entry_ref"]==RESUME_RUNTIME_REF
        and spec["efficient_training_caps_ref"]==manifest["bounds_ref"]
        and method["cuda_cap_bytes"]==request["declared_limits"]["cuda_cap_bytes"]==STRICT_RESUME_CUDA_BYTES
        and caps["max_cuda_reserved_bytes"]==spec["max_cuda_reserved_bytes"]==ORIGINAL_C3_CUDA_BYTES
        and method["cpu_rss_cap_bytes"]==caps["max_cpu_rss_bytes"]==spec["max_cpu_rss_bytes"]==ORIGINAL_C3_RSS_BYTES,
        "Additional24GiB abort/scheduling cap and original40GiB worker guard are distinct and unchanged")
    require(method["job_id"]!=intent["old_job_id"] and spec["output_directory"]!=old["output_directory"]
        and not (ROOT/spec["output_directory"]).is_relative_to(ROOT/old["output_directory"])
        and not (ROOT/old["output_directory"]).is_relative_to(ROOT/spec["output_directory"]),
        "Fresh resumed scientific job/output outside preserved original run")
    required=resume_transition_refs_v1(cap["prior_controller_completion_ref"])
    require(all(r in sources and r in request["runtime_source_refs"] for r in required),
        "Complete actual migration/intent/terminal/bounds/qualification/source family in new operational closure")
    direct_bindings(admin,required)
    # This exact reviewed runtime authenticates all original science caps/data
    # and direct bindings without importing Torch or reading checkpoint payloads.
    require(ref(RESUME_RUNTIME_REF["path"])==RESUME_RUNTIME_REF,"Exact reviewed efficient resume runtime")
    rs=importlib.util.spec_from_file_location("_finite_exact_C3_resume_runtime_metadata",ROOT/RESUME_RUNTIME_REF["path"])
    runtime=importlib.util.module_from_spec(rs);sys.modules[rs.name]=runtime;rs.loader.exec_module(runtime)
    runtime.authenticate_binding_v1(method,spec)

def validate_previous_controller_v1(record):
    """This distinct successor accepts only the actual H050 full-science/CLI1 branch."""
    require(read(record).get("schema")==UNKNOWN_SCHEMA,
        "Distinct H050 failed-CLI transport proof required, never old success-only authority")
    return validate_transport_unknown_prior_v1(record)

def validate_native_reuse_v1(record,method_job,transition_ref,final_rows):
    cat=catalogue_v1();proof=read(record);spec=read(method_job["input_ref"])
    require(method_job["input_ref"]==cat["known_unused_native_input_ref"]
        and proof.get("schema")=="GPU0_never_started_native_input_reuse_v1"
        and proof.get("root_authorizer")=="root_agent" and proof.get("reuse_authorized") is True
        and proof.get("prior_controller_completion_ref")==transition_ref,
        "Only exact retained never-started native input reuse, after actual controller wait")
    row=final_rows[method_job["job_id"]]
    require(row["state"]=="cancelled" and row["pid"] is None and row["process_identities"]==[]
        and row.get("leader_start_ticks") is None and row.get("started_unix") is None,
        "Genuine never-started queue cancellation; no interrupted science reuse")
    require(row["requested"]==method_job and row["requested"]["input_ref"]==method_job["input_ref"]
        and row["requested"]["storage_ref"]==method_job["storage_ref"]
        and row["requested"]["cuda_caps_ref"]==method_job["cuda_caps_ref"]
        and proof.get("unchanged_scientific_input_ref")==method_job["input_ref"]
        and proof.get("unchanged_scientific_storage_ref")==method_job["storage_ref"]
        and proof.get("unchanged_caps_ref")==method_job["cuda_caps_ref"]
        and proof.get("unchanged_job_id")==method_job["job_id"]
        and proof.get("unchanged_absolute_expiry")==spec["expires_at_utc"]
        and proof.get("repeat_scientific_storage_reservation") is False,
        "Original job/input/storage/caps/absoluteexpiry unchanged; never repeat7GiB charge")

def envelope_parts_v1(envelope_ref):
    envelope=read(envelope_ref)
    require(envelope.get("schema")=="GPU0_declared_operational_envelope_v1"
        and envelope.get("status")=="active" and envelope.get("root_authorizer")=="root_agent",
        "Fresh separately bound operational envelope")
    request=read(envelope["request_ref"])
    require(request.get("schema")=="GPU0_declared_operational_request_v1"
        and request.get("root_authorizer")=="root_agent" and request.get("status")=="active","Explicit finite root request")
    return envelope,request,request["method_job"]

def job_from_envelope_v1(envelope_ref):
    envelope,request,method=envelope_parts_v1(envelope_ref)
    limits=request["declared_limits"]
    return dict(worker_kind=method["worker_kind"],logical_slot=request["slot"],job_id=method["job_id"],
        module=method["module"],module_ref=method["module_ref"],runtime_entry_ref=ref(DISPATCHER),
        input_ref=envelope_ref,storage_ref=envelope["operational_storage_ref"],
        cuda_caps_ref=method["cuda_caps_ref"],cuda_cap_bytes=limits["cuda_cap_bytes"],
        cpu_rss_cap_bytes=limits["cpu_rss_cap_bytes"],operational_deadline_seconds=limits["operational_deadline_seconds"],
        margin_mib=2048,peak_mib=math.ceil(limits["cuda_cap_bytes"]/MIB)+2048,
        log_limit_bytes=limits["log_limit_bytes"],env=dict(method["env"]),
        args=["--execute","--input",envelope_ref["path"],"--expected-input-sha256",envelope_ref["sha256"]])

def job_expiry_v2(outer_job):
    """Authenticated existing per-method bound; no authority or clock is changed."""
    envelope,request,method=envelope_parts_v1(outer_job["input_ref"])
    require(job_from_envelope_v1(outer_job["input_ref"])==outer_job
        and envelope["operational_storage_ref"]==outer_job["storage_ref"],
        "Exact predeclared outer job for per-method expiry")
    spec=read(method["input_ref"])
    envelope_end=expiry(envelope["expires_at_utc"]);method_end=expiry(spec["expires_at_utc"])
    require(envelope_end<=method_end,"Operational envelope cannot extend scientific expiry")
    return envelope_end

def validate_jobs_v1(cap,common_required,admin,env_keys):
    cat=catalogue_v1();jobmap=read(cap["finite_job_map_ref"])
    proof,final_rows=validate_previous_controller_v1(cap["prior_controller_completion_ref"])
    require(jobmap.get("schema")=="GPU0_root_bound_finite_job_map_v1"
        and jobmap.get("status")=="active" and jobmap.get("root_authorizer")=="root_agent"
        and isinstance(jobmap.get("jobs"),dict) and 1<=len(jobmap["jobs"])<=len(cat["slots"]),
        "Finite explicitly root-bound job map")
    require(set(jobmap["jobs"])==set(cap["jobs"]),"No arbitrary undeclared logical job")
    require(len(jobmap["jobs"])==2 and {r["slot"] for r in jobmap["jobs"].values()}=={"C3_RESUME_TRAIN","NATIVE_MLM"},
        "Only the original native method and full-state C3 continuation, never H050 replay")
    if proof.get("C3_resume_manifest_ref") is not None:
        require(sum(v.get("slot")=="C3_RESUME_TRAIN" for v in jobmap["jobs"].values())==1,
            "An interrupted C3 handover must actually select its sole full-state continuation")
    sources=list(cap["source_refs"].values())
    direct_bindings(admin,common_required+[cap["plan_ref"],cap["finite_job_map_ref"],cap["prior_controller_completion_ref"]])
    ids={admin["job_id"]};outputs={admin["output_directory"]};used_slots=set()
    for name,record in jobmap["jobs"].items():
        require(isinstance(name,str) and NAME.fullmatch(name),"Safe finite logical job identity")
        require(record["slot"] in cat["slots"] and record["slot"] not in used_slots,"One job per exact finite scientific slot")
        used_slots.add(record["slot"]);contract=cat["slots"][record["slot"]]
        envelope,request,method=envelope_parts_v1(record["operational_input_ref"])
        require(request["slot"]==record["slot"] and method["job_id"]==name
            and job_from_envelope_v1(record["operational_input_ref"])==cap["jobs"][name],
            "Exact operational envelope and declared launch fields")
        for key in ("worker_kind","module","module_ref"):require(method[key]==contract[key],"Frozen scientific entry: "+key)
        require(ref(method["runtime_entry_ref"]["path"])==method["runtime_entry_ref"]
            and method["runtime_entry_ref"] in sources and method["module_ref"] in sources,
            "Actual exact runtime and science source in root closure")
        if contract.get("known_runtime_ref") is not None:
            require(method["runtime_entry_ref"]==contract["known_runtime_ref"],"Exact preserved per-method runtime")
        if contract.get("only_unchanged_admitted_input_ref"):
            require(method["input_ref"]==contract["only_unchanged_admitted_input_ref"],"No new/repeated native MLM scientific input or charge")
        spec=read(method["input_ref"]);caps=read(method["cuda_caps_ref"]);science=read(method["storage_ref"])
        for path,sha in spec.get("source_sha256",{}).items():
            require(dict(path=path,sha256=sha) in sources,"Honest unchanged original input source closure in actual controller plan")
        require(spec.get("status")=="active" and spec.get("root_authorizer")=="root_agent"
            and spec["job_id"]==science["job_id"]==name and spec["phase"]==contract["phase"]
            and spec["physical_gpu"]==0 and spec["refs"]["storage"]==method["storage_ref"]
            and spec["output_directory"]==science["output_directory"],"Unmodified original method identity/phase/storage")
        if "split" in contract:require(spec.get("split")==contract["split"],"Exact held-out TGB split slot")
        require(method["args"]==["--execute","--input",method["input_ref"]["path"],
            "--expected-input-sha256",method["input_ref"]["sha256"]],"Literal original method dispatcher arguments")
        limits=request["declared_limits"]
        for jobkey,capkey in (("cuda_cap_bytes","max_cuda_reserved_bytes"),("cpu_rss_cap_bytes","max_cpu_rss_bytes")):
            value=limits[jobkey]
            minimum=0 if jobkey=="cuda_cap_bytes" and contract.get("zero_cuda_cap_allowed_if_explicit") else 1
            if record["slot"]=="C3_RESUME_TRAIN" and jobkey=="cuda_cap_bytes":
                require(type(value) is int and value==method[jobkey]==STRICT_RESUME_CUDA_BYTES
                    and caps[capkey]==spec[capkey]==ORIGINAL_C3_CUDA_BYTES,
                    "Exact24GiB additional logical cap, original40GiB scientific CUDA guard retained")
            else:
                require(type(value) is int and value>=minimum and value==method[jobkey]==caps[capkey]==spec[capkey],
                    "Explicit exact original method cap, never inferred: "+jobkey)
        for key,ceilingkey in (("cuda_cap_bytes","known_cuda_ceiling_bytes"),("cpu_rss_cap_bytes","known_cpu_ceiling_bytes")):
            ceiling=contract[ceilingkey]
            require(ceiling is None or limits[key]<=ceiling,"Original reviewed method ceiling retained")
        require(method["margin_mib"]==2048 and method["peak_mib"]==math.ceil(limits["cuda_cap_bytes"]/MIB)+2048
            and limits["operational_deadline_seconds"]==method["operational_deadline_seconds"]
            and type(limits["operational_deadline_seconds"]) is int and limits["operational_deadline_seconds"]>0,
            "Declared exact deadline and unchanged2GiB margin")
        if "operational_deadline_seconds" in caps:
            require(limits["operational_deadline_seconds"]==caps["operational_deadline_seconds"],"Original phase deadline unchanged")
        if method["worker_kind"] in ("H050","C3"):
            require(limits["operational_deadline_seconds"]<=86400,"Original H050/C3 outer deadline ceiling")
        elif method["worker_kind"]=="native_ARGformer":
            require(limits["operational_deadline_seconds"]<=259200 and cap.get("native_40GiB_operational_ceiling_authorized") is True
                and limits["cuda_cap_bytes"]==40*2**30
                and (limits["operational_deadline_seconds"]<=86400 or cap.get("native_72h_operational_deadline_authorized") is True),
                "Original explicit native40GiB/atmost72h authority")
        for key,value in contract.get("required_exact_limits",{}).items():
            require(limits[key]==value,"Exact reviewed runtime limit remains explicit: "+key)
        if contract.get("required_scientific_storage_bytes") is not None:
            require(science["reserved_new_bytes"]==contract["required_scientific_storage_bytes"]
                and science["reservation"]["auxiliary_scopes"]==[],"Exact separately charged scientific output scope")
        # No numeric default is inferred from the scientific100epoch budget.
        require(type(limits["log_limit_bytes"]) is int and limits["log_limit_bytes"]>0
            and set(method["env"]).issubset(env_keys) and all(isinstance(x,str) for x in method["env"].values()),
            "Bounded explicit logs and original private environment keys")
        require(request.get("actual_pool_ref")==ref(POOL) and request.get("actual_launcher_ref")==ref(LAUNCHER)
            and request.get("actual_dispatcher_ref")==ref(DISPATCHER)
            and request.get("unchanged_scientific_input_ref")==method["input_ref"],
            "Honest new operational route versus immutable scientific input")
        require(request.get("prior_controller_completion_ref")==cap["prior_controller_completion_ref"],
            "Every future job uses same actual old-controller completion")
        require(expiry(envelope["expires_at_utc"])<=expiry(spec["expires_at_utc"])
            and expiry(spec["expires_at_utc"])<=cap["expires_unix"],
            "Each unchanged method/envelope authority fits controller; no scientific expiry extension")
        scope=read(envelope["operational_storage_ref"])
        require(type(scope["reserved_new_bytes"]) is int and 0<scope["reserved_new_bytes"]<=64*2**20
            and scope["reservation"]["auxiliary_scopes"]==[],"Explicit count-derived small outer operational cap; no minimum/default inferred")
        require(scope["job_id"] not in ids and scope["job_id"]!=name
            and scope["output_directory"] not in outputs and scope["output_directory"]!=science["output_directory"],
            "Small separate operational charge/output; scientific charge never repeated")
        ids.add(scope["job_id"]);outputs.add(scope["output_directory"])
        require(name not in ids and science["output_directory"] not in outputs,"Distinct immutable scientific scopes")
        ids.add(name);outputs.add(science["output_directory"])
        runtime_refs=request["runtime_source_refs"]
        require(isinstance(runtime_refs,list) and method["runtime_entry_ref"] in runtime_refs
            and method["module_ref"] in runtime_refs and all(r in sources for r in runtime_refs),
            "Every actual declared runtime/science source authenticated")
        for r in runtime_refs:require(ref(r["path"])==r,"Exact actual runtime/science source bytes")
        direct_bindings(scope,common_required+runtime_refs+[envelope["request_ref"],method["input_ref"],
            method["storage_ref"],method["cuda_caps_ref"],cap["prior_controller_completion_ref"]])
        # Original science reservation/bindings remain validated by the original
        # method runtime/guard. No source map or64-binding rule is rewritten.
        require(len(science["reservation"]["consumer_bindings"])<=64,"Original scientific64binding ceiling unchanged")
        require(all(r in runtime_refs for r in contract.get("required_runtime_source_refs",[])),
            "Exact reviewed per-slot runtime dependency closure")
        if contract.get("science_handoff_ref"):
            handoff=read(contract["science_handoff_ref"])
            required=contract["required_science_source_refs"]+[contract["science_handoff_ref"]]
            require(all(r in runtime_refs for r in required),"Frozen TGB3arm protocol/body/metric closure exact")
            require(spec.get("frozen_scientific_handoff_ref")==contract["science_handoff_ref"],"Exact frozen TGB campaign")
        if record["slot"]=="C3_RESUME_TRAIN":
            validate_C3_resume_method_v1(cap,proof,final_rows,method,spec,caps,science,request,sources,admin)
        if method["input_ref"]==cat["known_unused_native_input_ref"]:
            require(record["slot"]=="NATIVE_MLM" and request.get("native_never_started_reuse_ref") is not None,
                "Retained native MLM needs explicit never-started reuse evidence")
            validate_native_reuse_v1(request["native_never_started_reuse_ref"],method,
                cap["prior_controller_completion_ref"],final_rows)
            direct_bindings(scope,[request["native_never_started_reuse_ref"]])
    return jobmap


TRANSPORT_HELPER_REF={'path': 'research/reviews/C3_bootstrap_failed_controller_handover_source_20261016_v2/transport_handover_v1.py', 'sha256': 'f5ccbb9750b504af7b8f9be7a1692995e3a5358905698bc127cec06415e70d9b'}
HISTORICAL_TRANSPORT_HELPER_REF={'path': 'research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/transport_handover_v1.py', 'sha256': 'd0c6f150cdf4b987e67a1dd95c9918e4467e601e467c7705d4a7d6fabd80dc4e'}
HANDLE_LOSS_REF={"path":"research/reviews/GPU0_shared_root_integration_20261016_v1/observation_handle_loss_20261016_v1.json","sha256":"cb1b825cdbae6c74d74035ad00c8627ed46eff8979d2849837e1981ec6cc6e1e"}
OLD_C3_CUDA_REF={"path":"experiments/results/C3_GNN_cachev3_GPU0_train_v1/cuda_qualification/qualification.json","sha256":"7e5fddc163aa76ad66ae97ed63200e62957211b29881e7b65c3fdbd6bcef4ffd"}
OLD_H050_CUDA_REF={"path":"experiments/results/H050_GPU0_direct_readout_train_seed101_v1/cuda_qualification/qualification.json","sha256":"279c701d8120e21a5fd8998a5e74a57427373dbb95c1c96e0a89c92771167b2f"}
UNKNOWN_SCHEMA="GPU0_prior_controller_transport_unknown_H050_failed_CLI_handover_v1"
FAILED_CLI_VALIDATOR_REF={'path': 'research/reviews/H050_CPU_evaluation_failed_CLI_successor_source_20261016_v1/failed_cli_terminal_v1.py', 'sha256': 'db29f245b4c37d85404592bdbe8809c6b15bfc8c17557d9819185b4f11ed536f'}

def transport_helper_v1():
    require(ref(TRANSPORT_HELPER_REF["path"])==TRANSPORT_HELPER_REF,"Exact reviewed transport handover helper")
    spec=importlib.util.spec_from_file_location("_finite_transport_unknown_handover",ROOT/TRANSPORT_HELPER_REF["path"])
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def validate_logical_wait_records_v1(final):
    for row in final["jobs"]:
        if row["pid"] is None:continue
        require(row.get("leader_wait_status_known") is True and row.get("owned_drain_observed") is True
            and type(row.get("exit_code")) is int,"Every actual started logical worker has known wait/drain")
        ids={(p["pid"],p["start_ticks"]) for p in row["process_identities"]}
        for w in row["descendant_waits"]:
            require(w.get("wait_status_known") is True and type(w.get("raw_wait_status")) is int
                and type(w.get("returncode")) is int and os.waitstatus_to_exitcode(w["raw_wait_status"])==w["returncode"]
                and (w["pid"],w["start_ticks"]) in ids,"Authenticated actual adopted wait and saved PID/start")
            if row["requested"]["worker_kind"]=="H050":
                require(w["returncode"]==0,"Every H050 adopted worker actually succeeded")
        require(type(row.get("finished_unix")) in (int,float)
            and row["finished_unix"]<=final["updated_unix"],"Final registry follows actual logical terminal")
    require(final.get("safety_error") is None,"No unresolved controller supervision error")
    transport_helper_v1().known_roster_v1(final)

def validate_loss_revocation_v1(request,ready,final):
    require(request.get("observation_handle_loss_ref")==HANDLE_LOSS_REF
        and request.get("original_external_tool_session_id")==23022
        and request.get("original_external_tool_exit_status")=="UNKNOWN"
        and request.get("original_external_tool_exit_code") is None
        and request.get("original_external_tool_terminal_consumed") is False
        and request.get("original_controller_outer_wait0_claimed") is False
        and request.get("actual_controller_terminal_ref") is None
        and request.get("transport_unknown_closed_transition_explicitly_authorized") is True,
        "Explicit exact old transport loss, never invent original outer0")
    loss=read(HANDLE_LOSS_REF)
    require(loss["tool_session"]==23022 and "Unknown process id" in loss["tool_error"]
        and loss["model_terminal_claimed"] is False and ready["exec_session_id"]==23022
        and ready["controller_identity"]["pid"]==3118924
        and ready["controller_identity"]["start_ticks"]==478196608
        and any(p["pid"]==3118924 and p["start_ticks"]==478196608 for p in loss["actual_remote_identities"]),
        "Narrow original session/controller and genuine saved missing-handle evidence")
    revoke=read(request["explicit_old_capability_revocation_ref"])
    require(revoke.get("schema")=="GPU0_old_controller_permanent_capability_revocation_v1"
        and revoke.get("root_authorizer")=="root_agent" and revoke.get("active") is True
        and revoke.get("permanent") is True
        and revoke.get("prior_capability_ref")==request["prior_capability_ref"]
        and revoke.get("prior_ready_ref")==request["prior_ready_ref"]
        and revoke.get("controller_identity")==ready["controller_identity"]
        and revoke.get("all_old_controller_start_submit_requeue_worker_write_paths_revoked") is True
        and revoke.get("new_scientific_or_restart_authority_granted") is False,
        "Explicit permanent original capability revocation, no new science permission")
    c3=read(OLD_C3_CUDA_REF);h050=read(OLD_H050_CUDA_REF)
    old_c3=next(r for r in final["jobs"] if r["requested"]["worker_kind"]=="C3")
    old_h050=next(r for r in final["jobs"] if r["requested"]["worker_kind"]=="H050")
    for q,row in ((c3,old_c3),(h050,old_h050)):
        require(any(p["pid"]==q["pid"] for p in row["process_identities"]),"Same actual original qualified worker")
    a,b=c3["lease"],h050["lease"]
    require(a["lease_device"]==b["lease_device"]==67
        and a["lease_inode"]==b["lease_inode"]==9257095304247255386
        and a["lease_metadata_sha256"]==b["lease_metadata_sha256"]==ready["lease_sha256"]==final["lease_sha256"]
        and a["launcher_metadata"]==b["launcher_metadata"]==final["lease_metadata"]
        and a["lease_path"]==b["lease_path"]==ready["lease_path"]==final["lease_path"]
        and a["physical_gpu"]==b["physical_gpu"]==0
        and a["launcher_metadata"]["launcher_pid"]==ready["controller_identity"]["pid"],
        "Both original worker qualifications pin the exact actual old GPU0 lifetime inode")
    return dict(device=a["lease_device"],inode=a["lease_inode"],
        metadata_sha256=a["lease_metadata_sha256"],metadata=a["launcher_metadata"])

def observe_transport_unknown_closed_v1(request_ref,*,resource_check):
    require(callable(resource_check),"Root guarded CPU resource callback required")
    resource_check("before_transport_unknown_closed_metadata")
    request,rows=validate_transport_unknown_request_v1(request_ref)
    ready=read(request["prior_ready_ref"]);final=read(request["final_registry_ref"])
    expected=validate_loss_revocation_v1(request,ready,final)
    observation=transport_helper_v1().observe_v1(request_ref,final,expected,TRANSPORT_HELPER_REF,
        resource_check=resource_check)
    observation["request_validator_ref"]=ref(SOURCE)
    return observation

def normalize_actual_probe_session_v1(value):
    """A positive actual tool session, normalized before old-session exclusion."""
    require((type(value) is int and value>0)
        or (type(value) is str and re.fullmatch(r"[1-9][0-9]*",value) is not None),
        "Actual positive integer session or canonical ASCII decimal session string")
    result=int(value)
    require(result!=23022,"New metadata probe cannot reuse original lost session23022")
    return result

def validate_transport_unknown_prior_v1(proof_ref):
    proof=read(proof_ref)
    require(proof.get("schema")==UNKNOWN_SCHEMA and proof.get("status")=="closed_lifecycle_transport_unknown"
        and proof.get("root_authorizer")=="root_agent","Honest transport-UNKNOWN closed-lifecycle proof")
    request_ref=proof["transport_unknown_request_ref"]
    request,rows=validate_transport_unknown_request_v1(request_ref)
    keys=("prior_capability_ref","prior_ready_ref","final_registry_ref","old_controller_future_writes_revoked",
        "C3_resume_manifest_ref","original_external_tool_session_id","original_external_tool_exit_status",
        "original_external_tool_exit_code","original_external_tool_terminal_consumed","original_controller_outer_wait0_claimed","actual_controller_terminal_ref",
        "observation_handle_loss_ref","explicit_old_capability_revocation_ref",
        "H050_failed_CLI_certificate_ref","H050_full_science_actual_CLI_exit1_explicitly_authorized",
        "H050_training_replay_authorized","H050_success0_or_completed_scope_credit_claimed")
    require(all(proof.get(k)==request.get(k) for k in keys),"Exact separately bound closed request, never mutate original events")
    ready=read(request["prior_ready_ref"]);final=read(request["final_registry_ref"])
    expected=validate_loss_revocation_v1(request,ready,final);helper=transport_helper_v1()
    observation=read(proof["transport_unknown_observation_ref"])
    helper.validate_observation_v1(observation,request_ref,helper.known_roster_v1(final),expected,HISTORICAL_TRANSPORT_HELPER_REF)
    require(observation.get("request_validator_ref")=={'path': 'research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/finite_jobs_v1.py', 'sha256': '8dcd1e5fd3b79e59baea6378d5eeb7ad485e59478cd9d8943aca4669bf9568b1'},"Exact preserved historical request validator source")
    tool=read(proof["probe_actual_tool_terminal_ref"])
    require(tool.get("schema")=="GPU0_transport_unknown_probe_actual_tool_terminal_v1"
        and tool.get("root_authorizer")=="root_agent" and tool.get("actual_terminal_consumed") is True
        and type(tool.get("actual_exit_code")) is int and tool["actual_exit_code"]==0
        and normalize_actual_probe_session_v1(tool.get("exec_session_id"))>0
        and tool.get("terminal_chunk_id")
        and tool.get("observation_ref")==proof["transport_unknown_observation_ref"]
        and tool.get("request_ref")==request_ref and tool.get("probe_source_ref")==HISTORICAL_TRANSPORT_HELPER_REF
        and tool.get("metadata_observation_not_original_controller_exit") is True
        and tool.get("original_external_tool_exit_status")=="UNKNOWN"
        and tool.get("original_external_tool_exit_code") is None,
        "Actual NEW metadata probe tool0, never relabel as original controller outer0")
    if proof.get("bootstrap_failed_controller_case_v1") is not None:
        helper.validate_bootstrap_case_v1(proof_ref,proof,sys.modules[__name__])
    return proof,rows

def transport_transition_refs_v1(transition_ref):
    proof=read(transition_ref)
    if proof.get("schema")!=UNKNOWN_SCHEMA:return []
    return [TRANSPORT_HELPER_REF,HANDLE_LOSS_REF,OLD_C3_CUDA_REF,OLD_H050_CUDA_REF,
        proof["transport_unknown_request_ref"],proof["transport_unknown_observation_ref"],
        proof["probe_actual_tool_terminal_ref"],proof["explicit_old_capability_revocation_ref"],
        FAILED_CLI_VALIDATOR_REF,proof["H050_failed_CLI_certificate_ref"]]

def recheck_transport_unknown_before_start_v1(transition_ref,*,resource_check):
    proof,rows=validate_transport_unknown_prior_v1(transition_ref)
    if proof.get("bootstrap_failed_controller_case_v1") is not None:
        return transport_helper_v1().recheck_bootstrap_before_start_v1(
            transition_ref,proof,sys.modules[__name__],resource_check=resource_check)
    observation=observe_transport_unknown_closed_v1(proof["transport_unknown_request_ref"],resource_check=resource_check)
    request=read(proof["transport_unknown_request_ref"])
    expected=validate_loss_revocation_v1(request,read(request["prior_ready_ref"]),read(request["final_registry_ref"]))
    return observation,expected

def validate_transport_unknown_request_v1(record):
    """Root-authenticated actual terminal/drain metadata, never an absence probe."""
    cat=catalogue_v1();proof=read(record)
    require(proof.get("schema")=="GPU0_transport_unknown_H050_failed_CLI_closed_handover_request_v1"
        and proof.get("status")=="active" and proof.get("root_authorizer")=="root_agent"
        and proof.get("prior_capability_ref")==cat["known_prior_controller_capability_ref"]
        and proof.get("prior_ready_ref")==cat["known_prior_controller_ready_ref"],
        "Actual previous controller lineage")
    ready=read(proof["prior_ready_ref"])
    final=read(proof["final_registry_ref"])
    validate_loss_revocation_v1(proof,ready,final)
    require(final.get("capability_ref")==proof["prior_capability_ref"]
        and final.get("controller")==ready["controller_identity"] and final.get("physical_gpu")==0
        and final.get("lifecycle")=="closed" and not final.get("ambiguous_descendants_or_lease_holders")
        and proof.get("old_controller_future_writes_revoked") is True,
        "Closed actual registry, no ambiguous ownership, permanent old write revocation")
    rows={row["logical_job"]:row for row in final["jobs"]}
    prior=read(proof["prior_capability_ref"])
    require(len(rows)==len(final["jobs"]) and set(rows)==set(prior["jobs"])
        and all(row["requested"]==prior["jobs"][name] for name,row in rows.items()),
        "Exact complete immutable old declared-job rows, without duplicate or missing identity")
    require(proof.get("H050_full_science_actual_CLI_exit1_explicitly_authorized") is True
        and proof.get("H050_training_replay_authorized") is False
        and proof.get("H050_success0_or_completed_scope_credit_claimed") is False
        and proof.get("C3_resume_manifest_ref") is not None,
        "Explicit genuine failed-CLI science, no H050 replay/credit, actual full-state C3 migration")
    h050=failed_cli_validator_v1().validate_certificate(proof["H050_failed_CLI_certificate_ref"])
    require(h050["certificate"]["capability_ref"]==proof["prior_capability_ref"]
        and h050["registry"]["controller"]==ready["controller_identity"],
        "Same exact original shared controller and H050 failed-CLI certificate")
    for row in rows.values():
        if row["pid"] is None:
            require(row["state"]=="cancelled" and row.get("leader_start_ticks") is None
                and row["process_identities"]==[] and row.get("started_unix") is None,
                "Only explicitly cancelled never-started queued work may remain")
        elif row["requested"]["worker_kind"]=="C3" and proof.get("C3_resume_manifest_ref") is not None:
            validate_interrupted_C3_row_v1(proof,row,row["requested"])
        elif row["requested"]["worker_kind"]=="H050":
            require(row==h050["job"] and row["state"]=="failed" and row["exit_code"]==1,
                "Exact actual certified H050 failed row preserved in final closed registry")
        else:
            require(row["state"]=="succeeded" and row["exit_code"]==0
                and row.get("leader_wait_status_known") is True and row.get("owned_drain_observed") is True,
                "Every started logical job genuinely waited and drained successfully")
    trained={j["worker_kind"]:name for name,j in prior["jobs"].items() if j["worker_kind"] in ("H050","C3")}
    require(set(trained)=={"H050","C3"} and rows[trained["H050"]]["state"]=="failed"
        and rows[trained["H050"]]["exit_code"]==1
        and (rows[trained["C3"]]["state"]=="succeeded" if proof.get("C3_resume_manifest_ref") is None
             else rows[trained["C3"]]["state"]=="failed"),
        "Genuine H050 full scientific budget with actual CLI1, and explicit fully migrated interrupted C3")
    require(final.get("schema")=="GPU0_shared_logical_pool_registry_v1"
        and final.get("pool")==prior["pool"]
        and final.get("pool_job")=="gpu0_shared_v1:"+prior["pool"]+":gpu0",
        "Exact original closed lifecycle registry and shared physical0 pool")
    validate_logical_wait_records_v1(final)
    return proof,rows


def failed_cli_validator_v1():
    """Exact independently sealed typed evidence validator, imported only on root invocation."""
    require(ref(FAILED_CLI_VALIDATOR_REF["path"])==FAILED_CLI_VALIDATOR_REF,
        "Exact distinct H050 full-science actual-failed-CLI validator")
    spec=importlib.util.spec_from_file_location("_finite_H050_actual_failed_CLI_evidence",ROOT/FAILED_CLI_VALIDATOR_REF["path"])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

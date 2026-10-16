"""Finite source-bound job contracts; metadata only, no guard/probe/admission."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,math,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/GPU0_declared_C3_resume_pool_source_20261016_v1"
CATALOGUE=REL+"/finite_catalogue.v1.json"
SOURCE=REL+"/finite_jobs_v1.py"
POOL=REL+"/gpu0_declared_pool_v1.py"
LAUNCHER=REL+"/launch_declared_gpu0_v1.py"
DISPATCHER=REL+"/declared_runtime_v1.py"
BOOTSTRAP=REL+"/bootstrap_declared_v1.py"
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
    """Root-authenticated actual terminal/drain metadata, never an absence probe."""
    cat=catalogue_v1();proof=read(record)
    require(proof.get("schema")=="GPU0_prior_controller_closed_for_declared_pool_v1"
        and proof.get("status")=="completed" and proof.get("root_authorizer")=="root_agent"
        and proof.get("prior_capability_ref")==cat["known_prior_controller_capability_ref"]
        and proof.get("prior_ready_ref")==cat["known_prior_controller_ready_ref"],
        "Actual previous controller lineage")
    ready=read(proof["prior_ready_ref"]);terminal=read(proof["actual_controller_terminal_ref"])
    final=read(proof["final_registry_ref"])
    require(terminal.get("actual_exit_code")==0 and terminal.get("actual_terminal_consumed") is True
        and terminal.get("exec_session_id")==ready["exec_session_id"]
        and terminal.get("controller_identity")==ready["controller_identity"],
        "Root actual old-controller outer wait0 and exact identity")
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
    for row in rows.values():
        if row["pid"] is None:
            require(row["state"]=="cancelled" and row.get("leader_start_ticks") is None
                and row["process_identities"]==[] and row.get("started_unix") is None,
                "Only explicitly cancelled never-started queued work may remain")
        elif row["requested"]["worker_kind"]=="C3" and proof.get("C3_resume_manifest_ref") is not None:
            validate_interrupted_C3_row_v1(proof,row,row["requested"])
        else:
            require(row["state"]=="succeeded" and row["exit_code"]==0
                and row.get("leader_wait_status_known") is True and row.get("owned_drain_observed") is True,
                "Every started logical job genuinely waited and drained successfully")
    trained={j["worker_kind"]:name for name,j in prior["jobs"].items() if j["worker_kind"] in ("H050","C3")}
    require(set(trained)=={"H050","C3"} and rows[trained["H050"]]["state"]=="succeeded"
        and (rows[trained["C3"]]["state"]=="succeeded" if proof.get("C3_resume_manifest_ref") is None
             else rows[trained["C3"]]["state"]=="failed"),
        "Genuine H050 completion, and either completed C3 or explicit fully migrated interrupted C3")
    return proof,rows

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

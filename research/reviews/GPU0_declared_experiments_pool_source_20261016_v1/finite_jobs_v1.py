"""Finite source-bound job contracts; metadata only, no guard/probe/admission."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,re
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/GPU0_declared_experiments_pool_source_20261016_v1"
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
        else:
            require(row["state"]=="succeeded" and row["exit_code"]==0
                and row.get("leader_wait_status_known") is True and row.get("owned_drain_observed") is True,
                "Every started logical job genuinely waited and drained successfully")
    trained={j["worker_kind"]:name for name,j in prior["jobs"].items() if j["worker_kind"] in ("H050","C3")}
    require(set(trained)=={"H050","C3"} and all(rows[name]["state"]=="succeeded" for name in trained.values()),
        "Current H050/C3 TRAIN jobs complete before future controller")
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

def validate_jobs_v1(cap,common_required,admin,env_keys):
    cat=catalogue_v1();jobmap=read(cap["finite_job_map_ref"])
    proof,final_rows=validate_previous_controller_v1(cap["prior_controller_completion_ref"])
    require(jobmap.get("schema")=="GPU0_root_bound_finite_job_map_v1"
        and jobmap.get("status")=="active" and jobmap.get("root_authorizer")=="root_agent"
        and isinstance(jobmap.get("jobs"),dict) and 1<=len(jobmap["jobs"])<=len(cat["slots"]),
        "Finite explicitly root-bound job map")
    require(set(jobmap["jobs"])==set(cap["jobs"]),"No arbitrary undeclared logical job")
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
        require(expiry(spec["expires_at_utc"])>=cap["expires_unix"]
            and expiry(envelope["expires_at_utc"])>=cap["expires_unix"]
            and expiry(envelope["expires_at_utc"])<=expiry(spec["expires_at_utc"]),
            "No method/native absoluteexpiry extension through new facade")
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
        if method["input_ref"]==cat["known_unused_native_input_ref"]:
            require(record["slot"]=="NATIVE_MLM" and request.get("native_never_started_reuse_ref") is not None,
                "Retained native MLM needs explicit never-started reuse evidence")
            validate_native_reuse_v1(request["native_never_started_reuse_ref"],method,
                cap["prior_controller_completion_ref"],final_rows)
            direct_bindings(scope,[request["native_never_started_reuse_ref"]])
    return jobmap

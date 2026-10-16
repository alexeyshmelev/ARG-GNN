"""One genuine H050 full-science / failed-CLI outcome. Never a success0 or storage-credit certificate."""
from pathlib import Path
import ast,hashlib,json,math,os,re,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H050_CPU_evaluation_failed_CLI_successor_source_20261016_v1"
SOURCE=HERE+"/failed_cli_terminal_v1.py"
CERT_SCHEMA="H050_full_science_actual_failed_CLI_terminal_certificate_v1"
TOOL_SCHEMA="GPU0_shared_client_actual_tool_terminal_v1"
REG_SCHEMA="GPU0_shared_logical_pool_registry_v1"
TRAIN_POOL_REF=dict(path="research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_shared_pool_v1.py",sha256="28bb690f33c2384d04f88dbacff1f910694c398c81874f9b9751336f4fb6429c")

def require(ok,message):
    if not ok:raise ValueError(message)

def is_ref(r):
    return isinstance(r,dict) and set(r)=={"path","sha256"} and isinstance(r["path"],str) and isinstance(r["sha256"],str) and len(r["sha256"])==64 and all(c in "0123456789abcdef" for c in r["sha256"])

def path(name):
    require(isinstance(name,str) and not Path(name).is_absolute() and ".." not in Path(name).parts,"Contained relative path")
    p=ROOT/name
    require(p.resolve()==p and p.is_relative_to(ROOT) and not p.is_symlink(),"Canonical non-symlink metadata")
    return p

def ref(name):
    p=path(name);require(p.is_file(),"Existing metadata")
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(r):
    require(is_ref(r) and ref(r["path"])==r,"Exact immutable metadata bytes")
    data=path(r["path"]).read_bytes();require(len(data)<=33554432,"Bounded metadata")
    return json.loads(data)

def identity(v):
    return isinstance(v,dict) and type(v.get("pid")) is int and v["pid"]>0 and type(v.get("uid")) is int and v["uid"]>=0 and type(v.get("start_ticks")) is int and v["start_ticks"]>0

def completed_science_failed_cli_job(job):
    require(job.get("state")=="failed" and type(job.get("exit_code")) is int and job["exit_code"]==1
        and job.get("leader_wait_status_known") is True and job.get("owned_drain_observed") is True,
        "Actual failed CLI leader wait1 and owned drain")
    require(type(job.get("pid")) is int and job["pid"]>0 and type(job.get("leader_start_ticks")) is int and job["leader_start_ticks"]>0,
        "Actual leader PID/start identity")
    ids=job.get("process_identities")
    require(isinstance(ids,list) and ids and all(identity(v) for v in ids)
        and any(v["pid"]==job["pid"] and v["start_ticks"]==job["leader_start_ticks"] for v in ids),
        "Same leader and finite captured owned identities")
    require(len({(v["pid"],v["start_ticks"]) for v in ids})==len(ids),"Unique process identities")
    require(isinstance(job.get("descendant_waits"),list) and all(
        identity(dict(w,uid=next((v["uid"] for v in ids if v["pid"]==w.get("pid") and v["start_ticks"]==w.get("start_ticks")),-1)))
        and any(v["pid"]==w.get("pid") and v["start_ticks"]==w.get("start_ticks") for v in ids)
        and w.get("wait_status_known") is True and type(w.get("returncode")) is int and w["returncode"]==0
        and type(w.get("raw_wait_status")) is int and w["raw_wait_status"]==0
        for w in job["descendant_waits"]),"All actual adopted descendant waits0")
    require(not job.get("resource_abort") and not job.get("cancel_requested") and not job.get("owned_termination") and not job.get("error"),
        "No cancelled/resource-aborted/error job")
    require(all(type(job.get(k)) in (int,float) and math.isfinite(job[k]) and job[k]>0
        for k in ("started_unix","started_monotonic","finished_unix")) and job["finished_unix"]>=job["started_unix"],
        "Actual ordered finite terminal timestamps")
    return job

def validate_records(cert,registry,cap,spec,done,tool):
    require(cert.get("schema")==CERT_SCHEMA and cert.get("status")=="full_science_complete_actual_CLI_exit1_owned_drain"
        and cert.get("root_authorizer")=="root_agent" and cert.get("actual_execution") is True
        and cert.get("exclusive_wait_relabelled") is False,"Explicit honest root shared terminal certificate")
    for key in ("capability_ref","registry_snapshot_ref","client_tool_terminal_ref","input_ref","completed_ref","pool_source_ref","terminal_validator_ref"):
        require(is_ref(cert.get(key)),"Exact terminal refs")
    require(cap.get("schema")=="GPU0_shared_logical_pool_capability_v1" and cap.get("root_authorizer")=="root_agent"
        and cap.get("logical_job_sharing_authorized") is True and cap.get("physical_gpu")==0 and cap.get("gpu_uuid")=="GPU-ff06ca1f-403a-f2d0-b584-71e177452667" and cert["pool_source_ref"] in cap["source_refs"].values(),
        "Genuine source-bound pool capability")
    require(registry.get("schema")==REG_SCHEMA and registry.get("capability_ref")==cert["capability_ref"]
        and registry.get("pool")==cap["pool"] and registry.get("physical_gpu")==cap["physical_gpu"]
        and registry.get("pool_job")=="gpu0_shared_v1:"+cap["pool"]+":gpu"+str(cap["physical_gpu"])
        and registry.get("safety_error") is None and not registry.get("ambiguous_descendants_or_lease_holders")
        and identity(registry.get("controller")),"Same genuine pool/controller registry without ambiguity")
    name=cert["logical_job"];require(name in cap["jobs"],"Exact declared logical job")
    rows=[j for j in registry.get("jobs",[]) if j.get("logical_job")==name]
    require(len(rows)==1,"One terminal job in frozen registry");job=completed_science_failed_cli_job(rows[0]);requested=cap["jobs"][name]
    require(job.get("requested")==requested and requested.get("job_id")==name and requested.get("input_ref")==cert["input_ref"]
        and spec.get("job_id")==name and spec.get("physical_gpu")==cap["physical_gpu"]
        and spec.get("refs",{}).get("storage")==requested.get("storage_ref")
        and spec.get("max_cuda_reserved_bytes")==requested.get("cuda_cap_bytes")
        and spec.get("max_cpu_rss_bytes")==requested.get("cpu_rss_cap_bytes")
        and requested.get("args")==["--execute","--input",cert["input_ref"]["path"],"--expected-input-sha256",cert["input_ref"]["sha256"]],
        "Actual exact admitted job/input/source/caps/storage/arguments")
    require(spec.get("phase")==cert.get("phase") and spec.get("output_directory")==cert.get("output_directory")
        and cert["completed_ref"]["path"]==spec["output_directory"]+"/completed.json"
        and done.get("input_ref")==cert["input_ref"],"Same scientific marker and actual output path")
    require(tool.get("schema")==TOOL_SCHEMA and tool.get("root_authorizer")=="root_agent"
        and tool.get("observation_source")=="actual_root_exec_or_write_stdin_terminal"
        and tool.get("actual_terminal_consumed") is True and tool.get("command") in ("run","wait")
        and tool.get("capability_ref")==cert["capability_ref"] and tool.get("logical_job")==name
        and type(tool.get("tool_exit_code")) is int and tool["tool_exit_code"]==1
        and isinstance(tool.get("tool_session_id"),(str,int)) and bool(str(tool["tool_session_id"]))
        and isinstance(tool.get("terminal_chunk_id"),str) and tool["terminal_chunk_id"]
        and tool.get("terminal_job")==job,"Genuine root-consumed shared client tool terminal1 and exact failed terminal JSON")
    require(registry["updated_unix"]>=job["finished_unix"],"Snapshot taken after terminal")
    require(cert.get("scientific_marker_unchanged") is True,"Original scientific marker preserved")
    return dict(certificate=cert,registry=registry,capability=cap,input=spec,completion=done,job=job)

def validate_certificate(record,reader=None):
    decode=read if reader is None else reader.decode_json
    cert=decode(record)
    require(cert.get("terminal_validator_ref")==ref(SOURCE),"Exact declared terminal validator")
    for key in ("capability_ref","registry_snapshot_ref","client_tool_terminal_ref","input_ref","completed_ref","pool_source_ref"):
        require(is_ref(cert.get(key)),"Pinned terminal lineage ref")
    # Sources are authenticated separately from metadata decode.
    require(ref(cert["pool_source_ref"]["path"])==cert["pool_source_ref"],"Unchanged actual pool source")
    require(cert["pool_source_ref"]==TRAIN_POOL_REF,"Only reviewed TRAIN/downstream pool")
    cap=decode(cert["capability_ref"])
    for r in cap["source_refs"].values():
        require(is_ref(r) and ref(r["path"])==r,"Complete actual pool execution source/metadata closure unchanged")
    if cap.get("controller_execution_override_ref") is not None:
        require(cap["controller_execution_override_ref"] in cap["source_refs"].values()
            and cap["controller_metadata_scope_checker_ref"] in cap["source_refs"].values()
            and cap.get("original_canonical_pool_and_workers_unchanged") is True,
            "Actual additive TRAIN controller execution is declared without changing scientific workers")
    result=validate_records(cert,decode(cert["registry_snapshot_ref"]),cap,
        decode(cert["input_ref"]),decode(cert["completed_ref"]),decode(cert["client_tool_terminal_ref"]))
    storage=decode(result["input"]["refs"]["storage"]);j=result["job"]["requested"]
    require(storage.get("job_id")==cert["logical_job"] and storage.get("output_directory")==cert["output_directory"]
        and storage.get("active") is True and storage.get("reservation",{}).get("output")==str(ROOT/cert["output_directory"])
        and storage.get("reserved_new_bytes")==storage["reservation"].get("reserved_bytes")
        and j["module_ref"] in cap["source_refs"].values() and ref(j["module_ref"]["path"])==j["module_ref"],
        "Exact historical source-bound worker storage identity; no credit/rebase assertion")
    reconcile_failed_cli_provenance(result,decode)
    return result


def bind_certificate_v1(request_ref,output_directory):
    """Root-only metadata binding after actual consumed wait1, full-science proof and owned drain.
    Never generates a wait receipt, polls, signals or infers completion.
    """
    request=read(request_ref)
    require(request.get("schema")=="H050_root_full_science_failed_CLI_binding_request_v1"
        and request.get("status")=="active" and request.get("root_authorizer")=="root_agent",
        "Explicit root metadata binding request")
    cap=read(request["capability_ref"]);spec=read(request["input_ref"])
    cert=dict(schema=CERT_SCHEMA,status="full_science_complete_actual_CLI_exit1_owned_drain",root_authorizer="root_agent",
        actual_leader_exit_code=1,actual_tool_exit_code=1,automatic_completed_scope_credit=False,
        root_failed_CLI_observation_ref=ROOT_OUTCOME_REF,
        actual_execution=True,exclusive_wait_relabelled=False,scientific_marker_unchanged=True,
        capability_ref=request["capability_ref"],registry_snapshot_ref=request["registry_snapshot_ref"],
        client_tool_terminal_ref=request["client_tool_terminal_ref"],input_ref=request["input_ref"],
        completed_ref=request["completed_ref"],pool_source_ref=TRAIN_POOL_REF,terminal_validator_ref=ref(SOURCE),
        logical_job=spec["job_id"],phase=spec["phase"],output_directory=spec["output_directory"])
    require(spec.get("phase")=="train" and spec.get("hypothesis_id")=="H050","Only actual H050 full TRAIN")
    proof=validate_records(cert,read(cert["registry_snapshot_ref"]),cap,spec,read(cert["completed_ref"]),read(cert["client_tool_terminal_ref"]))
    reconcile_failed_cli_provenance(proof,read)
    folder=path(output_directory);require(not folder.exists() and folder.parent.is_dir(),"Fresh metadata folder")
    folder.mkdir()
    target=folder/"failed_CLI_scientific_completion_certificate.json"
    target.write_text(json.dumps(cert,indent=2,sort_keys=True)+"\n")
    result=ref(str(target.relative_to(ROOT)))
    validate_certificate(result)
    return result

# This validator is intentionally restricted to the one authenticated incident.
ROOT_OUTCOME_REF={"path":"research/reviews/H050_full_budget_failed_CLI_root_20261016_v1/actual_outcome.root_v1.json","sha256":"554e56ed22ee96910128b36580177e4c355ea51e0ad65115f4a5e3d8f6875819"}
REGISTRY_REF={"path":"research/reviews/H050_full_budget_failed_CLI_root_20261016_v1/registry_snapshot.root_v1.json","sha256":"eb9620030ff7fdd3a7ee4d166af26fc80be218f2abeca91999bcc5c136265c2b"}
CLIENT_REF={"path":"research/reviews/H050_full_budget_failed_CLI_root_20261016_v1/actual_wait_client_tool_terminal.root_v1.json","sha256":"741b3a374b86e1570f318f6a9bfb11e34b0e905b3a2de979fb17f98ba2ca448a"}
COMPLETION_REF={"path":"experiments/results/H050_GPU0_direct_readout_train_seed101_v1/completed.json","sha256":"8df6cdcf507a067e48a9a23769cbb6718d485323f29ead6f4d767a64a55fbb8c"}
LOG_REF={"path":"research/reviews/GPU0_shared_root_integration_20261016_v1/controller_output_v1/jobs/H050-direct-readout-GPU0-TRAIN-v1.log","sha256":"1afdb2db3f799cfa2e989be3945506b65eabe66cadecb19555c6cc37e19ed8c1"}
ORIGINAL_TERMINAL_REF={"path":"research/reviews/H050_GPU0_shared_eval_lineage_source_20261016_v1/shared_terminal_v1.py","sha256":"b20169f3244ff973e6d4db36a54004e0235b6c112a458b93ace56a94deec0b79"}

def raw(record, cap=32*2**20):
    require(isinstance(record,dict) and set(record)=={"path","sha256"}
        and isinstance(record["sha256"],str) and re.fullmatch("[0-9a-f]{64}",record["sha256"]),
        "Exact relative pathname/SHA256")
    p=path(record["path"]);before=p.lstat()
    signature=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=cap,
        "Bounded singly linked regular immutable input")
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),"rb") as f:
        require(signature(os.fstat(f.fileno()))==signature(before),"Unchanged opened identity")
        data=f.read(cap+1)
        require(signature(os.fstat(f.fileno()))==signature(before),"Unchanged read identity")
    require(len(data)<=cap and signature(p.lstat())==signature(before)
        and hashlib.sha256(data).hexdigest()==record["sha256"],"Exact immutable bytes")
    return data

def reconcile_failed_cli_provenance(proof,decode):
    cert,job,done=proof["certificate"],proof["job"],proof["completion"]
    outcome=decode(ROOT_OUTCOME_REF);tool=decode(CLIENT_REF)
    require(cert["registry_snapshot_ref"]==REGISTRY_REF and cert["client_tool_terminal_ref"]==CLIENT_REF
        and cert["completed_ref"]==COMPLETION_REF and cert["root_failed_CLI_observation_ref"]==ROOT_OUTCOME_REF,
        "Only exact root-frozen actual failed-CLI evidence, never mutable registry")
    require(cert.get("actual_leader_exit_code")==cert.get("actual_tool_exit_code")==1
        and cert.get("automatic_completed_scope_credit") is False,
        "Actual failed TRAIN outcome retained, no completed-success scope credit")
    require(tool["tool_session_id"]==71181 and tool["terminal_chunk_id"]=="3aa694"
        and tool["tool_exit_code"]==1 and tool["failed_exit_preserved"] is True,
        "Exact genuinely consumed failed CLI terminal")
    require(job["pid"]==3119593 and job["leader_start_ticks"]==478204526
        and job["log_bytes"]==99644 and job["log_dropped_bytes"]==0
        and job["log_path"]==str(path(LOG_REF["path"])),
        "Exact captured failed leader identity and complete bounded log")
    require(outcome.get("schema")=="H050_actual_full_budget_failed_CLI_root_observation_v1"
        and outcome["completed_ref"]==COMPLETION_REF and outcome["registry_snapshot_ref"]==REGISTRY_REF
        and outcome["actual_tool_terminal_ref"]==CLIENT_REF and outcome["log_ref"]==LOG_REF
        and outcome["actual_leader_exit_code"]==outcome["actual_tool_exit_code"]==1
        and outcome["success0_or_storage_credit_claimed"] is False
        and outcome["training_replay_authorized"] is False
        and outcome["H050_exact_leader_PID_absent"] is True
        and outcome["final_log_literal_equals_original_scientific_completion"] is True
        and outcome["full_passes"]==30 and outcome["total_optimizer_updates"]==3840
        and outcome["scheduled_checkpoints"]==24,"Genuine root observation, not proposed execution evidence")
    log=raw(LOG_REF,2*2**20)
    require(len(log)==99644 and b"Traceback (most recent call last):" not in log,
        "Complete actual log, no traceback substituted for return-value failure")
    final=log.decode("utf-8").rstrip().splitlines()[-1]
    require(isinstance(ast.literal_eval(final),dict) and ast.literal_eval(final)==done,
        "Actual final printed noninteger return value exactly equals original completed science")
    arms=["adjacent_marginal_null","h045_continued_training","stable_leaf_memory","triple_support_pair_memory"]
    require(done.get("schema")=="H050_four_arm_train_completion_v1"
        and done.get("status")=="completed_fixed30pass_four_arm_TRAIN"
        and set(done.get("arms",[]))==set(arms) and done.get("steps")=={a:960 for a in arms}
        and len(done.get("history",[]))==30
        and sum(map(len,done.get("checkpoint_groups",{}).values()))==24,
        "Genuine full-budget scientific outputs, not partial marker acceptance")
    # The unchanged detailed lineage validator separately checks every30 pass,
    # first-step qualification, same32 parents and all24 exact checkpoint refs.
    return proof

"""Two finite deferred root envelopes and metadata recipes. Stdlib only.
No writes, accounting calls, guard configuration, process probes or launch API.
All event-reference parameters must point to later genuine root evidence.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import copy,hashlib,importlib.util,json,math,re,sys

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_native_worker_module_route_fix_source_20261016_v2"
SOURCE=HERE+"/finite_envelopes_v1.py"
PREPARE_REF=dict(path=HERE+"/prepare_resume_v1.py",sha256="c86c16335672c1749dc9d4c226c83e8eafa2296839692534ba83363355f67d0b")
MIB=2**20
GIB=2**30

def prepare_v1():
    p=ROOT/PREPARE_REF["path"]
    if not (p.resolve()==p and p.is_file() and not p.is_symlink()
            and hashlib.sha256(p.read_bytes()).hexdigest()==PREPARE_REF["sha256"]):
        raise ValueError("Exact sealed owned preparation source")
    sp=importlib.util.spec_from_file_location("_C3_deferred_science_preparation_v1",p)
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
    return m

def pool_v1():
    m=prepare_v1()
    return m.load(m.FINITE_POOL_REF,"_C3_deferred_exact_finite_pool_metadata_v1")

def historical_guard_bindings_v1():
    """Historical metadata only. Root must later supply actual current bindings."""
    m=prepare_v1();old=m.read(m.OLD_INPUT_REF);storage=m.read(old["refs"]["storage"])
    original_authority=m.read(storage["reservation"]["root_admission"])
    return m.unique(list(original_authority["bindings"].values()))

def base_common_refs_v1():
    """Exact first two lists of finite common_required_v1, without future events."""
    m=prepare_v1();p=pool_v1();oldcap=m.read(m.OLD_CAP_REF)
    paths=[p.SOURCE,p.LAUNCHER,p.DISPATCHER,p.BOOTSTRAP,
        p.REVIEWED+"/launch_gpu0_shared_v1.py",p.REVIEWED+"/gpu0_runtime_adapter_v1.py",
        p.REVIEWED+"/gpu0_bootstrap_renewal_v1.py",p.reviewed.DOWN,p.reviewed.OLD,p.legacy.BASE]
    rows=[m.ref(x) for x in paths]
    rows += [p.CONTRACT_REF,p.CATALOGUE_REF,p.REVIEWED_POOL_REF,p.GUARD_REF,
        p.CHECKER_REF,p.MEMO_REF,p.V4_REF,
        dict(path=p.legacy.GUARD,sha256=p.legacy.GUARD_SHA),p.legacy.POLICY,p.legacy.REPAIR,
        p.legacy.RENEWAL_SOURCE_REF,oldcap["marker_repair_renewal_ref"],
        oldcap["GPU0_policy_ref"],oldcap["stop_evidence_ref"]]
    return m.unique(rows)

def transition_materials_v1(actual_prior_controller_completion_ref):
    """Reads only the real later terminal/migration/probe JSON family."""
    m=prepare_v1();p=pool_v1();oldcap=m.read(m.OLD_CAP_REF)
    proof,rows=p.contract.validate_previous_controller_v1(actual_prior_controller_completion_ref)
    m.need(proof.get("C3_resume_manifest_ref") is not None,"This recipe requires the genuine interrupted full-state C3 branch")
    common=p.common_required_v1(dict(
        marker_repair_renewal_ref=oldcap["marker_repair_renewal_ref"],
        stop_evidence_ref=oldcap["stop_evidence_ref"],
        prior_controller_completion_ref=actual_prior_controller_completion_ref),
        SimpleNamespace(POLICY_REF=oldcap["GPU0_policy_ref"]))
    common=m.unique(common)
    return m,p,oldcap,proof,rows,common

def native_reuse_proof_v1(actual_prior_controller_completion_ref):
    m,p,oldcap,proof,rows,common=transition_materials_v1(actual_prior_controller_completion_ref)
    method=m.native_method_unchanged_v1();row=rows[method["job_id"]]
    spec=m.read(method["input_ref"]);storage=m.read(method["storage_ref"])
    m.need(row["requested"]==method and row["state"]=="cancelled" and row["pid"] is None
        and row.get("leader_start_ticks") is None and row.get("started_unix") is None
        and row["process_identities"]==[],"Actual never-started exact original native cancellation")
    m.need(storage["reserved_new_bytes"]==7*GIB and spec["refs"]["storage"]==method["storage_ref"],
        "Original native7GiB allocation, never a replacement charge")
    return dict(schema="GPU0_never_started_native_input_reuse_v1",root_authorizer="root_agent",
        reuse_authorized=True,prior_controller_completion_ref=actual_prior_controller_completion_ref,
        unchanged_scientific_input_ref=method["input_ref"],unchanged_scientific_storage_ref=method["storage_ref"],
        unchanged_caps_ref=method["cuda_caps_ref"],unchanged_job_id=method["job_id"],
        unchanged_absolute_expiry=spec["expires_at_utc"],repeat_scientific_storage_reservation=False)

def runtime_refs_v1(slot,method,actual_prior_controller_completion_ref):
    m,p,oldcap,proof,rows,common=transition_materials_v1(actual_prior_controller_completion_ref)
    m.need(slot in ("C3_RESUME_TRAIN","NATIVE_MLM"),"Exactly two declared slots in this root recipe")
    rows=[method["runtime_entry_ref"],method["module_ref"]]
    if slot=="C3_RESUME_TRAIN":
        rows += p.contract.resume_transition_refs_v1(actual_prior_controller_completion_ref)
    return m.unique(rows)

def operational_request_v1(slot,method,actual_prior_controller_completion_ref,*,actual_native_reuse_ref=None):
    m,p,oldcap,proof,rows,common=transition_materials_v1(actual_prior_controller_completion_ref)
    cat=p.contract.catalogue_v1()["slots"]
    m.need(slot in ("C3_RESUME_TRAIN","NATIVE_MLM")
        and all(method[k]==cat[slot][k] for k in ("worker_kind","module","module_ref")),
        "Only exact C3-resume/native methods")
    spec=m.read(method["input_ref"])
    if slot=="NATIVE_MLM":
        m.need(method==m.native_method_unchanged_v1() and actual_native_reuse_ref is not None,
            "Entire original native nested method unchanged")
        p.contract.validate_native_reuse_v1(actual_native_reuse_ref,method,actual_prior_controller_completion_ref,rows)
    else:
        m.need(method==m.c3_method_v1(method["input_ref"]) and actual_native_reuse_ref is None
            and spec["efficient_resume_manifest_ref"]==proof["C3_resume_manifest_ref"],
            "Fresh strict24GiB full-state C3 continuation matches genuine transition")
    result=dict(schema="GPU0_declared_operational_request_v1",status="active",root_authorizer="root_agent",
        slot=slot,method_job=copy.deepcopy(method),
        declared_limits={k:method[k] for k in (
            "cuda_cap_bytes","cpu_rss_cap_bytes","operational_deadline_seconds","log_limit_bytes")},
        actual_pool_ref=m.FINITE_POOL_REF,actual_launcher_ref=m.FINITE_LAUNCHER_REF,
        actual_dispatcher_ref=m.ref(p.DISPATCHER),unchanged_scientific_input_ref=method["input_ref"],
        prior_controller_completion_ref=actual_prior_controller_completion_ref,
        runtime_source_refs=runtime_refs_v1(slot,method,actual_prior_controller_completion_ref))
    if actual_native_reuse_ref is not None:result["native_never_started_reuse_ref"]=actual_native_reuse_ref
    return result

def actual_request_v1(record):
    m=prepare_v1();request=m.read(record)
    expected=operational_request_v1(request["slot"],request["method_job"],
        request["prior_controller_completion_ref"],
        actual_native_reuse_ref=request.get("native_never_started_reuse_ref"))
    m.need(request==expected,"Actual saved exact finite request")
    return request

def outer_bindings_v1(actual_request_ref,*,actual_guard_bindings):
    m=prepare_v1();request=actual_request_v1(actual_request_ref);method=request["method_job"]
    common=transition_materials_v1(request["prior_controller_completion_ref"])[-1]
    rows=[*common,*actual_guard_bindings,*request["runtime_source_refs"],actual_request_ref,
        method["input_ref"],method["storage_ref"],method["cuda_caps_ref"],
        request["prior_controller_completion_ref"]]
    if request.get("native_never_started_reuse_ref") is not None:rows.append(request["native_never_started_reuse_ref"])
    return m.unique(rows)

def contained_output_v1(output):
    m=prepare_v1();p=Path(output)
    m.need(type(output) is str and not p.is_absolute() and ".." not in p.parts
        and p.as_posix()==output and output.startswith("experiments/results/"),
        "Fresh repository results namespace")
    return output

def outer_reserve_request_v1(actual_request_ref,*,job_id,output_directory,actual_guard_bindings):
    m=prepare_v1();request=actual_request_v1(actual_request_ref);method=request["method_job"]
    m.need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",job_id) is not None
        and job_id!=method["job_id"],"Separate explicit outer operational identity")
    output=contained_output_v1(output_directory);science=m.read(method["storage_ref"])
    a,b=ROOT/output,ROOT/science["output_directory"]
    m.need(a!=b and not a.is_relative_to(b) and not b.is_relative_to(a),
        "Separate outer output outside scientific scope")
    return dict(key="H041/"+job_id,output=str(a),auxiliary_scopes=[],reserved_bytes=4*MIB,
        consumer_bindings=outer_bindings_v1(actual_request_ref,actual_guard_bindings=actual_guard_bindings))

def storage_from_actual_reserve_v1(request,actual_reserve_receipt_ref,*,job_id,expires_at_utc,
                                  actual_budget_components,actual_policy_ref):
    """Pure record construction after root's unchanged guard performed reserve."""
    m=prepare_v1();receipt=m.read(actual_reserve_receipt_ref);own=receipt["value"]
    m.need(receipt.get("kind")=="concurrent_completed_scope_reserve_receipt_v1"
        and {k:v for k,v in own.items() if k!="root_admission"}==request
        and request["key"]=="H041/"+job_id,"Genuine actual exact reserve receipt")
    authority=m.read(own["root_admission"])
    m.need(authority["active"] is True and authority["operation"]=="reserve"
        and authority["request"]==request and authority["expires_at_utc"]==expires_at_utc,
        "Actual root authority matches receipt and absolute expiry")
    m.need(set(actual_budget_components)=={"implementation","launcher","consumer_adapter"}
        and all(r in request["consumer_bindings"] for r in actual_budget_components.values())
        and actual_policy_ref in request["consumer_bindings"],"All real original guard components direct")
    return dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode="prospective_980gb_v1",job_id=job_id,
        output_directory=Path(request["output"]).relative_to(ROOT).as_posix(),
        reserved_new_bytes=request["reserved_bytes"],reservation=own,
        initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=expires_at_utc,
        completed_overlay_policy_ref=actual_policy_ref,budget_components=copy.deepcopy(actual_budget_components))

def operational_envelope_v1(actual_request_ref,actual_operational_storage_ref,*,actual_guard_bindings):
    m=prepare_v1();request=actual_request_v1(actual_request_ref);method=request["method_job"]
    storage=m.read(actual_operational_storage_ref);spec=m.read(method["input_ref"])
    expected=outer_reserve_request_v1(actual_request_ref,job_id=storage["job_id"],
        output_directory=storage["output_directory"],actual_guard_bindings=actual_guard_bindings)
    m.need(storage["reserved_new_bytes"]==4*MIB and storage["expires_at_utc"]==spec["expires_at_utc"]
        and {k:v for k,v in storage["reservation"].items() if k!="root_admission"}==expected,
        "Exact separate4MiB/noaux outer scope within unchanged scientific expiry")
    return dict(schema="GPU0_declared_operational_envelope_v1",status="active",root_authorizer="root_agent",
        request_ref=actual_request_ref,operational_storage_ref=actual_operational_storage_ref,
        expires_at_utc=storage["expires_at_utc"])

def finite_job_map_v1(actual_C3_envelope_ref,actual_native_envelope_ref):
    m=prepare_v1();p=pool_v1();jobs={}
    for slot,record in (("C3_RESUME_TRAIN",actual_C3_envelope_ref),("NATIVE_MLM",actual_native_envelope_ref)):
        envelope,request,method=p.contract.envelope_parts_v1(record)
        actual_request_v1(envelope["request_ref"])
        m.need(request["slot"]==slot and method["job_id"] not in jobs,"Exactly two separate finite methods")
        jobs[method["job_id"]]=dict(slot=slot,operational_input_ref=record)
    return dict(schema="GPU0_root_bound_finite_job_map_v1",status="active",root_authorizer="root_agent",jobs=jobs)

def controller_plan_v1(actual_job_map_ref,actual_prior_controller_completion_ref,*,
                       pool,controller_job_id,expires_at_utc,drain_expires_at_utc,actual_guard_bindings):
    m,p,oldcap,proof,rows,common=transition_materials_v1(actual_prior_controller_completion_ref)
    jobmap=m.read(actual_job_map_ref)
    m.need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}",pool) is not None
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",controller_job_id) is not None,
        "Safe explicit fresh pool/controller names")
    m.need(jobmap.get("schema")=="GPU0_root_bound_finite_job_map_v1" and jobmap.get("status")=="active"
        and jobmap.get("root_authorizer")=="root_agent"
        and len(jobmap["jobs"])==2
        and {r["slot"] for r in jobmap["jobs"].values()}=={"C3_RESUME_TRAIN","NATIVE_MLM"},
        "Exact two-slot job map")
    sources=[*common,*actual_guard_bindings,PREPARE_REF,m.ref(SOURCE),actual_job_map_ref,actual_prior_controller_completion_ref]
    jobs={};ends=[]
    for name,row in jobmap["jobs"].items():
        envelope,request,method=p.contract.envelope_parts_v1(row["operational_input_ref"])
        actual_request_v1(envelope["request_ref"])
        m.need(request["slot"]==row["slot"] and method["job_id"]==name
            and request["prior_controller_completion_ref"]==actual_prior_controller_completion_ref,
            "Actual shared transition and exact finite method")
        jobs[name]=p.contract.job_from_envelope_v1(row["operational_input_ref"])
        spec=m.read(method["input_ref"])
        for path,sha in spec["source_sha256"].items():sources.append(dict(path=path,sha256=sha))
        sources += [*request["runtime_source_refs"],row["operational_input_ref"],envelope["request_ref"],
            envelope["operational_storage_ref"],method["input_ref"],method["storage_ref"],method["cuda_caps_ref"]]
        if request.get("native_never_started_reuse_ref") is not None:sources.append(request["native_never_started_reuse_ref"])
        ends.append(p.contract.expiry(spec["expires_at_utc"]))
    end=p.contract.expiry(expires_at_utc);drain=p.contract.expiry(drain_expires_at_utc)
    renewal=m.read(oldcap["marker_repair_renewal_ref"])
    m.need(max(ends)<=end and drain>=end+35
        and drain<p.contract.expiry(renewal["renewed_expires_at_utc"]),
        "Pool covers both actual scientific expiries with owned drain inside genuine renewal")
    m.need(controller_job_id not in jobs,"Distinct controller logical identity")
    return dict(schema="GPU0_declared_logical_pool_plan_v1",status="active",root_authorizer="root_agent",
        pool=pool,controller_job_id=controller_job_id,physical_gpu=0,gpu_uuid=m.GPU0_UUID,
        maximum_distinct_project_GPUs=1,maximum_concurrent_jobs=2,three_logical_jobs_authorized=False,
        logical_job_sharing_authorized=True,external_job_sharing_authorized=True,
        legacy_exclusive_coexistence_allowed=False,headroom_mib=4096,termination_grace_seconds=30,
        owned_termination_escalation_authorized=True,owned_drain_after_expiry_authorized=True,
        expires_unix=end,drain_expires_unix=drain,jobs=jobs,
        source_refs={r["path"]:r for r in m.unique(sources,limit=None)},
        marker_repair_renewal_ref=oldcap["marker_repair_renewal_ref"],GPU0_policy_ref=oldcap["GPU0_policy_ref"],
        stop_evidence_ref=oldcap["stop_evidence_ref"],finite_job_map_ref=actual_job_map_ref,
        prior_controller_completion_ref=actual_prior_controller_completion_ref,
        native_40GiB_operational_ceiling_authorized=True,native_72h_operational_deadline_authorized=True,
        efficient_C3_full_state_resume_authorized=True)

def controller_bindings_v1(actual_plan_ref,*,actual_guard_bindings):
    m=prepare_v1();plan=m.read(actual_plan_ref)
    common=transition_materials_v1(plan["prior_controller_completion_ref"])[-1]
    return m.unique([*common,*actual_guard_bindings,actual_plan_ref,
        plan["finite_job_map_ref"],plan["prior_controller_completion_ref"],PREPARE_REF,m.ref(SOURCE)])

def controller_reserve_request_v1(actual_plan_ref,*,output_directory,actual_guard_bindings):
    m=prepare_v1();plan=m.read(actual_plan_ref);output=contained_output_v1(output_directory)
    scopes=[]
    p=pool_v1()
    for row in m.read(plan["finite_job_map_ref"])["jobs"].values():
        envelope,request,method=p.contract.envelope_parts_v1(row["operational_input_ref"])
        scopes += [m.read(envelope["operational_storage_ref"])["output_directory"],
            m.read(method["storage_ref"])["output_directory"]]
    a=ROOT/output
    m.need(all(a!=ROOT/b and not a.is_relative_to(ROOT/b) and not (ROOT/b).is_relative_to(a) for b in scopes),
        "Fresh controller output separate from all outer/scientific scopes")
    return dict(key="H041/"+plan["controller_job_id"],output=str(a),auxiliary_scopes=[],reserved_bytes=32*MIB,
        consumer_bindings=controller_bindings_v1(actual_plan_ref,actual_guard_bindings=actual_guard_bindings))

def controller_capability_v1(actual_plan_ref,actual_controller_storage_ref,*,actual_guard_bindings):
    m=prepare_v1();plan=m.read(actual_plan_ref);storage=m.read(actual_controller_storage_ref)
    expected=controller_reserve_request_v1(actual_plan_ref,output_directory=storage["output_directory"],
        actual_guard_bindings=actual_guard_bindings)
    m.need(storage["job_id"]==plan["controller_job_id"] and storage["reserved_new_bytes"]==32*MIB
        and {k:v for k,v in storage["reservation"].items() if k!="root_admission"}==expected
        and pool_v1().contract.expiry(storage["expires_at_utc"])>=plan["drain_expires_unix"],
        "Genuine32MiB/noaux controller scope lasting through original owned drain")
    result=copy.deepcopy(plan);result.update(schema="GPU0_declared_logical_pool_capability_v1",
        plan_ref=actual_plan_ref,controller_storage_ref=actual_controller_storage_ref)
    return result

FUTURE_ROLES_V1=(
    "scientific_caps","scientific_request","scientific_config","scientific_runtime","scientific_closure",
    "scientific_authorization","scientific_input","scientific_storage",
    "resume_bounds","resume_qualification","migration_manifest","migration_intent","rollback_authorization",
    "old_C3_terminal","prior_controller_completion","H050_failed_CLI_certificate",
    "transport_request","transport_observation","probe_actual_terminal","old_capability_revocation",
    "C3_outer_request","C3_outer_storage","C3_outer_envelope",
    "native_outer_request","native_outer_storage","native_outer_envelope","native_reuse",
    "finite_job_map","controller_plan")

def proposed_future_paths_v1():
    """Uncreated path descriptors only. They are never path/SHA references."""
    return {role:HERE+"/future_root_metadata_v1/"+role+".json" for role in FUTURE_ROLES_V1}

def pre_stop_source_shape_v1(future_paths,*,include_rollback=True,transport_unknown=True):
    """Path-set arithmetic, never manufacture future refs or execute a validator."""
    m=prepare_v1();p=pool_v1();old=m.read(m.OLD_INPUT_REF);st=m.read(old["refs"]["storage"])
    m.need(set(future_paths)==set(FUTURE_ROLES_V1)
        and len(set(future_paths.values()))==len(future_paths)
        and type(include_rollback) is bool and type(transport_unknown) is bool,
        "Exactly named distinct future-path descriptors and explicit branches")
    for path in future_paths.values():
        q=Path(path)
        m.need(type(path) is str and not q.is_absolute() and ".." not in q.parts
            and q.suffix==".json" and q.as_posix()==path and path.startswith(HERE+"/"),
            "Only explicitly tagged owned future metadata paths")
    base={r["path"] for r in base_common_refs_v1()}
    guards={r["path"] for r in historical_guard_bindings_v1()}
    f=future_paths
    replacements={
        old["caps_ref"]["path"]:f["scientific_caps"],
        old["root_stage_request_ref"]["path"]:f["scientific_request"],
        old["refs"]["config"]["path"]:f["scientific_config"],
        old["refs"]["runtime"]["path"]:f["scientific_runtime"],
        old["transitive_source_closure_ref"]["path"]:f["scientific_closure"],
        old["actual_launcher_ref"]["path"]:m.FINITE_LAUNCHER_REF["path"],
        old["operational_pool_source_ref"]["path"]:m.FINITE_POOL_REF["path"],
        old["operational_root_entry_ref"]["path"]:m.SOURCE,
        old["operational_shared_route_ref"]["path"]:f["scientific_authorization"]}
    science={replacements.get(r["path"],r["path"]) for r in st["reservation"]["consumer_bindings"]}
    science |= {old["refs"]["runtime"]["path"],old["operational_GPU0_runtime_ref"]["path"]}
    science |= {r["path"] for r in (m.RESUME_RUNTIME_REF,m.INTEGRATION_REF,m.VALIDATOR_REF,
        m.STEP_REF,m.CACHE_REF,m.GPU0_ADMISSION_REF,m.FINITE_POOL_REF,m.FINITE_LAUNCHER_REF)}
    science |= {m.SOURCE}|guards|{f[k] for k in (
        "scientific_caps","scientific_request","scientific_config","scientific_runtime","scientific_closure",
        "scientific_authorization","resume_bounds","resume_qualification","migration_manifest","migration_intent")}
    resume={r["path"] for r in (m.RESUME_RUNTIME_REF,m.INTEGRATION_REF,m.VALIDATOR_REF,m.STEP_REF,m.CACHE_REF,m.OLD_INPUT_REF)}
    resume |= {f[k] for k in ("migration_manifest","migration_intent","resume_bounds","resume_qualification","old_C3_terminal")}
    if include_rollback:
        science.add(f["rollback_authorization"]);resume.add(f["rollback_authorization"])
    transport=set()
    if transport_unknown:
        transport={p.contract.TRANSPORT_HELPER_REF["path"],p.contract.HANDLE_LOSS_REF["path"],
            p.contract.OLD_C3_CUDA_REF["path"],p.contract.OLD_H050_CUDA_REF["path"],
            p.contract.FAILED_CLI_VALIDATOR_REF["path"],f["H050_failed_CLI_certificate"]}
        transport |= {f[k] for k in ("transport_request","transport_observation","probe_actual_terminal","old_capability_revocation")}
    common=base|resume|transport
    C3_outer=common|guards|{old["main_source_ref"]["path"]}|{f[k] for k in (
        "C3_outer_request","scientific_input","scientific_storage","scientific_caps","prior_controller_completion")}
    native=m.native_method_unchanged_v1()
    native_outer=common|guards|{native[k]["path"] for k in (
        "runtime_entry_ref","module_ref","input_ref","storage_ref","cuda_caps_ref")}
    native_outer |= {f[k] for k in ("native_outer_request","native_reuse","prior_controller_completion")}
    controller=common|guards|{f[k] for k in ("controller_plan","finite_job_map","prior_controller_completion")}
    controller |= {PREPARE_REF["path"],SOURCE}
    known_rows={}
    for path in science|C3_outer|native_outer|controller:
        if path not in f.values():known_rows[path]=m.ref(path)
    inverse={path:role for role,path in f.items()}
    def describe(paths):
        return dict(unique_path_count=len(paths),original_direct_ceiling=64,
            available_direct_slots=64-len(paths),passes_source_shape_ceiling=len(paths)<=64,
            actual_sources_or_saved_metadata=[known_rows[path] for path in sorted(paths) if path in known_rows],
            future_path_descriptors=[dict(role=inverse[path],path=path,status="UNCREATED_FUTURE_PATH_DESCRIPTOR",
                is_event_or_source_reference=False) for path in sorted(paths) if path in inverse])
    report=dict(schema="C3_resume_pre_stop_source_shape_inspection_v1",
        no_new_event_references_created=True,actual_admission=False,actual_execution=False,
        guard_configured_or_called=False,old_input_ref=m.OLD_INPUT_REF,
        old_scientific_direct_count=len(st["reservation"]["consumer_bindings"]),
        old_transitive_source_map_count=len(old["source_sha256"]),
        historical_actual_guard_binding_count=len(guards),base_common_count=len(base),
        migration_common_count=len(resume),transport_common_count=len(transport),
        complete_common_count=len(common),common_plus_guard_count=len(common|guards),
        include_rollback=include_rollback,transport_unknown=transport_unknown,
        future_paths_are_descriptors_only=True,
        actual_root_must_recompute_complete_union_before_every_reserve=True,
        native_runtime_direct_list_is_only_exact_frozen_module_and_runtime=True,
        no_transitive_source_map_entries_dropped=True,
        scientific=describe(science),C3_operational=describe(C3_outer),
        native_operational=describe(native_outer),controller=describe(controller))
    m.need(all(report[k]["passes_source_shape_ceiling"] for k in (
        "scientific","C3_operational","native_operational","controller")),
        "Required source-shape direct union exceeds original64")
    return report

"""Finite native and ready original-EVAL ROOT envelopes and metadata recipes. Stdlib only.
No writes, accounting calls, guard configuration, process probes or launch API.
All event-reference parameters must point to later genuine root evidence.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import copy,hashlib,importlib.util,json,math,re,sys

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2/controller_v1"
SOURCE=HERE+"/finite_envelopes_v1.py"
PREPARE_REF={'path': 'research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2/controller_v1/prepare_resume_v1.py', 'sha256': '9d59805f93003df5f13d380a71c0d7350b4afd56474d02789df29019776f1ee3'}
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

def base_common_refs_v1():
    """ROOT common refs after the actual final family approval/seal/core exist."""
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
    rows += p.contract.family_authority_refs_v2()
    return m.unique(rows)

def transition_materials_v1(actual_prior_controller_completion_ref):
    """Reads only the real later terminal/migration/probe JSON family."""
    m=prepare_v1();p=pool_v1();oldcap=m.read(m.OLD_CAP_REF)
    proof,rows=p.contract.validate_previous_controller_v1(actual_prior_controller_completion_ref)
    m.need(proof.get("C3_not_interrupted_for_this_transition") is True,"C3 must finish in the current unchanged controller")
    common=p.common_required_v1(dict(
        marker_repair_renewal_ref=oldcap["marker_repair_renewal_ref"],
        stop_evidence_ref=oldcap["stop_evidence_ref"],
        prior_controller_completion_ref=actual_prior_controller_completion_ref),
        SimpleNamespace(POLICY_REF=oldcap["GPU0_policy_ref"]))
    common=m.unique(common)
    return m,p,oldcap,proof,rows,common

def runtime_refs_v1(slot,method,actual_prior_controller_completion_ref):
    m,p,oldcap,proof,rows,common=transition_materials_v1(actual_prior_controller_completion_ref)
    m.need(slot in p.contract.catalogue_v1()["slots"],"Exact native or ready original EVAL parent slot")
    rows=[method["runtime_entry_ref"],method["module_ref"],*p.contract.catalogue_v1()["slots"][slot].get("required_runtime_source_refs",[])]
    return m.unique(rows)

def operational_request_v1(slot,method,actual_prior_controller_completion_ref,*,actual_native_reuse_ref=None):
    m,p,oldcap,proof,rows,common=transition_materials_v1(actual_prior_controller_completion_ref)
    m.need(actual_native_reuse_ref is None and slot in p.contract.catalogue_v1()['slots']
        and method==(m.native_method_v2(method['input_ref']) if slot=='NATIVE_MLM' else m.eval_method_v1(method['input_ref'])),
        'One exact fresh native or original EVAL method, no TRAIN/reuse/replay')
    return dict(schema='GPU0_declared_operational_request_v1',status='active',root_authorizer='root_agent',
        slot=slot,method_job=copy.deepcopy(method),
        declared_limits={k:method[k] for k in ('cuda_cap_bytes','cpu_rss_cap_bytes','operational_deadline_seconds','log_limit_bytes')},
        actual_pool_ref=m.FINITE_POOL_REF,actual_launcher_ref=m.FINITE_LAUNCHER_REF,
        actual_dispatcher_ref=m.ref(p.DISPATCHER),unchanged_scientific_input_ref=method['input_ref'],
        prior_controller_completion_ref=actual_prior_controller_completion_ref,
        runtime_source_refs=runtime_refs_v1(slot,method,actual_prior_controller_completion_ref))

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
        accounting_mode=authority["accounting_mode"],job_id=job_id,
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

def finite_job_map_v1(actual_native_envelope_ref,actual_eval_envelope_refs):
    m=prepare_v1()
    p=pool_v1()
    m.need(isinstance(actual_eval_envelope_refs,(tuple,list)) and 1<=len(actual_eval_envelope_refs)<=5,
        'Only genuinely prepared/admitted ready EVAL parents, never future SHA placeholders')
    jobs={}
    slots=set()
    for record in [actual_native_envelope_ref,*actual_eval_envelope_refs]:
        envelope,request,method=p.contract.envelope_parts_v1(record)
        actual_request_v1(envelope['request_ref'])
        slot=request['slot']
        m.need(slot in p.contract.catalogue_v1()['slots'] and slot not in slots and method['job_id'] not in jobs,
            'Fixed exact native and original VAL000..004 parent catalogue')
        slots.add(slot)
        jobs[method['job_id']]=dict(slot=slot,operational_input_ref=record)
    m.need('NATIVE_MLM' in slots and len(slots)>=2,'Native plus at least one genuinely ready original EVAL parent')
    return dict(schema='GPU0_root_bound_finite_job_map_v1',status='active',root_authorizer='root_agent',jobs=jobs)

def controller_plan_v1(actual_job_map_ref,actual_prior_controller_completion_ref,*,
                       pool,controller_job_id,expires_at_utc,drain_expires_at_utc,actual_guard_bindings):
    m,p,oldcap,proof,rows,common=transition_materials_v1(actual_prior_controller_completion_ref)
    jobmap=m.read(actual_job_map_ref)
    m.need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}",pool) is not None
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",controller_job_id) is not None,
        "Safe explicit fresh pool/controller names")
    m.need(jobmap.get("schema")=="GPU0_root_bound_finite_job_map_v1" and jobmap.get("status")=="active"
        and jobmap.get("root_authorizer")=="root_agent"
        and 2<=len(jobmap["jobs"])<=6
        and "NATIVE_MLM" in {r["slot"] for r in jobmap["jobs"].values()}
        and len({r["slot"] for r in jobmap["jobs"].values()})==len(jobmap["jobs"])
        and {r["slot"] for r in jobmap["jobs"].values()}<=set(p.contract.catalogue_v1()["slots"]),
        "One native and1..5 actual ready EVAL parent jobs")
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
        efficient_C3_full_state_resume_authorized=False,C3_EVAL_8GiB_operational_ceiling_authorized=True)

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

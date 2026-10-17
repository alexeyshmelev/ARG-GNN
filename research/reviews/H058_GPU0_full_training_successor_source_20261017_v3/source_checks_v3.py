"""Stdlib AST and isolated RAM checks only. No production module import or resource calls."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,hashlib,json,os
D=Path("research/reviews")/Path(__file__).parent.name
A=Path("research/reviews/H058_GPU0_full_training_successor_source_20261017_v2")
PRODUCTION=("__init__.py","common_v1.py","context_factory_v1.py","direct_gpu0_v1.py","root_driver_v1.py","producer_v1.py","observations_gpu_v1.py")
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(name)
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def parsed(p):return ast.parse(p.read_text())
def isolated_fn(path,name,ns):
    node=next(n for n in parsed(path).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<isolated source function>","exec"),ns)
    return ns[name]
def normalized(text):
    return (text.replace(str(D),str(A)).replace(str(D).replace("/","."),str(A).replace("/",".")).
        replace("H058_GPU0_v3_","H058_GPU0_v2_").replace("H058-GPU0-v3-","H058-GPU0-v2-").
        replace("native_resource_disposition_ref","native_noncoexistence_qualification_ref").
        replace("validate_native_resource_disposition_v1","validate_native_noncoexistence_v3").
        replace("all_old_native_launch_resume_write_routes_permanently_revoked","all_native_launch_resume_execution_routes_revoked_while_H058_attached").
        replace("after genuine native terminal and owned drain","after native successful completion"))
def dump(v):return ast.dump(v,include_attributes=False)
for name in PRODUCTION:
    ast.parse((D/name).read_text())
    if name!="common_v1.py":
        check(name+"_normalized_AST_unchanged",dump(parsed(A/name))==dump(ast.parse(normalized((D/name).read_text()))))
old=parsed(A/"common_v1.py");new=ast.parse(normalized((D/"common_v1.py").read_text()))
def remove_delta(tree):
    tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef)and n.name=="validate_native_noncoexistence_v3")
        and not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id.startswith("NATIVE_") for t in n.targets))]
    return tree
check("common_only_native_resource_validator_and_ref_names_changed",dump(remove_delta(old))==dump(remove_delta(new)))
ns=dict(Path=Path)
for n in parsed(D/"common_v1.py").body:
    if isinstance(n,ast.Assign):
        try:exec(compile(ast.Module(body=[n],type_ignores=[]),"<source constants>","exec"),ns)
        except (NameError,TypeError):pass
now=datetime.now(timezone.utc)
v=json.loads((D/"native_resource_disposition.INACTIVE.template.v1.json").read_text())
v.update(status="resources_released",created_at_utc=(now-timedelta(hours=1)).isoformat(),
    expires_at_utc=(now+timedelta(days=2)).isoformat(),observed_at_utc=(now-timedelta(seconds=1)).isoformat(),
    H058_job_id="RAM_H058",no_native_GPU0_attachment_alive=True,
    all_old_native_launch_resume_write_routes_permanently_revoked=True)
absence=json.loads((D/"native_targeted_absence.INACTIVE.template.v1.json").read_text())
absence.update(status="observed",observed_at_utc=v["observed_at_utc"],native_known_roots_and_contexts_absent=True,
    no_native_GPU0_attachment_alive=True,all_exact_native_identities_and_groups_absent=True,
    all_old_native_launch_resume_write_routes_permanently_revoked=True)
vr=dict(path="RAM/disposition.json",sha256="0"*64);ar=dict(path="RAM/absence.json",sha256="1"*64)
v["fresh_targeted_native_absence_ref"]=ar
records={vr["path"]:(vr,v),ar["path"]:(ar,absence)}
for key in ("NATIVE_RESOURCE_WAIT_REF","NATIVE_RESOURCE_OUTER_REF","NATIVE_RESOURCE_AMENDMENT_REF","NATIVE_ROOT_DISPOSITION_REF"):
    r=ns[key];check(key+"_genuine_exact_hash",ref(r["path"])==r);records[r["path"]]=(r,json.loads(Path(r["path"]).read_text()))
baseline=copy.deepcopy(records)
def reader(r):
    original,value=records[r["path"]];require(original==r,"Exact copied metadata reference")
    return copy.deepcopy(value)
ns.update(read=reader,ref=ref,require=require,datetime=datetime,timezone=timezone,os=os)
isolated_fn(D/"common_v1.py","active",ns)
fn=isolated_fn(D/"common_v1.py","validate_native_resource_disposition_v1",ns)
q=dict(job_id="RAM_H058",native_resource_disposition_ref=vr)
check("genuine_outer1_direct0_complete50_drain_accepted_for_resources_only",fn(q,fresh=True)["native_success_claimed"] is False)
def reject(name,target,key,value,fresh=True):
    global records
    records=copy.deepcopy(baseline);records[target][1][key]=value
    try:fn(q,fresh=fresh)
    except (ValueError,KeyError,TypeError):check(name,True)
    else:raise AssertionError(name+" was accepted")
wv=ns["NATIVE_RESOURCE_WAIT_REF"]["path"];ov=ns["NATIVE_RESOURCE_OUTER_REF"]["path"];dv=ns["NATIVE_ROOT_DISPOSITION_REF"]["path"]
for name,target,key,value in [
 ("outer0_relabel_rejected",ov,"actual_tool_terminal_exit_code",0),
 ("unknown_terminal_rejected",ov,"terminal_consumed",False),
 ("direct_nonzero_rejected",wv,"returncode",-15),
 ("unknown_direct_wait_rejected",wv,"actual_direct_worker_wait",False),
 ("incomplete_owned_drain_rejected",wv,"all_owned_processes_and_groups_drained",False),
 ("no_identity_absence_rejected",wv,"owned_identity_absence",False),
 ("no_EOF_rejected",wv,"pipe_EOF_observed",False),
 ("cleanup_error_rejected",wv,"cleanup_errors",["RAM_failure"]),
 ("failure_erasure_rejected",wv,"operational_failure",None),
 ("wait_amendment_conflict_rejected",wv,"actual_execution_amendment_ref",ar),
 ("root_route_revocation_missing_rejected",dv,"all_future_old_VAL000_MLM_launch_resume_write_routes_revoked",False),
 ("native_success_claim_rejected",vr["path"],"native_success_claimed",True),
 ("scientific_authority_claim_rejected",vr["path"],"native_scientific_acceptance_authorized",True),
 ("resource_outer0_relabel_rejected",vr["path"],"native_outer_exit_code",0),
 ("missing_permanent_revocation_rejected",vr["path"],"all_old_native_launch_resume_write_routes_permanently_revoked",False),
 ("partial_known_inventory_rejected",vr["path"],"native_process_identities",v["native_process_identities"][:-1]),
 ("unknown_group_inventory_rejected",vr["path"],"native_process_groups",[]),
 ("wrong_wait_ref_rejected",vr["path"],"native_actual_wait_ref",ar),
 ("wrong_outer_ref_rejected",vr["path"],"native_actual_outer_terminal_ref",ar),
 ("missing_root_disposition_ref_rejected",vr["path"],"native_ROOT_terminal_disposition_ref",ar),
 ("live_context_rejected",ar["path"],"no_native_GPU0_attachment_alive",False),
 ("absence_incomplete_rejected",ar["path"],"all_exact_native_identities_and_groups_absent",False),
 ("absence_inventory_conflict_rejected",ar["path"],"native_process_identities",[]),
 ("absence_group_conflict_rejected",ar["path"],"native_process_groups",[]),
 ("absence_wait_conflict_rejected",ar["path"],"native_actual_wait_ref",vr),
 ("absence_outer_conflict_rejected",ar["path"],"native_actual_outer_terminal_ref",vr),
 ("absence_no_permanent_revocation_rejected",ar["path"],"all_old_native_launch_resume_write_routes_permanently_revoked",False),
 ("absence_future_timestamp_rejected",vr["path"],"observed_at_utc",(now+timedelta(hours=1)).isoformat())]:
    reject(name,target,key,value)
records=copy.deepcopy(baseline)
for path in (vr["path"],ar["path"]):records[path][1]["observed_at_utc"]=(now-timedelta(minutes=10)).isoformat()
try:fn(q,fresh=True)
except ValueError:check("stale_absence_rejected_at_admission_launch",True)
else:raise AssertionError("stale absence accepted")
check("continuing_resource_checks_allow_historical_entry_stamp",fn(q,fresh=False)["native_outer_exit_code"]==1)
# Actual unchanged native scientific function, in isolated RAM; no completed/checkpoint payload reads.
native=Path("research/reviews/native_ARGformer_postdrain_pool_source_20261017_v2/downstream_context_v1.py")
complete_ref=dict(path="RAM/completed.json",sha256="2"*64)
wait=copy.deepcopy(baseline[wv][1]);wait["completed_ref"]=complete_ref
complete=dict(parent_id="american_c3_val_000",query_ancestry_truth_opened=False,child_input_ref=wait["input_ref"])
outer=dict(actual_exit_code=0,actual_terminal_consumed=True,actual_direct_worker_returncode=0,
    all_owned_processes_and_groups_drained=True,actual_wait_ref=ns["NATIVE_RESOURCE_WAIT_REF"],actual_tool_session_id=70445)
science_records={complete_ref["path"]:complete,wv:wait,ov:outer}
c=SimpleNamespace(require=require,read=lambda r:science_records[r["path"]],PARENT="american_c3_val_000",
    MANIFEST=wait["adapter_manifest_ref"]["path"],ref=lambda p:wait["adapter_manifest_ref"])
sci=isolated_fn(native,"_bundle_v1",dict(c=c))
bundle=dict(completion_ref=complete_ref,wait_ref=ns["NATIVE_RESOURCE_WAIT_REF"],outer_terminal_ref=ns["NATIVE_RESOURCE_OUTER_REF"])
try:sci(bundle,"contrastive")
except ValueError:check("unchanged_native_scientific_gate_rejects_real_operational_failure",True)
else:raise AssertionError("science accepted failure")
science_records[wv]["operational_failure"]=None;science_records[ov]["actual_exit_code"]=1
try:sci(bundle,"contrastive")
except ValueError:check("unchanged_native_scientific_gate_rejects_outer1",True)
else:raise AssertionError("science accepted outer1")
science_records[ov]["actual_exit_code"]=0
check("unchanged_native_success_predicate_control_passes",sci(bundle,"contrastive")[0]==complete)
# Exact source-derived direct role arithmetic, without a guard/journal/mutex call.
q_roles={k:dict(path="RAM/"+k,sha256="0"*64)for k in ("config_ref","source_review_ref","direct_attachment_authority_ref","native_resource_disposition_ref")}
q_roles["science_handoff_ref"]=ref(D/"producer_handoff.v1.json")
role_ns=dict(ns,q=q_roles,cfg=dict(source_manifest_ref=ref(D/"producer_manifest.v1.json")),
    request_ref=dict(path="RAM/root_request",sha256="0"*64),producer=ref(D/"producer_v1.py"),
    runtime=dict(path="RAM/runtime",sha256="0"*64),closure=dict(path="RAM/closure",sha256="0"*64),
    direct=SimpleNamespace(CURRENT_CAP=None))
for n in parsed(D/"direct_gpu0_v1.py").body:
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="CURRENT_CAP"for t in n.targets):
        role_ns["direct"].CURRENT_CAP=ast.literal_eval(n.value)
admit=next(n for n in parsed(D/"root_driver_v1.py").body if isinstance(n,ast.FunctionDef)and n.name=="admit_v1")
for n in admit.body:
    nodes=[]
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="needed"for t in n.targets):nodes=[n]
    if isinstance(n,ast.AugAssign)and isinstance(n.target,ast.Name)and n.target.id=="needed":nodes=[n]
    if isinstance(n,ast.If):nodes=[z for z in n.body if isinstance(z,ast.AugAssign)and isinstance(z.target,ast.Name)and z.target.id=="needed"]
    for z in nodes:exec(compile(ast.Module(body=[z],type_ignores=[]),"<isolated role list>","exec"),role_ns)
capture=json.loads(Path("research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json").read_text())
parent=list(capture["all_actual_parent_bindings"].values())
parent_paths={r["path"]for r in parent};base=parent_paths|{ns["SELECTOR_REF"]["path"]};needed={r["path"]for r in role_ns["needed"]}
arithmetic=dict(frozen_named_parent_roles=len(parent),frozen_distinct_parent_roles=len(parent_paths),
    current_selector_named_union=len(parent)+1,current_selector_distinct_union=len(base),
    new_direct_refs=len(needed-base),total_distinct=len(base|needed),direct_ceiling=64,
    actual_current_admission_evidence=False)
check("immutable_parent45named38distinct",len(parent)==45 and len(parent_paths)==38)
check("source_role_union_exactly64",arithmetic["total_distinct"]==64 and arithmetic["new_direct_refs"]==25)
(D/"ROLE_ARITHMETIC.v3.json").write_text(json.dumps(arithmetic,sort_keys=True,indent=2)+"\n")
# The old CPU dict predicate permits truthful null totals; no final update count is manufactured.
cpu=json.loads(Path("research/reviews/H058_CPU_ROOT_stop_for_GPU_20261017_v1/prior_CPU_disposition.root_active_v1.json").read_text())
check("CPU_null_totals_and_real1plus1_lower_bounds_preserved",
    cpu["actual_optimizer_steps"]=={"same_marker_aligned":None,"same_marker_alignment_null":None}
    and cpu["verified_optimizer_step_lower_bounds"]=={"same_marker_aligned":1,"same_marker_alignment_null":1}
    and cpu["new_GPU_budget"]==1920)
manifest=json.loads((D/"source_manifest.v1.json").read_text());extensions=set()
for r in manifest["refs"]:
    check("closure_exact:"+r["path"],ref(r["path"])==r)
    extensions.add(Path(r["path"]).suffix)
    if Path(r["path"]).suffix==".py":ast.parse(Path(r["path"]).read_text())
check("source_closure_has_no_binary_payloads",extensions<=set((".py",".json",".sh",".md")))
for r in json.loads((A/"MANIFEST.v2.json").read_text())["files"]:
    check("sealed_v2_unchanged:"+r["path"],ref(r["path"])["sha256"]==r["sha256"])
result=dict(schema="H058_v3_AST_isolated_RAM_source_checks_v1",status="PASS",check_count=len(checks),checks=checks,
    production_modules_imported=False,model_or_data_payloads_read=False,guards_journals_mutexes_resources_called=False,
    source_only=True,role_arithmetic=arithmetic)
(D/"source_checks.result.v3.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status="PASS",checks=len(checks),role_arithmetic=arithmetic),sort_keys=True))

"""Narrow source-only native14v3 audit and complete compatibility set counts; no project imports."""
import ast,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).parent
V2=HERE.parent/"native14_stopped_retention_successor_source_20261016_v2"
V1=HERE.parent/"native14_stopped_retention_successor_source_20261016_v1"
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=Path(path);raw=p.read_bytes();require(not p.is_symlink() and len(raw)<=32*2**20,"bounded source/metadata only")
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())
def read(path):return json.loads(Path(path).read_bytes())
def save(name,value):(HERE/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
def funcs(source):return {n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def paths(refs):return {r["path"] for r in refs}
def verify(refs):
    for r in refs:require(ref(r["path"])==r,"exact source/compact metadata ref "+r["path"])
seals=[]
for old,sha in [(V1,"a725a81c8844f7ba4dbba7d19cf898ace170ca85bb3c5daefb0567df08be7b5e"),(V2,"010b7dca3eed6b7702dc4feff1713cb43f837da65c3a41af4240243233fb2b8a")]:
    seal=ref(old/"source_handoff.v1.json");require(seal["sha256"]==sha,"immutable prior handoff")
    authored=read(old/"source_handoff.v1.json")["authored_files"];verify(authored)
    seals.append(dict(handoff_ref=seal,unchanged_authored_files=len(authored)))
review=HERE.parent/"native14_stopped_retention_successor_independent_review_20261016_v2/review.v2.json"
require(ref(review)["sha256"]=="c2a4fcd52a54085ff8230d875b967efdbd3743de9902671e9834011925c3b40c","immutable SOURCE_PASS review")
m=read(HERE/"source_manifest.v1.json");oldm=read(V2/"source_manifest.v1.json")
refs=m["runtime_verified_source_and_compact_metadata_refs"];verify(refs)
require(len(refs)==len(paths(refs))==222,"complete222 closure")
py=[r for r in refs if r["path"].endswith(".py")]
for r in py:ast.parse(Path(r["path"]).read_text())
require(len(py)==166 and m["static_unresolved_project_imports"]==[],"full166 source closure")
for k in ["original_reservation","original_reservation_sha256","original_scope_paths","original_reserved_bytes","original_expiry_unchanged","known_PID_start_rows","known_groups","recorded_filesystem_endpoints","all_original_evidence_refs_preserved"]:
    require(m[k]==oldm[k],"historical field preserved "+k)
historical=m["original_reservation"]["consumer_bindings"]
audit=read(m["evidence_audit_ref"]["path"])
require(m["original_reservation"]==audit["original"]["reservation"],"exact audited original reservation")
require(len(historical)==len(paths(historical))==45 and all(r in refs for r in historical),"all45 remain runtime authenticated")
digest=hashlib.sha256(json.dumps(m["original_reservation"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
require(digest==m["original_reservation_sha256"]=="d748739db39bfe8c4469014b6d815730cf06baf13ef6246fcd073afd640672da","exact original digest")
new=funcs((HERE/"native14_stopped_retention_v1.py").read_text())
old=funcs((V2/"native14_stopped_retention_v1.py").read_text())
preserved=[]
for name,fn in old.items():
    if name in {"_bindings","_native_manifest"}:continue
    normalized=ast.parse(ast.unparse(fn).replace(str(V2),str(HERE))).body[0]
    require(ast.dump(normalized)==ast.dump(new[name]),"preserved v2 function "+name);preserved.append(name)
require(len(preserved)==57,"only two production functions changed")
require((HERE/"retained_nodes_v1.py").read_bytes()==(V2/"retained_nodes_v1.py").read_bytes(),"helper byte-identical")
ordinary=read(V2/"source_static_checks.v1.json")["ordinary_AST_identical_functions_and_classes"]
parent=funcs(Path("research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py").read_text())
for n in ordinary:require(ast.dump(new[n])==ast.dump(parent[n]),"ordinary current Census API "+n)
require(len(ordinary)==30,"unchanged30 APIs")
base=read(V2/"direct_role_plan.v1.json")["current_Census23_refs"];verify(base)
authored=[ref(HERE/n) for n in ["native14_stopped_retention_v1.py","retained_nodes_v1.py","source_manifest.v1.json"]]
concrete=base+authored
require(len(paths(base))==23 and len(paths(concrete))==26,"exact23+3")
future=["actual_native14_policy","actual_native14_ROOT_activation","actual_native14_permanent_revocation","actual_native14_source_review"]
plan=dict(schema="native14_v3_exact_historical_manifest_auth_direct_role_plan",successor_revision=3,
    current_Census23_refs=base,authored_direct_refs=authored,concrete_union_refs=concrete,concrete_unique_count=26,
    future_actual_state_roles=future,future_overlay_state_binding_unique_upper_count=30,
    future_actual_retention_invocation_roles=["actual_targeted_observation","actual_retirement_operation"],
    complete_retention_invocation_union_upper_count=32,original_direct_ref_ceiling=64,
    original45_exact_refs=historical,original45_all_explicitly_raw_authenticated_at_runtime=True,
    original45_authenticated_via_direct_manifest_not_flattened=True,exact_audited_original_reservation_equality_required=True,
    all_current_ancestor_and_seven_native14_roles_direct=True,actual_future_unions_must_be_recomputed_under_original_mutex=True,
    no_authentication_omission_or_cap_change=True,source_only=True,actual_admission_or_capacity_claimed=False)
save("direct_role_plan.v3.json",plan)
valpath=Path("research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1/direct_role_plan.v1.json")
val=read(valpath);valcounts={}
for name,p in val["phases"].items():
    verify(p["concrete_refs"]);overlap=len(paths(p["concrete_refs"])&paths(concrete))
    count=len(paths(p["concrete_refs"])|paths(concrete))+len(p["future_actual_roles"])+4
    require(overlap==23 and count==65,"honest complete sealed VALv1 union")
    valcounts[name]=dict(original_complete_union=p["complete_upper_union"],overlap_with_native14_state=overlap,
        native14v3_complete_union=count,fits64=False,needs_separate_reviewed_operational_rebind=True)
evalpath=Path("research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1/direct_role_plan.v1.json")
ev=read(evalpath);verify(ev["known_direct_refs"])
evcount=len(paths(ev["known_direct_refs"])|paths(concrete))+len(ev["future_exact_roles"])+4
require(evcount==45,"current full30Mb evaluation complete union45")
legacypath=Path("research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2/direct_binding_role_plan.v2.json")
legacy=read(legacypath);legacycounts={};legacy_mismatches={}
for name,p in legacy["scopes"].items():
    for r in p["actual_existing_refs"]:
        actual=ref(r["path"])
        if actual!=r:legacy_mismatches[r["path"]]=dict(expected=r,actual=actual)
    overlap=len(set(p["complete_direct_paths"])&paths(concrete))
    count=len(set(p["complete_direct_paths"])|paths(concrete))+4
    require(overlap==19,"exact legacy overlap19")
    legacycounts[name]=dict(original_complete_union=p["direct_count"],overlap_with_native14_state=overlap,
        native14v3_complete_union=count,fits64=count<=64)
require(legacycounts["EVAL_outer_representative_VAL000"]["native14v3_complete_union"]==67,"legacy outer67 still fails")
downpath=Path("research/reviews/native_VAL000_downstream_execution_source_20261016_v2")
down=read(downpath/"ROOT_reserve_request.INACTIVE.template.v1.json")
def exactrefs(v):
    if isinstance(v,dict):
        if set(v)=={"path","sha256"}:return [v]
        return sum((exactrefs(x) for x in v.values()),[])
    if isinstance(v,list):return sum((exactrefs(x) for x in v),[])
    return []
# Four explicitly requested downstream consumers, not every authority reference nested in the draft.
downrefs=down["consumer_bindings"];verify(downrefs)
require(len(downrefs)==4 and len(paths(downrefs)|paths(concrete))+4==34,"illustrated master reserve34")
compat=dict(schema="native14_v3_source_only_exact_consumer_union_compatibility",source_only=True,
    original_direct_ceiling=64,native14v2_state=62,native14v2_VALv1_union=93,native14v3_state=30,native14v3_retention_invocation=32,
    sealed_VALv1_plan_ref=ref(valpath),sealed_VALv1_stages=valcounts,
    current_full30Mb_C3_evaluation=dict(plan_ref=ref(evalpath),original_complete_union=38,known_refs=34,future_roles=ev["future_exact_roles"],overlap=23,native14v3_complete_union=evcount,fits64=True,scope="Original5 VAL parents times6 epoch checkpoints,30 seals"),
    legacy_C3_native_concurrent=dict(plan_ref=ref(legacypath),scopes=legacycounts,repair_in_current_scope=False,declared_plan_role_count_only=True,source_ref_integrity_mismatches=list(legacy_mismatches.values()),actual_source_authenticated=not bool(legacy_mismatches),
        distinction="Legacy C3 GNN EVAL envelope alongside native MLM; separate from current full30Mb evaluation and native ARGformer downstream candidate"),
    separate_native65GiB_downstream=dict(handoff_ref=ref(downpath/"source_handoff.v2.json"),manifest_ref=ref(downpath/"source_manifest.v1.json"),
        illustrated_four_consumer_refs=downrefs,illustrated_master_reserve_union=34,with_distinct_active_reserve_authority=35,
        final_complete_union_proven=False,still_needs_source_review_ROOT_authority_phase_controller_and_guard_rebind=True),
    actual_future_refs_not_fabricated=True,no_actual_activation_admission_measurement_or_capacity_claimed=True)
save("deployment_compatibility.v3.json",compat)
for name in ["activation_policy.template.inactive.v1.json","frozen_parent_activation.template.inactive.v1.json","permanent_revocation.template.inactive.v1.json","retirement_operation.template.inactive.v1.json","targeted_observation.template.inactive.v1.json"]:
    v=read(HERE/name);require(v["active"] is False,"inactive template")
    for r in exactrefs(v):
        if r["path"].startswith(str(HERE)+"/"):require(ref(r["path"])==r,"exact own template ref")
fixture=read(HERE/"binding_contract_results.v3.json")
require(fixture["status"]=="PASS" and fixture["cases"]==72 and all(r["status"]=="PASS" for r in fixture["results"]),"72 fake contracts")
for p in HERE.glob("*.py"):compile(p.read_text(),str(p),"exec")
save("source_static_checks.v3.json",dict(schema="native14_v3_narrow_source_preservation_audit",status="SOURCE_STATIC_PASS_REVIEW_PENDING",
    created_at_utc=datetime.now(timezone.utc).isoformat(),prior_seals_unchanged=seals,prior_SOURCE_PASS_review_ref=ref(review),
    source_manifest_ref=ref(HERE/"source_manifest.v1.json"),main_source_ref=ref(HERE/"native14_stopped_retention_v1.py"),
    direct_role_plan_ref=ref(HERE/"direct_role_plan.v3.json"),deployment_compatibility_ref=ref(HERE/"deployment_compatibility.v3.json"),
    runtime_ref_count=222,parsed_python_count=166,ordinary_AST_identical_functions_and_classes=ordinary,
    unchanged_v2_functions_and_classes=preserved,only_changed_production_functions=["_bindings","_native_manifest"],
    retained_helper_byte_identical=True,all_original45_refs_authenticated_and_exact_audited_reservation=True,original_reservation_sha256=digest,
    all_original_history_scope_caps_authority_expiry_socket_fail_closed_semantics_preserved=True,
    direct_state30_retention32=True,sealed_VALv1_union65_current_C3_EVAL45_legacy_EVAL_outer67=True,
    fake_fixture_ref=ref(HERE/"binding_contract_results.v3.json"),fake_cases=72,actual_guard_accounting_process_or_scope_calls=0,
    scientific_payload_model_array_private_truth_stdout_tail_reads=0,source_only=True))
require(read(HERE/"source_static_checks.v3.json")["direct_role_plan_ref"]==ref(HERE/"direct_role_plan.v3.json"),"final role plan hash exact")
print(json.dumps(dict(status="PASS",runtime_refs=222,python_ASTs=166,ordinary_API_ASTs=30,preserved_v2_functions=57,fake_checks=72,native_state=30,retirement=32,VALv1=65,current_C3_EVAL=45,legacy_EVAL_outer=67)))

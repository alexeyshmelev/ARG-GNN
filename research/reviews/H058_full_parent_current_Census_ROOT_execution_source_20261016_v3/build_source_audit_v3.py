"""Reproducible source-only manifest/role/audit builder. No project imports or runtime calls."""
import ast,copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).parent;OLD=HERE.parent/"H058_full_parent_ROOT_execution_source_20261016_v2"
SCI=HERE.parent/"H058_full_parent_pilot_source_20261016_v2"
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=Path(path);data=p.read_bytes()
    require(p.is_file() and not p.is_symlink() and len(data)<=32*2**20,"bounded source/metadata")
    return dict(path=str(p),sha256=hashlib.sha256(data).hexdigest())
def read(path):return json.loads(Path(path).read_bytes())
def save(name,value):(HERE/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
def node(t,name):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def verify_seal(directory,expected):
    sealref=ref(directory/"source_handoff.v1.json");require(sealref["sha256"]==expected,"original handoff unchanged")
    seal=read(directory/"source_handoff.v1.json")
    for r in seal["files"]:require(ref(r["path"])=={k:r[k] for k in ["path","sha256"]},"original sealed file unchanged")
    return dict(handoff_ref=sealref,unchanged_file_count=len(seal["files"]))
root_seal=verify_seal(OLD,"fc0807a10c3c79526d31ff121c2ed6166f8372b165c8afc21c45bb328a30d75d")
science_seal=verify_seal(SCI,"509e299922ba504c0282b5d10c21964e7ebfb7d7048939bac40554bc4208e82e")
root_review_path=HERE.parent/"H058_full_parent_ROOT_execution_independent_review_20261016_v2/review.v1.json"
science_review_path=HERE.parent/"H058_full_parent_pilot_fresh_delta_review_20261016_v2/review.v2.json"
for p in [root_review_path,science_review_path]:
    review=read(p);require(review["verdict"]=="SOURCE_PASS" and review["blockers"]==[],"genuine predecessor source pass")
ancestors=read(HERE.parent/"European_outer8_preparation_source_20261016_v2/direct_role_plan.v2.json")["ancestor_refs"]
require(len({r["path"] for r in ancestors})==23,"actual current Census23 binding union")
for r in ancestors:require(ref(r["path"])==r,"actual current Census23 source/policy/activation pins")
ct=ast.parse((HERE/"common_v1.py").read_text());dt=ast.parse((HERE/"direct_gpu0_v1.py").read_text());rt=ast.parse((HERE/"root_driver_v1.py").read_text())
ns=dict(Path=Path)
for t in [ct,dt]:
    for n in t.body:
        if isinstance(n,ast.Assign):
            for call in [x for x in ast.walk(n.value) if isinstance(x,ast.Call)]:
                require(isinstance(call.func,ast.Name) and call.func.id in ["dict","Path"],"constant-only source configuration")
            exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),"<source-constants>","exec"),ns)
require(ns["HERE"]==str(HERE),"current ROOT package")
guard=Path(ns["GUARD"]["path"]);gt=ast.parse(guard.read_text())
require(ref(guard)==ns["GUARD"] and ref(ns["POLICY"]["path"])==ns["POLICY"],"actual current guard/policy")
api={}
for name in ["configure_with_renewal_v1","operation_draft_v1","_state_locked","apply_reserve_v1","budget_check_v9","validate_storage_v9"]:
    api[name]=ast.unparse(node(gt,name).args)
savefn=node(node(gt,"PilotOutputV9"),"save") if False else next(n for n in node(gt,"PilotOutputV9").body if isinstance(n,ast.FunctionDef) and n.name=="save")
writer=ast.get_source_segment(guard.read_text(),savefn)
literal="self.used+planned<=self.storage['reserved_new_bytes']"
require(writer.count(literal)==1 and ast.unparse(savefn.args)=="self, name, value, kind='json'","exact current strict writer clone API/expression")
modified_writer=writer.replace(literal,"self.used+planned<=min(self.storage['reserved_new_bytes'],256*2**20)")
compile(modified_writer,"<audit-only-current-writer-clone>","exec")
oldct=ast.parse((OLD/"common_v1.py").read_text())
caps=lambda t:next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="CAPS" for x in n.targets))
require(ast.dump(caps(ct),include_attributes=False)==ast.dump(caps(oldct),include_attributes=False),"unchanged original caps")
# Keep the reviewed predecessor closure, replacing only its five ROOT Python files.
refs={}
def add(r):
    require(ref(r["path"])==r,"exact production closure")
    require(r["path"] not in refs or refs[r["path"]]==r,"no conflicting closure pin")
    refs[r["path"]]=r
for r in read(OLD/"source_manifest.v1.json")["refs"]:
    if r["path"].startswith(str(OLD)+"/") and r["path"].endswith(".py"):
        add(ref(HERE/Path(r["path"]).name))
    else:add(r)
for r in ancestors:add(r)
for p in [OLD/"source_handoff.v1.json",OLD/"source_manifest.v1.json",root_review_path,science_review_path,
          SCI/"source_handoff.v1.json",SCI/"source_manifest.v1.json",SCI/"producer_v1.py",SCI/"protocol_v1.py"]:
    add(ref(p))
# Explicit source pins of the actual current guard; do not traverse its retired payload scopes.
for n in gt.body:
    if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id.endswith("_REF") for x in n.targets):
        v=n.value;r=None
        if isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=="dict":
            d={k.arg:ast.literal_eval(k.value) for k in v.keywords}
            if set(d)=={"path","sha256"}:r=d
        elif isinstance(v,ast.Dict):
            d=ast.literal_eval(v)
            if set(d)=={"path","sha256"}:r=d
        if r is not None:add(r)
manifest=dict(schema="H058_actual_current_Census_ROOT_source_manifest_v3",source_only=True,
    execution_authorized=False,refs=sorted(refs.values(),key=lambda r:r["path"]),
    predecessor_handoff_ref=root_seal["handoff_ref"],predecessor_independent_review_ref=ref(root_review_path),
    scientific_handoff_ref=science_seal["handoff_ref"],scientific_source_manifest_ref=ref(SCI/"source_manifest.v1.json"),
    scientific_independent_review_ref=ref(science_review_path),current_Census_guard_ref=ns["GUARD"],
    current_Census_policy_ref=ns["POLICY"],current_Census23_ancestor_refs=ancestors,
    science_payload_reads=False,accounting_calls=False,production_manifest_excludes_fixtures_audits_and_self=True)
save("source_manifest.v1.json",manifest)
mref=ref(HERE/"source_manifest.v1.json")
for name in ["request.template.inactive.v1.json","evaluate_request.template.inactive.v1.json","native_noncoexistence_qualification.template.inactive.v3.json"]:
    value=read(HERE/name);value["adapter_manifest_ref"]=mref
    require(value.get("status")=="SOURCE_ONLY_INACTIVE","inactive template")
    save(name,value)
# Evaluate only actual needed assignments from admission; future metadata roles stay symbolic, unhashed.
future={}
def symbolic(name):
    r=dict(path="ACTUAL_ROOT_ROLE/"+name,sha256=None);future[name]=r;return r
scientific=read(SCI/"source_handoff.v1.json")
role_ns=dict(ns,ref=ref,MANIFEST=str(HERE/"source_manifest.v1.json"),
    request_ref=symbolic("phase_request"),runtime=symbolic("phase_runtime"),
    closure=symbolic("phase_consumer_manifest"),producer=scientific["producer_ref"],
    cfg=dict(source_manifest_ref=scientific["source_manifest_ref"]),
    q=dict(config_ref=symbolic("active_scientific_config"),science_handoff_ref=science_seal["handoff_ref"],
           source_review_ref=symbolic("independent_v3_SOURCE_PASS_review"),
           direct_attachment_authority_ref=symbolic("TRAIN_borrowed_attachment_authority"),
           native_noncoexistence_qualification_ref=symbolic("TRAIN_native_noncoexistence_qualification")),
    direct=type("FakeSourceConstants",(),dict(CURRENT_CAP=ns["CURRENT_CAP"]))())
admit=node(rt,"admit_v1")
base_stmts=[n for n in admit.body if isinstance(n,(ast.Assign,ast.AugAssign)) and
    ((isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="needed" for x in n.targets))
     or isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=="needed")]
exec(compile(ast.fix_missing_locations(ast.Module(body=base_stmts,type_ignores=[])),"<admission-role-list>","exec"),role_ns)
base_needed=role_ns["needed"]
train_if=next(n for n in admit.body if isinstance(n,ast.If) and any(isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=="needed" for x in n.body))
train_add=next(n for n in train_if.body if isinstance(n,ast.AugAssign))
plan=dict(schema="H058_actual_current_Census_direct_role_plan_v3",source_only=True,current_Census23_refs=ancestors,
    direct_ceiling=64,scientific_transitive_manifest_ref=scientific["source_manifest_ref"],
    future_actual_roles=[dict(role=k,path_and_hash_must_be_supplied_by_ROOT=True) for k in future],
    role_lists_derived_from_actual_admit_needed_AST=True,actual_guard_or_accounting_not_called=True,phases={})
for phase in ["evaluate","train"]:
    role_ns["needed"]=list(base_needed)
    if phase=="train":exec(compile(ast.fix_missing_locations(ast.Module(body=[train_add],type_ignores=[])),"<train-role-list>","exec"),role_ns)
    union={r["path"]:r for r in ancestors+role_ns["needed"]}
    concrete=[r for r in union.values() if r["sha256"] is not None]
    symbolic_roles=[r["path"].removeprefix("ACTUAL_ROOT_ROLE/") for r in union.values() if r["sha256"] is None]
    require(len(union)<=64,"complete original64 admission union")
    for r in concrete:require(ref(r["path"])==r,"exact concrete direct role")
    plan["phases"][phase]=dict(concrete_refs=sorted(concrete,key=lambda r:r["path"]),concrete_count=len(concrete),
        unresolved_actual_roles=symbolic_roles,unresolved_actual_role_count=len(symbolic_roles),
        complete_upper_union=len(union),remaining_direct_slots=64-len(union))
save("direct_role_plan.v3.json",plan)
fixture=read(HERE/"fake_contract_results.v3.json");require(fixture["status"]=="PASS" and fixture["passed"]>=82,"source-only fixture pass")
compiled=[]
for p in HERE.glob("*.py"):
    compile(p.read_text(),str(p),"exec");compiled.append(ref(p))
# Ordinary static scientific values only. The protocol is source parsed, never imported.
pt=ast.parse((SCI/"protocol_v1.py").read_text());protocol=node(pt,"fixed_protocol_v1")
ret=next(n for n in protocol.body if isinstance(n,ast.Return))
values={k.arg:ast.literal_eval(k.value) for k in ret.value.keywords if isinstance(k.value,ast.Constant)}
require(values["passes"]==30 and values["updates_per_arm"]==960 and values["VAL_prediction_products_required_before_truth"]==96
    and values["first_step_qualification"]=="first_scheduled_step_retained_in_960_no_extra_updates","frozen science")
guard_apis={k:api[k] for k in api}
audit=dict(schema="H058_actual_current_Census_source_audit_v3",status="SOURCE_ONLY_CHECKS_PASS",
    source_manifest_ref=mref,source_manifest_ref_count=len(manifest["refs"]),direct_role_plan_ref=ref(HERE/"direct_role_plan.v3.json"),
    compiled_source_refs=compiled,fake_contract_results_ref=ref(HERE/"fake_contract_results.v3.json"),
    fake_checks_passed=fixture["passed"],protected_predecessor_functions_identical=fixture["unchanged_protected_function_count"],
    predecessor_ROOT_seal_unchanged=root_seal,scientific_seal_unchanged=science_seal,
    predecessor_ROOT_SOURCE_PASS_review_ref=ref(root_review_path),scientific_SOURCE_PASS_review_ref=ref(science_review_path),
    current_Census_api_signatures=guard_apis,current_Census_writer_save_signature=ast.unparse(savefn.args),
    writer_cap_literal_count=1,writer_clone_only_local_counter_cap_change=True,original_CAPS_AST_identical=True,
    original_capacity_decision_AST_identical=True,original_owned_cleanup_finally_AST_identical=True,
    RAMLog_AST_identical=True,scientific_protocol_values=values,source_only=True,
    scientific_or_project_imports=False,scientific_payload_reads=False,actual_guard_configure_or_accounting_calls=False,
    process_device_socket_or_resource_probes=False,admission_or_execution=False,
    new_independent_review_status="PENDING")
save("source_audit.v3.json",audit)
print(json.dumps(dict(status="PASS",manifest_refs=len(manifest["refs"]),checks=fixture["passed"],
    direct_unions={k:v["complete_upper_union"] for k,v in plan["phases"].items()},
    original_ROOT_files=root_seal["unchanged_file_count"],original_science_files=science_seal["unchanged_file_count"]),sort_keys=True))

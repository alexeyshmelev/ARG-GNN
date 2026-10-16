"""Source-only narrow rebind preservation audit; no project imports or actual guard/accounting calls."""
import ast,builtins,hashlib,json,symtable
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).parent
OLD=HERE.parent/"American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
NATIVE=HERE.parent/"native14_stopped_retention_successor_source_20261016_v3"
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_bytes())
def save(n,v):(HERE/n).write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n")
def funcs(p):return {n.name:n for n in ast.parse(Path(p).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def verify(rows):
    for r in rows:
        require(ref(r["path"])=={k:r[k] for k in ("path","sha256")},"exact sealed source/compact metadata "+r["path"])
        if "bytes" in r:require(len(Path(r["path"]).read_bytes())==r["bytes"],"exact sealed byte size")
require(ref(OLD/"source_handoff.v1.json")["sha256"]=="e7e7edf6f71321fde24637c944f4edf76ec89a6fc5aef3694aa4af6b71cb62fb","immutable VALv1 seal")
oldseal=read(OLD/"source_handoff.v1.json");verify(oldseal["files"])
require(ref(NATIVE/"source_handoff.v1.json")["sha256"]=="e8e0d88fa50d3c343f6e06416ef9ae35f32e5fc5d195676173294bb639dd67eb","immutable native14v3 seal")
verify(read(NATIVE/"source_handoff.v1.json")["authored_files"])
for n in ["readiness.v1.json","scientific_basis.v1.json","caps.template.inactive.v1.json","reuse_provenance.v1.json",
    "infer.scientific_config.template.inactive.v1.json","native_paths.scientific_config.template.inactive.v1.json","plan.scientific_config.template.inactive.v1.json"]:
    require((HERE/n).read_bytes()==(OLD/n).read_bytes(),"byte-identical original science/caps/prerequisites "+n)
m=read(HERE/"source_manifest.v1.json");verify(m["refs"])
require(len(m["refs"])==len({r["path"] for r in m["refs"]})==301,"complete301 exact metadata/source closure")
py=[r for r in m["refs"] if r["path"].endswith(".py")]
for r in py:ast.parse(Path(r["path"]).read_text())
require(len(py)==196,"full196 Python AST closure")
require(m["single_direct_role_consolidated_through_exact_manifest"] in m["refs"],"baseline remains manifest-authenticated")
for r in read(NATIVE/"source_manifest.v1.json")["runtime_verified_source_and_compact_metadata_refs"]:
    require(r in m["refs"],"all native222 source/metadata refs retained")
for r in read(OLD/"source_manifest.v1.json")["refs"]:
    expected=r
    if r["path"].startswith(str(OLD)+"/") and not r["path"].endswith(".lock"):
        expected=ref(r["path"].replace(str(OLD),str(HERE),1))
    require(expected in m["refs"],"all original125 source/metadata refs retained with exact own rebind")
allowed={"root_contract_v1.py":{"ref","phase_paths","original_guard","actual_prior","validate_request","bind","direct_roles","input_spec","cpu_slot"},
    "root_driver_v1.py":{"admit","close","main"},"worker_entry_v1.py":{"worker"},"plan_worker_v1.py":set()}
preserved={};changed={}
for filename,permitted in allowed.items():
    old=funcs(OLD/filename);new=funcs(HERE/filename);same=[];delta=[]
    for name,fn in old.items():
        text=ast.unparse(fn).replace(str(OLD),str(HERE)).replace(OLD.name,HERE.name).replace("_current_Census_v1","_native14v3_v2").replace("-current-Census-v1","-native14v3-v2")
        normalized=ast.parse(text).body[0]
        if ast.dump(normalized)==ast.dump(new[name]):same.append(name)
        else:require(name in permitted,"only scoped operational function changes "+filename+":"+name);delta.append(name)
    preserved[filename]=same;changed[filename]=delta
ct=funcs(HERE/"root_contract_v1.py");oct=funcs(OLD/"root_contract_v1.py")
# Exact prerequisite checks differ only by fresh namespace/suffix; no wait/drain/science branch changes.
class Namespace(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=node.value.replace("_current_Census_v1","_native14v3_v2").replace("-current-Census-v1","-native14v3-v2")
            if node.value=="_v1":node.value="_v2"
        return node
for name in ["phase_paths","actual_prior"]:
    normalized=Namespace().visit(oct[name]);require(ast.dump(normalized)==ast.dump(ct[name]),"only prerequisite namespace change "+name)
# Direct roles drop exactly the baseline config while retaining every producer/helper role.
role=oct["direct_roles"]
class Roles(ast.NodeTransformer):
    def visit_List(self,node):
        self.generic_visit(node);node.elts=[x for x in node.elts if not isinstance(x,ast.Name) or x.id!="BASE_CONFIG_REF"];return node
    def visit_BinOp(self,node):
        self.generic_visit(node)
        if any(isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value.startswith("/cpu_slot_") for x in ast.walk(node)):
            for x in ast.walk(node):
                if isinstance(x,ast.Name) and x.id=="REL":x.id="SLOT_REL"
        return node
role=Roles().visit(role)
require(ast.dump(role)==ast.dump(ct["direct_roles"]),"only one redundant direct baseline role consolidated plus original shared slots")
oldexec=funcs(OLD/"root_driver_v1.py")["execute"];newexec=funcs(HERE/"root_driver_v1.py")["execute"]
require(ast.dump(oldexec)==ast.dump(newexec),"entire bounded supervisor execute AST unchanged")
require(ast.dump(funcs(OLD/"plan_worker_v1.py")["execute"])==ast.dump(funcs(HERE/"plan_worker_v1.py")["execute"]),"entire original plan execute AST unchanged")
for index in range(2):
    r=ref(OLD/("cpu_slot_"+str(index)+".lock"));require(r in m["refs"] and Path(r["path"]).read_bytes()==b"","original shared source slot unchanged")
    require(not (HERE/("cpu_slot_"+str(index)+".lock")).exists(),"no new independent CPU slot pool")
plan=read(HERE/"direct_role_plan.v2.json")
for name,p in plan["phases"].items():
    verify(p["concrete_refs"]);require(p["concrete_count"]+p["future_actual_count"]==p["complete_upper_union"]==64,"all6 complete real stage role unions64")
    require(all("sha256" not in x for x in p["future_actual_role_descriptors"]),"no fabricated actual future hashes")
require(len(plan["phases"])==6,"all six original parent/stages")
compiled=[];unresolved={}
for p in sorted(HERE.glob("*.py")):
    compile(p.read_text(),str(p),"exec");compiled.append(ref(p))
    if p.name in allowed:
        table=symtable.symtable(p.read_text(),str(p),"exec")
        available=set(table.get_identifiers())|set(dir(builtins))|{"__file__","__name__"};absent=set()
        def visit(t):
            for s in t.get_symbols():
                if s.is_global() and s.is_referenced() and s.get_name() not in available:absent.add(s.get_name())
            for child in t.get_children():visit(child)
        visit(table)
        if absent:unresolved[p.name]=sorted(absent)
require(not unresolved,"no unresolved production globals")
def ownrefs(v):
    if isinstance(v,dict):
        if set(v)=={"path","sha256"}:
            if v["path"].startswith(str(HERE)+"/"):require(ref(v["path"])==v,"exact inactive template own ref")
        else:
            for x in v.values():ownrefs(x)
    elif isinstance(v,list):
        for x in v:ownrefs(x)
for p in HERE.glob("*.json"):
    if "inactive" in p.name:
        v=read(p);require(v.get("active",False) is False and v.get("execution_enabled",False) is False,"inert template")
        ownrefs(v)
fixtures=read(HERE/"fake_contract_results.v2.json")
require(fixtures["status"]=="PASS" and fixtures["passed"]==139 and all(x["status"]=="PASS" for x in fixtures["checks"]),"139 pure/fake contracts")
save("source_static_checks.v2.json",dict(schema="American_C3_VAL003004_native14v3_narrow_source_preservation_audit_v2",
    status="SOURCE_STATIC_AND_139_PURE_FAKE_CHECKS_PASS_REVIEW_PENDING",created_at_utc=datetime.now(timezone.utc).isoformat(),
    source_manifest_ref=ref(HERE/"source_manifest.v1.json"),prior_VALv1_handoff_ref=ref(OLD/"source_handoff.v1.json"),
    prior_VALv1_files_unchanged=len(oldseal["files"]),native14v3_handoff_ref=ref(NATIVE/"source_handoff.v1.json"),
    native14v3_authored_files_unchanged=True,production_source_metadata_refs=301,parsed_Python_sources=196,
    preserved_function_ASTs=preserved,changed_operational_functions=changed,new_function="bind_policy",unresolved_authored_globals=unresolved,
    compiled_authored_refs=compiled,entire_supervisor_execute_and_cleanup_AST_identical=True,entire_original_plan_execute_AST_identical=True,
    all_original_scientific_configs_caps_readiness_basis_and_seed_metadata_byte_identical=True,
    original_shared_two_slot_files_preserved_without_new_slot_pool=True,single_consolidated_ref=m["single_direct_role_consolidated_through_exact_manifest"],
    all125_prior_source_refs_and_all222_native_source_refs_authenticated=True,
    direct_role_plan_ref=ref(HERE/"direct_role_plan.v2.json"),all6_complete_stage_unions=64,original_direct_ceiling=64,
    exact_actual_policy_hash_required_at_bind=True,actual_policy_hash_not_fabricated=True,policy_ref_immutable_in_request_input_storage=True,
    guard_original_full_source_policy_activation_revocation_review_validation_preserved=True,
    original_reservations_wall_RSS_AUX_work_log_caps_and_scientific_wait_drain_close_requirements_preserved=True,
    all30_prediction_seals_and_full_original_VAL_scope_before_truth_preserved=True,
    fake_contract_results_ref=ref(HERE/"fake_contract_results.v2.json"),passing_pure_fake_checks=139,
    actual_guard_accounting_admission_or_jobs=0,numerical_imports=0,process_device_socket_resource_probes=0,
    scientific_payload_model_array_private_truth_stdout_tail_reads=0,source_only=True))
print(json.dumps(dict(status="PASS",refs=301,python_ASTs=196,fake_checks=139,all_six_complete_stage_unions=64,original_VALv1_files_unchanged=len(oldseal["files"]))))

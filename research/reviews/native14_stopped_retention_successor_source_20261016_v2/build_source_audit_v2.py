"""Source-only v2 preservation, exact refs and tiny-fixture audit; never import project code."""
import ast,builtins,hashlib,json,os,stat,symtable
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).parent
OLD=HERE.parent/"native14_stopped_retention_successor_source_20261016_v1"
PARENT=HERE.parent/"prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py"
REVIEW=HERE.parent/"native14_stopped_retention_successor_independent_review_20261016_v1/review.v1.json"
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=Path(path);data=p.read_bytes();require(p.is_file() and not p.is_symlink() and len(data)<=32*2**20,"bounded source/metadata only")
    return dict(path=str(p),sha256=hashlib.sha256(data).hexdigest())
def read(path):return json.loads(Path(path).read_bytes())
def save(name,value):(HERE/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
def funcs(source):return {n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
oldseal=read(OLD/"source_handoff.v1.json")
require(ref(OLD/"source_handoff.v1.json")["sha256"]=="a725a81c8844f7ba4dbba7d19cf898ace170ca85bb3c5daefb0567df08be7b5e","v1 sealed handoff unchanged")
for r in oldseal["authored_files"]:require(ref(r["path"])==r,"all17 v1 authored files unchanged")
require(ref(REVIEW)["sha256"]=="d5d5dd011f6ee850ebb46c0a8397107fb14bb34ce546b26295906e39c5bbdd7c","exact independent REQUEST_CHANGES")
review=read(REVIEW);require(review["verdict"]=="REQUEST_CHANGES_SOURCE_ONLY","review provenance")
m=read(HERE/"source_manifest.v1.json");refs=m["runtime_verified_source_and_compact_metadata_refs"]
require(len(refs)==222 and len({r["path"] for r in refs})==222,"unchanged full runtime ref count")
pyrefs=[]
for r in refs:
    require(ref(r["path"])==r,"exact runtime source/compact metadata ref")
    if r["path"].endswith(".py"):
        ast.parse(Path(r["path"]).read_text());pyrefs.append(r)
require(len(pyrefs)==166,"original full static Python closure count")
require(m["static_unresolved_project_imports"]==[],"preserved closed project imports")
require((HERE/"retained_nodes_v1.py").read_bytes()==(OLD/"retained_nodes_v1.py").read_bytes(),"unchanged no-follow metadata helper")
newsource=(HERE/"native14_stopped_retention_v1.py").read_text()
new=funcs(newsource);old=funcs((OLD/"native14_stopped_retention_v1.py").read_text());parent=funcs(PARENT.read_text())
oldchecks=read(OLD/"source_static_checks.v1.json")
ordinary=oldchecks["ordinary_AST_identical_functions_and_classes"]
for name in ordinary:
    require(ast.dump(new[name],include_attributes=False)==ast.dump(parent[name],include_attributes=False),"ordinary API exact current parent: "+name)
require(len(ordinary)==30,"all original30 ordinary AST checks")
allowed={"_kernel_endpoints","_native_socket_absent","_native_targets","_native_validate_targets","_native_measurement"}
unchanged=[]
for name,fn in old.items():
    if name in allowed:continue
    normalized=ast.parse(ast.unparse(fn).replace(str(OLD),str(HERE))).body[0]
    require(ast.dump(normalized,include_attributes=False)==ast.dump(new[name],include_attributes=False),"v1 function preserved outside socket delta: "+name)
    unchanged.append(name)
for name in ["_state_locked","apply_retire_stopped_native14_v1","_native_recorded_proof","root_observe_stopped_v1","_native_proof","_native_revocation"]:
    require(ast.dump(new[name],include_attributes=False)==ast.dump(old[name],include_attributes=False),"unchanged lifecycle/history/chronology body: "+name)
module=ast.parse(newsource)
constants={}
for n in module.body:
    if isinstance(n,ast.Assign):
        try:
            value=ast.literal_eval(n.value)
            for target in n.targets:
                if isinstance(target,ast.Name):constants[target.id]=value
        except (ValueError,TypeError):pass
require(constants["NATIVE_KEY"]==m["exact_key"]==oldseal["only_retention_key"],"exact key")
require(constants["NATIVE_RESERVED"]==m["original_reserved_bytes"]==15032385536,"exact original charge")
require(constants["NATIVE_ROOTS"]==m["original_scope_paths"]==oldseal["all3_original_charged_parent_roots"],"all3 complete roots")
require(constants["NATIVE_EXPIRY"]==m["original_expiry_unchanged"]=="2026-10-18T11:47:27.217781+00:00","original expiry")
require(constants["MODE"]=="native14_stopped_retention_v2" and constants["JOURNAL"]==".runtime/storage/native14_stopped_retention_v2","separate inactive v2 mode/journal")
require(len(m["known_PID_start_rows"])==150 and sum(r["start_ticks"]=="UNKNOWN" for r in m["known_PID_start_rows"])==6
    and len(m["known_groups"])==11 and len(m["recorded_filesystem_endpoints"])==24,"preserved lineage/targets")
require(m["known_PID_start_rows"]==read(OLD/"source_manifest.v1.json")["known_PID_start_rows"]
    and m["recorded_filesystem_endpoints"]==read(OLD/"source_manifest.v1.json")["recorded_filesystem_endpoints"],"exact original target identities")
plan=read(HERE/"direct_role_plan.v1.json")
for r in plan["concrete_union_refs"]:require(ref(r["path"])==r,"exact concrete direct ref")
require(len({r["path"] for r in plan["concrete_union_refs"]})==58
    and plan["future_overlay_state_binding_unique_upper_count"]==62
    and plan["complete_retention_invocation_union_upper_count"]==64
    and plan["original_direct_ref_ceiling"]==64,"complete role upper unions unchanged")
for name in ["activation_policy.template.inactive.v1.json","frozen_parent_activation.template.inactive.v1.json",
             "permanent_revocation.template.inactive.v1.json","retirement_operation.template.inactive.v1.json",
             "targeted_observation.template.inactive.v1.json"]:
    value=read(HERE/name);require(value.get("active") is False,"inactive authority template")
    def walk(v):
        if isinstance(v,dict):
            if set(v)=={"path","sha256"} and isinstance(v.get("path"),str) and v["path"].startswith(str(HERE)+"/"):
                require(ref(v["path"])==v,"fresh template exact own source/manifest ref")
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(value)
software=read(HERE/"software_contract_results.v1.json");sockets=read(HERE/"socket_address_contract_results.v2.json")
require(software["cases"]==43 and all(r["passed"] for r in software["results"]),"retained helper and API fixtures")
require(sockets["status"]=="PASS" and sockets["cases"]==56 and all(r["passed"] for r in sockets["results"]),"fake socket/catalog contracts")
compiled=[]
for p in sorted(HERE.glob("*.py")):
    compile(p.read_text(),str(p),"exec");compiled.append(ref(p))
unresolved={}
for p in [HERE/"native14_stopped_retention_v1.py",HERE/"retained_nodes_v1.py"]:
    table=symtable.symtable(p.read_text(),str(p),"exec")
    available=set(table.get_identifiers())|set(dir(builtins))|{"__file__","__name__"}
    absent=set()
    def visit(t):
        for s in t.get_symbols():
            if s.is_global() and s.is_referenced() and s.get_name() not in available:absent.add(s.get_name())
        for child in t.get_children():visit(child)
    visit(table)
    if absent:unresolved[str(p)]=sorted(absent)
require(not unresolved,"no unresolved authored globals")
# Preserve synthetic metadata/files; no alias target is followed by this inventory.
run=Path(software["fixture_directory"])
require(run.is_relative_to(HERE.resolve()/"software_fixtures"),"owned synthetic fixture inventory only")
inventory=[]
for p in sorted(run.rglob("*")):
    s=p.lstat();kind="directory" if stat.S_ISDIR(s.st_mode) else "regular" if stat.S_ISREG(s.st_mode) else "symlink" if stat.S_ISLNK(s.st_mode) else "socket" if stat.S_ISSOCK(s.st_mode) else "fifo" if stat.S_ISFIFO(s.st_mode) else "other"
    row=dict(path=str(p),kind=kind,stat_identity=[s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns])
    if kind=="regular":
        require(s.st_size<=1000000,"tiny synthetic regular payload")
        row["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    elif kind=="symlink":row["stored_link_text"]=os.readlink(p)
    inventory.append(row)
save("software_fixture_inventory.v1.json",dict(schema="native14_v2_owned_synthetic_fixture_inventory",fixture_directory=str(run),nodes=inventory,
    aliases_sockets_and_failures_preserved=True,regular_payload_bytes=software["regular_fixture_payload_bytes"],original_scope_access=False))
audit=dict(schema="native14_source_static_API_role_and_preservation_audit_v2",status="SOURCE_STATIC_AND_TINY_SOFTWARE_PASS_INDEPENDENT_REVIEW_PENDING",
    created_at_utc=datetime.now(timezone.utc).isoformat(),main_source_ref=ref(HERE/"native14_stopped_retention_v1.py"),
    measurement_source_ref=ref(HERE/"retained_nodes_v1.py"),source_manifest_ref=ref(HERE/"source_manifest.v1.json"),
    direct_role_plan_ref=ref(HERE/"direct_role_plan.v1.json"),authored_AST_compile_refs=compiled,source_metadata_refs=222,
    parsed_dependency_Python_sources=len(pyrefs),ordinary_AST_identical_functions_and_classes=ordinary,
    preserved_v1_function_ASTs_outside_socket_delta=unchanged,ordinary_reserve_and_completed_replay_AST_branches_identical=True,
    retained_node_helper_byte_identical=True,unresolved_authored_globals=unresolved,unresolved_project_imports=[],
    concrete_unique_refs=58,state_upper_unique_refs=62,retention_invocation_upper_unique_refs=64,original_direct_role_cap=64,
    original64KiB_authority_event_bound_unchanged=True,all89_original_evidence_refs_preserved=True,
    known_lineage_PID_rows=150,unknown_original_start_rows=6,known_groups=11,exact_recorded_filesystem_endpoints=24,
    original_reservation_sha256=m["original_reservation_sha256"],original_expiry_preserved=constants["NATIVE_EXPIRY"],
    original_all3_roots_and_15032385536B_exact_reservation_bound=True,
    ordinary_completed_close_for_native14_blocked=True,namespace02_only_no_namespace03_success_assumed=True,
    exact_target_identity_presence_chronology_and_final_operation_expiry_rechecked=True,
    unqualified_retained_filesystem_sockets_preserved_and_blocked=True,production_certificates_require_zero_sockets=True,
    historical_known_endpoint_observations_require_absent_paths=True,raw_kernel_path_nonmatch_never_certifies_socket_inactivity=True,
    no_assumed_original_bind_identity_mapping_from_lstat=True,strict_nonempty_complete_parseable_kernel_catalog=True,
    lexical_relative_slash_dot_same_basename_alias_ambiguities_block=True,renamed_or_different_basename_socket_nodes_still_block=True,
    no_external_endpoint_path_probe_or_symlink_resolution=True,no_socket_cleanup_or_namespace_change=True,
    no_actual_scope_acceptance_or_socket_absence_established=True,
    software_contract_cases=software["cases"],socket_fake_contract_cases=sockets["cases"],
    software_results_ref=ref(HERE/"software_contract_results.v1.json"),socket_results_ref=ref(HERE/"socket_address_contract_results.v2.json"),
    synthetic_fixture_inventory_ref=ref(HERE/"software_fixture_inventory.v1.json"),
    current_live_C3_Relate_native7GiB_caps_guards_and_writes_unchanged=True,
    no_actual_credit_capacity_admission_job_Git_or_installation=True,no_original_payload_model_truth_or_stdout_tail_reads=True,
    no_real_scope_measurement_or_process_device_socket_probe_or_accounting_operation=True,no_parent_prefix_or_projection_selected=True,
    no_expiry_completion_optimizer_or_historical_drain_invented=True,
    v1_seal_ref=ref(OLD/"source_handoff.v1.json"),v1_all17_authored_files_unchanged=True,predecessor_review_ref=ref(REVIEW),
    F1_resolution="Preserve and reject all unqualified retained filesystem sockets; spelling filters alone never grant inactivity",
    D1_resolution="This report binds the final exact direct_role_plan ref",independent_v2_review="PENDING")
save("source_static_checks.v1.json",audit)
require(read(HERE/"source_static_checks.v1.json")["direct_role_plan_ref"]==ref(HERE/"direct_role_plan.v1.json"),"D1 final exact roleplan hash")
print(json.dumps(dict(status="PASS",runtime_refs=222,Python_closure=len(pyrefs),ordinary_ASTs=30,
    concrete_roles=58,state_roles=62,invocation_roles=64,software_cases=43,socket_cases=56,
    preserved_v1_files=17,fixture_bytes=software["regular_fixture_payload_bytes"]),sort_keys=True))

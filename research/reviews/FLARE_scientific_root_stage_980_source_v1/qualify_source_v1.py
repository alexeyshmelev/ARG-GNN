"""Read-only source, pure-formula and bounded metadata qualification; never admits or executes scientific payloads."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,types

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/FLARE_scientific_root_stage_980_source_v1"
OLD=ROOT/"research/reviews/FLARE_scientific_root_stage_source_v1"
BUILD=ROOT/"research/reviews/FLARE_existing_JDK_source_v1"
checks=[]
read_paths=set()
synthetic={}
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(dict(name=name,passed=True))
def need(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project path")
    return p
def ref(value):
    p=local(value);rel=p.relative_to(ROOT).as_posix()
    if rel in synthetic:return dict(path=rel,sha256=hashlib.sha256(json.dumps(synthetic[rel],sort_keys=True).encode()).hexdigest())
    need(p.suffix in (".json",".py",".sh"),"Source/metadata only")
    need(p.stat().st_size<=262144,"Bounded source/metadata")
    read_paths.add(rel)
    return dict(path=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    need(ref(record["path"])==record,"Exact SHA")
    p=local(record["path"]);rel=p.relative_to(ROOT).as_posix()
    need(p.suffix==".json","JSON metadata only")
    if rel in synthetic:return copy.deepcopy(synthetic[rel])
    need(p.stat().st_size<=65536,"Compact metadata")
    return json.loads(p.read_text())
def unique(records):
    out=[];seen={}
    for r in records:
        need(set(r)=={"path","sha256"},"Exact refs")
        need(r["path"] not in seen or seen[r["path"]]==r,"No contradictory refs")
        if r not in out:out.append(r);seen[r["path"]]=r
    need(len(out)<=64,"At most64 direct bindings")
    return out
def synth(name,value):
    rel="research/reviews/FLARE_scientific_root_stage_980_source_v1/_SYNTHETIC_METADATA_ONLY_/"+name+".json"
    synthetic[rel]=copy.deepcopy(value)
    return ref(rel)
def expect_reject(name,fn):
    try:fn()
    except (ValueError,KeyError):check(name,True)
    else:check(name,False)
new_text=(HERE/"root_stage_980_driver.py").read_text()
old_text=(OLD/"root_stage_driver.py").read_text()
context_text=(HERE/"root_context_980_v1.py").read_text()
old_context=(BUILD/"root_context_v1.py").read_text()
new_tree=ast.parse(new_text);old_tree=ast.parse(old_text)
for name,source in [("driver",new_text),("context",context_text)]:
    compile(source,name,"exec");check(name+"_compiles_without_import",True)
def definitions(tree):
    return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
nd,od=definitions(new_tree),definitions(old_tree)
for name in ("metadata","argument_ref","target_v1","conversion_v1","config_v1"):
    check(name+"_AST_identical_to_frozen_e017",ast.dump(nd[name])==ast.dump(od[name]))
check("WaitWriter_AST_identical_to_frozen46d4",
      ast.dump(definitions(ast.parse(context_text))["WaitWriter"])==ast.dump(definitions(ast.parse(old_context))["WaitWriter"]))
class NormalizeContext(ast.NodeTransformer):
    replacements={
      "Separate prospective980GB root IO context for original FLARE scientific stages.":"Minimal current-credit-v3 root context for original FLARE scientific stages.",
      "FLARE_existing_JDK_current980_root_admission_v1":"FLARE_existing_JDK_current_v3_root_admission_v1",
      "Actual unchanged prospective980GB helper/policy/repair":"Actual unchanged current-credit-v3 helper/policy/repair",
      "prospective_980gb_v1":"concurrent_completed_scope_credits_v3"}
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id in ("SOURCE","GUARD_REF","POLICY_REF") for t in node.targets):return None
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value in self.replacements:node.value=self.replacements[node.value]
        return node
check("whole_context_AST_identical_except_exact_IO_declarations_schema_mode_and_message",
 ast.dump(NormalizeContext().visit(ast.parse(context_text)))==ast.dump(NormalizeContext().visit(ast.parse(old_context))))
names={"SOURCE","PACKET","BUILD_DRIVER_REF","CONTEXT_REF","GUARD_REF","POLICY_REF","REPAIR_REF","TEMPLATES","BUILD_INPUT_REF","CAPS","COMMIT","VERSION","STAGE_SOURCE","ROLES"}
selected=[]
for n in new_tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):selected.append(n)
    elif isinstance(n,ast.FunctionDef) and n.name in ("metadata","argument_ref","target_v1","conversion_v1","config_v1"):selected.append(n)
resource={}
exec(compile((BUILD/"resource_v1.py").read_text(),"pure_resource","exec"),resource)
class PureImport:
    @staticmethod
    def import_module(name):
        need(name=="research.reviews.FLARE_existing_JDK_source_v1.resource_v1","Only pure formula import")
        return types.SimpleNamespace(plan_v1=resource["plan_v1"])
b=types.SimpleNamespace(need=need,local=local,ref=ref,read=read,unique=unique,AUX_ALLOWANCE=524288,
 MANIFEST_REF=ref(BUILD/"source_manifest.json"),
 RUNTIME_BASIS_REF=ref("research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json"))
g=dict(ROOT=ROOT,Path=Path,b=b,importlib=PureImport)
exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),"extracted_metadata_only","exec"),g)
frozen=read(b.MANIFEST_REF)
check("original19_sources_unchanged",len(frozen["source_sha256"])==19 and all(ref(p)==dict(path=p,sha256=h) for p,h in frozen["source_sha256"].items()))
check("original_build_helper_unchanged",ref(g["BUILD_DRIVER_REF"]["path"])==g["BUILD_DRIVER_REF"])
check("original_e017_stage_driver_unchanged",ref(OLD/"root_stage_driver.py")["sha256"]=="e017a58d0a8d5374e1e8d70662a8ba1b8b445737e5c383bed19558c1e3b9f287")
check("original_source_packet_qualification_unchanged",read(ref(OLD/"source_static_receipt.json"))["passed"]==40)
check("new_context_exact_pin",ref(g["CONTEXT_REF"]["path"])==g["CONTEXT_REF"])
check("actual980_guard_policy_repair_exact_pins",all(ref(g[n]["path"])==g[n] for n in ("GUARD_REF","POLICY_REF","REPAIR_REF")))
target_ref=ref(OLD/"train000_target.json")
check("original_target_pin",target_ref["sha256"]=="a944251c7e5b0d7cc311192b2e953aac4f2d193235f4e8834c5f7745406e954b")
target,obs,count,plan,target_refs=g["target_v1"](target_ref)
check("actual_TRAIN000_metadata_and_full_count",target["scientific_role"]==g["ROLES"]["train"] and count["num_sites"]==226185)
check("original_formula_and_caps",g["CAPS"]==dict(convert=950000000,native=9000000000,seal=32000000) and
 plan["conversion_original_formula_bytes"]==876994450 and plan["native_original_formula_with1GiB_tmp_bytes"]==8499254001
 and plan["native_VCF_original_formula_bytes"]==7373083377 and plan["seal_original_formula_without_optional_tract_TSV_bytes"]==21897216)
bad=copy.deepcopy(target);bad["scientific_role"]=g["ROLES"]["val"]
expect_reject("TRAIN_cannot_be_relabelled_heldout",lambda:g["target_v1"](synth("bad_target_role",bad)))
focal=read(target["focal_completion_ref"]);bad_focal=copy.deepcopy(focal);bad_focal["query_ancestry_truth_opened"]=True
bad=copy.deepcopy(target);bad["focal_completion_ref"]=synth("bad_focal_truth",bad_focal)
expect_reject("truth_opened_focal_metadata_rejected",lambda:g["target_v1"](synth("bad_target_focal",bad)))
bad_done=read(target["observation_completion_ref"]);bad_done["status"]="partial"
bad=copy.deepcopy(target);bad["observation_completion_ref"]=synth("bad_observed_done",bad_done)
expect_reject("incomplete_observation_rejected",lambda:g["target_v1"](synth("bad_target_done",bad)))
args=types.SimpleNamespace(conversion=None,expected_conversion_sha256=None,build_completion=None,expected_build_completion_sha256=None,native_run=None,expected_native_run_sha256=None)
convert_cfg,convert_extra=g["config_v1"]("convert",target,count,plan,args)
check("convert_original_policies_and_all_sites",convert_cfg["max_sites"]==226185 and convert_cfg["multiallelic_policy"]=="preserve_all_supported_2to4_ACGT_alleles" and convert_cfg["unsupported_site_policy"]=="reject_no_silent_filter")
conv=dict(status="completed_C3_FLARE_observation_conversion",author_Table1_exact_reproduction=False,
 observation_ref=target["observation_ref"],focal_registry_ref=target["focal_registry_ref"],simulation_config_ref=target["simulation_config_ref"],
 parent_id=target["parent_id"],input_sites=count["num_sites"],reference_haplotypes=600,query_haplotypes=1000,native_expected_markers=100,
 common_focal_records=10000,unsupported_sites_dropped=0,query_ancestry_truth_opened=False,common_focal_registry_unchanged=True,
 native_options=convert_cfg["native_options"],projection=convert_cfg["projection"],native_source_commit=g["COMMIT"])
conv_ref=synth("future_conversion_COMPLETION_NOT_ACTUAL",conv)
args.conversion=conv_ref["path"];args.expected_conversion_sha256=conv_ref["sha256"]
native_cfg,native_extra=g["config_v1"]("native",target,count,plan,args)
check("actual_completed_build_metadata_and_actual_wait0_validates",len(native_extra)==9 and read(native_extra[2])["status"]=="completed_verified_official_source_build")
check("native_same_author_options_heap_and_VCF_cap",native_cfg["native_options"]==dict(probs=True,em=True,**{"update-p":False},array=False,**{"min-maf":.005,"min-mac":50},gen=10,**{"panel-probs":False},nthreads=4,seed=-99999)
 and native_cfg["heap_gib"]==16 and native_cfg["max_native_vcf_bytes"]==7373083377 and native_cfg["timeout_seconds"]==14400)
wait=dict(actual_direct_native_wait=True,returncode=0,owned_process_group_ESRCH=True,operational_failure=None)
wait_ref=synth("future_native_WAIT_NOT_ACTUAL",wait)
native=dict(status="completed_C3_native_FLARE",author_Table1_exact_reproduction=False,conversion_ref=conv_ref,parent_id=target["parent_id"],
 focal_registry_ref=target["focal_registry_ref"],query_ancestry_truth_opened=False,native_outputs_modified=False,native_options=native_cfg["native_options"],
 projection=native_cfg["projection"],source_commit=g["COMMIT"],native_version=g["VERSION"],native_actual_wait_ref=wait_ref,
 native_output_refs={suffix:dict(path="UNOPENED_"+suffix,sha256="0"*64) for suffix in (".anc.vcf.gz",".model",".global.anc.gz",".log")},
 binary_provenance_ref=native_cfg["binary_provenance_ref"])
native_ref=synth("future_native_COMPLETION_NOT_ACTUAL",native)
args.native_run=native_ref["path"];args.expected_native_run_sha256=native_ref["sha256"]
seal_cfg,seal_extra=g["config_v1"]("seal",target,count,plan,args)
check("seal_original_call_path_all_markers10000focals",seal_cfg["mode"]=="seal_native" and seal_cfg["max_markers"]==226185 and seal_cfg["max_focal_records"]==10000)
check("seal_only_optional_tract_table_disabled",seal_cfg["write_derived_marker_tracts"] is False and seal_cfg["max_tract_records"]==0)
bad_wait=copy.deepcopy(wait);bad_wait["returncode"]=1;bad=copy.deepcopy(native);bad["native_actual_wait_ref"]=synth("bad_wait",bad_wait)
bad_native_ref=synth("bad_native",bad)
args.native_run=bad_native_ref["path"];args.expected_native_run_sha256=bad_native_ref["sha256"]
expect_reject("seal_rejects_failed_native_wait",lambda:g["config_v1"]("seal",target,count,plan,args))
check("all_configs_truth_unopened",all(c["external_truth_ref"] is None for c in (convert_cfg,native_cfg,seal_cfg)))
roster_ref=ref("research/reviews/prospective_980GB_capacity_source_v1/admitted_simulate_v1/reserve_authority.json")
roster=read(roster_ref)["bindings"];resource_refs=unique(list(roster.values()))
check("actual_resource_roster_sources_and_metadata_unchanged",all(ref(r["path"])==r for r in resource_refs))
code=dict(frozen["source_sha256"])
for r in unique([ref(HERE/"root_stage_980_driver.py"),g["BUILD_DRIVER_REF"],g["CONTEXT_REF"],g["GUARD_REF"],*resource_refs]):
    if Path(r["path"]).suffix==".py":code[r["path"]]=r["sha256"]
code=dict(sorted(code.items()));closure=[dict(path=p,sha256=h) for p,h in code.items()]
counts={};stage_metacounts={}
for stage,extra in (("convert",convert_extra),("native",native_extra),("seal",seal_extra)):
    registration=synth("prospective_"+stage+"_registration",{})
    cfg_ref=synth("prospective_"+stage+"_config",{})
    runtime_ref=synth("prospective_"+stage+"_runtime",{})
    manifest_ref=synth("prospective_"+stage+"_manifest",{})
    own=ref(HERE/"root_stage_980_driver.py")
    meta=unique([r for r in frozen["required_metadata_refs"] if r!=frozen["observation_count_manifest_ref"]]+
      [b.MANIFEST_REF,g["BUILD_DRIVER_REF"],g["TEMPLATES"][stage],target_ref,*target_refs,registration,*extra,b.RUNTIME_BASIS_REF,
       g["POLICY_REF"],g["REPAIR_REF"],*[r for r in resource_refs if Path(r["path"]).suffix!=".py"]])
    helpers=unique([r for r in closure if any(s in r["path"] for s in ("project_budget","failed_scope_capacity","marker_repair","completed_scope_credits"))]+resource_refs)
    direct=unique([own,g["BUILD_DRIVER_REF"],g["CONTEXT_REF"],ref(g["STAGE_SOURCE"][stage]),cfg_ref,runtime_ref,manifest_ref,
      g["GUARD_REF"],g["POLICY_REF"],g["REPAIR_REF"],*helpers,*closure,*meta])
    consumers=unique([*direct,*roster.values()])
    counts[stage]=len(consumers);stage_metacounts[stage]=len(meta)
    check(stage+"_prospective_direct_bindings_at_most64",len(consumers)<=64)
    check(stage+"_all_actual_resources_and_full_source_closure_directly_bound",all(r in consumers for r in [*resource_refs,*closure]))
    check(stage+"_only_py_sh_json_direct_bindings",all(Path(r["path"]).suffix in (".py",".sh",".json") for r in consumers))
check("ordered_complete_source_closure",list(code)==sorted(code) and all(ref(r["path"])==r for r in closure))
admit=ast.get_source_segment(new_text,nd["admit"]);launch=ast.get_source_segment(new_text,nd["launch"]);worker=ast.get_source_segment(new_text,nd["worker"])
check("admit_exact_active980_configure_route_no_old_adapter","b.adapter(" not in new_text and "m.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)" in new_text and 'm.MODE=="prospective_980gb_v1"' in new_text)
check("guard_functions_original_unmemoized","getattr(m.budget.v4,n).__module__==m.budget.v4.__name__" in new_text)
check("one_fresh_state_before_full_bindings_and_reserve",admit.count("state=m._state_locked")==1 and admit.index("state=m._state_locked")<admit.index("resource_refs=")<admit.index("consumers=")<admit.index("m.apply_reserve_v1"))
check("worker_empty_pipe_ticket_before_admission_guard_and_science",worker.index("ticket=f.read(65)")<worker.index("load_admission(args)")<worker.index("configured_guard_v1()")<worker.index("context.execute_v1"))
check("precharge_before_Popen",launch.index("required_write_bytes=b.AUX_ALLOWANCE")<launch.index("subprocess.Popen"))
check("seed_owned_identity_and_RSS_then_fsynced_seed_before_release",launch.index("subprocess.Popen")<launch.index("owned[proc.pid]")<launch.index("b.observe(owned,groups)")<launch.index('b.put(aux/"launch_monitor_seed.json"')<launch.index("os.write(wfd"))
check("child_module_correct_and_pipe_explicitly_inherited",'"research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver"' in launch and "pass_fds=(rfd,)" in launch)
check("real_wait_minimal_before_group_checks_guard_or_logs",launch.index("rc=proc.wait(")<launch.index('b.put(aux/"actual_wait_minimal.json"')<launch.index("live=b.absent_groups(groups)")<launch.index("required_write_bytes=len(log)+65536"))
check("postwait_minimal_exclusive_fsynced_original_put",all(s in (BUILD/"root_build_driver.py").read_text() for s in ("os.O_EXCL","os.O_NOFOLLOW","os.fsync(f.fileno())")))
check("no_parent_guard_during_worker_writes","budget_check_v9" not in launch[launch.index("while proc.poll() is None:"):launch.index("finally:")])
check("monitor_reuses_original_process_helpers",all(s in launch for s in ("b.process_info(","b.observe(","b.signal_owned(","b.absent_groups(")))
check("polling_soft_bounds_disclosed","RSS_bound_is_polling_soft_bound_not_kernel_quota=True" in launch and "unobserved_between_poll_processes_possible=True" in launch)
check("no_score_admission_route",set(g["CAPS"])=={"convert","native","seal"} and 'choices=tuple(CAPS)' in new_text)
check("no_monkeypatch_or_guard_import_in_metadata_checks",not any(Path(p).suffix not in (".json",".py",".sh") for p in read_paths))
receipt=dict(schema="FLARE_scientific_root_stage_980_source_static_receipt_v1",checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source_ref=ref(HERE/"root_stage_980_driver.py"),context_ref=g["CONTEXT_REF"],qualifier_ref=ref(HERE/"qualify_source_v1.py"),
 passed=len(checks),checks=checks,prospective_direct_binding_counts=counts,prospective_required_metadata_counts=stage_metacounts,
 source_closure_count=len(closure),source_closure_refs=closure,actual_resource_roster_preview_ref=roster_ref,
 source_freeze_basis_ref=b.MANIFEST_REF,active_guard_ref=g["GUARD_REF"],active_policy_ref=g["POLICY_REF"],
 stage_caps_bytes=g["CAPS"],actual_TRAIN_target_metadata_validated=True,actual_successful_build_and_wait0_metadata_validated=True,
 genotype_focal_tree_JAR_or_query_truth_payload_opened=False,actual_guard_configuration_or_admission_executed=False,
 actual_Popen_Java_native_build_or_GPU_executed=False,Git_executed=False,actual_heldout_validation_performed=False,
 prospective_predecessor_fixtures_are_in_memory_synthetic_only=True,read_metadata_and_source_paths=sorted(read_paths),
 note="Source AST/compilation/hash checks, bounded actual JSON metadata, pure resource formulas and in-memory synthetic future predecessors only. Binding counts preview existing actual980 roster; root admission re-reads live roster and enforces64.")
data=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()
out=HERE/"source_static_receipt.json"
need(not out.exists(),"Preserve prior receipt")
with out.open("xb") as f:f.write(data)
print(json.dumps(dict(passed=len(checks),bindings=counts,source_closure_count=len(closure),source_ref=receipt["source_ref"],receipt_ref=ref(out)),sort_keys=True))

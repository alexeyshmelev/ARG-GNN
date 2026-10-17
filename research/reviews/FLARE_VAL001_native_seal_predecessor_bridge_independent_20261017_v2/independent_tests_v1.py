"""Independent source/AST and bounded actual JSON metadata review. No guards or science."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
C=ROOT/"research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2"
REVIEW=ROOT/"research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_independent_20261017_v2"
def ref(p):
    p=Path(p);p=p if p.is_absolute()else ROOT/p
    assert p.is_file()and not p.is_symlink()and p.stat().st_size<=32*2**20
    assert p.suffix in(".py",".json",".md",".txt",".sh",".patch",".yaml",".yml")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def jread(p):return json.loads((p if isinstance(p,Path)else ROOT/p).read_text())
def fn(text,name):return copy.deepcopy(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name))
# Reproduce candidate49 cases, redirecting only the final result path.
tree=ast.parse((C/"integration_check_v2.py").read_text())
outputs=[n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="p"for t in n.targets)]
assert len(outputs)==1
outputs[0].value=ast.parse('REVIEW/"candidate_reproduced_results.v2.json"',mode="eval").body
ast.fix_missing_locations(tree);candidate_ns=dict(REVIEW=REVIEW,__name__="_independently_reproduced_candidate_RAM")
exec(compile(tree,"<candidate49_checks_only_result_path_relocated>","exec"),candidate_ns)
assert candidate_ns["result"]["case_count"]==49
cases=[]
def check(name,ok=True):assert ok,name;cases.append(dict(name=name,status="PASS"))
def rejects(name,action):
    try:action()
    except (ValueError,KeyError,TypeError,AssertionError):check(name)
    else:raise AssertionError("Expected rejection: "+name)
bridge=candidate_ns["bridge"];legacy=candidate_ns["legacy"];key=candidate_ns["key"]
manifest=jread(C/"source_manifest.v2.json");baseline=jread("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2/source_manifest.v1.json")
check("B1_resolved_exact_full148metadata_list_in_original_order",manifest["metadata_refs"]==baseline["metadata_refs"]and len(manifest["metadata_refs"])==148)
check("all153_original_source_pins_preserved",len(baseline["source_sha256"])==153 and all(manifest["source_sha256"][p]==s for p,s in baseline["source_sha256"].items()))
for name in("__init__.py","predecessor_bridge_v1.py","root_driver_v1.py","worker_entry_v1.py","prepare_context_v1.py"):
    v2=(C/name).read_text();v1=(C.parent/(C.name[:-1]+"1")/name).read_text()
    normalized=v2.replace(C.name,C.name[:-1]+"1").replace("source_manifest.v2.json","source_manifest.v1.json").replace("source_handoff.v2.json","source_handoff.v1.json")
    check("v2_production_changes_namespace_paths_only:"+name,ast.dump(ast.parse(normalized),include_attributes=False)==ast.dump(ast.parse(v1),include_attributes=False))
# Genuine bounded actual metadata only. Never follow VCF/NPZ/JAR/tensor refs.
actual_reads={}
def actual_read(r):
    assert Path(r["path"]).suffix==".json"
    assert ref(r["path"])==r
    actual_reads[r["path"]]=r
    return jread(r["path"])
inp=actual_read(bridge.INPUT_REF);wait=actual_read(bridge.WAIT_REF);converted=actual_read(bridge.COMPLETED_REF)
close_ref=ref("research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/ordinary_close_receipt.root_v1.json")
check("actual_ordinary_close_receipt_exact3da38af7",close_ref["sha256"]=="3da38af704512caa5368d013cd4b4febaf135b593c8de06834515552954c2a7b")
closed=actual_read(close_ref)
outer_ref=ref("research/reviews/research_execution_ROOT_20261017_v22/FLARE_VAL001_conversion_actual_outer_consumed.root_v1.json");outer=actual_read(outer_ref)
check("actual_v4_full_source_mapping_and_input_wait_completion_join",inp["source_sha256"]==jread(bridge.V4_MANIFEST_REF["path"])["source_sha256"]and inp["root_driver_ref"]==bridge.V4_DRIVER_REF and inp["main_source_ref"]==bridge.V4_ENTRY_REF and wait["input_ref"]==bridge.INPUT_REF and wait["completed_ref"]==bridge.COMPLETED_REF)
check("actual_worker0_drained_no_failure_and_consumed_conversion_outer0",wait["actual_direct_worker_wait"]is True and wait["returncode"]==0 and wait["all_owned_processes_and_groups_drained"]is True and wait["operational_failure"]is None and outer["actual_terminal_consumed"]is True and outer["actual_exit_code"]==0 and outer["actual_wait_ref"]==bridge.WAIT_REF and outer["completed_ref"]==bridge.COMPLETED_REF)
check("actual26609native_markers_distinct_from_full227316input_sites",converted["input_sites"]==227316 and converted["native_expected_markers"]==26609 and converted["reference_haplotypes"]==600 and converted["query_haplotypes"]==1000 and converted["common_focal_records"]==10000)
reserve=inp["storage"]["reservation"]
check("actualclose_exact_original_reservation_hash_and_revocation",closed["key"]==bridge.KEY and closed["artifacts_deleted"]==0 and closed["value"]["future_writes_and_relaunch_revoked"]is True and closed["value"]["request"]["completed_ref"]==bridge.COMPLETED_REF and closed["value"]["request"]["direct_wait_ref"]==bridge.WAIT_REF and closed["value"]["reservation_sha256"]==hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest())
# Execute original full context validator/config builder against genuine JSONs only.
core_path=ROOT/"research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/completion_v1.py"
core_text=core_path.read_text();core=types.ModuleType("_independent_original_FLARE_metadata_only")
core.__dict__.update(HERE=core_path.parent,ROOT=ROOT,copy=copy,json=json,MISSING=tuple("american_c3_val_%03d"%i for i in range(1,5)),require=bridge.require,OPTIONS=candidate_ns["basis"]["configurations"]["native_config"]["native_options"],COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1",CLASSES=("EUR","EAS","AFR"),_tools=lambda:(actual_read,lambda *a:(_ for _ in()).throw(AssertionError("No payload bound")),ref))
for name in("_local","_local_ref","_sources","_context","_gzip_bound","_conversion_bound","_native_output_bound","inactive_stage_configs_v1"):
    exec(compile(ast.Module(body=[fn(core_text,name)],type_ignores=[]),"<exact_original_context_and_config_metadata_functions>","exec"),core.__dict__)
actual_request=actual_read(inp["root_stage_request_ref"]);context_ref=actual_request["context_ref"]
actual_context=core._context(context_ref,parent_id=bridge.PARENT)
check("exact_original_context_validator_accepts_actual_full30Mb_VAL001_JSON_lineage",actual_context["targets"][bridge.PARENT]["input_sites"]==227316 and actual_context["targets"][bridge.PARENT]["haplotypes"]==1600)
configs=core.inactive_stage_configs_v1(context_ref,bridge.PARENT,conversion_ref=bridge.COMPLETED_REF)
check("exact_original_count_derived_native_caps_use26609markers",configs["native"]["max_native_vcf_bytes"]==871184687 and configs["native"]["output_cap_bytes"]==1997355311)
check("exact_original_seal_support_and_native_options_unchanged",configs["seal"]["max_markers"]==26609 and configs["seal"]["max_focal_records"]==10000 and configs["seal"]["write_derived_marker_tracts"]is False and configs["seal"]["max_tract_records"]==0 and configs["native"]["heap_gib"]==16 and configs["native"]["timeout_seconds"]==14400 and configs["native"]["native_options"]==converted["native_options"])
# Whole wrapper/contract path with actual predecessor docs and actual original context.
def actual_fixture():
    c,records,req_ref,raw_calls,current_calls=candidate_ns["fixture"]()
    q=records[key(req_ref)];q["context_ref"]=context_ref
    c.completion=lambda:core
    original_read=c.read
    def read(r):
        if key(r)in records:return copy.deepcopy(records[key(r)])
        return actual_read(r)
    c.read=read
    target=actual_context["targets"][bridge.PARENT]
    records[key(q["registration_ref"])].update(observation_ref=target["observation_ref"],focal_registry_ref=target["focal_registry_ref"],count_manifest_ref=target["count_manifest_ref"])
    records[key(bridge.INPUT_REF)]=copy.deepcopy(inp);records[key(bridge.WAIT_REF)]=copy.deepcopy(wait);records[key(bridge.COMPLETED_REF)]=copy.deepcopy(converted)
    records[key(legacy["CLOSE"])]=copy.deepcopy(closed)
    # Raw genuine conversion terminal lacks the original contract's direct_wait_ref alias.
    records[key(legacy["OUTER"])]=copy.deepcopy(outer)
    caps=c.stage_caps("native",c.make_config(q));q["caps"]=caps
    records[key(q["root_directive_ref"])].update(context_ref=context_ref,caps=caps)
    return c,records,req_ref
c,docs,req=actual_fixture()
rejects("raw_actual_outer_schema_requires_truthful_canonical_wait_ref_alias",lambda:c.validate_request(req))
# This alias exists in RAM only and explicitly preserves the original observed record.
docs[key(legacy["OUTER"])].update(direct_wait_ref=bridge.WAIT_REF,original_actual_outer_terminal_ref=outer_ref)
q,caps,_=c.validate_request(req)
check("whole_actual_wrapper_contract_accepts_real_context_v4wait_input_and_actualclose_with_hypothetical_ROOT_certificates",q["stage"]=="native"and caps["reservation_bytes"]==2005743919 and caps["rss_bytes"]==24*2**30 and caps["cpu_threads"]==4)
for name,r,field,value in(("final_journal_pending",legacy["FINAL"],"final_original_journal_accepted",False),("wrong_final_selector",legacy["FINAL"],"storage_policy_selector_ref",legacy["E"]),("unconsumed_close_outer",legacy["CLOSE_OUTER"],"actual_terminal_consumed",False),("wrong_close_receipt",legacy["CLOSE_OUTER"],"ordinary_close_receipt_ref",legacy["E"]),("wrong_conversion_input",legacy["E"],"input_ref",legacy["E"]),("failed_worker",bridge.WAIT_REF,"returncode",1)):
    old=docs[key(r)][field];docs[key(r)][field]=value
    rejects("full_real_metadata_route_rejects:"+name,lambda:c.validate_request(req));docs[key(r)][field]=old
old=docs[key(legacy["ACCEPT"])].pop("final_original_journal_acceptance_ref")
rejects("missing_final_original_journal_certificate_blocks_whole_native_request",lambda:c.validate_request(req));docs[key(legacy["ACCEPT"])]["final_original_journal_acceptance_ref"]=old
# Seal predecessor remains genuine same-new-source native branch, not the v4 exception.
seal_c,seal_docs,seal_req,_,_=candidate_ns["fixture"]("seal")
seal_c.validate_request(seal_req);check("full_new_source_native_predecessor_for_seal_still_validated")
check("all_resource_calls_remain_RAM_stubs_no_runtime_guard_import",not any(n=="torch"or n.startswith(("torch.","numpy","arggnn","composer"))for n in sys.modules))
closure=jread(REVIEW/"source_closure.v1.json")
check("candidate17files_unchanged_after_tests",all(ref(p)["sha256"]==s for p,s in closure["candidate_baseline_sha256"].items()))
result=dict(schema="independent_FLARE_v2_source_full_actual_metadata_RAM_review_v1",status="PASS_SOURCE_AST_RAM_ONLY",candidate_reproduced=49,independent_checks=len(cases),total_checks=49+len(cases),cases=cases,actual_JSON_metadata_refs=list(actual_reads.values()),actual_close_receipt_ref=close_ref,conversion_actual_outer_ref=outer_ref,final_original_journal32481_consumption_or_acceptance_observed=False,pending_ROOT_certificates_in_RAM_only=True,real_guard_state_process_resource_GPU_payload_tensor_truth_numerical_admission_launch_or_Git_action=False)
with (REVIEW/"independent_results.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in result.items()if k not in("cases","actual_JSON_metadata_refs")},sort_keys=True))

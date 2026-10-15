"""Read-only source/compact-manifest checks. Imports no driver/guard/science."""
from pathlib import Path
import argparse,ast,copy,difflib,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/H018_observed_bridge_root_source_v1"
FOLDER=ROOT/REL

def raw(name):return (FOLDER/name).read_bytes()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
def data(name):return json.loads(raw(name))
def dump(node):return ast.dump(node,include_attributes=False)
def functions(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
def assignments(tree):return {t.id:n.value for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}

def check():
    old=raw("root_bridge_driver_v2.py").decode();new=raw("root_bridge_driver_v3.py").decode()
    ot,nt=ast.parse(old),ast.parse(new);of,nf=functions(ot),functions(nt);oa,na=assignments(ot),assignments(nt)
    oldm,m=data("source_manifest_v2.json"),data("source_manifest_v3.json")
    oldc,c=data("config.json"),data("config_v3.json")
    oldr,r=data("root_request.inactive_v2.json"),data("root_request.inactive_v3.json")
    checks={}
    checks["old_v2_source_unchanged"]=sha(FOLDER/"root_bridge_driver_v2.py")=="c06b56f138e8aa31386225935eb8549161a5a77610decb4170d9f3096b3cdd26"
    checks["old_v2_manifest_and_inactive_request_unchanged"]=sha(FOLDER/"source_manifest_v2.json")=="7797099ad30254563b714baec2daf25061883bedee800fe85af1fab72384187b" and sha(FOLDER/"root_request.inactive_v2.json")=="7614755ef40e92666c2ebcef473b04b36e6ccc14784134a2cd1ca44315de037f"
    checks["old_original_driver_config_manifest_unchanged"]=sha(FOLDER/"root_bridge_driver.py")=="e1433363ffdf0fd70b50f45edaa5ffa1fae28a577ecc466baabca4b2ba2233ef" and sha(FOLDER/"config.json")=="7a54190e9a1c02650d08020516ae5978ceae53560586339f200777d2468b3996" and sha(FOLDER/"source_manifest.json")=="3713aff850388ab18accf6f56aa6d42cf91fe816541b961777c0cf92eab25929"
    suffixes=ast.literal_eval(na["DIRECT_BINDING_SUFFIXES"])
    guard_text=(ROOT/"src/arggnn/project_budget_v4.py").read_text();guard_tree=ast.parse(guard_text)
    scope=next(n for n in guard_tree.body if isinstance(n,ast.FunctionDef) and n.name=="scope_spec")
    suffix_test=next(n for n in ast.walk(scope) if isinstance(n,ast.Compare) and isinstance(n.left,ast.Attribute) and n.left.attr=="suffix" and any(isinstance(op,ast.NotIn) for op in n.ops))
    checks["exact_unchanged_guard_direct_suffixes"]=set(suffixes)==ast.literal_eval(suffix_test.comparators[0])=={".py",".sh",".json"} and m["direct_binding_suffixes"]==list(suffixes)
    excluded={g:{name:rec for name,rec in m[g].items() if Path(rec["path"]).suffix not in suffixes} for g in ("source_refs","metadata_refs")}
    direct=[rec for g in ("source_refs","metadata_refs") for rec in m[g].values() if Path(rec["path"]).suffix in suffixes]
    checks["all4_excluded_full_refs_preserved_transitively"]=excluded==m["transitive_only_refs"] and set(excluded["source_refs"])=={"python_environment_history"} and set(excluded["metadata_refs"])=={"planned.sample","planned.sample_ages.generations","qualified.regional.map"}
    checks["all_source_refs_actual_hashes_match_without_payload_reads"]=all(ref(ROOT/rec["path"])==rec for rec in m["source_refs"].values())
    checks["entire_original_metadata_mapping_preserved_without_opening_targets"]=m["metadata_refs"]==oldm["metadata_refs"]
    checks["original_source_mapping_preserved_except_fresh_driver_config_plus_verifier"]=set(m["source_refs"])==set(oldm["source_refs"])|{"independent_serialization_verifier"} and all(m["source_refs"][k]==v for k,v in oldm["source_refs"].items() if k not in ("driver","config"))
    checks["all_other_original_manifest_fields_preserved"]=all(m[k]==v for k,v in oldm.items() if k!="source_refs")
    checks["config_changes_only_fresh_packet_and_keeps_all_budgets_pins"]=c["packet_directory"]==REL+"/admission_v2" and {k:v for k,v in c.items() if k!="packet_directory"}=={k:v for k,v in oldc.items() if k!="packet_directory"}
    changed={"packet_directory","config_ref","driver_ref","source_manifest_ref"}
    checks["request_changes_only_fresh_packet_and_exact_new_refs"]=all(r[k]==v for k,v in oldr.items() if k not in changed) and r["packet_directory"]==c["packet_directory"] and r["driver_ref"]==m["source_refs"]["driver"]==ref(FOLDER/"root_bridge_driver_v3.py") and r["config_ref"]==m["source_refs"]["config"]==ref(FOLDER/"config_v3.json") and r["source_manifest_ref"]==ref(FOLDER/"source_manifest_v3.json")
    checks["request_remains_inactive_no_registration_or_authority"]=r["status"]=="inactive_source_plan" and r["root_authorizer"] is None and r["registration_ref"] is None and r["registered_before_execution"] is False
    allowed_globals={"SOURCE","CONFIG","MANIFEST","PACKET"}
    checks["all_original_globals_budgets_guard_pins_output_aux_and_PYTHONPATH_kept"]=set(na)==set(oa)|{"DIRECT_BINDING_SUFFIXES"} and all(dump(na[k])==dump(v) for k,v in oa.items() if k not in allowed_globals) and new.count("PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT / \"src\"))")==2
    checks["new_identity_constants_exact"]=all(isinstance(na[k],ast.BinOp) and isinstance(na[k].left,ast.Name) and na[k].left.id=="REL" and ast.literal_eval(na[k].right)==v for k,v in dict(SOURCE="/root_bridge_driver_v3.py",CONFIG="/config_v3.json",MANIFEST="/source_manifest_v3.json",PACKET="/admission_v2").items())
    allowed_functions={"verify_request","admit","admitted_inputs"}
    checks["all_science_worker_supervision_wait_and_other_original_functions_AST_identical"]=set(nf)==set(of)|{"direct_authority_refs"} and all(dump(nf[k])==dump(v) for k,v in of.items() if k not in allowed_functions)
    verify=copy.deepcopy(nf["verify_request"])
    verify.body=[n for n in verify.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="direct_authority_refs")]
    checks["entire_original_request_and_all_full_SHA_validation_AST_preserved"]=dump(verify)==dump(of["verify_request"])
    helper=ast.get_source_segment(new,nf["direct_authority_refs"])
    checks["manifest_and_actual_closure_assert_all_full_maps_and_exclusions"]=all(s in helper for s in ("manifest[\"transitive_only_refs\"] == excluded","closure[group] == manifest[group]","closure[\"transitive_only_refs\"] == excluded","closure[\"direct_binding_suffixes\"] == manifest[\"direct_binding_suffixes\"]"))
    admit=ast.get_source_segment(new,nf["admit"]);worker_inputs=ast.get_source_segment(new,nf["admitted_inputs"])
    checks["admission_filters_only_exact_suffixes_and_keeps_every_actual_authority"]=all(s in admit for s in ("consumers = [x for x in candidates if Path(x[\"path\"]).suffix in DIRECT_BINDING_SUFFIXES]","*view[\"bindings\"].values()]","runtime, closure, *closure_direct, b.GUARD, b.POLICY, b.REPAIR,","for x in authorities)","all(x not in consumers for group in manifest[\"transitive_only_refs\"].values()","len(consumers) <= 64"))
    checks["worker_requires_full_transitive_closure_and_direct_runtime_guard_sources"]=all(s in worker_inputs for s in ("direct_authority_refs(manifest, closure)","b.GUARD, b.POLICY, b.REPAIR, *storage[\"budget_components\"].values()","Path(x[\"path\"]).suffix in DIRECT_BINDING_SUFFIXES and x in own[\"consumer_bindings\"]","All excluded refs retained only in bound manifest/closure JSON"))
    verifier=dict(path="research/reviews/H018_bridge_serialization_readback_source_v1/serialization_readback.py",sha256="eb32b6ad22b81276d88437ffe3c0bcef36e0ccb773788fe8fc19a4b11e12b6ff")
    checks["readback_verifier_exactly_source_bound_and_not_invoked"]=m["source_refs"]["independent_serialization_verifier"]==verifier and verifier in direct and dump(nf["worker"])==dump(of["worker"])
    expected_diff="".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=REL+"/root_bridge_driver_v2.py",tofile=REL+"/root_bridge_driver_v3.py"))
    checks["exact_source_diff_recorded"]=raw("driver_v2_to_v3.diff").decode()==expected_diff
    compile(new,REL+"/root_bridge_driver_v3.py","exec");compile(Path(__file__).read_text(),__file__,"exec")
    checks["compile_as_text_without_driver_or_guard_import"]=True
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    return dict(schema="H018_observed_bridge_v3_source_binding_checks_v1",all_passed=True,checks=checks,
        refs={name:ref(FOLDER/name) for name in ("root_bridge_driver_v3.py","config_v3.json","source_manifest_v3.json","root_request.inactive_v3.json","driver_v2_to_v3.diff","source_checks_v3.py")},
        transitive_only_refs=excluded,known_closure_direct_ref_count=len(direct),source_ref_count=len(m["source_refs"]),metadata_ref_count=len(m["metadata_refs"]),
        dynamic_guard_authority_consumer_count_uninspected=True,source_only=True,guard_or_science_imported=False,
        payloads_or_failed_packet_or_live_state_read=False,admissions_accounting_process_or_worker_execution=False,
        statement="All excluded refs remain in unchanged full source/metadata maps, additionally classified transitively and checked at runtime. Actual guard authority bindings are required directly and cannot be silently filtered. No actual guard acceptance or runtime result claimed.")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--receipt");args=parser.parse_args()
    result=check()
    if args.receipt:
        destination=Path(args.receipt);destination=destination if destination.is_absolute() else ROOT/destination
        assert destination.parent==FOLDER and not destination.exists()
        with destination.open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(dict(all_passed=result["all_passed"],checks=len(result["checks"]),known_closure_direct_ref_count=result["known_closure_direct_ref_count"],refs=result["refs"]),indent=2))

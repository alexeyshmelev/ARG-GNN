"""Read-only qualifier source checks. Never imports driver/guard/native/tskit."""
from pathlib import Path
import ast
import copy
import hashlib
import json

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H018_native_Relate_qualification_source_v1"
OLD=ROOT/"research/reviews/H018_QC_boolean_serialization_repair_source_v1/root_bridge_driver.py"


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def functions(tree):
    return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}


def dump(node):
    return ast.dump(node,include_attributes=False)


def check():
    text=(HERE/"root_qualifier.py").read_text()
    tree=ast.parse(text)
    funcs=functions(tree)
    old=OLD.read_text();of=functions(ast.parse(old))
    cfg=json.loads((HERE/"config.json").read_bytes())
    manifest=json.loads((HERE/"source_manifest.json").read_bytes())
    request=json.loads((HERE/"root_request.inactive.json").read_bytes())
    profile=json.loads((HERE/"fixture_profile.json").read_bytes())
    checks={}
    settings=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SETTINGS" for t in n.targets))
    checks["literal_config_exact_match"]=ast.literal_eval(settings)==cfg
    checks["default_request_inactive_no_authority"]=request["status"]=="inactive_source_plan" and request["root_authorizer"] is None and request["registration_ref"] is None and request["registered_before_execution"] is False and request["native_execution_authorized"] is False
    checks["only_admit_or_launch_no_worker_resume_or_phase_action"]='choices=("admit","launch")' in text and 'No replay, continuation or phase resume' in text
    checks["all_fresh_names_exact"]=cfg["job_id"]=="H018-native-Relate-synthetic-qualification-v1" and cfg["condition_id"]=="H018_Relate_fullsite_native_support_synthetic_v1" and cfg["output_directory"]=="data/H018_native_Relate_synthetic_qualification_v1" and cfg["packet_directory"]==HERE.relative_to(ROOT).as_posix()+"/admission_v1" and cfg["auxiliary_directory"]==HERE.relative_to(ROOT).as_posix()+"/worker_aux_v1"
    checks["exact50MB_combined_storage_and_original_runtime_bounds"]=cfg["reserved_bytes"]==50_000_000 and cfg["output_cap_bytes"]==32_000_000 and cfg["address_space_per_process_bytes"]==1_000_000_000 and cfg["combined_RSS_cap_bytes"]==2_000_000_000 and cfg["Relate_seconds"]==300 and cfg["converter_seconds"]==120 and cfg["total_controller_seconds"]==600 and cfg["termination_grace_seconds"]==10
    checks["CPU_only_no_real_GT_biological_native_or_GPU_authority"]=cfg["CPU_threads"]==1 and cfg["physical_gpu"] is None and cfg["real_GT_reads_authorized"] is False and cfg["biological_native_execution_authorized"] is False and cfg["GPU_authorized"] is False and 'CUDA_VISIBLE_DEVICES=""' in text
    checks["native_file_limit_confined_to_child_before_exec"]="resource.RLIMIT_FSIZE" not in ast.get_source_segment(text,funcs["limit_process"]) and "resource.RLIMIT_FSIZE" in ast.get_source_segment(text,funcs["native_limits"]) and "preexec_fn=native_limits" in text
    checks["copied_guard_independent_helpers_AST_unchanged"]=all(dump(funcs[n])==dump(of[n]) for n in ("require","utc","load","common","syncdir","save","identity","bounded_file","error_record","direct_authority_refs","write_bytes","group_absent"))
    waited=ast.get_source_segment(text,funcs["direct_wait_owned"]).replace("def direct_wait_owned(proc, started, control_error, seconds):","def direct_wait_owned(proc, started, control_error):").replace("now-started > seconds","now-started > SUPERVISOR_SECONDS")
    checks["actual_wait_and_safe_group_signal_logic_AST_unchanged_except_stage_deadline_parameter"]=dump(ast.parse(waited).body[0])==dump(of["direct_wait_owned"])
    checks["real_waitpid_no_guessed_return_code"]="os.waitpid(proc.pid, os.WNOHANG)" in waited and "os.waitstatus_to_exitcode(status)" in waited and 'actual_direct_wait_observed=False' in waited
    checks["wait_finally_wraps_monitor_setup_and_cleanup"]=all(x in ast.get_source_segment(text,funcs["supervise"]) for x in ("finally:","waited=direct_wait_owned","observed_before_risky_postprocessing=True","group_absent(proc.pid)","control_error","print(json.dumps"))
    launch=ast.get_source_segment(text,funcs["launch"])
    checks["both_native_calls_sequential_single_use_true_wait_before_next_stage"]=all(x in launch for x in ('for name,argv,seconds in stages:','finally:\n                if proc is not None:observation,log=supervise','name+"_direct_wait.json"','No next phase after failed/unknown wait or non-drain'))
    checks["tk_load_only_after_native_stage_loop"]=launch.index("import tskit")>launch.index('for name in ("native.anc","native.mut","native.trees")')>launch.index("for name,argv,seconds in stages:")
    checks["precise_fixture_byte_manifest_enforced_before_native"]='Exact frozen tiny synthetic fixture bytes' in launch and launch.index("Exact frozen tiny synthetic fixture bytes")<launch.index("subprocess.Popen")
    checks["no_unowned_process_scan"]=all(x not in text for x in ("pgrep","ps -","/proc/*","iterdir()")) and '"/task/"+str(child)+"/children"' in ast.get_source_segment(text,funcs["owned_rss"])
    checks["source_closure_preserves_all_exclusions_transitively"]=manifest["direct_binding_suffixes"]==[".json",".py",".sh"] and manifest["transitive_only_refs"]=={g:{k:r for k,r in manifest[g].items() if Path(r["path"]).suffix not in (".json",".py",".sh")} for g in ("source_refs","metadata_refs")}
    installation=json.loads((ROOT/cfg["installation_ref"]["path"]).read_bytes())
    checks["exact_installed_tool_and_licence_metadata_pins"]=all(cfg["native_binary_refs"][k]["sha256"]==v["sha256"] and cfg["native_binary_refs"][k]["path"]==Path(v["path"]).relative_to(ROOT).as_posix() for k,v in installation["binaries"].items()) and cfg["license_ref"]["sha256"]==installation["license_sha256"]
    skips={r["path"] for r in cfg["native_binary_refs"].values()}|{cfg["license_ref"]["path"]}
    checks["all_source_and_metadata_hashes_match_except_tools_licence_deferred_to_root"]=all(ref(ROOT/r["path"])==r for g in ("source_refs","metadata_refs") for r in manifest[g].values() if r["path"] not in skips)
    checks["runtime_tools_licence_and_wheel_records_hash_before_execution"]=all(x in text for x in ("verify_runtime_packages(b)","runtime_package_RECORD_refs","bounded_file(b,rec,64_000_000,retain=False)")) and "for group in" in ast.get_source_segment(text,funcs["verify_request"])
    checks["actual_guard_authorities_not_silently_filtered"]=all(x in text for x in ('*view["bindings"].values()','for x in authorities','len(consumers)<=64','*storage["budget_components"].values()'))
    checks["request_and_manifest_final_refs_exact"]=request["driver_ref"]==manifest["source_refs"]["driver"]==ref(HERE/"root_qualifier.py") and request["config_ref"]==manifest["source_refs"]["config"]==ref(HERE/"config.json") and request["source_manifest_ref"]==ref(HERE/"source_manifest.json")
    checks["fixture_profile_only_one_512_site40_haplotype_case"]=profile["sites"]==512 and profile["subjects"]==20 and profile["haplotypes"]==40 and len(profile["constant_rows0"])==6
    helper=(HERE/"fixture_and_verify.py").read_text()
    checks["every_native_site_vector_and_constant_fact_kept"]=all(x in helper for x in ("native_mut_identity_facts","all512_native_vector_facts","six_constant_vector_facts","all20480_biological_nucleotide_and_phase_calls","every_fixture_site_has_full_tree_support"))
    checks["double_sixdecimal_date_policy_and_exact_node_identity"]=all(x in helper for x in ('Decimal.from_float(age)','ROUND_HALF_EVEN','bound=0.5e-6+0.5*spacing','float(ts.node(node).time)==anc_ages[i]'))
    checks["no_tree_genotype_or_chronology_modifications"]=all(x not in helper for x in ("dump_tables(","tables.nodes.set_columns","tables.sites.add_row","tables.mutations.add_row","simplify(","keep_intervals(","set_times("))
    checks["unknown_or_failed_native_never_completes_observed_baseline"]='actual_observed2074_baseline_authorized=False' in text and 'actual_observed2074_baseline_executed=False' in text and 'controller_OS_exit_not_self_certified=True' in text
    for p in HERE.glob("*.py"):compile(p.read_text(),str(p),"exec")
    checks["all_sources_compile_only_without_driver_import"]=True
    checks["single_thread_before_preexec_and_no_implicit_bytecode_writes"]='len(os.listdir("/proc/self/task"))==1' in text and "sys.dont_write_bytecode=True" in text
    failed=[k for k,v in checks.items() if not v]
    assert not failed,failed
    eligible=[r for g in ("source_refs","metadata_refs") for r in manifest[g].values() if Path(r["path"]).suffix in (".json",".py",".sh")]
    return dict(schema="H018_Relate_qualification_source_checks_v1",all_passed=True,checks=checks,count=len(checks),
      eligible_direct_refs=len(eligible),dynamic_guard_bindings_not_inspected=True,
      source_only=True,driver_guard_native_or_tskit_imported=False,actual_GT_read=False,native_or_admission_execution=False,
      installed_binary_and_licence_byte_revalidation_deferred_to_root=True,
      monitored_aggregate_storage_is_not_an_OS_filesystem_quota=True)


if __name__=="__main__":
    print(json.dumps(check(),sort_keys=True,indent=2))

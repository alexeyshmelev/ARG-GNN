"""Bounded H047 root source checks: no guard/admission/runtime/data/CUDA calls."""
from pathlib import Path
from types import SimpleNamespace
import ast,contextlib,copy,hashlib,io,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from arggnn import train_cohort_risk_continuation_v1 as w
from research.reviews.H047_GPU45_cohort_risk_root_source_v1 import root_common_v1 as c,root_driver as driver
from arggnn import H047_GPU45_cohort_risk_worker_v1 as cli
D=ROOT/c.HERE
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError):checks.append(name)
    else:raise AssertionError(name)
def ref(path):
    p=ROOT/path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
cfg=json.loads((D/"config_inactive.json").read_text())
w.validate_protocol(cfg,require_registration=False)
check("inactive_protocol_accepts_without_registration",True)
rejects("runtime_protocol_rejects_missing_registration",lambda:w.validate_protocol(cfg))
registered=copy.deepcopy(cfg)
registered["registration_ref"]=dict(path=w.REGISTRATION_PATH,sha256="a"*64)
w.validate_protocol(registered)
check("runtime_protocol_accepts_exact_synthetic_registration_path_and_SHA",True)
bad=copy.deepcopy(registered);bad["registration_ref"]["path"]="wrong.json"
rejects("runtime_protocol_rejects_wrong_registration_path",lambda:w.validate_protocol(bad))
check("inactive_status_and_caps_require_root_freeze",cfg["status"]=="inactive_root_source_review_pending"
      and cfg["root_resource_caps_are_proposals"] is True and cfg["registration_ref"] is None)
check("fixed_banks128GiB_dense_predictions32GiB_CUDA32e9",cfg["CPU_bank_cap_bytes"]==128*2**30
      and cfg["RAM_prediction_cap_bytes"]==32*2**30 and c.CUDA_CAP==32000000000)
paths=[c.HERE+"/"+x for x in ("root_common_v1.py","context_factory_v1.py","root_driver.py")]
paths += [cli.SOURCE,w.SOURCE,w.CORE_REF["path"]]
for p in paths:compile((ROOT/p).read_text(),p,"exec")
check("all_six_owned_source_files_compile_without_bytecode",True)
check("core_worker_still_match_frozen_24check_receipt",
      ref(w.CORE_REF["path"])==w.CORE_REF and ref(w.SOURCE)["sha256"]=="d542dddc574e93b0bec7801ace9a15d0cc9bd28fb1774a773520f3e29cbd7557")
check("canonical_arggnn_module_and_actual_source_path_match",driver.CLI_MODULE=="arggnn.H047_GPU45_cohort_risk_worker_v1"
      and driver.CLI==cli.SOURCE=="src/"+driver.CLI_MODULE.replace(".","/")+".py")
launcher=(ROOT/"scripts/launch_reserved_gpu_full_storage_v1.py").read_text()
check("actual_launcher_canonical_module_predicate_satisfied",
      driver.CLI_MODULE.startswith("arggnn.") and all(x.isidentifier() for x in driver.CLI_MODULE.split("."))
      and 'args.module.startswith("arggnn.")' in launcher
      and 'module_path = "src/" + args.module.replace(".", "/") + ".py"' in launcher)
cli_source=(ROOT/cli.SOURCE).read_text();driver_source=(ROOT/(c.HERE+"/root_driver.py")).read_text()
context=(ROOT/(c.HERE+"/context_factory_v1.py")).read_text()
check("canonical_main_returns_integer0_after_scientific_call","run_context_v1(context,record);return 0" in cli_source)
check("canonical_actual_source_directly_reserved_and_bound",
      "ref(CLI)" in driver_source and "worker_cli_ref=ref(CLI)" in driver_source
      and 'spec["worker_cli_ref"]==common.ref(SOURCE)' in cli_source)
with contextlib.redirect_stdout(io.StringIO()):
    check("canonical_default_main_inert_return0",cli.main([])==0)
    check("root_driver_default_main_inert_return0",driver.main([])==0)
check("no_torch_CUDA_guard_runtime_modules_imported",
      "torch" not in sys.modules and not any("completed_scope_credits_v3" in n for n in sys.modules))
check("root_driver_requires_root_freeze_before_mutations",
      driver_source.index("Root freezes caps before admission") < driver_source.index("packet.mkdir()"))
check("exact_registration_directly_reserved_and_transitively_bound",
      'cfg["registration_ref"]' in driver_source and "hypothesis_registration_ref" in driver_source
      and 'manifest["hypothesis_registration_ref"]==cfg["registration_ref"]' in context)
manifest=dict(schema="root_H043_original40_observed_count_manifest_v1",
    counts_only_no_labels_or_model_metrics=True,outer_included=False,
    parents={p:dict(native_holds=i+2,native_nodes=(i+2)*1903,haplotypes=952)
             for i,p in enumerate(w.TRAIN+w.VAL)})
train=c.count_plan_v1(manifest,"train");evaluation=c.count_plan_v1(manifest,"evaluate")
check("synthetic_counts_exact32TRAIN8VAL_plans",train["parent_count"]==32 and evaluation["parent_count"]==8)
check("synthetic_allphase_flags_are_summed",train["all_phase_flag_payload_bytes"]==sum(2*manifest["parents"][p]["native_nodes"] for p in w.TRAIN))
sizes=[16*manifest["parents"][p]["native_holds"]*686 for p in w.TRAIN]
check("native_target_retained_plus_FP64_FP32_scratch_bound",train["all_phase_native_hold_FP32_mass_bytes"]==sum(sizes)
      and train["retained_masses_plus_largest_FP64_FP32_construction_upper_bound_bytes"]==sum(sizes)+2*max(sizes))
check("dense96_probability_bound_fits32GiB",evaluation["all96_original_grid_probability_upper_bound_bytes"]==21073920000
      and evaluation["all96_original_grid_probability_upper_bound_bytes"]<=cfg["RAM_prediction_cap_bytes"])
check("both_synthetic_output_plans_fit100MB",all(x["total_output_scope_upper_bound_bytes"]<=100000000 for x in (train,evaluation)))
bad=copy.deepcopy(manifest);bad["parents"].pop(w.VAL[-1])
rejects("counts_missing_original_parent_rejected",lambda:c.count_plan_v1(bad,"train"))
bad=copy.deepcopy(manifest);bad["outer_included"]=True
rejects("outer_counts_rejected",lambda:c.count_plan_v1(bad,"train"))
bad=copy.deepcopy(manifest);bad["parents"][w.TRAIN[0]]["native_holds"]=10000000
rejects("oversized_synthetic_output_forecast_rejected",lambda:c.count_plan_v1(bad,"train"))
pid=w.TRAIN[0];row=manifest["parents"][pid]
bank=SimpleNamespace(parent=SimpleNamespace(numel=lambda:row["native_nodes"]),
    sample_index=SimpleNamespace(shape=(row["native_holds"],952)),readout_hold=[0,1])
declared=w.h42.flag_preparation_plan(row["native_nodes"],952)
proof=c.preparation_check_v1(pid,bank,declared,cfg,manifest,rss_reader=lambda:1234)
check("preparation_uses_injected_RSS_and_actual_synthetic_shapes",
      proof["projected_RSS_planning_bytes"]==1234+2058*row["native_nodes"]+128*952)
rejects("preparation_bad_declared_dimensions_rejected",
        lambda:c.preparation_check_v1(pid,bank,{},cfg,manifest,rss_reader=lambda:0))
rejects("preparation_RSS_exhaustion_rejected",
        lambda:c.preparation_check_v1(pid,bank,declared,cfg,manifest,rss_reader=lambda:c.RSS_CAP))
small=dict(cfg,flag_preparation_cap_bytes=1)
rejects("preparation_scratch_cap_rejected",
        lambda:c.preparation_check_v1(pid,bank,declared,small,manifest,rss_reader=lambda:0))
small=dict(cfg,GPU_flag_cap_bytes=1)
rejects("complete_GPU_flag_payload_cap_rejected",
        lambda:c.preparation_check_v1(pid,bank,declared,small,manifest,rss_reader=lambda:0))
check("retained_target_preflight_in_driver_and_context",
      all('plan["retained_masses_plus_largest_FP64_FP32_construction_upper_bound_bytes"]<=cfg["CPU_target_cap_bytes"]' in s
          for s in (context,driver_source)))
check("TRAIN_lineage_requires_actual960_and_successful_direct_wait",
      all(s in (ROOT/(c.HERE+"/root_common_v1.py")).read_text() for s in
          ('"original_risk":960,"balanced_risk":960','wait.get("returncode")==0','wait.get("actual_direct_launcher_wait") is True')))
receipt=dict(schema="H047_root_bounded_source_static_checks_v1",hypothesis_id="H047",status="passed",
    check_count=len(checks),checks=checks,source_sha256={p:ref(p)["sha256"] for p in paths},
    check_ref=ref(c.HERE+"/source_static_check.py"),
    raw_data_count_corpus_checkpoint_or_truth_arrays_read=False,
    counts_used="tiny injected synthetic dimensions only",actual_process_RSS_probe=False,
    guard_or_native_execution=False,CUDA_operations=False,admission_or_training=False,
    synthetic_count_plans=dict(train=train,evaluate=evaluation))
(D/"source_static_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status="passed",check_count=len(checks),receipt_ref=ref(c.HERE+"/source_static_receipt.json"))))

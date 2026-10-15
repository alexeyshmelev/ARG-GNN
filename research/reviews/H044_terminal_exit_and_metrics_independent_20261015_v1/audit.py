"""Independent H044 source/compact-metadata audit. No project imports/runtime."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ast
import csv
import hashlib
import io
import json
import math
import statistics
import struct

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT/"research/reviews/H044_terminal_exit_and_metrics_independent_20261015_v1"
OUT = "experiments/results/H044_marker_cached_evaluate_seed101_v2/"
PACKET = "research/reviews/H044_cached_evaluation_root_v2/admitted_evaluate_v3/"
PRIMARY = "primary_present_day_182"
ARMS = ("localized", "tree_pooled")
PASSES = (5, 10, 15, 20, 25, 30)
PARENTS = tuple("ap_expanded_val_%03d"%i for i in range(8))
inputs, checks = {}, []


def raw(path):
    p = ROOT/path
    assert p.resolve() == p and p.is_file()
    assert p.suffix in (".json", ".py", ".md", ".csv"), "Source/compact metadata only"
    data = p.read_bytes()
    assert len(data) <= 2_000_000
    inputs[path] = dict(path=path, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return data


def read(path):
    return json.loads(raw(path))


def bound(rec):
    data = raw(rec["path"])
    assert hashlib.sha256(data).hexdigest() == rec["sha256"]
    return json.loads(data)


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(dict(name=name, passed=True))


def close(a, b, atol=1e-12):
    return a is None and b is None or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=atol)


def f32(x):
    return struct.unpack("f",struct.pack("f",x))[0]


def mean8f32(values):
    x = list(map(f32,values))
    assert len(x) == 8
    left = f32(f32(x[0]+x[1])+f32(x[2]+x[3]))
    right = f32(f32(x[4]+x[5])+f32(x[6]+x[7]))
    return f32(f32(left+right)/8)


def summary(values):
    x = [float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(mean=statistics.mean(x) if x else None,
                sample_SD=statistics.stdev(x) if len(x)>1 else None,n=len(x))


def ast_name(node):
    if isinstance(node,ast.Name):
        return node.id
    if isinstance(node,ast.Attribute):
        return ast_name(node.value)+"."+node.attr
    raise ValueError("Nonliteral callable")


def decode_repr(node):
    # Decode only syntax from printed metric metadata, never eval or import NumPy.
    if isinstance(node,ast.Constant):
        return node.value
    if isinstance(node,ast.Dict):
        return {decode_repr(k):decode_repr(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node,(ast.List,ast.Tuple)):
        return [decode_repr(v) for v in node.elts]
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
        return -decode_repr(node.operand)
    if isinstance(node,ast.Call) and ast_name(node.func) in ("array","np.float64"):
        assert len(node.args)==1 and not node.keywords
        return decode_repr(node.args[0])
    raise ValueError("Unsupported printed literal AST "+ast.dump(node)[:100])


def main():
    done = read(OUT+"completed.json")
    check("requested_H044_completion_SHA", inputs[OUT+"completed.json"]["sha256"] ==
          "bc46c8edfaae2ef0894cd8ae057a69ce1d6844bda8fb9262b849709e4b4524a4")
    waited = read(PACKET+"actual_wait.json")
    launch, log = bound(waited["launch_ref"]), bound(waited["log_ref"])
    inner_wait = read(OUT+"launcher_exit_full_storage_v1.json")
    execution = bound(done["input_ref"])
    check("actual_wait1_no_monitor_error_owned_drain", waited["actual_direct_launcher_wait"] is True
          and waited["returncode"] == 1 and waited["error"] is None and waited["owned_group_drained"] is True
          and waited["completed_ref"] == dict(path=OUT+"completed.json",sha256=inputs[OUT+"completed.json"]["sha256"]))
    check("inner_worker_wait1_and_exact_launcher_identity", inner_wait["returncode"] == 1
          and inner_wait["wait_status_known"] is True and inner_wait["launcher_pid"] == waited["launcher_pid"]
          and launch["launcher_pid"] == waited["launcher_pid"] and launch["input_ref"] == done["input_ref"])
    worker_path = "src/arggnn/evaluate_marker_resolution_cached_v2.py"
    producer_path = "research/reviews/H044_cached_evaluation_source_v2/evaluate_cached_v2.py"
    worker_source, producer_source = raw(worker_path).decode(), raw(producer_path).decode()
    check("actual_execution_pins_current_worker_producer_sources",
          execution["main_source_ref"] == dict(path=worker_path,sha256=inputs[worker_path]["sha256"])
          and execution["cache_producer_source_ref"] == dict(path=producer_path,sha256=inputs[producer_path]["sha256"]))
    for path in ("src/arggnn/train_marker_resolution_readout_v1.py",
                 "src/arggnn/mutation_routed_full_panel_primary_v1.py",
                 "src/arggnn/mutation_routed_fast_primary_v1.py",
                 "scripts/train_ancestralpaths_biological_v1.py",
                 "src/arggnn/train_mutation_routed_full_panel_pyg_v1.py",
                 "src/arggnn/train_anonymous_clade_flag_control_v1.py"):
        ast.parse(raw(path))
    check("dict_return_after_completed_write_and_SystemExit_source",
          "return producer.run_context_v2(ctx,record)" in worker_source
          and 'raise SystemExit(main())' in worker_source
          and 'output.save("completed.json",result);return result' in producer_source)
    lines = log["tail"].splitlines()
    first = next(i for i,line in enumerate(lines) if line.startswith("{'schema': 'H044_marker_paired_VAL_completion_v1'"))
    terminal_event = json.loads(lines[-1])
    literal = "\n".join(lines[first:-1])
    tree = ast.parse(literal,mode="eval")
    check("complete_printed_result_exactly_matches_written_completion", decode_repr(tree.body) == done)
    calls = dict(Counter(ast_name(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)))
    check("log_contains_worker_wait1_no_traceback_or_discard", terminal_event ==
          dict(event="supervisor_wait",worker_pid=inner_wait["worker_pid"],returncode=1,wait_status_known=True)
          and not any("Traceback (most recent call last)" in x for x in lines)
          and log["discarded_earlier_bytes"] == 0 and log["total_pipe_bytes"] == log["retained_raw_bytes"] ==
          len(log["tail"].encode()))
    progress = [json.loads(x) for x in lines[:first] if x.startswith('{"stage": "H044_VAL_prediction"')]
    check("log_progress_all96_before_result_print", [x["sealed_products"] for x in progress] == list(range(1,97))
          and all(x["total_products"] == 96 for x in progress))

    matrix = bound(done["complete_probability_matrix_manifest_ref"])
    train = bound(done["train_completion_ref"])
    config = bound(train["config_ref"])
    check("same_train_eval_seed_budget_and_original_score_protocol", execution["refs"]["config"]==train["config_ref"]
          and config["seed"]==101 and config["parameters_per_arm"]==31428
          and config["base_parameters"]==30788 and config["added_parameters"]==640
          and config["passes"]==30 and config["updates_per_pass"]==32 and config["updates_per_arm"]==960
          and config["VAL_influences_optimization"] is False and config["early_stopping"] is False
          and config["metrics"]["boundary_tolerance_bp"]==1000
          and config["metrics"]["probability_floor"] is False and config["metrics"]["purity_filter"] is False
          and config["metrics"]["posthoc_renormalization"] is False
          and config["metrics"]["support"]=="same_ARGMix_original100bp_center_bins_clipped_to_coverage_no_missing_bins_no_tail_extrapolation")
    observations = bound(done["observations_seal"])
    flag_receipt = bound(observations["flag_seal"])
    check("paired_train_completion_bindings", train["schema"]=="H044_marker_paired_train_completion_v1"
          and train["status"]=="completed_fixed30pass_paired_TRAIN" and train["arms"]==list(ARMS)
          and train["steps"]==dict(localized=960,tree_pooled=960)
          and train["VAL_truth_read"] is False and train["outer_truth_read"] is False)
    check("all96_matrix_cartesian_counts_and_before_truth_flags",
          matrix["schema"]=="H044_marker_all96_VAL_barrier_v1" and matrix["arms"]==list(ARMS)
          and matrix["checkpoint_passes"]==list(PASSES) and matrix["VAL_parent_ids"]==list(PARENTS)
          and matrix["complete_dense_prediction_products"]==len(matrix["packets"])==len(matrix["seals"])==96
          and matrix["all96_sealed_before_any_VAL_truth"] is True
          and matrix["outer_truth_read"] is False and done["all96_RAM_seals_before_any_VAL_truth"] is True)
    check("eight_observation_bindings_before_truth", observations["parent_ids"]==list(PARENTS)
          and set(observations["marker_recipes"])==set(PARENTS)
          and observations["truth_read"] is False and observations["all_observations_precede_phase_truth"] is True
          and observations["preparation_completion_ref"]==train["preparation_completion_ref"])
    products, geometry, total_bytes = {}, {}, 0
    for item in matrix["packets"]:
        packet, rec = item["packet"], item["record"]
        key = packet["arm"],packet["pass_index"],packet["parent_id"]
        check("unique_packet_"+str(key), key not in products)
        products[key] = rec
        serialized = (json.dumps(packet,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
        assert rec == dict(path=OUT+"H044_RAM_probability_packets/%s.pass_%02d.%s.json"%key,
                           sha256=hashlib.sha256(serialized).hexdigest())
        assert packet["checkpoint_ref"]==train["checkpoint_groups"][str(key[1])][key[0]]
        assert packet["core_ref"]==train["core_ref"] and packet["targets_opened_before_seal"] is False
        assert packet["prediction_array_durable"] is False
        p = packet["probability"]; count=observations["marker_recipes"][key[2]]["counts"]["scoring_centers"]
        assert p["dtype"]=="float32" and p["shape"]==[count,686,4] and p["bytes"]==count*686*4*4
        assert len(p["sha256"])==64
        g=packet["geometry"]
        assert set(g)=={"positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts","readout_hold"}
        assert all(g[k]["shape"]==[count] for k in ("positions","retained_indexes","readout_hold"))
        assert g["edges"]["shape"]==[count+1]
        assert all(g[k]["shape"]==[686] for k in ("query_slots","raw_query_nodes","cohorts"))
        assert key[2] not in geometry or geometry[key[2]]==g
        geometry[key[2]]=g; total_bytes += p["bytes"]
    check("all96_exact_keys_records_geometry_and_checkpoint_bindings",
          set(products)=={(a,p,i) for a in ARMS for p in PASSES for i in PARENTS}
          and matrix["seals"]==[products[k] for k in sorted(products)]
          and total_bytes==matrix["RAM_prediction_bytes"]==done["RAM_dense_probability_bytes"]==21071417472)
    core_data=raw(train["core_ref"]["path"])
    check("all96_core_ref_matches_source", hashlib.sha256(core_data).hexdigest()==train["core_ref"]["sha256"])
    check("no_durable_dense_predictions_and_no_extra_fit_or_outer_truth", done["dense_prediction_arrays_saved"] is False
          and done["virtual_seals_durable_in_matrix_manifest"] is True
          and matrix["virtual_packet_bytes_resolved_only_by_known_RAM_reader"] is True
          and done["optimization_performed"] is False and done["outer_truth_read"] is False
          and done["same_VAL_is_development_evidence"] is True
          and done["immutable_input_cache_v2"]["extra_fits"] is False
          and done["immutable_input_cache_v2"]["same_original_probability_products"] is True)
    cache=done["immutable_input_cache_v2"]["cache"]
    check("cache_all8_original_binding_recipe_ties", cache["builds"]==8 and set(cache["parents"])==set(PARENTS)
          and all(cache["parents"][pid]["binding_sha256"]==observations["marker_recipes"][pid]["binding_sha256"]
                  and cache["parents"][pid]["targets_used"] is False
                  and cache["parents"][pid]["learned_values_cached"] is False
                  and cache["parents"][pid]["arrays_durable"] is False for pid in PARENTS))

    history_summary, metrics, max_fast_full_difference = {}, {}, 0.0
    expected_counts = dict(primary_present_day_182=182,secondary_ancient_504=504,all_queries_686=686,
        age_stratum_Bronze_Age=162,age_stratum_BAA=14,age_stratum_Neo=302,age_stratum_Yam=26)
    full_by_arm = {}
    for arm in ARMS:
        history=done["history"][arm]
        assert [row["pass_index"] for row in history]==list(PASSES)
        history_summary[arm]=[]
        for row in history:
            cells=row["cells"]
            assert [cell["parent_id"] for cell in cells]==list(PARENTS)
            for c in cells:
                assert c["split"]=="val" and c["prediction_seal"]==products[arm,row["pass_index"],c["parent_id"]]
                g=c["groups"][PRIMARY]
                assert g["query_count"]==182 and g["denominator_query_bp"]==(g["represented_bin_support_bp"][1]-g["represented_bin_support_bp"][0])*182
                assert g["score_arithmetic"]=="float32" and g["backend"]=="numpy"
                assert g["probability_floor_used"] is False and g["posthoc_renormalization"] is False
            means=row["equal_parent_metrics"][PRIMARY]
            for key in ("accuracy","nll"):
                assert mean8f32([c["groups"][PRIMARY][key] for c in cells])==means[key]
            history_summary[arm].append(dict(pass_index=row["pass_index"],**means))
        best=min(history,key=lambda row:(-row["equal_parent_metrics"][PRIMARY]["accuracy"],
                                        row["equal_parent_metrics"][PRIMARY]["nll"],row["pass_index"]))
        check(arm+"_selected_pass30_exact_history_argmax", best==done["selected"][arm] and best["pass_index"]==30)
        full=done["selected_full_diagnostics"][arm]
        assert full["selection_revised"] is False and [x["parent_id"] for x in full["cells"]]==list(PARENTS)
        full_by_arm[arm]={x["parent_id"]:x for x in full["cells"]}
        for cell,fast in zip(full["cells"],best["cells"]):
            assert cell["prediction_seal"]==fast["prediction_seal"]
            groups=cell["groups"]
            assert set(groups)==set(expected_counts)
            for name,g in groups.items():
                q=expected_counts[name];left,right=g["represented_bin_support_bp"];span=right-left
                assert g["query_count"]==q and g["denominator_query_bp"]==g["query_bp"]==q*span
                assert g["represented_span_bp"]==span and g["represented_fraction_original_parent"]==span/2000000
                assert g["original_parent_interval_bp0"]==[0,2000000] and g["normalization_length_bp"]==2000000
                assert g["original_parent_query_bp"]==2000000*q
                cov=g["coverage_original_bp0"]
                assert g["physical_coverage_fraction"]==(cov[1]-cov[0])/2000000
                assert g["covered_span_without_retained_center_bp"]==cov[1]-cov[0]-span
                assert g["score_scope"]=="conditional on exact retained-center clipped-bin support; uncovered tails excluded"
                assert g["prediction_definition"]=="constant raw C4 probabilities over frozen physical 100-bp bins"
                assert g["truth_definition"]=="exact intersections with complete external truth intervals"
                assert g["probability_floor_used"] is False
                cm=g["confusion_truth_rows_prediction_columns_query_bp"]
                total=sum(map(sum,cm));diag=sum(cm[i][i] for i in range(4))
                assert total==g["denominator_query_bp"] and close(diag/total,g["accuracy"])
                per=[]
                for k,pc in enumerate(g["per_class"]):
                    true=sum(cm[k]);pred=sum(row[k] for row in cm);tp=cm[k][k]
                    precision=tp/pred if pred else None;recall=tp/true if true else None
                    f1=2*tp/(true+pred) if true else None
                    assert pc["true_query_bp"]==true and pc["predicted_query_bp"]==pred
                    assert close(precision,pc["precision"]) and close(recall,pc["recall"]) and close(f1,pc["f1"])
                    per.append((recall,f1))
                assert close(statistics.mean(x[0] for x in per if x[0] is not None),g["balanced_accuracy"])
                assert close(statistics.mean(x[1] for x in per if x[1] is not None),g["macro_f1"])
                b=g["boundaries"];nt,npred,m=b["true_boundaries"],b["predicted_boundaries"],b["matched"]
                assert b["missed"]==nt-m and b["extra"]==npred-m
                assert close(m/npred if npred else None,b["precision"])
                assert close(m/nt if nt else None,b["recall"])
                assert close(2*m/(nt+npred) if nt else None,b["f1"])
                calibration=g["calibration_reliability_bins"]
                assert sum(x["query_bp"] for x in calibration)==total and len(calibration)==15
                ece=sum(abs(x["accuracy"]-x["mean_confidence"])*x["query_bp"] for x in calibration if x["query_bp"])/total
                assert close(ece,g["calibration_ece"])
            for target,parts in (("all_queries_686",(PRIMARY,"secondary_ancient_504")),
                ("secondary_ancient_504",("age_stratum_Bronze_Age","age_stratum_BAA","age_stratum_Neo","age_stratum_Yam"))):
                assert close(sum(groups[s]["accuracy"]*groups[s]["query_count"] for s in parts)/groups[target]["query_count"],
                             groups[target]["accuracy"])
                for i in range(4):
                    for j in range(4):
                        assert sum(groups[s]["confusion_truth_rows_prediction_columns_query_bp"][i][j] for s in parts)==groups[target]["confusion_truth_rows_prediction_columns_query_bp"][i][j]
                for k in ("true_boundaries","predicted_boundaries","matched","missed","extra"):
                    assert sum(groups[s]["boundaries"][k] for s in parts)==groups[target]["boundaries"][k]
            max_fast_full_difference=max(max_fast_full_difference,abs(groups[PRIMARY]["accuracy"]-fast["groups"][PRIMARY]["accuracy"]))
            assert groups[PRIMARY]["represented_bin_support_bp"]==fast["groups"][PRIMARY]["represented_bin_support_bp"]
            assert groups[PRIMARY]["denominator_query_bp"]==fast["groups"][PRIMARY]["denominator_query_bp"]
        for name in expected_counts:
            gs=[c["groups"][name] for c in full["cells"]]
            means=full["equal_parent_metrics"][name]
            assert mean8f32([g["accuracy"] for g in gs])==means["accuracy"]
            assert mean8f32([g["nll"] for g in gs])==means["nll"]
            assert close(statistics.mean(g["brier"] for g in gs),means["brier"])
            assert close(statistics.mean(g["calibration_ece"] for g in gs),means["calibration_ece"])
            metrics[arm,name] = dict(accuracy=summary([g["accuracy"] for g in gs]),
                nll=summary([g["nll"] for g in gs]),brier=summary([g["brier"] for g in gs]),
                balanced_accuracy=summary([g["balanced_accuracy"] for g in gs]),
                macro_f1=summary([g["macro_f1"] for g in gs]),
                calibration_ece=summary([g["calibration_ece"] for g in gs]),
                boundary_f1=summary([g["boundaries"]["f1"] for g in gs]),
                boundary_precision=summary([g["boundaries"]["precision"] for g in gs]),
                boundary_recall=summary([g["boundaries"]["recall"] for g in gs]),
                boundary_counts_sum={k:sum(g["boundaries"][k] for g in gs) for k in ("true_boundaries","predicted_boundaries","matched","missed","extra")})
        check(arm+"_all_full_cohort_and_boundary_arithmetic_support",True)
        receipt=done["metric_receipts"][arm]
        check(arm+"_one_selected_full_call_no_target_model_or_loss", receipt["VAL_parents"]==8
              and receipt["full_selected_diagnostic_calls"]==1 and receipt["all_VAL_seals_before_truth"] is True
              and receipt["target_mass_model_input"] is False and receipt["target_mass_VAL_loss"] is False)

    h42comp=read("research/reviews/H042_completed_comparison_root_20261015_v1/summary.json")
    h42cohort=read("research/reviews/H042_completed_cohort_diagnostics_root_20261015_v1/summary.json")
    h42done=bound(h42comp["completion_ref"])
    assert h42cohort["completion_ref"]==h42comp["completion_ref"]
    h42fast={c["parent_id"]:c for c in h42done["selected"]["exact_clade_flags"]["cells"]}
    h42full={c["parent_id"]:c for c in h42done["selected_full_diagnostics"]["exact_clade_flags"]["cells"]}
    support_rows, per_parent = [], []
    for pid in PARENTS:
        original=h42comp["per_parent"][pid]
        assert original==h42fast[pid]["groups"][PRIMARY]
        seal=bound(h42fast[pid]["prediction_seal"])
        assert seal["parent_id"]==pid and seal["geometry"]==geometry[pid]
        row=dict(parent_id=pid,query_count=182,denominator_query_bp=original["denominator_query_bp"],
                 represented_bin_support_bp=original["represented_bin_support_bp"],
                 H042_H044_geometry_and_query_cohort_identity_equal=True)
        for arm in ARMS:
            fast=next(c for c in done["selected"][arm]["cells"] if c["parent_id"]==pid)["groups"][PRIMARY]
            g=full_by_arm[arm][pid]["groups"][PRIMARY]
            assert all(fast[k]==original[k] for k in ("query_count","denominator_query_bp","represented_bin_support_bp","score_arithmetic","probability_floor_used","posthoc_renormalization"))
            for name,old in h42full[pid]["groups"].items():
                current=full_by_arm[arm][pid]["groups"][name]
                assert all(current[k]==old[k] for k in ("query_count","denominator_query_bp","represented_bin_support_bp",
                    "coverage_original_bp0","represented_span_bp","original_parent_query_bp","score_scope"))
            per_parent.append(dict(arm=arm,parent_id=pid,accuracy=g["accuracy"],
                boundary_f1=g["boundaries"]["f1"],boundary_precision=g["boundaries"]["precision"],
                boundary_recall=g["boundaries"]["recall"],
                H044_minus_H042_accuracy_percentage_points=100*(g["accuracy"]-h42full[pid]["groups"][PRIMARY]["accuracy"])))
        support_rows.append(row)
    check("H042_H044_exact_parent_primary_denominator_support_and_sealed_geometry_match_all8",True)
    for name,row in h42cohort["cohorts"].items():
        x=[h42full[pid]["groups"][name]["accuracy"] for pid in PARENTS]
        assert close(statistics.mean(x),row["accuracy"]["mean"])
        assert close(statistics.stdev(x),row["accuracy"]["sample_SD"])
        assert close(statistics.mean(x)*100,h42comp["populations"][name]["accuracy_mean_percent"])
    check("H042_reference_cohort_summary_matches_written_full_cells",True)
    localized=[full_by_arm["localized"][p]["groups"][PRIMARY]["accuracy"] for p in PARENTS]
    pooled=[full_by_arm["tree_pooled"][p]["groups"][PRIMARY]["accuracy"] for p in PARENTS]
    paired=[100*(a-b) for a,b in zip(localized,pooled)]
    comparisons=dict(localized_minus_tree_pooled_primary_pp=summary(paired),
        positive_parents=sum(x>0 for x in paired),per_parent_percentage_points=dict(zip(PARENTS,paired)),
        localized_minus_tree_pooled_boundary_F1_pp=100*(metrics["localized",PRIMARY]["boundary_f1"]["mean"]-metrics["tree_pooled",PRIMARY]["boundary_f1"]["mean"]),
        H044_minus_H042_primary_mean_pp={a:100*(metrics[a,PRIMARY]["accuracy"]["mean"]-h42cohort["cohorts"][PRIMARY]["accuracy"]["mean"]) for a in ARMS},
        H042_comparison_is_descriptive_different_added_training_and_parameter_budget=True)
    report=dict(schema="H044_terminal_exit_and_metrics_independent_audit_v1",
        status="PRESERVED_DEVELOPMENT_OUTPUTS_VERIFIED_WITH_GENUINE_EXIT1_CAVEAT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),inputs=list(inputs.values()),
        actual_direct_launcher_returncode=1,actual_monitor_error=None,actual_owned_group_drained=True,
        wait0_claim=False,exit_cause="Saved result dictionary returned to SystemExit(main()); printed noninteger object exits1",
        log_printed_result_equals_completed=True,log_printed_result_AST_calls=calls,log_printed_result_characters=len(literal),
        log_result_first_line=first+1,log_result_last_line=len(lines)-1,
        all96_bindings_verified=True,all96_probability_bytes=total_bytes,actual_probability_arrays_rehashed=False,
        selected_passes={a:done["selected"][a]["pass_index"] for a in ARMS},
        full_diagnostic_summary_convention="float64 arithmetic means and sample SD over 8 independent VAL parent cells",
        source_selection_summary_convention="original float32 equal-parent means; exact standard-library IEEE754 pairwise8 reproduction",
        maximum_absolute_fast_vs_full_cell_accuracy_difference=max_fast_full_difference,
        history=history_summary,
        cohort_metrics={a:{name:metrics[a,name] for name in expected_counts} for a in ARMS},
        support_comparison_H042=support_rows,primary_per_parent=per_parent,comparisons=comparisons,
        checks=checks,check_count=len(checks),
        optimizer_seeds=[101],development_only=True,outer_truth_read=False,novelty_established=False,
        source_metadata_only=True,raw_datasets_checkpoints_targets_trees_models_read=False,
        project_guard_Torch_numpy_imported=False,scientific_native_GPU_process_accounting_admission_actions=False,
        limitations=["Genuine recorded exit1 remains a failed process exit; no wait0/credit/replay certificate",
          "Prediction arrays were RAM-only and are no longer rehashable by this audit; verify durable packets/refs and source gating only",
          "Metric arithmetic checked from written cells, not recomputed from raw probabilities/truth",
          "One seed, 8 VAL simulations used for selection and diagnostics; SD is parent SD, no outer test or generalization claim",
          "H044-vs-H042 support is identical, but H044 adds readout parameters and further training",
          "Localized gain is negligible with worse boundary F1; useful new marker-localization contribution is unsupported"])
    HERE.mkdir(exist_ok=True)
    (HERE/"audit_receipt.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    buf=io.StringIO();writer=csv.writer(buf)
    writer.writerow(["arm","cohort","n","accuracy_mean_percent","accuracy_sample_SD_pp","boundary_F1_n","boundary_F1_mean_percent","boundary_F1_sample_SD_pp","NLL_mean","NLL_sample_SD"])
    for arm in ARMS:
        for name in expected_counts:
            m=metrics[arm,name]
            writer.writerow([arm,name,m["accuracy"]["n"],100*m["accuracy"]["mean"],100*m["accuracy"]["sample_SD"],m["boundary_f1"]["n"],
                100*m["boundary_f1"]["mean"] if m["boundary_f1"]["mean"] is not None else None,
                100*m["boundary_f1"]["sample_SD"] if m["boundary_f1"]["sample_SD"] is not None else None,
                m["nll"]["mean"],m["nll"]["sample_SD"]])
    (HERE/"cohort_metrics.csv").write_text(buf.getvalue())
    print(json.dumps(dict(status=report["status"],checks=len(checks),
        primary={a:metrics[a,PRIMARY] for a in ARMS},comparisons=comparisons),indent=2))


if __name__=="__main__":
    main()

"""Summarize completed H042 metrics, with no inference or query-truth access."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, statistics
ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/H042_completed_cohort_diagnostics_root_20261015_v1"
COMPLETION = ROOT / "experiments/results/H042_GPU4_clade_flags_evaluate_seed101_memo_v1/completed.json"
NATIVE = ROOT / "research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics.csv"
ORDER = [("primary_present_day_182", "Present (182)"), ("age_stratum_Bronze_Age", "Bronze Age (162)"), ("age_stratum_BAA", "BAA (14)"), ("age_stratum_Neo", "Neo (302)"), ("age_stratum_Yam", "Yam (26)"), ("secondary_ancient_504", "Ancient (504)"), ("all_queries_686", "All (686)")]
def ref(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def stats(values):
    assert len(values) <= 8 and all(math.isfinite(x) for x in values)
    return dict(mean=statistics.mean(values) if values else None, sample_SD=statistics.stdev(values) if len(values)>1 else None, n=len(values))
def pct(value):
    return 100*value if value is not None else None
def formatted(mean, sd):
    return "undefined" if mean is None else ("%.4f ± %.4f" % (mean,sd) if sd is not None else "%.4f (one defined parent)" % mean)
def main():
    completed = json.loads(COMPLETION.read_text())
    assert completed["status"] == "completed_original_H038_selection_for_flag_control"
    assert completed["selected"]["exact_clade_flags"]["pass_index"] == 30
    cells = completed["selected_full_diagnostics"]["exact_clade_flags"]["cells"]
    assert sorted(c["parent_id"] for c in cells) == ["ap_expanded_val_%03d" % i for i in range(8)]
    result = dict(schema="H042_completed_cohort_diagnostics_summary_v1", updated_utc=datetime.now(timezone.utc).isoformat(), completion_ref=ref(COMPLETION), native_summary_ref=ref(NATIVE), boundary_tolerance_bp=1000, selected_additional_pass=30, optimizer_seeds=[101], SD_unit="independent validation simulation", development_only=True, new_inference=False, query_truth_opened=False, checkpoint_selection_changed=False, cohorts={})
    rows = []
    for group, label in ORDER:
        entries = [c["groups"][group] for c in cells]
        metrics = {key: stats([float(e[key]) for e in entries]) for key in ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier", "calibration_ece")}
        metrics["boundaries"] = {key: stats([float(e["boundaries"][key]) for e in entries if e["boundaries"][key] is not None]) for key in ("precision", "recall", "f1")}
        metrics["boundary_counts_sum"] = {key: sum(e["boundaries"][key] for e in entries) for key in ("matched", "missed", "extra", "true_boundaries", "predicted_boundaries")}
        pooled = [[sum(e["confusion_truth_rows_prediction_columns_query_bp"][i][j] for e in entries) for j in range(4)] for i in range(4)]
        metrics["pooled_confusion_query_bp_truth_rows_prediction_columns"] = pooled
        metrics["pooled_confusion_row_normalized"] = [[x/sum(row) if sum(row) else None for x in row] for row in pooled]
        metrics["per_parent"] = [{"parent_id": c["parent_id"], "accuracy": e["accuracy"], "boundary_precision": e["boundaries"]["precision"], "boundary_recall": e["boundaries"]["recall"], "boundary_f1": e["boundaries"]["f1"]} for c,e in zip(cells, entries)]
        result["cohorts"][group] = metrics
        rows.append(dict(model="H042_exact_clade_flags", cohort=group, queries=entries[0]["query_count"], accuracy_percent_mean=100*metrics["accuracy"]["mean"], accuracy_percent_sampleSD=100*metrics["accuracy"]["sample_SD"], boundary_f1_percent_mean=pct(metrics["boundaries"]["f1"]["mean"]), boundary_f1_percent_sampleSD=pct(metrics["boundaries"]["f1"]["sample_SD"]), nll_mean=metrics["nll"]["mean"], brier_mean=metrics["brier"]["mean"], ECE_mean=metrics["calibration_ece"]["mean"]))
    with (HERE/"summary.json").open("x") as f: json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    with (HERE/"cohort_metrics.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    native=list(csv.DictReader(NATIVE.open()))
    assert len(native) == 14
    models=list(dict.fromkeys(r["model"] for r in native))
    report=["# H042 completed population and boundary diagnostics", "", "Derived only from the already completed selected-pass30 evaluation. One optimizer seed101. Mean and sample SD are across eight validation simulations. No new predictions, query-truth access or selection changes.", "", "| Cohort | Accuracy, mean ± SD (%) | Boundary F1, mean ± SD (%) |", "|---|---:|---:|"]
    for row,(_,label) in zip(rows,ORDER):report.append("| %s | %s | %s |" % (label,formatted(row["accuracy_percent_mean"],row["accuracy_percent_sampleSD"]),formatted(row["boundary_f1_percent_mean"],row["boundary_f1_percent_sampleSD"])))
    p=result["cohorts"]["primary_present_day_182"]["boundaries"]
    report += ["", "Present182 boundary precision is %.4f ± %.4f%% and recall is %.4f ± %.4f%%, with 1 kb tolerance. Boundary detection remains weak despite high span accuracy." % (100*p["precision"]["mean"],100*p["precision"]["sample_SD"],100*p["recall"]["mean"],100*p["recall"]["sample_SD"]), "", "H042 has higher mean span accuracy than native AP and current native ARGMix in all seven reported cohorts. These are development-pipeline comparisons. Input features, training budgets and checkpoint selection differ. There is no original-paper superiority, independent-test confirmation or training-seed uncertainty claim.", "", "Boundary metrics remain undefined where no relevant boundaries exist. Their mean and SD use only parents with a defined value, with metric-specific counts in the JSON. Present182 uses all eight parents. The JSON also records per-parent boundary values and pooled span confusion matrices. Pooled confusion is weighted by query-base pairs, while table values give equal weight to each simulation.", "", "[Complete summary](summary.json) · [CSV](cohort_metrics.csv)"]
    (HERE/"README.md").write_text("\n".join(report)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.6))
    names={"native_AP":"Native AncestralPaths", "native_ARGMix":"Current native ARGMix", "H038_aligned":"H038 aligned", "H038_zero":"H038 zero", "H042_exact_clade_flags":"H042 local-clade flags"}
    # The native CSV's exact model names are retained in the output metadata.
    selected_models=[m for m in models if "zero" not in m.lower()]+["H042_exact_clade_flags"]
    series=native+rows
    for j,m in enumerate(selected_models):
        rr=[next(r for r in series if r["model"]==m and r["cohort"]==g) for g,_ in ORDER]
        y=np.asarray([float(r["accuracy_percent_mean"]) for r in rr]);sd=np.asarray([float(r["accuracy_percent_sampleSD"]) for r in rr]);x=np.arange(7)+(j-(len(selected_models)-1)/2)*0.17
        axes[0].errorbar(x,y,yerr=sd,fmt="o",capsize=2,markersize=4,label=names.get(m,m))
        r=rr[0];axes[1].errorbar(j,float(r["boundary_f1_percent_mean"]),yerr=float(r["boundary_f1_percent_sampleSD"]),fmt="o",capsize=3)
    axes[0].set_xticks(range(7),[label.split(" (")[0] for _,label in ORDER],rotation=30,ha="right");axes[0].set_ylabel("Span accuracy (%)");axes[0].set_ylim(90,100.7);axes[0].legend(loc="lower right",fontsize=8)
    axes[1].set_xticks(range(len(selected_models)),[names.get(m,m) for m in selected_models],rotation=25,ha="right");axes[1].set_ylabel("Present182 boundary F1 (%)");axes[1].set_ylim(bottom=0)
    for ax in axes:ax.grid(axis="y",alpha=0.25)
    fig.suptitle("Development simulations: mean ± sample SD across eight parents, one optimizer seed",fontsize=11);fig.tight_layout();fig.savefig(HERE/"accuracy_and_boundary_mean_SD.png",dpi=160);plt.close(fig)
    print(json.dumps(dict(cohort_count=len(rows),native_models=models,Present182=result["cohorts"]["primary_present_day_182"]["boundaries"],new_inference=False)))
if __name__=="__main__":main()

"""Reproduce three comparison figures using only pinned, completed metric JSONs.

Run on anogena from /workspace-SR003.nfs2/ARG-GNN:
  . scripts/project_env.sh
  "$ARGGNN_PYTHON" -B research/reviews/current_comparison_figures_20261015_v1/plot_comparisons.py
No model, checkpoint, prediction array, genotype, or truth payload is opened.
"""
from pathlib import Path
import hashlib
import json
import os
import math
import statistics
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/current_comparison_figures_20261015_v1"
os.environ["MPLCONFIGDIR"] = str(HERE / ".matplotlib")
os.environ["MPLBACKEND"] = "Agg"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CLASSES = ["ANA", "CHG", "WHG", "EHG"]
PRIMARY = "primary_present_day_182"
PARENTS = [f"ap_expanded_val_{i:03d}" for i in range(8)]
METHODS = ["native_ARGMix", "native_AP", "H045_foreign_episode"]
LABELS = {"native_ARGMix": "Native ARGMix", "native_AP": "AncestralPaths derived C4",
          "H045_foreign_episode": "H045 foreign_episode",
          "H047_original_risk": "H047 original risk", "H047_balanced_risk": "H047 balanced risk"}
SHORT = {"native_ARGMix": "ARGMix", "native_AP": "AP derived C4",
         "H045_foreign_episode": "H045 foreign", "H047_original_risk": "H047 original",
         "H047_balanced_risk": "H047 balanced"}
COLORS = {"native_ARGMix": "#166D9B", "native_AP": "#C47721", "H045_foreign_episode": "#25836A",
          "H047_original_risk": "#6862A5", "H047_balanced_risk": "#AF5381"}
COHORTS = [PRIMARY, "age_stratum_Bronze_Age", "age_stratum_BAA", "age_stratum_Neo",
           "age_stratum_Yam", "secondary_ancient_504", "all_queries_686"]
COHORT_LABELS = ["Present182", "BronzeAge162", "BAA14", "Neo302", "Yam26", "Ancient504", "All686"]
checks = []
reads = []

def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append(name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_ref(ref):
    rel = ref["path"]
    path = ROOT / rel
    check("read_stays_inside_repo:" + rel, path.resolve().is_relative_to(ROOT))
    check("no_symlink_input:" + rel, not path.is_symlink())
    check("only_metric_JSON:" + rel, path.suffix == ".json")
    raw = path.read_bytes()
    check("pinned_input:" + rel, hashlib.sha256(raw).hexdigest() == ref["sha256"])
    reads.append(ref)
    return json.loads(raw)

def stat(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "sample_SD": float(np.std(values, ddof=1)), "n": len(values)}

def save_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")

def save_figure(fig, stem):
    for ext in ("png", "svg", "pdf"):
        kwargs = {"dpi": 190} if ext == "png" else {}
        if ext == "pdf":
            kwargs["metadata"] = {"Title": stem, "Author": "ARG-GNN research",
                                  "Subject": "Existing completed development metric receipts only",
                                  "CreationDate": None, "ModDate": None}
        if ext == "svg":
            kwargs["metadata"] = {"Date": None}
        fig.savefig(HERE / (stem + "." + ext), facecolor="white", **kwargs)
    plt.close(fig)

def main():
    check("dedicated_remote_repository", Path.cwd().resolve() == ROOT)
    spec = json.loads((HERE / "evidence_spec.json").read_text())
    native = load_ref(spec["inputs"]["native_summary"])
    raw_native = load_ref(spec["inputs"]["native_per_parent"])
    native_done = load_ref(spec["inputs"]["native_completed"])
    h045_summary = load_ref(spec["inputs"]["H045_summary"])
    h045 = load_ref(spec["inputs"]["H045_completed"])
    h047 = load_ref(spec["inputs"]["H047_summary"])
    check("class_order_source_exact", native["class_order"] == ["Ana", "CHG", "WHG", "EHG"])
    check("native_original_completion", native_done["actual_computation_completed"] and
          native_done["all8_raw_seals_before_truth"] and native["all8_seals_verified_before_truth"])
    check("native_summary_binding", native_done["summary_ref"] == spec["inputs"]["native_summary"])
    check("native_no_new_predictions_or_fit", not native["new_predictions"] and not native["retraining"])
    check("H045_completion_binding", h045_summary["completion_ref"] == spec["inputs"]["H045_completed"])
    check("H045_completed_diagnostic_schema", h045["schema"] == "H045_episode_VAL_completion_v1" and
          h045["status"] == "completed_original_H038_selection")
    check("H045_sealed_original_development", h045["all96_RAM_seals_before_any_VAL_truth"] and
          h045["same_VAL_is_development_evidence"] and not h045["outer_truth_read"])
    check("H047_completed_root_schema", h047["schema"] == "H047_actual_completed_comparison_root_v1" and
          h047["actual_exec_exit_code"] == 0 and not h047["outer_test_accessed"])

    method_cells = dict(raw_native)
    method_cells["H045_foreign_episode"] = h045["selected_full_diagnostics"]["foreign_episode"]["cells"]
    by_method = {}
    support = {}
    for method in METHODS:
        cells = method_cells[method]
        ids = [c["parent_id"] for c in cells]
        check(method + ":exact_original_eight_VAL", sorted(ids) == PARENTS and len(set(ids)) == 8)
        cells_by_id = {c["parent_id"]: c for c in cells}
        matrices, perclass, accuracy, supports = [], {"precision": [], "recall": [], "f1": []}, [], []
        for pid in PARENTS:
            cell = cells_by_id[pid]
            g = cell["groups"][PRIMARY]
            C = np.asarray(g["confusion_truth_rows_prediction_columns_query_bp"], dtype=np.float64)
            check(method + ":" + pid + ":val_split", cell["split"].lower() == "val")
            check(method + ":" + pid + ":finite_integer_4x4_counts",
                  C.shape == (4, 4) and np.isfinite(C).all() and (C >= 0).all() and np.equal(C, np.floor(C)).all())
            true = C.sum(axis=1)
            predicted = C.sum(axis=0)
            check(method + ":" + pid + ":all_true_rows_available", (true > 0).all())
            check(method + ":" + pid + ":182_queries", g["query_count"] == 182)
            check(method + ":" + pid + ":recorded_denominator_exact", C.sum() == g["denominator_query_bp"])
            check(method + ":" + pid + ":recorded_accuracy_exact",
                  math.isclose(float(np.trace(C) / C.sum()), g["accuracy"], rel_tol=0, abs_tol=1e-14))
            values = {"precision": np.diag(C) / predicted, "recall": np.diag(C) / true,
                      "f1": 2 * np.diag(C) / (true + predicted)}
            for metric, v in values.items():
                check(method + ":" + pid + ":" + metric + "_matches_recorded_per_class",
                      np.allclose(v, [x[metric] for x in g["per_class"]], atol=1e-14, rtol=0))
                perclass[metric].append(v)
            matrices.append(C)
            accuracy.append(g["accuracy"])
            supports.append(dict(parent_id=pid, true_class_query_bp=true.tolist(),
                denominator_query_bp=g["denominator_query_bp"], query_count=g["query_count"],
                coverage_original_bp0=g["coverage_original_bp0"],
                original_parent_interval_bp0=g["original_parent_interval_bp0"],
                represented_fraction_original_parent=g["represented_fraction_original_parent"]))
        counts = np.asarray(matrices)
        # Critical operation order: normalize each parent's true row FIRST.
        normalized = counts / counts.sum(axis=2, keepdims=True)
        mean = normalized.mean(axis=0)
        sd = normalized.std(axis=0, ddof=1)
        pooled = counts.sum(axis=0)
        pooled_rows = pooled / pooled.sum(axis=1, keepdims=True)
        check(method + ":all_parent_rows_sum_one", np.allclose(normalized.sum(axis=2), 1, atol=1e-14))
        check(method + ":mean_rows_sum_one", np.allclose(mean.sum(axis=1), 1, atol=1e-14))
        # An independent stdlib calculation checks the aggregation implementation.
        independent_mean = [[statistics.mean(normalized[:, i, j].tolist()) for j in range(4)] for i in range(4)]
        independent_sd = [[statistics.stdev(normalized[:, i, j].tolist()) for j in range(4)] for i in range(4)]
        check(method + ":independent_mean_SD", np.allclose(mean, independent_mean, atol=1e-15) and
              np.allclose(sd, independent_sd, atol=1e-15))
        perclass_stats = {metric: {"mean": np.mean(v, axis=0).tolist(),
                                  "sample_SD": np.std(v, axis=0, ddof=1).tolist(),
                                  "per_parent": np.asarray(v).tolist()}
                          for metric, v in perclass.items()}
        if method.startswith("native_"):
            row = next(r for r in native["models"][method]["rows"] if r["cohort"] == PRIMARY)
            check(method + ":recorded_pooled_counts_exact", np.array_equal(pooled, row["pooled_confusion_query_bp"]))
            for metric in perclass:
                check(method + ":" + metric + "_aggregate_matches_recorded",
                      np.allclose(perclass_stats[metric]["mean"], [x[metric]["mean"] for x in row["per_class_mean_SD"]], atol=1e-14) and
                      np.allclose(perclass_stats[metric]["sample_SD"], [x[metric]["sample_sd"] for x in row["per_class_mean_SD"]], atol=1e-14))
            source_accuracy = row["accuracy"]
            source_sd = source_accuracy["sample_sd"]
        else:
            source_accuracy = h045_summary["arms"]["foreign_episode"]["groups"][PRIMARY]["accuracy"]
            source_sd = source_accuracy["sample_SD"]
        check(method + ":summary_accuracy_mean_SD_matches_counts",
              math.isclose(np.mean(accuracy), source_accuracy["mean"], rel_tol=0, abs_tol=1e-14) and
              math.isclose(np.std(accuracy, ddof=1), source_sd, rel_tol=0, abs_tol=1e-14))
        by_method[method] = {"label": LABELS[method], "parent_ids": PARENTS,
            "confusion_query_bp_per_parent": counts.tolist(),
            "row_normalized_fraction_per_parent": normalized.tolist(),
            "row_normalized_fraction_mean": mean.tolist(), "row_normalized_fraction_sample_SD": sd.tolist(),
            "per_class": perclass_stats, "accuracy": stat(accuracy),
            "largest_absolute_difference_from_pooled_normalization_pp": float(np.max(abs(mean - pooled_rows)) * 100)}
        support[method] = supports
    check("exact_shared_per_parent_true_class_and_query_bp_support",
          support[METHODS[0]] == support[METHODS[1]] == support[METHODS[2]])
    check("all_confusion_counts_available", len(by_method) == 3)

    population = {}
    for method in METHODS:
        population[method] = {}
        for cohort in COHORTS:
            values = [c["groups"][cohort]["accuracy"] for c in sorted(method_cells[method], key=lambda c:c["parent_id"])]
            population[method][cohort] = stat(values)
    for arm in ("original_risk", "balanced_risk"):
        method = "H047_" + arm
        population[method] = {}
        for cohort in COHORTS:
            v = h047["arms"][arm]["cohorts"][cohort]["accuracy"]
            check(method + ":" + cohort + ":eight_defined_parents", v["defined_parent_count"] == 8)
            population[method][cohort] = {"mean": v["mean"], "sample_SD": v["sample_SD"], "n": 8}

    common_metrics = {}
    for method in METHODS + ["H047_original_risk", "H047_balanced_risk"]:
        common_metrics[method] = {}
        for cohort in COHORTS:
            if method.startswith("native_"):
                source_row = next(r for r in native["models"][method]["rows"] if r["cohort"] == cohort)
                metric_source = {k: source_row[k] for k in
                    ("accuracy", "nll", "macro_f1", "boundary_f1", "brier", "calibration_ece")}
            elif method == "H045_foreign_episode":
                source_row = h045_summary["arms"]["foreign_episode"]["groups"][cohort]
                metric_source = {k: source_row["boundary_F1" if k == "boundary_f1" else k] for k in
                    ("accuracy", "nll", "macro_f1", "boundary_f1", "brier", "calibration_ece")}
            else:
                source_row = h047["arms"][method.removeprefix("H047_")]["cohorts"][cohort]
                metric_source = {k: source_row["boundaries"]["f1"] if k == "boundary_f1" else source_row[k] for k in
                    ("accuracy", "nll", "macro_f1", "boundary_f1", "brier", "calibration_ece")}
            normalized_stats = {}
            for metric, source_stat in metric_source.items():
                n = source_stat.get("defined_parent_count", source_stat.get("defined_parents", source_stat.get("n")))
                normalized_stats[metric] = {"mean": source_stat["mean"],
                    "sample_SD": source_stat.get("sample_SD", source_stat.get("sample_sd")),
                    "defined_parent_count": n, "undefined_parent_count": source_stat.get("undefined_parent_count", 8-n)}
                normalized_stats[metric]["source_stat"] = source_stat
                normalized_stats[metric]["state"] = "available" if source_stat["mean"] is not None else "unavailable_source_aggregate"
                if method.startswith("native_") and metric == "nll":
                    infinite_parents = sum(c["groups"][cohort]["infinite_nll_query_bp"] > 0
                                           for c in method_cells[method])
                    normalized_stats[metric]["recorded_infinite_parent_count"] = infinite_parents
                    if infinite_parents:
                        check(method + ":" + cohort + ":infinite_NLL_preserves_null_source_aggregate",
                              source_stat["mean"] is None and source_stat.get("sample_sd") is None)
                        normalized_stats[metric]["state"] = "infinite_NLL_sample_SD_undefined"
            check(method + ":" + cohort + ":table_accuracy_agrees_with_figure",
                  math.isclose(normalized_stats["accuracy"]["mean"], population[method][cohort]["mean"], rel_tol=0, abs_tol=1e-14) and
                  math.isclose(normalized_stats["accuracy"]["sample_SD"], population[method][cohort]["sample_SD"], rel_tol=0, abs_tol=1e-14))
            common_metrics[method][cohort] = normalized_stats
    save_json("cohort_scores.json", {"schema":"completed_common_cohort_scores_v1",
        "methods":common_metrics, "input_refs":reads,
        "units":{"accuracy":"percent after multiplying source fraction by100; SD in percentage points",
                 "macro_f1":"percent after multiplying source fraction by100; SD in percentage points",
                 "boundary_f1":"percent after multiplying source fraction by100; SD in percentage points",
                 "nll":"native negative log-likelihood units; SD in same units",
                 "brier":"source Brier units; SD in same units",
                 "calibration_ece":"source fraction units; SD in same units"},
        "SD_definition":"sample SD, ddof1, across defined independent VAL parents; one completed fit per method",
        "development_only":True, "significance_claim":False})
    def fmt_stat(v, percent=False):
        if v["state"] == "infinite_NLL_sample_SD_undefined":
            return f"infinite; SD undefined ({v['recorded_infinite_parent_count']}/8 parents)"
        if v["mean"] is None:
            return f"unavailable (n={v['defined_parent_count']}/8 defined)"
        factor=100 if percent else 1
        digits=4 if percent else 6
        mean=v["mean"]
        sd=v["sample_SD"]
        text=(f"{mean*factor:.{digits}f}" if isinstance(mean,(int,float)) else str(mean))
        text+=" +/- "+(f"{sd*factor:.{digits}f}" if isinstance(sd,(int,float)) else "unavailable")
        if v["defined_parent_count"]<8:
            text+=f" (n={v['defined_parent_count']})"
        return text
    methods=METHODS+["H047_original_risk","H047_balanced_risk"]
    lines=["# Completed European C4 cohort scores","",
        "The same original eight internal VAL parents, nominal2Mb,952input haplotypes and686queries. Present182 is primary. Native method inputs, objectives and selection/training budgets differ. These are descriptive development scores.",
        "", "Accuracy is percent mean +/- sample SD in percentage points. SD uses ddof1 across eight independent simulation parents, one completed fit per method, not optimizer-seed SD, SE or a confidence interval.","",
        "| Cohort | "+" | ".join(SHORT[m] for m in methods)+" |",
        "|---|"+"---:|"*5]
    for cohort,label in zip(COHORTS,COHORT_LABELS):
        lines.append("| "+label+" | "+" | ".join(fmt_stat(common_metrics[m][cohort]["accuracy"],True) for m in methods)+" |")
    lines+=["","Native ARGMix:11epochs, native early stopping. AncestralPaths:5epochs, native fixed schedule, recorded six-path probabilities collapsed to C4. H045foreign_episode and both H047 arms: H042warmstart +30passes/960updates per arm, seed101. H045 is the current selected development fit, not an independently confirmed best architecture.","",
        "## Other common recorded metrics","",
        "Rates below are percent with SD in percentage points. NLL, Brier and ECE retain original source units. n below8 identifies undefined-parent exclusions in the original aggregate. Native AP NLL is marked infinite when recorded infinite_nll_query_bp is positive in any parent; its sample SD is undefined. A null published aggregate without such an infinite-NLL witness stays unavailable.","",
        "| Cohort | Setup | NLL +/- SD | Macro F1 % +/- SD | Boundary F1 % +/- SD | Brier +/- SD | ECE +/- SD |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for cohort,label in zip(COHORTS,COHORT_LABELS):
        for method in methods:
            v=common_metrics[method][cohort]
            lines.append("| "+label+" | "+SHORT[method]+" | "+" | ".join([
                fmt_stat(v["nll"]),fmt_stat(v["macro_f1"],True),fmt_stat(v["boundary_f1"],True),
                fmt_stat(v["brier"]),fmt_stat(v["calibration_ece"])])+" |")
    lines+=["","BronzeAge is not assumed identical to the paper's Post-Neolithic cohort. No significance, causal demographic mechanism, architecture attribution or original-paper reproduction claim is made. All common metrics and exact source hashes are in cohort_scores.json. Confusion figures use per-parent row normalization before the mean, never pooled-row normalization.","","## Provenance",""]
    lines += ["- "+r["path"]+"; SHA256 "+r["sha256"] for r in reads]
    (HERE/"cohort_score_table.md").write_text("\n".join(lines)+"\n")

    evidence = {"schema": "recorded_C4_confusion_comparison_v1", "class_order": CLASSES,
        "scope": "European C4 original eight internal VAL parents; Present182; shared recorded physical query-bp support",
        "normalization": "C_parent[true,pred]/sum_pred(C_parent[true,pred]); then unweighted mean across8parents",
        "uncertainty": "sample SD with ddof1 across8independent simulation parents, not SE, CI or optimizer-seed SD",
        "confusion_units": "fractions in JSON; plots show percent mean and SD in percentage points",
        "per_class_formulas": {"recall": "C_ii/true_row_i", "precision": "C_ii/predicted_column_i",
                               "f1": "2*C_ii/(true_row_i+predicted_column_i)"},
        "support": support[METHODS[0]], "methods": by_method, "population_accuracy": population,
        "input_refs": reads, "unavailable": [],
        "caveats": spec["caveats"], "checks_passed": len(checks), "checks": checks}
    save_json("measured_evidence.json", evidence)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False, "savefig.dpi": 190,
        "svg.hashsalt": "EuropeanC4RecordedConfusion20261015"})

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.1))
    fig.subplots_adjust(left=.065, right=.91, top=.76, bottom=.23, wspace=.23)
    fig.suptitle("Present182 ancestry confusion on shared recorded support", x=.49, y=.96, fontsize=17, weight="bold")
    fig.text(.49, .895, "Normalize true-class rows within each VAL parent, then average eight parents", ha="center", fontsize=11)
    fig.text(.49, .85, "Cell labels: mean %  |  +/- sample SD in percentage points", ha="center", fontsize=10, color="#49525B")
    for ax, method in zip(axes, METHODS):
        m = np.asarray(by_method[method]["row_normalized_fraction_mean"]) * 100
        s = np.asarray(by_method[method]["row_normalized_fraction_sample_SD"]) * 100
        image = ax.imshow(m, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        ax.set_title(LABELS[method], pad=12, fontweight="bold")
        ax.set_xticks(range(4), CLASSES)
        ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("Predicted ancestry", labelpad=8)
        ax.set_ylabel("True ancestry" if method == METHODS[0] else "")
        for i in range(4):
            for j in range(4):
                color = "white" if m[i, j] > 65 else "#152D3A"
                ax.text(j, i-.09, f"{m[i,j]:.2f}", ha="center", va="center", color=color, weight="bold", fontsize=11)
                ax.text(j, i+.16, f"+/- {s[i,j]:.2f}", ha="center", va="center", color=color, fontsize=8.8)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.935,.268,.013,.45])
    fig.colorbar(image, cax=cax, label="Mean conditional assignment (%)")
    fig.text(.065, .116, "European C4 | 32 TRAIN / 8 VAL | 2 Mb nominal | 182 present-day queries per parent", fontsize=10, weight="bold")
    fig.text(.065, .077, "Common recorded support verified exactly across methods. Inputs, objectives and training budgets differ.", fontsize=10)
    fig.text(.065, .04, "Development evidence; no architecture attribution or original-paper reproduction claim. SD is across parents, not optimizer seeds.", fontsize=9, color="#49525B")
    save_figure(fig, "present182_confusion_mean_sd")

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.8), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, top=.75, bottom=.27, wspace=.10)
    fig.suptitle("Present182 per-class ancestry diagnostics", y=.97, fontsize=17, weight="bold")
    fig.text(.5,.9,"Metrics calculated separately from each parent's exact query-bp confusion counts",ha="center",fontsize=11)
    x = np.arange(4)
    offsets = [-.20, 0, .20]
    min_lower = 100
    max_upper = 0
    for ax, metric, title in zip(axes, ["recall","precision","f1"], ["Recall: true ancestry recovered","Precision: predicted ancestry correct","F1: precision-recall balance"]):
        for method, off in zip(METHODS, offsets):
            p = by_method[method]["per_class"][metric]
            mean = np.array(p["mean"])*100
            sd = np.array(p["sample_SD"])*100
            vals = np.array(p["per_parent"])*100
            min_lower=min(min_lower,float((mean-sd).min()),float(vals.min()))
            max_upper=max(max_upper,float((mean+sd).max()),float(vals.max()))
            for j in range(8):
                ax.scatter(x+off, vals[j], s=11, color=COLORS[method], alpha=.20, zorder=2)
            ax.errorbar(x+off,mean,yerr=sd,fmt="o",color=COLORS[method],markersize=6,
                        linewidth=1.7,capsize=4,label=LABELS[method],zorder=3)
        ax.set_xticks(x,CLASSES)
        ax.set_title(title,fontsize=11,pad=10,fontweight="bold")
        ax.set_xlim(-.5,3.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E0E5EA", linewidth=.8)
        ax.set_xlabel("Ancestry class")
    axes[0].set_ylabel("Metric (%)")
    axes[0].set_ylim(math.floor(min_lower/2)*2-1,math.ceil(max_upper/2)*2+1)
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.145),ncol=3,frameon=False)
    fig.text(.065,.092,"Dots: eight independent VAL parents. Large markers: equal-parent mean. Error bars: sample SD, ddof1.",fontsize=10)
    fig.text(.065,.052,"One completed fit per method. Native ARGMix: 11 epochs; AP: 5 epochs; H045: H042 warmstart +30 passes /960 updates, seed101.",fontsize=9)
    fig.text(.065,.018,"Same present-day query-bp support; different inputs, objectives and selection budgets. No independent TEST or architecture-level attribution.",fontsize=9,color="#49525B")
    save_figure(fig,"present182_per_class_mean_sd")

    pop_methods = METHODS + ["H047_original_risk","H047_balanced_risk"]
    fig, axes = plt.subplots(2,4,figsize=(14.8,8.1))
    fig.subplots_adjust(left=.105,right=.985,top=.80,bottom=.17,wspace=.22,hspace=.44)
    fig.suptitle("Population accuracy across the shared development panel",y=.97,fontsize=17,weight="bold")
    fig.text(.5,.92,"Equal-parent means +/- sample SD across eight VAL simulations; one completed fit per method",ha="center",fontsize=11)
    fig.text(.5,.87,"H047 matched original-risk and balanced-risk arms are shown separately",ha="center",fontsize=10,color="#49525B")
    for idx,(cohort,label) in enumerate(zip(COHORTS,COHORT_LABELS)):
        ax=axes.flat[idx]
        lows=[];highs=[]
        for y,method in enumerate(pop_methods):
            v=population[method][cohort]
            mean=v["mean"]*100;sd=v["sample_SD"]*100
            ax.errorbar(mean,y,xerr=sd,fmt="o",color=COLORS[method],markersize=5,capsize=3,linewidth=1.4)
            lows.append(mean-sd);highs.append(mean+sd)
        ax.set_ylim(4.6,-.6)
        ax.set_yticks(range(5),[SHORT[m] for m in pop_methods] if idx%4==0 else [])
        ax.set_xlim(math.floor((min(lows)-.15)*2)/2,min(101,math.ceil((max(highs)+.15)*2)/2))
        ax.set_title(label,weight="bold",fontsize=12,pad=8)
        ax.set_xlabel("Accuracy (%)")
        ax.set_axisbelow(True)
        ax.grid(axis="x",color="#E0E5EA",linewidth=.8)
        ax.tick_params(axis="x",labelsize=9)
    axes.flat[7].axis("off")
    legend = [Line2D([0],[0],marker="o",color=COLORS[m],linestyle="",label=LABELS[m],markersize=6) for m in pop_methods]
    axes.flat[7].legend(handles=legend,loc="center left",frameon=False,fontsize=9.5,labelspacing=1.3)
    fig.text(.105,.108,"Each panel uses its own horizontal range. SD bars describe parent variation, not confidence intervals or optimizer-seed variation.",fontsize=10)
    fig.text(.105,.068,"Present182 is primary; the other cohorts are descriptive. BronzeAge is not assumed identical to the paper's Post-Neolithic cohort.",fontsize=9)
    fig.text(.105,.03,"Native objectives/budgets are retained. H045 and both H047 arms use H042 warmstarts +30 passes /960 updates per arm, seed101.",fontsize=9)
    save_figure(fig,"population_accuracy_mean_sd")

    directed = {}
    for method in METHODS:
        m=np.asarray(by_method[method]["row_normalized_fraction_mean"])
        s=np.asarray(by_method[method]["row_normalized_fraction_sample_SD"])
        ordered=sorted([(float(m[i,j]),i,j) for i in range(4) for j in range(4) if i!=j],reverse=True)
        directed[method]=[{"true_class":CLASSES[i],"predicted_class":CLASSES[j],
                          "mean_percent":v*100,"sample_SD_pp":float(s[i,j])*100}
                         for v,i,j in ordered[:3]]
    save_json("error_pattern_summary.json",{"schema":"recorded_error_pattern_summary_v1",
        "top_directed_confusions":directed,
        "class_recall_mean_percent":{m:(np.diag(by_method[m]["row_normalized_fraction_mean"])*100).tolist() for m in METHODS},
        "interpretation_limit":"Descriptive errors on shared internal VAL support; no causal demographic mechanism, significance, architecture attribution or original-paper reproduction claim."})
    output_files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in (".png",".svg",".pdf")]
    check("nine_expected_figure_exports",len(output_files)==9)
    total=sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    check("all_outputs_under_10MB",total<10_000_000)
    save_json("verification.json",{"status":"PASS","checks_passed":len(checks),"checks":checks,
        "figure_exports":[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output_files)],
        "measured_evidence_sha256":sha(HERE/"measured_evidence.json"),
        "source_sha256":sha(Path(__file__)),"bytes_at_generation":total,
        "software":{"python":sys.version,"numpy":np.__version__,"matplotlib":matplotlib.__version__},
        "project_input_refs":reads,"unavailable":[]})
    print(json.dumps({"status":"PASS","checks":len(checks),"figure_exports":len(output_files),"bytes":total}))

if __name__ == "__main__":
    main()

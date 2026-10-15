"""Bounded actual-metadata regression checks for H047 ledger logging.

No tensors, predictions, truth, checkpoints, GPU or storage-state reads.
The old/new generators first run with all output writes captured in memory.
"""
from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from unittest.mock import patch
import csv
import hashlib
import io
import json

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OWNED = ROOT / "research/reviews/H047_ledger_logging_source_v1"
SOURCE = ROOT / "scripts/update_research_ledger.py"
SUMMARY = ROOT / "research/reviews/H047_completed_comparison_root_20261015_v1/summary.json"
COMPLETED = ROOT / "experiments/results/H047_GPU4_cohort_risk_evaluate_seed101_v1/completed.json"
checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def load_namespace(source):
    ns = {"__name__": "ledger_source_check"}
    exec(compile(source, str(SOURCE), "exec"), ns)
    return ns

def capture(ns):
    outputs = {rel: (ROOT / rel).read_text() for rel in (
        "research/experiment_ledger.json", "research/experiment_results.csv",
        "research/experiment_ledger.md")}
    def write(path, content, *args, **kwargs):
        outputs[str(path.relative_to(ROOT))] = content
        return len(content)
    out = io.StringIO()
    with patch.object(Path, "write_text", write), redirect_stdout(out):
        ns["main"]()
    return outputs, json.loads(out.getvalue())

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))

before_source = (OWNED / "update_research_ledger.before.py").read_text()
after_source = SOURCE.read_text()
old = load_namespace(before_source)
new = load_namespace(after_source)
summary = json.loads(SUMMARY.read_text())
completed = json.loads(COMPLETED.read_text())
records, rows = new["h047_comparison_records"]()
check("exact_two_measured_arm_records", len(records) == len(rows) == 2)
check("both_arms_distinct", {r["method"] for r in records} == {"balanced_risk", "original_risk"})
check("arm_record_identities_unique", len({(r["study"], r["artifact"], r["method"]) for r in records}) == 2)
check("source_summary_hash_exact", all(r["sha256"] == hashlib.sha256(SUMMARY.read_bytes()).hexdigest() for r in records))
check("source_completion_hash_exact", all(r["context_receipts"] == [summary["actual_source_ref"]] for r in records))
check("only_performance_records", all(r["kind"] == "reported_model_result" for r in records))

for record, row in zip(records, rows):
    arm = record["method"]
    expected = {}
    for cohort, values in summary["arms"][arm]["cohorts"].items():
        for metric, stat in values.items():
            if metric == "boundaries":
                cells = [("boundary_" + name, cell) for name, cell in stat.items()]
            elif metric == "per_class":
                cells = [("class_" + str(item["class_0based"]) + "." + name, cell)
                         for item in stat for name, cell in item.items() if name != "class_0based"]
            else:
                cells = [(metric, stat)]
            for name, cell in cells:
                for k in ("mean", "sample_SD", "defined_parent_count", "undefined_parent_count"):
                    expected[cohort + "." + name + "." + k] = cell[k]
    actual = {k: v for k, v in record["metrics"].items() if "." in k}
    check(arm + "_all_cohort_mean_sd_support_exact", actual == expected)
    check(arm + "_all_seven_cohorts", len({k.split(".")[0] for k in actual}) == 7)
    check(arm + "_scalar_only", all(v is None or type(v) in (int, float) for v in record["metrics"].values()))
    primary = summary["arms"][arm]["cohorts"]["primary_present_day_182"]
    check(arm + "_primary_mean_sd_source_units", [
        record["metrics"]["mean_accuracy"], record["metrics"]["mean_accuracy_sample_SD"],
        record["metrics"]["validation_nll"], record["metrics"]["validation_nll_sample_SD"]
    ] == [primary["accuracy"]["mean"], primary["accuracy"]["sample_SD"],
          primary["nll"]["mean"], primary["nll"]["sample_SD"]])
    check(arm + "_csv_cohort_stats_exact", json.loads(row["cohort_metrics_json"]) == expected)
    check(arm + "_csv_primary_SD_exact", row["mean_accuracy_sample_SD"] == primary["accuracy"]["sample_SD"])
    check(arm + "_undefined_boundary_support_preserved",
          actual["age_stratum_Yam.boundary_f1.defined_parent_count"] == 7 and
          actual["age_stratum_Yam.boundary_f1.undefined_parent_count"] == 1)
    check(arm + "_checkpoint_metadata_only_binding",
          record["bindings"]["checkpoint_ref"] == completed["selected"][arm]["checkpoint"])
    check(arm + "_development_scope_explicit",
          record["bindings"]["same_VAL_is_development_evidence"] is True and
          "no outer TEST" in record["bindings"]["evaluation_split"])

original_reader = new["read_json"]
for name, mutate in [
    ("wrong_schema_rejected", lambda x: x.update(schema="unrelated_engineering_schema")),
    ("missing_arm_rejected", lambda x: x["arms"].pop("original_risk")),
    ("changed_completion_hash_rejected", lambda x: x["actual_source_ref"].update(sha256="0"*64)),
    ("changed_checkpoint_binding_rejected", lambda x: x["arms"]["balanced_risk"]["checkpoint_ref"].update(sha256="0"*64)),
]:
    bad = deepcopy(summary)
    mutate(bad)
    def reader(path, bad=bad):
        return (bad, "mock_summary_digest") if path == SUMMARY else original_reader(path)
    new["read_json"] = reader
    try:
        new["h047_comparison_records"]()
    except ValueError:
        checks.append(name)
    else:
        raise AssertionError(name)
new["read_json"] = original_reader

old_outputs, old_report = capture(old)
new_outputs, new_report = capture(new)
old_inventory = json.loads(old_outputs["research/experiment_ledger.json"])["result_inventory"]
new_inventory = json.loads(new_outputs["research/experiment_ledger.json"])["result_inventory"]
check("old_inventory_preserved_exactly", new_inventory[:-2] == old_inventory)
check("only_two_measured_inventory_records_added", new_inventory[-2:] == records)
old_csv = list(csv.DictReader(io.StringIO(old_outputs["research/experiment_results.csv"])))
new_csv = list(csv.DictReader(io.StringIO(new_outputs["research/experiment_results.csv"])))
extra = {"mean_accuracy_sample_SD", "validation_nll_sample_SD", "cohort_metrics_json"}
check("all_existing_CSV_rows_preserved", [{k:v for k,v in r.items() if k not in extra} for r in new_csv[:-2]] == old_csv)
check("two_distinct_CSV_arms_added", {r["method"] for r in new_csv[-2:]} == {"balanced_risk", "original_risk"})
check("existing_rows_have_no_fabricated_SD", all(not any(r[k] for k in extra) for r in new_csv[:-2]))
check("performance_count_adds_exact_two", new_report["performance_result_count"] == old_report["performance_result_count"] + 2)
check("scalar_count_adds_exact_two", new_report["scalar_result_record_count"] == old_report["scalar_result_record_count"] + 2)
check("indexed_count_adds_exact_two", new_report["indexed_receipt_count"] == old_report["indexed_receipt_count"] + 2)
old_keys = Counter((r.get("study"), r.get("artifact"), r.get("method")) for r in old_inventory)
new_keys = Counter((r.get("study"), r.get("artifact"), r.get("method")) for r in new_inventory)
check("no_existing_record_duplicated", all(new_keys[k] == v for k,v in old_keys.items()))
check("original_completion_not_double_counted_as_performance",
      all(r["kind"] == "engineering_receipt" for r in new_inventory
          if r.get("artifact") == str(COMPLETED.relative_to(ROOT))))
second_outputs, second_report = capture(new)
check("generator_idempotent", second_outputs == new_outputs)
check("second_run_counts_identical", all(second_report[k] == new_report[k] for k in
      ("performance_result_count", "scalar_result_record_count", "indexed_receipt_count")))

report = {
    "schema": "H047_ledger_logging_source_checks_v1", "status": "PASS",
    "check_count": len(checks), "checks": checks,
    "before_generator_report": old_report, "after_generator_report": new_report,
    "new_arm_metric_scalar_counts": {r["method"]: len(r["metrics"]) for r in records},
    "actual_summary_ref": {"path": str(SUMMARY.relative_to(ROOT)),
                           "sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest()},
    "actual_completion_ref": summary["actual_source_ref"],
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "scope": "actual metadata and mocked malformed-summary/write tests only; no live resource actions"
}
(OWNED / "source_checks_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": report["status"], "check_count": len(checks),
                  "before": old_report, "after": new_report,
                  "new_arm_metric_scalar_counts": report["new_arm_metric_scalar_counts"]}))

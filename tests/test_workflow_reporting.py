"""Repository-local temporary reports and mocked Git; no scientific jobs."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from arggnn.benchmarking import workflow_reporting as reporting


class WorkflowReportingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=reporting.c.ROOT / ".runtime")
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(reporting.c, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temporary.cleanup)
        self.now = 1000000.
        self.clock = patch.object(reporting.time, "time", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.git_calls = []
        self.git = patch.object(reporting, "_git", side_effect=self.mock_git)
        self.git.start()
        self.addCleanup(self.git.stop)
        self.original = "# Research\n\n## Existing scores\nKeep these exact old values.\n"
        (self.root / "README.md").write_text(self.original)
        self.tasks = [dict(id=name, resource=resource, factory_module="arggnn.benchmarking.example",
                           factory_function=stage + "_factory", factory_arguments=dict(stage=stage))
                      for name, resource, stage in (("prepare", "cpu", "prepare"),
                                                    ("fit", "gpu0", "fit"),
                                                    ("benchmark_score", "cpu", "score"))]
        self.plan_ref = self.write("plan.json", dict(schema="arggnn_executable_workflow_v1",
                                 physical_gpu=0, tasks=self.tasks))
        self.state = dict(plan_ref=self.plan_ref, tasks={task["id"]: dict(status="pending") for task in self.tasks},
                          updated_utc="poll one", complete=False, run_status="running")
        self.state_path = "state.json"
        self.save_state()

    def mock_git(self, *args):
        self.git_calls.append(args)
        return subprocess.CompletedProcess(args, 1 if args[:3] == ("diff", "--cached", "--quiet") else 0, "", "")

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
        return reporting.c.file_ref(target)

    def save_state(self):
        self.write(self.state_path, self.state)

    def read(self, path):
        return json.loads((self.root / path).read_text())

    def publish(self, **kwargs):
        self.save_state()
        return reporting.publish(self.plan_ref, self.state_path, **kwargs)

    def score(self, summary):
        source_ref = self.write("scientific/summary.json", summary)
        result_ref = self.write("scientific/result.json", dict(value=source_ref))
        self.state["tasks"]["benchmark_score"].update(status="succeeded", result_ref=result_ref)
        return source_ref

    def aggregate(self, mean=.7, unit="fraction"):
        return dict(mean=mean, sample_SD=.02, parents=5, unit=unit)

    def american(self):
        return dict(schema=reporting.AMERICAN, status="completed_shared15parent_paired_scores",
                    all_predictions_sealed_before_any_heldout_truth=True,
                    parents=15, validation_parents=5, test_parents=10,
                    records_per_parent=10000, records_per_method=150000,
                    probabilities_modified=False, probability_floor=False,
                    aggregation="equal_independent_parent_mean_and_sample_SD_ddof1",
                    native_one_hot_transports_excluded_from_NLL_Brier=True, exact_paper_reproduction=False,
                    equal_parent_summaries={"test": {
                        "treeloom": {"accuracy": self.aggregate(.81), "NLL": dict(
                            mean=None, sample_SD=None, parents=10, unit="nats_per_focal", status="positive_infinity")},
                        "argformer": {"accuracy": self.aggregate(.75)}}},
                    paired_deltas={"test": {"treeloom_minus_argformer": {"accuracy": dict(
                        self.aggregate(.06), direction="positive_favors_first_method", per_parent={"secret": .06})}}},
                    truth_provenance={"private": "NEVER_COPY_TRUTH"}, per_parent={"private": ["NEVER_COPY_ARRAY"]})

    def test_counts_stages_running_names_preserve_readme_and_limit_git_paths(self):
        self.state["tasks"]["prepare"]["status"] = "succeeded"
        self.state["tasks"]["fit"].update(status="running", pid=123, started_at=1)
        result = self.publish()
        self.assertEqual(result["status"], "published")
        report = self.read(reporting.STATUS_PATH)
        self.assertEqual(report["counts"], dict(succeeded=1, failed=0, running=1, pending=1, blocked=0, unknown=0))
        self.assertEqual(report["running_job_names"], ["fit"])
        self.assertEqual(report["stages"]["fit"]["running"], 1)
        readme = (self.root / "README.md").read_text()
        self.assertTrue(readme.startswith(self.original))
        commit = next(args for args in self.git_calls if args[0] == "commit")
        self.assertIn("--only", commit)
        self.assertEqual(commit[commit.index("--") + 1:], reporting.REPORT_PATHS)
        self.assertEqual(self.git_calls[-1], ("push", "origin", "HEAD"))

    def test_poll_only_changes_do_not_rewrite_or_commit(self):
        self.publish()
        old = (self.root / reporting.STATUS_PATH).read_bytes()
        self.git_calls.clear()
        self.now += 1900
        self.state["updated_utc"] = "poll two"
        result = self.publish()
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(old, (self.root / reporting.STATUS_PATH).read_bytes())
        self.assertEqual(self.git_calls, [])

    def test_routine_throttle_then_new_failure_bypasses_once(self):
        self.publish()
        self.git_calls.clear()
        self.now += 10
        self.state["tasks"]["fit"]["status"] = "running"
        self.assertEqual(self.publish()["status"], "updated")
        self.assertEqual(self.git_calls, [])
        self.state["tasks"]["fit"].update(status="failed", error="genuine failure")
        self.assertEqual(self.publish()["status"], "published")
        self.git_calls.clear()
        self.assertEqual(self.publish()["status"], "unchanged")
        self.assertEqual(self.git_calls, [])
        self.now += 1801
        self.state["tasks"]["prepare"]["status"] = "succeeded"
        self.assertEqual(self.publish()["status"], "published")

    def test_american_exact_aggregates_no_truth_or_per_parent_and_score_bypass(self):
        self.publish()
        self.git_calls.clear()
        self.now += 1
        summary = self.american()
        source_ref = self.score(summary)
        self.assertEqual(self.publish()["status"], "published")
        score = self.read(reporting.SCORE_PATH)["results"]["benchmark_score"]
        self.assertEqual(score["source_summary_ref"], source_ref)
        self.assertEqual(score["equal_parent_summaries"], summary["equal_parent_summaries"])
        self.assertNotIn("per_parent", score["paired_deltas"]["test"]["treeloom_minus_argformer"]["accuracy"])
        serialized = json.dumps(score)
        self.assertNotIn("NEVER_COPY", serialized)
        self.assertNotIn("truth_provenance", serialized)
        self.assertIn("positive infinity", (self.root / "README.md").read_text())
        self.assertNotIn("NLL", score["equal_parent_summaries"]["test"]["argformer"])

    def test_native_six_exact_overall_and_cohort_aggregates(self):
        summary = dict(schema=reporting.EUROPEAN, all_method_all_parent_prediction_barrier_before_truth=True,
                       records_per_parent=10000, comparison_scope="released Dense with common sampling",
                       published_CNN_reproduction=False,
                       summary={method: {group: {"accuracy": dict(mean=.8, sample_SD=.03,
                                independent_parents=5, unit="fraction")} for group in ("overall", "cohort")}
                                for method in ("ancestralpaths", "treeloom")},
                       paired_accuracy_differences={"overall": dict(mean=.01, sample_SD=.02,
                                unit="fraction", direction="TreeLoom minus AncestralPaths", per_parent=[1, 2, 3])},
                       rows={"private": ["NEVER_COPY"]}, test_targets={"private": "NEVER_COPY"})
        self.score(summary)
        self.assertEqual(self.publish()["status"], "published")
        compact = self.read(reporting.SCORE_PATH)["results"]["benchmark_score"]
        self.assertEqual(compact["summary"], summary["summary"])
        self.assertNotIn("NEVER_COPY", json.dumps(compact))
        self.assertNotIn("per_parent", compact["paired_accuracy_differences"]["overall"])

    def c4(self):
        return dict(schema=reporting.C4,
                    all_registered_endpoint_predictions_before_any_TEST_truth=True,
                    source_class_order=["Ana", "CHG", "WHG", "EHG"], records_per_parent=10000,
                    uncertainty="equal-parent mean and sample SD, ddof1",
                    separate_genuine_four_head_TreeLoom_training=True,
                    published_ARGMix_paper_reproduction=False,
                    scope="available release on common AP200Mb data, separate from native-six AP benchmark",
                    summary={method: {group: {metric: dict(mean=.8, sample_SD=.03,
                             independent_parents=5) for metric in ("accuracy", "macro_F1", "Brier", "raw_NLL")}
                             for group in ("overall", "cohort")} for method in ("argmix", "treeloom")},
                    paired_accuracy_differences={"overall": dict(mean=.01, sample_SD=.02,
                             direction="TreeLoom minus ARGMix", per_parent=["NEVER_COPY_PER_PARENT"])},
                    rows={"private": "NEVER_COPY_ROWS"},
                    test_targets=dict(path="private/forbidden_truth.json", sha256="0" * 64),
                    predictions=dict(path="private/forbidden_arrays.json", sha256="0" * 64),
                    native6_predictions=dict(path="private/forbidden_native6.json", sha256="0" * 64))

    def test_c4_exact_aggregates_units_and_no_private_reference_reads(self):
        self.publish()
        self.now += 1
        source = self.c4()
        # A genuine zero posterior is represented as unavailable in the exact summary.
        source["summary"]["argmix"]["overall"]["raw_NLL"].update(mean=None, sample_SD=None)
        summary_ref = self.score(source)
        allowed = {self.plan_ref["path"], "scientific/result.json", summary_ref["path"]}
        original_read = reporting.c.read_ref
        observed = []
        def metadata_only(ref):
            self.assertIn(ref["path"], allowed, "Reporter attempted a private/per-parent reference read")
            observed.append(ref["path"])
            return original_read(ref)
        with patch.object(reporting.c, "read_ref", side_effect=metadata_only):
            self.assertEqual(self.publish()["status"], "published")
        score = self.read(reporting.SCORE_PATH)["results"]["benchmark_score"]
        self.assertTrue(score["metrics_reported"])
        self.assertEqual(score["summary"], source["summary"])
        self.assertEqual(score["source_summary_ref"], summary_ref)
        self.assertNotIn("unit", score["summary"]["treeloom"]["overall"]["accuracy"])
        self.assertEqual(score["metric_units"]["accuracy"], "fraction")
        self.assertEqual(score["metric_units"]["raw_NLL"], "nats_per_record")
        self.assertNotIn("per_parent", score["paired_accuracy_differences"]["overall"])
        self.assertNotIn("NEVER_COPY", json.dumps(score))
        self.assertNotIn("private/", json.dumps(score))
        self.assertEqual(set(observed), allowed)
        readme = (self.root / "README.md").read_text()
        self.assertIn("European C4 matched evaluation", readme)
        self.assertIn("Published ARGMix paper reproduction: false", readme)
        self.assertIn("nats_per_record", readme)
        self.assertIn("argmix | 5 |", readme)

    def test_c4_requires_cross_endpoint_barrier_and_five_parents(self):
        for broken in ("barrier", "support", "classes"):
            with self.subTest(broken=broken):
                source = self.c4()
                if broken == "barrier":
                    source["all_registered_endpoint_predictions_before_any_TEST_truth"] = False
                elif broken == "support":
                    source["summary"]["argmix"]["overall"]["accuracy"]["independent_parents"] = 4
                else:
                    source["source_class_order"] = ["wrong"]
                self.score(source)
                result = self.publish(force=True)
                self.assertEqual(self.read(reporting.SCORE_PATH)["results"], {})
                self.assertEqual(len(result["reporting_errors"]), 1)

    def test_unknown_schema_has_no_metrics(self):
        self.score(dict(schema="unreviewed", accuracy=1., truth=[0, 1]))
        self.publish()
        compact = self.read(reporting.SCORE_PATH)["results"]["benchmark_score"]
        self.assertFalse(compact["metrics_reported"])
        self.assertNotIn("accuracy", compact)
        self.assertIn("No supported authenticated matched score", (self.root / "README.md").read_text())

    def test_tampered_score_is_recorded_never_displayed(self):
        self.score(self.american())
        (self.root / "scientific/summary.json").write_text('{"schema":"tampered"}')
        result = self.publish()
        self.assertEqual(result["status"], "published")
        self.assertEqual(self.read(reporting.SCORE_PATH)["results"], {})
        self.assertEqual(len(result["reporting_errors"]), 1)
        self.assertIn("Metadata changed", result["reporting_errors"][0]["error"])

    def test_failed_git_is_recorded_and_throttled(self):
        with patch.object(reporting, "_git", return_value=subprocess.CompletedProcess([], 1, "", "mock failure")):
            result = self.publish()
        self.assertEqual(result["status"], "error")
        self.assertIn("mock failure", result["error"])
        self.assertTrue((self.root / result["error_path"]).exists())
        self.git_calls.clear()
        self.now += 5
        self.assertEqual(self.publish()["status"], "updated")
        self.assertEqual(self.git_calls, [])

    def test_push_failure_retries_without_duplicate_commit(self):
        original = self.mock_git
        def fail_push(*args):
            result = original(*args)
            if args[0] == "push":
                return subprocess.CompletedProcess(args, 1, "", "mock offline")
            return result
        with patch.object(reporting, "_git", side_effect=fail_push):
            self.assertEqual(self.publish()["status"], "error")
        self.git_calls.clear()
        self.now += 1801
        self.assertEqual(self.publish()["status"], "published")
        self.assertEqual(self.git_calls, [("push", "origin", "HEAD")])

    def test_malformed_markers_and_wrong_plan_do_not_modify_reports(self):
        (self.root / "README.md").write_text(self.original + reporting.BEGIN)
        self.assertEqual(self.publish()["status"], "error")
        self.assertFalse((self.root / reporting.STATUS_PATH).exists())
        self.assertEqual(self.git_calls, [])
        (self.root / "README.md").write_text(self.original)
        self.state["plan_ref"] = dict(path="other", sha256="0" * 64)
        self.assertEqual(self.publish()["status"], "error")
        self.assertFalse((self.root / reporting.STATUS_PATH).exists())

    def test_existing_marked_block_preserves_exact_surrounding_bytes(self):
        prefix = "# Existing\n\nLegacy metric 0.42.\n"
        suffix = "\n\n## Later score\nAccuracy 0.91.\n"
        (self.root / "README.md").write_text(prefix + reporting.BEGIN + "old" + reporting.END + suffix)
        self.publish()
        text = (self.root / "README.md").read_text()
        self.assertTrue(text.startswith(prefix))
        self.assertTrue(text.endswith(suffix))
        self.assertEqual(text.count(reporting.BEGIN), 1)

    def test_external_live_queue_is_separate_from_workflow_counts(self):
        self.write("external.json", dict(jobs=[dict(admission=dict(job_id="named_external"), status="running")]))
        self.tasks.append(dict(id="external", resource="external", queue_path="external.json", job_id="named_external"))
        self.plan_ref = self.write("plan.json", dict(schema="arggnn_executable_workflow_v1", physical_gpu=0, tasks=self.tasks))
        self.state["plan_ref"] = self.plan_ref
        self.state["tasks"]["external"] = dict(status="pending", external=True)
        self.publish()
        report = self.read(reporting.STATUS_PATH)
        self.assertEqual(report["counts"]["running"], 0)
        self.assertEqual(report["counts"]["pending"], 4)
        self.assertEqual(report["external_queue_jobs"][0]["status"], "running")
        self.assertIn("named_external", (self.root / "README.md").read_text())

    def test_completion_and_force_bypass_interval(self):
        self.publish()
        self.git_calls.clear()
        self.now += 1
        self.state["tasks"]["prepare"]["status"] = "succeeded"
        self.assertEqual(self.publish(force=True)["status"], "published")
        self.state["tasks"]["fit"]["status"] = "succeeded"
        self.state["tasks"]["benchmark_score"]["status"] = "blocked"
        self.publish()
        self.state["tasks"]["benchmark_score"]["status"] = "succeeded"
        self.score(self.american())
        self.state["complete"] = True
        self.assertEqual(self.publish()["status"], "published")
        self.assertTrue(self.read(reporting.STATUS_PATH)["complete"])


if __name__ == "__main__":
    unittest.main()

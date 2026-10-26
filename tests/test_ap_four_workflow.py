"""Synthetic workflow integration checks. No benchmark data, GPU or full fit."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tskit

from arggnn.ancestralpaths_native_bridge import parse_observed_panel
from arggnn.benchmarking import european_sampling as sampling
from arggnn.benchmarking.paper_ancestralpaths_four import CONDITION, PATH_NAMES
from arggnn.benchmarking.paper_ancestralpaths_four.features import extract_four_path_features
from arggnn.benchmarking.paper_ancestralpaths_four import workflow


def ref(name):
    return {"path": name, "sha256": "0" * 64}


def source_plan():
    return sampling.paper_sampling_plan(
        [f"train_{i}" for i in range(3)], [f"test_{i}" for i in range(5)], seed=7)


def registry_for(parent, plan):
    expected = plan["parents"][parent]
    records = []
    for cohort_number, cohort in enumerate(sampling.COHORTS):
        for _ in range(expected["cohort_quotas"][cohort]):
            i = len(records)
            records.append(dict(example_index=i, query_node_id=2 * cohort_number,
                sample_index=2 * cohort_number, sample_id=f"d{cohort_number}", haplotype=0,
                cohort=cohort, native_site_id=i, position_local_bp=float(i),
                position_original_bp=float(i)))
    # Canonical IDs intentionally differ from container order, as may occur
    # after the genuine registry's sort by genomic position.
    return dict(schema="ap_paper_sampled_registry_v1", parent_id=parent,
        split=expected["split"], cohort_quotas=deepcopy(expected["cohort_quotas"]),
        examples=len(records), records=list(reversed(records)),
        labels_present=False, query_truth_read=False,
        native_path_names=list(sampling.NATIVE_PATHS), observation_ref=ref("observed/" + parent))


def two_tree_observations():
    cohorts = ("present_day", "Bronze_Age", "BAA", "Ana", "CHG")
    tables = tskit.TableCollection(100)
    for _ in range(10):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for left, right, order in [(0, 50, [2, 1, 3, 4]), (50, 100, [4, 2, 3, 1])]:
        pairs = []
        for group in range(5):
            pair = tables.nodes.add_row(time=1.5)
            for sample in (2 * group, 2 * group + 1):
                tables.edges.add_row(left, right, pair, sample)
            pairs.append(pair)
        child = pairs[0]
        for step, group in enumerate(order, 1):
            ancestor = tables.nodes.add_row(time=10.9 * step)
            tables.edges.add_row(left, right, ancestor, child)
            tables.edges.add_row(left, right, ancestor, pairs[group])
            child = ancestor
    tables.sort()
    labels = "sample population group sex\n" + "".join(
        f"d{i} {cohort} {cohort} NA\n" for i, cohort in enumerate(cohorts))
    return tables.tree_sequence(), parse_observed_panel(labels)


class FourRegistryAndExtractionTests(unittest.TestCase):
    def test_selected_records_preserve_original_ids_without_renumbering(self):
        plan = source_plan()
        registry = registry_for("test_0", plan)
        original = deepcopy(registry)
        records, indices = workflow.selected_records(registry, plan)
        expected = list(range(4000)) + list(range(6000, 10000))
        np.testing.assert_array_equal(indices, expected)
        self.assertEqual([r["example_index"] for r in records], expected)
        self.assertNotIn("BAA", {r["cohort"] for r in records})
        self.assertEqual(registry, original)

    def test_registry_missing_duplicate_and_truth_bearing_rows_are_rejected(self):
        plan = source_plan()
        original = registry_for("test_0", plan)
        mutations = [
            lambda r: r["records"].pop(),
            lambda r: r["records"][0].update(example_index=r["records"][1]["example_index"]),
            lambda r: r.update(labels_present=True),
            lambda r: r.update(query_truth_read=True),
            lambda r: r["records"][0].update(cohort="BAA"),
        ]
        for change in mutations:
            candidate = deepcopy(original)
            change(candidate)
            with self.subTest(change=change), self.assertRaises(ValueError):
                workflow.selected_records(candidate, plan)

    def test_multi_tree_duplicates_boundaries_and_original_order_match_per_row(self):
        ts, panel = two_tree_observations()
        pairs = [(99.999, 0), (0., 1), (50., 0), (49.999, 0),
                 (50., 0), (0., 2), (25., 0), (50., 3)]
        records = [dict(example_index=17 + i, position_local_bp=p, query_node_id=q)
                   for i, (p, q) in enumerate(pairs)]
        expected = np.stack([extract_four_path_features(ts.at(p), [q], panel)[0]
                             for p, q in pairs]).astype(np.float32)
        check = Mock()
        with patch.object(workflow.features, "extract_four_path_features",
                          wraps=extract_four_path_features) as extraction:
            actual = workflow.extract_rows(ts, panel, records, check=check)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(extraction.call_count, 2)
        self.assertEqual(check.call_count, 2)
        for call in extraction.call_args_list:
            query_ids = call.args[1]
            self.assertEqual(len(query_ids), len(set(query_ids)))
        np.testing.assert_array_equal(actual[2], actual[4])
        self.assertFalse(np.array_equal(actual[2], actual[3]))

    def test_outside_half_open_genomic_domain_is_rejected(self):
        ts, panel = two_tree_observations()
        for position in (-1, 100, np.nan, np.inf):
            with self.subTest(position=position), self.assertRaises(ValueError):
                workflow.extract_rows(ts, panel,
                    [dict(position_local_bp=position, query_node_id=0)], check=lambda: None)
        with self.assertRaises(ValueError):
            workflow.extract_rows(ts, panel, [], check=lambda: None)


class FourLoadedFeatureTests(unittest.TestCase):
    def test_cached_six_width_or_drifted_selected_ids_are_rejected(self):
        plan, plan_ref = source_plan(), ref("source_plan")
        registry = registry_for("test_0", plan)
        records, indices = workflow.selected_records(registry, plan)
        registry_ref, feature_ref = ref("registry/test_0"), ref("four_features/test_0")
        product = dict(schema="ap_four_features_v1", condition=CONDITION,
            source_plan_ref=plan_ref, split="test", parent_id="test_0",
            feature_shape=[5, 8], feature_spec=workflow.features.feature_spec(),
            query_truth_read=False, registry_ref=registry_ref,
            inference_ref=registry["observation_ref"], rows=8000,
            rows_per_cohort={c: 2000 for c in workflow.QUERY_COHORTS},
            arrays_ref=ref("four_arrays/test_0"))
        store = {registry_ref["path"]: registry, feature_ref["path"]: product}
        for defect in ("native_width", "selected_id_order"):
            values = dict(features=np.zeros((8000, 5, 8), dtype=np.float32),
                          example_indices=indices.copy())
            if defect == "native_width":
                values["features"] = np.zeros((8000, 5, 9), dtype=np.float32)
            else:
                values["example_indices"][[0, 1]] = values["example_indices"][[1, 0]]
            with self.subTest(defect=defect), ExitStack() as stack:
                stack.enter_context(patch.object(workflow.original_data, "checked_plan", return_value=plan))
                stack.enter_context(patch.object(workflow.io, "_read", side_effect=lambda r, **_: store[r["path"]]))
                stack.enter_context(patch.object(workflow.io, "_arrays", return_value=values))
                with self.assertRaises(ValueError):
                    workflow.load_features(feature_ref, plan_ref, split="test")


class FourTrainingInputTests(unittest.TestCase):
    def make_fixture(self):
        plan = source_plan()
        parent = plan["train_parent_ids"][0]
        registry = registry_for(parent, plan)
        records, indices = workflow.selected_records(registry, plan)
        feature = dict(parent_id=parent, registry_ref=ref("registry/" + parent),
                       inference_ref=registry["observation_ref"])
        original_feature = dict(registry_ref=feature["registry_ref"],
                                inference_ref=feature["inference_ref"])
        target = dict(schema="ap_paper_TRAIN_native_targets_v1", parent_id=parent,
            split="train", features_ref=ref("original_features/" + parent),
            registry_ref=feature["registry_ref"], arrays_ref=ref("labels/" + parent),
            native_path_names=list(sampling.NATIVE_PATHS), no_label_dependent_selection=True)
        labels = np.zeros(registry["examples"], dtype=np.int64)
        for r in registry["records"]:
            if r["cohort"] == "BAA":
                labels[r["example_index"]] = 4
        products = {p: dict(features_ref=ref("four_features/" + p),
                            source_targets_ref=ref("targets/" + p)) for p in plan["train_parent_ids"]}
        loaded = (feature, registry, records, indices, np.zeros((len(indices), 5, 8), dtype=np.float32))
        return SimpleNamespace(plan=plan, parent=parent, registry=registry, feature=feature,
            original_feature=original_feature, target=target, labels=labels,
            products=products, loaded=loaded)

    def assert_rejected_before_training(self, fixture):
        tensor = Mock()
        arrays = Mock(return_value={"native_labels_0based": fixture.labels})
        store = {"targets/" + fixture.parent: fixture.target,
                 "original_features/" + fixture.parent: fixture.original_feature}
        with ExitStack() as stack:
            stack.enter_context(patch.object(workflow.io, "_ready"))
            stack.enter_context(patch.object(workflow.original_data, "checked_plan", return_value=fixture.plan))
            stack.enter_context(patch.object(workflow, "load_features", return_value=fixture.loaded))
            stack.enter_context(patch.object(workflow.io, "_read", side_effect=lambda r, **_: store[r["path"]]))
            stack.enter_context(patch.object(workflow.io, "_arrays", arrays))
            build = stack.enter_context(patch.object(workflow.model, "build_cnn"))
            with self.assertRaises(ValueError):
                workflow.fit(ref("source_plan"), fixture.products, 7, "portable_nhwc", 1,
                    "unused-output", tensorflow=tensor, guard=lambda _: None,
                    execution_check=lambda: None, write_headroom_bytes=1)
            build.assert_not_called()
            self.assertEqual(tensor.mock_calls, [])
        return arrays

    def test_test_targets_are_rejected_before_label_arrays_or_tensorflow(self):
        f = self.make_fixture()
        f.target.update(schema="ap_paper_TEST_native_targets_v1", split="test")
        self.assert_rejected_before_training(f).assert_not_called()

    def test_target_and_original_feature_alignment_mismatches_are_rejected(self):
        for field in ("registry_ref", "inference_ref"):
            f = self.make_fixture()
            f.original_feature[field] = ref("wrong-parent/" + field)
            with self.subTest(field=field):
                self.assert_rejected_before_training(f).assert_not_called()
        f = self.make_fixture()
        f.target["registry_ref"] = ref("wrong-registry")
        self.assert_rejected_before_training(f).assert_not_called()

    def test_missing_parent_and_wrong_target_length_are_rejected(self):
        f = self.make_fixture()
        del f.products[f.plan["train_parent_ids"][-1]]
        self.assert_rejected_before_training(f).assert_not_called()
        f = self.make_fixture()
        f.labels = f.labels[:-1]
        self.assert_rejected_before_training(f).assert_called_once()

    def test_native_baa_label_in_selected_non_baa_row_is_not_collapsed(self):
        f = self.make_fixture()
        f.labels[f.loaded[3][0]] = 4
        self.assert_rejected_before_training(f).assert_called_once()


class TruthReached(Exception):
    pass


class FourPredictionBarrierTests(unittest.TestCase):
    def make_fixture(self, methods=None):
        methods = methods or [CONDITION, "second_declared_four_output_method"]
        plan = source_plan()
        plan_ref = ref("source_plan")
        metadata, arrays, predictions, targets = {}, {}, {}, {}
        for parent in plan["test_parent_ids"]:
            registry = registry_for(parent, plan)
            registry_ref = ref("registry/" + parent)
            metadata[registry_ref["path"]] = registry
            records, indices = workflow.selected_records(registry, plan)
            targets[parent] = dict(generation_ref=ref("truth/" + parent + "/generation"),
                target_schema_ref=ref("truth/" + parent + "/schema"),
                truth_ref=ref("truth/" + parent + "/values"))
            for method in methods:
                name = method + "/" + parent
                pred_ref, array_ref = ref("pred/" + name), ref("arrays/" + name)
                predictions.setdefault(method, {})[parent] = pred_ref
                metadata[pred_ref["path"]] = dict(schema="ap_four_predictions_v1",
                    condition=method, parent_id=parent, split="test", rows=8000,
                    source_plan_ref=plan_ref, class_names=list(PATH_NAMES), query_truth_read=False,
                    probability_merging=False, registry_ref=registry_ref, predictions_ref=array_ref)
                arrays[array_ref["path"]] = dict(
                    probabilities=np.full((8000, 4), .25, dtype=np.float32),
                    example_indices=indices.copy(),
                    example_ids=np.asarray([parent + ":" + str(i) for i in indices]))
        return SimpleNamespace(plan=plan, plan_ref=plan_ref, metadata=metadata, arrays=arrays,
            predictions=predictions, targets=targets, methods=methods, events=[])

    def invoke(self, f, exception):
        def read(r, **kwargs):
            f.events.append(("metadata", r["path"], kwargs.get("observed", False)))
            if r["path"].startswith("truth/"):
                raise TruthReached(r["path"])
            self.assertTrue(kwargs.get("observed"), "Prediction metadata must be observed-only")
            return f.metadata[r["path"]]

        def load_arrays(r, **kwargs):
            f.events.append(("arrays", r["path"], kwargs.get("observed", False)))
            if r["path"].startswith("truth/"):
                raise TruthReached(r["path"])
            self.assertTrue(kwargs.get("observed"), "Prediction arrays must be observed-only")
            return f.arrays[r["path"]]

        with ExitStack() as stack:
            stack.enter_context(patch.object(workflow.io, "_ready"))
            stack.enter_context(patch.object(workflow.original_data, "checked_plan", return_value=f.plan))
            stack.enter_context(patch.object(workflow.io, "_read", side_effect=read))
            stack.enter_context(patch.object(workflow.io, "_arrays", side_effect=load_arrays))
            with self.assertRaises(exception):
                workflow.score(f.plan_ref, f.predictions, f.targets, f.methods, "unused-output",
                    guard=lambda _: None, execution_check=lambda: None,
                    write_headroom_bytes=1, max_truth_uncompressed_bytes=1)

    def assert_no_truth_read(self, f):
        self.assertFalse(any(path.startswith("truth/") for _, path, _ in f.events))

    def last_prediction(self, f):
        r = f.predictions[f.methods[-1]][f.plan["test_parent_ids"][-1]]
        return f.metadata[r["path"]]

    def test_truth_phase_is_reached_only_after_every_method_and_parent(self):
        f = self.make_fixture()
        self.invoke(f, TruthReached)
        reads = [path for kind, path, _ in f.events if kind == "arrays"]
        self.assertEqual(len(reads), 10)
        self.assertEqual(set(reads), set(f.arrays))
        self.assertTrue(f.events[-1][1].startswith("truth/"))
        self.assertFalse(f.events[-1][2])

    def test_missing_last_method_parent_prevents_all_truth_reads(self):
        f = self.make_fixture()
        del f.predictions[f.methods[-1]][f.plan["test_parent_ids"][-1]]
        self.invoke(f, ValueError)
        self.assert_no_truth_read(f)

    def test_six_output_array_at_last_parent_cannot_pass_four_output_barrier(self):
        f = self.make_fixture()
        last = self.last_prediction(f)
        f.arrays[last["predictions_ref"]["path"]]["probabilities"] = np.full((8000, 6), 1 / 6)
        self.invoke(f, ValueError)
        self.assert_no_truth_read(f)
        self.assertEqual(sum(kind == "arrays" for kind, _, _ in f.events), 10)

    def test_merged_six_output_metadata_cannot_pass(self):
        f = self.make_fixture()
        self.last_prediction(f)["probability_merging"] = True
        self.invoke(f, ValueError)
        self.assert_no_truth_read(f)

    def test_wrong_example_identity_nonfinite_or_incomplete_predictions_block_truth(self):
        for defect in ("example_ids", "example_indices", "nonfinite", "short"):
            f = self.make_fixture()
            values = f.arrays[self.last_prediction(f)["predictions_ref"]["path"]]
            if defect == "example_ids":
                values["example_ids"][0] = "other-parent:0"
            elif defect == "example_indices":
                values["example_indices"][0] = -1
            elif defect == "nonfinite":
                values["probabilities"][0, 0] = np.nan
            else:
                values["probabilities"] = values["probabilities"][:-1]
            with self.subTest(defect=defect):
                self.invoke(f, ValueError)
                self.assert_no_truth_read(f)

    def test_competitors_cannot_silently_use_different_focal_registries(self):
        f = self.make_fixture()
        last = self.last_prediction(f)
        alternate = ref("registry/alternate-identical-content")
        f.metadata[alternate["path"]] = deepcopy(f.metadata[last["registry_ref"]["path"]])
        last["registry_ref"] = alternate
        self.invoke(f, ValueError)
        self.assert_no_truth_read(f)


if __name__ == "__main__":
    unittest.main()

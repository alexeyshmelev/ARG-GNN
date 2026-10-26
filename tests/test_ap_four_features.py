"""Synthetic CPU checks for the paper's BAA-excluded distribution experiment."""
import unittest

import numpy as np
import tskit

from arggnn.ancestralpaths_native_bridge import extract_native_features, parse_observed_panel
from arggnn.benchmarking.paper_ancestralpaths_four import FEATURE_NAMES
from arggnn.benchmarking.paper_ancestralpaths_four.data import (
    four_path_targets, non_baa_row_indices,
)
from arggnn.benchmarking.paper_ancestralpaths_four.features import (
    extract_four_path_features, feature_spec,
)


def panel_for(cohorts, ids=None):
    text = "sample population group sex\n" + "".join(
        f"d{i} {cohort} {cohort} NA\n" for i, cohort in enumerate(cohorts))
    return parse_observed_panel(text, haplotype_node_ids=ids)


def star(cohorts, *, noncontiguous=False):
    tables = tskit.TableCollection(100)
    ids = []
    for _ in range(2 * len(cohorts)):
        if noncontiguous:
            tables.nodes.add_row(time=0)
        ids.append(tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0))
    root = tables.nodes.add_row(time=120.9)
    for leaf in ids:
        tables.edges.add_row(0, 100, root, leaf)
    tables.sort()
    return tables.tree_sequence().first(), panel_for(cohorts, ids)


def chain():
    cohorts = ("present_day", "Bronze_Age", "BAA", "Yam", "Neo", "WHG", "EHG", "Ana", "CHG")
    tables = tskit.TableCollection(100)
    for _ in range(18):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    child = tables.nodes.add_row(time=1.9)
    for leaf in (0, 1):
        tables.edges.add_row(0, 100, child, leaf)
    for group in range(8):
        pair = tables.nodes.add_row(time=2.3)
        for leaf in (2 + 2 * group, 3 + 2 * group):
            tables.edges.add_row(0, 100, pair, leaf)
        ancestor = tables.nodes.add_row(time=10.9 * (group + 1))
        for lower in (child, pair):
            tables.edges.add_row(0, 100, ancestor, lower)
        child = ancestor
    tables.sort()
    return tables.tree_sequence().first(), panel_for(cohorts)


class FourFeaturesTests(unittest.TestCase):
    def test_exclusion_before_normalization_preserves_tree_and_panel(self):
        tree, panel = star(("present_day", "BAA", "Ana", "CHG"))
        before = tree.tree_sequence.dump_tables()
        original_groups = dict(panel.group_by_node)
        x = extract_four_path_features(tree, [0, 1], panel)
        expected = np.zeros((2, 5, 8))
        expected[:, 0, 5:7] = .5
        expected[:, 0, 7] = 120 / 1500
        expected[:, 1:] = -15
        np.testing.assert_array_equal(x, expected)
        self.assertTrue(before.equals(tree.tree_sequence.dump_tables()))
        self.assertEqual(panel.group_by_node, original_groups)
        old = extract_native_features(tree, [0], panel)
        self.assertEqual(old[0, 0, 1], 1 / 3)
        self.assertEqual(x[0, 0, :7].sum(), 1.)

    def test_baa_only_shell_skipped_and_fifth_refilled_beyond_cached_rows(self):
        tree, panel = chain()
        x = extract_four_path_features(tree, [0], panel)[0]
        expected = np.zeros((5, 8))
        expected[np.arange(5), np.arange(5)] = 1
        expected[:, 7] = np.asarray([10, 32, 43, 54, 65]) / 1500
        np.testing.assert_array_equal(x, expected)
        old = extract_native_features(tree, [0], panel)[0]
        self.assertGreater(x[-1, -1], old[-1, -1])

    def test_no_baa_matches_original_feature_arithmetic_exactly(self):
        tree, panel = star(("present_day", "Bronze_Age", "Ana", "CHG", "WHG"))
        new = extract_four_path_features(tree, [0, 2], panel)
        old = extract_native_features(tree, [0, 2], panel)
        np.testing.assert_array_equal(new, old[:, :, (0, 2, 3, 4, 5, 6, 7, 8)])

    def test_only_baa_and_modern_leaves_produce_padding(self):
        tree, panel = star(("present_day", "BAA"))
        x = extract_four_path_features(tree, [0, 1], panel)
        np.testing.assert_array_equal(x, np.full((2, 5, 8), -15.))

    def test_ancient_query_self_inclusion_and_noncontiguous_ids(self):
        tree, panel = star(("present_day", "Bronze_Age", "BAA"), noncontiguous=True)
        modern, bronze = panel.haplotype_node_ids[0], panel.haplotype_node_ids[2]
        x = extract_four_path_features(tree, [bronze, modern], panel)
        self.assertEqual(x[0, 0, 0], 1.)
        np.testing.assert_array_equal(x[0], x[1])
        self.assertEqual(tree.tree_sequence.num_samples, 6)

    def test_age_modes_and_explicit_output_dtype(self):
        tree, panel = star(("present_day", "Bronze_Age", "BAA"))
        x = extract_four_path_features(tree, [0], panel, age_mode="painting_float", dtype=np.float32)
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(float(x[0, 0, 7]), 120.9 / 1500, places=7)
        with self.assertRaises(ValueError):
            extract_four_path_features(tree, [0], panel, dtype=np.int64)

    def test_float32_is_cast_after_author_float64_accumulation(self):
        tree, panel = star(("present_day", "BAA") + ("Bronze_Age",) * 5 + ("Neo",) * 2)
        x64 = extract_four_path_features(tree, [0], panel)
        x32 = extract_four_path_features(tree, [0], panel, dtype=np.float32)
        np.testing.assert_array_equal(x32, x64.astype(np.float32))

    def test_baa_pure_reference_duplicate_empty_and_unknown_queries_rejected(self):
        tree, panel = star(("present_day", "BAA", "Ana"))
        for queries in ([2], [4], [0, 0], [], [999], [True]):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                extract_four_path_features(tree, queries, panel)

    def test_spec_marks_reconstruction_without_topology_changes(self):
        spec = feature_spec()
        self.assertEqual(spec["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(spec["input_shape"], [5, 8])
        self.assertFalse(spec["genealogies_pruned"])
        self.assertFalse(spec["genealogies_reinferred"])
        self.assertTrue(spec["cached_native_six_features_are_insufficient"])


class FourDataTests(unittest.TestCase):
    def test_observed_selection_preserves_existing_row_order(self):
        cohorts = ["BAA", "present_day", "Yam", "BAA", "Neo", "Bronze_Age"]
        np.testing.assert_array_equal(non_baa_row_indices(cohorts), [1, 2, 4, 5])

    def test_no_label_collapse_or_probability_merging(self):
        cohorts = ["present_day", "Bronze_Age", "Neo", "Yam"]
        labels = np.asarray([0, 1, 2, 3], dtype=np.int32)
        result = four_path_targets(labels, cohorts)
        np.testing.assert_array_equal(result, labels)
        self.assertEqual(result.dtype, np.int64)
        result[0] = 2
        self.assertEqual(labels[0], 0)
        for invalid in ([0, 1, 4, 5], [-1, 0, 1, 2], [0., 1., 2., 3.], [True] * 4,
                        [[0, 1, 2, 3]], [0, 1]):
            with self.subTest(labels=invalid), self.assertRaises(ValueError):
                four_path_targets(invalid, cohorts)

    def test_baa_must_be_removed_by_observed_cohort_before_labels(self):
        with self.assertRaisesRegex(ValueError, "BAA query rows"):
            four_path_targets([0, 1], ["present_day", "BAA"])
        for cohorts in ([], ["BAA"], ["unknown"], ["Ana"], [0]):
            with self.subTest(cohorts=cohorts), self.assertRaises(ValueError):
                non_baa_row_indices(cohorts)


if __name__ == "__main__":
    unittest.main()

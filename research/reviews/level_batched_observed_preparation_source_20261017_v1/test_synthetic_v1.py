"""Small synthetic CPU fixtures for ROOT to execute; not executed by author.

No production data, truth, checkpoint, GPU, learned backward or timing probe.
The legacy NumPy routine remains the reference, including its own accumulation.
"""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from arggnn.mutation_routed_tree_observations_v1 import (
    NativeObservedTreeV1, _layout, branch_concordance_support_v1)
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.observations_v1 import _layout_with_isolated_samples
from research.reviews.level_batched_observed_preparation_source_20261017_v1.level_batched_support_v1 import (
    prepare_branch_level_layout_v1, descendant_allele_chunks_v1, level_batched_branch_support_v1)


def tree(parent, time, sample_node, sample_id=None):
    ids = list(range(len(sample_node))) if sample_id is None else sample_id
    return NativeObservedTreeV1(np.array(parent, dtype=np.int64),
        np.array(time, dtype=np.float64), np.array(sample_node, dtype=np.int64),
        np.array(ids, dtype=np.int64))


def legacy_counts(native, genotype, allele_count, ids):
    """Original site/postorder integer primitive, exposed solely for tiny tests."""
    p, _, sn, sid, postorder, _, _, _, _ = _layout(native)
    columns = np.array([list(ids).index(identity) for identity in sid], dtype=np.int64)
    width = int(np.max(allele_count, initial=1))
    result = np.zeros((len(genotype), len(p), width), dtype=np.int64)
    for site, row in enumerate(genotype[:, columns]):
        called = row >= 0
        result[site, sn[called], row[called]] = 1
        for node in postorder:
            if p[node] >= 0:
                result[site, p[node]] += result[site, node]
    return result


class ObservedPreparationFixtures(unittest.TestCase):
    def compare(self, native, rows, alleles, ids=None):
        ids = np.array(native.sample_id if ids is None else ids, dtype=np.int64)
        g = np.array(rows, dtype=np.int8).reshape(-1, len(ids))
        k = np.array(alleles, dtype=np.int64)
        layout = prepare_branch_level_layout_v1(_layout(native))
        expected = branch_concordance_support_v1(native, g, k, ids)
        counts = legacy_counts(native, g, k, ids)
        for chunk in (1, 2, 3, 32):
            actual = level_batched_branch_support_v1(layout, g, k, ids,
                marker_chunk_size=chunk)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertFalse(actual.requires_grad)
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-6)
            self.assertEqual(actual.shape, (len(native.parent), 5))
            observed = list(descendant_allele_chunks_v1(layout, g, k, ids,
                marker_chunk_size=chunk))
            if len(g):
                restored = torch.cat([value[2].permute(1, 0, 2) for value in observed]).numpy()
                np.testing.assert_array_equal(restored, counts)
                self.assertEqual(observed[0][0], 0)
                self.assertEqual(observed[-1][1], len(g))
                self.assertTrue(all(stop-start <= chunk for start, stop, *_ in observed))
            else:
                self.assertEqual(observed, [])
        return layout, g, k, expected

    def test_binary_level_age_ties_missing_multiallelic_and_chunk_axes(self):
        native = tree([4,4,5,5,6,6,-1], [0,0,0,0,2,2,6], [0,1,2,3])
        self.compare(native, [[0,0,1,1],[0,1,0,1],[-1,0,1,-1],
            [0,1,2,3],[0,0,0,0],[-1,-1,-1,-1],[1,-1,-1,-1]],
            [2,2,2,4,1,2,2])

    def test_nonbinary_root_retains_every_sample_and_allele(self):
        native = tree([4,4,4,4,-1], [0,0,0,0,3], [0,1,2,3])
        self.compare(native, [[0,1,2,3],[0,0,1,1],[-1,2,1,0]], [4,2,3])

    def test_unary_complementary_split_and_native_ties(self):
        native = tree([3,4,4,5,5,-1], [0,0,0,1,1,3], [0,1,2])
        _, _, _, support = self.compare(native, [[0,1,1]], [2])
        np.testing.assert_allclose(support[[0,3,4],0], np.ones(3)/3, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(support[:,3], [2,2,2,2,2,0])

    def test_native_indices_and_identity_column_permutation(self):
        # Internal sampled ancestor, non-contiguous identity keys, genotype
        # columns reordered independently of native sampled-node order.
        native = tree([2,2,3,-1], [0,0,1,3], [0,1,2], [30,10,90])
        self.compare(native, [[2,0,1],[-1,1,1],[0,0,0]], [3,2,1], ids=[90,30,10])
        perm = np.array([3,1,0,2], dtype=np.int64)
        inverse = np.argsort(perm)
        oldp = native.parent[perm]
        parent = np.where(oldp >= 0, inverse[np.maximum(oldp, 0)], -1)
        moved = tree(parent, native.node_time[perm], inverse[native.sample_node], native.sample_id)
        self.compare(moved, [[2,0,1],[-1,1,1]], [3,2], ids=[90,30,10])

    def test_raw_positive_duration_with_rounded_FP32_age_tie(self):
        native = tree([2,2,-1], [100000000.,0.,100000001.], [0,1])
        self.assertEqual(np.float32(native.node_time[0]), np.float32(native.node_time[2]))
        self.compare(native, [[0,1],[-1,1]], [2,2])

    def test_empty_and_noninformative_connected_holds(self):
        native = tree([2,2,-1], [0,0,1], [0,1])
        self.compare(native, [], [])
        self.compare(native, [[0,0],[-1,-1],[0,-1]], [1,2,2])
        singleton = tree([-1], [4], [0], [41])
        self.compare(singleton, [[0],[-1]], [1,2], ids=[41])

    def test_all_sampled_isolated_zero_SNP_without_virtual_edges(self):
        native = tree([-1,-1,-1], [0,2,0], [2,0,1], [30,10,20])
        layout = prepare_branch_level_layout_v1(_layout_with_isolated_samples(native))
        g, k = np.empty((0,3), dtype=np.int8), np.empty(0,dtype=np.int64)
        out = level_batched_branch_support_v1(layout, g, k, [10,20,30])
        expected = np.zeros((3,5), dtype=np.float32); expected[:,4] = 1.
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertEqual(layout.branches.numel(), 0)
        with self.assertRaisesRegex(ValueError, "zero-SNP"):
            level_batched_branch_support_v1(layout, np.array([[0,1,0]]), [2], [10,20,30])

    def test_bound_chunks_do_not_drop_sites_and_failstop_one_marker(self):
        native = tree([2,2,-1], [0,0,1], [0,1])
        layout, g, k, expected = self.compare(native, [[0,1],[1,0],[0,1]], [2,2,2])
        chunks = list(descendant_allele_chunks_v1(layout, g, k, [0,1],
            marker_chunk_size=32, max_count_elements=6))
        self.assertEqual([(a,b) for a,b,*_ in chunks], [(0,1),(1,2),(2,3)])
        actual = level_batched_branch_support_v1(layout, g, k, [0,1],
            max_count_elements=6)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-6)
        with self.assertRaisesRegex(ValueError, "One marker"):
            level_batched_branch_support_v1(layout, g, k, [0,1], max_count_elements=5)

    def test_invalid_alleles_joins_and_unsupported_disconnected_topology(self):
        native = tree([2,2,-1], [0,0,1], [0,1])
        layout = prepare_branch_level_layout_v1(_layout(native))
        for g, k, ids in [([[0,2]], [2], [0,1]), ([[-2,0]], [2], [0,1]),
                          ([[0,1]], [2], [0,0]), ([[0,1]], [2], [0,9])]:
            with self.assertRaises(ValueError):
                level_batched_branch_support_v1(layout, np.array(g), np.array(k), ids)
        partial = (np.array([2,-1,-1]), None, np.array([0,1]), np.array([0,1]),
            None, None, np.array([0,0,1]), None, None)
        with self.assertRaisesRegex(ValueError, "connected tree"):
            prepare_branch_level_layout_v1(partial)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()

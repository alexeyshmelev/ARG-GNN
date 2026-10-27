"""Small deterministic CPU equivalence cases; no scientific dataset or GPU use."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast
from pathlib import Path
import unittest
import warnings
import numpy as np
from research.preprocessing_optimization_20261026.tree import episodes as original
from research.efficiency_20261027.independent_efficiency_challenge.episode_integer_vectorization import episodes_v1 as candidate
from research.efficiency_20261027.independent_efficiency_challenge.episode_integer_vectorization import integer_writes_v1 as writes


def function_ast(module, name):
    return next(node for node in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def call_with_receipt(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        try:
            output = fn(*args, **kwargs)
            error = None
        except Exception as exc:
            output = None
            error = (type(exc).__name__, str(exc))
    return output, error, [(item.category.__name__, str(item.message)) for item in observed]


class IntegerWriteEquivalence(unittest.TestCase):
    def compare(self, current, previous, *, n=16, members=0, episode_count=0,
                hold=7, initial_sizes=None):
        buffers = [np.full(n, -123, dtype=np.int32) for _ in range(3)]
        buffers.append(np.zeros(n, dtype=np.int32) if initial_sizes is None else initial_sizes.copy())
        left, right = [x.copy() for x in buffers], [x.copy() for x in buffers]
        a = call_with_receipt(writes.scalar_integer_writes, current, previous, hold,
                              *left, members, episode_count, n)
        b = call_with_receipt(writes.write_episode_integers, current, previous, hold,
                              *right, members, episode_count, n)
        self.assertEqual(a, b)
        if a[0] is not None:
            self.assertEqual(list(a[0][0].items()), list(b[0][0].items()))
            self.assertTrue(all(type(value) is int for value in b[0][0].values()))
        for x, y in zip(left, right):
            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x.tobytes(), y.tobytes())
        return a, left

    def test_reference_loop_is_original_source(self):
        builder = function_ast(original, 'build_clade_episode_arrays_v1')
        loop = next(node for node in builder.body if isinstance(node, ast.For))
        reference = function_ast(writes, 'scalar_integer_writes')
        self.assertEqual(ast.dump(ast.Module(body=loop.body[-3:-1], type_ignores=[])),
                         ast.dump(ast.Module(body=reference.body[1:-1], type_ignores=[])))

    def test_builder_only_replaces_membership_loop(self):
        left = function_ast(original, 'build_clade_episode_arrays_v1')
        right = function_ast(candidate, 'build_clade_episode_arrays_v1')
        left.body.pop(0)
        right.body.pop(0)
        old_loop = next(node for node in left.body if isinstance(node, ast.For))
        new_loop = next(node for node in right.body if isinstance(node, ast.For))
        old_loop.body[-3:-1] = new_loop.body[-2:-1]
        self.assertEqual(ast.dump(left), ast.dump(right))
        self.assertIs(original._span_denominators, candidate._span_denominators)
        self.assertIs(original._forest_index, candidate._forest_index)
        self.assertIs(original._tree_clades, candidate._tree_clades)

    def test_empty_at_buffer_bounds(self):
        self.compare({}, {}, n=3, members=3, episode_count=3)

    def test_new_persistent_disappearing_reappearing_order(self):
        cases = [({}, {}, 0), ({5: 6, 3: 4}, {}, 0),
                 ({3: 8, 5: 9, 6: 10}, {5: 0, 3: 1}, 2),
                 ({6: 12}, {3: 1, 5: 0, 6: 2}, 3),
                 ({5: 13, 6: 14}, {6: 2}, 3)]
        for current, previous, count in cases:
            with self.subTest(current=current):
                self.compare(current, previous, episode_count=count)

    def test_duplicate_episode_ids_count_repeatedly(self):
        self.compare({3: 4, 5: 6, 6: 7}, {3: 0, 5: 0, 6: 0}, episode_count=1)

    def test_exact_member_and_episode_boundary(self):
        self.compare({3: 4, 5: 6}, {}, n=8, members=6, episode_count=6)
        self.compare({3: 4, 5: 6}, {3: 0, 5: 1}, n=8, members=6, episode_count=8)

    def test_membership_rejection_preserves_partial_writes(self):
        receipt, buffers = self.compare({3: 4, 5: 6}, {}, n=8, members=7)
        self.assertEqual(receipt[1], ('ValueError', 'At most one proper-clade membership per native vertex'))
        self.assertEqual(int(buffers[0][7]), 4)

    def test_episode_rejection_preserves_partial_writes_and_guard_order(self):
        receipt, buffers = self.compare({3: 4, 5: 6}, {}, n=8, episode_count=7)
        self.assertEqual(receipt[1], ('ValueError', 'Episode buffer bound exceeded'))
        self.assertEqual(int(buffers[0][0]), 4)
        receipt, _ = self.compare({3: 4}, {}, n=8, members=8, episode_count=8)
        self.assertEqual(receipt[1], ('ValueError', 'At most one proper-clade membership per native vertex'))

    def test_int32_max_and_warning_behavior(self):
        for initial in (np.iinfo(np.int32).max - 2, np.iinfo(np.int32).max - 1,
                        np.iinfo(np.int32).max, np.iinfo(np.int32).min):
            with self.subTest(initial=initial):
                sizes = np.zeros(8, dtype=np.int32)
                sizes[0] = initial
                self.compare({3: 4, 5: 6}, {3: 0, 5: 0}, n=8,
                             episode_count=1, initial_sizes=sizes)


BINARY = np.array([4, 4, 5, 5, 6, 6, -1], dtype=np.int64)
ALTERNATE = np.array([4, 5, 4, 5, 6, 6, -1], dtype=np.int64)
UNARY = np.array([4, 4, 6, 6, 5, 7, 7, -1], dtype=np.int64)
ISOLATED = np.full(4, -1, dtype=np.int64)


def forest(trees):
    parents, samples, offset = [], [], 0
    for tree in trees:
        parents.append(np.where(tree >= 0, tree + offset, -1))
        samples.append(np.arange(offset, offset + 4, dtype=np.int64))
        offset += len(tree)
    return np.concatenate(parents), np.stack(samples)


class CompleteBuilderEquivalence(unittest.TestCase):
    def compare(self, trees, **kwargs):
        parent, samples = forest(trees)
        spans = np.asarray([0.1 + index * 0.071 for index in range(len(trees))], dtype=np.float32)
        a = call_with_receipt(original.build_clade_episode_arrays_v1, parent, samples, spans, **kwargs)
        b = call_with_receipt(candidate.build_clade_episode_arrays_v1, parent, samples, spans, **kwargs)
        self.assertEqual(a[1:], b[1:])
        self.assertIsNone(a[1])
        aa, am = a[0]
        ba, bm = b[0]
        self.assertEqual(am, bm)
        self.assertEqual(tuple(aa), tuple(ba))
        for key in aa:
            self.assertEqual(aa[key].dtype, ba[key].dtype)
            self.assertEqual(aa[key].shape, ba[key].shape)
            self.assertEqual(aa[key].tobytes(), ba[key].tobytes(), key)

    def test_single_hold_and_persistent_episodes(self):
        self.compare([BINARY])
        self.compare([BINARY] * 7)

    def test_transitions_and_true_unary_vertices(self):
        self.compare([BINARY, BINARY, ALTERNATE, UNARY, UNARY, BINARY])

    def test_isolated_native_roots_and_empty_memberships(self):
        self.compare([ISOLATED], isolated_hold_indices=(0,))
        self.compare([BINARY, ISOLATED, BINARY, UNARY, ISOLATED], isolated_hold_indices=(1, 4))

    def test_public_rejections_match(self):
        parent, samples = forest([BINARY])
        cases = [((parent, samples, np.array([0.1], dtype=np.float64)), {}),
                 ((parent, samples, np.array([0], dtype=np.float32)), {}),
                 ((parent, samples, np.array([0.1], dtype=np.float32)), {'max_payload_bytes': 1}),
                 ((np.array([-2, -1, -1], dtype=np.int64), np.array([[0, 1, 2]], dtype=np.int64), np.array([1], dtype=np.float32)), {}),
                 ((ISOLATED, np.array([[0, 1, 2, 3]], dtype=np.int64), np.array([1], dtype=np.float32)), {})]
        for args, kwargs in cases:
            with self.subTest(kwargs=kwargs, parent=args[0].tolist()):
                a = call_with_receipt(original.build_clade_episode_arrays_v1, *args, **kwargs)
                b = call_with_receipt(candidate.build_clade_episode_arrays_v1, *args, **kwargs)
                self.assertIsNotNone(a[1])
                self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)

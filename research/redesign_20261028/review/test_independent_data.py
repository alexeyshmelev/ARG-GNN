"""Independent CPU edge-case checks. Only hand-authored small fixtures are used."""
import random
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tskit
from research.redesign_20261028.data.window_data import ReplacementTreeReader, SortedVCF, tree_from_tskit

HERE = Path('research/redesign_20261028/review')


class IndependentDataChecks(unittest.TestCase):
    def test_future_insertion_cannot_extend_current_tree(self):
        tables = tskit.TableCollection(10)
        for _ in range(3):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=2)
        tables.edges.add_row(0, 10, 3, 0)
        tables.edges.add_row(0, 10, 3, 1)
        tables.edges.add_row(5, 10, 4, 3)
        tables.edges.add_row(5, 10, 4, 2)
        tables.sort()
        ts = tables.tree_sequence()
        with tempfile.TemporaryDirectory(dir=HERE, prefix='native_interval_') as folder:
            path = Path(folder) / 'fixture.trees'
            ts.dump(path)
            reader = ReplacementTreeReader(path, ts.samples())
            expected = tree_from_tskit(ts.at(2), ts.samples())
            try:
                actual = reader.at(2)
            except ValueError as error:
                self.assertTrue(any(word in str(error).lower() for word in ['replacement', 'persistent', 'insertion', 'fallback']))
            finally:
                for array in reader.columns.arrays.values():
                    if isinstance(array, np.memmap):
                        array._mmap.close()
            if 'actual' in locals():
                self.assertEqual((actual.left, actual.right), (expected.left, expected.right))
                np.testing.assert_array_equal(actual.parent, expected.parent)

    def test_previous_removal_cannot_extend_current_tree_left(self):
        tables = tskit.TableCollection(10)
        for _ in range(3):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=2)
        tables.edges.add_row(0, 5, 3, 0)
        tables.edges.add_row(2, 10, 4, 1)
        tables.edges.add_row(2, 10, 4, 2)
        tables.sort()
        ts = tables.tree_sequence()
        with tempfile.TemporaryDirectory(dir=HERE, prefix='previous_removal_') as folder:
            path = Path(folder) / 'fixture.trees'
            ts.dump(path)
            reader = ReplacementTreeReader(path, ts.samples())
            expected = tree_from_tskit(ts.at(6), ts.samples())
            try:
                actual = reader.at(6)
            except ValueError as error:
                self.assertTrue(any(word in str(error).lower() for word in ['replacement', 'persistent', 'removal', 'fallback']))
            finally:
                for array in reader.columns.arrays.values():
                    if isinstance(array, np.memmap):
                        array._mmap.close()
            if 'actual' in locals():
                self.assertEqual((actual.left, actual.right), (expected.left, expected.right))
                np.testing.assert_array_equal(actual.parent, expected.parent)

    def test_plain_vcf_random_lengths_exact_rows_and_missing_positions(self):
        rng = random.Random(20261028)
        panel = [dict(sample_id='sample', haplotype=h) for h in (0, 1)]
        with tempfile.TemporaryDirectory(dir=HERE, prefix='seek_') as folder:
            for trial in range(20):
                path = Path(folder) / ('fixture_%02d.vcf' % trial)
                count = rng.randint(2, 120)
                positions = np.cumsum([rng.randint(2, 40) for _ in range(count)]).tolist()
                lines = ['##fileformat=VCFv4.2', '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample']
                for i, position in enumerate(positions):
                    info = 'NOTE=' + 'x' * rng.choice([1, 2, 30, 256, 1024, 8192])
                    genotype = '.|1' if i % 5 == 0 else '0|1'
                    lines.append(f'22\t{position}\t{i}\tA\tC\t.\tPASS\t{info}\tGT:DP\t{genotype}:40')
                path.write_text('\n'.join(lines) + ('\n' if trial % 2 == 0 else ''))
                reader = SortedVCF(path, panel)
                try:
                    order = list(range(count))
                    rng.shuffle(order)
                    for i in order:
                        calls, alleles = reader.at(positions[i])
                        self.assertEqual(calls.tolist(), [-1, 1] if i % 5 == 0 else [0, 1])
                        self.assertEqual(alleles, ['A', 'C'])
                    for position in [0, positions[0] - 1, positions[-1] + 1]:
                        with self.assertRaisesRegex(ValueError, 'not found'):
                            reader.at(position)
                finally:
                    reader.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)

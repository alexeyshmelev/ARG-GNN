"""Synthetic tests for European concrete preparation; no scientific runtime."""
import copy
import importlib.util
import io
import sys
import types
import unittest
from unittest.mock import Mock, patch

from arggnn.benchmarking import european_execution as execution
from arggnn.benchmarking import european_recipes as recipes


class MemoryPath:
    def __init__(self, store, path):
        self.store, self.path = store, path

    def __truediv__(self, name):
        return MemoryPath(self.store, self.path + "/" + name)

    def exists(self):
        return self.path in self.store

    def mkdir(self):
        if self.exists():
            raise FileExistsError(self.path)
        self.store[self.path] = "DIRECTORY"

    def read_text(self):
        return self.store[self.path]

    def open(self):
        return io.StringIO(self.read_text())

    def __str__(self):
        return self.path


def rows_for(recipe):
    result = []
    for cohort in recipe["parameters"]["cohorts"]:
        for _ in range(cohort["diploids"]):
            sample = len(result) // 2
            for hap in (0, 1):
                result.append(dict(sample_index=len(result), sample_id=f"s{sample}",
                    haplotype=hap, observed_cohort=cohort["id"], age_generations=0,
                    role="query" if cohort["role"] == "query" else "pure_source_reference",
                    observed_reference_class=None))
    return result


class SyntheticFeatureRows:
    def __init__(self, nodes):
        self.nodes, self.shape = nodes, (len(nodes), 5, 9)

    def __getitem__(self, index):
        node = self.nodes[index]
        return types.SimpleNamespace(astype=lambda dtype: ("FEATURE", node, dtype))


class TinyVector(list):
    def astype(self, dtype):
        return TinyVector(int(v) for v in self)

    def tolist(self):
        return list(self)

    def __sub__(self, value):
        return TinyVector(v - value for v in self)

    def __gt__(self, other):
        return TinyVector(a > b for a, b in zip(self, other))

    def __and__(self, value):
        return TinyVector(v & value for v in self)

    def __getitem__(self, value):
        if isinstance(value, TinyVector):
            return TinyVector(v for v, keep in zip(self, value) if keep)
        return super().__getitem__(value)


class FakeTable(types.SimpleNamespace):
    def clear(self):
        pass

    def drop_metadata(self):
        pass

    def set_columns(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class EuropeanExecutionTests(unittest.TestCase):
    def setUp(self):
        self.recipe = recipes.load_recipe("ancestralpaths_european")

    def test_import_does_not_import_science_or_launch(self):
        spec = importlib.util.spec_from_file_location("arggnn.benchmarking.european_execution_inert", execution.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("importlib.import_module", side_effect=AssertionError("No runtime import")):
            spec.loader.exec_module(module)

    def test_approval_guard_and_sources_precede_runtime(self):
        events = []
        with patch.object(recipes, "verify_sources", side_effect=lambda recipe: events.append("sources")), \
             patch.object(execution, "_runtime", side_effect=RuntimeError("synthetic runtime stop")):
            with self.assertRaisesRegex(RuntimeError, "synthetic runtime stop"):
                execution.prepare_relate_inputs(self.recipe, {}, "RAM", guard=lambda n: events.append("guard"),
                    execution_check=lambda: events.append("approval"), write_headroom_bytes=1)
        self.assertEqual(events, ["approval", "guard", "sources", "sources"])

    def test_canonical_metadata_rejects_mutation_before_approval(self):
        self.recipe["parameters"]["length_bp"] = 2000000
        approve = Mock()
        with self.assertRaisesRegex(ValueError, "Canonical registered"):
            execution.prepare_relate_inputs(self.recipe, {}, "RAM", guard=Mock(),
                execution_check=approve, write_headroom_bytes=1)
        approve.assert_not_called()

    def test_callbacks_and_headroom_are_required(self):
        for guard, approve, bound in ((None, Mock(), 1), (Mock(), None, 1),
                                      (Mock(), Mock(), 0), (Mock(), Mock(), True)):
            with self.assertRaises(ValueError):
                execution.prepare_relate_inputs(self.recipe, {}, "RAM", guard=guard,
                    execution_check=approve, write_headroom_bytes=bound)

    def test_prepare_uses_full_200mb_map_and_original_filter(self):
        rows = rows_for(self.recipe)
        store = {"source.vcf": "FAKE_VCF", "ages": "FAKE_AGES", "labels": "FAKE_LABELS"}
        destination = MemoryPath(store, "inputs")
        paths = {name: MemoryPath(store, path) for name, path in (
            ("genotypes.vcf", "source.vcf"), ("sample_ages.generations", "ages"),
            ("population_labels.txt", "labels"))}
        assets = dict(rows=rows, names=[row["sample_id"] for row in rows[::2]], paths=paths,
                      position_join={200000000: dict(VCF_position_1based=200000000,
                          tree_position_0based=199999999, site_id=0)}, source_sha256={"VCF": "f" * 64})
        retained = [{"fields": ["FAKE", "FILTERED", "SITE"]}]
        filter_vcf = Mock(return_value=(["FAKE_HEADER"], retained, {"retained": 1}))

        def write(path, value, guard):
            guard(0)
            store[str(path)] = value

        full = types.SimpleNamespace(confined=lambda x: x, roster=lambda: rows,
                                     text_write=write, json_write=write)
        helper = types.SimpleNamespace(filter_vcf=filter_vcf)
        with patch.object(execution, "_approved", return_value=200000000), \
             patch.object(execution, "_runtime", return_value=(None, None, full, helper)):
            actual = execution.prepare_relate_inputs(self.recipe, assets, destination,
                guard=Mock(), execution_check=Mock(), write_headroom_bytes=100)
        self.assertEqual(actual, (retained, {"retained": 1}))
        filter_vcf.assert_called_once()
        self.assertEqual(store["inputs/input.map"], "pos COMBINED_rate Genetic_Map\n1 1 0\n200000001 1 200\n")
        self.assertEqual(store["inputs/input.vcf"], "FAKE_HEADER\nFAKE\tFILTERED\tSITE\n")
        self.assertEqual(store["inputs/preparation.json"]["length_bp"], 200000000)
        self.assertFalse(store["inputs/preparation.json"]["truth_or_oracle_files_read"])

    def test_features_stream_one_tree_at_a_time_and_preserve_repeated_draws(self):
        trees = [types.SimpleNamespace(interval=types.SimpleNamespace(left=0, right=100)),
                 types.SimpleNamespace(interval=types.SimpleNamespace(left=100, right=200))]
        ts = types.SimpleNamespace(sequence_length=200, trees=lambda: iter(trees))
        panel = types.SimpleNamespace(haplotype_node_ids=tuple(range(952)))
        sampled = [
            dict(example_index=2, position_local_bp=20, query_node_id=0),
            dict(example_index=0, position_local_bp=20, query_node_id=0),
            dict(example_index=1, position_local_bp=170, query_node_id=1)]
        extract = Mock(side_effect=lambda tree, nodes, panel, **kwargs: SyntheticFeatureRows(nodes))
        bridge = types.SimpleNamespace(extract_native_features=extract)
        np = types.SimpleNamespace(float64="Float64", float32="Float32")
        with patch.object(execution, "_approved"), \
             patch.object(execution, "_runtime", return_value=(np, None, None, None)), \
             patch.object(execution.importlib, "import_module", return_value=bridge):
            stream = execution.iter_native_sampled_features(
                self.recipe, ts, panel, tuple(range(686)), sampled,
                guard=Mock(), execution_check=Mock(), write_headroom_bytes=100)
            self.assertFalse(extract.called)
            self.assertEqual(next(stream), (2, ("FEATURE", 0, "Float32")))
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(next(stream), (0, ("FEATURE", 0, "Float32")))
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(next(stream), (1, ("FEATURE", 1, "Float32")))
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(list(stream), [])
        self.assertEqual(extract.call_args_list[0].args[1], [0])
        self.assertEqual(extract.call_args_list[0].kwargs, dict(age_mode="training_int", dtype="Float64"))

    def test_stream_rejects_unsorted_registry_and_duplicate_identity(self):
        tree = types.SimpleNamespace(interval=types.SimpleNamespace(left=0, right=200))
        ts = types.SimpleNamespace(sequence_length=200, trees=lambda: iter([tree]))
        panel = types.SimpleNamespace(haplotype_node_ids=tuple(range(952)))
        for rows in (
            [dict(example_index=0, position_local_bp=20, query_node_id=0),
             dict(example_index=1, position_local_bp=10, query_node_id=0)],
            [dict(example_index=0, position_local_bp=20, query_node_id=0),
             dict(example_index=0, position_local_bp=30, query_node_id=0)],
            [dict(example_index=0, position_local_bp=200, query_node_id=0)]):
            with self.subTest(rows=rows), patch.object(execution, "_approved"), \
                 patch.object(execution, "_runtime", return_value=(None, None, None, None)), \
                 patch.object(execution.importlib, "import_module"):
                with self.assertRaises(ValueError):
                    list(execution.iter_native_sampled_features(
                        self.recipe, ts, panel, tuple(range(686)), rows,
                        guard=Mock(), execution_check=Mock(), write_headroom_bytes=100))

    def test_cleanup_preserves_full_200mb_coordinates_without_dense_grid(self):
        store = {"native": "FAKE_NATIVE"}
        native, output = MemoryPath(store, "native"), MemoryPath(store, "clean")
        edges = FakeTable(left=TinyVector([1, 1]), right=TinyVector([200000001, 200000001]),
                          parent=TinyVector([952, 952]), child=TinyVector([0, 1]))
        tables = FakeTable(
            edges=edges, nodes=FakeTable(flags=TinyVector([1] * 952 + [0])),
            sites=FakeTable(position=TinyVector([1, 200000000])), mutations=FakeTable(),
            migrations=FakeTable(), populations=FakeTable(), individuals=FakeTable(),
            provenances=FakeTable(), sort=lambda: None)
        clean = types.SimpleNamespace(num_nodes=953, num_sites=2, num_trees=1,
            dump=lambda path: store.__setitem__(str(path), "FAKE_CLEAN"))
        tables.tree_sequence = lambda: setattr(clean, "sequence_length", tables.sequence_length) or clean
        inferred = types.SimpleNamespace(tables=tables, num_nodes=953, nbytes=100, dump_tables=lambda: tables)
        mapping = [dict(sample_index=i, inferred_tree_node_id=i) for i in range(952)]
        helper = types.SimpleNamespace(inferred_sample_mapping=lambda ts, assets: mapping)
        full = types.SimpleNamespace(confined=lambda value: value, sha=lambda path: "f" * 64)
        np = types.SimpleNamespace(
            int64="Int64", int32="Int32",
            array_equal=lambda a, b: list(a) == list(b),
            maximum=lambda values, bound: TinyVector(max(v, bound) for v in values),
            minimum=lambda values, bound: TinyVector(min(v, bound) for v in values),
            full=lambda size, value, dtype: TinyVector([value] * size))
        tskit = types.SimpleNamespace(NULL=-1, NODE_IS_SAMPLE=1, MetadataSchema=lambda value: value,
                                      load=lambda path: inferred)
        assets = dict(rows=rows_for(self.recipe),
            position_join={1: dict(site_id=0), 200000000: dict(site_id=1)})
        with patch.object(execution, "_approved", return_value=200000000), \
             patch.object(execution, "_runtime", return_value=(np, tskit, full, helper)):
            result = execution.clean_converted_tree(
                self.recipe, native, assets, {1: {}, 200000000: {}}, output,
                guard=Mock(), execution_check=Mock(), write_headroom_bytes=1)
        self.assertEqual(result["coverage_original_bp0"], [0, 200000000])
        self.assertEqual(result["normalization_length_bp"], 200000000)
        self.assertEqual(result["site_join"][-1]["local_position_0based"], 199999999)
        self.assertEqual(tables.edges.right, [200000000, 200000000])
        self.assertEqual(tables.sites.position, [0, 199999999])
        self.assertEqual(len(result["sample_index_to_inferred_tree_node_id"]), 952)
        self.assertFalse(result["auxiliary_grid_included"])
        self.assertNotIn("retained_fixed100bp_indexes", result)
        self.assertEqual(store["clean"], "FAKE_CLEAN")

    def test_cleanup_refuses_invalid_grid_choice_before_runtime(self):
        with patch.object(execution, "_approved", return_value=200000000), \
             patch.object(execution, "_runtime") as load:
            with self.assertRaisesRegex(ValueError, "auxiliary-grid"):
                execution.clean_converted_tree(self.recipe, "RAM", {}, {}, "OUT",
                    guard=Mock(), execution_check=Mock(), write_headroom_bytes=1,
                    include_auxiliary_grid=1)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

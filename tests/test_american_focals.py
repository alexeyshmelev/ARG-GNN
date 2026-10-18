"""Tiny fake-tree parity against the frozen original full-corpus selector."""
from pathlib import Path
import os
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from arggnn import american_c3_paths_v1 as original
from arggnn.benchmarking import american_focals as compact


class FakeTree:
    def __init__(self, index, left, right, phases):
        self.index = index
        self.interval = SimpleNamespace(left=left, right=right)
        self.parents = {}
        for sample, phase in enumerate(phases):
            base = 100+10*sample
            path = ([sample, base, base+1, 1000+phase] if phase < 2 else
                    [sample, base+2, 1000] if phase == 2 else [sample])
            self.parents.update(zip(path, path[1:]))
    def parent(self, node):
        return self.parents.get(node, -1)


class FakeTS:
    def __init__(self):
        self.num_samples, self.num_trees, self.num_nodes = 6, 8, 1100
        self.num_migrations = self.num_populations = 0
        self.tables = SimpleNamespace(nodes=SimpleNamespace(population=np.full(1100, -1)))
        self.breaks = [0., 1., 4., 7., 8., 13., 22., 31., 40.]
        phases = [[0,0,1,0,2,2,1,3], [1,0,1,0,1,0,1,0],
                  [0,0,1,0,2,2,1,3], [0,1,0,1,0,1,0,1],
                  [2,2,0,1,2,3,0,1], [3,0,3,0,1,2,1,2]]
        self._trees = [FakeTree(i, self.breaks[i], self.breaks[i+1],
                                [p[i] for p in phases]) for i in range(8)]
    def samples(self):
        return np.arange(self.num_samples)
    def trees(self):
        return iter(self._trees)
    def breakpoints(self):
        return iter(self.breaks)


class FocalsCaptured(Exception):
    """Stop the original immediately after its exact focal array production."""


class MemoryWriter:
    def __init__(self):
        self.arrays = {}
        self.jsons = {}
    def npz(self, name, **arrays):
        self.arrays[name] = {k:v.copy() for k,v in arrays.items()}
        if name == "focals.npz":
            raise FocalsCaptured()
        return dict(path="mock/"+name, sha256="a"*64)
    def pickle(self, name, value, planned):
        return dict(path="mock/"+name, sha256="a"*64)
    def json(self, name, value):
        self.jsons[name] = value
        return dict(path="mock/"+name, sha256="a"*64)


class ExactCompactFocalsTest(unittest.TestCase):
    def setUp(self):
        self.ts = FakeTS()
        self.rows = [dict(sample_index=i, observed_node_id=i,
                          role="pure_source_reference" if i<2 else "query",
                          observed_reference_class=i if i<2 else None) for i in range(6)]
        self.observation = dict(parent_id="synthetic_parent", class_order=["EUR","EAS","AFR"],
                                genealogy_kind=compact.observation.GENEALOGY,
                                schema="american_c3_observed_v1", query_ancestry_truth_in_inputs=False,
                                roster_ref="mock_roster", tree_ref="mock_tree")
        self.roster = dict(parent_id="synthetic_parent", class_order=self.observation["class_order"],
                           samples=self.rows)
        self.cfg = dict(original.DEFAULTS, observation_ref="mock_observation",
                        max_unique_paths=100, max_path_tokens=2000, max_path_occurrences=48,
                        max_vocabulary_nodes=2000, max_raw_path_nodes=10,
                        max_sample_unique_paths=20, max_sample_path_tokens=200,
                        query_focal_count=6, output_cap_bytes=10**9)
        self.fake_tskit = SimpleNamespace(NULL=-1, load=lambda _:self.ts)

    def run_pair(self, seed, count=6):
        cfg = dict(self.cfg, path_sampling_seed=seed, query_focal_count=count)
        writer = MemoryWriter()
        with patch.dict(sys.modules, {"tskit":self.fake_tskit}):
            functions = original.author_functions()
            def read(ref):
                return self.observation if ref == "mock_observation" else self.roster
            with patch.object(original, "read", side_effect=read), \
                 patch.object(original, "bound", return_value=Path("mock_tree")), \
                 patch.object(original, "ref", return_value=dict(path="mock_author", sha256="b"*64)):
                with self.assertRaises(FocalsCaptured):
                    original.execute(cfg, writer)
            arrays, counts = compact._exact_focals(self.ts, self.rows, cfg,
                path_key=functions["get_path_key"], resource_check=lambda _:None)
        expected = writer.arrays["focals.npz"]
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, expected[key], err_msg=key)
        self.assertEqual(counts["unique_paths"], 20)
        self.assertEqual(counts["path_tokens"], 64)
        self.assertEqual(counts["full_native_path_occurrences"], 48)
        return arrays, counts

    def test_identical_original_ids_rng_first_intervals_and_sample_support(self):
        for seed in (0, 1, 92001):
            with self.subTest(seed=seed):
                self.run_pair(seed)
        arrays, counts = self.run_pair(17, count=14)
        self.assertEqual(counts["reference_unique_paths"], 6)
        self.assertEqual(counts["query_unique_paths"], 14)
        self.assertEqual(set(arrays["sample_indices"]), {2,3,4,5})
        # Two paths share their first three raw nodes, but differ at the root.
        # Repeated occurrences of the first path must keep its original interval.
        first = np.flatnonzero(arrays["record_ids"] == 6).item()
        second = np.flatnonzero(arrays["record_ids"] == 7).item()
        np.testing.assert_array_equal(arrays["first_occurrence_intervals_bp"][first], [0.,1.])
        np.testing.assert_array_equal(arrays["first_occurrence_intervals_bp"][second], [4.,7.])
        self.assertEqual(arrays["positions_bp0"][first], .5)
        self.assertLessEqual(counts["actual_path_occurrences_traversed"], 96)

    def test_caps_fail_instead_of_sampling_or_truncating(self):
        with patch.dict(sys.modules, {"tskit":self.fake_tskit}):
            path_key = original.author_functions()["get_path_key"]
        for key, value in (("max_unique_paths",1), ("max_path_tokens",2),
                           ("max_path_occurrences",47), ("max_sample_unique_paths",1),
                           ("max_sample_path_tokens",2), ("max_raw_path_nodes",3)):
            with self.subTest(cap=key), self.assertRaises(ValueError):
                compact._exact_focals(self.ts, self.rows, dict(self.cfg, **{key:value}),
                                     path_key=path_key, resource_check=lambda _:None)

    def test_metadata_plan_requires_complete_inference_and_query_truth_isolation(self):
        reference = lambda name: dict(path="experiments/results/mock/"+name, sha256="a"*64)
        refs = {k:reference(k+".json") for k in ("input", "observation", "completion", "roster")}
        observed = dict(self.observation, parent_id="argformer_american_train_999", split="train",
                        sequence_length_bp=30000000, sites_and_sample_identities_verified=True,
                        migrations_and_internal_population_labels_removed=True,
                        roster_ref=refs["roster"], **{k:reference(k) for k in
                        ("tree_ref","genotypes_ref","positions_ref","alleles_ref")})
        rows = [dict(sample_index=i, observed_node_id=i, age_generations=0.,
                     sample_id="sample_%d" % (i//2), haplotype=i%2,
                     role="pure_source_reference" if i<600 else "query",
                     observed_reference_class=i//200 if i<600 else None) for i in range(1600)]
        roster = dict(parent_id=observed["parent_id"], class_order=observed["class_order"], samples=rows)
        source = dict(schema="american_fresh_inference_ready_v1", parent_id=observed["parent_id"],
                      split="train", requires_tsinfer_tsdate=False, query_truth_in_training_inputs=False,
                      truth_scoring_authorized=False, exact_paper_reproduction=False,
                      declared_conventions=compact.generation.CONVENTIONS,
                      declared_inference_conventions=compact.observation.INFERENCE_CONVENTIONS,
                      seeds=dict(sampling=92001), observation_ref=refs["observation"],
                      stage_completion_ref=refs["completion"])
        completion = dict(status="completed_current_C3_inference", observation_ref=refs["observation"])
        metadata = {refs[k]["path"]:value for k,value in
                    (("input",source),("observation",observed),("completion",completion),("roster",roster))}
        spec = dict(schema=compact.SCHEMA, input_ref=refs["input"], source_refs=[],
                    output_directory="experiments/results/mock_output/argformer_american_train_999/focals",
                    declared_focal_conventions=compact.CONVENTIONS,
                    **{k:10**8 for k in compact.CAP_KEYS})
        with patch.object(compact,"requirements",return_value=dict(source_refs=[])), \
             patch.object(compact,"_read",side_effect=lambda ref:metadata[ref["path"]]):
            plan = compact.prepare(spec)
            self.assertFalse(plan["execution_authorized"])
            self.assertEqual(len(plan["rows"]),1600)
            for broken, key, value in ((source,"query_truth_in_training_inputs",True),
                                      (observed,"sites_and_sample_identities_verified",False),
                                      (completion,"observation_ref",reference("different.json")),
                                      (rows[1599],"observed_reference_class",1)):
                previous = broken[key]
                broken[key] = value
                with self.assertRaises(ValueError):
                    compact.prepare(spec)
                broken[key] = previous

    def test_inert_import_and_private_ref_rejection(self):
        code = ("import sys; sys.path.insert(0, %r); " % str(Path(compact.__file__).resolve().parents[2])
                + "from arggnn.benchmarking import american_focals; "
                "assert not any(n in sys.modules for n in ('numpy','tskit','torch','torch_geometric'))")
        subprocess.run([sys.executable,"-B","-s","-c",code], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        with self.assertRaisesRegex(ValueError, "private/truth/oracle"):
            compact._public_ref(dict(path="experiments/results/private/truth.json",sha256="a"*64))


if __name__ == "__main__":
    unittest.main()

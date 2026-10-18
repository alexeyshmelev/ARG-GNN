"""Synthetic observed metadata and fake native runner only, with no subprocess."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import tempfile
import unittest

from arggnn.benchmarking import european_native_relate as native


ROOT = native.stages.recipes.ROOT


class NativeAPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".ap_native_tests_", dir=ROOT / "tests")
        self.base = Path(self.temporary.name)
        self.observed = self.base / "observed"
        self.observed.mkdir()
        self.parent = dict(parent_id="ancestralpaths_european_train_000", split="train",
                           seeds=dict(relate_initial=1234, relate_size_base=5678))
        self.recipe = dict(parameters=dict(
            length_bp=200000000, recombination_rate=1e-8,
            native_inference=dict(initial_haploid_effective_size=30000,
                mutation_rate=1.25e-8, population_size_iterations=5,
                final_branch_length_reestimate=True, years_per_generation=28)))
        self.roster = []
        cohorts = ("present_day", "Bronze_Age", "BAA", "Neo", "Yam",
                   "WHG", "EHG", "Ana", "CHG")
        counts = (91, 81, 7, 151, 13, 51, 43, 25, 14)
        diploid = 0
        for cohort, count in zip(cohorts, counts):
            for _ in range(count):
                for hap in (0, 1):
                    self.roster.append(dict(
                        sample_index=len(self.roster), sample_id="s" + str(diploid),
                        haplotype=hap, observed_cohort=cohort, age_generations=0,
                        role="query" if diploid < 343 else "pure_source_reference",
                        observed_reference_class=None if diploid < 343 else 0))
                diploid += 1
        rows = [dict(row, observed_node_id=row["sample_index"]) for row in self.roster]
        self.write("sample_manifest.json", json.dumps(rows))
        self.write("sample_order.txt", "\n".join(r["sample_id"] for r in rows[::2]) + "\n")
        self.write("sample_ages.generations", "0\n" * 952)
        self.write("population_labels.txt", "sample population group sex\n" + "".join(
            f'{r["sample_id"]} {r["observed_cohort"]} {r["observed_cohort"]} NA\n'
            for r in rows[::2]))
        self.write("uniform_recombination_map.txt",
                   "Position(bp)\tRate(cM/Mb)\tMap(cM)\n1\t1\t0\n200000001\t1\t200\n")
        self.write("site_coordinate_manifest.json", json.dumps(dict(
            length_bp=200000000, contig_id="22", rows=[
                dict(site_id=0, tree_position_0based=0, VCF_position_1based=1),
                dict(site_id=1, tree_position_0based=199999998,
                     VCF_position_1based=199999999)])))
        self.write("observation_manifest.json", json.dumps(dict(
            parent_id=self.parent["parent_id"], split="train",
            config_kind="ancestralpaths_european", length_bp=200000000,
            samples=952, diploids=476, query_haplotypes=686,
            pure_reference_haplotypes=266, sites=2,
            census_or_ancestry_annotations_present=False,
            cleaned_oracle_exported=False, Relate_executed=False,
            sample_template_sha256=self.digest(self.roster))))
        self.write("genotypes.vcf", "synthetic VCF handled by fake preparation only\n")
        # If a dense-grid loader is accidentally used, this invalid NPZ fails.
        self.write("scoring_coordinates.npz", "diagnostic grid deliberately not loaded")
        self.refs = self.current_refs()
        self.tools = {}
        for key in ("bin/RelateFileFormats", "scripts/RelateParallel/RelateParallel.sh",
                    "scripts/EstimatePopulationSize/EstimatePopulationSize.sh"):
            path = self.base / "release" / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fake native tool, never executed")
            self.tools[key] = str(path)
        self.read_paths = []
        self.full = SimpleNamespace(
            ROOT=ROOT, confined=lambda p: Path(p).resolve(), sha=self.sha,
            digest=self.digest, roster=lambda: self.roster, native_tools=lambda _: self.tools,
            ref=lambda p: dict(path=p, sha256=self.sha(ROOT / p)),
            text_write=lambda p, value, guard: Path(p).write_text(value),
            json_write=lambda p, value, guard: Path(p).write_text(json.dumps(value)),
            haps_check=lambda p, eligible: [r["pos1"] for r in eligible])
        self.helper = SimpleNamespace(canonical_sample_text=lambda text, names: "canonical sample")
        self.native_config = dict(native_step_headroom_bytes=1024,
                                  relate=dict(timeout_seconds_per_step=600))
        self.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def sha(self, path):
        self.read_paths.append(str(path)) if hasattr(self, "read_paths") else None
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, name, text):
        (self.observed / name).write_text(text)

    def current_refs(self):
        return {name: dict(path=str((self.observed / name).relative_to(ROOT)),
                           sha256=self.sha(self.observed / name))
                for name in native.OBSERVED_FILES}

    def prepare(self, recipe, assets, destination, **kwargs):
        destination.mkdir()
        for name in ("converted.sample", "input.haps", "input.poplabels",
                     "sample_ages.txt", "input.map"):
            (destination / name).write_text("synthetic native input")
        return [dict(pos1=p) for p in assets["position_join"]], {"retained": 2}

    def clean(self, recipe, native_file, assets, retained, destination, **kwargs):
        self.assertFalse(kwargs["include_auxiliary_grid"])
        self.assertIsInstance(retained, set)
        self.assertEqual(retained, {1, 199999999})
        destination.write_text("fake sanitized tree, not a scientific array")
        return dict(samples=952, sites=2, trees=1,
            sample_index_to_inferred_tree_node_id=[
                dict(sample_index=r["sample_index"], inferred_tree_node_id=1000 + r["sample_index"])
                for r in assets["rows"]],
            query_keys=[[r["sample_id"], r["haplotype"]]
                        for r in assets["rows"] if r["role"] == "query"])

    def step(self, argv, work, journal, label, config, guard):
        self.calls.append((label, argv))
        if label == "relate_initial":
            (work / "pre_size.anc").write_text("initial anc")
            (work / "pre_size.mut").write_text("initial mut")
        if label == "convert_post_size":
            (work / "post_size.trees").write_text("fake native converter output")
        journal.append(dict(label=label, argv=argv, returncode=0, status="complete"))
        (work / "commands.json").write_text(json.dumps({"steps": journal}))

    def run_infer(self, runner=None):
        return native.infer_parent(
            self.recipe, self.parent, self.observed, self.base / "native",
            native_config=self.native_config, expected_input_refs=self.refs,
            native_step=runner or self.step, guard=lambda _: None,
            execution_check=lambda: None, write_headroom_bytes=4096,
            max_metadata_bytes=1024 * 1024)

    def contexts(self):
        return (
            patch.object(native.stages, "_approved", return_value=200000000),
            patch.object(native.stages, "_runtime", return_value=(None, None, self.full, self.helper)),
            patch.object(native.stages, "prepare_relate_inputs", side_effect=self.prepare),
            patch.object(native.stages, "clean_converted_tree", side_effect=self.clean))

    def test_full_domain_assets_retain_all_cohorts_without_grid_or_truth_read(self):
        a, b, _, _ = self.contexts()
        with a, b:
            assets = native.load_observed_assets(
                self.recipe, self.parent, self.observed, self.refs,
                guard=lambda _: None, execution_check=lambda: None,
                write_headroom_bytes=4096, max_metadata_bytes=1024 * 1024)
        self.assertEqual(len(assets["rows"]), 952)
        self.assertEqual(len({r["observed_cohort"] for r in assets["rows"]}), 9)
        self.assertEqual(sum(r["role"] == "query" for r in assets["rows"]), 686)
        self.assertIn(199999999, assets["position_join"])
        self.assertEqual(set(assets["paths"]), set(native.OBSERVED_FILES))
        self.assertFalse(any("/truth/" in p for p in self.read_paths))

    def test_native_pipeline_preserves_commands_order_settings_and_sample_identity(self):
        a, b, c, d = self.contexts()
        with a, b, c, d:
            receipt = self.run_infer()
        self.assertEqual([label for label, _ in self.calls], [
            "convert_vcf", "annotate_observed_cohorts", "relate_initial",
            "five_population_size_iterations", "convert_post_size"])
        initial = self.calls[2][1]
        sizes = self.calls[3][1]
        self.assertEqual(initial[initial.index("-N") + 1], "30000")
        self.assertEqual(initial[initial.index("-m") + 1], "1.25e-8")
        self.assertEqual(sizes[sizes.index("--num_iter") + 1], "5")
        self.assertEqual(sizes[sizes.index("--years_per_gen") + 1], "28")
        rows = json.loads((self.base / "native/observed_sample_manifest.json").read_text())
        self.assertEqual(len(rows), 952)
        self.assertEqual(rows[0]["source_observed_node_id"], 0)
        self.assertEqual(rows[0]["observed_node_id"], 1000)
        self.assertTrue(receipt["native_six_path_endpoint_retained"])
        self.assertFalse(receipt["truth_or_oracle_read"])
        self.assertFalse(receipt["sampled_registry_prepared"])
        self.assertFalse(receipt["paper_comparison_completed"])
        self.assertEqual(receipt["operational_adapter"]["native_memory_gb"], 2)
        self.assertEqual(receipt["operational_adapter"]["explicit_released_defaults"],
                         {"threshold": 0.5, "years_per_generation": 28})
        self.assertNotIn("--noanc", sizes)

    def test_stale_2mb_map_and_changed_bound_input_rejected_before_native(self):
        self.write("uniform_recombination_map.txt",
                   "Position(bp) Rate(cM/Mb) Map(cM)\n1 1 0\n2000001 1 2\n")
        a, b, c, d = self.contexts()
        with a, b, c, d:
            with self.assertRaisesRegex(ValueError, "hashes"):
                self.run_infer()
            self.refs = self.current_refs()
            with self.assertRaisesRegex(ValueError, "complete-domain"):
                self.run_infer()
        self.assertEqual(self.calls, [])

    def test_failed_native_step_stops_without_retry_or_completed_receipt(self):
        def fail(argv, work, journal, label, config, guard):
            if label == "relate_initial":
                self.calls.append((label, argv))
                raise RuntimeError("synthetic native failure")
            self.step(argv, work, journal, label, config, guard)
        a, b, c, d = self.contexts()
        with a, b, c, d:
            with self.assertRaisesRegex(RuntimeError, "synthetic native failure"):
                self.run_infer(fail)
        self.assertEqual([v[0] for v in self.calls],
                         ["convert_vcf", "annotate_observed_cohorts", "relate_initial"])
        self.assertFalse((self.base / "native/inference.json").exists())

    def test_explicit_metadata_cap_rejects_without_cropping(self):
        a, b, _, _ = self.contexts()
        with a, b, self.assertRaisesRegex(ValueError, "read cap"):
            native.load_observed_assets(
                self.recipe, self.parent, self.observed, self.refs,
                guard=lambda _: None, execution_check=lambda: None,
                write_headroom_bytes=4096, max_metadata_bytes=8)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()

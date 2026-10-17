"""European paper planning and RAM-only AP writer tests.

No scientific libraries, scientific payloads, subprocesses or filesystem
outputs are used. The full writer uses symbolic grids and fake runtime objects.
"""
import builtins
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import PurePosixPath
import types
import unittest
from unittest.mock import Mock, patch

from arggnn.benchmarking import european_recipes as recipes


class SymbolicGrid:
    """Describe a large grid without allocating coordinates."""
    def __init__(self, expression):
        self.expression = expression

    def __getitem__(self, value):
        return SymbolicGrid(("slice", self.expression, value.start, value.stop, value.step))

    def __add__(self, other):
        return SymbolicGrid(("add", self.expression, other.expression))

    def __truediv__(self, value):
        return SymbolicGrid(("divide", self.expression, value))


class TinyVector(list):
    def astype(self, dtype):
        return TinyVector(int(value) for value in self)

    def tolist(self):
        return list(self)

    def __add__(self, value):
        return TinyVector(item + value for item in self)


class MemoryPath:
    """The writer's complete output tree lives only in this dictionary."""
    def __init__(self, store, path="output"):
        self.store, self.path = store, PurePosixPath(path)

    def __truediv__(self, value):
        return MemoryPath(self.store, self.path / value)

    def __lt__(self, other):
        return str(self.path) < str(other.path)

    def __str__(self):
        return str(self.path)

    def is_dir(self):
        return self.store.get(str(self.path)) == ("directory",)

    def is_file(self):
        return str(self.path) in self.store and not self.is_dir()

    def mkdir(self):
        if str(self.path) in self.store:
            raise FileExistsError(str(self.path))
        self.store[str(self.path)] = ("directory",)

    def iterdir(self):
        return iter(MemoryPath(self.store, name) for name in list(self.store)
                    if PurePosixPath(name).parent == self.path)

    def relative_to(self, parent):
        return self.path.relative_to(parent.path)

    def stat(self):
        return types.SimpleNamespace(st_size=len(repr(self.store[str(self.path)])))

    def open(self, mode):
        if mode != "x" or str(self.path) in self.store:
            raise AssertionError("Only fresh fake VCF writes are supported")
        path = self

        class Capture(io.StringIO):
            def __exit__(self, *args):
                path.store[str(path.path)] = self.getvalue()
                return super().__exit__(*args)
        return Capture()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class FakeRuntime:
    def __init__(self, recipe):
        self.calls, self.store = [], {"output": ("directory",)}
        self.output = MemoryPath(self.store)
        self.parameters = recipe["parameters"]
        self.population_names = [item["name"] for item in self.parameters["demography"]["populations"]]
        self.rows = []
        for cohort in self.parameters["cohorts"]:
            for _ in range(cohort["diploids"]):
                diploid = len(self.rows) // 2
                population = "present_bronze" if cohort["id"] in ("present_day", "Bronze_Age") else cohort["id"]
                for haplotype in (0, 1):
                    self.rows.append(dict(
                        sample_index=len(self.rows), sample_id=f"sample_{diploid}",
                        haplotype=haplotype, observed_cohort=cohort["id"],
                        sampling_population=population,
                        age_generations=0 if cohort["id"] == "present_day" else 120,
                        role="query" if cohort["role"] == "query" else "pure_source_reference",
                        observed_reference_class=None))
        self.truth = types.SimpleNamespace(
            samples=lambda: range(952),
            node=lambda index: types.SimpleNamespace(
                time=self.rows[index]["age_generations"],
                population=self.population_names.index(self.rows[index]["sampling_population"]),
                individual=index // 2),
            num_trees=2, nbytes=40,
            dump=lambda path: self.store.__setitem__(str(path), "FAKE_TREE_ONLY"))
        self.observed = types.SimpleNamespace(
            samples=lambda: range(952), num_sites=2,
            tables=types.SimpleNamespace(sites=types.SimpleNamespace(position=TinyVector([0, self.parameters["length_bp"] - 1]))),
            write_vcf=self.write_vcf)
        runtime = self

        class Demography:
            def __init__(self):
                self.populations, self.admixtures, self.splits = [], [], []

            def add_population(self, **item):
                self.populations.append(item)

            def add_admixture(self, **item):
                self.admixtures.append(item)

            def add_population_split(self, **item):
                self.splits.append(item)

            def add_census(self, *, time):
                self.census = time

            def sort_events(self):
                self.sorted = True

        def sample_set(count, **options):
            return dict(count=count, **options)

        def ancestry(**options):
            runtime.calls.append(("ancestry", options))
            return runtime.truth

        def mutations(clean, **options):
            runtime.calls.append(("mutations", options))
            return runtime.observed

        def arange(start, stop, step, *, dtype):
            runtime.calls.append(("grid", (start, stop, step, dtype)))
            return SymbolicGrid(("range", start, stop, step, dtype))

        def write(path, value, guard):
            guard(0)
            runtime.store[str(path)] = copy.deepcopy(value)

        simulation = types.SimpleNamespace(
            POPULATIONS=self.population_names,
            external_truth_intervals=lambda truth, rows: {
                "native_paths_1based": types.SimpleNamespace(shape=(2, 686)),
                "query_raw_node_ids": TinyVector(range(686))},
            strip_truth_annotations=lambda truth: ("FAKE_CLEAN", list(range(952))),
            assert_observed_only=lambda observed: runtime.calls.append(("observed_only", True)))
        full = types.SimpleNamespace(
            confined=lambda output: output, helpers=lambda: (simulation, None),
            roster=lambda: self.rows, npz_write=write, json_write=write, text_write=write,
            digest=canonical_digest, sha=lambda path: "f" * 64)
        msprime = types.SimpleNamespace(
            SampleSet=sample_set, Demography=Demography, sim_ancestry=ancestry,
            sim_mutations=mutations, JC69=lambda: "FAKE_JC69")
        np = types.SimpleNamespace(
            arange=arange, float64="float64", int64="int64",
            array_equal=lambda left, right: list(left) == list(right),
            asarray=lambda values, dtype: TinyVector(values))
        self.modules = (msprime, np, types.SimpleNamespace(NULL=-1), full)

    def write_vcf(self, stream, **options):
        self.calls.append(("vcf", options))
        transformed = options["position_transform"](TinyVector([0, self.parameters["length_bp"] - 1]))
        if transformed != [1, self.parameters["length_bp"]]:
            raise AssertionError("VCF coordinates must use the declared full-length domain")
        stream.write("FAKE_VCF_ONLY\n")


class RecipeMetadataTests(unittest.TestCase):
    def test_import_and_planning_do_not_import_science(self):
        original_import = builtins.__import__
        seen = []

        def limited_import(name, *args, **kwargs):
            seen.append(name)
            if name.split(".")[0] in {"msprime", "numpy", "tskit", "torch", "tensorflow"}:
                raise AssertionError("Planning must remain standard-library only")
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("european_inert_test", recipes.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=limited_import):
            spec.loader.exec_module(module)
            plan = module.prepare_recipe("ancestralpaths_european")
            module.prepare_recipe("argmix_european")
            module.parent_registry()
        self.assertTrue(seen)
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse(plan["scientific_payloads_opened"])
        self.assertEqual(plan["jobs_started"], 0)

    def test_ap_paper_scale_roster_replicates_and_native_endpoint(self):
        recipe = recipes.load_recipe("ancestralpaths_european")
        p = recipe["parameters"]
        self.assertEqual(p["length_bp"], 200000000)
        self.assertEqual((p["diploids"], p["haplotypes"], p["query_haplotypes"]), (476, 952, 686))
        self.assertEqual(sum(item["diploids"] for item in p["cohorts"]), 476)
        self.assertEqual(p["replicates"], {"train": 3, "evaluation": 5})
        self.assertEqual(p["truth_endpoint"]["primary"], "native_six_census350_paths")
        self.assertEqual(p["truth_endpoint"]["derived_C4_probability_collapse"], [[1, 5], [2, 6], [3], [4]])
        self.assertEqual(len(p["age_roster_ref"]["sha256"]), 64)
        self.assertEqual(recipe["generator"]["function"], "generate_ancestralpaths_parent")

    def test_source_refs_are_present_and_current(self):
        for recipe_id in recipes.RECIPES:
            recipes.verify_sources(recipes.load_recipe(recipe_id))

    def test_paper_precedence_clears_generation_but_keeps_training_gap(self):
        recipe = recipes.load_recipe("ancestralpaths_european")
        self.assertFalse(recipes.material_blockers(recipe))
        baa = next(item for item in recipe["parameters"]["demography"]["admixtures"]
                   if item["derived"] == "BAA")
        self.assertEqual(baa["time"], 170)
        released = next(item for item in recipe["evidence"]
                        if item["parameter"] == "demography.BAA_released_code_derivation")
        self.assertEqual(released["value"]["released_event_time_generations"], 169)
        self.assertEqual(released["agreement"], "author_code")
        self.assertEqual(len(recipes.material_blockers(recipe, stage="training")), 1)
        self.assertTrue(recipes.material_blockers(recipe, stage="training")[0].startswith("training:"))
        self.assertFalse(recipes.material_blockers(recipe, stage="inference"))

    def test_prepare_recipe_verifies_sources_and_rejects_stale_pins(self):
        with patch.object(recipes, "verify_sources", side_effect=ValueError("synthetic stale source")) as verify:
            with self.assertRaisesRegex(ValueError, "synthetic stale source"):
                recipes.prepare_recipe("ancestralpaths_european")
        verify.assert_called_once()
        stale = recipes.load_recipe("ancestralpaths_european")
        stale["source_refs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Pinned European recipe source changed"):
            recipes.verify_sources(stale)

    def test_argmix_unknowns_are_not_filled_from_prospective_reconstruction(self):
        recipe = recipes.load_recipe("argmix_european")
        p = recipe["parameters"]
        self.assertEqual((p["diploids"], p["haplotypes"]), (1611, 3222))
        self.assertIsNone(p["length_bp"])
        self.assertIsNone(p["individual_age_and_ID_roster_ref"])
        self.assertFalse(recipe["generator"]["supports_paper_parameters"])
        blockers = recipes.material_blockers(recipe)
        for field in ("length_bp", "demography", "map", "genotype_error", "individual_age_and_ID_roster_ref"):
            self.assertTrue(any(item.startswith(field + ":") for item in blockers), field)
        self.assertFalse(any("printed_plus_minus_definition:" in item or "seed_policy:" in item for item in blockers))

    def test_fresh_seed_registry_is_independent_and_inert(self):
        records = recipes.parent_registry()
        self.assertEqual([item["split"] for item in records], ["train"] * 3 + ["evaluation"] * 5)
        self.assertTrue(all("seeds" not in item for item in records))
        seeds = {item["parent_id"]: {"ancestry": 2 * i + 1, "mutations": 2 * i + 2}
                 for i, item in enumerate(records)}
        frozen = recipes.parent_registry(seeds=seeds)
        self.assertEqual(frozen[-1]["seeds"], {"ancestry": 15, "mutations": 16})
        seeds[records[-1]["parent_id"]]["ancestry"] = 1
        with self.assertRaisesRegex(ValueError, "Every fresh"):
            recipes.parent_registry(seeds=seeds)

    def test_invalid_seed_values_and_counts_are_rejected(self):
        for value in ({"ancestry": 1, "mutations": 1},
                      {"ancestry": True, "mutations": 2},
                      {"ancestry": 0, "mutations": 2},
                      {"ancestry": 2**32, "mutations": 2},
                      {"ancestry": 1}, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                recipes._seeds(value)
        for options in ({"train": 0}, {"evaluation": -1}, {"train": True}, {"train": 10000}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                recipes.parent_registry(**options)


class ConfigurableWriterTests(unittest.TestCase):
    def setUp(self):
        self.recipe = recipes.load_recipe("ancestralpaths_european")
        self.parent = dict(parent_id="ap_paper200mb_train_000", split="train",
                           seeds={"ancestry": 101, "mutations": 102})

    def invoke(self, **options):
        arguments = dict(guard=Mock(), execution_check=Mock(), write_headroom_bytes=1024,
                         require_paper_match=False)
        arguments.update(options)
        return recipes.generate_ancestralpaths_parent(self.recipe, self.parent, "RAM_ONLY", **arguments)

    def test_strict_registered_recipe_reaches_execution_check_despite_training_gap(self):
        approve = Mock(side_effect=RuntimeError("synthetic stop at execution check"))
        guard = Mock()
        with patch.object(recipes, "_load_ap_runtime") as load:
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                self.invoke(execution_check=approve, guard=guard, require_paper_match=True)
        approve.assert_called_once()
        guard.assert_not_called()
        load.assert_not_called()

    def test_registered_material_generation_conflict_still_blocks_before_execution(self):
        self.recipe["evidence"].append(dict(
            parameter="synthetic_generation_gap", value=None, agreement="unresolved",
            material=True, stages=["generation"], note="Synthetic registered gap"))
        approve, guard = Mock(), Mock()
        with patch.object(recipes, "load_recipe", return_value=copy.deepcopy(self.recipe)), \
             patch.object(recipes, "_load_ap_runtime") as load:
            with self.assertRaisesRegex(ValueError, "Paper-matched generation blocked"):
                self.invoke(execution_check=approve, guard=guard, require_paper_match=True)
        approve.assert_not_called()
        guard.assert_not_called()
        load.assert_not_called()

    def test_strict_mode_rejects_changed_parameters_or_stale_evidence_before_execution(self):
        for field in ("parameters", "generator", "evidence", "source_refs"):
            with self.subTest(field=field):
                self.recipe = recipes.load_recipe("ancestralpaths_european")
                if field == "parameters":
                    self.recipe[field]["demography"]["admixtures"][0]["time"] += 1
                elif field == "generator":
                    self.recipe[field]["function"] = "unregistered_producer"
                elif field == "evidence":
                    self.recipe[field][0]["note"] = "Unregistered evidence"
                else:
                    self.recipe[field][0]["sha256"] = "0" * 64
                approve, guard = Mock(), Mock()
                with patch.object(recipes, "_load_ap_runtime") as load:
                    with self.assertRaisesRegex(ValueError, "canonical registered"):
                        self.invoke(execution_check=approve, guard=guard, require_paper_match=True)
                approve.assert_not_called()
                guard.assert_not_called()
                load.assert_not_called()
        self.recipe = recipes.load_recipe("ancestralpaths_european")
        self.recipe["parameters"]["genotype_error_rate"] = False
        with self.assertRaisesRegex(ValueError, "canonical registered"):
            self.invoke(require_paper_match=True)

    def test_callbacks_and_write_bound_are_mandatory(self):
        for options in ({"guard": None}, {"execution_check": None},
                        {"write_headroom_bytes": 0}, {"write_headroom_bytes": True},
                        {"require_paper_match": 1}):
            with self.subTest(options=options), patch.object(recipes, "_load_ap_runtime") as load:
                with self.assertRaises(ValueError):
                    self.invoke(**options)
                load.assert_not_called()

    def test_unsupported_scientific_settings_reject_before_execution(self):
        for field, value in (("mutation_model", "binary"), ("genotype_error_rate", .001),
                             ("discrete_genome", False), ("ancestry_model", "dtwf"),
                             ("recombination_rate", float("nan")), ("mutation_rate", 0)):
            with self.subTest(field=field):
                self.recipe["parameters"][field] = value
                approve = Mock()
                with patch.object(recipes, "_load_ap_runtime") as load:
                    with self.assertRaises(ValueError):
                        self.invoke(execution_check=approve)
                approve.assert_not_called()
                load.assert_not_called()
                self.recipe = recipes.load_recipe("ancestralpaths_european")
        self.recipe["parameters"]["demography"]["census_time"] = 351
        with self.assertRaisesRegex(ValueError, "census-350"):
            self.invoke()

    def test_hold_and_guard_can_stop_before_runtime(self):
        for stop_at in ("execution_check", "guard"):
            approve, guard = Mock(), Mock()
            chosen = approve if stop_at == "execution_check" else guard
            chosen.side_effect = RuntimeError("synthetic execution hold")
            with patch.object(recipes, "_load_ap_runtime") as load:
                with self.assertRaisesRegex(RuntimeError, "execution hold"):
                    self.invoke(execution_check=approve, guard=guard)
                load.assert_not_called()
            if stop_at == "execution_check":
                guard.assert_not_called()

    def test_complete_200mb_writer_is_symbolic_and_keeps_every_length_output(self):
        runtime = FakeRuntime(self.recipe)
        events = []
        approve = lambda: events.append("approved")
        guard = lambda size: events.append(("guard", size))
        with patch.object(recipes, "verify_sources", side_effect=lambda recipe: events.append("verified")), \
             patch.object(recipes, "_load_ap_runtime", side_effect=lambda: events.append("runtime") or runtime.modules):
            receipt = recipes.generate_ancestralpaths_parent(
                self.recipe, self.parent, runtime.output, guard=guard,
                execution_check=approve, write_headroom_bytes=1024, require_paper_match=True)
        self.assertEqual(events[:4], ["approved", ("guard", 1024), "verified", "runtime"])
        ancestry = dict(runtime.calls)["ancestry"]
        self.assertEqual(ancestry["sequence_length"], 200000000)
        self.assertEqual(sum(item["count"] * item["ploidy"] for item in ancestry["samples"]), 952)
        self.assertEqual(len(ancestry["samples"]), 386)
        self.assertEqual(ancestry["samples"][0], dict(count=91, ploidy=2, population="present_bronze", time=0))
        self.assertEqual(ancestry["demography"].census, 350)
        self.assertTrue(ancestry["demography"].sorted)
        self.assertEqual(dict(runtime.calls)["grid"], (0, 200000001, 100, "float64"))
        self.assertEqual(dict(runtime.calls)["mutations"]["random_seed"], 102)
        self.assertEqual(runtime.store["output/observed/uniform_recombination_map.txt"].splitlines()[-1],
                         "200000001\t1\t200")
        for manifest in ("site_coordinate_manifest.json", "observation_manifest.json"):
            self.assertEqual(runtime.store["output/observed/" + manifest]["length_bp"], 200000000)
        self.assertEqual(len(runtime.store["output/observed/sample_manifest.json"]), 952)
        self.assertEqual(len(runtime.store["output/observed/sample_order.txt"].splitlines()), 476)
        target = runtime.store["output/truth/target_schema.json"]
        self.assertEqual(target["primary_endpoint"], "native_six_census350_paths")
        self.assertEqual(len(target["query_keys"]), 686)
        self.assertEqual(len(target["query_sample_indices"]), 686)
        self.assertTrue(target["C4_is_separately_named_derived_endpoint"])
        self.assertFalse(receipt["paper_reproduction_claimed"])
        self.assertFalse(receipt["Relate_executed"])
        self.assertFalse(receipt["material_paper_generation_conflicts"])
        self.assertEqual(receipt["sequence_length_bp"], 200000000)
        self.assertEqual(receipt["recipe_canonical_sha256"], canonical_digest(self.recipe))
        self.assertEqual(receipt["parameters_canonical_sha256"], canonical_digest(self.recipe["parameters"]))
        self.assertEqual(len(receipt["files"]), 12)
        self.assertIsInstance(runtime.store["output/observed/scoring_coordinates.npz"]["bin_edges_bp"], SymbolicGrid)

    def test_declared_length_and_rate_drive_outputs_without_source_rewriting(self):
        self.recipe["parameters"].update(length_bp=4000, recombination_rate=2e-8)
        runtime = FakeRuntime(self.recipe)
        with patch.object(recipes, "verify_sources"), \
             patch.object(recipes, "_load_ap_runtime", return_value=runtime.modules):
            receipt = recipes.generate_ancestralpaths_parent(
                self.recipe, self.parent, runtime.output, guard=Mock(),
                execution_check=Mock(), write_headroom_bytes=1024, require_paper_match=False)
        self.assertEqual(dict(runtime.calls)["ancestry"]["sequence_length"], 4000)
        self.assertEqual(dict(runtime.calls)["grid"], (0, 4001, 100, "float64"))
        self.assertEqual(runtime.store["output/observed/uniform_recombination_map.txt"].splitlines()[-1],
                         "4001\t2\t0.008")
        self.assertEqual(runtime.store["output/observed/observation_manifest.json"]["length_bp"], 4000)
        self.assertEqual(receipt["sequence_length_bp"], 4000)

    def test_nonempty_output_rejects_before_simulation(self):
        runtime = FakeRuntime(self.recipe)
        runtime.store["output/unexpected"] = "metadata"
        with patch.object(recipes, "verify_sources"), \
             patch.object(recipes, "_load_ap_runtime", return_value=runtime.modules):
            with self.assertRaisesRegex(ValueError, "fresh empty"):
                recipes.generate_ancestralpaths_parent(
                    self.recipe, self.parent, runtime.output, guard=Mock(),
                    execution_check=Mock(), write_headroom_bytes=1024, require_paper_match=False)
        self.assertFalse(runtime.calls)


if __name__ == "__main__":
    unittest.main()

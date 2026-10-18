"""Fresh CPU American benchmark generation over the pinned numerical simulator.

Paper-stated model, length, rates and population counts are preserved. Haploid
sample units, JC69/discrete mutations and recorded-pulse truth are explicitly
declared reproducible conventions where the paper is silent. These outputs are
fresh shared data, never relabelled historical C3 parents or exact paper bytes.

ROOT's admitted CPU worker calls run(spec, resource_check=..., execution_check=...).
No process, numerical import, GPU, historical accounting or work starts at import.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIMULATION_MODULE = "arggnn.american_c3_simulate_v1"
SOURCE_REFS = (
    {"path": "src/arggnn/american_c3_simulate_v1.py",
     "sha256": "d6a6d8a704589b1bfc337eacec21c98fe5df351fcbd9bb1b21485c6274f1b2c5"},
    {"path": "src/arggnn/american_c3_common_v1.py",
     "sha256": "615d3892f1df7e9de39e57991eab883b2eca1af9d13e424661541aac4b2d2abf"},
    {"path": ".runtime/env/lib/python3.11/site-packages/stdpopsim/catalog/HomSap/demographic_models.py",
     "sha256": "ca1adb03f251b7fc293323ef9fe4e77ec9e705a9ad38c21382946bed7c791e1c"},
)
CONVENTIONS = {
    "sample_unit": "haploid", "mutation_model": "JC69", "discrete_genome": True,
    "truth_rule": "recorded_admixture_pulse12_lineage_destination",
    "genealogy_predictor_input": "requires_shared_tsinfer_tsdate_not_simulated_true_topology",
}
SPLITS = {"train": "train", "validation": "val", "test": "test"}
CAP_KEYS = ("max_sites", "max_tables_bytes", "max_truth_intervals",
            "max_output_bytes", "max_seconds", "max_rss_bytes")


def _path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if ".." in path.parts or not path.is_relative_to(ROOT) or path.resolve() != path:
        raise ValueError("Canonical nonsymlink repository path required")
    return path


def _reference(path):
    path = _path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest.hexdigest()}


def _read(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Exact immutable metadata reference required")
    path = _path(ref["path"])
    if not path.is_file() or path.stat().st_size > 2 * 2**20 or _reference(path) != ref:
        raise ValueError("Missing, oversized or changed campaign metadata")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected bounded metadata object")
    return value


def _verify_sources():
    for ref in SOURCE_REFS:
        if _reference(ref["path"]) != ref:
            raise ValueError("American numerical source changed: " + ref["path"])


def prepare(spec):
    """Validate one frozen campaign task and return an inert exact config.

    All 35 simulation identities/streams are checked before selecting one.
    There is no truth-file or historical-data reader.
    """
    if spec.get("schema") != "american_fresh_preparation_v1":
        raise ValueError("Named fresh American preparation spec required")
    if json.dumps(spec.get("declared_conventions"), sort_keys=True, allow_nan=False) != json.dumps(CONVENTIONS, sort_keys=True):
        raise ValueError("Explicit current benchmark conventions must be frozen in the spec")
    if any(type(spec.get(k)) is not int or spec[k] <= 0 for k in CAP_KEYS):
        raise ValueError("Finite positive full-parent count/output/RSS/wall caps required")
    campaign = _read(spec["campaign_ref"])
    design = _read(spec["comparison_ref"])
    recipe = _read(spec["recipe_ref"])
    if (campaign.get("schema") != "benchmark_campaign_v1"
            or design.get("schema") != "paper_comparison_design_v1"
            or design.get("recipe_id") != "argformer_american"
            or design.get("split_counts") != {"train": 20, "validation": 5, "test": 10}
            or recipe.get("id") != "argformer_american"):
        raise ValueError("Exact fresh 20 TRAIN / 5 validation / 10 TEST design required")
    simulation = recipe["parameters"]["simulation"]
    known = dict(species="HomSap", demographic_model="AmericanAdmixture_4B18",
                 sequence_length_bp=30000000, mutation_rate_per_bp_generation=1.25e-8,
                 recombination_rate_per_bp_generation=1e-8, genotype_error=0, phasing_error=0,
                 sample_counts_as_written={"AFR": 200, "EUR": 200, "ASIA": 200, "ADMIX": 1000})
    if any(simulation.get(k) != v for k, v in known.items()):
        raise ValueError("Paper-stated American biological parameters changed")
    tasks = [row for row in campaign["tasks"]
             if row.get("recipe") == "argformer_american" and row.get("stage") == "simulate_once"]
    parents, ids, streams, seeds = set(), set(), set(), []
    counts = dict.fromkeys(SPLITS, 0)
    for task in tasks:
        split, parent = task["split"], task["shared_parent_id"]
        if (split not in SPLITS or parent in parents or task["id"] in ids
                or task["simulation_stream"] in streams
                or not parent.startswith("argformer_american_" + split + "_")
                or task.get("methods") != ["argformer", "flare", "treeloom"]
                or set(task["seeds"]) != {"ancestry", "mutations", "observation", "sampling"}):
            raise ValueError("Independent shared parent/split/stream identities required")
        values = list(task["seeds"].values())
        if any(type(value) is not int or not 0 < value < 2**32 for value in values):
            raise ValueError("Fresh positive 32-bit streams required")
        parents.add(parent); ids.add(task["id"]); streams.add(task["simulation_stream"])
        seeds.extend(values); counts[split] += 1
    if counts != design["split_counts"] or len(seeds) != len(set(seeds)):
        raise ValueError("Exact independent split counts and globally distinct streams required")
    selected = [row for row in tasks if row["id"] == spec["task_id"]]
    if len(selected) != 1:
        raise ValueError("Exactly one frozen fresh simulation task required")
    task = selected[0]
    output = _path(spec["output_directory"])
    if (not output.is_relative_to(ROOT / "experiments/results") or output == ROOT / "experiments/results"
            or task["shared_parent_id"] not in output.parts or output.exists()):
        raise ValueError("Fresh parent-specific isolated experiments/results output required")
    config = dict(
        schema="american_c3_simulation_reconstruction_v1", execution_enabled=True,
        species=known["species"], model=known["demographic_model"], sequence_length_bp=30000000,
        mutation_rate=1.25e-8, recombination_rate=1e-8, admixture_generation=12.,
        sample_unit=CONVENTIONS["sample_unit"], sample_counts=known["sample_counts_as_written"],
        class_order=["EUR", "EAS", "AFR"], parent_id=task["shared_parent_id"], split=SPLITS[task["split"]],
        ancestry_seed=task["seeds"]["ancestry"], mutation_seed=task["seeds"]["mutations"],
        max_sites=spec["max_sites"], max_tables_bytes=spec["max_tables_bytes"],
        max_truth_intervals=spec["max_truth_intervals"], output_cap_bytes=spec["max_output_bytes"],
        interpretation="Fresh shared benchmark using paper-stated biology and explicitly declared unspecified conventions; exact published numerical reproduction not claimed")
    _verify_sources()
    return dict(config=config, task=copy.deepcopy(task), output_directory=str(output),
                paper_known_parameters=known, declared_conventions=copy.deepcopy(CONVENTIONS),
                exact_paper_reproduction=False, execution_authorized=False)


class _Writer:
    """The existing simulator's I/O interface with an exclusive bounded scope."""
    def __init__(self, path, cap, check):
        self.path, self.cap, self.check = path, cap, check
        self.pending, self.finished = {}, 0
        check("american_generation_create_scope")
        path.mkdir(parents=True, exist_ok=False)

    def prepare(self, name, planned):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Producer output must stay in its isolated scope")
        if relative.parts[0] == "truth":
            relative = Path("private") / relative
        elif name == "completed.json":
            relative = Path("private/generation_completed.json")
        path = _path(self.path / relative)
        if (type(planned) is not int or planned < 0 or path.exists() or path in self.pending
                or self.finished + sum(self.pending.values()) + planned > self.cap):
            raise ValueError("Exclusive whole-scope output cap exceeded; no truncation or retry")
        self.check("american_generation_prepare_write")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pending[path] = planned
        return path

    def finish(self, path, planned):
        size = path.stat().st_size
        if self.pending.get(path) != planned or size > planned:
            raise ValueError("Actual serialization exceeded its declared count bound")
        self.check("american_generation_finish_write")
        del self.pending[path]
        self.finished += size
        return _reference(path)

    def json(self, name, value):
        data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        path = self.prepare(name, len(data))
        with path.open("xb") as stream:
            stream.write(data)
        return self.finish(path, len(data))

    def npy(self, name, array):
        np = importlib.import_module("numpy")
        if array.dtype.hasobject:
            raise ValueError("No object arrays")
        planned = int(array.nbytes) + 4096
        path = self.prepare(name, planned)
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        return self.finish(path, planned)

    def npz(self, name, **arrays):
        np = importlib.import_module("numpy")
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("No object arrays")
        planned = sum(int(array.nbytes) for array in arrays.values()) + 65536
        path = self.prepare(name, planned)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        return self.finish(path, planned)


def run(spec, *, resource_check, execution_check):
    """Generate one full fresh parent once inside ROOT's supervised CPU worker.

    resource_check enforces admitted current storage/RSS/deadline at boundaries;
    the outer CPU supervisor must also enforce wall/RSS while msprime is busy.
    The function launches no process and supplies no automatic retry.
    """
    if not callable(resource_check) or not callable(execution_check):
        raise ValueError("Actual caller admission and resource checks required")
    plan = prepare(spec)
    execution_check()
    import os
    if Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit CPU-only repository worker required")
    resource_check("american_generation_before_scientific_import")
    simulation = importlib.import_module(SIMULATION_MODULE)
    config = plan["config"]
    counts = dict(config["sample_counts"])
    if simulation.storage_bound(config, counts) > spec["max_output_bytes"]:
        raise ValueError("Full declared production bounds exceed the admitted output cap")
    writer = _Writer(_path(plan["output_directory"]), spec["max_output_bytes"], resource_check)
    completion_ref = simulation.execute(config, writer)
    resource_check("american_generation_complete")
    completed = _read(completion_ref)
    if (completed.get("status") != "completed_C3_simulation_reconstruction"
            or completed.get("parent_id") != config["parent_id"] or writer.pending):
        raise ValueError("Genuine complete fresh simulation receipt required")
    # Publish only observed input references. In particular, TEST truth references
    # stay in private/generation_completed.json and never enter a training input.
    public = dict(
        schema="american_fresh_shared_observation_ready_v1", task_id=spec["task_id"],
        parent_id=config["parent_id"], split=config["split"],
        observation_ref=completed["observation_ref"], requires_tsinfer_tsdate=True,
        simulated_true_topology_is_predictor_input=False,
        query_truth_in_training_inputs=False, exact_paper_reproduction=False,
        paper_known_parameters=plan["paper_known_parameters"],
        declared_conventions=plan["declared_conventions"],
        campaign_ref=spec["campaign_ref"], comparison_ref=spec["comparison_ref"], recipe_ref=spec["recipe_ref"],
        source_refs=list(SOURCE_REFS), seeds=plan["task"]["seeds"],
        sites=completed["sites"], truth_scoring_authorized=False,
        count_limit_failure_policy="fail_whole_parent_without_crop_redraw_or_reuse")
    return writer.json("observation_ready.json", public)

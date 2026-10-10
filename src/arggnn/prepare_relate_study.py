"""Prepare hash-bound paired Relate inputs for a train/validation-only study.

Every parent is processed. Nonzero inference statuses fail the panel instead of
silently dropping parents. Truth is hashed for audit binding, never loaded into
inferred model inputs. Inference and tensor preparation are sequential CPU work.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from .resources import ROOT, project_path, require_storage, set_research_title

BENCHMARK = "ancient-europe-stdpopsim-0.3.0-relate-inferred"
SEED_SCHEME = "SHA256 UTF-8 ARG-GNN/Relate/v1.2.4/{master_seed}/{split}/{replicate}; first 8 bytes unsigned big endian modulo (2**31-1), plus 1"
COMMON_CHOICES = (
    "master_seed", "simulation", "splits", "reference_adapter", "windows", "models",
    "hidden_dim", "num_layers", "epochs", "patience", "learning_rate",
    "weight_decay", "dropout", "selection", "comparison_note",
)


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def array_sha(value: Any) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def inference_seed(master_seed: int, split: str, replicate: int) -> int:
    if split not in {"train", "val"} or replicate < 0:
        raise ValueError("Only train/validation parent identities are permitted")
    key = f"ARG-GNN/Relate/v1.2.4/{master_seed}/{split}/{replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31 - 1) + 1


def query_keys(case: Any) -> list[list[Any]]:
    rows = {int(row["observed_node_id"]): row for row in case.metadata["samples"]}
    keys = []
    for node in case.sequence.query_node_ids.tolist():
        row = rows[int(node)]
        if row["cohort_role"] != "query" or row.get("reference_class") is not None:
            raise RuntimeError("Query identity includes a reference ancestry annotation")
        keys.append([row["sample_id"], row["haplotype"]])
    return keys


def validate_protocol(settings: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding = settings["source_parent"]
    config_path = project_path(binding["config"])
    manifest_path = project_path(binding["manifest"])
    if sha(config_path) != binding["config_sha256"] or sha(manifest_path) != binding["manifest_sha256"]:
        raise RuntimeError("Source parent config or manifest integrity failure")
    source_settings = json.loads(config_path.read_text())
    source = json.loads(manifest_path.read_text())
    if source["protocol_sha256"] != binding["config_sha256"] or source["protocol"] != source_settings:
        raise RuntimeError("Source parent manifest is not bound to its frozen protocol")
    if settings["benchmark"] != BENCHMARK or source_settings["benchmark"] != binding["benchmark"]:
        raise ValueError("Unexpected inferred or source benchmark family")
    for choice in COMMON_CHOICES:
        if settings[choice] != source_settings[choice]:
            raise ValueError(f"Paired protocol changes frozen common choice: {choice}")
    if settings["splits"] != {"train": 16, "val": 4, "test": 0}:
        raise ValueError("This inferred pilot permits exactly 16 train and 4 validation parents")
    inference = settings["inference"]
    if (inference["method"], inference["version"], inference["seed_scheme"]) != ("Relate", "1.2.4", SEED_SCHEME):
        raise ValueError("Inference method or seed scheme differs from the pinned protocol")
    if inference["threads"] != 4 or inference["coordinate_offset"] != 1.:
        raise ValueError("Pinned inference uses four CPU threads and the official converter offset")
    records = source["replicates"]
    expected = {(split, replicate) for split in ("train", "val") for replicate in range(settings["splits"][split])}
    identities = [(record["split"], record["replicate"]) for record in records]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise RuntimeError("Parent panel identity mismatch or test parents present")
    seeds = [inference_seed(inference["master_seed"], *identity) for identity in identities]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Independent per-parent inference seed collision")
    return source, sorted(records, key=lambda r: (0 if r["split"] == "train" else 1, r["replicate"]))


def parent_binding(record: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    directory = project_path(record["directory"])
    for path_key, hash_key, expected_name in (
        ("metadata", "metadata_sha256", "metadata.json"),
        ("tensor_file", "tensor_sha256", "tensors.pt.gz"),
        ("prepared_tensor_file", "prepared_tensor_sha256", "prepared.pt"),
    ):
        path = project_path(record[path_key])
        if path != (directory / expected_name).resolve() or sha(path) != record[hash_key]:
            raise RuntimeError(f"Parent {path_key} directory/hash binding failed")
    # Bind the external target's source artifact without reading its genealogies.
    truth = directory / "truth.trees.gz"
    observed = directory / "observed.trees.gz"
    return {
        "directory": str(directory), "split": record["split"], "replicate": record["replicate"],
        "seeds": record["seeds"], "metadata_sha256": record["metadata_sha256"],
        "tensor_sha256": record["tensor_sha256"],
        "prepared_tensor_sha256": record["prepared_tensor_sha256"],
        "truth_archive_sha256": sha(truth), "clean_observed_archive_sha256": sha(observed),
        "source_protocol_sha256": settings["source_parent"]["config_sha256"],
        "source_manifest_sha256": settings["source_parent"]["manifest_sha256"],
    }


def verify_pair(parent: Any, case: Any) -> dict[str, Any]:
    import torch
    indexes = torch.tensor(case.metadata["retained_parent_grid_indexes"], dtype=torch.long)
    if indexes.ndim != 1 or len(indexes) == 0 or not bool((indexes[1:] > indexes[:-1]).all()):
        raise RuntimeError("Retained parent grid indexes must be nonempty and strictly ordered")
    if not torch.equal(case.labels, parent.labels[indexes]):
        raise RuntimeError("Inferred targets differ from paired source targets")
    if not torch.equal(case.sequence.positions, parent.sequence.positions[indexes]):
        raise RuntimeError("Inferred scoring coordinates differ from paired source coordinates")
    if query_keys(case) != query_keys(parent):
        raise RuntimeError("Inferred query order differs from the paired source order")
    for identity in ("split", "replicate", "seeds", "master_seed"):
        if case.metadata[identity] != parent.metadata[identity]:
            raise RuntimeError(f"Paired parent identity changed: {identity}")
    return {
        "targets_equal": True, "positions_equal": True, "query_order_equal": True,
        "parent_target_sha256": array_sha(parent.labels),
        "retained_target_sha256": array_sha(case.labels),
        "retained_positions_sha256": array_sha(case.sequence.positions),
        "query_haplotype_keys": query_keys(case), "retained_parent_grid_indexes": indexes.tolist(),
    }


def verify_loaded_panel(manifest: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .simulate import load_replicate, split_fixed_windows
    from .train_study import load_cases
    records = []
    counts = {}
    for split in ("train", "val"):
        cases = load_cases(manifest, split)
        counts[split] = len(cases)
        sources = {r["replicate"]: r for r in manifest["replicates"] if r["split"] == split}
        for case in cases:
            record = sources[case.metadata["replicate"]]
            parent = load_replicate(record["source_parent"]["directory"])
            alignment = verify_pair(parent, case)
            tiles = split_fixed_windows(case, manifest["protocol"]["windows"])
            if not torch.equal(torch.cat([t.labels for t in tiles]), case.labels):
                raise RuntimeError("Fixed tiles do not retain every paired target in original order")
            if not torch.equal(torch.cat([t.sequence.positions for t in tiles]), case.sequence.positions):
                raise RuntimeError("Fixed tiles change original global grid coordinates")
            if list(Path(record["directory"]).glob("truth.trees*")):
                raise RuntimeError("A raw truth genealogy leaked into the inferred case directory")
            for tile in tiles:
                tile.sequence.validate()
            records.append({"split": split, "replicate": case.metadata["replicate"],
                            "prepared_feature_dim": case.sequence.x.shape[1], "fixed_tiles": len(tiles),
                            "tile_bounds": [t.metadata["window_bounds"] for t in tiles],
                            "global_tile_bounds": [t.metadata["global_window_bounds"] for t in tiles],
                            **alignment})
    if counts != {"train": 16, "val": 4}:
        raise RuntimeError("Loaded panel split counts differ from the frozen protocol")
    return {"loader": "arggnn.train_study.load_cases", "train_parents": counts["train"],
            "validation_parents": counts["val"], "test_parents": 0,
            "all_raw_metadata_tensor_and_adapter_hashes_verified": True,
            "all_retained_target_query_coordinate_pairs_exact": True,
            "all_fixed_tiles_cover_retained_coordinates_once": True,
            "inferred_case_truth_genealogies_present": False, "replicates": records}


def prepare_panel(config_path: Path, target: Path) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    set_research_title()
    import numpy as np
    import torch
    import tskit
    from .reference_encoder import ReferenceFeatureAdapter
    from .relate_pipeline import (
        COPYRIGHT_NOTICE, RelateConfig, load_ancient_parent, prepare_relate_inputs,
        run_paired_relate, verified_relate_installation,
    )
    torch.set_num_threads(4)
    settings = json.loads(config_path.read_text())
    _, source_records = validate_protocol(settings)
    installation = verified_relate_installation()
    if installation["archive_sha256"] != settings["inference"]["archive_sha256"]:
        raise RuntimeError("Inference tool archive differs from the frozen protocol")
    if target.exists():
        raise RuntimeError("Use a fresh versioned inferred dataset directory")
    require_storage(headroom_bytes=500_000_000)
    # Verify every parent before the first native inference run.
    bindings = [parent_binding(record, settings) for record in source_records]
    target.mkdir(parents=True)
    started = time.monotonic()
    code_paths = [Path(__file__), *[ROOT / "src" / "arggnn" / name for name in
        ("relate_pipeline.py", "reference_encoder.py", "simulate.py", "data.py", "train_study.py")]]
    manifest = {
        "schema_version": 2, "protocol": settings, "protocol_sha256": sha(config_path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty_at_generation": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "generation_code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code_paths},
        "source_parent": settings["source_parent"], "relate_installation": installation,
        "versions": {"python": __import__("sys").version, "torch": torch.__version__,
                     "numpy": np.__version__, "tskit": tskit.__version__},
        "copyright_notice": COPYRIGHT_NOTICE, "replicates": [],
    }
    for source_record, binding in zip(source_records, bindings):
        require_storage(headroom_bytes=500_000_000)
        split, replicate = source_record["split"], source_record["replicate"]
        replicate_started = time.monotonic()
        parent, observed = load_ancient_parent(binding["directory"])
        if parent.metadata["seeds"] != source_record["seeds"] or parent.metadata["replicate"] != replicate or parent.metadata["split"] != split:
            raise RuntimeError("Source parent identity does not match verified record")
        inference = settings["inference"]
        for name in ("mutation_rate", "recombination_rate"):
            if parent.metadata["config"][name] != inference[name]:
                raise RuntimeError(f"Inference {name} differs from the paired source setting")
        directory = target / split / f"rep{replicate:03d}"
        prep = prepare_relate_inputs(parent, observed, directory / "inputs",
                                    min_mac=inference["min_mac"], recombination_rate=inference["recombination_rate"])
        seed = inference_seed(inference["master_seed"], split, replicate)
        native_config = RelateConfig(**{name: inference[name] for name in
            ("mutation_rate", "effective_haploid_size", "memory_gb", "timeout_seconds", "threads")}, seed=seed)
        case = run_paired_relate(parent, prep, directory / "inference", config=native_config)
        alignment = verify_pair(parent, case)
        observation = case.tree_sequence
        if (observation.tables.nodes.metadata.size or observation.tables.edges.metadata.size or
            observation.tables.nodes.population.max() != tskit.NULL or observation.num_migrations):
            raise RuntimeError("Inferred observation retained internal ancestry annotations")
        raw = directory / "inference" / "dataset"
        tensor_path, metadata_path = raw / "tensors.pt.gz", raw / "metadata.json"
        # Native runner saves raw tensors once. Finalize protocol/provenance metadata
        # before hashing it, without ever writing a source truth tree here.
        case.metadata.update({
            "benchmark": BENCHMARK, "parent_benchmark": parent.metadata["benchmark"],
            "source_parent": binding, "paired_alignment": alignment,
            "inference_configuration": asdict(native_config), "inference_seed": seed,
            "inference_seed_scheme": inference["seed_scheme"],
            "inference_master_seed": inference["master_seed"], "protocol_sha256": manifest["protocol_sha256"],
            "grid_switch_counts_per_query": (case.labels[1:] != case.labels[:-1]).sum(0).tolist(),
            "grid_class_counts": torch.bincount(case.labels.flatten(), minlength=case.metadata["num_classes"]).tolist(),
        })
        case.metadata["files"] = {p.name: p.stat().st_size for p in raw.iterdir() if p.is_file() and p.name != "metadata.json"}
        write_json(metadata_path, case.metadata)
        tensor_hash = sha(tensor_path)
        adapter = ReferenceFeatureAdapter(case.sequence.x.shape[1], **settings["reference_adapter"])
        prepared = adapter.precompute(case.sequence)
        cached = raw / "prepared.pt"
        provenance = adapter.manifest_provenance(case.metadata["feature_names"], tensor_hash)
        torch.save({"sequence": prepared.as_dict(), "adapter": provenance}, cached)
        statuses = [step["returncode"] for step in case.metadata["inference_provenance"]["steps"]]
        if statuses != [0, 0]:
            raise RuntimeError("Successful parent requires both native inference return codes zero")
        retained = torch.tensor(alignment["retained_parent_grid_indexes"], dtype=torch.long)
        record = {
            "directory": str(raw), "metadata": str(metadata_path), "metadata_sha256": sha(metadata_path),
            "tensor_file": str(tensor_path), "tensor_sha256": tensor_hash,
            "prepared_tensor_file": str(cached), "prepared_tensor_sha256": sha(cached),
            "adapter_configuration_sha256": provenance["configuration_sha256"],
            "split": split, "replicate": replicate, "seeds": parent.metadata["seeds"],
            "source_parent": binding, "inference_seed": seed, "inference_configuration": asdict(native_config),
            "observed_tree_file": str(raw / "observed.trees.gz"), "observed_tree_sha256": sha(raw / "observed.trees.gz"),
            "native_inferred_tree_sha256": sha(directory / "inference" / "inferred.trees"),
            "preparation_sha256": sha(directory / "inputs" / "preparation.json"),
            "execution_sha256": sha(directory / "inference" / "execution.json"),
            "true_switches": parent.metadata["total_true_switches"],
            "grid_switches": sum(case.metadata["grid_switch_counts_per_query"]),
            "parent_grid_points": case.metadata["parent_grid_points"], "retained_grid_points": len(retained),
            "physical_coverage_bp": case.metadata["coverage_physical_bp"],
            "physical_coverage_fraction": case.metadata["physical_coverage_fraction"],
            "grid_coverage_fraction": case.metadata["grid_coverage_fraction"], "paired_alignment": alignment,
            "exported_sites": prep["num_sites"], "site_filter_counts": prep["filter_counts"],
            "inferred_trees": case.sequence.num_graphs, "packed_nodes": len(case.sequence.x),
            "native_returncodes": statuses,
            "inference_elapsed_seconds": sum(step["elapsed_seconds"] for step in case.metadata["inference_provenance"]["steps"]),
            "preparation_elapsed_seconds": time.monotonic() - replicate_started,
            "inferred_nearest_tmrca_accuracy": float((case.baseline_probabilities.argmax(-1) == case.labels).float().mean()),
            "paired_oracle_nearest_tmrca_accuracy": float((parent.baseline_probabilities[retained].argmax(-1) == case.labels).float().mean()),
            "bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
        }
        manifest["replicates"].append(record)
        write_json(target / "manifest.partial.json", manifest)
        print(json.dumps({"event": "paired_inferred_replicate_ready", "split": split, "replicate": replicate,
            "inference_seed": seed, "inferred_trees": record["inferred_trees"], "exported_sites": prep["num_sites"],
            "retained_grid_points": len(retained), "parent_grid_points": record["parent_grid_points"],
            "physical_coverage_fraction": record["physical_coverage_fraction"],
            "native_returncodes": statuses, "elapsed_seconds": record["preparation_elapsed_seconds"]}), flush=True)
    verification = verify_loaded_panel(manifest)
    write_json(target / "verification.json", verification)
    manifest["verification_file"] = str(target / "verification.json")
    manifest["verification_sha256"] = sha(target / "verification.json")
    records = manifest["replicates"]
    manifest["summary"] = {
        "train_parents": 16, "validation_parents": 4, "test_parents": 0,
        "all_native_returncodes_zero": all(r["native_returncodes"] == [0, 0] for r in records),
        "total_retained_grid_points": sum(r["retained_grid_points"] for r in records),
        "total_parent_grid_points": sum(r["parent_grid_points"] for r in records),
        "min_physical_coverage_fraction": min(r["physical_coverage_fraction"] for r in records),
        "mean_physical_coverage_fraction": float(np.mean([r["physical_coverage_fraction"] for r in records])),
        "min_grid_coverage_fraction": min(r["grid_coverage_fraction"] for r in records),
        "total_inference_elapsed_seconds": sum(r["inference_elapsed_seconds"] for r in records),
        "total_preparation_elapsed_seconds": time.monotonic() - started,
        "data_bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file()),
    }
    write_json(target / "manifest.json", manifest)
    (target / "manifest.partial.json").unlink()
    print(json.dumps({"event": "inferred_dataset_complete", "manifest": str(target / "manifest.json"),
                      "protocol_sha256": manifest["protocol_sha256"], **manifest["summary"],
                      "storage_bytes": require_storage()}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_panel(project_path(args.config), project_path(args.output))


if __name__ == "__main__":
    main()

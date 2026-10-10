"""Integrity-bound age-operator sidecars for verified observed-only boundary tiles."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import platform
import torch
from .resources import ROOT, project_path
from .boundary_dataset import (load_boundary_dataset, sha256_file, StorageAdmission,
    _digest, _json_bytes, _torch_bytes)
from .simulate import split_fixed_windows
from .train_study import load_cases

SCHEMA_VERSION = 1
DEPENDENCIES = ("age_dataset.py", "age_interface.py", "kron_interface.py",
    "boundary_dataset.py", "models.py", "simulate.py", "train_study.py",
    "data.py", "reference_encoder.py", "genealogy_baselines.py")

def implementation_hashes():
    return {name: sha256_file(ROOT/"src/arggnn"/name) for name in DEPENDENCIES}

def _require_equal(actual, expected, message):
    if actual != expected:
        raise RuntimeError(message)

def _source_tiles(dataset):
    source = json.loads(Path(dataset.manifest["source"]["manifest_path"]).read_text())
    parents = {c.parent_id: c for c in dataset.parent_cases}
    for split in dataset.manifest["preparation"]["splits"]:
        for case in load_cases(source, split):
            identity = f'{dataset.manifest["source"]["manifest_sha256"]}:{split}:{case.metadata["replicate"]}'
            parent = parents[identity]
            verified = {tile.metadata["window_index"]: tile for tile in parent.tiles}
            for case_tile in split_fixed_windows(case, dataset.manifest["preparation"]["windows"]):
                tile = verified[case_tile.metadata["window_index"]]
                if not torch.equal(case_tile.sequence.positions, tile.bank.positions):
                    raise RuntimeError("Source/verified tile coordinate mismatch")
                yield tile, case_tile.sequence

def prepare_age_dataset(boundary_directory, output_directory, *,
        expected_boundary_manifest_sha256, expected_boundary_preparation_sha256,
        age_scales=(350., 1400.), strength=3., progress=None):
    from .age_interface import precompute_age_interfaces
    output = project_path(output_directory)
    if output.exists():
        raise RuntimeError("Choose a fresh versioned age sidecar directory")
    dataset = load_boundary_dataset(boundary_directory,
        expected_manifest_sha256=expected_boundary_manifest_sha256,
        expected_preparation_sha256=expected_boundary_preparation_sha256)
    code = implementation_hashes()
    settings = {"age_scales": list(age_scales), "strength": float(strength),
        "normalization": "shared_plain_mean_boundary_diagonal",
        "labels_in_cache": False, "cache_dtype": "float64",
        "source_geometry": "exact observed clipped sequence, unchanged plain bank"}
    digest = _digest(settings)
    records = {r["tile_id"]: r for r in dataset.records}
    manifest = {"schema_version": SCHEMA_VERSION,
        "boundary_directory": str(dataset.directory),
        "boundary_manifest_sha256": dataset.manifest_sha256,
        "boundary_preparation_sha256": dataset.preparation_sha256,
        "source_manifest_sha256": dataset.manifest["source"]["manifest_sha256"],
        "preparation": settings, "preparation_sha256": digest,
        "implementation_sha256": code,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "records": []}
    storage = StorageAdmission()
    storage.admit(100_000_000)
    output.mkdir(parents=True)
    storage.write(output/"preparation.json", _json_bytes({**manifest, "status": "incomplete"}))
    preparation = dataset.manifest["preparation"]
    for tile, sequence in _source_tiles(dataset):
        _require_equal(implementation_hashes(), code, "Age preparation implementation changed")
        cache = precompute_age_interfaces(sequence, plain_bank=tile.bank,
            time_column=preparation["time_column"],
            parent_branch_column=preparation["parent_branch_column"],
            time_scale=preparation["time_scale"],
            resistance_scale=preparation["resistance_scale"],
            feature_columns=preparation["feature_columns"],
            max_condition=preparation["max_condition"],
            age_scales=age_scales, strength=strength,
            reference_ids=tile.bank.query_cache.reference_ids).operator_cache
        record = records[tile.provenance["tile_id"]]
        relative = Path(record["bank_file"]).with_name("age_operators.pt")
        provenance = {"tile_id": record["tile_id"],
            "plain_bank_sha256": record["bank_sha256"],
            "preparation_sha256": digest}
        storage.refresh()
        info = storage.write(output/relative, _torch_bytes({"schema_version": SCHEMA_VERSION,
            "operator_cache": cache.as_dict(), "provenance": provenance}))
        manifest["records"].append({**provenance, "operator_file": str(relative),
            "operator_sha256": info["sha256"], "operator_bytes": info["bytes"]})
        if progress:
            progress({"event": "age_tile_ready", "tile_id": record["tile_id"],
                "bytes": info["bytes"]})
    _require_equal(set(r["tile_id"] for r in manifest["records"]), set(records),
        "Age sidecars do not cover verified boundary tiles")
    _require_equal(implementation_hashes(), code, "Age source changed before sealing")
    _require_equal(sha256_file(dataset.directory/"manifest.json"),
        dataset.manifest_sha256, "Boundary manifest changed before sealing")
    storage.write(output/"manifest.json", _json_bytes({**manifest, "status": "complete"}))
    return manifest

def _check_partition(cache, bank, age_scales):
    plain = bank.query_cache
    if cache.normalization != "shared_plain_mean_boundary_diagonal":
        raise RuntimeError("Age normalization mismatch")
    for name in ("query_ids", "reference_ids", "query_indices", "reference_indices"):
        if not torch.equal(getattr(cache, name), getattr(plain, name)):
            raise RuntimeError("Age partition identity/order mismatch")
    if not torch.equal(cache.normalization_scale, plain.normalization_scale):
        raise RuntimeError("Age normalization does not use unchanged plain scales")
    if not torch.equal(cache.cqq[:,0], plain.cqq) or not torch.equal(cache.cqr[:,0], plain.cqr):
        raise RuntimeError("Age plain channel differs from source bank")
    if tuple(cache.operator_scales) != (0., *age_scales):
        raise RuntimeError("Age scale ordering mismatch")

def load_age_caches(directory, dataset, *, expected_manifest_sha256,
        expected_preparation_sha256):
    from .age_interface import AgeInterfaceCache
    directory = project_path(directory)
    path = directory/"manifest.json"
    _require_equal(sha256_file(path), expected_manifest_sha256, "Age manifest integrity failure")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise RuntimeError("Incomplete/unsupported age dataset")
    _require_equal(manifest["preparation_sha256"], _digest(manifest["preparation"]),
        "Age preparation integrity failure")
    _require_equal(manifest["preparation_sha256"], expected_preparation_sha256,
        "Unexpected age preparation")
    _require_equal(manifest["implementation_sha256"], implementation_hashes(),
        "Age implementation/dependency drift")
    _require_equal(manifest["boundary_directory"], str(dataset.directory),
        "Unexpected plain boundary directory")
    _require_equal(manifest["boundary_manifest_sha256"], dataset.manifest_sha256,
        "Age sidecars refer to another boundary dataset")
    _require_equal(manifest["boundary_preparation_sha256"], dataset.preparation_sha256,
        "Unexpected plain preparation")
    records = {r["tile_id"]: r for r in manifest["records"]}
    plain = {r["tile_id"]: r for r in dataset.records}
    if len(records) != len(manifest["records"]) or set(records) != set(plain):
        raise RuntimeError("Missing/duplicate/extraneous age tile identity")
    result = {}
    for parent in dataset.parent_cases:
        for tile in parent.tiles:
            identity = tile.provenance["tile_id"]
            record, source = records[identity], plain[identity]
            relative = str(Path(source["bank_file"]).with_name("age_operators.pt"))
            _require_equal(record["operator_file"], relative, "Age artifact binding mismatch")
            filename = project_path(directory/relative)
            if directory not in filename.parents:
                raise RuntimeError("Age artifact escapes dataset")
            _require_equal(record["plain_bank_sha256"], source["bank_sha256"],
                "Age/plain artifact binding mismatch")
            _require_equal(record["preparation_sha256"], expected_preparation_sha256,
                "Age tile preparation mismatch")
            _require_equal(sha256_file(filename), record["operator_sha256"],
                "Age operator integrity failure")
            _require_equal(filename.stat().st_size, record["operator_bytes"],
                "Age operator byte-count mismatch")
            payload = torch.load(filename, map_location="cpu", weights_only=True)
            if set(payload) != {"schema_version", "operator_cache", "provenance"}:
                raise RuntimeError("Unexpected age envelope, possible supervision leakage")
            _require_equal(payload["schema_version"], SCHEMA_VERSION, "Age payload schema mismatch")
            provenance = {name: record[name] for name in (
                "tile_id", "plain_bank_sha256", "preparation_sha256")}
            _require_equal(payload["provenance"], provenance, "Age payload provenance mismatch")
            cache = AgeInterfaceCache.from_dict(payload["operator_cache"])
            _check_partition(cache, tile.bank, manifest["preparation"]["age_scales"])
            result[identity] = cache
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundary-manifest-sha256", required=True)
    parser.add_argument("--boundary-preparation-sha256", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare_age_dataset(args.boundary_data, args.output,
        expected_boundary_manifest_sha256=args.boundary_manifest_sha256,
        expected_boundary_preparation_sha256=args.boundary_preparation_sha256,
        progress=lambda event: print(json.dumps(event), flush=True))

if __name__ == "__main__":
    main()

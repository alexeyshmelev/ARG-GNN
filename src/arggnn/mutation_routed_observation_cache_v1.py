"""Compressed, observed-only four-parent CPU bank cache. No pickle or targets."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import hashlib
import io
import json
import os
import uuid

import numpy as np

from .mutation_routed_parent_loader_v1 import ROOT, PARENTS, PRIMARY_REFS, _verified_scratch_path_v1
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1, validate_inputs_v1

SCHEMA = "mutation_routed_observation_cache_v1"
BUNDLE_FIELDS = {"parents", "train_parent_ids", "val_parent_ids", "sham_bin_edges", "receipt", "efficiency_provenance"}
SCORING_FIELDS = {"parent_id", "split", "samples", "query_slots", "raw_query_nodes", "cohorts", "positions", "edges", "retained_indexes", "coverage_original", "query_ids", "positions_original", "bin_edges_original", "query_sample_index", "original_query_nodes", "parent_manifest_ref", "observed_parent_entry_ref"}


def _json_bytes(value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError("Only JSON metadata and separate observed arrays may be cached")
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=scalar) + "\n").encode()


def _canonical(value):
    return json.loads(_json_bytes(value))


def _tuple_paths(value, prefix=()):
    paths = [list(prefix)] if isinstance(value, tuple) else []
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
    for key, child in children:
        paths.extend(_tuple_paths(child, prefix + (key,)))
    return paths


def _restore_tuples(metadata):
    for path in sorted(metadata.pop("tuple_paths", []), key=len, reverse=True):
        parent = metadata
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = tuple(parent[path[-1]])
    return metadata


def _check_pins(observed_parent_refs, source_refs):
    if set(observed_parent_refs) != set(PARENTS) or not source_refs:
        raise ValueError("Four frozen observed parents and explicit observation/model source pins required")
    for refs in observed_parent_refs.values():
        if set(refs) != PRIMARY_REFS:
            raise ValueError("Only qualified prepared/entry/roster observation refs accepted")
    records = list(source_refs.values()) + [ref for refs in observed_parent_refs.values() for ref in refs.values()]
    for ref in records:
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str) or not isinstance(ref["sha256"], str)
                or len(ref["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in ref["sha256"])):
            raise ValueError("Every provenance pin requires exact path and SHA256")
    for ref in source_refs.values():
        path = Path(ref["path"])
        if path.is_absolute():
            path = path.relative_to(ROOT)
        if ".." in path.parts or not path.as_posix().startswith("src/arggnn/mutation_routed_"):
            raise ValueError("Cache source pins must identify the frozen observed/model implementation")


def _check_parent(pid, item, observed_refs):
    if set(item) != {"model_inputs", "observations_receipt", "scoring"}:
        raise ValueError("Only model observations, observation receipt and original scoring metadata accepted")
    receipt = item["observations_receipt"]
    if receipt.get("targets_read") is not False or receipt.get("parent_id") != pid:
        raise ValueError("Explicit observed-only parent provenance required")
    if {key: receipt["refs"][key] for key in PRIMARY_REFS} != observed_refs:
        raise ValueError("Cache parent observation lineage differs from requested frozen refs")
    if set(item["scoring"]) - SCORING_FIELDS:
        raise ValueError("Target masses and extra scoring arrays are excluded from the observation cache")


def _encode_parent(item, *, expected_haplotypes):
    bank = item["model_inputs"]
    validate_inputs_v1(bank, expected_haplotypes=expected_haplotypes)
    arrays, model_spec = {}, {}
    for field in fields(bank):
        value = getattr(bank, field.name)
        if isinstance(value, tuple):
            keys = ["model." + field.name + "." + str(i) for i in range(len(value))]
            arrays.update({key: tensor.detach().numpy() for key, tensor in zip(keys, value)})
            model_spec[field.name] = keys
        else:
            key = "model." + field.name
            arrays[key] = value.detach().numpy()
            model_spec[field.name] = key
    scoring = {}
    for name, value in item["scoring"].items():
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("Object arrays and pickle are forbidden")
            key = "scoring." + name
            arrays[key] = value
            scoring[name] = {"array_key": key}
        else:
            scoring[name] = {"json": value}
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    return payload.getvalue(), dict(model_spec=model_spec, scoring=scoring,
        observations_receipt=item["observations_receipt"])


def _write_fresh_atomic(path, data):
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as file:
            file.write(data); file.flush(); os.fsync(file.fileno()); os.fchmod(file.fileno(), 0o444)
        os.link(temporary, path)  # Atomic publication with no existing-file overwrite.
    finally:
        temporary.unlink(missing_ok=True)


def write_observation_cache_v1(bundle, cache_directory, *, observed_parent_refs,
        source_refs, guard_exact_bytes, expected_haplotypes=280):
    """Serialize in RAM, call root's exact-byte guard, then publish fresh files.

    The caller owns the shared cache scope/cap. No cache budget audit occurs here.
    The manifest is published last and is the only completed-cache entry point.
    """
    _check_pins(observed_parent_refs, source_refs)
    if (set(bundle) - BUNDLE_FIELDS or set(bundle["parents"]) != set(PARENTS)
            or bundle["receipt"].get("targets_read") is not False):
        raise ValueError("Only the frozen observed parent collection may be cached")
    directory = _verified_scratch_path_v1(ROOT, cache_directory)
    payloads, parent_meta = {}, {}
    for pid in PARENTS:
        item = bundle["parents"][pid]
        _check_parent(pid, item, observed_parent_refs[pid])
        data, metadata = _encode_parent(item, expected_haplotypes=expected_haplotypes)
        filename = pid + ".npz"
        payloads[filename] = data
        parent_meta[pid] = dict(metadata, array_ref={"path": (directory / filename).relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest()}, compressed_bytes=len(data))
    metadata = dict(schema=SCHEMA, observed_parent_refs=observed_parent_refs, source_refs=source_refs,
        parents=parent_meta, bundle_metadata={key: value for key, value in bundle.items() if key != "parents"},
        expected_haplotypes=expected_haplotypes, observations_only=True, targets_cached=False,
        pickle_used=False, compressed_payload_bytes=sum(map(len, payloads.values())))
    metadata["tuple_paths"] = _tuple_paths(metadata)
    manifest_data = _json_bytes(metadata)
    payloads["manifest.json"] = manifest_data
    exact_bytes = sum(map(len, payloads.values()))
    if guard_exact_bytes(exact_bytes) is False:
        raise ValueError("Root exact-byte cache guard rejected the compressed write")
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in payloads):
        raise FileExistsError("Observation cache files must be exclusively fresh")
    for name, data in payloads.items():
        _write_fresh_atomic(directory / name, data)
    return dict(path=(directory / "manifest.json").relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(manifest_data).hexdigest())


def read_observation_cache_v1(manifest_ref, reader, *, observed_parent_refs,
        source_refs, expected_haplotypes=280):
    """Verify readonly cache/source pins and reconstruct the exact CPU bank."""
    _check_pins(observed_parent_refs, source_refs)
    for ref in source_refs.values():
        reader.read(ref)
    data = reader.read(manifest_ref)
    if hashlib.sha256(data).hexdigest() != manifest_ref["sha256"]:
        raise ValueError("Cache manifest hash differs")
    metadata = _restore_tuples(json.loads(data))
    if (metadata.get("schema") != SCHEMA or metadata.get("targets_cached") is not False
            or metadata.get("pickle_used") is not False or metadata.get("observations_only") is not True
            or metadata.get("expected_haplotypes") != expected_haplotypes
            or metadata.get("observed_parent_refs") != _canonical(observed_parent_refs)
            or metadata.get("source_refs") != _canonical(source_refs) or set(metadata["parents"]) != set(PARENTS)):
        raise ValueError("Frozen cache schema, source or observed parent provenance differs")
    bundle = metadata["bundle_metadata"]
    if set(bundle) - (BUNDLE_FIELDS - {"parents"}) or bundle["receipt"].get("targets_read") is not False:
        raise ValueError("Target-bearing or extra cache metadata forbidden")
    bundle["parents"] = {}
    for pid in PARENTS:
        item = metadata["parents"][pid]
        payload = reader.read(item["array_ref"])
        if hashlib.sha256(payload).hexdigest() != item["array_ref"]["sha256"] or len(payload) != item["compressed_bytes"]:
            raise ValueError("Compressed observed cache bytes differ")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            spec = item["model_spec"]
            wanted = {key for value in spec.values() for key in (value if isinstance(value, list) else [value])}
            wanted.update(value["array_key"] for value in item["scoring"].values() if "array_key" in value)
            if set(archive.files) != wanted:
                raise ValueError("Only exact model and original scoring array keys may be decoded")
            arrays = {name: tuple(archive[key].copy() for key in value) if isinstance(value, list)
                      else archive[value].copy() for name, value in spec.items()}
            scoring = {name: archive[value["array_key"]].copy() if "array_key" in value else value["json"]
                       for name, value in item["scoring"].items()}
        bank = MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
        validate_inputs_v1(bank, expected_haplotypes=expected_haplotypes)
        restored = dict(model_inputs=bank, scoring=scoring, observations_receipt=item["observations_receipt"])
        _check_parent(pid, restored, observed_parent_refs[pid])
        bundle["parents"][pid] = restored
    return bundle


class VerifiedObservationCacheIndexV1:
    """Manifest/source verification once per process, then parent-local reads."""
    def __init__(self, metadata, manifest_ref, observed_parent_refs, source_refs, expected_haplotypes):
        self.metadata = metadata
        self.manifest_ref = manifest_ref
        self.observed_parent_refs = observed_parent_refs
        self.source_refs = source_refs
        self.expected_haplotypes = expected_haplotypes


def read_observation_cache_index_v1(manifest_ref, reader, *, observed_parent_refs,
        source_refs, expected_haplotypes=280):
    """Read only JSON/source pins, never any parent NPZ; suitable for workers."""
    _check_pins(observed_parent_refs, source_refs)
    for ref in source_refs.values():
        reader.read(ref)
    data = reader.read(manifest_ref)
    if hashlib.sha256(data).hexdigest() != manifest_ref["sha256"]:
        raise ValueError("Cache manifest hash differs")
    metadata = _restore_tuples(json.loads(data))
    if (metadata.get("schema") != SCHEMA or metadata.get("targets_cached") is not False
            or metadata.get("pickle_used") is not False or metadata.get("observations_only") is not True
            or metadata.get("expected_haplotypes") != expected_haplotypes
            or metadata.get("observed_parent_refs") != _canonical(observed_parent_refs)
            or metadata.get("source_refs") != _canonical(source_refs) or set(metadata["parents"]) != set(PARENTS)):
        raise ValueError("Frozen cache schema, source or observed parent provenance differs")
    bundle = metadata["bundle_metadata"]
    if set(bundle) - (BUNDLE_FIELDS - {"parents"}) or bundle["receipt"].get("targets_read") is not False:
        raise ValueError("Target-bearing or extra cache metadata forbidden")
    return VerifiedObservationCacheIndexV1(metadata, dict(manifest_ref),
        _canonical(observed_parent_refs), _canonical(source_refs), expected_haplotypes)


def read_observation_cache_parent_v1(parent_id, index, reader):
    """Decode only one complete native parent from a verified readonly index."""
    if not isinstance(index, VerifiedObservationCacheIndexV1) or parent_id not in PARENTS:
        raise ValueError("Verified observed cache index and frozen parent ID required")
    item = index.metadata["parents"][parent_id]
    payload = reader.read(item["array_ref"])
    if hashlib.sha256(payload).hexdigest() != item["array_ref"]["sha256"] or len(payload) != item["compressed_bytes"]:
        raise ValueError("Compressed observed cache bytes differ")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        spec = item["model_spec"]
        wanted = {key for value in spec.values() for key in (value if isinstance(value, list) else [value])}
        wanted.update(value["array_key"] for value in item["scoring"].values() if "array_key" in value)
        if set(archive.files) != wanted:
            raise ValueError("Only exact model and original scoring array keys may be decoded")
        arrays = {name: tuple(archive[key].copy() for key in value) if isinstance(value, list)
                  else archive[value].copy() for name, value in spec.items()}
        scoring = {name: archive[value["array_key"]].copy() if "array_key" in value else value["json"]
                   for name, value in item["scoring"].items()}
    bank = MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
    validate_inputs_v1(bank, expected_haplotypes=index.expected_haplotypes)
    restored = dict(model_inputs=bank, scoring=scoring, observations_receipt=item["observations_receipt"])
    _check_parent(parent_id, restored, index.observed_parent_refs[parent_id])
    return restored

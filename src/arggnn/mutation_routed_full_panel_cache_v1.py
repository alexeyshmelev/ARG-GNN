"""Registry-bound full952 observed cache and complete-parent PyG Dataset."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from torch_geometric.data import Dataset
from .census_tsinfer_fit_v1 import ConsumerReaderV1
from . import mutation_routed_observation_cache_v1 as old
from .mutation_routed_parent_loader_v1 import ROOT, _verified_scratch_path_v1
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from .mutation_routed_full_panel_observations_v1 import validate_registry_v1, validate_full_bank_v1
from .mutation_routed_pyg_dataset_v1 import (
    MutationRoutedCachedParentDatasetV1, MutationRoutedParentDataV1,
    make_parent_dataloader_v1, shutdown_parent_dataloader_v1,
)
SCHEMA="mutation_routed_full_panel_observed_cache_v1"
FIELDS={"schema","registry","entry_refs","source_refs","parents","sham_bin_edges","receipt",
        "targets_cached","observations_only","pickle_used","tuple_paths"}


def _validate_source_refs_v1(source_refs):
    if not source_refs:raise ValueError("Explicit observed implementation source pins required")
    for ref in source_refs.values():
        if (set(ref)!={"path","sha256"} or not isinstance(ref["path"],str)
                or not ref["path"].startswith("src/arggnn/") or not ref["path"].endswith(".py")
                or ".." in Path(ref["path"]).parts or len(ref["sha256"])!=64
                or any(c not in "0123456789abcdef" for c in ref["sha256"])):
            raise ValueError("Observed cache provenance accepts implementation Python sources only")


def _check_item_v1(parent_id,item,entry_ref,roles):
    if set(item)!={"model_inputs","scoring","observations_receipt"}:
        raise ValueError("Observed bank/scoring/receipt fields only")
    validate_full_bank_v1(item["model_inputs"])
    score=item["scoring"];receipt=item["observations_receipt"]
    if (set(score)!=old.SCORING_FIELDS or score["parent_id"]!=parent_id
            or score["split"]!=roles[parent_id] or len(score["query_slots"])!=686
            or score["parent_manifest_ref"]!=entry_ref or score["observed_parent_entry_ref"]!=entry_ref
            or receipt["entry_ref"]!=entry_ref or receipt["targets_read"] is not False
            or receipt["observations_only"] is not True
            or not np.array_equal(item["model_inputs"].query_column.numpy(),score["query_slots"])):
        raise ValueError("Registry/query/input-provenance cache binding differs")


def write_full_cache_v1(bundle,directory,*,source_refs,guard_exact_bytes):
    """Root-owned observed publication. Each actual compressed write is guarded."""
    if set(bundle)!={"parents","registry","entry_refs","sham_bin_edges","receipt"}:
        raise ValueError("No targets, score caches or extra fields accepted")
    registry=bundle["registry"];validate_registry_v1(registry);_validate_source_refs_v1(source_refs)
    roles={r["parent_id"]:r["split"] for r in registry}
    entries=bundle["entry_refs"];items=bundle["parents"]
    if set(items)!=set(entries) or not set(items).issubset(roles) or not source_refs:
        raise ValueError("Exact observed entries and source provenance required")
    if bundle["receipt"].get("targets_read") is not False:
        raise ValueError("Target-bearing cache forbidden")
    directory=_verified_scratch_path_v1(ROOT,directory)
    metadata=dict(schema=SCHEMA,registry=registry,entry_refs=entries,source_refs=source_refs,
        parents={},sham_bin_edges=bundle["sham_bin_edges"],receipt=bundle["receipt"],
        targets_cached=False,observations_only=True,pickle_used=False)
    # One parent serialized/published at a time; only final manifest certifies completion.
    for parent_id,item in items.items():
        _check_item_v1(parent_id,item,entries[parent_id],roles)
        payload,record=old._encode_parent(item,expected_haplotypes=952)
        path=directory/(parent_id+".npz")
        record.update(array_ref={"path":path.relative_to(ROOT).as_posix(),
            "sha256":hashlib.sha256(payload).hexdigest()},compressed_bytes=len(payload))
        if guard_exact_bytes(len(payload)) is False:
            raise ValueError("Root compressed observed write guard rejected")
        directory.mkdir(parents=True,exist_ok=True)
        old._write_fresh_atomic(path,payload)
        metadata["parents"][parent_id]=record
    metadata["tuple_paths"]=old._tuple_paths(metadata)
    payload=old._json_bytes(metadata)
    if guard_exact_bytes(len(payload)) is False:
        raise ValueError("Root observed manifest write guard rejected")
    old._write_fresh_atomic(directory/"manifest.json",payload)
    return {"path":(directory/"manifest.json").relative_to(ROOT).as_posix(),
            "sha256":hashlib.sha256(payload).hexdigest()}


def read_full_cache_index_v1(manifest_ref,reader,*,registry,entry_refs,source_refs):
    validate_registry_v1(registry);_validate_source_refs_v1(source_refs)
    for ref in source_refs.values():reader.read(ref)
    payload=reader.read(manifest_ref)
    if hashlib.sha256(payload).hexdigest()!=manifest_ref["sha256"]:
        raise ValueError("Observed manifest hash differs")
    raw=json.loads(payload)
    if set(raw)!=FIELDS:
        raise ValueError("Strict observed cache manifest fields required")
    value=old._restore_tuples(raw)
    if (value["schema"]!=SCHEMA or value["registry"]!=old._canonical(registry)
            or value["entry_refs"]!=old._canonical(entry_refs)
            or value["source_refs"]!=old._canonical(source_refs)
            or value["targets_cached"] is not False or value["observations_only"] is not True
            or value["pickle_used"] is not False or value["receipt"].get("targets_read") is not False
            or set(value["parents"])!=set(entry_refs)):
        raise ValueError("Registry/source/entry/observed cache contract differs")
    return value


def read_full_cache_parent_v1(parent_id,index,reader):
    if parent_id not in index["parents"]:raise ValueError("Parent absent from observed cache")
    metadata=index["parents"][parent_id]
    payload=reader.read(metadata["array_ref"])
    if len(payload)!=metadata["compressed_bytes"] or hashlib.sha256(payload).hexdigest()!=metadata["array_ref"]["sha256"]:
        raise ValueError("Observed compressed byte count differs")
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        spec=metadata["model_spec"]
        wanted={k for value in spec.values() for k in (value if isinstance(value,list) else [value])}
        wanted.update(v["array_key"] for v in metadata["scoring"].values() if "array_key" in v)
        if set(archive.files)!=wanted:raise ValueError("Only exact observed arrays permitted")
        arrays={name:tuple(archive[k].copy() for k in keys) if isinstance(keys,list)
                else archive[keys].copy() for name,keys in spec.items()}
        scoring={name:archive[value["array_key"]].copy() if "array_key" in value else value["json"]
                 for name,value in metadata["scoring"].items()}
    item=dict(model_inputs=MutationRoutedWholeTreeInputsV1.from_arrays(arrays),
        scoring=scoring,observations_receipt=metadata["observations_receipt"])
    _check_item_v1(parent_id,item,index["entry_refs"][parent_id],
                  {r["parent_id"]:r["split"] for r in index["registry"]})
    return item


class FullPanelParentDatasetV1(MutationRoutedCachedParentDatasetV1):
    """Two-worker loader-compatible complete PyG parents; no target inputs."""
    def __init__(self,manifest_ref,*,registry,entry_refs,source_refs,parent_ids,
                 worker_tmp_directory,allow_outer=False):
        Dataset.__init__(self,root=None,transform=None,pre_transform=None,pre_filter=None)
        roles=validate_registry_v1(registry)
        selected=tuple(parent_ids)
        allowed=set(roles["outer"] if allow_outer else roles["train"]+roles["val"])
        if not selected or len(set(selected))!=len(selected) or not set(selected).issubset(allowed):
            raise ValueError("Registered permitted complete observed parent subset required")
        self.registry=registry;self.entry_refs=entry_refs;self.source_refs=source_refs
        self.manifest_ref=dict(manifest_ref);self.parent_ids=selected
        self.worker_tmp_directory=str(_verified_scratch_path_v1(ROOT,worker_tmp_directory))
        self._reader=self._index=self._process_id=None

    def _process_state(self):
        if self._process_id!=os.getpid():
            self._reader=ConsumerReaderV1()
            self._index=read_full_cache_index_v1(self.manifest_ref,self._reader,
                registry=self.registry,entry_refs=self.entry_refs,source_refs=self.source_refs)
            self._process_id=os.getpid()
        return self._reader,self._index

    def get(self,index):
        reader,cache=self._process_state()
        parent_id=self.parent_ids[int(index)]
        item=read_full_cache_parent_v1(parent_id,cache,reader)
        return MutationRoutedParentDataV1(item["model_inputs"],parent_id=parent_id,
            scoring=item["scoring"],observations_receipt=item["observations_receipt"])

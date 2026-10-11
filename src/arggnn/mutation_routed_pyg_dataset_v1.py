"""PyG complete cached-parent Dataset with CPU-only spawn workers.

One dataset item is one entire original parent forest. No target is accepted,
no tree/sample/readout index is rebased and no original hold is omitted.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import os
import tempfile
import sys
import uuid
import multiprocessing.connection as mp_connection

import torch
from torch.utils.data import DataLoader, get_worker_info
from torch_geometric.data import Data, Dataset

from .census_tsinfer_fit_v1 import ConsumerReaderV1
from .mutation_routed_parent_loader_v1 import ROOT, PARENTS, _verified_scratch_path_v1
from .mutation_routed_observation_cache_v1 import (
    read_observation_cache_index_v1, read_observation_cache_parent_v1,
)
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1

BANK_FIELDS = tuple(field.name for field in fields(MutationRoutedWholeTreeInputsV1))
LEVEL_FIELDS = {"upward_levels", "upward_children", "upward_parent_rank", "downward_levels"}


class MutationRoutedParentDataV1(Data):
    """All observed bank tensors plus explicit native child-to-parent edges."""
    def __init__(self, model_inputs=None, *, parent_id=None, scoring=None, observations_receipt=None):
        super().__init__()
        if model_inputs is not None:
            for name in BANK_FIELDS:
                self[name] = getattr(model_inputs, name)
            child = torch.nonzero(model_inputs.parent >= 0, as_tuple=False).flatten()
            self.edge_index = torch.stack((child, model_inputs.parent[child]))
            self.num_nodes = model_inputs.parent.numel()
            self.parent_id = parent_id
            self.scoring = scoring
            self.observations_receipt = observations_receipt
            self.worker_pid = os.getpid()
            self.worker_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            self.worker_tmp_directory = os.environ.get("TMPDIR")
            self.worker_cpu_threads = torch.get_num_threads()

    def to_model_inputs(self):
        values = {name: tuple(self[name]) if name in LEVEL_FIELDS else self[name] for name in BANK_FIELDS}
        return MutationRoutedWholeTreeInputsV1(**values)

    def __inc__(self, key, value, *args, **kwargs):
        # The supported loader never batches parents. Explicitly prevent PyG's
        # automatic sample_index/edge_index increments if inspected elsewhere.
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        return None

    def pin_memory(self, *args):
        # Pin only scientific tensors/levels, never observation/scorer metadata.
        for name in BANK_FIELDS + ("edge_index",):
            value = self[name]
            self[name] = tuple(tensor.pin_memory() for tensor in value) if name in LEVEL_FIELDS else value.pin_memory()
        return self


class MutationRoutedCachedParentDatasetV1(Dataset):
    """Lazy per-process index, one NPZ per get; no four-parent reread."""
    def __init__(self, manifest_ref, *, observed_parent_refs, source_refs,
            parent_ids=PARENTS, expected_haplotypes=280, worker_tmp_directory):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
        self.manifest_ref = dict(manifest_ref)
        self.observed_parent_refs = observed_parent_refs
        self.source_refs = source_refs
        self.parent_ids = tuple(parent_ids)
        self.expected_haplotypes = expected_haplotypes
        self.worker_tmp_directory = str(_verified_scratch_path_v1(ROOT, worker_tmp_directory))
        if not self.parent_ids or len(set(self.parent_ids)) != len(self.parent_ids) or not set(self.parent_ids).issubset(PARENTS):
            raise ValueError("Unique frozen observed parent items required")
        self._reader = self._index = None
        self._process_id = None

    def len(self):
        return len(self.parent_ids)

    def _process_state(self):
        if self._process_id != os.getpid():
            self._reader = ConsumerReaderV1()
            self._index = read_observation_cache_index_v1(self.manifest_ref, self._reader,
                observed_parent_refs=self.observed_parent_refs, source_refs=self.source_refs,
                expected_haplotypes=self.expected_haplotypes)
            self._process_id = os.getpid()
        return self._reader, self._index

    def get(self, index):
        reader, cache_index = self._process_state()
        parent_id = self.parent_ids[int(index)]
        item = read_observation_cache_parent_v1(parent_id, cache_index, reader)
        return MutationRoutedParentDataV1(item["model_inputs"], parent_id=parent_id,
            scoring=item["scoring"], observations_receipt=item["observations_receipt"])

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_reader=None, _index=None, _process_id=None)
        return state


def _complete_parent_identity_v1(data):
    return data


_ORIGINAL_ARBITRARY_ADDRESS_V1 = mp_connection.arbitrary_address


def _worker_ipc_address_v1(family):
    # Linux abstract UNIX sockets avoid the108-byte pathname limit encountered
    # by Torch tensor FD transport under long project-local worker TMPDIR paths.
    # This creates no external filesystem artifact and changes no sample data.
    if family == "AF_UNIX" and sys.platform.startswith("linux"):
        return b"\0arggnn-pyg-" + str(os.getpid()).encode() + b"-" + uuid.uuid4().hex.encode()
    return _ORIGINAL_ARBITRARY_ADDRESS_V1(family)


def _cpu_worker_init_v1(worker_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    mp_connection.arbitrary_address = _worker_ipc_address_v1
    dataset = get_worker_info().dataset
    base = _verified_scratch_path_v1(ROOT, dataset.worker_tmp_directory)
    directory = _verified_scratch_path_v1(ROOT, base / ("worker_" + str(os.getpid())))
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(directory)
    tempfile.tempdir = str(directory)


def make_parent_dataloader_v1(dataset, *, pin_memory=True):
    """Two persistent CPU spawn workers; complete parent items, no batching.

    Iterate once, transfer to the GPU only in the main process, and retain the
    returned model banks there across optimizer updates when memory permits.
    No Dataset/DataLoader iteration is needed for each training update.
    """
    if not isinstance(dataset, MutationRoutedCachedParentDatasetV1):
        raise TypeError("A complete observed cached-parent PyG Dataset is required")
    return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2,
        multiprocessing_context="spawn", persistent_workers=True, pin_memory=pin_memory,
        prefetch_factor=1, collate_fn=_complete_parent_identity_v1,
        worker_init_fn=_cpu_worker_init_v1)


def parent_worker_pids_v1(loader):
    iterator = getattr(loader, "_iterator", None)
    return [] if iterator is None else [worker.pid for worker in iterator._workers]


def shutdown_parent_dataloader_v1(loader):
    """Drain only this loader's registered workers before completion sealing."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return dict(getattr(loader, "_arggnn_worker_drain_receipt",
            dict(worker_pids=[], worker_exitcodes=[], drained=True)))
    workers = tuple(iterator._workers)
    pids = [worker.pid for worker in workers]
    iterator._shutdown_workers()  # Torch2.6 persistent-worker lifecycle hook.
    loader._iterator = None
    receipt = dict(worker_pids=pids, worker_exitcodes=[worker.exitcode for worker in workers],
        drained=not any(worker.is_alive() for worker in workers))
    loader._arggnn_worker_drain_receipt = receipt
    return dict(receipt)


# Equivalent descriptive alias for callers that prefer drain terminology.
drain_parent_dataloader_v1 = shutdown_parent_dataloader_v1

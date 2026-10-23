"""Fresh C3 parent identities over the unchanged selected TreeLoom equations.

Only registry membership and explicit static module/type bindings differ from the
verified historical implementation. TEST uses its observation-only
val_prediction execution phase. The public boundary retains the true split.
This module provides no launcher, checkpoint reader, scorer or TEST truth API.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from types import MappingProxyType
import zipfile

from arggnn.benchmarking import contracts as c

ROOT = c.ROOT
CLASSES = ("EUR", "EAS", "AFR")
SCHEMA = "treeloom_fresh_parent_registry_v1"
LENGTH = 30_000_000
FOCALS = 10_000
INPUT_REFS = ("observation_ref", "focal_registry_ref", "native_preparation_ref")
IDENTITY = ("record_ids", "sample_nodes", "sample_indices", "positions_bp0",
            "parent_id", "class_order")


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _ref(value):
    _need(isinstance(value, dict) and set(value) == {"path", "sha256"}
          and isinstance(value["path"], str) and value["path"]
          and isinstance(value["sha256"], str) and len(value["sha256"]) == 64
          and all(c in "0123456789abcdef" for c in value["sha256"]),
          "Explicit immutable project file reference required")
    return dict(value)


def _read(record):
    from arggnn.american_c3_common_v1 import read
    return read(_ref(record))


@dataclass(frozen=True)
class FreshParentRegistry:
    """Authenticated independent parents, without any ancestry truth path."""

    reference_items: tuple
    parents: tuple

    @classmethod
    def from_ref(cls, reference):
        reference = _ref(reference)
        payload = _read(reference)
        _need(isinstance(payload, dict) and payload.get("schema") == SCHEMA
              and payload.get("split_unit") == "independent_simulation_parent"
              and payload.get("class_order") == list(CLASSES)
              and payload.get("sequence_length_bp") == LENGTH,
              "Explicit independent C3 30Mb parent registry required")
        rows = payload.get("parents")
        _need(isinstance(rows, list) and rows, "Nonempty fresh parent registry")
        parents = []
        for row in rows:
            _need(isinstance(row, dict)
                  and isinstance(row.get("parent_id"), str) and row["parent_id"]
                  and row.get("split") in ("train", "validation", "test")
                  and isinstance(row.get("simulation_stream"), str)
                  and row["simulation_stream"],
                  "Explicit parent identity, split and simulation stream required")
            parents.append((row["parent_id"], row["split"], row["simulation_stream"]))
        _need(len({p[0] for p in parents}) == len(parents)
              and len({p[2] for p in parents}) == len(parents)
              and {p[1] for p in parents} == {"train", "validation", "test"},
              "Disjoint parent identities/streams and all three splits required")
        return cls(tuple(sorted(reference.items())), tuple(parents))

    @property
    def reference(self):
        return dict(self.reference_items)

    def ids(self, split):
        _need(split in ("train", "validation", "test"), "Declared registry split required")
        return tuple(p for p, s, _ in self.parents if s == split)

    def split_for(self, parent_id):
        values = [s for p, s, _ in self.parents if p == parent_id]
        _need(len(values) == 1, "Parent must belong to the immutable fresh registry")
        return values[0]

    def epoch_order(self, epoch):
        _need(type(epoch) is int and 1 <= epoch <= 30, "Declared training epoch 1..30 required")
        rows = list(self.ids("train"))
        random.Random(70002 + epoch).shuffle(rows)
        return rows


def _identity_sha(records):
    import numpy as np
    h = hashlib.sha256()
    for key in IDENTITY:
        value = np.ascontiguousarray(records[key])
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


class FreshTrainTargetJoin:
    """Resolve only registered TRAIN truth after all observed identities seal."""

    def __init__(self, registry, resolve_train_truth, *, train_inputs,
                 max_truth_uncompressed_bytes):
        _need(type(registry) is FreshParentRegistry and callable(resolve_train_truth),
              "Authenticated fresh registry and separate TRAIN resolver required")
        _need(type(max_truth_uncompressed_bytes) is int
              and max_truth_uncompressed_bytes > 0,
              "Explicit bounded TRAIN truth payload required")
        _need(isinstance(train_inputs, dict)
              and set(train_inputs) == set(registry.ids("train")),
              "Exact input/truth bindings for every declared TRAIN parent required")
        bindings = {}
        for parent, values in train_inputs.items():
            _need(isinstance(values, dict)
                  and set(values) == {*INPUT_REFS, "truth_ref"},
                  "Observed and private truth refs must be separately explicit")
            bindings[parent] = MappingProxyType(
                {name: MappingProxyType(_ref(value)) for name, value in values.items()})
        self.registry = registry
        self.bindings = MappingProxyType(bindings)
        self.resolve = resolve_train_truth
        self.cap = max_truth_uncompressed_bytes

    def join(self, parent_id, *, split, plan, observation_seal):
        # Deny held-out parents before a resolver call or private path lookup.
        _need(split == "train" and self.registry.split_for(parent_id) == "train",
              "Only registered TRAIN truth may be resolved")
        binding = self.bindings[parent_id]
        _need(plan.get("parent_id") == parent_id
              and plan.get("sequence_length_bp") == LENGTH
              and observation_seal.get("parent_id") == parent_id
              and observation_seal.get("observation_only") is True
              and observation_seal.get("all_blocks_sealed_before_truth") is True
              and observation_seal.get("parent_registry_ref") == self.registry.reference
              and plan.get("plan_ref") == observation_seal.get("plan_ref")
              and isinstance(plan.get("plan_ref"), dict),
              "Complete fresh observation plan must seal before TRAIN truth")
        for name in INPUT_REFS:
            expected = dict(binding[name])
            _need(plan.get(name) == expected and observation_seal.get(name) == expected,
                  "Same-parent immutable observed inputs required before truth")

        from arggnn.american_c3_evaluate_v1 import registry as load_focals
        from arggnn.american_c3_common_v1 import bound
        from arggnn.american_c3_gnn_targets_v1 import TrainFocalTargetVectorV1
        import numpy as np
        import torch

        focal_ref = dict(binding["focal_registry_ref"])
        records = load_focals(focal_ref)
        identity = _identity_sha(records)
        _need(records["parent_id"].item() == parent_id
              and records["class_order"].tolist() == list(CLASSES)
              and len(records["record_ids"]) == FOCALS
              and identity == observation_seal.get("focal_identity_sha256"),
              "All original frozen fresh focal identities required")
        truth_ref = self.resolve(parent_id)
        _need(truth_ref == dict(binding["truth_ref"]),
              "Resolver must return the exact same-parent TRAIN truth reference")
        path = bound(_ref(truth_ref))
        keys = ("intervals_bp", "query_sample_indices", "source_classes_0based",
                "parent_id", "class_order")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            _need(len(members) == len(keys)
                  and {m.filename for m in members} == {k + ".npy" for k in keys}
                  and sum(m.file_size for m in members) <= self.cap,
                  "Bounded exact TRAIN truth NPZ without duplicate members required")
        with np.load(path, allow_pickle=False) as archive:
            truth = {k: archive[k] for k in archive.files}
        _need(np.array_equal(truth["parent_id"], records["parent_id"])
              and np.array_equal(truth["class_order"], records["class_order"]),
              "TRAIN truth parent/class identity must match observed focals")
        intervals = truth["intervals_bp"]
        samples = truth["query_sample_indices"]
        labels = truth["source_classes_0based"]
        _need(intervals.dtype == np.float64 and intervals.ndim == 2
              and intervals.shape[1] == 2 and len(intervals) > 0
              and np.isfinite(intervals).all() and intervals[0, 0] == 0
              and intervals[-1, 1] == LENGTH
              and (intervals[:, 1] > intervals[:, 0]).all()
              and np.array_equal(intervals[:-1, 1], intervals[1:, 0])
              and samples.dtype == np.int64
              and np.array_equal(samples, np.arange(600, 1600, dtype=np.int64))
              and labels.dtype == np.uint8
              and labels.shape == (len(intervals), 1000) and (labels < 3).all(),
              "Complete C3 half-open truth and actual query sample columns required")
        positions = records["positions_bp0"]
        _need(((positions >= 0) & (positions < LENGTH)).all()
              and np.isin(records["sample_indices"], samples).all(),
              "Every frozen focal must have its actual query sample and coordinate")
        lookup = {int(v): i for i, v in enumerate(samples)}
        rows = np.searchsorted(intervals[:, 1], positions, side="right")
        columns = np.asarray([lookup[int(v)] for v in records["sample_indices"]],
                             dtype=np.int64)
        targets = labels[rows, columns].astype(np.int64, copy=True)
        receipt = dict(schema="treeloom_fresh_TRAIN_target_join_v1",
            parent_id=parent_id, split="train",
            parent_registry_ref=self.registry.reference,
            truth_ref=dict(truth_ref), focal_registry_ref=focal_ref,
            focal_identity_sha256=identity, records=FOCALS,
            class_order=list(CLASSES), labels_in_observation_bank=False,
            purity_filter=False, nearest_or_majority_fallback=False,
            interval_rule="half_open_searchsorted_right_actual_sample_index")
        return TrainFocalTargetVectorV1(
            parent_id, focal_ref, identity,
            torch.tensor(targets, dtype=torch.long, device="cpu"), receipt)


class FreshTreeLoom:
    """One isolated numerical namespace bound to one immutable fresh registry."""

    def __init__(self, registry):
        _need(type(registry) is FreshParentRegistry, "Authenticated fresh registry required")
        from . import static_model_v2, static_preparation_v2, bounded_execution_v2
        self.registry = registry
        self.status_path = None
        self.frozen = static_model_v2
        self.preparation = static_preparation_v2
        self.execution = bounded_execution_v2
        train = registry.ids("train")
        heldout = registry.ids("validation") + registry.ids("test")
        _need(len(train) == 20, "Complete original20-parent TRAIN cohort required")
        # A fresh task interpreter owns this static namespace. Bind once only.
        for module in (self.frozen, self.preparation):
            _need(not module.TRAIN or (module.TRAIN == train and module.VAL == heldout),
                  "One immutable registry per fresh static interpreter required")
            module.TRAIN, module.VAL = train, heldout

    def build_model(self):
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(70001)
            model = self.frozen.AmericanTreeLoomC3FullV1()
        _need(len(list(model.parameters())) == 49
              and sum(p.numel() for p in model.parameters()) == 32132
              and all(p.device.type == "cpu" and p.dtype == torch.float32
                      for p in model.parameters()),
              "Unchanged selected 49-tensor C3 model required")
        from .gru_v2 import install_chunked_gru
        return install_chunked_gru(model, chunk_holds=1024)

    def prepare_observation(self, state, *, split, resource_check, **kwargs):
        _need(callable(resource_check), "The actual worker resource callback is required")
        parent = state["manifest"]["parent_id"]
        _need(self.registry.split_for(parent) == split,
              "Observed parent must retain its true fresh registry split")
        _need(split in ("train", "validation", "test"), "Declared fresh split required")
        observation_ref = _ref(kwargs["observation_ref"])
        _need(_read(observation_ref) == state["manifest"],
              "Loaded observations must match their immutable source manifest")
        # The immutable implementation has one observation-only prediction phase.
        phase = "train" if split == "train" else "val_prediction"
        bundle, metadata = self.preparation.prepare_from_loaded_observation_v1(
            state, split=phase, resource_check=resource_check, **kwargs)
        metadata = dict(metadata, parent_registry_ref=self.registry.reference,
                        declared_split=split,
                        internal_execution_phase=phase,
                        fresh_parent_identity_preserved=True)
        return bundle, metadata

    def _check_bundle(self, model, bundle, metadata, expected_split=None):
        _need(type(model) is self.frozen.AmericanTreeLoomC3FullV1
              and type(bundle) is self.frozen.AmericanTreeLoomWholeParentV1,
              "Matching static model and bundle types required")
        split = self.registry.split_for(bundle.parent_id)
        _need((expected_split is None or split == expected_split)
              and metadata.get("parent_id") == bundle.parent_id
              and metadata.get("declared_split") == split
              and metadata.get("parent_registry_ref") == self.registry.reference
              and bundle.split == ("train" if split == "train" else "val_prediction"),
              "Fresh parent identity/split must survive the numerical boundary")
        bundle.validate()
        return split

    def train_step(self, model, bundle, targets, optimizer, metadata, *,
                   resource_check, profile=None):
        self._check_bundle(model, bundle, metadata, "train")
        _need(targets.receipt.get("parent_registry_ref") == self.registry.reference
              and targets.receipt.get("split") == "train",
              "Separate fresh registered TRAIN target join required")
        from .step_v2 import train_step
        return train_step(model, bundle, targets, optimizer, metadata,
            resource_check=resource_check, frozen_module=self.frozen,
            status_path=self.status_path, holds_per_block=128,
            haplotypes_per_gru=64, history_holds_per_window=1024, profile=profile)

    def predict_logits(self, model, bundle, metadata, *, resource_check, profile=None):
        split = self._check_bundle(model, bundle, metadata)
        _need(split in ("validation", "test") and callable(resource_check),
              "Registered observation-only held-out prediction required")
        import torch
        model.eval()
        resource_check("fresh_TreeLoom_prediction")
        with torch.no_grad():
            executor = self.execution.BoundedExecutionV1(
                model, bundle, metadata, resource_check=resource_check,
                holds_per_block=128, haplotypes_per_gru=64, history_holds_per_window=1024, profile=profile)
            return executor.forward()

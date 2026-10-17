"""Observed-only parent descriptors, with an optional lazy Torch DataLoader.

Workers transport small immutable references, never prepared parent tensors.
They do not decode files, open query truth, run a model, or create CUDA state.
The admitted main process authenticates and decodes a descriptor through the
pinned TreeLoom codec/facade. Expensive whole-parent CPU preparation belongs in
a separately bounded CPU stage, not in a DataLoader worker returning ~50 GB of
tensors through multiprocessing IPC.

No new cache format, cache producer, cache capacity claim, or filesystem reader
is introduced here. Importing this module does not import Torch.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
from typing import Mapping, Sequence

SCHEMA = "arggnn.prepared_parent_descriptor.v1"
MAX_DESCRIPTOR_BYTES = 16384
_REFERENCE_FIELDS = (
    "protocol_ref", "preparation_source_ref", "observation_ref",
    "focal_registry_ref", "native_preparation_ref",
    "prepared_bundle_ref", "reference_labels_ref",
)
_REQUIRED_FIELDS = {
    "schema", "parent_id", "split", "observed_only", "protocol_ref",
    "preparation_source_ref", "observation_ref", "focal_registry_ref",
    "native_preparation_ref",
}
_OPTIONAL_FIELDS = {"prepared_bundle_ref", "reference_labels_ref", "identity_key"}


def _encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class ImmutableRef:
    """Repository-relative immutable handle; validation performs no file I/O."""

    path: str
    sha256: str

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path or len(self.path) > 2048:
            raise ValueError("A bounded repository-relative reference is required")
        path = PurePosixPath(self.path)
        if (path.is_absolute() or ".." in path.parts or "\\" in self.path
                or "\0" in self.path or str(path) != self.path or self.path == "."):
            raise ValueError("References must be canonical repository-relative paths")
        if (not isinstance(self.sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None):
            raise ValueError("Reference SHA256 must be an exact lowercase digest")

    @classmethod
    def from_value(cls, value: ImmutableRef | Mapping[str, str]) -> ImmutableRef:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ValueError("A reference contains only path and SHA256, never arrays or payloads")
        return cls(value["path"], value["sha256"])

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


def _observed_ref(value) -> ImmutableRef:
    ref = ImmutableRef.from_value(value)
    # Reject explicitly named query-truth/target handles. The pinned reader still
    # verifies the actual sanitized observation; filenames alone prove no content.
    parts = [part.lower() for part in PurePosixPath(ref.path).parts]
    if any(part in {"truth", "query_truth", "query_ancestry", "targets"}
           or part.startswith(("query_truth.", "query_ancestry.", "query_ancestry_truth"))
           for part in parts):
        raise ValueError("Observed descriptors cannot contain query-truth or target handles")
    return ref


@dataclass(frozen=True)
class PreparedParentDescriptor:
    """Only observed references and an immutable protocol/preparation identity.

    Reference-population labels may be supplied explicitly. Query ancestry,
    supervised targets, tensors and arbitrary extra fields are never accepted.
    An optional prepared_bundle_ref names an existing pinned preparation artifact;
    this module does not serialize, decode, verify or produce that artifact.
    """

    parent_id: str
    split: str
    protocol_ref: ImmutableRef | Mapping[str, str]
    preparation_source_ref: ImmutableRef | Mapping[str, str]
    observation_ref: ImmutableRef | Mapping[str, str]
    focal_registry_ref: ImmutableRef | Mapping[str, str]
    native_preparation_ref: ImmutableRef | Mapping[str, str]
    prepared_bundle_ref: ImmutableRef | Mapping[str, str] | None = None
    reference_labels_ref: ImmutableRef | Mapping[str, str] | None = None

    def __post_init__(self):
        if (not isinstance(self.parent_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.parent_id) is None
                or self.split not in {"train", "validation", "test"}):
            raise ValueError("An explicit parent and protocol split are required")
        for name in _REFERENCE_FIELDS:
            value = getattr(self, name)
            if value is None and name in {"prepared_bundle_ref", "reference_labels_ref"}:
                continue
            ref = (ImmutableRef.from_value(value)
                   if name in {"protocol_ref", "preparation_source_ref"} else _observed_ref(value))
            object.__setattr__(self, name, ref)
        if len(_encoded(self._payload())) > MAX_DESCRIPTOR_BYTES - 128:
            raise ValueError("Only small metadata descriptors may cross the worker boundary")

    def _payload(self) -> dict:
        value = {"schema": SCHEMA, "parent_id": self.parent_id,
                 "split": self.split, "observed_only": True}
        for name in _REFERENCE_FIELDS:
            ref = getattr(self, name)
            value[name] = None if ref is None else ref.as_dict()
        return value

    @property
    def identity_key(self) -> str:
        """Hash metadata identity, not a newly measured scientific payload."""
        return hashlib.sha256(_encoded(self._payload())).hexdigest()

    def as_dict(self) -> dict:
        return dict(self._payload(), identity_key=self.identity_key)

    @classmethod
    def from_mapping(cls, value: Mapping) -> PreparedParentDescriptor:
        if (not isinstance(value, Mapping) or not _REQUIRED_FIELDS <= set(value)
                or set(value) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
                or value["schema"] != SCHEMA or value["observed_only"] is not True):
            raise ValueError("Expected only declared observed-parent descriptor fields")
        descriptor = cls(**{name: value[name] for name in
                           ("parent_id", "split", *_REFERENCE_FIELDS) if name in value})
        if "identity_key" in value and value["identity_key"] != descriptor.identity_key:
            raise ValueError("Descriptor protocol/source/artifact identity changed")
        return descriptor


class PreparedParentDataset:
    """Map-style Dataset protocol with exact caller-declared parent order.

    This intentionally does not inherit from Torch's Dataset: importing a
    descriptor should not import a scientific runtime. DataLoader accepts its
    __len__/__getitem__ protocol. Repeated visits to the same immutable parent
    are allowed when explicitly present in parent_order.
    """

    def __init__(self, descriptors: Sequence[PreparedParentDescriptor | Mapping], *,
                 parent_order: Sequence[str]):
        self._descriptors = tuple(
            item if isinstance(item, PreparedParentDescriptor)
            else PreparedParentDescriptor.from_mapping(item) for item in descriptors)
        self.parent_order = tuple(parent_order)
        if tuple(item.parent_id for item in self._descriptors) != self.parent_order:
            raise ValueError("Descriptors must already be in the exact declared parent order")
        identities = {}
        common = None
        for item in self._descriptors:
            protocol = (item.protocol_ref, item.preparation_source_ref, item.split)
            if common is None:
                common = protocol
            if protocol != common:
                raise ValueError("A dataset cannot mix protocol/source versions or splits")
            if item.parent_id in identities and identities[item.parent_id] != item.identity_key:
                raise ValueError("Repeated parent visits must use the same immutable identity")
            identities[item.parent_id] = item.identity_key

    def __len__(self) -> int:
        return len(self._descriptors)

    def __getitem__(self, index: int) -> PreparedParentDescriptor:
        return self._descriptors[index]


def _descriptor_only(value):
    """Top-level picklable identity collator; no conversion, stacking or pinning."""
    if not isinstance(value, PreparedParentDescriptor):
        raise TypeError("Descriptor loader may return only immutable observed descriptors")
    return value


def _cpu_worker_init(_worker_id: int):
    """Runs only inside a fresh spawned DataLoader worker, never in the main process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def make_descriptor_loader(dataset: PreparedParentDataset, *, num_workers: int = 0):
    """Lazily create a CPU descriptor loader, retaining exact order and small IPC.

    Positive worker counts use spawn rather than inheriting the main process's
    CUDA context. CUDA/model/codec work remains in the admitted main process.
    A private CPU generator prevents DataLoader worker seeding from consuming
    the scientific training RNG. Creating this loader starts no benchmark; the
    caller still owns explicit execution approval and every resource admission.
    """
    if not isinstance(dataset, PreparedParentDataset):
        raise TypeError("A features-only PreparedParentDataset is required")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be an explicitly bounded nonnegative integer")
    import torch
    from torch.utils.data import DataLoader
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    options = dict(batch_size=None, shuffle=False, num_workers=num_workers,
                   collate_fn=_descriptor_only, pin_memory=False, generator=generator,
                   persistent_workers=num_workers > 0)
    if num_workers:
        options.update(prefetch_factor=1, multiprocessing_context="spawn",
                       worker_init_fn=_cpu_worker_init)
    return DataLoader(dataset, **options)

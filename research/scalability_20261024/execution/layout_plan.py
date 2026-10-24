"""Immutable, reusable CPU layouts for exact TreeLoom block execution.

This isolated module does not launch work or modify frozen training. Its arrays
match the current executor's eleven-field block tuple. Int32 storage is safe
because every index is block-local; transfer adapters must request torch.long
as the frozen executor already does. All grouping preserves ascending row order.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from research.repairs.benchmark_recovery_20261020.training.layout import block_indices


def _array(value):
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise ValueError("CPU immutable observation arrays required")
        return value.detach().numpy()
    return np.asarray(value)


def _immutable(value):
    value = np.asarray(value, dtype=np.int32)
    return np.frombuffer(value.tobytes(order="C"), dtype=np.int32).reshape(value.shape)


def _token(value):
    if hasattr(value, "_version"):
        return (value.data_ptr(), value._version, tuple(value.shape),
                tuple(value.stride()), str(value.dtype), str(value.device))
    value = np.asarray(value)
    return (id(value), value.shape, value.strides, value.dtype.str)


def _stamp(bank, counts, member_hold):
    return (tuple(counts), _token(bank.parent), _token(bank.sample_index),
            tuple(_token(x) for x in bank.upward_levels),
            tuple(_token(x) for x in bank.downward_levels), _token(member_hold))


def structural_digest(bank, counts, member_hold, holds_per_block):
    """Content key for one-time binding after restoring an immutable bank."""
    digest = hashlib.sha256()
    digest.update(str(holds_per_block).encode())
    values = [np.asarray(counts, dtype=np.int64), _array(bank.parent),
              _array(bank.sample_index), _array(member_hold),
              *[_array(x) for x in bank.upward_levels],
              np.array([-7654321], dtype=np.int64),
              *[_array(x) for x in bank.downward_levels]]
    for value in values:
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LayoutBlock:
    first_hold: int
    last_hold: int
    first_node: int
    last_node: int
    parent: np.ndarray
    upward_levels: tuple
    upward_children: tuple
    upward_parent_rank: tuple
    downward_levels: tuple
    first_member: int
    last_member: int

    def executor_tuple(self):
        return (self.first_hold, self.last_hold, self.first_node, self.last_node,
                self.parent, self.upward_levels, self.upward_children,
                self.upward_parent_rank, self.downward_levels,
                self.first_member, self.last_member)


@dataclass(frozen=True)
class BoundLayout:
    plan: object
    observation_stamp: tuple

    def assert_binding(self, bank, counts, member_hold):
        """O(number of levels), no scan or rebuilt topology on repeated steps."""
        if self.observation_stamp != _stamp(bank, counts, member_hold):
            raise ValueError("Layout binding changed; restore and content-rebind explicitly")
        return self.plan.executor_layout()


@dataclass(frozen=True)
class LayoutPlan:
    blocks: tuple
    holds_per_block: int
    structural_sha256: str
    native_holds: int
    native_nodes: int

    @classmethod
    def build(cls, bank, counts, member_hold, *, holds_per_block=64):
        counts = tuple(int(x) for x in counts)
        parent, sample = _array(bank.parent), _array(bank.sample_index)
        members = _array(member_hold)
        if (not counts or not 1 <= holds_per_block <= 256
                or any(x <= 0 for x in counts)
                or sum(counts) != len(parent)
                or sample.ndim != 2 or len(sample) != len(counts)
                or members.ndim != 1
                or np.any(members[1:] < members[:-1])):
            raise ValueError("Complete ordered native node/sample/member partition required")
        offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        n = len(parent)
        height = np.empty(n, dtype=np.int16)
        depth = np.zeros(n, dtype=np.int16)
        assigned = np.zeros(n, dtype=bool)
        for k, rows in enumerate(bank.upward_levels):
            rows = _array(rows)
            if k >= 32767 or np.any(assigned[rows]):
                raise ValueError("Original unique finite upward levels required")
            height[rows] = k
            assigned[rows] = True
        if not assigned.all():
            raise ValueError("Upward levels must cover every native node")
        for k, rows in enumerate(bank.downward_levels, start=1):
            if k >= 32767:
                raise ValueError("Original finite downward levels required")
            depth[_array(rows)] = k
        blocks = []
        for a in range(0, len(counts), holds_per_block):
            b = min(a + holds_per_block, len(counts))
            lo, hi = int(offsets[a]), int(offsets[b])
            if hi - lo >= 2**31:
                raise ValueError("Block-local int32 index bound exceeded")
            p = parent[lo:hi].copy()
            nonroot = p >= 0
            if (np.any(p[nonroot] < lo) or np.any(p[nonroot] >= hi)
                    or np.any(sample[a:b] < lo) or np.any(sample[a:b] >= hi)):
                raise ValueError("Every original edge/sample must remain within its block")
            p[nonroot] -= lo
            levels, children, ranks, downward = block_indices(
                p, height[lo:hi], depth[lo:hi])
            ml, mr = (int(x) for x in np.searchsorted(members, [a, b]))
            blocks.append(LayoutBlock(a, b, lo, hi, _immutable(p),
                tuple(_immutable(x) for x in levels),
                tuple(_immutable(x) for x in children),
                tuple(_immutable(x) for x in ranks),
                tuple(_immutable(x) for x in downward), ml, mr))
        return cls(tuple(blocks), holds_per_block,
                   structural_digest(bank, counts, member_hold, holds_per_block),
                   len(counts), n)

    def executor_layout(self):
        return tuple(block.executor_tuple() for block in self.blocks)

    def bind(self, bank, counts, member_hold):
        """One content scan when a compact bank is restored to different storage."""
        if structural_digest(bank, counts, member_hold,
                             self.holds_per_block) != self.structural_sha256:
            raise ValueError("Restored observation topology does not match layout plan")
        return BoundLayout(self, _stamp(bank, counts, member_hold))

    @property
    def array_bytes(self):
        total = 0
        for z in self.blocks:
            for value in (z.parent, *z.upward_levels, *z.upward_children,
                          *z.upward_parent_rank, *z.downward_levels):
                total += value.nbytes
        return total

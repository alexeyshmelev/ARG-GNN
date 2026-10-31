"""H033 efficiency hypothesis: repack frozen 32-readout forests eight at a time.

Only block boundaries change. The frozen tree solve, TV operator, model math,
parameters and state-dict keys are reused. No speed or accuracy claim is made.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from .interleaved_tree_tv_gnn_v1 import (
    InterleavedTreeTVBankV1,
    InterleavedTreeTVGNNV1,
    PackedTreeBlockV1,
    prepare_interleaved_tree_tv_bank_v1,
)

SOURCE_READOUT_BLOCK_SIZE = 32
REPACK_FACTOR = 8
READOUT_BLOCK_SIZE = SOURCE_READOUT_BLOCK_SIZE * REPACK_FACTOR


def _merge_tree_blocks_v1(blocks: tuple[PackedTreeBlockV1, ...]) -> PackedTreeBlockV1:
    """Concatenate complete forests and merge equal-depth elimination schedules."""
    parts = {name: [] for name in (
        "parent", "roots", "query_nodes", "source_mask", "conductance",
        "weighted_degree", "readout_indexes", "origin_node_indexes",
    )}
    schedules = {name: [] for name in (
        "upward_levels", "upward_children", "downward_levels",
    )}
    cursor = 0
    for block in blocks:
        # Negative root sentinels are values, not local node indexes.
        parts["parent"].append(torch.where(
            block.parent >= 0, block.parent + cursor, block.parent))
        for name in ("roots", "query_nodes"):
            parts[name].append(getattr(block, name) + cursor)
        for name in ("source_mask", "conductance", "weighted_degree",
                     "readout_indexes", "origin_node_indexes"):
            parts[name].append(getattr(block, name))
        for name, levels in schedules.items():
            for depth, nodes in enumerate(getattr(block, name)):
                if depth == len(levels):
                    levels.append([])
                levels[depth].append(nodes + cursor)
        cursor += block.parent.numel()
    merged = {name: torch.cat(values) for name, values in parts.items()}
    merged.update({name: tuple(torch.cat(level) for level in levels)
                   for name, levels in schedules.items()})
    return replace(blocks[0], **merged)


def repack_interleaved_tree_tv_bank_v1(
        bank: InterleavedTreeTVBankV1) -> InterleavedTreeTVBankV1:
    """Merge every eight original ragged blocks, retaining a complete final tail.

    The input is the ordered frozen 32-readout packing. All node rows survive,
    readout and global origin indexes retain their values and order, and the
    reference, TV and age-group tensors remain the same objects.
    """
    if (not bank.blocks or bank.metadata.get("readout_block_size")
            != SOURCE_READOUT_BLOCK_SIZE):
        raise ValueError("A nonempty frozen 32-readout bank is required")
    sizes = [block.readout_indexes.numel() for block in bank.blocks]
    if (any(size != SOURCE_READOUT_BLOCK_SIZE for size in sizes[:-1])
            or not 0 < sizes[-1] <= SOURCE_READOUT_BLOCK_SIZE
            or sum(sizes) != bank.readout_count):
        raise ValueError("Complete ordered 32-readout blocks and one optional tail required")
    indexes = torch.cat([block.readout_indexes for block in bank.blocks])
    expected = torch.arange(bank.readout_count, device=indexes.device, dtype=indexes.dtype)
    if not torch.equal(indexes, expected):
        raise ValueError("Original readout indexes must be consecutive and ordered")
    blocks = tuple(_merge_tree_blocks_v1(bank.blocks[start:start + REPACK_FACTOR])
                   for start in range(0, len(bank.blocks), REPACK_FACTOR))
    readout_sizes = [block.readout_indexes.numel() for block in blocks]
    vertex_sizes = [block.parent.numel() for block in blocks]
    metadata = dict(bank.metadata,
        packed_block_count=len(blocks), readout_block_size=READOUT_BLOCK_SIZE,
        packed_block_readout_sizes=readout_sizes, packed_block_vertex_sizes=vertex_sizes,
        replicated_vertices=sum(vertex_sizes), maximum_packed_vertices=max(vertex_sizes),
        source_readout_block_size=SOURCE_READOUT_BLOCK_SIZE, repack_factor=REPACK_FACTOR,
        packing_revision="interleaved_tree_tv_repack_v1")
    return replace(bank, blocks=blocks, metadata=metadata)


def prepare_interleaved_tree_tv_repack_bank_v1(*args, **kwargs) -> InterleavedTreeTVBankV1:
    """Prepare with the frozen source/cut policy, then change only block boundaries."""
    return repack_interleaved_tree_tv_bank_v1(
        prepare_interleaved_tree_tv_bank_v1(*args, **kwargs))


class InterleavedTreeTVRepackGNNV1(InterleavedTreeTVGNNV1):
    """Explicit 256-readout model declaration with identical inherited operators.

    The parent constructor creates the same 52 parameters under the same names.
    Its frozen 32-only admission is used internally; this revision declares 256
    after construction. No module, function or class is patched at runtime.
    """
    def __init__(self, *, initial_tree_strength=.1, initial_tv_strength=.1,
                 readout_block_size=READOUT_BLOCK_SIZE, checkpoint_tree=True):
        if readout_block_size != READOUT_BLOCK_SIZE:
            raise ValueError("This repacking revision requires declared 256-readout blocks")
        super().__init__(initial_tree_strength=initial_tree_strength,
                         initial_tv_strength=initial_tv_strength,
                         readout_block_size=SOURCE_READOUT_BLOCK_SIZE,
                         checkpoint_tree=checkpoint_tree)
        self.readout_block_size = READOUT_BLOCK_SIZE

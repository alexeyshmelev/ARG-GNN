"""Exact original windows with one advancing native-tree cursor per worker."""
from __future__ import annotations
from arggnn.benchmarking.paper_gst_lai_runtime_v2 import dataset as original

Window = original.Window
from_native_trees = original.from_native_trees
load_parents = original.load_parents
loader = original.loader
ParentOrderSampler = original.ParentOrderSampler


class CompleteTreeSets(original.CompleteTreeSets):
    """Keep the original parent sampler and deterministic eight-tree encoding.

    A map-style DataLoader worker receives an ascending strided subsequence.
    Advancing across intervening trees avoids allocating and seeking a new Tree
    from its null state on every window. Only requested windows are encoded.
    There is one live cursor per worker, not one cursor per parent.
    """
    def __init__(self, states, records):
        super().__init__(states, records)
        self._cursor_parent = None
        self._cursor = None
        self.cursor_counts = dict(starts=0, next_calls=0, reverse_seeks=0, windows=0)

    def __getstate__(self):
        state = super().__getstate__()
        state["_cursor_parent"] = None
        state["_cursor"] = None
        state["cursor_counts"] = dict(starts=0, next_calls=0, reverse_seeks=0, windows=0)
        return state

    def __getitem__(self, index):
        parent, start = self.windows[index]
        state = self.states[parent]
        ts = state["ts"]
        if parent not in self.workspaces:
            self.workspaces[parent] = original.observations.native_workspace_v1(ts)
        if self._cursor_parent != parent or self._cursor is None or self._cursor.index == -1:
            self._cursor_parent = parent
            self._cursor = ts.first()
            self.cursor_counts["starts"] += 1
        tree = self._cursor
        if tree.index > start:
            # Random-access callers retain exact semantics. The production
            # parent sampler never needs a reverse seek within a parent pass.
            tree.seek_index(start)
            self.cursor_counts["reverse_seeks"] += 1
        while tree.index < start:
            original.io._need(tree.next(), "Requested original native tree must exist")
            self.cursor_counts["next_calls"] += 1
        original.io._need(tree.index == start, "Exact original native window start")
        trees, intervals = [], []
        for offset in range(min(8, ts.num_trees-start)):
            original.io._need(tree.index == start+offset, "Consecutive complete native trees")
            trees.append(original.observations.native_tree(tree, state["nodes"],
                node_time=state["node_time"], workspace=self.workspaces[parent]))
            intervals.append(tuple(tree.interval))
            tree.next()
            self.cursor_counts["next_calls"] += 1
        self.cursor_counts["windows"] += 1
        return from_native_trees(parent, start, trees, intervals, state, self.records[parent])

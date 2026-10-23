"""Compact observed-only 25-reference + 3-ancestor ARGMix examples.

Distances are directed, measured from row-node age to the pair MRCA. Donor
ranking uses unquantized TMRCA and stable identities before any distance cap.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
import numpy as np
from . import paper_argmix_generator as g


class ScopeGuard:
    """A finite local failstop; Root must separately admit and supervise the job."""
    def __init__(self, scope, maximum_bytes, timeout_seconds):
        self.scope = g.confined(scope)
        g.need(maximum_bytes > 0 and timeout_seconds > 0, "Finite scope and deadline required")
        self.maximum_bytes = maximum_bytes
        self.deadline = time.monotonic() + timeout_seconds

    def __call__(self, required_write_bytes=0):
        g.need(time.monotonic() <= self.deadline, "Stage deadline exceeded")
        used = 0
        if self.scope.exists():
            for path in self.scope.rglob("*"):
                g.need(not path.is_symlink(), "Output scope contains a symlink")
                try:
                    if path.is_file():
                        used += path.stat().st_size
                except FileNotFoundError:
                    # Native Relate can retire its own transient stage files.
                    continue
        g.need(used + required_write_bytes <= self.maximum_bytes, "Stage output cap exceeded")


class LCA:
    """Vectorized binary lifting with an explicit disconnected sentinel."""
    def __init__(self, tree):
        self.n = tree.tree_sequence.num_nodes
        # Relate sequences can have millions of global nodes. Only the few
        # thousand active nodes in this tree enter the lifting table.
        self.nodes = np.sort(np.fromiter(tree.nodes(), dtype=np.int64))
        self.global_ids = np.append(self.nodes,self.n)
        parent_global = np.asarray(tree.parent_array[self.nodes],dtype=np.int64)
        parent_global = np.where(parent_global < 0,self.n,parent_global)
        parent = np.append(np.searchsorted(self.nodes,parent_global),len(self.nodes))
        self.up = [parent]
        jump = parent.copy()
        depth = (np.arange(len(parent)) != parent).astype(np.int64)
        while np.any(jump[jump] != jump):
            depth = depth + depth[jump]
            jump = jump[jump]
            self.up.append(jump.copy())
        self.depth = depth
        self.times = np.append(tree.tree_sequence.nodes_time[self.nodes],np.inf)

    def local(self, values):
        values = np.asarray(values,dtype=np.int64)
        local = np.searchsorted(self.nodes,values)
        g.need(np.all(self.global_ids[local] == values),"Inactive tree node requested")
        return local

    def node_time(self, values):
        return self.times[self.local(values)]

    def parent(self, values):
        return self.global_ids[self.up[0][self.local(values)]]

    def mrca(self, left, right):
        left, right = np.broadcast_arrays(left, right)
        a, b = self.local(left),self.local(right)
        swap = self.depth[a] < self.depth[b]
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        delta = self.depth[a] - self.depth[b]
        for bit, ancestor in enumerate(self.up):
            a = np.where((delta & (1 << bit)) != 0, ancestor[a], a)
        for ancestor in reversed(self.up):
            change = (a != b) & (ancestor[a] != ancestor[b])
            a, b = np.where(change, ancestor[a], a), np.where(change, ancestor[b], b)
        return self.global_ids[np.where(a == b, a, self.up[0][a])]


def quantize_positive(values):
    values = np.asarray(values)
    g.need(np.all(np.isfinite(values)) and np.all(values >= 0), "Finite nonnegative times required")
    return np.floor(values + 0.5).astype(np.int64)


def capped_distance(row_times, mrca_times, cap=1000, mode="linear"):
    delta = np.abs(quantize_positive(row_times) - quantize_positive(mrca_times))
    if mode == "log2":
        delta = np.floor(np.log2(1.0 + delta)).astype(np.int64)
    else:
        g.need(mode == "linear", "Unknown distance transform")
    return np.minimum(delta, cap).astype(np.uint16)


def feature_batch(tree, roster, focal_indices, features):
    """No truth paths, labels, populations or target ancestry are accepted here."""
    lca = LCA(tree)
    node_by_sample = np.asarray([r["inferred_tree_node_id"] for r in roster], dtype=np.int64)
    donors = [i for i, r in enumerate(roster) if r["reference_eligible"]]
    donors.sort(key=lambda i: roster[i]["stable_haplotype_id"])
    donor_nodes = node_by_sample[donors]
    focal_indices = np.asarray(focal_indices, dtype=np.int64)
    focal_nodes = node_by_sample[focal_indices]
    ancestry = lca.mrca(focal_nodes[:, None], donor_nodes[None, :])
    raw = lca.node_time(ancestry)
    for i, focal in enumerate(focal_indices):
        eligible = np.asarray([roster[d]["sample_id"] != roster[focal]["sample_id"] for d in donors])
        g.need(np.count_nonzero(eligible) >= features["num_references"] and np.all(np.isfinite(raw[i, eligible])),
               "Insufficient connected eligible donors")
        raw[i, ~eligible] = np.inf
    # Donors are already sorted by stable identity. Stable sorting preserves ties.
    selected = np.argsort(raw, axis=1, kind="stable")[:, :features["num_references"]]
    reference_samples = np.asarray(donors)[selected]
    reference_nodes = node_by_sample[reference_samples]
    batch, width = len(focal_indices), features["genealogy_tokens"]
    nodes = np.full((batch, width), lca.n, dtype=np.int64)
    tokens = np.full((batch, width), features["tokens"]["PAD"], dtype=np.uint8)
    nodes[:, 0], tokens[:, 0] = focal_nodes, features["tokens"]["SAMPLE"]
    stop = 1 + features["num_references"]
    nodes[:, 1:stop] = reference_nodes
    for i in range(batch):
        for j, donor in enumerate(reference_samples[i]):
            tokens[i, 1+j] = features["tokens"][roster[donor]["observed_cohort"]]
    ancestor = focal_nodes.copy()
    for column in range(stop, width):
        ancestor = lca.parent(ancestor)
        valid = ancestor != lca.n
        nodes[:, column] = ancestor
        tokens[valid, column] = features["tokens"]["ANC"]
    valid = nodes != lca.n
    pair_valid = valid[:, :, None] & valid[:, None, :]
    mrca = lca.mrca(nodes[:, :, None], nodes[:, None, :])
    g.need(np.all(mrca[pair_valid] != lca.n), "Disconnected selected nodes")
    # Sentinel values are replaced only for the ignored PAD pairs.
    times = np.where(valid, lca.node_time(nodes), 0)
    mrca_times = np.where(pair_valid, lca.node_time(mrca), 0)
    distance = capped_distance(times[:, :, None], mrca_times,
                               features["max_rel_distance"], features["distance_mode"])
    distance[~pair_valid] = 0
    stored_nodes = np.where(valid, nodes, -1).astype(np.int32)
    return tokens, distance, stored_nodes


def read_targets(path):
    with g.confined(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    g.need(all(int(r["target_row"]) == i for i, r in enumerate(rows)), "Target row order differs")
    return rows


def extract(config_path, parent_id, inferred_output, support_output, output, *, guard):
    """Write only frozen sampled examples. No all-interval native archive is built."""
    config, parent = g.parent_record(config_path, parent_id)
    import tskit
    bridge_path = g.confined(inferred_output) / "completed.json"
    support_path = g.confined(support_output) / "completed.json"
    bridge, support = g.metadata(bridge_path), g.metadata(support_path)
    g.need(bridge["parent_id"] == support["parent_id"] == parent_id and
           bridge["config_ref"] == support["config_ref"] == g.file_ref(config_path) and
           support["truth_or_labels_read"] is False and support["inferred_bridge_ref"] == g.file_ref(bridge_path),
           "Exact observed-only parent and target provenance required")
    ts = tskit.load(g.verify_ref(bridge["tree_ref"]))
    roster = g.metadata(g.verify_ref(bridge["roster_ref"]))
    targets = read_targets(g.verify_ref(support["targets_ref"]))
    n, width = len(targets), config["features"]["genealogy_tokens"]
    g.need(n == parent["planned_examples"], "Frozen example count differs")
    g.need(width == 1 + config["features"]["num_references"] + config["features"]["num_ancestors"], "Geometry mismatch")
    output = g._fresh_directory(output, guard)
    expected_bytes = n * (width + 2*width*width + 4*width) + 3*128
    g.need(expected_bytes + 1024*1024 <= config["bounds"]["compact_parent_bytes"], "Compact parent cap too small")
    guard(expected_bytes)
    tokens = np.lib.format.open_memmap(output/"tokens.npy", mode="w+", dtype=np.uint8, shape=(n,width))
    distances = np.lib.format.open_memmap(output/"distances.npy", mode="w+", dtype=np.uint16, shape=(n,width,width))
    nodes = np.lib.format.open_memmap(output/"nodes.npy", mode="w+", dtype=np.int32, shape=(n,width))
    order = sorted(range(n), key=lambda i: (int(targets[i]["inferred_tree_index"]), i))
    tree, cursor = ts.first(), 0
    started = time.monotonic()
    while cursor < n:
        tree_index = int(targets[order[cursor]]["inferred_tree_index"])
        tree.seek_index(tree_index)
        end = cursor
        while end < n and int(targets[order[end]]["inferred_tree_index"]) == tree_index:
            end += 1
        for chunk_start in range(cursor, end, config["bounds"]["feature_chunk_examples"]):
            indices = order[chunk_start:min(end,chunk_start+config["bounds"]["feature_chunk_examples"])]
            for index in indices:
                row = targets[index]
                sample = roster[int(row["sample_index"])]
                position = float(row["local_midpoint_bp"])
                g.need(tree.interval.left <= position < tree.interval.right and sample["query_eligible"] and
                       row["sample_id"] == sample["sample_id"] and int(row["haplotype"]) == sample["haplotype"] and
                       row["cohort"] == sample["observed_cohort"], "Target coordinate or identity mismatch")
            x, d, selected = feature_batch(tree, roster, [int(targets[i]["sample_index"]) for i in indices], config["features"])
            tokens[indices], distances[indices], nodes[indices] = x,d,selected
            guard(0)
        cursor = end
    for array in (tokens,distances,nodes):
        array.flush()
    receipt = dict(schema="paper_argmix_compact_features_v1", parent_id=parent_id, split=parent["split"],
                   config_ref=g.file_ref(config_path), source_ref=g.file_ref(__file__),
                   inferred_bridge_ref=g.file_ref(bridge_path), support_ref=g.file_ref(support_path),
                   examples=n, geometry=config["features"],
                   arrays={name:g.file_ref(output/(name+".npy")) for name in ("tokens","distances","nodes")},
                   dtype={"tokens":"uint8","distances":"uint16","nodes":"int32"},
                   truth_read=False, merging=False, seconds=time.monotonic()-started)
    g.Writes(1024*1024,guard).json(output/"completed.json",receipt)
    return receipt

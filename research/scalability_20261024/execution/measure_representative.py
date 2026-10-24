"""Bounded observed TRAIN slice measurement; no model/training launch or truth I/O."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import resource
import statistics
import time
from types import SimpleNamespace
import numpy as np
import torch
import tskit
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
from research.repairs.benchmark_recovery_20261020.runtime import observations
from research.scalability_20261024.execution.layout_plan import LayoutPlan

OUT = Path("research/scalability_20261024/execution")
MANIFEST = Path("experiments/results/benchmark_campaign_20261018/fresh_american/argformer_american_train_000/inference/observed/manifest.json")
HOLDS = 128


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sparse_native(tree, sample_nodes, node_time):
    """Exact original preorder with local search arrays, no global rank/stamp."""
    nodes = np.asarray(list(tree.nodes()), dtype=np.int64)
    order = np.argsort(nodes)
    sorted_ids = nodes[order]
    raw_parent = tree.parent_array[nodes]
    nonroot = raw_parent >= 0
    parent = np.full(len(nodes), -1, dtype=np.int64)
    parent[nonroot] = order[np.searchsorted(sorted_ids, raw_parent[nonroot])]
    sample = order[np.searchsorted(sorted_ids, sample_nodes)].copy()
    if not np.array_equal(nodes[sample], sample_nodes):
        raise ValueError("Every exact native sample must remain active")
    return NativeObservedTreeV1(parent, node_time[nodes].copy(), sample,
                                np.arange(len(sample), dtype=np.int64))


def baseline_layout(bank, counts, member_hold, holds_per_block=64):
    """Plaintext reference of frozen bounded_execution_v2 constructor layout."""
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    parent, sample = bank.parent.numpy(), bank.sample_index.numpy()
    height = np.empty(len(parent), dtype=np.int16)
    depth = np.zeros(len(parent), dtype=np.int16)
    for k, rows in enumerate(bank.upward_levels):
        height[rows.numpy()] = k
    for k, rows in enumerate(bank.downward_levels, start=1):
        depth[rows.numpy()] = k
    members = member_hold.numpy()
    result = []
    for a in range(0, len(counts), holds_per_block):
        b = min(a + holds_per_block, len(counts))
        lo, hi = int(offsets[a]), int(offsets[b])
        p = parent[lo:hi].copy()
        nonroot = p >= 0
        p[nonroot] -= lo
        ht, dp = height[lo:hi], depth[lo:hi]
        levels = tuple(np.flatnonzero(ht == k) for k in range(int(ht.max()) + 1))
        rank = np.empty(hi - lo, dtype=np.int64)
        for rows in levels:
            rank[rows] = np.arange(len(rows))
        child = np.flatnonzero(nonroot)
        children = tuple(child[ht[p[child]] == k] for k in range(len(levels)))
        ranks = tuple(rank[p[x]] for x in children)
        downward = tuple(np.flatnonzero(dp == k) for k in range(1, int(dp.max()) + 1))
        ml, mr = (int(x) for x in np.searchsorted(members, [a, b]))
        result.append((a, b, lo, hi, p, levels, children, ranks, downward, ml, mr))
    return tuple(result)


def array_bytes(layout):
    return sum(x.nbytes for z in layout for value in z
               for x in ((value,) if isinstance(value, np.ndarray)
                         else value if isinstance(value, tuple) else ())
               if isinstance(x, np.ndarray))


def assert_same(a, b):
    assert len(a) == len(b)
    for old, new in zip(a, b):
        for x, y in zip(old, new):
            if isinstance(x, np.ndarray):
                assert np.array_equal(x, y)
            elif isinstance(x, tuple):
                assert len(x) == len(y)
                assert all(np.array_equal(xx, yy) for xx, yy in zip(x, y))
            else:
                assert x == y


def bench(fn, repeats):
    values = []
    for _ in range(repeats):
        t = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - t)
    return result, values


def dependency_ids(bank, counts, site_global_start):
    """Strict source identity; no reuse based only on current numeric states."""
    parent = bank.parent.numpy()
    times = bank.node_time.numpy()
    root = parent < 0
    sample = bank.sample_index.numpy()
    spans = bank.spans.numpy()
    site_hold = bank.site_hold.numpy()
    ids = np.empty(len(parent), dtype=np.int64)
    key_ids = {}
    for hold, count in enumerate(counts):
        lo = sum(counts[:hold])
        hi = lo + count
        columns = {int(node): col for col, node in enumerate(sample[hold])}
        site_lo, site_hi = np.searchsorted(site_hold, [hold, hold + 1])
        for node in range(lo, hi):
            key = ("node_base", times[node].tobytes(), bool(root[node]), node in columns)
            if node in columns:
                col = columns[node]
                key += ("site_source", int(site_lo) + site_global_start,
                        int(site_hi) + site_global_start, spans[hold].tobytes(),
                        col, bank.reference_class[col].numpy().tobytes(),
                        bank.sample_age[col].numpy().tobytes(),
                        "global256_chunks", int(site_lo + site_global_start) // 256)
            ids[node] = key_ids.setdefault(key, len(key_ids))
    gate_raw = np.concatenate((bank.edge_geometry.numpy(),
                               bank.branch_support_true.numpy()), axis=1)
    gate_ids = np.empty(len(parent), dtype=np.int64)
    seen = {}
    for i, row in enumerate(gate_raw):
        gate_ids[i] = seen.setdefault(row.tobytes(), len(seen))
    return ids, gate_ids


def subtree_census(parent, children, base_ids=None, gate_ids=None):
    """Exact ordered upward expression DAG keys; no outside/history claim."""
    ids = np.empty(len(parent), dtype=np.int64)
    seen = {}
    leaf_count = 0
    for node in range(len(parent) - 1, -1, -1):
        # Native tskit preorder guarantees every child follows its parent.
        child = children[node]
        if not child:
            leaf_count += 1
        if base_ids is None:
            key = ("shape", tuple(int(ids[c]) for c in child))
        else:
            key = ("up", int(base_ids[node]),
                   tuple((int(gate_ids[c]), int(ids[c])) for c in child))
        ids[node] = seen.setdefault(key, len(seen))
    return {"occurrences":len(parent), "unique_ordered_subexpressions":len(seen),
            "duplicate_occurrences":len(parent)-len(seen),
            "unique_fraction":len(seen)/len(parent), "leaf_occurrences":leaf_count}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["split"] == "train" and manifest["query_ancestry_truth_in_inputs"] is False
    tree_path = Path(manifest["tree_ref"]["path"])
    assert sha(tree_path) == manifest["tree_ref"]["sha256"]
    ts, load_times = bench(lambda: tskit.load(str(tree_path)), 1)
    samples = np.asarray(ts.samples(), dtype=np.int64)
    node_time = np.asarray(ts.nodes_time, dtype=np.float64)
    workspace = observations.native_workspace_v1(ts)
    def collect(native):
        trees, intervals, native_ids = [], [], []
        for tree in itertools.islice(ts.trees(), HOLDS):
            trees.append(native(tree))
            intervals.append(tuple(tree.interval))
            native_ids.append(np.asarray(list(tree.nodes()), dtype=np.int64))
        return trees, intervals, native_ids
    dense, dense_times = bench(lambda: collect(lambda t: observations.native_tree(
        t, samples, node_time=node_time, workspace=workspace)), 3)
    sparse, sparse_times = bench(lambda: collect(lambda t: sparse_native(
        t, samples, node_time)), 3)
    for a, b in zip(dense[0], sparse[0]):
        for f in dataclasses.fields(a):
            assert np.array_equal(getattr(a, f.name), getattr(b, f.name))
    trees, intervals, native_ids = dense
    counts = [len(t.parent) for t in trees]
    print(json.dumps({"phase":"native_slice_complete","holds":HOLDS,
                      "expanded_nodes":sum(counts),"dense_seconds":dense_times,
                      "sparse_seconds":sparse_times}), flush=True)
    positions = np.load(manifest["positions_ref"]["path"], mmap_mode="r")
    lo, hi = (int(x) for x in np.searchsorted(positions, [intervals[0][0], intervals[-1][1]]))
    genotypes = np.load(manifest["genotypes_ref"]["path"], mmap_mode="r")
    genotype = np.asarray(genotypes[lo:hi]).copy()
    allele_counts = []
    with Path(manifest["alleles_ref"]["path"]).open() as f:
        for i, line in enumerate(f):
            if i >= hi:
                break
            if i >= lo:
                allele_counts.append(len(json.loads(line)))
    roster = json.loads(Path(manifest["roster_ref"]["path"]).read_text())["samples"]
    labels = np.asarray([-1 if r["observed_reference_class"] is None
                         else r["observed_reference_class"] for r in roster], dtype=np.int64)
    ages = np.asarray([r["age_generations"] for r in roster], dtype=np.float64)
    start = time.perf_counter()
    bank, metadata = observations.encode_block(trees, intervals, genotype,
        np.asarray(allele_counts, dtype=np.int8), np.asarray(positions[lo:hi]),
        labels, ages, num_classes=3, age_split=350.,
        query_columns=np.flatnonzero(labels < 0))
    encode_seconds = time.perf_counter() - start
    members = torch.empty(0, dtype=torch.long)
    reference, reference_times = bench(lambda: baseline_layout(bank, counts, members), 5)
    plan, build_times = bench(lambda: LayoutPlan.build(
        bank, counts, members, holds_per_block=64), 5)
    assert_same(reference, plan.executor_layout())
    bound = plan.bind(bank, counts, members)
    _, reuse_times = bench(lambda: bound.assert_binding(bank, counts, members), 100)
    for block in plan.blocks:
        for value in (block.parent, *block.upward_levels, *block.upward_children,
                      *block.upward_parent_rank, *block.downward_levels):
            try:
                value.flags.writeable = True
            except ValueError:
                pass
            else:
                raise AssertionError("Plan must have immutable bytes backing")
    cloned = dataclasses.replace(bank, parent=bank.parent.clone())
    rebound = plan.bind(cloned, counts, members)
    assert_same(reference, rebound.assert_binding(cloned, counts, members))
    changed = dataclasses.replace(bank, parent=bank.parent.clone())
    changed.parent[0] = 0
    try:
        plan.bind(changed, counts, members)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed topology must be rejected")
    # A no-op tensor mutation still invalidates a fast bound through version.
    probe = dataclasses.replace(bank, parent=bank.parent.clone())
    probe_bound = plan.bind(probe, counts, members)
    probe.parent.add_(0)
    try:
        probe_bound.assert_binding(probe, counts, members)
    except ValueError:
        pass
    else:
        raise AssertionError("In-place mutation must invalidate fast binding")
    base_ids, gate_ids = dependency_ids(bank, counts, lo)
    parent = bank.parent.numpy()
    child_lists = [[] for _ in range(len(parent))]
    for c, p in enumerate(parent):
        if p >= 0:
            child_lists[int(p)].append(c)
    topology = subtree_census(parent, child_lists)
    full = subtree_census(parent, child_lists, base_ids, gate_ids)
    packet = {}
    for f in dataclasses.fields(bank):
        value = getattr(bank, f.name)
        if torch.is_tensor(value):
            packet[f.name] = value.numpy()
        elif isinstance(value, tuple):
            for j, x in enumerate(value):
                packet[f.name + "_" + str(j)] = x.numpy()
    packet.update(node_offsets=np.concatenate(([0],np.cumsum(counts))),
                  native_node_ids=np.concatenate(native_ids),
                  base_dependency_id=base_ids, gate_dependency_id=gate_ids,
                  site_global_index=np.arange(lo,hi,dtype=np.int64),
                  site_position_bp0=np.asarray(positions[lo:hi]),
                  intervals=np.asarray(intervals))
    packet_path = OUT/"representative_slice.npz"
    np.savez(packet_path, **packet)
    packet_meta = {
        "schema":"observed_TreeLoom_first_sweep_packet_v1",
        "policy":"strict_source_identity_lower_bound",
        "parent_id":manifest["parent_id"],"split":"train","native_hold_range":[0,HOLDS],
        "whole_parent_retained_in_frozen_run":True,
        "diagnostic_slice_only_not_training_or_data_change":True,
        "truth_opened":False,"model_executed":False,
        "source_manifest_ref":{"path":str(MANIFEST),"sha256":sha(MANIFEST)},
        "tree_ref":manifest["tree_ref"],"genotypes_ref":manifest["genotypes_ref"],
        "positions_ref":manifest["positions_ref"],"roster_ref":manifest["roster_ref"],
        "alleles_ref":manifest["alleles_ref"],
        "packet_ref":{"path":str(packet_path),"sha256":sha(packet_path)},
        "shapes_dtypes":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in packet.items()},
        "level_tuple_lengths":{f.name:len(getattr(bank,f.name)) for f in dataclasses.fields(bank)
                               if isinstance(getattr(bank,f.name),tuple)},
        "site_source_range":[lo,hi],"site_chunk_size":256,
        "site_global_chunk_alignment":lo % 256,
        "site_chunk_boundary_note":"Frozen full-parent execution uses global256-site chunks. This packet contains the exact source sites for these128holds; omitted neighboring-hold rows in a shared boundary chunk are not a site-encoder parity claim.",
        "base_dependency_policy":"Internal base IDs key exact node_time/root/sampled. Sampled nodes also key persistent column, exact site source range, span bytes, reference label/age sources and global256chunk alignment. No current-state numeric equality merges.",
        "gate_dependency_policy":"Exact bytes of immutable4 geometry +5 branch-support fields; gates are shared parameter functions. Learned states, history, second sweep and parameter updates are excluded.",
        "reduction_order":"Ascending native local node row within each original upward level; child rows as in encoder upward_children. Native IDs preserve tskit preorder.",
        "topology_only_census":topology,"strict_first_sweep_upward_census":full}
    (OUT/"representative_slice.json").write_text(json.dumps(packet_meta,indent=2)+"\n")
    results = {
        "schema":"TreeLoom_layout_reuse_measurement_v1",
        "observed_unix":time.time(),"parent_id":manifest["parent_id"],
        "whole_parent":{"global_nodes":ts.num_nodes,"marginal_trees":ts.num_trees,
                        "edges":ts.num_edges,"sites":ts.num_sites,"samples":ts.num_samples,
                        "tree_file_bytes":tree_path.stat().st_size},
        "bounded_slice":{"holds":HOLDS,"expanded_nodes":sum(counts),
                         "native_node_count_minmax":[min(counts),max(counts)],"sites":hi-lo},
        "ts_load_seconds":load_times,"legacy_dense_native_seconds":dense_times,
        "sparse_local_remap_seconds":sparse_times,
        "sparse_native_exact_fields":True,
        "dense_rank_stamp_workspace_bytes":ts.num_nodes*16,
        "sparse_decision":"Prefer measured lower median converter; no claim that dense workspace is American bottleneck.",
        "encode_observed_slice_seconds":encode_seconds,
        "legacy_layout_build_seconds":reference_times,
        "immutable_layout_cold_build_seconds":build_times,
        "bound_layout_reuse_seconds":reuse_times,
        "median_legacy_layout_seconds":statistics.median(reference_times),
        "median_cold_plan_seconds":statistics.median(build_times),
        "median_reuse_seconds":statistics.median(reuse_times),
        "legacy_layout_array_bytes":array_bytes(reference),
        "plan_array_bytes":plan.array_bytes,
        "plan_values_and_row_order_exact":True,
        "fast_binding_mutation_rejection":True,
        "restored_content_rebind_exact":True,
        "changed_content_rejected":True,
        "immutable_bytes_backing":True,
        "maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "topology_only_census":topology,"strict_first_sweep_upward_census":full,
        "source_refs":[{"path":p,"sha256":sha(p)} for p in (
            "research/benchmark_rebuild_20261024/american/treeloom_speed/bounded_execution_v2.py",
            "research/benchmark_rebuild_20261024/american/treeloom_speed/static_preparation_v2.py",
            "research/repairs/benchmark_recovery_20261020/runtime/observations.py",
            "research/repairs/benchmark_recovery_20261020/training/layout.py",
            "research/scalability_20261024/execution/layout_plan.py",
            "research/scalability_20261024/execution/measure_representative.py")],
        "limitations":["Only128contiguous TRAIN trees; rates are not full-parent speedup claims.",
                       "No GPU timing, full-parent build, model launch or live integration.",
                       "Layout plan requires retained cache or persisted sidecar across passes; it does not by itself remove expanded feature banks.",
                       "Sparse remap retains one tskit Tree cursor and its own dense internal arrays. It only removes custom rank/stamp workspaces."]}
    (OUT/"measurement.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({k:v for k,v in results.items() if "seconds" not in k or k.startswith("median")},indent=2),flush=True)


if __name__ == "__main__":
    main()

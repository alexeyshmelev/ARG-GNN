"""Tiny RAM-only equivalence against the pinned original native focal writer."""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import tskit
from research.reviews.TreeLoom_compact_original_focal_source_20261017_v1 import compact_focals_v1 as c


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def fixture_v1():
    tables = tskit.TableCollection(100.)
    for _ in range(4):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.)
    # IDs4/5 and8/9 are alternative clades; roots6/7 differ in time.
    for time in (1., 1., 2., 3., 1., 1.):
        tables.nodes.add_row(time=time)
    for left, right, kind in ((0., 10., "A"), (10., 30., "B"),
                             (30., 60., "A"), (60., 100., "C")):
        if kind == "A":
            edges = ((4, 0), (4, 2), (5, 1), (5, 3), (6, 4), (6, 5))
        elif kind == "B":
            edges = ((4, 0), (4, 2), (5, 1), (5, 3), (7, 4), (7, 5))
        else:
            edges = ((8, 0), (8, 3), (9, 1), (9, 2), (6, 8), (6, 9))
        for parent, child in edges:
            tables.edges.add_row(left, right, parent, child)
    tables.sort()
    ts = tables.tree_sequence()
    rows = [dict(sample_index=i, observed_node_id=i,
        role="pure_source_reference" if i < 2 else "query",
        observed_reference_class=i if i < 2 else None) for i in range(4)]
    return ts, rows


class OriginalFocalCaptured(Exception):
    pass


def original_focals_v1(ts, rows, cfg):
    """Execute original code unchanged through its focal NPZ write, then stop.

    Metadata and tskit.load are replaced only inside this fixture, with RAM
    objects. No actual observation, corpus, model or truth payload is read.
    """
    original = c.original_native_v1()
    saved_read, saved_bound, saved_load = original.read, original.bound, tskit.load
    captured = {}
    calls = []
    observation = dict(schema="american_c3_observed_v1", parent_id=cfg["parent_id"],
        class_order=list(c.CLASSES), genealogy_kind=c.GENEALOGY,
        query_ancestry_truth_in_inputs=False, roster_ref="RAM_ROSTER", tree_ref="RAM_TREE")
    roster = dict(parent_id=cfg["parent_id"], class_order=list(c.CLASSES), samples=rows)
    class CaptureWriter:
        def npz(self, name, **arrays):
            calls.append(name)
            need(name == "focals.npz", "First original scientific write must be focal NPZ")
            captured.update({key: value.copy() for key, value in arrays.items()})
            raise OriginalFocalCaptured()
    try:
        original.read = lambda record: observation if record == "RAM_OBSERVATION" else roster if record == "RAM_ROSTER" else (_ for _ in ()).throw(AssertionError("Unexpected metadata read"))
        original.bound = lambda record: Path("RAM_TREE") if record == "RAM_TREE" else (_ for _ in ()).throw(AssertionError("Unexpected payload reference"))
        tskit.load = lambda path: ts if path == "RAM_TREE" else (_ for _ in ()).throw(AssertionError("Unexpected tree load"))
        try:
            original.execute(dict(cfg, observation_ref="RAM_OBSERVATION",
                genealogy_kind=c.GENEALOGY, output_cap_bytes=10**9), CaptureWriter())
        except OriginalFocalCaptured:
            pass
        else:
            raise AssertionError("Expected exact original focal writer capture")
    finally:
        original.read, original.bound, tskit.load = saved_read, saved_bound, saved_load
    need(calls == ["focals.npz"] and len(captured) == 7, "Only bounded original focal output captured")
    return captured


def npz_bytes_v1(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def run_v1():
    ts, rows = fixture_v1()
    need(ts.num_trees == 4 and ts.num_samples == 4 and ts.num_sites == 0, "Tiny four-hold/four-sample RAM fixture")
    key = c.original_native_v1().author_functions()["get_path_key"]
    tree0, tree2 = ts.at_index(0), ts.at_index(2)
    need(all(key(tree0, i) == key(tree2, i) for i in range(4)), "Nonadjacent repeated full paths")
    results = []
    for seed, count in ((92001, 4), (92002, 5), (92020, 6)):
        cfg = dict(parent_id="RAM_original_focal_fixture", max_unique_paths=64,
            max_path_tokens=512, max_path_occurrences=16, max_vocabulary_nodes=32,
            max_raw_path_nodes=16, query_focal_count=count, path_sampling_seed=seed)
        expected = original_focals_v1(ts, rows, cfg)
        actual, statistics = c.focal_arrays_v1(ts, rows, cfg)
        need(list(expected) == list(actual), "Exact original NPZ field insertion order")
        for name in expected:
            a, b = expected[name], actual[name]
            need(a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes(order="C") == b.tobytes(order="C"), "Bitwise original array identity: " + name)
        expected_bytes, actual_bytes = npz_bytes_v1(expected), npz_bytes_v1(actual)
        need(expected_bytes == actual_bytes, "Same compressed NPZ bytes under the exact same serializer")
        need(c.identity_sha_v1(expected) == c.identity_sha_v1(actual), "Exact original focal identity hash")
        need(statistics == dict(unique_paths=12, path_tokens=36, path_occurrences=16,
            observed_vocabulary_node_count=10, unique_query_paths=6), "Complete traversal statistics")
        need((actual["record_ids"] >= 6).all(), "Reference paths consume global IDs before queries")
        need(np.all(np.diff(actual["record_ids"]) > 0), "Original sorted chosen global IDs")
        need(not np.isin(actual["positions_bp0"], [45.]).any(), "Repeated paths retain first occurrence, never later midpoint")
        results.append(dict(seed=seed, focal_records=count, all_seven_arrays_bitwise_equal=True,
            serialized_npz_bytes_equal=True, focal_identity_sha256=c.identity_sha_v1(actual),
            record_ids=actual["record_ids"].tolist(), positions_bp0=actual["positions_bp0"].tolist()))
    rejected = []
    base = dict(parent_id="RAM_original_focal_fixture", max_unique_paths=64,
        max_path_tokens=512, max_path_occurrences=16, max_vocabulary_nodes=32,
        max_raw_path_nodes=16, query_focal_count=4, path_sampling_seed=92001)
    for field, limit in (("max_unique_paths", 2), ("max_path_tokens", 2),
                         ("max_path_occurrences", 15), ("max_vocabulary_nodes", 2),
                         ("max_raw_path_nodes", 2), ("query_focal_count", 7)):
        try:
            c.focal_arrays_v1(ts, rows, dict(base, **{field: limit}))
        except ValueError:
            rejected.append(field)
        else:
            raise AssertionError("Whole-parent failstop must reject: " + field)
    fake_ref = dict(path="RAM_NOT_OPENED", sha256="0"*64)
    configs = [c.make_config_v1(c.TRAIN[i], fake_ref) for i in (0, 19)]
    need([v["path_sampling_seed"] for v in configs] == [92001, 92020]
        and all(v["execution_enabled"] is False for v in configs), "Original frozen parent seed endpoints; default inactive")
    return dict(schema="TreeLoom_compact_original_focal_RAM_equivalence_v1",
        status="PASS", cases=results, whole_parent_failstop_rejections=rejected,
        real_observation_payloads_read=False, native_corpus_payloads_read=False,
        truth_payloads_read=False, model_executed=False, training_executed=False,
        actual_resource_operations=False, actual_full_parent_equivalence_claimed=False)


if __name__ == "__main__":
    print(json.dumps(run_v1(), sort_keys=True, indent=2))

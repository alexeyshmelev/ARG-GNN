"""Compact TRAIN focal preparation with the original native path identities.

Root supplies the actual admitted CPU context and existing Writer. This module
has no launcher, accounting operation, retry, cleanup, truth resolver or GPU use.
The distinct completion describes focal preparation, never a native path corpus.
"""
from __future__ import annotations
import ast
import hashlib
import importlib
import json
from pathlib import Path

from arggnn.american_c3_common_v1 import ROOT, CLASSES, require, bound, read

HERE = "research/reviews/TreeLoom_compact_original_focal_source_20261017_v1"
GENEALOGY = "inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
TRAIN = tuple("american_c3_train_%03d" % i for i in range(20))
STATUS = "completed_TreeLoom_original_identity_focal_preparation_without_native_corpus"
CONFIG_SCHEMA = "TreeLoom_compact_original_focal_preparation_config_v1"
COMPLETION_SCHEMA = "TreeLoom_compact_original_focal_preparation_completion_v1"
ORIGINAL_NATIVE = dict(path="src/arggnn/american_c3_paths_v1.py",
    sha256="3b32f16e0a81160852e5b93fe9033abf9bc2eeb56c3b15ca552d73f4aacf374c")
ORIGINAL_AUTHOR = dict(path=".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py",
    sha256="e7e66a029694b21a2ba1f9c9a6c4fbc193d81b8831282580ba9639a9e224726c")
ORIGINAL_IDENTITY = dict(path="src/arggnn/american_c3_gnn_targets_v1.py",
    sha256="065160f5d1e05ed00a663c1a374116fbc4f46bfc9446869185e891630689ce46")
ORIGINAL_PIPELINE_MANIFEST = dict(
    path="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2/source_manifest.v1.json",
    sha256="2c9a08835aa60dafa0a25ccf51ac1933655797e89f9e1aaa49a25ff97d8d9e8b")
ORIGINAL_LIMITS = dict(max_path_occurrences=153726000, max_path_tokens=395840004,
    max_raw_path_nodes=1024, max_unique_paths=31968234, max_vocabulary_nodes=133462)
FOCALS = 10000
FOCAL_NPZ_BOUND_BYTES = 2**20
PRIMARY_OUTPUT_CAP_BYTES = 2**21


def original_native_v1():
    """Use the pinned original author function extractor, without executing it."""
    expected = bound(ORIGINAL_NATIVE)
    bound(ORIGINAL_AUTHOR)
    native = importlib.import_module("arggnn.american_c3_paths_v1")
    require(Path(native.__file__).resolve() == expected,
        "Pinned original native path source, no substituted module")
    return native


def make_config_v1(parent_id, observation_ref, *, execution_enabled=False):
    """Freeze original parent seed and reject changes to original failstop limits."""
    require(parent_id in TRAIN and type(execution_enabled) is bool,
        "Only the original twenty TRAIN parents")
    condition = read(ORIGINAL_PIPELINE_MANIFEST)["condition"]
    template = condition["stage_templates"]["native_paths"]
    require({k: template[k] for k in ORIGINAL_LIMITS} == ORIGINAL_LIMITS
        and template["query_focal_count"] == FOCALS
        and template["genealogy_kind"] == GENEALOGY,
        "Unchanged original native traversal failstop limits")
    matches = [r for r in condition["registry"]["parents"] if r["parent_id"] == parent_id]
    require(len(matches) == 1 and matches[0]["split"] == "train",
        "Exact original registered TRAIN parent")
    index = TRAIN.index(parent_id)
    require(matches[0]["path_sampling_seed"] == 92001 + index,
        "Original fixed parent focal RNG seed")
    require(isinstance(observation_ref, dict) and set(observation_ref) == {"path", "sha256"}
        and isinstance(observation_ref["path"], str)
        and isinstance(observation_ref["sha256"], str)
        and len(observation_ref["sha256"]) == 64,
        "Canonical observation ref shape; actual payload authenticated only on execute")
    return dict(schema=CONFIG_SCHEMA, execution_enabled=execution_enabled,
        parent_id=parent_id, observation_ref=dict(observation_ref),
        genealogy_kind=GENEALOGY, **ORIGINAL_LIMITS, query_focal_count=FOCALS,
        path_sampling_seed=matches[0]["path_sampling_seed"],
        output_cap_bytes=PRIMARY_OUTPUT_CAP_BYTES)


def focal_arrays_v1(ts, rows, cfg):
    """Original first-seen full-path IDs and focal arrays; no corpus serialization.

    This generic RAM core is also the tiny fixture entry. Production execute
    separately requires the exact full30Mb/1600 original TRAIN observation.
    Every bound rejects the entire parent; none samples or truncates traversal.
    """
    import numpy as np
    for name in (*ORIGINAL_LIMITS, "query_focal_count"):
        require(type(cfg[name]) is int and cfg[name] > 0, "Positive finite original traversal bound")
    require(type(cfg["path_sampling_seed"]) is int, "Explicit dedicated focal seed")
    require([r["sample_index"] for r in rows] == list(range(len(rows)))
        and list(map(int, ts.samples())) == [r["observed_node_id"] for r in rows],
        "Original ordered sampled identity")
    require(ts.num_nodes < 2**31
        and ts.num_samples * ts.num_trees <= cfg["max_path_occurrences"],
        "Original full-parent occurrence and32bit node failstops")
    f = original_native_v1().author_functions()
    keys = {}
    seqs = []
    first = []
    tokens = 0
    observed_nodes = set()
    # Keep this exact original samples-outer / trees-inner traversal.
    # All reference paths consume global IDs before subsequent query paths.
    for sample in ts.samples():
        for tree in ts.trees():
            key = f["get_path_key"](tree, int(sample))
            require(len(key) <= cfg["max_raw_path_nodes"],
                "Complete raw path length cap; no truncated deduplication")
            if key not in keys:
                require(len(seqs) < cfg["max_unique_paths"],
                    "Unique-path cap; no sampling/subset fallback")
                tokens += len(key)
                require(tokens <= cfg["max_path_tokens"], "Complete path-token cap")
                keys[key] = len(seqs)
                seqs.append(list(key))
                first.append(tree.index)
                observed_nodes.update(key)
    # The original vocabulary is sorted(unique raw node IDs). Count the same
    # set to preserve its failstop without materializing token strings or vocab.
    require(len(observed_nodes) <= cfg["max_vocabulary_nodes"],
        "Original observed vocabulary node-count failstop")
    sample_meta = {r["observed_node_id"]: r for r in rows}
    trees = list(ts.breakpoints())
    query_ids = [keys[tuple(s)] for s in seqs if sample_meta[s[0]]["role"] == "query"]
    require(len(query_ids) >= cfg["query_focal_count"], "Enough complete unique query paths")
    chosen = np.random.default_rng(cfg["path_sampling_seed"]).choice(
        np.asarray(query_ids, dtype=np.int64), size=cfg["query_focal_count"], replace=False)
    chosen.sort()
    samples = np.asarray([seqs[int(i)][0] for i in chosen], dtype=np.int64)
    left = np.asarray([trees[first[int(i)]] for i in chosen], dtype=np.float64)
    right = np.asarray([trees[first[int(i)] + 1] for i in chosen], dtype=np.float64)
    positions = (left + right) / 2
    # Same field insertion order, dtype, shape and values as original execute.
    arrays = dict(record_ids=chosen, sample_nodes=samples,
        sample_indices=np.asarray([sample_meta[int(u)]["sample_index"] for u in samples], dtype=np.int64),
        parent_id=np.asarray(cfg["parent_id"]), class_order=np.asarray(CLASSES),
        positions_bp0=positions, first_occurrence_intervals_bp=np.column_stack((left, right)))
    require(sum(a.nbytes for a in arrays.values()) + 65536 <= FOCAL_NPZ_BOUND_BYTES,
        "Bounded complete focal NPZ, no reduced focal set")
    statistics = dict(unique_paths=len(seqs), path_tokens=tokens,
        path_occurrences=ts.num_samples * ts.num_trees,
        observed_vocabulary_node_count=len(observed_nodes), unique_query_paths=len(query_ids))
    return arrays, statistics


def identity_sha_v1(arrays):
    """Execute the exact original identity fields/helper, after source binding."""
    import numpy as np
    source = bound(ORIGINAL_IDENTITY).read_text()
    nodes = ast.parse(source).body
    keep = [n for n in nodes if
        (isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id == "IDENTITY" for v in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name == "identity_sha")]
    require(len(keep) == 2, "Exact original IDENTITY tuple and identity_sha helper")
    namespace = dict(np=np, hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=keep, type_ignores=[]),
        ORIGINAL_IDENTITY["path"] + "#unchanged-focal-identity", "exec"), namespace)
    return namespace["identity_sha"](arrays)


def execute(cfg, writer):
    """CPU science entry only; the unchanged original guard owns every write."""
    import tskit
    require(cfg == make_config_v1(cfg["parent_id"], cfg["observation_ref"], execution_enabled=True),
        "Verbatim original-seed compact TRAIN config; explicit admitted execution")
    m = read(cfg["observation_ref"])
    roster = read(m["roster_ref"])
    rows = roster["samples"]
    require(m["schema"] == "american_c3_observed_v1"
        and m["parent_id"] == roster["parent_id"] == cfg["parent_id"]
        and m["split"] == "train" and m["sequence_length_bp"] == 30000000
        and m["class_order"] == roster["class_order"] == list(CLASSES)
        and m["genealogy_kind"] == GENEALOGY
        and m["query_ancestry_truth_in_inputs"] is False
        and len(rows) == 1600, "Exact original complete sanitized TRAIN observation")
    require([r["role"] for r in rows] == ["pure_source_reference"] * 600 + ["query"] * 1000
        and [r["observed_reference_class"] for r in rows] == [2] * 200 + [0] * 200 + [1] * 200 + [None] * 1000,
        "Original known references and query role order; no query ancestry")
    ts = tskit.load(str(bound(m["tree_ref"])))
    require(ts.sequence_length == 30000000 and ts.num_samples == 1600
        and ts.num_migrations == 0 and ts.num_populations == 0
        and (ts.tables.nodes.population == -1).all(),
        "Same complete inferred tree sequence; no migration/internal source labels")
    arrays, statistics = focal_arrays_v1(ts, rows, cfg)
    focal_ref = writer.npz("focals.npz", **arrays)
    identity = identity_sha_v1(arrays)
    seal_ref = writer.json("focal_seal.json", dict(
        schema="TreeLoom_original_native_focal_identity_seal_v1",
        parent_id=m["parent_id"], observation_ref=cfg["observation_ref"],
        focal_registry_ref=focal_ref, focal_identity_sha256=identity, focal_records=FOCALS,
        query_truth_opened=False, identity_source_ref=ORIGINAL_IDENTITY,
        preparation_kind="compact_original_identity_focals_without_native_corpus"))
    return writer.json("completed.json", dict(schema=COMPLETION_SCHEMA, status=STATUS,
        observation_ref=cfg["observation_ref"], parent_id=m["parent_id"],
        genealogy_kind=GENEALOGY, class_order=list(CLASSES), focals_ref=focal_ref,
        focal_seal_ref=seal_ref, focal_identity_sha256=identity,
        query_focal_count=FOCALS, path_sampling_seed=cfg["path_sampling_seed"],
        focal_coordinate_rule="label-blind first-occurrence midpoint: reconstruction assumption, not recovered Table1 rule",
        source_original_native_preparation_ref=ORIGINAL_NATIVE,
        source_author_function_ref=ORIGINAL_AUTHOR, identity_source_ref=ORIGINAL_IDENTITY,
        original_registry_manifest_ref=ORIGINAL_PIPELINE_MANIFEST,
        **statistics, full_original_path_traversal=True,
        reference_paths_counted_in_global_record_ids=True,
        native_corpus_materialized=False, native_dataset_files_written=0,
        native_vocabulary_materialized=False, native_train_val_split_executed=False,
        sequence_to_tree_occurrence_mapping_materialized=False,
        query_ancestry_truth_opened=False, genotypes_read_for_focal_selection=False,
        author_Table1_exact_reproduction=False, model_executed=False,
        ancestry_accuracy=None, automatic_close_or_credit=False))

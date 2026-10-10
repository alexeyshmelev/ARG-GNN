#!/usr/bin/env python
"""Frozen CPU cosine corpus retrieval on completed ARGformer native path embeddings."""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import struct
import sys
import time
import traceback
import zipfile

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SCRIPT = ROOT / "scripts/score_official_argformer_retrieval_biological_v1.py"
CONFIG = ROOT / "configs/official_argformer_retrieval_biological_v1.json"
LEGACY = ROOT / "scripts/score_official_argformer_biological_v1.py"
LEGACY_CONFIG = ROOT / "configs/official_argformer_biological_readout_v1.json"
AUTHOR = ROOT / ".runtime/author_sources/argformer_f7545be7bca0"
INPUTS = {"path": "experiments/results/official_argformer_biological_readout_v1_sequence/root_inputs.json",
          "sha256": "28abd1777ca982024cb3f527cfca64c591b5f54853a18064c65264a275561921"}
SCHEMA = {"path": "experiments/results/official_argformer_retrieval_biological_root_schema_v1/schema.json",
          "sha256": "bf3a49e9738eb212e6032d8d518f6e1a82b73a4eeb4cd408dc5d464b7cb695b7"}
PARENTS = tuple(f"bio_paths_pilot_{split}_{i:03d}" for split in ("train", "val") for i in range(2))
VAL = PARENTS[2:]
CLASSES = ("ANA", "CHG", "WHG", "EHG")
RULE = {
    "primary_k": 1, "sensitivity_k": 5, "neighbor_count_saved": 6,
    "metric": "dot_of_source_F_normalize_float32_vectors_eps1e-12",
    "score_dtype": "float32", "votes": "unweighted_raw_neighbor_class_counts_divided_by_k",
    "corpus": "every_unique_full_path_of_pure96_in_same_ARG_across_native_path_train_and_val",
    "query_paths": "unique_full_path_ids_needed_by_all_common_retained_query120_centers",
    "path_multiplicity": "one_per_full_path_ID_no_genomic_span_weighting",
    "identical_embedding_rows": "retained_if_full_path_ID_differs",
    "neighbor_ties": "exact_computed_score_ties_then_ascending_original_full_path_ID",
    "class_vote_ties": "numpy_argmax_first_in_fixed_C4_order",
    "query_batch": 128, "corpus_chunk": 8192, "BLAS_threads": 4,
    "torch_threads": 4, "torch_interop_threads": 1, "embedding_dim": 768,
    "normalization_epsilon": 1e-12, "probability_floor": False,
    "calibration": False, "class_balance": False, "span_weighting": False,
    "target_based_eligibility": False, "training": False, "GPU": False,
    "max_exhaustive_pairs_per_parent": 40_000_000_000,
    "storage_headroom_bytes": 2_000_000_000,
    "primary_endpoints": ["accuracy", "macro_f1"],
    "NLL": "raw_hard_vote_zero_retained_and_positive_truth_mass_at_zero_is_infinite",
    "published_simulated_LAI_k": "unresolved_main_text_unspecified_source_default1",
    "paper_top5_scope": "explicit_archaic_and_Oceania_real_population_retrieval",
    "published_corpus_curation_and_tie_order": "not_fully_resolved",
    "paper_task_reproduction": False, "adapted_biological_task": True,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local(value, observed=False):
    p = Path(value)
    if ".." in p.parts or (observed and any("truth" in x.lower() or "oracle" in x.lower() for x in p.parts)):
        raise ValueError("Traversal or truth/oracle observed path forbidden")
    p = p if p.is_absolute() else ROOT / p
    if p.resolve() != p or not p.is_relative_to(ROOT):
        raise ValueError("Canonical project-local path required")
    return p


def bound(record, observed=False):
    if not isinstance(record, dict) or not re.fullmatch("[0-9a-f]{64}", record.get("sha256", "")):
        raise ValueError("Full factual path/hash binding required")
    p = local(record["path"], observed)
    if sha(p) != record["sha256"]:
        raise ValueError("Bound file changed: " + str(p))
    return p


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON requires explicit representation")
    return value


def new_json(path, value):
    with Path(path).open("x") as f:
        json.dump(jsonable(value), f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def policy(expected=None):
    if Path.cwd().resolve() != ROOT or Path(__file__).resolve() != SCRIPT:
        raise RuntimeError("Fixed remote repository and source path required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Explicitly hidden CUDA is required for every mode")
    if expected and sha(CONFIG) != expected:
        raise ValueError("Reviewed policy changed")
    cfg = json.loads(CONFIG.read_text())
    if (cfg["kind"] != "official_argformer_retrieval_biological_v1" or cfg["schema_version"] != 1
            or cfg["implementation_sha256"] != sha(SCRIPT) or cfg["retrieval_rule"] != RULE
            or cfg["root_inputs"] != INPUTS or cfg["root_schema"] != SCHEMA
            or cfg["parent_ids"] != list(PARENTS) or cfg["VAL_parent_ids"] != list(VAL)
            or cfg["class_order"] != list(CLASSES) or cfg["all_VAL_all_k_sealed_before_truth"] is not True):
        raise ValueError("Frozen retrieval method/data/implementation differs")
    for name, digest in cfg["frozen_files_sha256"].items():
        bound({"path": name, "sha256": digest})
    inventory = {str(p.relative_to(AUTHOR)): sha(p) for p in sorted(AUTHOR.rglob("*")) if p.is_file()}
    tree = hashlib.sha256((json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    if tree != cfg["author_tree_sha256"]:
        raise RuntimeError("Immutable author source tree changed")
    legacy = json.loads(LEGACY_CONFIG.read_text())
    if (cfg["common_metrics"] != legacy["metrics"] or cfg["external_truth_manifest"] != legacy["external_truth_manifest"]
            or cfg["legacy_scoring_config_sha256"] != sha(LEGACY_CONFIG)):
        raise ValueError("Unchanged common scoring/support binding differs")
    return cfg


def npz_headers(path):
    """Read only declared NPY header bytes, never any array payload."""
    result = {}
    with zipfile.ZipFile(path) as z:
        for entry in z.infolist():
            if not entry.filename.endswith(".npy") or "/" in entry.filename:
                raise ValueError("Simple native NPZ members required")
            with z.open(entry) as f:
                if f.read(6) != b"\x93NUMPY":
                    raise ValueError("NPY header magic differs")
                version = f.read(2)
                if version[0] not in (1, 2, 3):
                    raise ValueError("Unsupported NPY header")
                length = struct.unpack("<H" if version[0] == 1 else "<I", f.read(2 if version[0] == 1 else 4))[0]
                if length > 65536:
                    raise ValueError("Unexpected unbounded NPY header")
                header = ast.literal_eval(f.read(length).decode("utf8" if version[0] == 3 else "latin1").strip())
            if set(header) != {"descr", "fortran_order", "shape"} or header["fortran_order"] is not False:
                raise ValueError("Declared C-order native NPZ schema differs")
            result[entry.filename[:-4]] = header
    return result


def metadata_preflight(cfg):
    """Completed JSON and NPZ-header provenance only. No tensor or join payload load."""
    bindings = json.loads(bound(INPUTS, True).read_text())
    evidence = json.loads(bound(SCHEMA, True).read_text())
    if (bindings["kind"] != "official_argformer_biological_readout_inputs_v1"
            or set(bindings["parents"]) != set(PARENTS)
            or evidence["kind"] != "actual_completed_native_embedding_metadata_v1"
            or evidence["CPU_only"] is not True or evidence["query_ancestry_truth_read"] is not False
            or evidence["no_distances_or_scores_computed"] is not True):
        raise ValueError("Actual completed native input/schema evidence required")
    observed_path = bound(bindings["observed_manifest"], True)
    observed = json.loads(observed_path.read_text())
    entries = {r["parent_id"]: r for r in observed["replicates"]}
    if observed["status"] != "four_development_parents_inferred_pre_and_post_five_iterations" or set(entries) != set(PARENTS):
        raise ValueError("Exact completed observed four-parent inputs required")
    schema_records = {r["parent_id"]: r for r in evidence["records"]}
    if set(schema_records) != set(VAL):
        raise ValueError("Both factual VAL embedding schemas required")
    native = json.loads(local("configs/official_argformer_biological_pilot_v1.json").read_text())
    records = {}
    for name in PARENTS:
        binding = bindings["parents"][name]
        if set(binding) != {"prepared_manifest", "fit_receipt"}:
            raise ValueError("Exact native preparation/fit bindings required")
        prepared_path = bound(binding["prepared_manifest"], True)
        prepared = json.loads(prepared_path.read_text())
        receipt_path = bound(binding["fit_receipt"], True)
        receipt = json.loads(receipt_path.read_text())
        if (prepared["parent_id"] != name or prepared["schema"] != "official_argformer_prepared_observed_arg_v1"
                or prepared["status"] != "prepared_observed_paths_only" or prepared["true_query_ancestry_read"] is not False
                or prepared["namespace"] != name + ":" + entries[name]["inferred_tree"]["sha256"]
                or prepared["policy_sha256"] != sha(local("configs/official_argformer_biological_pilot_v1.json"))
                or prepared["input_manifest_sha256"] != bindings["observed_manifest"]["sha256"]
                or receipt["status"] != "passed" or receipt["exit_code"] != 0 or receipt["action"] != "fit"
                or receipt["root_actual_execution"] is not True or receipt["true_query_ancestry_read"] is not False
                or receipt["namespace"] != prepared["namespace"] or receipt["parent_id"] != name
                or receipt["prepared_manifest_sha256"] != binding["prepared_manifest"]["sha256"]
                or receipt["config_sha256"] != prepared["policy_sha256"]
                or receipt["implementation_sha256"] != cfg["native_fit_implementation_sha256"]
                or receipt["classifier_unchanged_unused"] is not True):
            raise ValueError("Passed native mean-head fit/preparation identity differs")
        for phase, epochs in (("pretraining", 5), ("contrastive", 1)):
            record = receipt[phase]
            if (record["completed_epochs"] != epochs or record["execution"] != "full_batch"
                    or record["actual_optimizer_updates"] <= 0
                    or record["initial_model_state_sha256"] == record["final_model_state_sha256"]
                    or record["true_query_ancestry_read"] is not False):
                raise ValueError("Completed unchanged native MLM5/contrastive1 required")
            directory = "mlm" if phase == "pretraining" else "contrastive"
            phase_path = bound({"path": str(receipt_path.parent / directory / "phase_receipt.json"),
                                "sha256": receipt["artifacts_sha256"][directory + "/phase_receipt.json"]}, True)
            if json.loads(phase_path.read_text()) != record:
                raise ValueError("Actual native phase receipt differs")
        for relative in ("corpus.pkl", "embedding/all/raw_all_sequences_and_vocab.pkl", "coordinate_path_joins.npz",
                         "observed_roster.json", "observed_parent_entry.json"):
            bound({"path": str(prepared_path.parent / relative), "sha256": prepared["files_sha256"][relative]}, True)
        embedding = {"path": str(receipt_path.parent / "native_path_embeddings.pt"),
                     "sha256": receipt["artifacts_sha256"]["native_path_embeddings.pt"]}
        original_mean = {"path": str(receipt_path.parent / "downstream_C4_local_readout.npz"),
                         "sha256": receipt["artifacts_sha256"]["downstream_C4_local_readout.npz"]}
        bound(original_mean, True)
        n = len(prepared["coordinate_contract"]["retained_fixed100bp_indexes"])
        headers = npz_headers(prepared_path.parent / "coordinate_path_joins.npz")
        shapes = {"grid_indexes": (n,), "original_positions_bp": (n,), "local_positions_bp": (n,),
                  "query_nodes": (120,), "reference_nodes": (96,), "reference_classes": (96,),
                  "query_path_ids": (n, 120), "reference_path_ids": (n, 96)}
        if set(headers) != set(shapes) or any(headers[k]["shape"] != v for k, v in shapes.items()):
            raise ValueError("Exact native per-center join header shapes differ")
        for key in ("grid_indexes", "query_nodes", "reference_nodes", "reference_classes", "query_path_ids", "reference_path_ids"):
            if re.fullmatch("[<>=|]?[iu][1248]", headers[key]["descr"]) is None:
                raise ValueError("Integer native path/subject/grid header required")
        if name in VAL:
            record = schema_records[name]
            detail = record["schema"]
            if (record["fit_receipt"] != binding["fit_receipt"]
                    or local(record["file"]["path"], True) != local(embedding["path"], True)
                    or record["file"]["sha256"] != embedding["sha256"]
                    or detail["embeddings"] != {"type": "torch.Tensor", "shape": [prepared["unique_full_paths"], RULE["embedding_dim"]],
                        "dtype": "torch.float32", "device": "cpu", "finite": True, "contiguous": True}
                    or detail["native_pooling"] != {"type": "str", "value": "trained_author_mean_head"}
                    or record["true_query_ancestry_read"] is not False or record["tensor_payload_values_printed"] is not False):
                raise ValueError("Actual completed native embedding tensor schema differs")
            bound(embedding, True)
        refs = sum(prepared["path_counts"]["contrastive"][split] for split in ("train", "val"))
        upper_query = prepared["unique_full_paths"] - refs
        records[name] = {"namespace": prepared["namespace"], "unique_full_paths": prepared["unique_full_paths"],
            "reference_full_path_count": refs, "query_unique_upper_bound_including_mixed": upper_query,
            "exhaustive_dot_pairs_upper_bound": refs * upper_query,
            "dot_FLOPs_upper_bound": 2 * refs * upper_query * RULE["embedding_dim"],
            "embeddings": embedding, "original_mean_readout": original_mean,
            "NPZ_headers_only": headers, "binding": binding}
    if native["query_ancestry_targets_read"] is not False:
        raise ValueError("Native encoder policy target exclusion differs")
    return records


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Only primitive native path/vocab metadata is admitted")


def primitive_load(path):
    with Path(path).open("rb") as f:
        return PrimitiveUnpickler(f).load()


def corpus_roles(paths, query_nodes, reference_nodes, reference_classes):
    """Pure donor full-path rows only. Embedding duplicates are not path duplicates."""
    import numpy as np
    if not isinstance(paths, tuple) or not paths or any(not isinstance(p, tuple) for p in paths) or len(set(paths)) != len(paths):
        raise ValueError("Nonempty unique complete path IDs required")
    if (np.asarray(query_nodes).dtype.kind not in 'iu' or np.asarray(reference_nodes).dtype.kind not in 'iu'
            or np.asarray(reference_classes).dtype.kind not in 'iu' or len(reference_nodes) != len(reference_classes)):
        raise ValueError('Typed observed donor/query IDs and C4 reference labels required')
    query = set(map(int, query_nodes))
    ref = dict(zip(map(int, reference_nodes), map(int, reference_classes)))
    if len(query) != len(query_nodes) or len(ref) != len(reference_nodes) or query & set(ref):
        raise ValueError("Disjoint unique actual query and pure reference leaf IDs required")
    if set(ref.values()) != {0, 1, 2, 3}:
        raise ValueError("Every biological source requires pure observed references")
    ids, labels = [], []
    for index, path in enumerate(paths):
        if not path or any(type(x) is not int or x < 0 for x in path) or len(set(path)) != len(path):
            raise ValueError("Unique acyclic integer observed full paths required")
        if path[0] in ref:
            ids.append(index)
            labels.append(ref[path[0]])
    if not ids:
        raise ValueError("No pure-reference path corpus")
    return np.asarray(ids, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def validate_support(paths, occurrences, namespace, expected_namespace, joins, local_length):
    import numpy as np
    if namespace != expected_namespace:
        raise ValueError("Retrieval corpus and queries require the same ARG namespace")
    positions = joins["local_positions_bp"]
    if (positions.ndim != 1 or not len(positions) or not np.isfinite(positions).all()
            or np.any(np.diff(positions) <= 0) or np.any(positions < 0) or np.any(positions >= local_length)):
        raise ValueError("Retained centers must lie in the declared half-open observed domain")
    leaves = {p[0] for p in paths}
    grouped = {leaf: [] for leaf in leaves}
    for pid, leaf, left, right in occurrences:
        if (type(pid) is not int or not 0 <= pid < len(paths) or paths[pid][0] != leaf
                or not math.isfinite(left) or not math.isfinite(right) or not 0 <= left < right <= local_length):
            raise ValueError("Observed occurrence/path/leaf coverage differs")
        grouped[leaf].append((left, right, pid))
    for records in grouped.values():
        records.sort()
        if (not records or records[0][0] != 0 or records[-1][1] != local_length
                or any(a[1] != b[0] for a, b in zip(records, records[1:]))):
            raise ValueError("Full observed per-leaf physical partition required")
    for nodes_key, ids_key in (("query_nodes", "query_path_ids"), ("reference_nodes", "reference_path_ids")):
        nodes, ids = joins[nodes_key], joins[ids_key]
        if (ids.dtype.kind not in "iu" or ids.shape != (len(joins["local_positions_bp"]), len(nodes))
                or np.any(ids < 0) or np.any(ids >= len(paths))):
            raise ValueError("Every retained query/reference center must have an actual path")
        for column, leaf in enumerate(nodes):
            if int(leaf) not in grouped:
                raise ValueError("Actual queried/reference leaf absent from full corpus")
            records = grouped[int(leaf)]
            ends = np.asarray([r[1] for r in records])
            selected = np.searchsorted(ends, joins["local_positions_bp"], side="right")
            if np.any(selected >= len(records)):
                raise ValueError("Retained center outside half-open observed domain")
            expected = np.asarray([r[2] for r in records], dtype=np.int64)[selected]
            if not np.array_equal(ids[:, column], expected):
                raise ValueError("Per-center actual full-path/leaf occurrence join differs")
    return True


def lexicographic_topk(scores, ids, k):
    """Exhaustive score descending, exact ties by original path ID ascending."""
    import numpy as np
    if scores.ndim != 2 or not np.isfinite(scores).all() or ids.shape != (scores.shape[1],) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Finite candidate scores and unique original IDs required")
    take = min(k, scores.shape[1])
    threshold = np.partition(scores, scores.shape[1] - take, axis=1)[:, scores.shape[1] - take]
    values = np.empty((len(scores), take), dtype=scores.dtype)
    selected = np.empty((len(scores), take), dtype=np.int64)
    for row in range(len(scores)):
        high = np.flatnonzero(scores[row] > threshold[row])
        equal = np.flatnonzero(scores[row] == threshold[row])
        equal = equal[np.argsort(ids[equal], kind="stable")]
        choice = np.r_[high, equal[:take - len(high)]]
        order = np.lexsort((ids[choice], -scores[row, choice]))
        choice = choice[order]
        values[row], selected[row] = scores[row, choice], ids[choice]
    return values, selected


def exhaustive_neighbors(query, corpus, corpus_ids, corpus_labels, q_batch, e_chunk, save_k=6, progress=None):
    import numpy as np
    if (query.ndim != 2 or corpus.ndim != 2 or query.shape[1] != corpus.shape[1]
            or query.dtype != np.float32 or corpus.dtype != np.float32
            or not np.isfinite(query).all() or not np.isfinite(corpus).all()
            or len(corpus) < save_k or corpus_ids.shape != (len(corpus),)
            or corpus_labels.shape != (len(corpus),) or np.any((corpus_labels < 0) | (corpus_labels > 3))
            or len(set(corpus_ids.tolist())) != len(corpus_ids) or q_batch < 1 or e_chunk < 1):
        raise ValueError("Declared finite float32 exhaustive retrieval inputs required")
    scores_all = np.empty((len(query), save_k), dtype=np.float32)
    ids_all = np.empty((len(query), save_k), dtype=np.int64)
    nearest_ties = np.zeros((len(query), 4), dtype=np.int64)
    started = time.monotonic()
    for qs in range(0, len(query), q_batch):
        qe = min(qs + q_batch, len(query))
        best_values = best_ids = None
        peak = np.full(qe - qs, -np.inf, dtype=np.float32)
        peak_counts = np.zeros((qe - qs, 4), dtype=np.int64)
        for es in range(0, len(corpus), e_chunk):
            ee = min(es + e_chunk, len(corpus))
            scores = query[qs:qe] @ corpus[es:ee].T
            local_peak = scores.max(axis=1)
            for row in np.flatnonzero(local_peak >= peak):
                tied = np.flatnonzero(scores[row] == local_peak[row])
                counts = np.bincount(corpus_labels[es:ee][tied], minlength=4)
                if local_peak[row] > peak[row]:
                    peak_counts[row] = counts
                else:
                    peak_counts[row] += counts
            peak = np.maximum(peak, local_peak)
            values, ids = lexicographic_topk(scores, corpus_ids[es:ee], save_k)
            if best_values is None:
                best_values, best_ids = values, ids
            else:
                merged_values = np.concatenate((best_values, values), axis=1)
                merged_ids = np.concatenate((best_ids, ids), axis=1)
                best_values = np.empty((qe - qs, min(save_k, merged_values.shape[1])), dtype=np.float32)
                best_ids = np.empty(best_values.shape, dtype=np.int64)
                for row in range(qe - qs):
                    order = np.lexsort((merged_ids[row], -merged_values[row]))[:save_k]
                    best_values[row], best_ids[row] = merged_values[row, order], merged_ids[row, order]
            del scores
        scores_all[qs:qe], ids_all[qs:qe] = best_values, best_ids
        nearest_ties[qs:qe] = peak_counts
        if progress:
            progress(qe, len(query), time.monotonic() - started)
    return scores_all, ids_all, nearest_ties


def hard_votes(neighbor_ids, corpus_ids, corpus_labels, k):
    import numpy as np
    if k not in (1, 5) or neighbor_ids.ndim != 2 or neighbor_ids.shape[1] < k:
        raise ValueError("Only prespecified k1/k5 hard votes")
    lookup = {int(pid): int(label) for pid, label in zip(corpus_ids, corpus_labels)}
    if len(lookup) != len(corpus_ids) or any(label not in range(4) for label in lookup.values()):
        raise ValueError("Distinct pure-reference IDs and C4 labels required")
    try:
        labels = np.asarray([[lookup[int(pid)] for pid in row[:k]] for row in neighbor_ids], dtype=np.int64)
    except KeyError as e:
        raise ValueError("Neighbor is not an eligible pure full path") from e
    counts = np.stack([(labels == c).sum(axis=1) for c in range(4)], axis=1)
    return counts.astype(np.float32) / np.float32(k)


def project_grid(unique_ids, probabilities, query_path_ids):
    import numpy as np
    if (unique_ids.ndim != 1 or np.any(np.diff(unique_ids) <= 0)
            or probabilities.shape != (len(unique_ids), 4) or query_path_ids.dtype.kind not in "iu"):
        raise ValueError("Complete unique-query probability support required")
    indices = np.searchsorted(unique_ids, query_path_ids)
    if np.any(indices >= len(unique_ids)) or not np.array_equal(unique_ids[indices], query_path_ids):
        raise ValueError("Missing query path cannot shrink common support")
    return probabilities[indices]


@contextmanager
def cpu_libraries(cfg):
    threads = cfg["retrieval_rule"]["BLAS_threads"]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    torch.set_num_threads(cfg["retrieval_rule"]["torch_threads"])
    torch.set_num_interop_threads(cfg["retrieval_rule"]["torch_interop_threads"])
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU retrieval must not initialize CUDA")
    with threadpool_limits(limits=threads, user_api="blas"):
        info = threadpool_info()
        if not info or any(r["num_threads"] > threads for r in info if r["user_api"] == "blas"):
            raise RuntimeError("Actual bounded BLAS thread policy differs")
        yield np, torch, info


def load_legacy(cfg):
    spec = importlib.util.spec_from_file_location("_frozen_ARGformer_common_scoring", LEGACY)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    common, helper = legacy.source_policy(cfg["legacy_scoring_config_sha256"])
    common, parents, states = legacy.validate_inputs(common, helper, INPUTS)
    return legacy, common, helper, parents, states


def tensor_sha(tensor):
    data = memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B")
    h = hashlib.sha256()
    for start in range(0, len(data), 16 * 1024 * 1024):
        h.update(data[start:start + 16 * 1024 * 1024])
    return h.hexdigest()


def seal_parent(parent, state, cfg, metadata, output, torch):
    import numpy as np
    prepared = state["prepared"]
    prepared_dir = local(state["records"]["prepared_manifest"]["path"], True).parent
    p = bound({"path": str(prepared_dir / "corpus.pkl"), "sha256": prepared["files_sha256"]["corpus.pkl"]}, True)
    corpus = primitive_load(p)
    if set(corpus) != {"paths", "occurrences", "namespace"} or len(corpus["paths"]) != prepared["unique_full_paths"]:
        raise ValueError("Native full observed corpus schema differs")
    paths = corpus["paths"]
    if {path[0] for path in paths} != set(parent["observation"]["nodes"].tolist()):
        raise ValueError("Full native path corpus must retain all280 actual observed leaves")
    source_sequences = primitive_load(bound({"path": str(prepared_dir / "embedding/all/raw_all_sequences_and_vocab.pkl"),
            "sha256": prepared["files_sha256"]["embedding/all/raw_all_sequences_and_vocab.pkl"]}, True))
    if set(source_sequences) != {"raw_sequences", "vocab"} or len(source_sequences["raw_sequences"]) != len(paths) or any(
            tuple(row) != path for row, path in zip(source_sequences["raw_sequences"], paths)):
        raise ValueError("Source native embedding row/full-path order differs")
    del source_sequences
    joins = state["joins"]
    corpus_ids, corpus_labels = corpus_roles(paths, joins["query_nodes"], joins["reference_nodes"], joins["reference_classes"])
    validate_support(paths, corpus["occurrences"], corpus["namespace"], prepared["namespace"], joins,
                     parent["coverage_original"][1] - parent["coverage_original"][0])
    unique_queries = np.unique(joins["query_path_ids"])
    if any(paths[int(pid)][0] not in set(joins["query_nodes"].tolist()) for pid in unique_queries):
        raise ValueError("Unique retrieval query path has wrong actual query leaf")
    if set(corpus_ids.tolist()) & set(unique_queries.tolist()):
        raise ValueError("Query self-match must not enter reference corpus")
    refs_expected = metadata["reference_full_path_count"]
    if len(corpus_ids) != refs_expected or len(unique_queries) * len(corpus_ids) > RULE["max_exhaustive_pairs_per_parent"]:
        raise RuntimeError("Declared exact CPU corpus/count/cost admission differs")
    payload = torch.load(bound(metadata["embeddings"], True), map_location="cpu", weights_only=True)
    vectors = payload["embeddings"]
    if (set(payload) != {"embeddings", "native_pooling"} or payload["native_pooling"] != "trained_author_mean_head"
            or vectors.shape != (prepared["unique_full_paths"], RULE["embedding_dim"])
            or vectors.dtype != torch.float32 or vectors.device.type != "cpu" or not vectors.is_contiguous()
            or not torch.isfinite(vectors).all() or tensor_sha(vectors) != state["receipt"]["readout"]["native_embedding_sha256"]):
        raise ValueError("Actual immutable native embedding order/dtype/tensor hash differs")
    import torch.nn.functional as F
    zero_rows = int((vectors.norm(p=2, dim=1) <= RULE["normalization_epsilon"]).sum())
    q = F.normalize(vectors[torch.from_numpy(unique_queries)], p=2, dim=1, eps=RULE["normalization_epsilon"]).numpy()
    e = F.normalize(vectors[torch.from_numpy(corpus_ids)], p=2, dim=1, eps=RULE["normalization_epsilon"]).numpy()
    del payload, vectors, corpus
    def progress(done, total, elapsed):
        print(json.dumps({"stage": "exact_CPU_retrieval", "parent_id": parent["parent_id"],
                          "unique_query_paths_completed": done, "unique_query_paths": total,
                          "elapsed_seconds": elapsed}), flush=True)
    scores, neighbors, ties = exhaustive_neighbors(q, e, corpus_ids, corpus_labels,
        RULE["query_batch"], RULE["corpus_chunk"], RULE["neighbor_count_saved"], progress)
    del q, e
    grids = {}
    diagnostics = {"unique_query_paths": int(len(unique_queries)), "pure_reference_full_paths": int(len(corpus_ids)),
        "exhaustive_pairs": int(len(unique_queries) * len(corpus_ids)), "native_embedding_zero_or_epsilon_rows": zero_rows,
        "top1_multiple_nearest_reference_rows": int(np.sum(ties.sum(axis=1) > 1)),
        "top1_cross_class_exact_max_ties": int(np.sum((ties > 0).sum(axis=1) > 1)),
        "top5_cutoff_exact_score_ties": int(np.sum(scores[:, 4] == scores[:, 5])), "k": {}}
    for k in (1, 5):
        p = hard_votes(neighbors, corpus_ids, corpus_labels, k)
        grid = project_grid(unique_queries, p, joins["query_path_ids"])
        if grid.shape != (len(parent["positions"]), 120, 4) or not np.isfinite(grid).all() or not np.allclose(grid.sum(-1), 1, atol=1e-6, rtol=0):
            raise ValueError("Full common C4 retained-query grid unsupported")
        grids["C4_probabilities_raw_k" + str(k)] = grid
        maxima = grid.max(-1, keepdims=True)
        diagnostics["k"][str(k)] = {"grid_cells": int(grid.shape[0] * 120),
            "class_vote_argmax_tie_grid_cells": int(np.sum((grid == maxima).sum(-1) > 1)),
            "raw_zero_class_probability_cells": int(np.sum(grid == 0))}
    neighbor_path = output / (parent["parent_id"] + "_original_retrieval_neighbors.npz")
    with neighbor_path.open("xb") as f:
        np.savez_compressed(f, query_full_path_ids=unique_queries, neighbor_full_path_ids=neighbors,
            cosine_scores_float32=scores, nearest_tie_reference_class_counts=ties,
            corpus_full_path_ids=corpus_ids, corpus_observed_reference_classes=corpus_labels)
    path = output / (parent["parent_id"] + "_sealed_retrieval_C4.npz")
    with path.open("xb") as f:
        np.savez_compressed(f, **grids, positions_bp=parent["positions"], bin_edges_bp=parent["edges"],
            retained_fixed100bp_indexes=parent["retained_indexes"], query_sample_indexes=parent["query_slots"],
            query_nodes=joins["query_nodes"], query_raw_node_ids=parent["raw_query_nodes"],
            query_path_ids=joins["query_path_ids"], coverage_original_bp0=parent["coverage_original"],
            normalization_length_bp=np.asarray(2000000))
    return {"parent_id": parent["parent_id"], "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "namespace": prepared["namespace"], "neighbors": {"path": str(neighbor_path.relative_to(ROOT)), "sha256": sha(neighbor_path)},
        "root_native_fit": state["records"]["fit_receipt"], "native_embeddings": metadata["embeddings"],
        "original_mean_readout": metadata["original_mean_readout"], "diagnostics": diagnostics,
        "all_common_retained_support": True, "truth_opened": False, "both_prespecified_k": [1, 5]}


def score_parent(parent, seal, common, helper, legacy, aggregate):
    import numpy as np
    with np.load(bound(seal, True), allow_pickle=False) as z:
        grids = {k: z["C4_probabilities_raw_k" + str(k)].copy() for k in (1, 5)}
        for key, expected in (("positions_bp", parent["positions"]), ("bin_edges_bp", parent["edges"]),
                             ("query_sample_indexes", parent["query_slots"]), ("query_raw_node_ids", parent["raw_query_nodes"])):
            if not np.array_equal(z[key], expected):
                raise ValueError("Sealed exact support/query identity differs")
    truth = legacy.ExternalVALC4Truth(parent, common, aggregate)
    groups = {"primary_present_day_64": np.flatnonzero(parent["cohorts"] == "present_day"),
              "secondary_ancient_56": np.flatnonzero(parent["cohorts"] != "present_day"), "all_queries_120": np.arange(120)}
    results = {}
    for k, p in grids.items():
        metrics = {}
        for name, columns in groups.items():
            item = helper.full_span_score(p, parent, truth, columns, common["metrics"])
            item["nll_is_infinite"] = bool(item["infinite_nll_query_bp"] > 0)
            item["nll_null_representation"] = "positive exact truth mass at raw hard-vote probability zero" if item["nll_is_infinite"] else None
            metrics[name] = item
        results[str(k)] = {"condition": "primary_source_default_top1" if k == 1 else "prespecified_paper_top5_sensitivity",
                          "neighbor_count": k, "primary_endpoints": ["accuracy", "macro_f1"], "exact_support_metrics": metrics}
    return {"parent_id": parent["parent_id"], "sealed_prediction": seal, "results_by_k": results,
        "external_truth_provenance": truth.provenance, "adapted_biological_task": True, "paper_task_reproduction": False,
        "training": False, "calibration_fit": False, "query_target_based_corpus_or_support": False,
        "old_same_coordinate_mean_exp_readout_preserved": True, "final_holdout_evaluation": False}


def run(cfg, metadata, output):
    output.mkdir(exist_ok=False)
    receipt = {"kind": "official_argformer_retrieval_biological_v1_execution", "status": "running",
        "implementation_sha256": sha(SCRIPT), "config_sha256": sha(CONFIG), "root_inputs": INPUTS,
        "root_schema": SCHEMA, "retrieval_rule": RULE, "predictions": [], "metrics": [], "exit_code": 1,
        "model_inference": False, "training": False, "GPU": False, "calibration": False,
        "original_fits_and_mean_readout_modified": False}
    started = time.monotonic()
    try:
        from arggnn.resources import require_storage
        receipt["storage_used_bytes"] = require_storage(RULE["storage_headroom_bytes"])
        with cpu_libraries(cfg) as (np, torch, libraries):
            receipt["CPU_runtime"] = {"torch": str(torch.__version__), "numpy": np.__version__, "BLAS": libraries,
                                     "torch_threads": torch.get_num_threads(), "CUDA_initialized": torch.cuda.is_initialized()}
            legacy, common, helper, parents, states = load_legacy(cfg)
            for name in VAL:
                receipt["predictions"].append(seal_parent(parents[name], states[name], cfg, metadata[name], output, torch))
            aggregate_path = output / "both_VAL_all_k_predictions_sealed_before_truth.json"
            new_json(aggregate_path, {"status": "both_VAL_predictions_sealed_before_truth", "predictions": receipt["predictions"],
                "root_inputs": INPUTS, "all_k": [1, 5], "truth_scoring_started": False})
            aggregate = {"path": str(aggregate_path.relative_to(ROOT)), "sha256": sha(aggregate_path)}
            receipt["aggregate_prediction_seal"] = aggregate
            for seal in receipt["predictions"]:
                result = score_parent(parents[seal["parent_id"]], seal, common, helper, legacy, aggregate)
                path = output / (seal["parent_id"] + "_external_retrieval_VAL_metrics.json")
                new_json(path, result)
                receipt["metrics"].append({"parent_id": seal["parent_id"], "path": str(path.relative_to(ROOT)), "sha256": sha(path)})
            policy(receipt["config_sha256"])
            metadata_preflight(cfg)
            legacy.validate_inputs(common, helper, INPUTS)
            bound(common["external_truth_manifest"])
            if torch.cuda.is_initialized():
                raise RuntimeError("CPU readout unexpectedly initialized CUDA")
            receipt.update(status="passed", exit_code=0)
    except BaseException as error:
        receipt.update(status="failed_partial_preserved", failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        new_json(output / "receipt.json", receipt)
        with (output / "exit_status").open("x") as f:
            f.write(str(receipt["exit_code"]) + "\n")
    return receipt["exit_code"]


def handmade_tests():
    """Handmade vectors/paths only. No source models, native payloads or truth."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    from threadpoolctl import threadpool_limits
    handmade_thread_limit = threadpool_limits(limits=1, user_api="blas")
    tests = []
    e = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [.5, .5], [.5, -.5], [-.5, .5]], dtype=np.float32)
    q = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    ids = np.array([21, 11, 41, 31, 9, 8, 7], dtype=np.int64)
    labels = np.array([0, 1, 2, 3, 1, 0, 2], dtype=np.int64)
    expected_scores, expected_ids = lexicographic_topk(q @ e.T, ids, 6)
    for qb, ec in ((1, 1), (2, 3), (3, 7)):
        scores, selected, ties = exhaustive_neighbors(q, e, ids, labels, qb, ec)
        assert np.array_equal(selected, expected_ids) and np.array_equal(scores, expected_scores)
    tests.append("exhaustive_cross_chunk_matches_full_metric_and_canonical_ties")

    import torch
    import torch.nn.functional as F
    raw = torch.tensor([[3., 4.], [0., 0.], [-4., 3.]], dtype=torch.float32)
    normalized = F.normalize(raw, p=2, dim=1, eps=1e-12)
    assert torch.equal(normalized[1], torch.zeros(2))
    assert torch.allclose(normalized[0], torch.tensor([.6, .8]))
    norm_corpus = torch.cat([normalized, normalized, normalized[:1]], dim=0)
    norm_labels = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)
    ns, ni, _ = exhaustive_neighbors(normalized.numpy(), norm_corpus.numpy(), np.arange(7), norm_labels, 2, 3)
    reference_score = (normalized @ norm_corpus.T).numpy()
    rs, ri = lexicographic_topk(reference_score, np.arange(7), 6)
    assert np.allclose(ns, rs, atol=1e-7, rtol=0)
    assert np.array_equal(ni[:, 0], ri[:, 0])
    provider_near_tie_order_differs = bool(not np.array_equal(ni, ri))
    assert not torch.cuda.is_initialized()
    tests.append("source_F_normalize_cosine_score_parity_zero_vector_and_disclosed_provider_near_ties")
    dup = np.tile(np.array([[1, 0]], dtype=np.float32), (7, 1))
    duplicate_labels = np.array([0, 1, 2, 3, 3, 3, 3])
    dids = np.array([9, 3, 8, 1, 4, 6, 2])
    _, neighbors, tie_counts = exhaustive_neighbors(q[:1], dup, dids, duplicate_labels, 1, 2)
    assert neighbors.tolist() == [[1, 2, 3, 4, 6, 8]]
    assert tie_counts.tolist() == [[1, 1, 1, 4]]
    vote1 = hard_votes(neighbors, dids, duplicate_labels, 1)
    vote5 = hard_votes(neighbors, dids, duplicate_labels, 5)
    assert vote1.tolist() == [[0., 0., 0., 1.]]
    assert np.array_equal(vote5, np.array([[0, .2, 0, .8]], dtype=np.float32))
    assert np.any(vote1 == 0) and vote1.sum() == 1
    tests.append("identical_embedding_rows_retained_ties_by_full_path_ID_raw_k_votes_and_zeros")
    paths = ((0, 10), (1, 10), (2, 10), (3, 11), (4, 11), (5, 11), (6, 11))
    ref_ids, ref_labels = corpus_roles(paths, [0], [1, 2, 3, 4], [0, 1, 2, 3])
    assert ref_ids.tolist() == [1, 2, 3, 4] and ref_labels.tolist() == [0, 1, 2, 3]
    tests.append("pure_reference_corpus_excludes_query_and_unlabeled_mixed")
    unique = np.array([2, 5], dtype=np.int64)
    probability = np.eye(4, dtype=np.float32)[[0, 3]]
    grid_ids = np.array([[5, 2], [2, 5]], dtype=np.int64)
    assert np.array_equal(project_grid(unique, probability, grid_ids), probability[[[1, 0], [0, 1]]])
    try:
        project_grid(unique, probability, np.array([[4]], dtype=np.int64))
        raise AssertionError("Missing support accepted")
    except ValueError:
        pass
    tests.append("repeated_center_path_projection_complete_support_and_missing_path_rejection")
    sample_paths = ((0, 10), (0, 11), (1, 10), (1, 11))
    occurrences = ((0, 0, 0., 5.), (1, 0, 5., 10.), (2, 1, 0., 5.), (3, 1, 5., 10.))
    joins = {"local_positions_bp": np.array([2., 7.]), "query_nodes": np.array([0]), "reference_nodes": np.array([1]),
             "query_path_ids": np.array([[0], [1]]), "reference_path_ids": np.array([[2], [3]])}
    assert validate_support(sample_paths, occurrences, "A", "A", joins, 10.)
    for namespace, corrupted in (("B", joins), ("A", dict(joins, query_path_ids=np.array([[0], [-1]])))):
        try:
            validate_support(sample_paths, occurrences, namespace, "A", corrupted, 10.)
            raise AssertionError("Invalid actual support/namespace accepted")
        except ValueError:
            pass
    tests.append("half_open_occurrence_path_joins_reject_namespace_mismatch_and_unsupported_cells")
    handmade_thread_limit.restore_original_limits()
    return {"status": "passed", "tests": tests, "FP32_provider_near_tie_neighbor_order_can_differ": provider_near_tie_order_differs,
            "actual_tensor_payload_loaded": False,
            "native_model_executed": False, "truth_or_scores_read": False, "GPU": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--test-handmade", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = policy(args.expected_config_sha256)
    if args.test_handmade:
        if args.execute:
            parser.error("Handmade source tests and actual execution are separate")
        result = handmade_tests()
        if args.output:
            new_json(local(args.output), result)
        print(json.dumps(result), flush=True)
        return 0
    metadata = metadata_preflight(cfg)
    if not args.execute:
        print(json.dumps({"status": "source_JSON_NPZ_header_preflight_only", "config_sha256": sha(CONFIG),
            "parents": metadata, "actual_tensor_payload_loaded": False, "embedding_distances_computed": False,
            "truth_or_scores_read": False, "model_executed": False, "GPU": False}), flush=True)
        return 0
    if not args.expected_config_sha256 or not args.output:
        parser.error("Actual CPU readout requires frozen config hash and fresh dedicated output")
    sys.path.insert(0, str(ROOT / "src"))
    output = local(args.output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh output with existing project parent required")
    return run(cfg, metadata, output)


if __name__ == "__main__":
    raise SystemExit(main())

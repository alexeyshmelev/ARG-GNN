"""Common ancestry metrics. This module does not load benchmark data."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contracts import ContractError, fingerprint, validate_protocol


@dataclass(frozen=True)
class Prediction:
    method_id: str
    protocol_id: str
    parent_id: str
    class_order: tuple[str, ...]
    keys: tuple[tuple[str, str, float], ...]
    hard_calls: np.ndarray
    probabilities: np.ndarray | None
    observation_fingerprint: str
    reference_query_fingerprint: str


def _validated(prediction: Prediction):
    n, c = len(prediction.keys), len(prediction.class_order)
    if not n or c < 2 or len(set(prediction.class_order)) != c:
        raise ContractError("Empty or ambiguous prediction task")
    if len(set(prediction.keys)) != n:
        raise ContractError("Duplicate query/position targets")
    for parent, sample, position in prediction.keys:
        if parent != prediction.parent_id or not sample or not math.isfinite(position):
            raise ContractError("Target identities must include this parent, sample and coordinate")
    hard = np.asarray(prediction.hard_calls)
    if hard.shape != (n,) or hard.dtype.kind not in "iu" or np.any((hard < 0) | (hard >= c)):
        raise ContractError("Native hard calls must be present for every target")
    probs = prediction.probabilities
    if probs is not None:
        probs = np.asarray(probs)
        if probs.shape != (n, c) or not np.all(np.isfinite(probs)):
            raise ContractError("Posterior shape, support or finiteness mismatch")
        if np.any((probs < 0) | (probs > 1)) or not np.allclose(probs.sum(1), 1, atol=1e-5, rtol=1e-5):
            raise ContractError("Posteriors must sum to one without hidden repair")
    return hard, probs


def score_parent(prediction: Prediction, truth_keys: Sequence,
                 truth: np.ndarray, *, expected_support_fingerprint: str,
                 weights=None, empirical=False) -> dict:
    if empirical:
        raise ContractError("Do not invent supervised accuracy for empirical samples")
    hard, probs = _validated(prediction)
    truth_keys = tuple(tuple(k) for k in truth_keys)
    if len(set(truth_keys)) != len(truth_keys):
        raise ContractError("Duplicate truth targets")
    if set(truth_keys) != set(prediction.keys):
        raise ContractError("Every method must cover the full frozen support")
    # This identity must be frozen from the protocol's focal registry before
    # predictions are produced, not calculated from whichever rows survived.
    support = fingerprint(sorted(truth_keys))
    if support != expected_support_fingerprint:
        raise ContractError("Targets differ from the pre-frozen evaluation support")
    labels = np.asarray(truth)
    n, c = len(truth_keys), len(prediction.class_order)
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= c)):
        raise ContractError("Truth and class order must match")
    index = {key: i for i, key in enumerate(prediction.keys)}
    order = np.asarray([index[key] for key in truth_keys])
    hard = hard[order]
    probs = probs[order] if probs is not None else None
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w <= 0) or w.sum() <= 0:
        raise ContractError("Declared target weights must have positive total support")
    # Canonical reporting identity is independent of file row order.
    canonical = np.asarray(sorted(range(n), key=lambda i: truth_keys[i]))
    confusion = np.zeros((c, c), dtype=float)
    np.add.at(confusion, (labels, hard), w)
    tp = np.diag(confusion)
    precision = np.divide(tp, confusion.sum(0), out=np.zeros(c), where=confusion.sum(0) != 0)
    recall = np.divide(tp, confusion.sum(1), out=np.zeros(c), where=confusion.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(c), where=precision + recall != 0)
    result = {
        "method_id": prediction.method_id, "protocol_id": prediction.protocol_id,
        "parent_id": prediction.parent_id, "class_order": list(prediction.class_order),
        "targets": n, "total_weight": float(w.sum()),
        "support_fingerprint": support,
        "weight_fingerprint": fingerprint(w[canonical].tolist()),
        "truth_fingerprint": fingerprint(labels[canonical].tolist()),
        "observation_fingerprint": prediction.observation_fingerprint,
        "reference_query_fingerprint": prediction.reference_query_fingerprint,
        "accuracy": float(tp.sum() / w.sum()), "macro_f1": float(f1.mean()),
        "precision": precision.tolist(), "recall": recall.tolist(),
        "f1": f1.tolist(), "confusion_truth_rows_prediction_columns": confusion.tolist(),
        "undefined_class_metric_policy": "zero_including_declared_macro_classes",
        "hard_calls_replaced_by_posterior_argmax": False,
        "probability_floor": False,
    }
    if probs is not None:
        selected = probs[np.arange(n), labels]
        positive = w > 0
        if np.any(selected[positive] == 0):
            result.update(nll=None, nll_status="infinite_zero_true_class_probability")
        else:
            result.update(nll=float(np.dot(-np.log(selected[positive]), w[positive]) / w.sum()),
                          nll_status="finite")
        target = np.eye(c)[labels]
        result["brier"] = float(np.dot(((probs - target) ** 2).sum(1), w) / w.sum())
    else:
        result.update(nll=None, nll_status="unavailable_native_probabilities_not_provided")
    return result


def collapse_probabilities(probabilities, groups: Sequence[Sequence[int]]):
    """Explicit derived endpoint, for example six native paths to four sources."""
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2 or not np.all(np.isfinite(p)):
        raise ContractError("Expected a finite nonempty native posterior matrix")
    if np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=1e-5):
        raise ContractError("Validate the native simplex before collapsing classes")
    indices = [i for group in groups for i in group]
    if (any(type(i) is not int for i in indices)
            or sorted(indices) != list(range(p.shape[1])) or any(not group for group in groups)):
        raise ContractError("The declared endpoint must partition all native classes exactly")
    return np.column_stack([p[:, list(group)].sum(1) for group in groups])


def summarize_parents(rows: Sequence[dict], metric: str = "accuracy") -> dict:
    if not rows or len({r["parent_id"] for r in rows}) != len(rows):
        raise ContractError("Each independent simulation parent contributes once")
    for name in ("method_id", "protocol_id", "class_order"):
        if any(r[name] != rows[0][name] for r in rows):
            raise ContractError("Do not pool different methods, protocols or class definitions")
    values = np.asarray([r[metric] for r in rows], dtype=float)
    classwise = metric in {"precision", "recall", "f1"}
    shape = (len(rows), len(rows[0]["class_order"])) if classwise else (len(rows),)
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ContractError("Expected one finite scalar or declared class vector per parent")
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else None
    return {"method_id": rows[0]["method_id"], "protocol_id": rows[0]["protocol_id"],
            "metric": metric, "mean": mean.tolist(),
            "sample_sd": None if sd is None else sd.tolist(),
            "class_order": list(rows[0]["class_order"]) if classwise else None,
            "independent_parents": len(values),
            "unit": ("nats" if metric == "nll" else "squared_probability_sum"
                     if metric == "brier" else "fraction"
                     if metric in {"accuracy", "macro_f1", "precision", "recall", "f1"}
                     else "dimensionless"),
            "aggregation": "equal_parent_mean_sample_sd"}


def assert_paired_support(results: dict[str, Sequence[dict]], *, protocol: dict,
                          split: str,
                          expected_support_fingerprints: Mapping[str, str]) -> None:
    """Validate against a pre-frozen protocol, not just agreement among outputs.

    Expected support fingerprints come from the frozen focal registries before
    prediction. Recomputing them from surviving prediction/truth rows defeats
    this boundary and is not an acceptable caller implementation.
    """
    validate_protocol(protocol)
    if split not in {"validation", "test"}:
        raise ContractError("Declare the evaluation split")
    parents = set(protocol["split_parents"][split])
    if not parents or set(expected_support_fingerprints) != parents:
        raise ContractError("Freeze support for every declared evaluation parent")
    if not results or set(results) != set(protocol["methods"]):
        raise ContractError("Every declared comparison method must be present")
    expected = None
    for method_id, rows in results.items():
        keys = {}
        for row in rows:
            parent = row["parent_id"]
            if row["method_id"] != method_id or parent in keys:
                raise ContractError("Method identity or independent-parent duplication")
            if (parent not in parents or row["protocol_id"] != protocol["id"]
                    or row["class_order"] != protocol["class_order"]
                    or row["support_fingerprint"] != expected_support_fingerprints[parent]):
                raise ContractError("Results do not match the frozen protocol and target support")
            keys[parent] = tuple(
                fingerprint(row[k]) for k in ("protocol_id", "class_order",
                "support_fingerprint", "weight_fingerprint", "truth_fingerprint",
                "observation_fingerprint", "reference_query_fingerprint"))
        if set(keys) != parents:
            raise ContractError("A method omitted a declared evaluation parent")
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ContractError("Pair the same parents, observations, queries, truth and support")

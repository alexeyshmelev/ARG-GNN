"""Inactive metadata CLI plus ROOT-callable Chickenpox completion functions.

No dataset, Torch, PyG, model, guard or resource module is imported by the
default CLI. Numerical callables require a separately admitted ROOT caller.
This file creates no processes and writes no files.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import operator
from pathlib import Path
import random
import re
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "dc306b1845c86616d2a72c202deeefac8fd1fcf0a34b3067cf6721e377d9da44"
COUNTIES = ("BACS", "BARANYA", "BEKES", "BORSOD", "BUDAPEST", "CSONGRAD",
            "FEJER", "GYOR", "HAJDU", "HEVES", "JASZ", "KOMAROM", "NOGRAD",
            "PEST", "SOMOGY", "SZABOLCS", "TOLNA", "VAS", "VESZPREM", "ZALA")
SEEDS = (101, 202, 303)
ARMS = ("full", "no_memory")
HISTORY, TRAIN_END, VAL_END, OBSERVATIONS = 52, 365, 443, 522
EPOCHS, BATCH_SIZE = 40, 16
RIDGE_CANDIDATES = (.01, .1, 1., 10., 100., 1000.)


@dataclass(frozen=True)
class PreparedChickenpoxV1:
    counts: Any                     # NumPy [522,20]; development tail is NaN/read-only
    mean: Any                       # NumPy [20], rows0:365 only
    scale: Any                      # NumPy [20], population SD, zero replaced by1
    edges: Any                      # NumPy [2,82], native non-self directed edges
    native_edges: Any               # Original102records, for exact legacy representation
    counties: tuple[str, ...]
    dates: tuple[str, ...]
    phase: str


def protocol_v1():
    return json.loads((HERE / "protocol.inactive.v1.json").read_text())


def _seal_set(records, expected_kind):
    required = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if not isinstance(records, list) or len(records) != len(required):
        raise ValueError("All six fixed selections/predictions must be sealed")
    seen = set()
    for row in records:
        if (not isinstance(row, dict) or row.get("kind") != expected_kind
                or not isinstance(row.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("Exact externally authenticated six-product seal metadata")
        if type(row.get("seed")) is not int or not isinstance(row.get("arm"), str):
            raise ValueError("Typed fixed seed/arm identities")
        key = (row.get("seed"), row.get("arm"))
        if key in seen:
            raise ValueError("Duplicate product does not satisfy the barrier")
        seen.add(key)
    if seen != required:
        raise ValueError("Missing or extra seed/arm product")


def _prepare_loaded_v1(data, phase):
    import numpy as np
    if data.archive_sha256 != ARCHIVE_SHA256 or tuple(data.counties) != COUNTIES:
        raise ValueError("Exact source bytes and county-name join required")
    counts = np.array(data.counts, dtype=np.float64, copy=True)
    if counts.shape != (OBSERVATIONS, 20) or len(data.dates) != OBSERVATIONS:
        raise ValueError("Original complete522weekly/20county support")
    if phase == "development":
        counts[VAL_END:] = np.nan
    elif phase != "test":
        raise ValueError("Explicit development or released heldout-test phase")
    available_end = VAL_END if phase == "development" else OBSERVATIONS
    if not np.isfinite(counts[:available_end]).all():
        raise ValueError("Available case observations must be finite")
    mean = counts[:TRAIN_END].mean(axis=0)
    scale = counts[:TRAIN_END].std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    raw_edges = np.asarray(data.edge_index, dtype=np.int64)
    if (raw_edges.shape != (2, 102) or int((raw_edges[0] == raw_edges[1]).sum()) != 20
            or raw_edges.min() != 0 or raw_edges.max() != 19):
        raise ValueError("Exact102native records/20self loops and complete20counties")
    edges = raw_edges[:, raw_edges[0] != raw_edges[1]].copy()
    pairs = set(map(tuple, edges.T.tolist()))
    if len(pairs) != 82 or any((v, u) not in pairs for u, v in pairs):
        raise ValueError("All41undirected native county links retained exactly once each direction")
    raw_edges = raw_edges.copy()
    for value in (counts, mean, scale, edges, raw_edges):
        value.setflags(write=False)
    return PreparedChickenpoxV1(counts, mean, scale, edges, raw_edges, COUNTIES, tuple(data.dates), phase)


def load_development_v1(archive_path):
    # The existing pinned loader performs bounded original archive/schema/name checks.
    from arggnn.chickenpox import load_chickenpox_archive
    return _prepare_loaded_v1(load_chickenpox_archive(archive_path), "development")


def load_test_v1(archive_path, *, release_selections: Callable):
    # ROOT authenticates real saved references and grants access before this callback returns.
    _seal_set(release_selections(), "selected_checkpoint")
    from arggnn.chickenpox import load_chickenpox_archive
    return _prepare_loaded_v1(load_chickenpox_archive(archive_path), "test")


def target_indices_v1(split):
    if split == "train":
        return tuple(range(HISTORY, TRAIN_END))
    if split == "validation":
        return tuple(range(TRAIN_END, VAL_END))
    if split == "test":
        return tuple(range(VAL_END, OBSERVATIONS))
    raise ValueError("Fixed chronological train, validation or test target split")


def history_batch_v1(prepared, targets, *, shuffle_prefix=False):
    import numpy as np
    import torch
    from .generic_core_v1 import ObservedHistoryV1
    targets = tuple(operator.index(t) for t in targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Distinct nonempty target indices per batch")
    if any(t < HISTORY or t >= (VAL_END if prepared.phase == "development" else OBSERVATIONS)
           for t in targets):
        raise ValueError("No unavailable target/future rows may enter this phase")
    histories, content = [], []
    for at in targets:
        # The ONLY numeric source slice used for this forecast.
        values = (prepared.counts[at-HISTORY:at] - prepared.mean) / prepared.scale
        order = list(range(HISTORY))
        if shuffle_prefix:
            prefix = order[:-1]
            random.Random(9031 + at).shuffle(prefix)
            order = prefix + [HISTORY-1]
        values = values[order]
        histories.append(values)
        content.append(tuple(at-HISTORY+i for i in order))
    x = torch.as_tensor(np.stack(histories)[..., None].copy(), dtype=torch.float32)
    return ObservedHistoryV1(
        x=x, edge_index=torch.tensor(prepared.edges.copy(), dtype=torch.long),
        durations=torch.ones(HISTORY, dtype=torch.float32), entity_ids=prepared.counties,
        target_indices=targets, content_indices=tuple(content)).validate()


def _chunks(values, size=BATCH_SIZE):
    for start in range(0, len(values), size):
        yield tuple(values[start:start+size])


def predict_v1(model, prepared, targets, *, boundary: Callable, shuffle_prefix=False):
    import numpy as np
    import torch
    targets = tuple(targets)
    output = []
    model.eval()
    with torch.no_grad():
        for ids in _chunks(targets):
            boundary(dict(event="predict_batch", targets=list(ids), shuffled=shuffle_prefix))
            predicted = model(history_batch_v1(prepared, ids, shuffle_prefix=shuffle_prefix))
            output.append(predicted.detach().cpu().numpy().astype(np.float64))
    standardized = np.concatenate(output, axis=0)
    return standardized * prepared.scale + prepared.mean


def metrics_v1(prediction, truth):
    import numpy as np
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 20:
        raise ValueError("Complete same-support[weeks,20] forecasts and external targets")
    if not (np.isfinite(prediction).all() and np.isfinite(truth).all()):
        raise ValueError("No omitted/nonfinite forecast or target")
    error = prediction - truth
    return dict(mae_cases=float(np.abs(error).mean()),
                rmse_cases=float(np.sqrt(np.square(error).mean())),
                per_county_mae=np.abs(error).mean(axis=0).tolist(),
                per_week_mae=np.abs(error).mean(axis=1).tolist(),
                forecast_weeks=len(error), prediction_clipping="none")


def fit_development_pair_v1(prepared, *, seed: int, boundary: Callable,
                            checkpoint_sink: Callable):
    """Return two selected states through ROOT's sink, without owning any output path.

    boundary MUST enforce the actual admitted runtime/storage policy in the
    caller. This algorithm is not a resource guard or an execution authority.
    """
    import numpy as np
    import torch
    from .generic_core_v1 import matched_pair_v1
    if prepared.phase != "development" or seed not in SEEDS:
        raise ValueError("Only fixed TRAIN/VAL completion and predeclared seed")
    if not callable(boundary) or not callable(checkpoint_sink):
        raise ValueError("ROOT boundary and bounded checkpoint sink are mandatory")
    # Small fixed graph protocol is CPU-only. No CUDA/device discovery is performed.
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    family = matched_pair_v1(seed)
    training, validation = target_indices_v1("train"), target_indices_v1("validation")
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
                  for arm, model in family.items()}
    best = {arm: dict(mae=float("inf"), epoch=None, state=None) for arm in ARMS}
    history = {arm: [] for arm in ARMS}
    updates = {arm: 0 for arm in ARMS}
    for epoch in range(1, EPOCHS+1):
        order = list(training)
        random.Random(seed + epoch).shuffle(order)
        # Exactly the same batch exposure/order for each independently optimized arm.
        for arm in ARMS:
            model, optimizer = family[arm], optimizers[arm]
            model.train()
            for ids in _chunks(order):
                boundary(dict(event="train_batch", seed=seed, arm=arm,
                              epoch=epoch, update=updates[arm]+1, targets=list(ids)))
                observed = history_batch_v1(prepared, ids)
                labels = torch.as_tensor(
                    ((prepared.counts[list(ids)] - prepared.mean)/prepared.scale).copy(),
                    dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(observed)
                loss = torch.nn.functional.mse_loss(prediction, labels)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite loss; no fallback/reduced support")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates[arm] += 1
            prediction = predict_v1(model, prepared, validation, boundary=boundary)
            score = metrics_v1(prediction, prepared.counts[list(validation)])
            history[arm].append(dict(epoch=epoch, updates=updates[arm], **score))
            # Strict improvement preserves earliest epoch ties.
            if score["mae_cases"] < best[arm]["mae"]:
                best[arm] = dict(mae=score["mae_cases"], epoch=epoch,
                                state={k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()})
    result = {}
    for arm in ARMS:
        if updates[arm] != 800 or best[arm]["state"] is None:
            raise RuntimeError("Incomplete40epochs/800updates or no selected finite state")
        record = dict(seed=seed, arm=arm, selected_epoch=best[arm]["epoch"],
                      selected_validation_mae_cases=best[arm]["mae"],
                      epochs_completed=EPOCHS, optimizer_updates=updates[arm],
                      training_targets=list(training), validation_targets=list(validation),
                      family_contract=family[arm].interface_contract(),
                      parameters=sum(p.numel() for p in family[arm].parameters()),
                      history=history[arm], test_accessed=False)
        boundary(dict(event="save_selected_checkpoint", seed=seed, arm=arm))
        checkpoint_ref = checkpoint_sink(seed, arm, best[arm]["state"], record)
        result[arm] = dict(record, selected_checkpoint_ref=checkpoint_ref)
    return result


def predict_test_v1(model, prepared, *, boundary: Callable, shuffle_prefix=False):
    if prepared.phase != "test":
        raise ValueError("Heldout raw observations require the separate six-selection release")
    return predict_v1(model, prepared, target_indices_v1("test"),
                      boundary=boundary, shuffle_prefix=shuffle_prefix)


def score_test_v1(all_predictions, prepared, *, release_prediction_seals: Callable):
    """ROOT first seals ALL six complete chronological prediction products."""
    _seal_set(release_prediction_seals(), "test_prediction")
    required = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if prepared.phase != "test" or set(all_predictions) != required:
        raise ValueError("All six complete fixed model products before TEST scoring")
    import numpy as np
    truth = prepared.counts[list(target_indices_v1("test"))]
    if any(np.asarray(value).shape != (79, 20) for value in all_predictions.values()):
        raise ValueError("Full79weeks/all20counties, no paired-denominator reduction")
    return {key: metrics_v1(value, truth) for key, value in all_predictions.items()}


def classical_predictions_v1(prepared):
    """No fitting: observed last value and lag52 on the identical TEST targets."""
    import numpy as np
    if prepared.phase != "test":
        raise ValueError("Classical test forecasts follow the same heldout release")
    ids = target_indices_v1("test")
    return {"last_value": np.stack([prepared.counts[t-1] for t in ids]),
            "seasonal_lag52": np.stack([prepared.counts[t-HISTORY] for t in ids])}


def paired_block_interval_v1(full_by_seed, no_memory_by_seed, truth):
    """Conditional uncertainty on one series, after complete TEST product seals.

    Average paired absolute errors across seeds/counties first; resample common
    circular13-week blocks, not independent counties or overlapping windows.
    """
    import numpy as np
    full, null, truth = np.asarray(full_by_seed), np.asarray(no_memory_by_seed), np.asarray(truth)
    if full.shape != null.shape or full.shape != (3, 79, 20) or truth.shape != (79, 20):
        raise ValueError("All3seeds/79weeks/20counties on identical support")
    differences = (np.abs(full-truth[None]) - np.abs(null-truth[None])).mean(axis=(0, 2))
    rng = np.random.default_rng(7721)
    samples = np.empty(10000, dtype=np.float64)
    for draw in range(len(samples)):
        starts = rng.integers(0, 79, size=7)
        index = np.concatenate([(start+np.arange(13)) % 79 for start in starts])[:79]
        samples[draw] = differences[index].mean()
    return dict(mean_delta_mae_cases=float(differences.mean()),
                interval95_cases=np.quantile(samples, [.025, .975]).tolist(),
                block_weeks=13, draws=10000, seed=7721,
                interpretation="conditional single-series temporal block interval; not independent populations")


def fit_frozen_ridge_v1(development):
    """Reconstruct the already selected alpha1000 coefficients on TRAIN only.

    This completes a previously unopened TEST endpoint. It is not a new alpha
    search or a rerun of the answered validation pilot.
    """
    if development.phase != "development":
        raise ValueError("Ridge coefficients must be built from TRAIN/VAL-restricted data")
    from types import SimpleNamespace
    from arggnn.chickenpox import ForecastProtocol, TrainingScaler, LinearRidgeForecaster
    scaler = TrainingScaler(development.mean, development.scale, TRAIN_END)
    data = SimpleNamespace(counts=development.counts)
    return LinearRidgeForecaster.fit_training(data, ForecastProtocol(), scaler, 1000.)


def predict_frozen_ridge_v1(fitted, prepared, *, boundary):
    import numpy as np
    if prepared.phase != "test":
        raise ValueError("Same released heldout endpoint as the current core")
    output = []
    for at in target_indices_v1("test"):
        boundary(dict(event="frozen_ridge_prediction", target=at))
        output.append(fitted.predict(prepared.counts[at-HISTORY:at]))
    return np.stack(output)


def predict_verified_legacy_gru_v1(model, prepared, *, authenticate_legacy, boundary):
    """No optimization or checkpoint loading here; ROOT supplies the verified old model.

    Source changed since the old audit. A strict state load alone does not prove
    identical constructor semantics. The separate ROOT review must establish them.
    """
    if prepared.phase != "test":
        raise ValueError("Same heldout release, no prior-model TEST tuning")
    proof = authenticate_legacy()
    if (proof.get("checkpoint_sha256") !=
            "0a311637a0e011a60e691a53aff2088e5d296b3c512b0b2dd77545a4a57b10bc"
            or proof.get("source_constructor_verified") is not True
            or proof.get("optimization_performed") is not False):
        raise ValueError("Exact reused seed101 selected GRU, independently authenticated constructor")
    from types import SimpleNamespace
    import numpy as np
    import torch
    from arggnn.chickenpox import TrainingScaler, forecast_sequence
    data = SimpleNamespace(counts=prepared.counts, edge_index=prepared.native_edges)
    scaler = TrainingScaler(prepared.mean, prepared.scale, TRAIN_END)
    output = []
    model.eval()
    with torch.no_grad():
        for at in target_indices_v1("test"):
            boundary(dict(event="reused_legacy_gru_prediction", target=at))
            observed = forecast_sequence(data, scaler, at, HISTORY)
            standardized = model(observed)[0, :, 0].detach().cpu().numpy()
            output.append(scaler.inverse(standardized))
    return np.stack(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)  # Deliberately offers no data/training/evaluation CLI flag.
    print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE", execution_authorized=False,
                          protocol=protocol_v1(), torch_imported=False,
                          data_opened=False, optimizer_updates=0), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pure metrics: native-six primary and explicitly derived founder-four."""
from __future__ import annotations

from .. import european_native_models as io
from ..european_sampling import NATIVE_PATHS

C4 = ("Ana", "CHG", "WHG", "EHG")


def metrics(probabilities, labels, class_names):
    import numpy as np
    p, y = np.asarray(probabilities), np.asarray(labels)
    k = len(class_names)
    io._need(p.shape == (len(y), k) and y.shape == (len(y),) and len(y) > 0
             and y.dtype.kind in "iu" and ((y >= 0) & (y < k)).all()
             and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
             and np.allclose(p.sum(1), 1, atol=1e-6, rtol=1e-5),
             "Complete finite normalized raw probabilities and actual labels required")
    calls = p.argmax(1)
    confusion = np.bincount(y * k + calls, minlength=k*k).reshape(k, k)
    per_class = {}
    for i, name in enumerate(class_names):
        tp, actual, predicted = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        per_class[name] = dict(precision=tp / predicted if predicted else 0.,
            recall=tp / actual if actual else 0.,
            F1=2 * tp / (actual + predicted) if actual + predicted else 0.,
            actual_count=actual, predicted_count=predicted)
    true_p = p[np.arange(len(y)), y]
    entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.)), 0), axis=1)
    confidence = p.max(1)
    reliability = []
    for i in range(10):
        mask = (confidence >= i/10) & ((confidence < (i+1)/10) if i < 9 else confidence <= 1)
        n = int(mask.sum())
        reliability.append(dict(lower=i/10, upper=(i+1)/10, records=n,
            mean_confidence=float(confidence[mask].mean()) if n else None,
            accuracy=float((calls[mask] == y[mask]).mean()) if n else None))
    zero = int((true_p == 0).sum())
    return dict(records=len(y), accuracy=float((calls == y).mean()),
        macro_F1=float(np.mean([v["F1"] for v in per_class.values()])),
        per_class=per_class, confusion=confusion.tolist(),
        confusion_convention="truth rows, prediction columns",
        source_style_confusion_predicted_rows_truth_columns=confusion.T.tolist(),
        raw_NLL=None if zero else float(-np.log(true_p).mean()),
        raw_NLL_is_infinite=bool(zero), zero_probability_truth_count=zero,
        Brier=float(np.square(p - np.eye(k)[y]).sum(1).mean()),
        mean_predictive_entropy_nats=float(entropy.mean()),
        mean_max_probability=float(confidence.mean()), reliability_bins=reliability,
        ECE_10_equal_width=float(sum(r["records"] / len(y) *
            abs(r["mean_confidence"] - r["accuracy"]) for r in reliability if r["records"])),
        calibrated_probabilities_claimed=False, probabilities_clipped_or_renormalized=False)


def endpoints(probabilities, labels):
    import numpy as np
    p, y = np.asarray(probabilities), np.asarray(labels)
    collapsed = np.column_stack((p[:, 0] + p[:, 4], p[:, 1] + p[:, 5], p[:, 2], p[:, 3]))
    return dict(native6=metrics(p, y, NATIVE_PATHS),
        derived_C4=metrics(collapsed, np.asarray([0, 1, 2, 3, 0, 1])[y], C4),
        primary="native6", derived_C4_is_separate_from_four_path_training=True)


def parent_summary(values):
    import numpy as np
    io._need(len(values) == 5, "Exactly five independent parents required")
    return dict(mean=None if any(v is None for v in values) else float(np.mean(values)),
        sample_SD=None if any(v is None for v in values) else float(np.std(values, ddof=1)),
        independent_parents=5, weighting="equal parent", ddof=1)

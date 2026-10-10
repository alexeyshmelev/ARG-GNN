"""Pure CPU metrics for a future frozen ancestry analysis.

No data loader, training, test generation or scientific-phase promotion lives here.
Callers must bind ordered parent/seed IDs to predictions before joining labels.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t


def logit_metrics(logits, labels, *, bins=15, diploid_pairs=None):
    """Unweighted fixed-grid metrics for one parent from original raw logits."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if z.ndim != 3 or z.shape[:2] != y.shape or not y.size or z.shape[-1] < 2:
        raise ValueError("Require nonempty aligned [position,query,class] logits")
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= z.shape[-1]):
        raise ValueError("Labels must be integer class slots")
    if not np.isfinite(z).all():
        raise ValueError("Raw logits must be finite")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("Positive integer bin count required")
    shifted = z - z.max(axis=-1, keepdims=True)
    logp = shifted - logsumexp(shifted, axis=-1, keepdims=True)
    if not np.isfinite(logp).all():
        raise ValueError("Log-softmax exceeds float64 representable range")
    p = np.exp(logp)
    k = z.shape[-1]
    chosen = np.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    pred = z.argmax(axis=-1)
    confusion = np.bincount((y*k+pred).ravel(), minlength=k*k).reshape(k, k)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    tp = confusion.diagonal()
    per_class = []
    all_class_f1 = []
    for c in range(k):
        prec = float(tp[c]/predicted_support[c]) if predicted_support[c] else None
        recall = float(tp[c]/support[c]) if support[c] else None
        denom = int(support[c]+predicted_support[c])
        f1 = float(2*tp[c]/denom) if denom else None
        all_class_f1.append(0. if f1 is None else f1)
        per_class.append(dict(class_slot=c, support=int(support[c]),
            predicted_support=int(predicted_support[c]), precision=prec, recall=recall, f1=f1))
    onehot = np.eye(k)[y]
    confidence = p.max(axis=-1)
    correct = pred == y
    edges = np.linspace(0., 1., bins+1)
    assignments = np.minimum(np.floor(confidence*bins).astype(int), bins-1)
    calibration = []
    for b in range(bins):
        mask = assignments == b
        count = int(mask.sum())
        calibration.append(dict(lower=float(edges[b]), upper=float(edges[b+1]),
            observations=count, mean_confidence=float(confidence[mask].mean()) if count else None,
            accuracy=float(correct[mask].mean()) if count else None))
    classwise = []
    for c in range(k):
        membership = np.minimum(np.floor(p[..., c]*bins).astype(int), bins-1)
        records = []
        for b in range(bins):
            mask = membership == b
            count = int(mask.sum())
            records.append(dict(lower=float(edges[b]), upper=float(edges[b+1]),
                observations=count, mean_probability=float(p[..., c][mask].mean()) if count else None,
                frequency=float((y[mask] == c).mean()) if count else None))
        classwise.append(dict(class_slot=c, bins=records))
    result = dict(observations=int(y.size), nll=float(-chosen.mean()),
        historical_floor_1e12_nll=float(-np.maximum(chosen, np.log(1e-12)).mean()),
        accuracy=float(correct.mean()), brier=float(((p-onehot)**2).sum(axis=-1).mean()),
        macro_f1_all_classes_fixed_zero=float(np.mean(all_class_f1)),
        confusion=confusion.tolist(), per_class=per_class, calibration=calibration,
        classwise_reliability=classwise, nll_policy="float64 stable log-softmax of uncalibrated logits",
        weighting="uniform fixed coordinates and query haplotypes within one parent")
    if diploid_pairs is not None:
        pairs = np.asarray(diploid_pairs)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or
                not np.issubdtype(pairs.dtype, np.integer) or not pairs.size or
                np.any(pairs < 0) or np.any(pairs >= z.shape[1]) or
                len(np.unique(pairs)) != pairs.size):
            raise ValueError("Explicit nonoverlapping valid diploid haplotype pairs required")
        estimated = p[:, pairs[:, 0]] + p[:, pairs[:, 1]]
        true = onehot[:, pairs[:, 0]] + onehot[:, pairs[:, 1]]
        result["diploid_dosage_mae"] = float(np.abs(estimated-true).mean())
        result["diploid_dosage_mae_per_class"] = np.abs(estimated-true).mean(axis=(0, 1)).tolist()
        result["diploid_pairs"] = pairs.tolist()
    return result


def holm_adjust(p_values):
    """Holm FWER adjusted p-values, retaining original contrast order."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Finite vector of probability p-values required")
    if not len(p):
        return []
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order])
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., adjusted_sorted)
    return adjusted.tolist()


def paired_parent_report(candidate_nll, baseline_nll, *, parent_ids, training_seeds,
                         bootstrap_seed=20261115, bootstrap_resamples=10000,
                         practical_improvement=0.010):
    """Paired t and bootstrap analysis of N parents by S fixed trained models.

    Each parent contributes one seed-averaged difference. Seed cells, queries and
    positions are never counted as independent sampling units. No ensembles.
    """
    c = np.asarray(candidate_nll, dtype=np.float64)
    b = np.asarray(baseline_nll, dtype=np.float64)
    if c.ndim != 2 or c.shape != b.shape or c.shape[0] < 2 or c.shape[1] < 1:
        raise ValueError("Require aligned [independent parent,fixed training seed] loss matrices")
    if not np.isfinite(c).all() or not np.isfinite(b).all() or np.any(c < 0) or np.any(b < 0):
        raise ValueError("All parent/seed losses must be finite nonnegative values")
    if (len(parent_ids) != c.shape[0] or len(set(parent_ids)) != len(parent_ids) or
            len(training_seeds) != c.shape[1] or len(set(training_seeds)) != len(training_seeds)):
        raise ValueError("Distinct explicitly ordered parent IDs and training seeds required")
    if (isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or
            bootstrap_resamples < 1 or not np.isfinite(practical_improvement) or practical_improvement <= 0):
        raise ValueError("Positive bootstrap budget and practical improvement required")
    cells = c-b
    delta = cells.mean(axis=1)
    n = len(delta)
    estimate = float(delta.mean())
    sd = float(delta.std(ddof=1))
    se = sd/np.sqrt(n)
    critical = float(student_t.ppf(.975, n-1))
    interval = [estimate-critical*se, estimate+critical*se]
    if sd == 0:
        statistic = None
        pvalue = 1. if estimate == 0 else 0.
    else:
        statistic = estimate/se
        pvalue = float(2*student_t.sf(abs(statistic), n-1))
    rng = np.random.default_rng(bootstrap_seed)
    sample_means = np.empty(bootstrap_resamples, dtype=np.float64)
    for start in range(0, bootstrap_resamples, 1000):
        count = min(1000, bootstrap_resamples-start)
        indices = rng.integers(0, n, size=(count, n))
        sample_means[start:start+count] = delta[indices].mean(axis=1)
    boot = np.quantile(sample_means, [.025, .975]).tolist()
    seed_means = cells.mean(axis=0)
    t_favorable = bool(interval[1] < 0)
    bootstrap_favorable = bool(boot[1] < 0)
    return dict(independent_parents=n, fixed_training_models=c.shape[1],
        parent_ids=list(parent_ids), training_seeds=list(training_seeds),
        candidate_mean_nll=float(c.mean(axis=1).mean()),
        baseline_mean_nll=float(b.mean(axis=1).mean()),
        difference_candidate_minus_baseline=estimate, paired_parent_sd=sd,
        paired_parent_standard_error=float(se), degrees_of_freedom=n-1,
        t_statistic=statistic, two_sided_t_pvalue=pvalue, t_interval95=interval,
        achieved_t_interval_halfwidth=float(critical*se),
        bootstrap_interval95=boot, bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        parent_seed_differences=cells.tolist(), paired_parent_differences=delta.tolist(),
        seed_specific_mean_differences=seed_means.tolist(),
        seed_mean_range=[float(seed_means.min()), float(seed_means.max())],
        seed_mean_sd=float(seed_means.std(ddof=1)) if len(seed_means) > 1 else None,
        practical_improvement_nats=float(practical_improvement),
        t_upper_below_zero=t_favorable, bootstrap_upper_below_zero=bootstrap_favorable,
        favorable_interval_agreement=t_favorable == bootstrap_favorable,
        t_upper_below_negative_practical_threshold=bool(interval[1] < -practical_improvement),
        inference_scope="conditional on the frozen training cohort and fixed selected checkpoints",
        resampling_unit="whole independent parent vector across fixed seeds and methods",
        phase_policy="This calculation alone does not establish an independent confirmatory experiment.")


def boundary_match(true_boundaries, predicted_boundaries, *, tolerance_bp=1000.):
    """Maximum-cardinality sorted one-to-one physical boundary matching.

    Matching uses earliest feasible pairs. Localization distances describe that
    deterministic pairing, not a globally minimum-distance assignment.
    """
    actual = np.asarray(true_boundaries, dtype=np.float64)
    predicted = np.asarray(predicted_boundaries, dtype=np.float64)
    if (actual.ndim != 1 or predicted.ndim != 1 or not np.isfinite(actual).all() or
            not np.isfinite(predicted).all() or not np.isfinite(tolerance_bp) or tolerance_bp < 0):
        raise ValueError("Finite sorted boundary coordinates and nonnegative tolerance required")
    if np.any(np.diff(actual) <= 0) or np.any(np.diff(predicted) <= 0):
        raise ValueError("Boundary coordinates must be strictly increasing")
    i = j = 0
    matches = []
    while i < len(actual) and j < len(predicted):
        if abs(actual[i]-predicted[j]) <= tolerance_bp:
            matches.append(dict(true_index=i, predicted_index=j,
                true_bp=float(actual[i]), predicted_bp=float(predicted[j]),
                absolute_error_bp=float(abs(actual[i]-predicted[j]))))
            i += 1
            j += 1
        elif predicted[j] < actual[i]-tolerance_bp:
            j += 1
        else:
            i += 1
    count = len(matches)
    nt, npred = len(actual), len(predicted)
    return dict(true_boundaries=nt, predicted_boundaries=npred, matched=count,
        missed=nt-count, extra=npred-count, matches=matches,
        precision=float(count/npred) if npred else None,
        recall=float(count/nt) if nt else None,
        f1=float(2*count/(nt+npred)) if nt else None,
        absolute_switch_count_error=abs(nt-npred), tolerance_bp=float(tolerance_bp),
        matching="maximum-cardinality earliest-feasible one-to-one ordered matching")

import numpy as np
import pytest
from scipy.stats import ttest_1samp
from arggnn.confirmatory_metrics import logit_metrics, holm_adjust, paired_parent_report, boundary_match


def test_extreme_logits_keep_uncapped_nll():
    z = np.array([[[1000., -1000., 0., 1.]]])
    result = logit_metrics(z, np.array([[1]]))
    assert result["nll"] == 2000.
    assert result["historical_floor_1e12_nll"] == pytest.approx(-np.log(1e-12))
    assert result["brier"] == 2.
    assert sum(b["observations"] for b in result["calibration"]) == 1
    assert result["calibration"][-1]["mean_confidence"] == 1.


def test_missing_class_support_is_visible_without_fabricated_recall():
    result = logit_metrics(np.zeros((3, 2, 4)), np.zeros((3, 2), dtype=int))
    assert result["nll"] == pytest.approx(np.log(4))
    assert result["per_class"][0]["recall"] == 1.
    for row in result["per_class"][1:]:
        assert row["support"] == 0
        assert row["recall"] is None and row["f1"] is None
    assert result["macro_f1_all_classes_fixed_zero"] == .25


def test_logits_shift_and_grid_duplication_do_not_change_metric():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(7, 4, 4))
    y = rng.integers(0, 4, size=(7, 4))
    a = logit_metrics(z, y)
    b = logit_metrics(np.repeat(z+1000., 3, axis=0), np.repeat(y, 3, axis=0))
    for key in ("nll", "accuracy", "brier", "macro_f1_all_classes_fixed_zero"):
        assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_diploid_pairing_is_explicit_and_phase_invariant():
    z = np.array([[[4., 0.], [0., 4.], [1., 2.], [3., 1.]]])
    y = np.array([[0, 1, 0, 0]])
    a = logit_metrics(z, y, diploid_pairs=[[0, 1], [2, 3]])
    perm = [1, 0, 3, 2]
    b = logit_metrics(z[:, perm], y[:, perm], diploid_pairs=[[0, 1], [2, 3]])
    assert a["diploid_dosage_mae"] == pytest.approx(b["diploid_dosage_mae"])
    with pytest.raises(ValueError):
        logit_metrics(z, y, diploid_pairs=[[0, 1], [1, 2]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_logits_rejected(bad):
    z = np.zeros((1, 1, 4))
    z[0, 0, 0] = bad
    with pytest.raises(ValueError):
        logit_metrics(z, np.zeros((1, 1), dtype=int))


def test_noninteger_labels_rejected():
    with pytest.raises(ValueError):
        logit_metrics(np.zeros((1, 1, 4)), np.zeros((1, 1)))


def test_paired_unit_is_parent_and_agrees_with_independent_scipy_ttest():
    delta = np.array([-.04, -.02, .01, -.03, -.01])
    c = 1. + delta[:, None] + np.array([-.1, 0., .1])[None, :]
    b = np.ones_like(c)
    r = paired_parent_report(c, b, parent_ids=list(range(5)), training_seeds=[101, 202, 303])
    oracle = ttest_1samp(delta, 0.)
    assert r["degrees_of_freedom"] == 4
    assert r["independent_parents"] == 5 and r["fixed_training_models"] == 3
    assert r["two_sided_t_pvalue"] == pytest.approx(oracle.pvalue)
    assert r["difference_candidate_minus_baseline"] == pytest.approx(delta.mean())
    assert r["paired_parent_sd"] == pytest.approx(delta.std(ddof=1))
    assert r["seed_mean_sd"] > .09


def test_bootstrap_preserves_seed_cancellation_and_is_reproducible():
    c = np.ones((7, 3)) + np.array([-.4, .1, .3])
    b = np.ones_like(c)
    kwargs = dict(parent_ids=list(range(7)), training_seeds=[101, 202, 303])
    r = paired_parent_report(c, b, **kwargs)
    repeat = paired_parent_report(c, b, **kwargs)
    assert r == repeat
    assert np.max(np.abs(r["bootstrap_interval95"])) < 1e-15
    assert r["achieved_t_interval_halfwidth"] < 1e-15


def test_equal_parent_weighting_does_not_count_seed_cells_as_replicates():
    c = np.array([[.1, .1, .1], [1.5, 1.5, 1.5]])
    b = np.ones_like(c)
    r = paired_parent_report(c, b, parent_ids=["small", "large"], training_seeds=[1, 2, 3])
    assert r["difference_candidate_minus_baseline"] == pytest.approx(-.2)
    assert r["degrees_of_freedom"] == 1
    assert r["seed_mean_sd"] == pytest.approx(0.)


def test_loss_correspondence_and_complete_case_contract():
    kwargs = dict(parent_ids=["a", "a"], training_seeds=[1, 2, 3])
    with pytest.raises(ValueError):
        paired_parent_report(np.ones((2, 3)), np.ones((2, 3)), **kwargs)
    kwargs["parent_ids"] = ["a", "b"]
    with pytest.raises(ValueError):
        paired_parent_report(np.ones((2, 3)), np.ones((2, 2)), **kwargs)
    z = np.ones((2, 3))
    z[0, 0] = np.nan
    with pytest.raises(ValueError):
        paired_parent_report(z, np.ones((2, 3)), **kwargs)


def test_holm_matches_manual_and_preserves_order():
    assert holm_adjust([.03, .01, .04]) == pytest.approx([.06, .03, .06])
    assert holm_adjust([.1, .1, .1]) == pytest.approx([.3, .3, .3])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        holm_adjust([.1, np.nan])


def test_one_to_one_boundary_matching_retains_missed_extra():
    result = boundary_match([100., 1100., 4000.], [50., 120., 1000., 8000.], tolerance_bp=200.)
    assert result["matched"] == 2 and result["missed"] == 1 and result["extra"] == 2
    assert result["precision"] == .5
    assert result["recall"] == pytest.approx(2/3)
    assert len({m["predicted_index"] for m in result["matches"]}) == 2
    assert result["absolute_switch_count_error"] == 1


def test_no_true_boundaries_is_undefined_with_visible_false_cost():
    result = boundary_match([], [100., 200.])
    assert result["recall"] is None and result["f1"] is None
    assert result["extra"] == 2 and result["absolute_switch_count_error"] == 2
    empty = boundary_match([], [])
    assert empty["precision"] is None
    missed = boundary_match([100.], [])
    assert missed["f1"] == 0. and missed["recall"] == 0.

"""
Unit tests runnable right now, offline, with no checkpoints and no GPU --
these validate the reproduction's own code (the part the proposal commits
to writing student-side) before it ever touches a real checkpoint.

Run with:  pytest tests/ -v
"""
import numpy as np
import pytest

from src.alignment import align_prediction, least_squares_scale_shift
from src.decision_rule import DecisionInputs, Verdict, apply_decision_rule
from src.metrics import (
    DA2KPairResult,
    abs_rel,
    bootstrap_ci,
    bootstrap_paired_difference,
    da2k_accuracy,
    da2k_accuracy_by_scene,
    da2k_pair_correct,
    delta1,
)


# ---------------------------------------------------------------------------
# alignment.py
# ---------------------------------------------------------------------------

def test_scale_shift_recovers_known_transform():
    rng = np.random.default_rng(0)
    gt = rng.uniform(1, 10, size=(50, 50)).astype(np.float32)
    true_s, true_t = 2.5, 0.3
    pred = (gt - true_t) / true_s  # so that true_s * pred + true_t == gt
    mask = np.ones_like(gt, dtype=bool)

    s, t = least_squares_scale_shift(pred, gt, mask)
    assert s == pytest.approx(true_s, rel=1e-3)
    assert t == pytest.approx(true_t, abs=1e-2)


def test_scale_shift_too_few_pixels_raises():
    pred = np.ones((10, 10))
    gt = np.ones((10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :5] = True  # only 5 valid pixels
    with pytest.raises(ValueError):
        least_squares_scale_shift(pred, gt, mask)


def test_align_prediction_gives_zero_error_on_noiseless_case():
    gt = np.full((20, 20), 5.0, dtype=np.float32)
    pred = np.full((20, 20), 2.0, dtype=np.float32)
    mask = np.ones_like(gt, dtype=bool)
    aligned = align_prediction(pred, gt, mask)
    assert np.allclose(aligned, gt, atol=1e-4)


# ---------------------------------------------------------------------------
# metrics.py: Track A
# ---------------------------------------------------------------------------

def test_abs_rel_zero_for_perfect_prediction():
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.ones(4, dtype=bool)
    assert abs_rel(gt.copy(), gt, mask) == pytest.approx(0.0)


def test_abs_rel_known_value():
    gt = np.array([2.0, 4.0])
    pred = np.array([1.0, 5.0])  # errors: 1/2=0.5, 1/4=0.25 -> mean 0.375
    mask = np.ones(2, dtype=bool)
    assert abs_rel(pred, gt, mask) == pytest.approx(0.375)


def test_delta1_all_within_threshold():
    gt = np.array([10.0, 10.0])
    pred = np.array([10.5, 9.5])  # ratios ~1.05, well within 1.25
    mask = np.ones(2, dtype=bool)
    assert delta1(pred, gt, mask) == pytest.approx(1.0)


def test_delta1_partial():
    gt = np.array([10.0, 10.0])
    pred = np.array([10.0, 20.0])  # second ratio = 2.0, fails threshold
    mask = np.ones(2, dtype=bool)
    assert delta1(pred, gt, mask) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# metrics.py: Track B (DA-2K)
# ---------------------------------------------------------------------------

def test_da2k_pair_correct_higher_is_closer():
    pred_depth = np.zeros((10, 10))
    pred_depth[2, 2] = 5.0  # point1: high value = "closer" under default convention
    pred_depth[7, 7] = 1.0  # point2: low value = "farther"
    correct = da2k_pair_correct(
        pred_depth, point1_hw=(2, 2), point2_hw=(7, 7),
        closer_point="point1", higher_is_closer=True,
    )
    assert correct is True


def test_da2k_pair_incorrect_when_flipped():
    pred_depth = np.zeros((10, 10))
    pred_depth[2, 2] = 1.0
    pred_depth[7, 7] = 5.0
    correct = da2k_pair_correct(
        pred_depth, point1_hw=(2, 2), point2_hw=(7, 7),
        closer_point="point1", higher_is_closer=True,
    )
    assert correct is False


def test_da2k_pair_lower_is_closer_convention():
    pred_depth = np.zeros((10, 10))
    pred_depth[2, 2] = 1.0  # low value = "closer" under this convention
    pred_depth[7, 7] = 5.0
    correct = da2k_pair_correct(
        pred_depth, point1_hw=(2, 2), point2_hw=(7, 7),
        closer_point="point1", higher_is_closer=False,
    )
    assert correct is True


def test_da2k_accuracy_and_by_scene():
    results = [
        DA2KPairResult("img1", "indoor", True, (0, 0), (1, 1), "point1"),
        DA2KPairResult("img1", "indoor", False, (0, 0), (1, 1), "point1"),
        DA2KPairResult("img2", "outdoor", True, (0, 0), (1, 1), "point1"),
    ]
    assert da2k_accuracy(results) == pytest.approx(2 / 3)
    by_scene = da2k_accuracy_by_scene(results)
    assert by_scene["indoor"] == pytest.approx(0.5)
    assert by_scene["outdoor"] == pytest.approx(1.0)


def test_da2k_accuracy_empty_raises():
    with pytest.raises(ValueError):
        da2k_accuracy([])


# ---------------------------------------------------------------------------
# metrics.py: bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_true_mean_for_constant_array():
    values = np.full(100, 0.7)
    point, lo, hi = bootstrap_ci(values, n_boot=200, seed=1)
    assert point == pytest.approx(0.7)
    # Degenerate (zero-variance) input: all bootstrap resamples equal 0.7,
    # so lo/hi collapse to 0.7 up to floating-point noise.
    assert lo == pytest.approx(0.7)
    assert hi == pytest.approx(0.7)


def test_bootstrap_ci_narrows_with_less_variance():
    rng = np.random.default_rng(0)
    low_var = rng.normal(0, 0.01, size=500)
    high_var = rng.normal(0, 1.0, size=500)
    _, lo1, hi1 = bootstrap_ci(low_var, n_boot=500, seed=1)
    _, lo2, hi2 = bootstrap_ci(high_var, n_boot=500, seed=1)
    assert (hi1 - lo1) < (hi2 - lo2)


def test_bootstrap_paired_difference_recovers_gap():
    rng = np.random.default_rng(0)
    a = rng.normal(0.95, 0.05, size=200)  # e.g. V2-S DA-2K correctness
    b = rng.normal(0.88, 0.05, size=200)  # e.g. V1-S DA-2K correctness
    point, lo, hi = bootstrap_paired_difference(a, b, n_boot=500, seed=1)
    assert point == pytest.approx(0.07, abs=0.02)
    assert lo < point < hi


# ---------------------------------------------------------------------------
# decision_rule.py
# ---------------------------------------------------------------------------

def test_decision_rule_strong_on_paper_reference_values():
    inputs = DecisionInputs(
        delta_da2k_pp=6.8, delta_abs_rel=-0.003, delta_delta1=0.003,
    )
    verdict, _ = apply_decision_rule(inputs)
    assert verdict == Verdict.STRONG


def test_decision_rule_partial_when_one_threshold_missed():
    # DA-2K gain big enough, delta1 fine, but AbsRel moved too much
    inputs = DecisionInputs(
        delta_da2k_pp=6.0, delta_abs_rel=-0.03, delta_delta1=0.005,
    )
    verdict, _ = apply_decision_rule(inputs)
    assert verdict == Verdict.PARTIAL


def test_decision_rule_not_reproduced_when_no_asymmetry():
    inputs = DecisionInputs(
        delta_da2k_pp=1.0, delta_abs_rel=-0.03, delta_delta1=0.05,
    )
    verdict, _ = apply_decision_rule(inputs)
    assert verdict == Verdict.NOT_REPRODUCED


def test_decision_rule_not_reproduced_wrong_direction():
    inputs = DecisionInputs(
        delta_da2k_pp=-2.0, delta_abs_rel=0.0, delta_delta1=0.0,
    )
    verdict, _ = apply_decision_rule(inputs)
    assert verdict == Verdict.NOT_REPRODUCED

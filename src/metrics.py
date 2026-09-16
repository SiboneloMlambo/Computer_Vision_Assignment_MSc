"""
Metric implementations for both reproduction tracks.

Track A (DIODE, conventional): AbsRel, delta1 on aligned depth.
Track B (DA-2K, benchmark-sensitivity): point-pair ordinal-depth accuracy.

Everything here is student-written per the proposal's Implementation
section -- only the model checkpoints/architecture come from the
authors' repository.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Track A: DIODE conventional metrics
# ---------------------------------------------------------------------------

def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute relative error: mean(|pred - gt| / gt) over valid pixels."""
    p, g = pred[mask], gt[mask]
    return float(np.mean(np.abs(p - g) / g))


def delta1(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of valid pixels with max(pred/gt, gt/pred) < 1.25."""
    p, g = pred[mask], gt[mask]
    ratio = np.maximum(p / g, g / p)
    return float(np.mean(ratio < 1.25))


@dataclass
class DiodeImageResult:
    image_id: str
    abs_rel: float
    delta1: float
    n_valid_px: int
    scale: float
    shift: float
    split: str = "unknown"  # "indoors" or "outdoor"


# ---------------------------------------------------------------------------
# Track B: DA-2K ordinal-depth accuracy
# ---------------------------------------------------------------------------

@dataclass
class DA2KPairResult:
    image_id: str
    scene_type: str
    correct: bool
    point1: tuple[int, int]
    point2: tuple[int, int]
    closer_point: str


def da2k_pair_correct(
    pred_depth: np.ndarray,
    point1_hw: tuple[int, int],
    point2_hw: tuple[int, int],
    closer_point: str,
    higher_is_closer: bool = True,
) -> bool:
    """Score one DA-2K point-pair annotation against a predicted disparity map.

    higher_is_closer=True matches the authors' own convention: the raw
    network output is documented as disparity-like, so a larger predicted
    value means a nearer point. VERIFY THIS on a real checkpoint during the
    smoke test (src/smoke_test.py::calibrate_sign_convention) before
    trusting the full run -- log the result either way.
    """
    h1, w1 = point1_hw
    h2, w2 = point2_hw
    v1 = pred_depth[h1, w1]
    v2 = pred_depth[h2, w2]

    if higher_is_closer:
        predicted_closer = "point1" if v1 > v2 else "point2"
    else:
        predicted_closer = "point1" if v1 < v2 else "point2"

    return predicted_closer == closer_point


def da2k_accuracy(results: list[DA2KPairResult]) -> float:
    if not results:
        raise ValueError("No DA-2K results to score.")
    return float(np.mean([r.correct for r in results]))


def da2k_accuracy_by_scene(results: list[DA2KPairResult]) -> dict[str, float]:
    scenes = sorted(set(r.scene_type for r in results))
    out = {}
    for scene in scenes:
        subset = [r for r in results if r.scene_type == scene]
        out[scene] = float(np.mean([r.correct for r in subset]))
    return out


# ---------------------------------------------------------------------------
# Uncertainty: bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    statistic=np.mean,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for an arbitrary per-item statistic (e.g. per-pair
    correctness for DA-2K, or per-image AbsRel for DIODE).

    Returns (point_estimate, lower, upper).
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty array.")

    point = float(statistic(values))
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    alpha = (1 - ci) / 2
    lower = float(np.quantile(boot_stats, alpha))
    upper = float(np.quantile(boot_stats, 1 - alpha))
    return point, lower, upper


def bootstrap_paired_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for a mean difference on the same items."""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.shape != values_b.shape:
        raise ValueError("Paired arrays must have the same shape.")
    rng = np.random.default_rng(seed)

    point = float(np.mean(values_a) - np.mean(values_b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        indices = rng.integers(0, len(values_a), size=len(values_a))
        diffs[i] = np.mean(values_a[indices] - values_b[indices])

    alpha = (1 - ci) / 2
    lower = float(np.quantile(diffs, alpha))
    upper = float(np.quantile(diffs, 1 - alpha))
    return point, lower, upper

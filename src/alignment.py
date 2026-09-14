"""
Scale-and-shift alignment for relative-depth predictions.

Depth Anything V2 (the base checkpoints, as opposed to the separate
metric_depth fine-tunes) outputs a *relative* disparity map: the raw
network output is documented in the authors' own `app.py` as
"16-bit raw output (can be considered as disparity)", i.e. LARGER
predicted value = CLOSER point. It carries no metric scale or shift.

To score this against a metric benchmark like DIODE (AbsRel, delta1),
the standard protocol from the relative-depth literature (Ranftl et al.,
"Towards Robust Monocular Depth Estimation", the same alignment MiDaS
and Depth Anything's own zero-shot tables use) is:

    depth_aligned = s * disparity + t

with (s, t) fit per-image by least squares against the valid ground
truth pixels, then AbsRel/delta1 are computed on depth_aligned.

This is a *documented assumption*, not something stated verbatim in
Table 2 of the paper -- log it in run_log.md as exactly that.
"""
from __future__ import annotations

import numpy as np


def least_squares_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Solve min_{s,t} sum_{mask} (s*pred + t - gt)^2 in closed form.

    Args:
        pred: (H, W) raw network disparity output.
        gt: (H, W) ground-truth depth (metres), same shape as pred.
        mask: (H, W) boolean array, True where gt is valid.

    Returns:
        (s, t): scale and shift.
    """
    if mask.sum() < 10:
        raise ValueError(f"Too few valid pixels to fit alignment: {mask.sum()}")

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    A = np.stack([p, np.ones_like(p)], axis=1)  # (N, 2)
    # Least squares: A @ [s, t] = g
    solution, *_ = np.linalg.lstsq(A, g, rcond=None)
    s, t = solution
    return float(s), float(t)


def align_prediction(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return pred rescaled into gt's metric space via least-squares scale+shift."""
    s, t = least_squares_scale_shift(pred, gt, mask)
    aligned = s * pred + t
    return aligned

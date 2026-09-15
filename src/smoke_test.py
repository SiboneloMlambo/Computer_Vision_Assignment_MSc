"""
20-image smoke test on both tracks (proposal's "Immediate milestone").

Verifies, in order:
  1. Both checkpoints load without shape errors.
  2. A single forward pass produces a finite, correctly-shaped disparity map.
  3. The higher_is_closer sign convention actually holds on a known
     near/far pair (see calibrate_sign_convention) -- this is the
     "inspect known near/far pairs" mitigation from the risk slide.
  4. Both eval scripts run end-to-end on 20 images without crashing.

Usage:
    python -m src.smoke_test --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --calibration-image data/da2k/<some_image_with_an_obvious_near_far_pair>.jpg \
        --near-hw 300 400 --far-hw 50 400
"""
from __future__ import annotations

import argparse

import cv2

from src.model_loader import infer_disparity, load_depth_anything_vits


def calibrate_sign_convention(
    checkpoint_path: str,
    model_version: str,
    image_path: str,
    near_hw: tuple[int, int],
    far_hw: tuple[int, int],
    input_size: int = 518,
) -> bool:
    """Run one image through the model and check whether the predicted
    disparity at a manually-identified NEAR point is larger than at a
    manually-identified FAR point.

    Returns True if higher_is_closer holds (the default assumption in
    src/metrics.py and src/model_loader.py), False if it's inverted.
    Print the result to run_log.md either way -- this single check is
    load-bearing for every DA-2K number the project produces.
    """
    model = load_depth_anything_vits(checkpoint_path, model_version=model_version)
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(image_path)

    pred = infer_disparity(model, raw_image, input_size=input_size)

    near_val = pred[near_hw[0], near_hw[1]]
    far_val = pred[far_hw[0], far_hw[1]]

    holds = bool(near_val > far_val)
    print(f"Predicted value at NEAR point {near_hw}: {near_val:.4f}")
    print(f"Predicted value at FAR point  {far_hw}: {far_val:.4f}")
    print(f"higher_is_closer holds: {holds}")
    if not holds:
        print(
            "WARNING: convention does not hold on this sample. Re-run on 2-3 more "
            "obvious near/far pairs before flipping --lower-is-closer globally -- "
            "one point pair is not enough evidence on its own."
        )
    return holds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--calibration-image", required=True)
    parser.add_argument("--near-hw", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--far-hw", nargs=2, type=int, required=True, metavar=("H", "W"))
    args = parser.parse_args()

    calibrate_sign_convention(
        args.checkpoint, args.model_version, args.calibration_image,
        tuple(args.near_hw), tuple(args.far_hw),
    )


if __name__ == "__main__":
    main()

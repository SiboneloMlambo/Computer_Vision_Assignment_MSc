"""
Track A: zero-shot DIODE evaluation for V1-S and V2-S.

Usage:
    python -m src.run_track_a_diode \
        --diode-root data/diode/val \
        --checkpoint-v1 checkpoints/depth_anything_v1_vits.pth \
        --checkpoint-v2 checkpoints/depth_anything_v2_vits.pth \
        --outdir results/diode \
        --smoke-test        # optional: only run 20 images
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.alignment import align_prediction, least_squares_scale_shift
from src.diode_dataset import find_diode_triples, load_depth_and_mask, sample_smoke_test_subset
from src.metrics import DiodeImageResult, abs_rel, delta1, bootstrap_ci
from src.model_loader import infer_disparity, load_depth_anything_vits


def evaluate_checkpoint(model, frames, input_size: int = 518) -> list[DiodeImageResult]:
    results = []
    for frame in tqdm(frames, desc="DIODE eval"):
        raw_image = cv2.imread(str(frame.image_path))
        if raw_image is None:
            print(f"[run_track_a] WARNING: could not read {frame.image_path}, skipping")
            continue

        gt, mask = load_depth_and_mask(frame)
        if mask.sum() < 10:
            continue

        pred = infer_disparity(model, raw_image, input_size=input_size)
        if pred.shape != gt.shape:
            # Ground truth and prediction should already match resolution
            # (infer_image upsamples back to input resolution); if they
            # don't, resize the prediction and log it -- do not silently
            # crop.
            print(
                f"[run_track_a] WARNING: shape mismatch pred={pred.shape} "
                f"gt={gt.shape} for {frame.image_path.name}, resizing pred"
            )
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        try:
            s, t = least_squares_scale_shift(pred, gt, mask)
            aligned = s * pred + t
            # Guard against non-physical aligned depths (<= 0) before division.
            valid_after_align = mask & (aligned > 1e-6) & (gt > 1e-6)
            if valid_after_align.sum() < 10:
                continue
            a_rel = abs_rel(aligned, gt, valid_after_align)
            d1 = delta1(aligned, gt, valid_after_align)
        except ValueError as e:
            print(f"[run_track_a] WARNING: {frame.image_path.name}: {e}")
            continue

        results.append(
            DiodeImageResult(
                image_id=frame.image_path.stem,
                abs_rel=a_rel,
                delta1=d1,
                n_valid_px=int(valid_after_align.sum()),
                scale=s,
                shift=t,
                split=frame.split,
            )
        )
    return results


def summarize(results: list[DiodeImageResult]) -> dict:
    abs_rels = np.array([r.abs_rel for r in results])
    delta1s = np.array([r.delta1 for r in results])
    ar_point, ar_lo, ar_hi = bootstrap_ci(abs_rels)
    d1_point, d1_lo, d1_hi = bootstrap_ci(delta1s)
    return {
        "n_images": len(results),
        "abs_rel_mean": ar_point,
        "abs_rel_ci95": [ar_lo, ar_hi],
        "delta1_mean": d1_point,
        "delta1_ci95": [d1_lo, d1_hi],
    }


def summarize_by_split(results: list[DiodeImageResult]) -> dict:
    """Indoor/outdoor/combined view of the same per-image results.

    DIODE's val set is split into "indoors" and "outdoor" scenes; the
    combined AbsRel/delta1 can mask a large gap between the two, so report
    each split alongside the pooled ("combined") figure already produced
    by `summarize`.
    """
    by_split = {"combined": summarize(results)}
    for split in sorted(set(r.split for r in results)):
        subset = [r for r in results if r.split == split]
        by_split[split] = summarize(subset)
    return by_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diode-root", required=True)
    parser.add_argument("--checkpoint-v1", required=True)
    parser.add_argument("--checkpoint-v2", required=True)
    parser.add_argument("--outdir", default="results/diode")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--smoke-test", action="store_true", help="only run on 20 images")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = find_diode_triples(args.diode_root)
    print(f"Found {len(frames)} DIODE val frames")
    if args.smoke_test:
        frames = sample_smoke_test_subset(frames, n_images=20)
        print(f"Smoke test: using {len(frames)} frames")

    all_results = {}
    for version, model_version, ckpt in [
        ("v1s", "v1", args.checkpoint_v1),
        ("v2s", "v2", args.checkpoint_v2),
    ]:
        print(f"\n=== Evaluating {version} ({ckpt}) ===")
        model = load_depth_anything_vits(ckpt, model_version=model_version)
        results = evaluate_checkpoint(model, frames, input_size=args.input_size)
        all_results[version] = results

        per_image_path = outdir / f"diode_per_image_{version}.json"
        with open(per_image_path, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)

        summary = summarize(results)
        print(json.dumps(summary, indent=2))
        with open(outdir / f"diode_summary_{version}.json", "w") as f:
            json.dump(summary, f, indent=2)

        by_split = summarize_by_split(results)
        print(f"\n--- {version} by split (indoor / outdoor / combined) ---")
        print(json.dumps(by_split, indent=2))
        with open(outdir / f"diode_summary_by_split_{version}.json", "w") as f:
            json.dump(by_split, f, indent=2)

    v1_summary = summarize(all_results["v1s"])
    v2_summary = summarize(all_results["v2s"])
    v1_by_split = summarize_by_split(all_results["v1s"])
    v2_by_split = summarize_by_split(all_results["v2s"])
    by_split_comparison = {
        split: {
            "delta_abs_rel": v2_by_split[split]["abs_rel_mean"] - v1_by_split[split]["abs_rel_mean"],
            "delta_delta1": v2_by_split[split]["delta1_mean"] - v1_by_split[split]["delta1_mean"],
            "v1s": v1_by_split[split],
            "v2s": v2_by_split[split],
        }
        for split in v1_by_split
    }
    comparison = {
        "delta_abs_rel": v2_summary["abs_rel_mean"] - v1_summary["abs_rel_mean"],
        "delta_delta1": v2_summary["delta1_mean"] - v1_summary["delta1_mean"],
        "v1s": v1_summary,
        "v2s": v2_summary,
        "by_split": by_split_comparison,
    }
    with open(outdir / "diode_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\n=== DIODE comparison ===")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

"""
Track B: DA-2K ordinal-depth accuracy for V1-S and V2-S.

Usage:
    python -m src.run_track_b_da2k \
        --da2k-root data/da2k \
        --checkpoint-v1 checkpoints/depth_anything_v1_vits.pth \
        --checkpoint-v2 checkpoints/depth_anything_v2_vits.pth \
        --outdir results/da2k \
        --smoke-test
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.da2k_dataset import load_da2k_annotations, sample_smoke_test_subset
from src.metrics import DA2KPairResult, da2k_accuracy, da2k_accuracy_by_scene, da2k_pair_correct, bootstrap_ci
from src.model_loader import infer_disparity, load_depth_anything_vits


def evaluate_checkpoint(model, annotations, da2k_root: Path, input_size: int = 518,
                         higher_is_closer: bool = True) -> list[DA2KPairResult]:
    results = []
    # Group by image so each image is only loaded/inferred once.
    by_image = defaultdict(list)
    for ann in annotations:
        by_image[ann.image_path].append(ann)

    for image_path, anns in tqdm(by_image.items(), desc="DA-2K eval"):
        full_path = da2k_root / image_path
        raw_image = cv2.imread(str(full_path))
        if raw_image is None:
            print(f"[run_track_b] WARNING: could not read {full_path}, skipping {len(anns)} pairs")
            continue

        pred = infer_disparity(model, raw_image, input_size=input_size)

        for ann in anns:
            correct = da2k_pair_correct(
                pred, ann.point1, ann.point2, ann.closer_point,
                higher_is_closer=higher_is_closer,
            )
            results.append(
                DA2KPairResult(
                    image_id=image_path,
                    scene_type=ann.scene_type,
                    correct=correct,
                    point1=ann.point1,
                    point2=ann.point2,
                    closer_point=ann.closer_point,
                )
            )
    return results


def summarize(results: list[DA2KPairResult]) -> dict:
    correctness = np.array([r.correct for r in results], dtype=float)
    point, lo, hi = bootstrap_ci(correctness)
    return {
        "n_pairs": len(results),
        "accuracy": point,
        "accuracy_ci95": [lo, hi],
        "accuracy_by_scene": da2k_accuracy_by_scene(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--da2k-root", required=True, help="directory containing annotations.json and images")
    parser.add_argument("--checkpoint-v1", required=True)
    parser.add_argument("--checkpoint-v2", required=True)
    parser.add_argument("--outdir", default="results/da2k")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--smoke-test", action="store_true", help="only run on 20 images")
    parser.add_argument(
        "--lower-is-closer", action="store_true",
        help="flip the sign convention if the smoke-test calibration (src/smoke_test.py) "
             "shows the default higher_is_closer=True is wrong for your checkpoints",
    )
    args = parser.parse_args()

    da2k_root = Path(args.da2k_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    annotations = load_da2k_annotations(da2k_root / "annotations.json")
    print(f"Loaded {len(annotations)} DA-2K pairs across "
          f"{len(set(a.image_path for a in annotations))} images")

    if args.smoke_test:
        annotations = sample_smoke_test_subset(annotations, n_images=20)
        print(f"Smoke test: using {len(annotations)} pairs")

    higher_is_closer = not args.lower_is_closer

    all_results = {}
    for version, model_version, ckpt in [
        ("v1s", "v1", args.checkpoint_v1),
        ("v2s", "v2", args.checkpoint_v2),
    ]:
        print(f"\n=== Evaluating {version} ({ckpt}) ===")
        model = load_depth_anything_vits(ckpt, model_version=model_version)
        results = evaluate_checkpoint(
            model, annotations, da2k_root, input_size=args.input_size,
            higher_is_closer=higher_is_closer,
        )
        all_results[version] = results

        per_pair_path = outdir / f"da2k_per_pair_{version}.json"
        with open(per_pair_path, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)

        summary = summarize(results)
        print(json.dumps(summary, indent=2))
        with open(outdir / f"da2k_summary_{version}.json", "w") as f:
            json.dump(summary, f, indent=2)

    v1_acc = np.array([r.correct for r in all_results["v1s"]], dtype=float)
    v2_acc = np.array([r.correct for r in all_results["v2s"]], dtype=float)
    delta_pp = 100 * (v2_acc.mean() - v1_acc.mean())

    comparison = {
        "v1s_accuracy": float(v1_acc.mean()),
        "v2s_accuracy": float(v2_acc.mean()),
        "delta_da2k_pp": float(delta_pp),
    }
    with open(outdir / "da2k_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\n=== DA-2K comparison ===")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

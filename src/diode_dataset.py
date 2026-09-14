"""
Loader for the DIODE validation set (zero-shot Track A).

DIODE's released layout (val/{indoors,outdoor}/scene_*/scan_*/) ships,
per frame, a matched triple:

    <id>.png              RGB image
    <id>_depth.npy         float32 depth map, metres, shape (H, W, 1)
    <id>_depth_mask.npy    0/1 validity mask, shape (H, W)

This loader walks the val directory recursively and pairs files by
their shared <id> stem. If your download has a different layout, fix
`find_diode_triples` and note the deviation in run_log.md -- this is
exactly the kind of preprocessing detail the proposal flags as worth
logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DiodeFrame:
    image_path: Path
    depth_path: Path
    mask_path: Path
    split: str  # "indoors" or "outdoor"


def find_diode_triples(diode_val_root: str | Path) -> list[DiodeFrame]:
    root = Path(diode_val_root)
    frames = []
    for depth_path in sorted(root.rglob("*_depth.npy")):
        stem = depth_path.name[: -len("_depth.npy")]
        image_path = depth_path.parent / f"{stem}.png"
        mask_path = depth_path.parent / f"{stem}_depth_mask.npy"
        if not image_path.exists() or not mask_path.exists():
            print(f"[diode_dataset] WARNING: skipping {stem}, missing image or mask")
            continue
        split = "indoors" if "indoors" in depth_path.parts else "outdoor"
        frames.append(DiodeFrame(image_path, depth_path, mask_path, split))
    return frames


def load_depth_and_mask(frame: DiodeFrame) -> tuple[np.ndarray, np.ndarray]:
    depth = np.load(frame.depth_path).squeeze()  # (H, W)
    mask = np.load(frame.mask_path).squeeze().astype(bool)  # (H, W)
    return depth, mask


def sample_smoke_test_subset(frames: list[DiodeFrame], n_images: int = 20, seed: int = 0) -> list[DiodeFrame]:
    import random

    rng = random.Random(seed)
    return rng.sample(frames, min(n_images, len(frames)))

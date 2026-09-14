"""
Loader for the DA-2K benchmark's annotations.json.

Format (from the official DA-2K.md, copied here for reference -- do not
redistribute the actual images/annotations in this repository; see
data/README.md for the licence/provenance check the proposal commits to
doing before download):

{
  "image_path": [
    {"point1": [h1, w1], "point2": [h2, w2], "closer_point": "point1"},
    ...
  ],
  ...
}

Scene type is not encoded in the JSON itself; the official benchmark
ships images under per-scene-type subfolders, so we infer scene_type
from the image path's parent directory. If your downloaded copy is
structured differently, fix `infer_scene_type` and note it in run_log.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DA2KAnnotation:
    image_path: str
    scene_type: str
    point1: tuple[int, int]
    point2: tuple[int, int]
    closer_point: str


def infer_scene_type(image_path: str) -> str:
    """DA-2K's eight scene types: indoor, outdoor, non_real,
    transparent_reflective, adverse_style, aerial, underwater, object.
    Assumes the released layout of <scene_type>/<image>.* -- verify
    against your actual download and adjust if needed."""
    parts = Path(image_path).parts
    known = {
        "indoor", "outdoor", "non_real", "transparent_reflective",
        "adverse_style", "aerial", "underwater", "object",
    }
    for p in parts:
        if p in known:
            return p
    return "unknown"


def load_da2k_annotations(annotations_json_path: str | Path) -> list[DA2KAnnotation]:
    with open(annotations_json_path) as f:
        raw = json.load(f)

    out = []
    for image_path, pairs in raw.items():
        scene_type = infer_scene_type(image_path)
        for pair in pairs:
            out.append(
                DA2KAnnotation(
                    image_path=image_path,
                    scene_type=scene_type,
                    point1=tuple(pair["point1"]),
                    point2=tuple(pair["point2"]),
                    closer_point=pair["closer_point"],
                )
            )
    return out


def sample_smoke_test_subset(
    annotations: list[DA2KAnnotation], n_images: int = 20, seed: int = 0
) -> list[DA2KAnnotation]:
    """Pick all pairs belonging to a random subset of n_images distinct
    images, for the 20-image smoke test described in the proposal."""
    import random

    rng = random.Random(seed)
    image_paths = sorted(set(a.image_path for a in annotations))
    chosen = set(rng.sample(image_paths, min(n_images, len(image_paths))))
    return [a for a in annotations if a.image_path in chosen]

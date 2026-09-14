"""
Loads the authors' released V1-S / V2-S checkpoints using the authors'
own model definition (cloned from github.com/DepthAnything/Depth-Anything-V2).

This file is the one place the "checkpoints and model definitions come
from the authors' official repositories" clause of the proposal lives --
everything downstream (data loading, metrics, comparison, plots) is
student-written per src/metrics.py, src/alignment.py, run_track_*.py.

Setup expected before this module is imported (see README.md / environment_colab.md):

    git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
    pip install -r external/Depth-Anything-V2/requirements.txt

    checkpoints/depth_anything_v2_vits.pth       <- V2-S, from the authors' releases
    checkpoints/depth_anything_v1_vits.pth       <- V1-S, from the authors' V1 repo/releases
                                                     (V1 uses the same vits DPT head shape;
                                                     confirm the state_dict keys match when
                                                     you load it -- log any renaming needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPO = REPO_ROOT / "external" / "Depth-Anything-V2"

if str(EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_REPO))

# ViT-S config, copied verbatim from the official run.py's model_configs['vits'].
VITS_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_depth_anything_vits(checkpoint_path: str | Path, device: str | None = None):
    """Load a ViT-S DepthAnythingV2 model (works for both V1-S and V2-S
    checkpoints, since V1 and V2 share the same ViT-S DPT architecture --
    only the trained weights and training data differ, which is exactly
    what this reproduction is testing).
    """
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
    except ImportError as e:
        raise ImportError(
            "Could not import depth_anything_v2. Did you clone the official repo into "
            f"{EXTERNAL_REPO}? See README.md's setup section."
        ) from e

    device = device or get_device()
    model = DepthAnythingV2(**VITS_CONFIG)

    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[model_loader] WARNING loading {checkpoint_path}: "
            f"{len(missing)} missing keys, {len(unexpected)} unexpected keys. "
            "Log this in run_log.md -- a V1 checkpoint may use different key names."
        )

    model = model.to(device).eval()
    return model


def infer_disparity(model, raw_image_bgr, input_size: int = 518):
    """Run inference, returning the raw disparity map (H, W), float32,
    at the ORIGINAL image resolution (the model internally resizes to
    `input_size` on the long/short side per the paper's 518px convention,
    then upsamples the prediction back -- see infer_image in the official
    dpt.py). Larger value = closer point (see alignment.py docstring for
    the source of this convention).
    """
    depth = model.infer_image(raw_image_bgr, input_size)
    return depth

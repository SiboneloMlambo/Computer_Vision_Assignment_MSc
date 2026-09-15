"""
Loads the authors' released V1-S / V2-S checkpoints using each release's
own model definition. V1 and V2 are deliberately not loaded through the
same class: a permissive cross-version load can run with missing weights
and silently invalidate the comparison.

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

import cv2
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPO_V1 = REPO_ROOT / "external" / "Depth-Anything"
EXTERNAL_REPO_V2 = REPO_ROOT / "external" / "Depth-Anything-V2"

# ViT-S config, copied verbatim from the official run.py's model_configs['vits'].
VITS_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_depth_anything_vits(
    checkpoint_path: str | Path,
    model_version: str,
    device: str | None = None,
):
    """Load V1-S or V2-S with the matching official implementation.

    Loading is strict. Any missing or unexpected key stops the run instead
    of producing plausible-looking output from partially initialized weights.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if model_version == "v1":
        if str(EXTERNAL_REPO_V1) not in sys.path:
            sys.path.insert(0, str(EXTERNAL_REPO_V1))
        try:
            from depth_anything.dpt import DepthAnything
        except ImportError as e:
            raise ImportError(
                "Could not import Depth Anything V1. Clone the official V1 repo into "
                f"{EXTERNAL_REPO_V1}."
            ) from e
        model = DepthAnything(VITS_CONFIG)
    elif model_version == "v2":
        if str(EXTERNAL_REPO_V2) not in sys.path:
            sys.path.insert(0, str(EXTERNAL_REPO_V2))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            raise ImportError(
                "Could not import Depth Anything V2. Clone the official V2 repo into "
                f"{EXTERNAL_REPO_V2}."
            ) from e
        model = DepthAnythingV2(**VITS_CONFIG)
    else:
        raise ValueError("model_version must be 'v1' or 'v2'")

    device = device or get_device()
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device).eval()
    model._reproduction_version = model_version
    model._reproduction_device = device
    return model


def infer_disparity(model, raw_image_bgr, input_size: int = 518):
    """Run inference, returning the raw disparity map (H, W), float32,
    at the ORIGINAL image resolution (the model internally resizes to
    `input_size` on the long/short side per the paper's 518px convention,
    then upsamples the prediction back -- see infer_image in the official
    dpt.py). Larger value = closer point (see alignment.py docstring for
    the source of this convention).
    """
    if getattr(model, "_reproduction_version", None) == "v2":
        return model.infer_image(raw_image_bgr, input_size)

    if getattr(model, "_reproduction_version", None) != "v1":
        raise ValueError("Model was not loaded by load_depth_anything_vits().")

    # V1 has no infer_image helper, so reproduce its official run.py
    # preprocessing and resize the raw network output back to image size.
    from torchvision.transforms import Compose
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

    image = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB) / 255.0
    height, width = image.shape[:2]
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    image = transform({"image": image})["image"]
    tensor = torch.from_numpy(image).unsqueeze(0).to(model._reproduction_device)
    with torch.no_grad():
        depth = model(tensor)
    depth = F.interpolate(
        depth[None], (height, width), mode="bilinear", align_corners=False
    )[0, 0]
    return depth.detach().cpu().numpy()

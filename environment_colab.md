# Colab environment setup

This project is inference-only on two ViT-S checkpoints, so a free T4
Colab session is enough (matches the proposal's feasibility claim).

```bash
# 1. Clone this project repository (student code)
git clone https://github.com/SiboneloMlambo/Computer_Vision_Assignment_MSc.git depth-anything-v2-reproduction
cd depth-anything-v2-reproduction

# 2. Clone both official repos. Each checkpoint is loaded with its own
#    release's architecture; strict loading prevents partial-weight runs.
git clone https://github.com/LiheYoung/Depth-Anything.git external/Depth-Anything
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
pip install -r external/Depth-Anything/requirements.txt
pip install -r external/Depth-Anything-V2/requirements.txt

# 3. Install this project's own (student-written evaluation code) dependencies
pip install -r requirements.txt

# 4. Download checkpoints into checkpoints/
#    - V2-S: from the official Depth Anything V2 release
#      (https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models)
#      -> checkpoints/depth_anything_v2_vits.pth
#    - V1-S: from the Depth Anything V1 release
#      -> checkpoints/depth_anything_v1_vits.pth
#    Record the exact URLs and download date in run_log.md.

# 5. Download data into data/ -- see data/README.md FIRST, before downloading
#    DA-2K: licence/provenance must be checked before this step.
```

## Verify the install (no GPU, no checkpoints, no data needed)

```bash
pytest tests/ -v
```

All 19 tests should pass. This only checks the student-written metrics,
alignment, and decision-rule code -- it does not touch the model or
real data, and passing it is not evidence the reproduction itself
worked. Run it first anyway: if these fail, nothing downstream can be
trusted.

## Smoke test (needs one real checkpoint + one real image)

```bash
python -m src.smoke_test \
    --checkpoint checkpoints/depth_anything_v2_vits.pth \
    --model-version v2 \
    --calibration-image data/da2k/<pick an image with an obvious near/far pair> \
    --near-hw <h> <w> --far-hw <h> <w>
```

Confirms the `higher_is_closer` sign convention (see `src/alignment.py`
docstring) before it gets baked into every DA-2K number.

## Full runs

```bash
python -m src.run_track_a_diode \
    --diode-root data/diode/val \
    --checkpoint-v1 checkpoints/depth_anything_v1_vits.pth \
    --checkpoint-v2 checkpoints/depth_anything_v2_vits.pth \
    --smoke-test   # drop this flag once the 20-image check looks right

python -m src.run_track_b_da2k \
    --da2k-root data/da2k \
    --checkpoint-v1 checkpoints/depth_anything_v1_vits.pth \
    --checkpoint-v2 checkpoints/depth_anything_v2_vits.pth \
    --smoke-test   # drop this flag once the 20-image check looks right

python -m src.apply_decision --results-dir results
```

# Run log

Per the course's "the interesting content of a reproduction is the
gap" note -- record every guessed preprocessing detail, every failed
run, and every deviation from the paper here, as you go, not
retroactively at the end.

## Checkpoint provenance

| Checkpoint | Source URL | Date downloaded | SHA256 (optional) |
|---|---|---|---|
| depth_anything_v2_vits.pth | https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth | | |
| depth_anything_v1_vits.pth | https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth | | |

(URLs filled in before the Colab run; fill in date + SHA256 once actually downloaded there.)

## Data licence/provenance check (DA-2K)

- Date checked: 2026-09-14
- Licence found: Apache-2.0, per the dataset card at
  https://huggingface.co/datasets/depth-anything/DA-2K. Permits
  commercial/private use and modification; requires attribution --
  cite "Depth Anything V2" (Yang et al., arXiv:2406.09414, 2024).
- Access method: images are hosted directly on the Hugging Face
  dataset repo (not links-to-originals) -- 1,035 images, ~682MB total,
  plus an annotations.json.
- Decision: proceed.
- NOTE: the licence/size figures above came from an automated page
  fetch, not a manual read of the HF dataset card -- confirm directly
  on the page before treating this as final, then delete this note.

## Sign-convention calibration (smoke_test.calibrate_sign_convention)

- Calibration image used:
- Near point (h, w):
- Far point (h, w):
- Predicted values:
- `higher_is_closer` holds?:

## Preprocessing decisions not stated explicitly in the paper

- Scale-and-shift alignment for DIODE (see `src/alignment.py` docstring):
  least-squares per-image, following the standard relative-depth
  evaluation protocol (Ranftl et al.) -- the paper's Table 2 does not
  state its own alignment method, so this is an assumption. Note here
  once you've run it whether results look sane against the reference
  0.073/0.076 AbsRel scale, since a badly wrong alignment protocol
  would produce absurd AbsRel values, not just slightly-off ones.

## 20-image smoke test results

- DIODE Track A: pass / fail, notes:
- DA-2K Track B: pass / fail, notes:

## Full run

- Runtime:
- Any Colab session interruptions / checkpointing needed:
- Final verdict (from `src/apply_decision.py`):

## What didn't reproduce (fill in last, mostly this)

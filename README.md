# Reproducing Depth Anything V2's benchmark-sensitivity claim

COMS7050A reproduction project. Sibonelo Mlambo.

## The claim being tested

At matched ViT-S encoder size, V2 shows a clearly larger improvement
over V1 on DA-2K's ordinal-depth accuracy (Table 3) than on DIODE's
conventional zero-shot AbsRel/delta1 (Table 2). Full statement,
hypotheses, and pre-registered decision rule: see the proposal PDF and
`src/decision_rule.py`.

- H0: V2's gain over V1 is not meaningfully larger on DA-2K than on DIODE.
- H1: V2 has a large positive DA-2K accuracy gain while DIODE AbsRel/delta1
  stay close or show only a small/mixed change.

Published values (88.5% -> 95.3% on DA-2K; 0.076 -> 0.073 AbsRel and
0.939 -> 0.942 delta1 on DIODE) are reference targets only and are
never reported here as reproduced results -- see the self-check at the
bottom of `src/decision_rule.py`, which is labelled as exactly that.

## What comes from the authors vs. what's student-written

- **From the authors** (`LiheYoung/Depth-Anything` and
  `DepthAnything/Depth-Anything-V2`): each release's matching model
  architecture, inference code (`infer_image`), and released checkpoint.
- **Student-written** (this repository): data loading (`src/da2k_dataset.py`,
  `src/diode_dataset.py`), the scale-and-shift alignment protocol
  (`src/alignment.py`), all metric implementations (`src/metrics.py`),
  the pre-registered decision rule (`src/decision_rule.py`), and all
  comparison/plotting code.

## Repository layout

```
src/
  alignment.py          scale-and-shift alignment for relative depth -> metric DIODE
  metrics.py             AbsRel, delta1, DA-2K ordinal accuracy, bootstrap CIs
  model_loader.py         wraps the authors' DepthAnythingV2 ViT-S architecture
  da2k_dataset.py         DA-2K annotations.json loader
  diode_dataset.py        DIODE val set loader
  run_track_a_diode.py    Track A: conventional zero-shot DIODE eval, end-to-end
  run_track_b_da2k.py     Track B: DA-2K ordinal accuracy, end-to-end
  smoke_test.py           sign-convention calibration + 20-image checks
  decision_rule.py        pre-registered strong/partial/not-reproduced thresholds
  apply_decision.py       combines both tracks' outputs and applies the rule
tests/
  test_metrics.py         19 unit tests, synthetic data, no GPU/checkpoints needed
notebooks/
  colab_reproduction.ipynb   end-to-end Colab notebook
data/README.md            licence/provenance checklist, expected layouts
run_log.md                 template for logging every guess and failure
environment_colab.md       exact setup commands
```

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v          # 19/19 should pass -- see environment_colab.md
```

Then follow `environment_colab.md` for the full Colab setup (cloning
the two official repos, downloading checkpoints/data, running both tracks,
and applying the decision rule).

## Status

- [x] Student-written evaluation code (metrics, alignment, decision
      rule) implemented and unit-tested (19/19 passing, synthetic data).
- [ ] Checkpoints verified loadable (needs Colab + real checkpoint files).
- [ ] DA-2K licence/provenance check completed (`data/README.md`).
- [ ] Sign-convention calibrated on a real checkpoint (`src/smoke_test.py`).
- [ ] 20-image smoke test, both tracks.
- [ ] Full DIODE + DA-2K runs.
- [ ] Decision rule applied to real results.

This matches the proposal's Week 9 milestone: repo created, checkpoints
verified, 20-image smoke tests on both tracks, DA-2K access/licence
recorded, preliminary output table produced.

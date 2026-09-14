# Data

Nothing in this repository redistributes DA-2K or DIODE data or images.
Download both yourself after the checks below.

## DA-2K

Source: https://huggingface.co/datasets/depth-anything/DA-2K

Before downloading:
- [ ] Check the dataset's licence on the Hugging Face page.
- [ ] Check whether images are hosted directly or link out to original
      sources (the proposal flags this as a real risk -- DA-2K's images
      are drawn from varied sources and some benchmarks like this one
      distribute only links, not pixels).
- [ ] Record what you find here, with the date checked, before running
      anything further. If the images are not lawfully accessible to
      you, stop and say so in run_log.md rather than substituting a
      different dataset silently.

Expected layout after download:
```
data/da2k/
├── annotations.json
├── indoor/...
├── outdoor/...
├── non_real/...
├── transparent_reflective/...
├── adverse_style/...
├── aerial/...
├── underwater/...
└── object/...
```
(`src/da2k_dataset.py::infer_scene_type` assumes this; fix it if your
download differs and note the fix in run_log.md.)

## DIODE

Source: https://diode-dataset.org/ (validation split only is needed)

Expected layout after download:
```
data/diode/val/
├── indoors/scene_*/scan_*/*.png, *_depth.npy, *_depth_mask.npy
└── outdoor/scene_*/scan_*/*.png, *_depth.npy, *_depth_mask.npy
```
(`src/diode_dataset.py::find_diode_triples` walks this recursively by
filename pattern, so it should tolerate small layout differences --
but check the frame count it finds against DIODE's published val size
before trusting it.)

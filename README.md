# 🧠 NeuroMontage
Create publication-ready **mosaic images** *or* **animated GIFs** of brain slices with colored lesion overlays or outlines – all from the command line.

---

## At a Glance
| Mode | What it does | Typical use-case |
|------|--------------|------------------|
| **Overlay** (default) | Renders a *single* cumulative segmentation volume as a semi-transparent colormap on top of T1/T2 anatomy. | Visualise lesion load, probabilistic maps, heat-maps, etc. |
| **Outline** (`--outline` **or** ≥2 masks) | Draws coloured contours for one or more binary lesion masks (each above its own threshold). | Compare multiple lesion masks, multi-time-point studies, algorithm evaluation. |

Additional niceties:

* Optional **slice highlighting** (`--highlight`) – adds a red border around slices containing any lesion voxels.
* **Flexible slice selection** – choose start/end indices and subsampling step.
* **Customisable layout** – set number of slices per row.
* **Animated GIFs** – auto-resized to 800 px height to keep file sizes reasonable.
* **L/R orientation labels** – stamped automatically in every output.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/neuromontage.git
cd neuromontage

# 2. (Recommended) create a fresh environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .\.venv\Scripts\activate  # Windows PowerShell

# 3. Install requirements
pip install -r requirements.txt
````

<details>
<summary><strong>Minimal dependencies</strong></summary>

* Python ≥ 3.8
* [`numpy`](https://numpy.org/)
* [`nibabel`](https://nipy.org/nibabel/)
* [`matplotlib`](https://matplotlib.org/)
* [`Pillow`](https://python-pillow.org/)
* [`scikit-image`](https://scikit-image.org/)

</details>

> **Tip:** All dependencies are pure-Python, so installation works on any OS where Python and a C compiler are available.

---

## Quick Start

### 1. Overlay Mode (one input file)

```bash
python neuromontage.py \
  -b sub-01_T1w.nii.gz \
  -s sub-01_lesion_prob.nii.gz \
  -o sub-01_mosaic.jpg \
  --alpha 0.6 \
  --colormap magma \
  --log_scale \
  --slice_step 2 \
  --slices_per_row 8
```

### 2. Outline Mode (multiple masks)

```bash
python neuromontage.py \
  -b sub-01_T1w.nii.gz \
  -s acute.nii.gz chronic.nii.gz edematous.nii.gz \
  -o outlines.gif \
  --gif --duration 6 \
  --threshold 0.3 0.5 0.1 \
  --highlight
```

See **more worked examples** in the [Usage Examples](#usage-examples) section.

---

## Command-line Reference

```text
usage: neuromontage.py [-h] -b BRAIN -s SEG ... -o OUTPUT [--gif]
                       [--duration SEC] [--outline] [--highlight]
                       [--slice_step N] [--start_slice I] [--end_slice I]
                       [--alpha VAL] [--colormap NAME] [--log_scale]
                       [--threshold T [T ...]] [--slices_per_row N]
```

| Flag                              | Meaning                                     | Default      |
| --------------------------------- | ------------------------------------------- | ------------ |
| `-b`, `--brain_file`              | Structural brain NIfTI (T1/T2/FLAIR)        | *required*   |
| `-s`, `--input_files`             | One or more lesion/segmentation NIfTIs      | *required*   |
| `-o`, `--output_file`             | Destination `.jpg` (mosaic) **or** `.gif`   | *required*   |
| `--gif`                           | Produce an animated GIF instead of a mosaic | off          |
| `--duration SEC`                  | Total GIF runtime in seconds                | `10.0`       |
| `--outline`                       | Force outline mode (even with one input)    | off          |
| `--highlight`                     | Red border around slices containing lesions | off          |
| `--slice_step N`                  | Use every *N*th valid slice                 | `2`          |
| `--start_slice I / --end_slice I` | Slice range (inclusive)                     | whole volume |
| `--alpha VAL`                     | Overlay opacity (overlay mode)              | `0.7`        |
| `--colormap NAME`                 | Any Matplotlib colormap                     | `viridis`    |
| `--log_scale`                     | Log-scale colour mapping (overlay mode)     | off          |
| `--threshold T [T ...]`           | Single or per-mask thresholds               | `0.0`        |
| `--slices_per_row N`              | Layout width in mosaic                      | `8`          |

---

## Usage Examples

<details>
<summary>Click to expand full examples</summary>

```bash
# (1) Single cumulative segmentation → linear overlay
python neuromontage.py -b brain_T1.nii.gz -s cumulative.nii.gz \
  -o overlay.jpg --start_slice 10 --end_slice 50 --slice_step 2 \
  --alpha 0.7 --slices_per_row 8

# (2) Single cumulative segmentation → log-scale overlay (thr=5)
python neuromontage.py -b brain_T1.nii.gz -s cumulative.nii.gz \
  -o overlay_log.jpg --log_scale --alpha 0.5 \
  --slices_per_row 6 --threshold 5

# (3) Two masks → identical threshold, coloured outlines
python neuromontage.py -b brain_T1.nii.gz -s lesion1.nii.gz lesion2.nii.gz \
  -o outlines.jpg --threshold 0.5 --slices_per_row 7

# (4) Two masks → different thresholds
python neuromontage.py -b brain_T1.nii.gz -s lesion1.nii.gz lesion2.nii.gz \
  -o outlines_diff.jpg --threshold 0.5 1.0

# (5) Force outline mode with one file
python neuromontage.py -b brain_T1.nii.gz -s mask.nii.gz \
  -o outline_single.jpg --outline

# (6) Mosaic with highlighted slices
python neuromontage.py -b brain_T1.nii.gz -s seg1.nii.gz seg2.nii.gz \
  -o mosaic_highlight.jpg --highlight --slices_per_row 5

# (7) Animated GIF (outline) with three masks
python neuromontage.py -b brain_T1.nii.gz -s A.nii.gz B.nii.gz C.nii.gz \
  -o multi_outlines.gif --gif --duration 5 --outline \
  --highlight --threshold 0.5 1.0 0.2

# (8) Animated GIF (overlay) with log-scale + highlight
python neuromontage.py -b brain_T1.nii.gz -s cumulative.nii.gz \
  -o overlay_animation.gif --gif --duration 8 \
  --alpha 0.6 --colormap plasma --log_scale \
  --highlight --threshold 2.5
```

</details>

---

## Output Samples

| Output type      | Example                                    |
| ---------------- | ------------------------------------------ |
| **Mosaic JPEG**  | ![mosaic example](docs/example_mosaic.jpg) |
| **Animated GIF** | ![gif example](docs/example_animation.gif) |

*(Screenshots omitted from repository to keep size small – generate them with the sample commands above.)*

---

## Tips & Best Practices

* **Large volumes** – use `--slice_step` to down-sample slices and keep mosaics manageable.
* **Colourblind-friendly overlays** – try `--colormap magma`, `plasma`, or `cividis`.
* **Log scale** is helpful for probability maps with long-tailed distributions.
* **Thresholds** accept floats; integer masks often work with `--threshold 0.5`.
* Combine **highlighting** with outlines to spot lesion-bearing slices instantly.

---

## Contributing

Pull requests are very welcome! If you:

1. Found a bug
2. Need a new feature
3. Want to improve documentation or tests

please open an issue first to discuss the change.
Make sure `pre-commit` hooks pass (`black`, `flake8`, `isort`).

---

## Citation

If this tool helped your research, please cite it:

```bibtex
@misc{neuromontage,
  author       = {YOUR NAME},
  title        = {NeuroMontage: Brain Overlay & Lesion Mosaic Utility},
  year         = {2025},
  url          = {https://github.com/YOUR_USERNAME/neuromontage},
  version      = {1.5}
}
```

---

## License

Distributed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

## Changelog

* **1.5** – Highlighting, GIF resize, multi-threshold support, robust L/R labels
* **1.4** – Log-scale overlays, coloured contour palettes
* **1.3** – Animated GIF support
* **1.2** – Slice range & subsampling, opacity control
* **1.1** – Outline mode for ≥2 masks
* **1.0** – Initial release (overlay mosaics)

---

> *May your montages be crisp and your reviewers impressed!*

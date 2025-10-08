# 🧠 NeuroMontage
Create publication-ready **mosaic images** *or* **animated GIFs** of brain slices with colored lesion overlays or outlines – all from the command line.

---

## At a Glance
| Mode | What it does | Typical use-case |
|------|--------------|------------------|
| **Brain-only** (no segmentation) | Display structural brain images in an optimal grid layout | Quick quality checks, anatomical reference images |
| **Overlay** (single segmentation file) | Renders a cumulative segmentation volume as a semi-transparent colormap on top of T1/T2 anatomy | Visualize lesion load, probabilistic maps, heat-maps |
| **Outline** (multiple masks **or** `--outline` flag) | Draws colored contours for one or more binary lesion masks | Compare multiple lesion masks, multi-time-point studies, algorithm evaluation |
| **Alternating** (`--alternate` flag) | Alternates rows between brain-only and brain+overlay views | Side-by-side comparison of anatomy with and without overlay |

Additional features:

* **Smart resolution management** – Automatically optimizes output quality without upscaling, keeping native resolution where possible
* **Adaptive sampling** – For large datasets, intelligently reduces slice count to maintain image quality and reasonable file sizes
* **4D volume support** – Creates "super mosaics" with each timepoint/volume displayed in its own row, including intensity indicator bars
* **Slice highlighting** (`--highlight`) – Adds a red border around slices containing lesion voxels
* **Flexible slice selection** – Choose start/end indices and subsampling step
* **Animated GIFs** – Auto-resized to 800px height for manageable file sizes
* **L/R orientation labels** – Automatically stamped on every output

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/mdschirmer/neuromontage.git
cd neuromontage

# 2. (Recommended) create a fresh environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .\.venv\Scripts\activate  # Windows PowerShell

# 3. Install requirements
pip install -r requirements.txt
```

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

### 1. Brain-only visualization

```bash
python neuromontage.py \
  -b sub-01_T1w.nii.gz \
  -o brain_mosaic.jpg \
  --start_slice 10 \
  --end_slice 50
```

### 2. Overlay Mode (single segmentation file)

```bash
python neuromontage.py \
  -b sub-01_T1w.nii.gz \
  -s sub-01_lesion_prob.nii.gz \
  -o overlay_mosaic.jpg \
  --alpha 0.6 \
  --colormap magma \
  --log_scale
```

### 3. Outline Mode (multiple masks)

```bash
python neuromontage.py \
  -b sub-01_T1w.nii.gz \
  -s acute.nii.gz chronic.nii.gz edematous.nii.gz \
  -o outlines.jpg \
  --threshold 0.3 0.5 0.1 \
  --highlight
```

### 4. Alternating Mode (compare with/without overlay)

```bash
python neuromontage.py \
  -b sub-01_T1w.nii.gz \
  -s lesion_mask.nii.gz \
  -o comparison.jpg \
  --alternate \
  --alpha 0.7
```

### 5. 4D Volume Super Mosaic

```bash
python neuromontage.py \
  -b diffusion_4d.nii.gz \
  -o diffusion_mosaic.jpg
```

See **more worked examples** in the [Usage Examples](#usage-examples) section.

---

## Understanding Resolution Modes

NeuroMontage offers intelligent resolution management to balance quality and file size:

### Auto Mode (Recommended)
```bash
python neuromontage.py -b brain.nii.gz -o output.jpg
# No --resolution flag needed (auto is default)
```

**What it does:**
- Maintains native slice resolution whenever possible (no upscaling)
- Automatically samples slices if needed to keep file size reasonable
- Ensures output dimensions stay within 4000×4000 pixels
- Optimizes grid layout for best use of space

**When to use:** Almost always! This mode provides the best quality-to-file-size ratio.

### Preset Resolutions (HD/2K/4K)
```bash
python neuromontage.py -b brain.nii.gz -o output.jpg --resolution 4k
```

**What it does:**
- Forces output to specific dimensions (HD: 1920×1080, 2K: 2560×1440, 4K: 3840×2160)
- May significantly downsample for large datasets
- Never upscales (maintains quality)

**When to use:** When you need consistent output dimensions across multiple datasets, or for presentation/publication with specific size requirements.

---

## Command-line Reference

```text
usage: neuromontage.py [-h] -b BRAIN [-s SEG ...] -o OUTPUT 
                       [--resolution {auto,hd,2k,4k}] [--gif] [--duration SEC]
                       [--outline] [--highlight] [--alternate]
                       [--slice_step N] [--start_slice I] [--end_slice I]
                       [--alpha VAL] [--colormap NAME] [--log_scale]
                       [--threshold T [T ...]]
```

### Required Arguments

| Flag | Description |
|------|-------------|
| `-b`, `--brain_file` | Structural brain NIfTI (T1/T2/FLAIR, 3D or 4D) |
| `-o`, `--output_file` | Destination `.jpg` (mosaic) **or** `.gif` (animation) |

### Segmentation Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--input_files` | One or more lesion/segmentation NIfTIs (optional, 3D only) | none |
| `--outline` | Force outline mode (even with one input) | off |
| `--highlight` | Red border around slices with lesions | off |
| `--alternate` | Alternate rows: brain-only vs. brain+overlay | off |

### Resolution & Layout

| Flag | Description | Default |
|------|-------------|---------|
| `--resolution` | Output size: `auto`, `hd`, `2k`, `4k` | `auto` |
| `--slice_step N` | Use every *N*th slice | `1` |
| `--start_slice I` | First slice index (inclusive) | `0` |
| `--end_slice I` | Last slice index (inclusive) | all |

### Overlay Appearance

| Flag | Description | Default |
|------|-------------|---------|
| `--alpha VAL` | Overlay opacity (0-1, overlay mode only) | `0.7` |
| `--colormap NAME` | Matplotlib colormap (overlay mode only) | `viridis` |
| `--log_scale` | Log-scale color mapping (overlay mode only) | off |
| `--threshold T [T ...]` | Single or per-mask threshold values | `0.0` |

### Animation Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gif` | Create animated GIF (3D data only) | off |
| `--duration SEC` | Total GIF runtime in seconds | `10.0` |

---

## Usage Examples

<details>
<summary>Click to expand comprehensive examples</summary>

```bash
# (1) Brain-only mosaic with auto resolution
python neuromontage.py -b brain_T1.nii.gz -o brain_mosaic.jpg

# (2) Brain-only with specific slice range
python neuromontage.py -b brain_T1.nii.gz -o brain_subset.jpg \
  --start_slice 20 --end_slice 60 --slice_step 2

# (3) Brain-only forced to 4K resolution
python neuromontage.py -b brain_T1.nii.gz -o brain_4k.jpg --resolution 4k

# (4) Single segmentation overlay with log scale
python neuromontage.py -b brain_T1.nii.gz -s cumulative.nii.gz \
  -o overlay_log.jpg --log_scale --alpha 0.6 --colormap plasma

# (5) Multiple lesion masks with colored outlines
python neuromontage.py -b brain_T1.nii.gz \
  -s lesion1.nii.gz lesion2.nii.gz lesion3.nii.gz \
  -o outlines.jpg --threshold 0.5

# (6) Different thresholds per mask
python neuromontage.py -b brain_T1.nii.gz \
  -s acute.nii.gz chronic.nii.gz \
  -o multi_threshold.jpg --threshold 0.3 0.7

# (7) Force outline mode with single file
python neuromontage.py -b brain_T1.nii.gz -s mask.nii.gz \
  -o outline_single.jpg --outline

# (8) Highlighted slices with segmentation
python neuromontage.py -b brain_T1.nii.gz -s lesions.nii.gz \
  -o highlighted.jpg --highlight --alpha 0.5

# (9) Alternating rows (brain vs. brain+overlay)
python neuromontage.py -b brain_T1.nii.gz -s lesions.nii.gz \
  -o alternating.jpg --alternate --colormap hot

# (10) 4D volume super mosaic with auto resolution
python neuromontage.py -b diffusion_4d.nii.gz -o diffusion.jpg

# (11) 4D volume with forced 2K output
python neuromontage.py -b fmri_4d.nii.gz -o fmri_2k.jpg --resolution 2k

# (12) Animated GIF with multiple outlines
python neuromontage.py -b brain_T1.nii.gz \
  -s mask1.nii.gz mask2.nii.gz mask3.nii.gz \
  -o animation.gif --gif --duration 6 --highlight

# (13) Animated GIF with overlay and log scale
python neuromontage.py -b brain_T1.nii.gz -s cumulative.nii.gz \
  -o overlay_anim.gif --gif --duration 8 \
  --alpha 0.6 --colormap magma --log_scale --highlight
```

</details>

---

## Understanding 4D Visualization

When you provide a 4D NIfTI file (e.g., diffusion, fMRI, or multi-echo data), NeuroMontage creates a "super mosaic":

**Layout:**
- Each volume/timepoint gets its own horizontal row
- All slices from that volume are arranged left-to-right
- Rows are labeled (Vol 1, Vol 2, etc.)

**Intensity Indicators:**
- Each row has a grayscale intensity bar on the right
- An orange horizontal line shows where that volume's median intensity falls
- This helps you quickly identify volumes with different contrast or signal characteristics

**Adaptive Sampling:**
- For 4D data with many slices, auto mode intelligently samples to maintain quality
- You'll see messages like "showing every 2 slice(s)" if sampling is applied

---

## Tips & Best Practices

### Resolution Selection
* **Use `auto` mode** for best results in most cases – it prevents quality loss from upscaling
* **Use preset resolutions** only when you need consistent dimensions across datasets
* For large datasets (>100 slices), auto mode may adaptively sample – this is normal and preserves quality

### Visualization Modes
* **Brain-only mode** is great for quality control and anatomical reference
* **Overlay mode** works best for continuous probability maps or intensity-based segmentations
* **Outline mode** is ideal for comparing discrete lesion masks or multi-timepoint data
* **Alternate mode** provides direct visual comparison – especially useful for presentations

### Color & Contrast
* For overlay mode, try colorblind-friendly colormaps: `magma`, `plasma`, `cividis`
* Use `--log_scale` for probability maps with long-tailed distributions
* Adjust `--alpha` (0.5-0.8) to balance visibility of anatomy vs. overlay

### File Sizes
* JPEG mosaics are automatically optimized (quality=95)
* GIFs are resized to 800px height to keep file sizes manageable
* For very large datasets, consider using `--slice_step` or letting auto mode handle sampling

### Thresholds
* Default threshold (0.0) works for most binary masks
* For probability maps, try thresholds like 0.3-0.7 depending on your confidence level
* You can specify different thresholds for each mask in outline mode

---

## Output Examples

### 3D Brain-only Mosaic
High-quality anatomical reference with automatic layout optimization.

### 3D Overlay Mosaic
Semi-transparent probability map overlaid on structural image with custom colormap.

### 3D Outline Mosaic
Multiple lesion masks shown as colored contours using colorblind-friendly palette.

### 3D Alternating Mosaic
Odd rows show brain-only, even rows show brain+overlay for direct comparison.

### 4D Super Mosaic
Each volume in its own row with intensity indicator bars showing relative signal characteristics.

### Animated GIF
Frame-by-frame visualization through all valid slices with optional highlighting.

*(Full resolution examples omitted from repository to keep size small – generate them with the sample commands above.)*

---

## Contributing

Pull requests are very welcome! If you:

1. Found a bug
2. Need a new feature
3. Want to improve documentation or tests

please open an issue first to discuss the change.

---

## Citation

If this tool helped your research, please cite it:

```bibtex
@software{neuromontage,
  author       = {Markus D. Schirmer},
  title        = {NeuroMontage: Brain Overlay & Lesion Mosaic Utility},
  year         = {2025},
  url          = {https://github.com/mdschirmer/neuromontage},
  version      = {2.2}
}
```

---

## License

Distributed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

## Changelog

* **2.2** – Alternating mode, robust normalization, brain-only mode, improved 4D intensity indicators
* **2.1** – Smart resolution management, adaptive sampling, 4D super mosaics
* **2.0** – Resolution presets (HD/2K/4K), optimal layout calculation
* **1.5** – Highlighting, GIF resize, multi-threshold support, robust L/R labels
* **1.4** – Log-scale overlays, colorblind-friendly contour palettes
* **1.3** – Animated GIF support
* **1.2** – Slice range & subsampling, opacity control
* **1.1** – Outline mode for ≥2 masks
* **1.0** – Initial release (overlay mosaics)

---
# 🧠 NeuroMontage
Create publication-ready **mosaic images** *or* **animations** (GIF/WebP) of brain slices with colored lesion overlays or outlines – all from the command line.

---

## At a Glance
| Mode | What it does | Typical use-case |
|------|--------------|------------------|
| **Brain-only** (no segmentation) | Display structural brain images in an optimal grid layout | Quick quality checks, anatomical reference images |
| **Overlay** (single segmentation file) | Renders a segmentation volume semi-transparent on top of T1/T2 anatomy: label maps (up to 7 whole-number values) in Okabe-Ito colors, other maps with a colormap | Visualize lesion load, probabilistic maps, heat-maps |
| **Outline** (multiple masks **or** `--outline` flag) | Draws colored contours for one or more binary lesion masks | Compare multiple lesion masks, multi-time-point studies, algorithm evaluation |
| **Alternating** (`--alternate` flag) | Alternates rows between brain-only and brain+overlay views | Side-by-side comparison of anatomy with and without overlay |

Additional features:

* **Smart resolution management** – Automatically optimizes output quality, keeping native resolution where possible. Slices smaller than 100px are enlarged by a whole-number factor, without interpolation
* **Adaptive sampling** – For large datasets, intelligently reduces slice count to maintain image quality and reasonable file sizes
* **4D volume support** – Creates "super mosaics" with each timepoint/volume displayed in its own row, including intensity indicator bars
* **Slice highlighting** (`--highlight`) – Adds a red border around slices containing lesion voxels
* **Flexible slice selection** – Choose start/end indices and subsampling step
* **Animations** (`--animate`, GIF or WebP) – Auto-resized to 800px height for manageable file sizes
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

Each mask gets a color and line style by its position (orange solid, sky blue dashed, green dotted, ...). To fix them per mask, for example when a mask may be missing, use `--colors` and `--line-styles` with one value per mask:

```bash
python neuromontage.py -b flair.nii.gz -s gmwm.nii.gz ventricles.nii.gz -o qc.jpg \
  --outline --colors orange green --line-styles solid dotted
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
- Maintains native slice resolution whenever possible (slices smaller than 100px are enlarged 2×, 3×, … without interpolation)
- Automatically samples slices if needed to keep file size reasonable
- Ensures 3D mosaics stay within 3840×2160 pixels (4D mosaics: 3840 pixels wide)
- Optimizes grid layout for best use of space

**When to use:** Almost always! This mode provides the best quality-to-file-size ratio.

### Preset Resolutions (HD/2K/4K)
```bash
python neuromontage.py -b brain.nii.gz -o output.jpg --resolution 4k
```

**What it does:**
- Gives exactly this output size (HD: 1920×1080, 2K: 2560×1440, 4K: 3840×2160); the mosaic is centered on a black (3D) or white (4D) frame
- May significantly downsample for large datasets
- Enlarges or reduces slices to fill the frame; enlarging uses no interpolation

**When to use:** When you need consistent output dimensions across multiple datasets, or for presentation/publication with specific size requirements.

---

## Command-line Reference

```text
usage: neuromontage.py [-h] [--version] -b BRAIN [-s SEG ...] -o OUTPUT 
                       [--resolution {auto,hd,2k,4k}] [--animate] [--duration SEC]
                       [--outline] [--highlight] [--alternate]
                       [--rows N] [--slice_step N] [--start_slice I] [--end_slice I]
                       [--alpha VAL] [--colormap NAME] [--log_scale]
                       [--threshold T [T ...]]
```

### Required Arguments

| Flag | Description |
|------|-------------|
| `-b`, `--brain_file` | Structural brain NIfTI (T1/T2/FLAIR, 3D or 4D) |
| `-o`, `--output_file` | Destination `.jpg`, `.jpeg`, `.png` or `.webp` (mosaic) **or** `.gif` / `.webp` with `--animate` (animation). `.webp` gives the smallest files |

### Segmentation Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-s`, `--input_files` | One or more lesion/segmentation NIfTIs (optional, 3D only) | none |
| `--outline` | Force outline mode (even with one input) | off |
| `--colors` | Outline mode: one color per mask, palette name (`orange`, `skyblue`, `green`, `yellow`, `blue`, `vermillion`, `pink`, `black`) or `#RRGGBB` | palette order |
| `--line-styles` | Outline mode: one line style per mask (`solid`, `dashed`, `dotted`, `long-dash`, `short-dash`, `very-long-dash`, `medium-dash`, `dense-dash`) | in that order |
| `--highlight` | Red border around slices with lesions | off |
| `--alternate` | Alternate rows: brain-only vs. brain+overlay | off |

### Resolution & Layout

| Flag | Description | Default |
|------|-------------|---------|
| `--resolution` | Output size: `auto`, `hd`, `2k`, `4k` | `auto` |
| `--rows N` | Fixed row count for 3D mosaics (columns derived; must be even with `--alternate`, odd values round down) | automatic |
| `--slice_step N` | Use every *N*th slice | `1` |
| `--start_slice I` | First slice index (inclusive) | `0` |
| `--end_slice I` | Last slice index (inclusive) | all |

### Overlay Appearance

| Flag | Description | Default |
|------|-------------|---------|
| `--alpha VAL` | Overlay opacity (0-1, overlay mode only) | `0.7` |
| `--colormap NAME` | Matplotlib colormap (overlay mode only). If set, it is also used for label maps | Okabe-Ito for label maps (up to 7 values), else `viridis` |
| `--log_scale` | Log-scale color mapping (overlay mode only) | off |
| `--threshold T [T ...]` | Single or per-mask threshold values | `0.0` |

### Animation Options

| Flag | Description | Default |
|------|-------------|---------|
| `--animate` | Create an animation (3D data only), as `.gif` or `.webp` (smaller; lossless when outlines are drawn). Optional for `.gif`. `--gif` is an alias | off |
| `--duration SEC` | Total animation runtime in seconds | `10.0` |

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

# (12) Animation with multiple outlines (WebP: smaller than GIF)
python neuromontage.py -b brain_T1.nii.gz \
  -s mask1.nii.gz mask2.nii.gz mask3.nii.gz \
  -o animation.webp --animate --duration 6 --highlight

# (13) Animated GIF with overlay and log scale
python neuromontage.py -b brain_T1.nii.gz -s cumulative.nii.gz \
  -o overlay_anim.gif --duration 8 \
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
* JPEG mosaics use quality 95 with full-resolution color, so thin outlines keep their color
* WebP mosaics (`.webp`) are about 45% smaller than JPEG at the same quality; they are lossless when outlines are drawn
* PNG mosaics are lossless
* Animations are resized to 800px height to keep file sizes manageable; animated WebP is about 40% smaller than GIF and not limited to 256 colors
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

### Animation
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
  url          = {https://github.com/mdschirmer/neuromontage}
}
```

To cite a specific version, run `python neuromontage.py --version`.

---

## License

Distributed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

## Changelog

* **2.3** – Segmentation overlays for 4D images. Line style for each mask. `--colors` and `--line-styles` options. 16:9 layouts. `--rows` option. Robust NIfTI loading. `--version` option.
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
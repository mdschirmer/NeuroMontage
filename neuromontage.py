#!/usr/bin/env python3
"""
NeuroMontage (v1.5)

:Summary:
    Create a brain‐overlay mosaic (or animated GIF) with either:
      1) A single “cumulative segmentation” NIfTI → shown as a colormap overlay, or
      2) One or more “lesion‐segmentation” NIfTIs → shown as their colored outlines (voxels > threshold) on each slice.

    • By default, if exactly one input is given, it’s overlay mode (colormap).
    • If two or more inputs are given, it’s outline mode (draw contours).
    • You can force outline mode with `--outline` even if only one file is passed.
    • Use `--highlight` to draw a red border around any slice that contains segmentation (voxels > threshold).
    • When multiple segmentation files are provided, you can supply either:
        – A single threshold (applied to all segmentations), or
        – A list of thresholds, one per segmentation file.
      If omitted, threshold defaults to 0.0 for all.

    When generating a GIF, each frame is resized to 800 pixels tall, preserving aspect ratio.

:Description:
    • Load a structural brain NIfTI (e.g. T1) via `-b/--brain_file`.
    • Load one or more lesion/segmentation NIfTIs via `-s/--input_files`.
        – If exactly 1 file is given and `--outline` is not set, treat it as a “cumulative lesion segmentation” → shown as a colormap overlay.
        – Otherwise (multiple files or `--outline`), treat each input as a binary mask (voxels > corresponding threshold) → draw colored contour outlines on each slice.
    • Use `--highlight` to draw a red border around any slice containing segmentation (voxels > threshold).
    • Supports specifying:
        – A start/end slice index range
        – A slice step (i.e., subsample every Nth valid slice)
        – Log‐scale vs. linear colormap for overlay mode
        – Overlay alpha/transparency
        – How many slices per row in a mosaic
        – Optionally output as an animated GIF instead of a single JPEG mosaic (rescaled to 800px tall)
        – Force outline mode (`--outline`)

:Usage Examples:

  # (1) Single “cumulative segmentation” → colormap overlay (linear), default threshold 0
  python3 neuromontage.py \
    -b brain_T1.nii.gz \
    -s cumulative_segmentation.nii.gz \
    -o mosaic_overlay.jpg \
    --start_slice 10 --end_slice 50 \
    --slice_step 2 \
    --alpha 0.7 \
    --slices_per_row 8

  # (2) Single “cumulative segmentation” → log‐scale colormap overlay, threshold=5
  python3 neuromontage.py \
    -b brain_T1.nii.gz \
    -s cumulative_segmentation.nii.gz \
    -o mosaic_log.jpg \
    --log_scale \
    --alpha 0.5 \
    --slices_per_row 6 \
    --threshold 5

  # (3) Two lesion masks → draw each in a different color outline, single threshold=0.5
  python3 neuromontage.py \
    -b brain_T1.nii.gz \
    -s lesion1.nii.gz lesion2.nii.gz \
    -o mosaic_outlines.jpg \
    --start_slice 20 --end_slice 60 \
    --slice_step 3 \
    --slices_per_row 7 \
    --threshold 0.5

  # (4) Two lesion masks → different thresholds per mask (0.5 for lesion1, 1.0 for lesion2)
  python3 neuromontage.py \
    -b brain_T1.nii.gz \
    -s lesion1.nii.gz lesion2.nii.gz \
    -o mosaic_outlines_diff_thresh.jpg \
    --threshold 0.5 1.0

  # (5) Force outline mode with a single file, threshold defaults to 0 → outlines for any positive voxel
  python3 neuromontage.py \
    -b brain_T1.nii.gz \
    -s cumulative_segmentation.nii.gz \
    -o outline_single.jpg \
    --outline

  # (6) Mosaic with highlighted slices containing segmentation:
  python3 neuromontage.py \
    -b brain_T1.nii.gz \
    -s seg1.nii.gz seg2.nii.gz \
    -o mosaic_highlight.jpg \
    --highlight \
    --slices_per_row 5

  # (7) Animated GIF in “outline” mode (three lesion masks), rescaled to 800px tall, with highlights:
  python3 neuromontage.py \
    -b brain_T1.nii.gz \
    -s lesionA.nii.gz lesionB.nii.gz lesionC.nii.gz \
    -o outlines.gif \
    --gif \
    --duration 5.0 \
    --outline \
    --highlight \
    --threshold 0.5 1.0 0.2

  # (8) Animated GIF in “overlay” mode (single file), rescaled to 800px tall, highlighting any slice with segmentation:
  python3 neuromontage.py \
    -b brain_T1.nii.gz \
    -s cumulative_segmentation.nii.gz \
    -o overlay_animation.gif \
    --gif \
    --duration 8.0 \
    --alpha 0.6 \
    --colormap plasma \
    --log_scale \
    --highlight \
    --threshold 2.5

:Requires:
    numpy, nibabel, matplotlib, Pillow, scikit‐image
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from PIL import Image
import argparse
import skimage.measure


def get_slice(img_data, idx):
    """
    Return the idx‐th axial slice (rotated 90° clockwise for a consistent orientation).
    If idx is out of bounds, use the last valid slice.
    """
    if idx >= img_data.shape[2]:
        idx = img_data.shape[2] - 1
    slice_2d = img_data[:, :, idx]
    return np.rot90(slice_2d, 1)


def find_contours_on_slice(binary_slice):
    """
    Given a 2D binary array `binary_slice` (values 0 or 1), find contours at threshold = 0.5.
    Returns a list of (N_i × 2) arrays of (row, col) floats.
    """
    return skimage.measure.find_contours(binary_slice, 0.5)


def generate_lr_labels():
    """
    Generate two 25×25 “L” and “R” binary masks (values 0 or 255) for orientation labels.
    <Copied from plane_brain.py>
    """
    L = np.zeros((50, 50), dtype=np.uint8)
    L[5:45, 10:15] = 255
    L[40:45, 15:35] = 255

    R = np.zeros((50, 50), dtype=np.uint8)
    R[5:45, 10:15] = 255
    R[5:10, 15:27] = 255
    R[20:25, 15:27] = 255
    for i in range(8):
        R[5 + i : 10 + i, 27 + i : 31 + i] = 255
        R[20 - i : 25 - i, 27 + i : 31 + i] = 255
    R[25:27, 15:25] = 255
    for i in range(21):
        R[21 + i : 25 + i, 15 + i : 20 + i] = 255

    return L[::2, ::2], R[::2, ::2]


def parse_args():
    parser = argparse.ArgumentParser(
        description='NeuroMontage: mosaic/GIF of brain slices with either a cumulative‐segmentation overlay (single input) or colored lesion‐mask outlines (>=2 inputs).'
    )
    parser.add_argument(
        '-b', '--brain_file', type=str, required=True,
        help='Structural brain NIfTI (e.g. T1).'
    )
    parser.add_argument(
        '-s', '--input_files', type=str, nargs='+', required=True,
        help=(
            'One or more lesion/segmentation NIfTI files. '
            '• If exactly 1 file is provided and --outline is not set, it is used as a “cumulative lesion segmentation” → shown as a colormap overlay. '
            '• Otherwise (multiple files or --outline), each input is treated as a lesion mask → draw each mask’s colored contour outlines (voxels > threshold) on each slice.'
        )
    )
    parser.add_argument(
        '--outline', action='store_true',
        help='Force outline mode even if only one input file is provided.'
    )
    parser.add_argument(
        '--highlight', action='store_true',
        help='Draw a red border around any slice that contains segmentation (voxels > threshold).'
    )
    parser.add_argument(
        '-o', '--output_file', type=str, required=True,
        help='Output file path: .jpg for a single mosaic, .gif for an animated GIF.'
    )
    parser.add_argument(
        '--gif', action='store_true',
        help='Produce an animated GIF (iterate through all valid slices).'
    )
    parser.add_argument(
        '--duration', type=float, default=10.0,
        help='GIF duration in seconds (only used if --gif is set). Default=10.0'
    )
    parser.add_argument(
        '--slice_step', type=int, default=2,
        help='Step between valid slices in mosaic (default=2).'
    )
    parser.add_argument(
        '--start_slice', type=int, default=None,
        help='First slice index (inclusive). Default=0.'
    )
    parser.add_argument(
        '--end_slice', type=int, default=None,
        help='Last slice index (inclusive). Default=last slice.'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.7,
        help='Overlay transparency [0..1] (only in single‐input overlay mode). Default=0.7'
    )
    parser.add_argument(
        '--colormap', type=str, default='viridis',
        help='Matplotlib colormap for the cumulative segmentation overlay (only in single‐input overlay mode). Default=viridis'
    )
    parser.add_argument(
        '--log_scale', action='store_true',
        help='Use logarithmic normalization on the cumulative segmentation overlay (only if exactly 1 input file without --outline).'
    )
    parser.add_argument(
        '--threshold', type=float, nargs='+', default=[0.0],
        help=(
            'One or more thresholds. '
            '• If a single value is provided, it is applied to all segmentations. '
            '• If N values are provided for N input_files, each threshold applies to the corresponding file. '
            '• In overlay mode (single file), threshold determines voxels > threshold to show. '
            '• In outline mode (multiple files), each mask is binarized at its threshold to find contours.'
        )
    )
    parser.add_argument(
        '--slices_per_row', type=int, default=8,
        help='Number of slices per row in the mosaic (default=8).'
    )
    parser.add_argument(
        '--upscale', type=float, default=1.0,
        help='Upscale factor for final image (e.g., 2.0 = double size, 4.0 = quadruple). Default=1.0'
    )

    return parser.parse_args()


def create_mosaic(
    brain_file,
    input_files,
    output_file,
    slice_step=2,
    thresholds=None,
    alpha=0.7,
    colormap_name='viridis',
    log_scale=False,
    slices_per_row=8,
    start_slice=None,
    end_slice=None,
    outline=False,
    highlight=False,
    upscale=1.0
):
    """
    Build and save a single mosaic (JPEG) at `output_file`.
    Modes:
      - Overlay mode: if exactly 1 input AND outline=False → colormap overlay (voxels > threshold).
      - Outline mode: if outline=True OR len(input_files) >= 2 → draw colored contours (voxels > threshold).
    If highlight=True, draw a red border around any slice containing segmentation (voxels > threshold).
    thresholds: list of float. If length 1, apply to all. If length == len(input_files), apply per file.
    """
    # 1) Validate/load inputs
    brain_nii = nib.load(brain_file)
    brain_data_full = brain_nii.get_fdata()

    num_inputs = len(input_files)
    single_file = (num_inputs == 1)
    outline_mode = (not single_file) or outline

    # Prepare thresholds list
    if thresholds is None:
        thresholds = [0.0]
    if len(thresholds) == 1:
        thresholds = thresholds * num_inputs
    elif len(thresholds) != num_inputs:
        raise ValueError(
            f"Number of thresholds ({len(thresholds)}) must be 1 or equal to number of input_files ({num_inputs})."
        )

    if not outline_mode:
        # Overlay mode: exactly one file, threshold = thresholds[0]
        cum_nii = nib.load(input_files[0])
        cum_data_full = cum_nii.get_fdata()
        if brain_data_full.shape != cum_data_full.shape:
            raise ValueError("Structural and cumulative segmentation volumes must match in shape.")
    else:
        # Outline mode: load each mask and apply binarization using corresponding threshold
        all_masks = []
        for seg_path in input_files:
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata()
            if seg_data.shape != brain_data_full.shape:
                raise ValueError(f"Segmentation {seg_path} has a different shape than the structural volume.")
            all_masks.append(seg_data)

    num_slices_total = brain_data_full.shape[2]

    # 2) Determine start/end slice indices
    if start_slice is None:
        start_slice = 0
    if end_slice is None or end_slice >= num_slices_total:
        end_slice = num_slices_total - 1

    if not (0 <= start_slice < num_slices_total):
        raise ValueError(f"start_slice ({start_slice}) out of bounds [0, {num_slices_total-1}]")
    if not (0 <= end_slice < num_slices_total):
        raise ValueError(f"end_slice ({end_slice}) out of bounds [0, {num_slices_total-1}]")
    if start_slice > end_slice:
        raise ValueError("start_slice cannot exceed end_slice.")

    # 3) Determine orientation from structural affine
    orientation = 'R-L' if brain_nii.affine[0, 0] < 0 else 'L-R'

    # 4) Build list of non‐zero slice indices within [start_slice..end_slice]
    valid_slices = [
        idx for idx in range(start_slice, end_slice + 1)
        if np.any(brain_data_full[:, :, idx] > 0)
    ]
    used_slices = valid_slices[::slice_step]
    if not used_slices:
        raise ValueError(f"No non‐zero slices found between indices {start_slice} and {end_slice}.")

    # 5) Determine slice dimensions
    example_slice = get_slice(brain_data_full, used_slices[0])
    slice_h, slice_w = example_slice.shape

    # 6) Layout mosaic grid
    n_cols = slices_per_row
    n_rows = int(np.ceil(len(used_slices) / n_cols))
    canvas_h = n_rows * slice_h
    canvas_w = n_cols * slice_w

    # 7) Prepare blank canvases
    base_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)       # structural grayscale
    overlay_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)    # overlay values (if single‐input)
    alpha_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)        # transparency mask (single‐input)
    outline_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)   # RGB mask outlines (multi‐input)

    # 8) Normalization for overlay mode
    if not outline_mode:
        norm_brain = Normalize(vmin=np.min(brain_data_full), vmax=np.max(brain_data_full))
        thr0 = thresholds[0]
        if log_scale:
            pos_vals = cum_data_full[cum_data_full > thr0]
            if pos_vals.size == 0:
                raise ValueError("No positive cumulative‐segmentation values above threshold for log scaling.")
            norm_overlay = LogNorm(vmin=pos_vals.min(), vmax=pos_vals.max())
        else:
            norm_overlay = Normalize(vmin=np.min(cum_data_full), vmax=np.max(cum_data_full))
        cmap = plt.get_cmap(colormap_name)
    else:
        # Outline mode: define colors for each mask
        contour_colors = [
            np.array([255, 0, 0], dtype=np.uint8),      # red
            np.array([0, 255, 0], dtype=np.uint8),      # green
            np.array([0, 0, 255], dtype=np.uint8),      # blue
            np.array([255, 165, 0], dtype=np.uint8),    # orange
            np.array([255, 0, 255], dtype=np.uint8),    # magenta
            np.array([0, 255, 255], dtype=np.uint8),    # cyan
            np.array([255, 255, 0], dtype=np.uint8),    # yellow
            np.array([128, 0, 128], dtype=np.uint8),    # purple
        ]
        num_colors = len(contour_colors)

    # 9) Loop through each slice
    for idx, slice_idx in enumerate(used_slices):
        row_i = idx // n_cols
        col_i = idx % n_cols
        y0, y1 = row_i * slice_h, (row_i + 1) * slice_h
        x0, x1 = col_i * slice_w, (col_i + 1) * slice_w

        # 9a) Structural slice → grayscale
        base_slice = get_slice(brain_data_full, slice_idx)
        if not outline_mode:
            base_canvas[y0:y1, x0:x1] = norm_brain(base_slice)
        else:
            base_canvas[y0:y1, x0:x1] = base_slice / np.max(brain_data_full)

        # 9b) Overlay mode: fill overlay_canvas + alpha_mask
        if not outline_mode:
            cum_slice = get_slice(cum_data_full, slice_idx)
            thr0 = thresholds[0]
            mask_vals = np.where(cum_slice > thr0, alpha, 0.0)
            overlay_canvas[y0:y1, x0:x1] = norm_overlay(cum_slice)
            alpha_mask[y0:y1, x0:x1] = mask_vals

        # 9c) Outline mode: draw each mask’s contours (voxels > threshold) in its assigned color
        else:
            for m_i, mask_vol in enumerate(all_masks):
                seg_slice = get_slice(mask_vol, slice_idx)
                thr_i = thresholds[m_i]
                binary_slice = (seg_slice > thr_i).astype(np.uint8)
                contours = find_contours_on_slice(binary_slice)
                color = contour_colors[m_i % num_colors]

                for contour in contours:
                    # Thicken by drawing a 3×3 neighborhood around each contour point
                    for (r, c) in contour:
                        rr = int(round(r))
                        cc = int(round(c))
                        if 0 <= rr < slice_h and 0 <= cc < slice_w:
                            outline_canvas[y0 + rr, x0 + cc, :] = color

    # 10) Build the final RGB composite
    if not outline_mode:
        rgb_overlay = cmap(overlay_canvas)[..., :3]        # floats in [0,1]
        rgb_base = plt.cm.gray(base_canvas)[..., :3]       # floats in [0,1]
        composite = (rgb_overlay * alpha_mask[..., None] +
                     rgb_base * (1.0 - alpha_mask[..., None]))
        composite_uint8 = (composite * 255).astype(np.uint8)
    else:
        rgb_base = plt.cm.gray(base_canvas)[..., :3]       # floats in [0,1]
        composite_uint8 = (rgb_base * 255).astype(np.uint8)
        mask_idx = np.any(outline_canvas > 0, axis=2)
        composite_uint8[mask_idx] = outline_canvas[mask_idx]

    # 11) Highlight slices containing segmentation (red border)
    if highlight:
        red_border = np.array([255, 0, 0], dtype=np.uint8)
        for idx, slice_idx in enumerate(used_slices):
            has_seg = False
            if not outline_mode and single_file:
                sl = cum_data_full[:, :, slice_idx]
                if np.any(sl > thresholds[0]):
                    has_seg = True
            else:
                for m_i, mask_vol in enumerate(all_masks):
                    sl = mask_vol[:, :, slice_idx]
                    if np.any(sl > thresholds[m_i]):
                        has_seg = True
                        break

            if has_seg:
                row_i = idx // n_cols
                col_i = idx % n_cols
                y0, y1 = row_i * slice_h, (row_i + 1) * slice_h
                x0, x1 = col_i * slice_w, (col_i + 1) * slice_w
                for t in range(2):
                    composite_uint8[y0 + t, x0:x1, :] = red_border
                    composite_uint8[y1 - 1 - t, x0:x1, :] = red_border
                    composite_uint8[y0:y1, x0 + t, :] = red_border
                    composite_uint8[y0:y1, x1 - 1 - t, :] = red_border

    # 12) Add L/R labels in top corners (white ×3 intensity)
    left_label, right_label = generate_lr_labels()
    pad = 10
    if orientation == 'L-R':
        composite_uint8[
            pad : pad + left_label.shape[0],
            pad : pad + left_label.shape[1],
            :
        ] = left_label[..., None] * 3

        composite_uint8[
            pad : pad + right_label.shape[0],
            -pad - right_label.shape[1] : -pad,
            :
        ] = right_label[..., None] * 3
    else:  # R-L
        composite_uint8[
            pad : pad + left_label.shape[0],
            -pad - left_label.shape[1] : -pad,
            :
        ] = left_label[..., None] * 3

        composite_uint8[
            pad : pad + right_label.shape[0],
            pad : pad + right_label.shape[1],
            :
        ] = right_label[..., None] * 3

    # 13) Save as JPEG
    final_image = Image.fromarray(composite_uint8)
    if upscale != 1.0:
        new_w = int(final_image.width * upscale)
        new_h = int(final_image.height * upscale)
        final_image = final_image.resize((new_w, new_h), resample=Image.LANCZOS)
    final_image.save(output_file, format='JPEG', quality=95)
    print(f"[NeuroMontage] Mosaic saved to {output_file}")


def create_gif(
    brain_file,
    input_files,
    output_file,
    gif_duration=10.0,
    alpha=0.7,
    colormap_name='viridis',
    log_scale=False,
    thresholds=None,
    outline=False,
    highlight=False,
    upscale=1.0
):
    """
    Build and save an animated GIF at `output_file`.  
    Modes:
      - Overlay mode: if exactly 1 input AND outline=False → colormap overlay (voxels > threshold).
      - Outline mode: if outline=True OR len(input_files) >= 2 → draw colored contours (voxels > threshold).
    Each frame is resized to 800px tall before being added to the GIF.
    If highlight=True, draw a red border around frames that contain segmentation (voxels > threshold).
    thresholds: list of float. If length 1, apply to all. If length == len(input_files), apply per file.
    """
    brain_nii = nib.load(brain_file)
    brain_data_full = brain_nii.get_fdata()

    num_inputs = len(input_files)
    single_file = (num_inputs == 1)
    outline_mode = (not single_file) or outline

    # Prepare thresholds list
    if thresholds is None:
        thresholds = [0.0]
    if len(thresholds) == 1:
        thresholds = thresholds * num_inputs
    elif len(thresholds) != num_inputs:
        raise ValueError(
            f"Number of thresholds ({len(thresholds)}) must be 1 or equal to number of input_files ({num_inputs})."
        )

    if not outline_mode:
        # Single‐input overlay: load cumulative segmentation & set up norms
        cum_nii = nib.load(input_files[0])
        cum_data_full = cum_nii.get_fdata()
        if brain_data_full.shape != cum_data_full.shape:
            raise ValueError("Structural and cumulative segmentation volumes must match in shape.")

        norm_brain = Normalize(vmin=np.min(brain_data_full), vmax=np.max(brain_data_full))
        thr0 = thresholds[0]
        if log_scale:
            pos_vals = cum_data_full[cum_data_full > thr0]
            if pos_vals.size == 0:
                raise ValueError("No positive cumulative‐segmentation values above threshold for log scaling.")
            norm_overlay = LogNorm(vmin=pos_vals.min(), vmax=pos_vals.max())
        else:
            norm_overlay = Normalize(vmin=np.min(cum_data_full), vmax=np.max(cum_data_full))
        cmap = plt.get_cmap(colormap_name)

    else:
        # Multi‐input or forced outline mode: load each mask and set up colors
        all_masks = []
        for seg_path in input_files:
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata()
            if seg_data.shape != brain_data_full.shape:
                raise ValueError(f"Segmentation {seg_path} has a different shape.")
            if seg_data.ndim == 4:
                seg_data = seg_data[:, :, :, 0]
            all_masks.append(seg_data)

        contour_colors = [
            np.array([255, 0, 0], dtype=np.uint8),      # red
            np.array([0, 255, 0], dtype=np.uint8),      # green
            np.array([0, 0, 255], dtype=np.uint8),      # blue
            np.array([255, 165, 0], dtype=np.uint8),    # orange
            np.array([255, 0, 255], dtype=np.uint8),    # magenta
            np.array([0, 255, 255], dtype=np.uint8),    # cyan
            np.array([255, 255, 0], dtype=np.uint8),    # yellow
            np.array([128, 0, 128], dtype=np.uint8),    # purple
        ]
        num_colors = len(contour_colors)

    # Determine all valid slices (where brain_data_full > 0)
    valid_slices = [
        i for i in range(brain_data_full.shape[2])
        if np.any(brain_data_full[:, :, i] > 0)
    ]
    if not valid_slices:
        raise ValueError("No valid slices (brain_data > 0) found.")

    images = []
    for slice_idx in valid_slices:
        # 1) Structural base
        base_slice = get_slice(brain_data_full, slice_idx)
        slice_h, slice_w = base_slice.shape

        if not outline_mode:
            # Single‐input overlay frame
            cum_slice = get_slice(cum_data_full, slice_idx)
            thr0 = thresholds[0]
            alpha_slice = np.where(cum_slice > thr0, alpha, 0.0)

            base_norm    = norm_brain(base_slice)
            overlay_norm = norm_overlay(cum_slice)

            rgb_base    = plt.cm.gray(base_norm)[..., :3]
            rgb_overlay = cmap(overlay_norm)[..., :3]
            frame_rgb   = (rgb_overlay * alpha_slice[..., None] +
                           rgb_base * (1.0 - alpha_slice[..., None]))
            frame_uint8 = (frame_rgb * 255).astype(np.uint8)

        else:
            # Outline mode: structural grayscale background
            rgb_base = plt.cm.gray(base_slice / np.max(brain_data_full))[..., :3]
            frame_uint8 = (rgb_base * 255).astype(np.uint8)

            # Draw colored, thickened outlines for each mask (voxels > threshold)
            for m_i, mask_vol in enumerate(all_masks):
                seg_slice = get_slice(mask_vol, slice_idx)
                thr_i = thresholds[m_i]
                binary_slice = (seg_slice > thr_i).astype(np.uint8)
                contours = find_contours_on_slice(binary_slice)
                color = contour_colors[m_i % num_colors]

                for contour in contours:
                    for (r, c) in contour:
                        rr = int(round(r))
                        cc = int(round(c))
                        # Draw a 3×3 neighborhood for thickness
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                rrr = rr + dr
                                ccc = cc + dc
                                if 0 <= rrr < slice_h and 0 <= ccc < slice_w:
                                    frame_uint8[rrr, ccc, :] = color

        # Add L/R labels per‐frame
        left_label, right_label = generate_lr_labels()
        pad = 10
        orientation = 'R-L' if brain_nii.affine[0, 0] < 0 else 'L-R'
        if orientation == 'L-R':
            frame_uint8[
                pad : pad + left_label.shape[0],
                pad : pad + left_label.shape[1],
                :
            ] = left_label[..., None] * 3
            frame_uint8[
                pad : pad + right_label.shape[0],
                -pad - right_label.shape[1] : -pad,
                :
            ] = right_label[..., None] * 3
        else:
            frame_uint8[
                pad : pad + left_label.shape[0],
                -pad - left_label.shape[1] : -pad,
                :
            ] = left_label[..., None] * 3
            frame_uint8[
                pad : pad + right_label.shape[0],
                pad : pad + right_label.shape[1],
                :
            ] = right_label[..., None] * 3

        # Highlight this frame if segmentation present and highlight=True
        if highlight:
            has_seg = False
            if not outline_mode and single_file:
                sl = cum_data_full[:, :, slice_idx]
                if np.any(sl > thresholds[0]):
                    has_seg = True
            else:
                for m_i, mask_vol in enumerate(all_masks):
                    sl = mask_vol[:, :, slice_idx]
                    if np.any(sl > thresholds[m_i]):
                        has_seg = True
                        break
            if has_seg:
                red_border = np.array([255, 0, 0], dtype=np.uint8)
                for t in range(2):
                    frame_uint8[t, :, :] = red_border
                    frame_uint8[slice_h - 1 - t, :, :] = red_border
                    frame_uint8[:, t, :] = red_border
                    frame_uint8[:, slice_w - 1 - t, :] = red_border

        # Convert to PIL and resize to 800px tall
        pil_frame = Image.fromarray(frame_uint8).convert("RGBA")
        scale = 800.0 / slice_h
        new_w = int(slice_w * scale)
        pil_frame_resized = pil_frame.resize((new_w, 800), resample=Image.LANCZOS)
        images.append(pil_frame_resized)

    # Save animated GIF
    if upscale != 1.0:
        upscaled_images = []
        for img in images:
            new_w = int(img.width * upscale)
            new_h = int(img.height * upscale)
            upscaled_img = img.resize((new_w, new_h), resample=Image.LANCZOS)
            upscaled_images.append(upscaled_img)
        images = upscaled_images

    duration_per_frame = int(gif_duration * 1000 / len(images))
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration_per_frame,
        loop=0
    )

    print(f"[NeuroMontage] GIF saved to {output_file}")


def main():
    args = parse_args()

    if args.gif:
        create_gif(
            brain_file=args.brain_file,
            input_files=args.input_files,
            output_file=args.output_file,
            gif_duration=args.duration,
            alpha=args.alpha,
            colormap_name=args.colormap,
            log_scale=args.log_scale,
            thresholds=args.threshold,
            outline=args.outline,
            highlight=args.highlight,
            upscale=args.upscale
        )
    else:
        create_mosaic(
            brain_file=args.brain_file,
            input_files=args.input_files,
            output_file=args.output_file,
            slice_step=args.slice_step,
            thresholds=args.threshold,
            alpha=args.alpha,
            colormap_name=args.colormap,
            log_scale=args.log_scale,
            slices_per_row=args.slices_per_row,
            start_slice=args.start_slice,
            end_slice=args.end_slice,
            outline=args.outline,
            highlight=args.highlight,
            upscale=args.upscale
        )


if __name__ == '__main__':
    main()

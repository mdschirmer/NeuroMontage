"""
GIF animation creation for NeuroMontage.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

from image_utils import (get_slice, find_contours_on_slice, generate_lr_labels, 
                         get_orientation, get_contour_colors, should_draw_pixel, resolve_outline_styles,
                         calculate_robust_normalization, calculate_overlay_normalization, load_nifti_robust)


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
    outline_colors=None,
    line_styles=None
):
    """
    Build and save an animated GIF. Each frame is 800px tall.
    """
    brain_nii, brain_data_full = load_nifti_robust(brain_file)
    
    if len(brain_data_full.shape) == 4:
        raise ValueError("[NeuroMontage] GIF creation not supported for 4D data. Use mosaic mode instead.")

    has_segmentation = input_files is not None and len(input_files) > 0
    
    if has_segmentation:
        num_inputs = len(input_files)
        single_file = (num_inputs == 1)
        outline_mode = (not single_file) or outline

        if thresholds is None:
            thresholds = [0.0]
        if len(thresholds) == 1:
            thresholds = thresholds * num_inputs
        elif len(thresholds) != num_inputs:
            raise ValueError(f"Number of thresholds ({len(thresholds)}) must be 1 or equal to number of input_files ({num_inputs}).")

        if not outline_mode:
            _, cum_data_full = load_nifti_robust(input_files[0])
            if brain_data_full.shape != cum_data_full.shape:
                raise ValueError("Structural and cumulative segmentation volumes must match in shape.")

            # Robust normalization for brain
            norm_brain = calculate_robust_normalization(brain_data_full)
            
            thr0 = thresholds[0]
            if log_scale:
                pos_vals = cum_data_full[cum_data_full > thr0]
                if pos_vals.size == 0:
                    raise ValueError("No positive cumulative-segmentation values above threshold for log scaling.")
                norm_overlay = LogNorm(vmin=pos_vals.min(), vmax=pos_vals.max())
            else:
                # Colors start at the threshold, so every shown value is visible
                norm_overlay = calculate_overlay_normalization(cum_data_full, thr0)
            cmap = plt.get_cmap(colormap_name)

        else:
            all_masks = []
            for seg_path in input_files:
                _, seg_data = load_nifti_robust(seg_path)
                if seg_data.shape != brain_data_full.shape:
                    raise ValueError(f"Segmentation {seg_path} has a different shape.")
                if seg_data.ndim == 4:
                    seg_data = seg_data[:, :, :, 0]
                all_masks.append(seg_data)

            # Robust normalization for brain
            norm_brain = calculate_robust_normalization(brain_data_full)
            
            contour_colors, contour_patterns = resolve_outline_styles(len(all_masks), outline_colors, line_styles)
    else:
        outline_mode = False
        single_file = False
        # Robust normalization for brain-only mode
        norm_brain = calculate_robust_normalization(brain_data_full)

    valid_slices = [
        i for i in range(brain_data_full.shape[2])
        if np.any(brain_data_full[:, :, i] > 0)
    ]
    if not valid_slices:
        raise ValueError("No valid slices found.")

    images = []
    orientation = get_orientation(brain_nii.affine)
    
    for slice_idx in valid_slices:
        base_slice = get_slice(brain_data_full, slice_idx)
        slice_h, slice_w = base_slice.shape

        if has_segmentation and not outline_mode:
            cum_slice = get_slice(cum_data_full, slice_idx)
            thr0 = thresholds[0]
            alpha_slice = np.where(cum_slice > thr0, alpha, 0.0)

            base_norm = norm_brain(base_slice)
            overlay_norm = norm_overlay(cum_slice)

            rgb_base = plt.cm.gray(base_norm)[..., :3]
            rgb_overlay = cmap(overlay_norm)[..., :3]
            frame_rgb = (rgb_overlay * alpha_slice[..., None] +
                        rgb_base * (1.0 - alpha_slice[..., None]))
            frame_uint8 = (frame_rgb * 255).astype(np.uint8)

        elif has_segmentation and outline_mode:
            base_norm = norm_brain(base_slice)
            rgb_base = plt.cm.gray(base_norm)[..., :3]
            frame_uint8 = (rgb_base * 255).astype(np.uint8)

            # Draw masks in order (solid first, dashed, dotted on top)
            for m_i, mask_vol in enumerate(all_masks):
                seg_slice = get_slice(mask_vol, slice_idx)
                thr_i = thresholds[m_i]
                binary_slice = (seg_slice > thr_i).astype(np.uint8)
                contours = find_contours_on_slice(binary_slice)
                color = contour_colors[m_i]

                for contour in contours:
                    for pt_idx, (r, c) in enumerate(contour):
                        if not should_draw_pixel(pt_idx, contour_patterns[m_i]):
                            continue
                        rr = int(round(r))
                        cc = int(round(c))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                rrr = rr + dr
                                ccc = cc + dc
                                if 0 <= rrr < slice_h and 0 <= ccc < slice_w:
                                    frame_uint8[rrr, ccc, :] = color
        else:
            base_norm = norm_brain(base_slice)
            rgb_base = plt.cm.gray(base_norm)[..., :3]
            frame_uint8 = (rgb_base * 255).astype(np.uint8)

        # Add L/R labels
        left_label, right_label = generate_lr_labels()
        pad = 10
        
        if orientation == 'L-R':
            placements = [(left_label, pad), (right_label, slice_w - pad - right_label.shape[1])]
        else:
            placements = [(right_label, pad), (left_label, slice_w - pad - left_label.shape[1])]
        
        # Blend the labels into the frame (keep the brighter pixel)
        for label, x0 in placements:
            h, w = label.shape
            region = frame_uint8[pad:pad + h, x0:x0 + w]
            np.maximum(region, (label * 0.9).astype(np.uint8)[..., None], out=region)

        # Highlight
        if highlight and has_segmentation:
            has_seg = False
            if not outline_mode and single_file:
                sl = cum_data_full[:, :, slice_idx]
                if np.any(sl > thresholds[0]):
                    has_seg = True
            elif outline_mode:
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

        # Convert to PIL and resize to 800px tall (no interpolation when enlarging)
        pil_frame = Image.fromarray(frame_uint8).convert("RGBA")
        scale = 800.0 / slice_h
        new_w = int(slice_w * scale)
        resample = Image.NEAREST if scale >= 1.0 else Image.LANCZOS
        pil_frame_resized = pil_frame.resize((new_w, 800), resample=resample)
        images.append(pil_frame_resized)

    duration_per_frame = int(gif_duration * 1000 / len(images))
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration_per_frame,
        loop=0
    )

    print(f"[NeuroMontage] GIF saved to {output_file}")
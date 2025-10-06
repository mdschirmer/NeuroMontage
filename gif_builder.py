"""
GIF animation creation for NeuroMontage.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

from image_utils import (get_slice, find_contours_on_slice, generate_lr_labels, 
                         get_orientation, get_contour_colors, calculate_robust_normalization)


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
    highlight=False
):
    """
    Build and save an animated GIF. Each frame is 800px tall.
    """
    brain_nii = nib.load(brain_file)
    brain_data_full = brain_nii.get_fdata()
    
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
            cum_nii = nib.load(input_files[0])
            cum_data_full = cum_nii.get_fdata()
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
                # Robust normalization for overlay
                norm_overlay = calculate_robust_normalization(cum_data_full)
            cmap = plt.get_cmap(colormap_name)

        else:
            all_masks = []
            for seg_path in input_files:
                seg_nii = nib.load(seg_path)
                seg_data = seg_nii.get_fdata()
                if seg_data.shape != brain_data_full.shape:
                    raise ValueError(f"Segmentation {seg_path} has a different shape.")
                if seg_data.ndim == 4:
                    seg_data = seg_data[:, :, :, 0]
                all_masks.append(seg_data)

            # Robust normalization for brain
            norm_brain = calculate_robust_normalization(brain_data_full)
            
            contour_colors = get_contour_colors()
            num_colors = len(contour_colors)
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
            frame_uint8[pad:pad + left_label.shape[0], pad:pad + left_label.shape[1], :] = left_label[..., None] * 3
            frame_uint8[pad:pad + right_label.shape[0], -pad - right_label.shape[1]:-pad, :] = right_label[..., None] * 3
        else:
            frame_uint8[pad:pad + left_label.shape[0], -pad - left_label.shape[1]:-pad, :] = left_label[..., None] * 3
            frame_uint8[pad:pad + right_label.shape[0], pad:pad + right_label.shape[1], :] = right_label[..., None] * 3

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

        # Convert to PIL and resize to 800px tall
        pil_frame = Image.fromarray(frame_uint8).convert("RGBA")
        scale = 800.0 / slice_h
        new_w = int(slice_w * scale)
        pil_frame_resized = pil_frame.resize((new_w, 800), resample=Image.LANCZOS)
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
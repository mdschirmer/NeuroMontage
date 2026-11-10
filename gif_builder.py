"""
Animation (GIF or WebP) creation for NeuroMontage.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from image_utils import (get_slice, generate_lr_labels,
                         get_orientation, resolve_outline_styles,
                         calculate_robust_normalization, calculate_overlay_colors, load_nifti_robust,
                         fill_non_finite, slice_has_content, prepare_thresholds,
                         draw_contours, draw_border, segmentation_in_slice)


def create_gif(
    brain_file,
    input_files,
    output_file,
    gif_duration=10.0,
    alpha=0.7,
    colormap_name=None,
    log_scale=False,
    thresholds=None,
    outline=False,
    highlight=False,
    outline_colors=None,
    line_styles=None
):
    """
    Build and save an animation (.gif or animated .webp). Each frame is 800px tall.
    """
    brain_nii, brain_data_full = load_nifti_robust(brain_file)
    brain_data_full = fill_non_finite(brain_data_full, 'Brain')
    
    if len(brain_data_full.shape) == 4:
        raise ValueError("[NeuroMontage] GIF creation not supported for 4D data. Use mosaic mode instead.")

    has_segmentation = input_files is not None and len(input_files) > 0
    
    if has_segmentation:
        num_inputs = len(input_files)
        single_file = (num_inputs == 1)
        outline_mode = (not single_file) or outline

        thresholds = prepare_thresholds(thresholds, num_inputs)

        if not outline_mode:
            _, cum_data_full = load_nifti_robust(input_files[0])
            cum_data_full = fill_non_finite(cum_data_full, 'Overlay', ceiling=thresholds[0])
            if brain_data_full.shape != cum_data_full.shape:
                raise ValueError("Structural and cumulative segmentation volumes must match in shape.")

            # Robust normalization for brain
            norm_brain = calculate_robust_normalization(brain_data_full)
            
            thr0 = thresholds[0]
            norm_overlay, cmap = calculate_overlay_colors(cum_data_full, thr0, colormap_name, log_scale)

        else:
            all_masks = []
            for m_i, seg_path in enumerate(input_files):
                _, seg_data = load_nifti_robust(seg_path)
                seg_data = fill_non_finite(seg_data, f'Mask {m_i + 1}', ceiling=thresholds[m_i])
                if seg_data.shape != brain_data_full.shape:
                    raise ValueError(f"Segmentation {seg_path} has a different shape.")
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
        if slice_has_content(brain_data_full[:, :, i])
    ]
    if not valid_slices:
        raise ValueError(f"No slices with image content: every slice is constant "
                         f"(e.g. all 0 or all NaN). Image shape: {brain_data_full.shape}.")

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
                draw_contours(frame_uint8, binary_slice, contour_colors[m_i], contour_patterns[m_i],
                              offsets=(-1, 0, 1))
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
            seg_volumes = all_masks if outline_mode else [cum_data_full]
            if segmentation_in_slice(seg_volumes, thresholds, slice_idx):
                draw_border(frame_uint8, 2, np.array([255, 0, 0], dtype=np.uint8))

        # Convert to PIL and resize to 800px tall (no interpolation when enlarging)
        pil_frame = Image.fromarray(frame_uint8).convert("RGBA")
        scale = 800.0 / slice_h
        new_w = int(slice_w * scale)
        resample = Image.NEAREST if scale >= 1.0 else Image.LANCZOS
        pil_frame_resized = pil_frame.resize((new_w, 800), resample=resample)
        images.append(pil_frame_resized)

    duration_per_frame = int(gif_duration * 1000 / len(images))
    if output_file.lower().endswith('.webp'):
        # Same rule as for mosaics: lossless with outlines (lossy WebP smears thin
        # colored lines), quality 95 otherwise. Both are smaller than GIF.
        if has_segmentation and outline_mode:
            webp_options = dict(lossless=True, method=6)
        else:
            webp_options = dict(quality=95, method=4)
        images = [im.convert('RGB') for im in images]
        images[0].save(
            output_file,
            format='WEBP',
            save_all=True,
            append_images=images[1:],
            duration=duration_per_frame,
            loop=0,
            **webp_options
        )
    else:
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=duration_per_frame,
            loop=0
        )

    print(f"[NeuroMontage] GIF saved to {output_file}")
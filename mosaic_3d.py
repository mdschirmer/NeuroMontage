"""
3D mosaic creation for NeuroMontage.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from skimage.transform import resize

from layout_utils import (RESOLUTION_PRESETS, calculate_auto_resolution_and_sampling, 
                          calculate_optimal_layout)
from image_utils import (get_slice, find_contours_on_slice, generate_lr_labels, 
                         get_orientation, get_contour_colors, calculate_robust_normalization)

# Increase PIL's image size limit to handle large mosaics
Image.MAX_IMAGE_PIXELS = None


def create_mosaic(
    brain_file,
    input_files,
    output_file,
    resolution='auto',
    slice_step=1,
    thresholds=None,
    alpha=0.7,
    colormap_name='viridis',
    log_scale=False,
    start_slice=None,
    end_slice=None,
    outline=False,
    highlight=False,
    alternate=False
):
    """
    Build and save a single mosaic (JPEG) with resolution-based layout.
    
    If alternate=True and segmentation files are provided:
        - Each row of slices is shown twice
        - First occurrence: brain-only
        - Second occurrence: brain+segmentation overlay
        Example: Row 1 = slices 1-6 (brain), Row 2 = slices 1-6 (brain+overlay),
                 Row 3 = slices 7-12 (brain), Row 4 = slices 7-12 (brain+overlay), etc.
    """
    # Load brain data
    brain_nii = nib.load(brain_file)
    brain_data_full = brain_nii.get_fdata()
    
    # Validate 3D data
    if len(brain_data_full.shape) != 3:
        raise ValueError("create_mosaic only supports 3D data. Use create_super_mosaic_4d for 4D data.")
    
    has_segmentation = input_files is not None and len(input_files) > 0
    
    # Validate alternate mode
    if alternate and not has_segmentation:
        print("[NeuroMontage] Warning: --alternate flag requires segmentation files. Ignoring --alternate.")
        alternate = False
    
    if has_segmentation:
        num_inputs = len(input_files)
        single_file = (num_inputs == 1)
        outline_mode = (not single_file) or outline

        # Prepare thresholds
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
        else:
            all_masks = []
            for seg_path in input_files:
                seg_nii = nib.load(seg_path)
                seg_data = seg_nii.get_fdata()
                if seg_data.shape != brain_data_full.shape:
                    raise ValueError(f"Segmentation {seg_path} has a different shape than the structural volume.")
                all_masks.append(seg_data)
    else:
        outline_mode = False
        single_file = False

    num_slices_total = brain_data_full.shape[2]

    # Determine slice range
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

    orientation = get_orientation(brain_nii.affine)

    # Build list of slices to use
    valid_slices = [
        idx for idx in range(start_slice, end_slice + 1)
        if np.any(brain_data_full[:, :, idx] > 0)
    ]
    
    # Apply user-specified slice_step first
    used_slices = valid_slices[::slice_step]
    if not used_slices:
        raise ValueError(f"No non-zero slices found between indices {start_slice} and {end_slice}.")

    # Get slice dimensions for resolution calculation
    example_slice = get_slice(brain_data_full, used_slices[0])
    slice_h, slice_w = example_slice.shape
    
    # If alternate mode, we need to display each slice twice (once plain, once with overlay)
    if alternate:
        display_count = len(used_slices) * 2
        print(f"[NeuroMontage] Alternate mode: displaying {len(used_slices)} slices twice = {display_count} total")
    else:
        display_count = len(used_slices)
    
    # Determine resolution and check if adaptive sampling needed
    if resolution == 'auto':
        target_width, target_height, adaptive_step, final_display_count = \
            calculate_auto_resolution_and_sampling(
                display_count, 
                slice_h=slice_h, 
                slice_w=slice_w,
                is_4d=False
            )
        
        # Apply adaptive sampling if recommended
        if adaptive_step > 1:
            used_slices = used_slices[::adaptive_step]
            if alternate:
                display_count = len(used_slices) * 2
            else:
                display_count = len(used_slices)
            print(f"[NeuroMontage] Final display count: {display_count} ({slice_h}x{slice_w} slices)")
        
        print(f"[NeuroMontage] Auto resolution: {target_width}×{target_height}")
    else:
        target_width, target_height = RESOLUTION_PRESETS[resolution]
        print(f"[NeuroMontage] Using preset {resolution.upper()} resolution: {target_width}×{target_height}")
    
    # Calculate optimal layout based on display count
    cols, rows, scale, actual_width, actual_height = calculate_optimal_layout(
        display_count, slice_h, slice_w, target_width, target_height
    )
    
    print(f"[NeuroMontage] Layout: {cols}×{rows} grid")
    print(f"[NeuroMontage] Output dimensions: {actual_width}×{actual_height} (scale: {scale:.2f}x)")
    if resolution != 'auto':
        print(f"[NeuroMontage] Target was: {target_width}×{target_height} ({resolution.upper()})")
    
    # Apply uniform scale to slice dimensions
    scaled_slice_h = int(slice_h * scale)
    scaled_slice_w = int(slice_w * scale)
    
    # Create canvases
    canvas_h = rows * scaled_slice_h
    canvas_w = cols * scaled_slice_w
    
    base_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    if has_segmentation:
        overlay_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        alpha_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        outline_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Robust normalization for brain
    norm_brain = calculate_robust_normalization(brain_data_full)
    
    if has_segmentation and not outline_mode:
        thr0 = thresholds[0]
        if log_scale:
            # For log scale, still need to handle thresholding
            pos_vals = cum_data_full[cum_data_full > thr0]
            if pos_vals.size == 0:
                raise ValueError("No positive cumulative-segmentation values above threshold for log scaling.")
            norm_overlay = LogNorm(vmin=pos_vals.min(), vmax=pos_vals.max())
        else:
            # Use robust normalization for overlay as well
            norm_overlay = calculate_robust_normalization(cum_data_full)
        cmap = plt.get_cmap(colormap_name)
    elif has_segmentation and outline_mode:
        contour_colors = get_contour_colors()
        num_colors = len(contour_colors)

    # Build the display sequence
    display_sequence = []
    if alternate:
        # Group slices into chunks of 'cols' size
        for chunk_start in range(0, len(used_slices), cols):
            chunk_end = min(chunk_start + cols, len(used_slices))
            chunk_slices = used_slices[chunk_start:chunk_end]
            
            # First row: brain-only for this chunk
            for slice_idx in chunk_slices:
                display_sequence.append((slice_idx, False))
            
            # Second row: brain+overlay for this chunk
            for slice_idx in chunk_slices:
                display_sequence.append((slice_idx, True))
    else:
        for slice_idx in used_slices:
            display_sequence.append((slice_idx, has_segmentation))

    # Process each display slot
    for display_idx, (slice_idx, show_overlay) in enumerate(display_sequence):
        row_i = display_idx // cols
        col_i = display_idx % cols
        
        if row_i >= rows:
            break
        
        y0, y1 = row_i * scaled_slice_h, (row_i + 1) * scaled_slice_h
        x0, x1 = col_i * scaled_slice_w, (col_i + 1) * scaled_slice_w

        # Get and scale structural slice
        base_slice = get_slice(brain_data_full, slice_idx)
        base_slice_norm = norm_brain(base_slice)
        
        if scale != 1.0:
            base_slice_scaled = resize(base_slice_norm, (scaled_slice_h, scaled_slice_w), 
                                      order=1, anti_aliasing=True, preserve_range=True)
        else:
            base_slice_scaled = base_slice_norm
        
        base_canvas[y0:y1, x0:x1] = base_slice_scaled

        if has_segmentation and show_overlay:
            if not outline_mode:
                # Overlay mode
                cum_slice = get_slice(cum_data_full, slice_idx)
                thr0 = thresholds[0]
                
                if scale != 1.0:
                    cum_slice_scaled = resize(cum_slice, (scaled_slice_h, scaled_slice_w),
                                            order=0, anti_aliasing=False, preserve_range=True)
                    mask_vals = np.where(cum_slice_scaled > thr0, alpha, 0.0)
                else:
                    cum_slice_scaled = cum_slice
                    mask_vals = np.where(cum_slice > thr0, alpha, 0.0)
                
                overlay_canvas[y0:y1, x0:x1] = norm_overlay(cum_slice_scaled)
                alpha_mask[y0:y1, x0:x1] = mask_vals

            else:
                # Outline mode
                for m_i, mask_vol in enumerate(all_masks):
                    seg_slice = get_slice(mask_vol, slice_idx)
                    thr_i = thresholds[m_i]
                    
                    if scale != 1.0:
                        seg_slice_scaled = resize(seg_slice, (scaled_slice_h, scaled_slice_w),
                                                order=0, anti_aliasing=False, preserve_range=True)
                    else:
                        seg_slice_scaled = seg_slice
                    
                    binary_slice = (seg_slice_scaled > thr_i).astype(np.uint8)
                    contours = find_contours_on_slice(binary_slice)
                    color = contour_colors[m_i % num_colors]

                    for contour in contours:
                        for (r, c) in contour:
                            rr = int(round(r))
                            cc = int(round(c))
                            if 0 <= rr < scaled_slice_h and 0 <= cc < scaled_slice_w:
                                outline_canvas[y0 + rr, x0 + cc, :] = color

    # Build final composite
    if has_segmentation and not outline_mode:
        rgb_overlay = cmap(overlay_canvas)[..., :3]
        rgb_base = plt.cm.gray(base_canvas)[..., :3]
        composite = (rgb_overlay * alpha_mask[..., None] +
                     rgb_base * (1.0 - alpha_mask[..., None]))
        composite_uint8 = (composite * 255).astype(np.uint8)
    elif has_segmentation and outline_mode:
        rgb_base = plt.cm.gray(base_canvas)[..., :3]
        composite_uint8 = (rgb_base * 255).astype(np.uint8)
        mask_idx = np.any(outline_canvas > 0, axis=2)
        composite_uint8[mask_idx] = outline_canvas[mask_idx]
    else:
        rgb_base = plt.cm.gray(base_canvas)[..., :3]
        composite_uint8 = (rgb_base * 255).astype(np.uint8)

    # Highlight slices with segmentation
    if highlight and has_segmentation:
        red_border = np.array([255, 0, 0], dtype=np.uint8)
        border_width = max(2, int(scale * 2))
        
        for display_idx, (slice_idx, show_overlay) in enumerate(display_sequence):
            if display_idx // cols >= rows:
                break
            
            # Only highlight slices that show overlay
            if not show_overlay:
                continue
                
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
                row_i = display_idx // cols
                col_i = display_idx % cols
                y0, y1 = row_i * scaled_slice_h, (row_i + 1) * scaled_slice_h
                x0, x1 = col_i * scaled_slice_w, (col_i + 1) * scaled_slice_w
                
                for t in range(border_width):
                    composite_uint8[y0 + t, x0:x1, :] = red_border
                    composite_uint8[y1 - 1 - t, x0:x1, :] = red_border
                    composite_uint8[y0:y1, x0 + t, :] = red_border
                    composite_uint8[y0:y1, x1 - 1 - t, :] = red_border

    # Add L/R labels
    left_label, right_label = generate_lr_labels()
    pad = max(5, int(scale * 10))
    label_scale = min(1.5, scale * 2)
    
    scaled_l_h = int(left_label.shape[0] * label_scale)
    scaled_l_w = int(left_label.shape[1] * label_scale)
    scaled_r_h = int(right_label.shape[0] * label_scale)
    scaled_r_w = int(right_label.shape[1] * label_scale)
    
    if scaled_l_h > 0 and scaled_l_w > 0:
        left_label_scaled = resize(left_label, (scaled_l_h, scaled_l_w), order=0, anti_aliasing=False, preserve_range=True)
    else:
        left_label_scaled = left_label
        
    if scaled_r_h > 0 and scaled_r_w > 0:
        right_label_scaled = resize(right_label, (scaled_r_h, scaled_r_w), order=0, anti_aliasing=False, preserve_range=True)
    else:
        right_label_scaled = right_label
    
    if orientation == 'L-R':
        if pad + scaled_l_h < canvas_h and pad + scaled_l_w < canvas_w:
            composite_uint8[pad:pad + scaled_l_h, pad:pad + scaled_l_w, :] = \
                np.maximum(composite_uint8[pad:pad + scaled_l_h, pad:pad + scaled_l_w, :],
                          left_label_scaled[..., None] * 0.9)
        
        if pad + scaled_r_h < canvas_h and canvas_w - pad - scaled_r_w > 0:
            composite_uint8[pad:pad + scaled_r_h, -pad - scaled_r_w:-pad if pad > 0 else None, :] = \
                np.maximum(composite_uint8[pad:pad + scaled_r_h, -pad - scaled_r_w:-pad if pad > 0 else None, :],
                          right_label_scaled[..., None] * 0.9)
    else:  # R-L
        if pad + scaled_r_h < canvas_h and pad + scaled_r_w < canvas_w:
            composite_uint8[pad:pad + scaled_r_h, pad:pad + scaled_r_w, :] = \
                np.maximum(composite_uint8[pad:pad + scaled_r_h, pad:pad + scaled_r_w, :],
                          right_label_scaled[..., None] * 0.9)
        
        if pad + scaled_l_h < canvas_h and canvas_w - pad - scaled_l_w > 0:
            composite_uint8[pad:pad + scaled_l_h, -pad - scaled_l_w:-pad if pad > 0 else None, :] = \
                np.maximum(composite_uint8[pad:pad + scaled_l_h, -pad - scaled_l_w:-pad if pad > 0 else None, :],
                          left_label_scaled[..., None] * 0.9)

    # Save
    final_image = Image.fromarray(composite_uint8)
    final_image.save(output_file, format='JPEG', quality=95)
    print(f"[NeuroMontage] Mosaic saved to {output_file}")
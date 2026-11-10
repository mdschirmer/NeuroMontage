"""
3D mosaic creation for NeuroMontage.
"""

import numpy as np
from matplotlib import colormaps
from PIL import Image
from skimage.transform import resize

from layout_utils import (RESOLUTION_PRESETS, MAX_WIDTH, MAX_HEIGHT_3D, MIN_SLICE_SIZE,
                          calculate_auto_resolution_and_sampling,
                          calculate_optimal_layout)
from image_utils import (get_slice, generate_orientation_labels, display_layout, to_display,
                         resolve_outline_styles,
                         calculate_robust_normalization, calculate_overlay_colors, load_nifti_robust,
                         fill_non_finite, slice_has_content, save_mosaic, prepare_thresholds,
                         draw_contours, draw_border, segmentation_in_slice)

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
    colormap_name=None,
    log_scale=False,
    start_slice=None,
    end_slice=None,
    outline=False,
    highlight=False,
    alternate=False,
    rows=None,
    outline_colors=None,
    line_styles=None
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
    # Load brain data with robust error handling
    brain_nii, brain_data_full, brain_rgb = load_nifti_robust(brain_file, return_rgb=True)
    brain_data_full = fill_non_finite(brain_data_full, 'Brain')
    
    # Show the slices in their own plane (axial, coronal, sagittal), upright
    layout = display_layout(brain_data_full.shape, brain_nii.affine)
    brain_data_full = to_display(brain_data_full, layout)
    if brain_rgb is not None:
        brain_rgb = to_display(brain_rgb, layout)
    print(f"[NeuroMontage] Showing {layout['plane']} slices")
    
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

        thresholds = prepare_thresholds(thresholds, num_inputs)

        if not outline_mode:
            _, cum_data_full = load_nifti_robust(input_files[0])
            cum_data_full = to_display(fill_non_finite(cum_data_full, 'Overlay', ceiling=thresholds[0]), layout)
            if brain_data_full.shape != cum_data_full.shape:
                raise ValueError("Structural and cumulative segmentation volumes must match in shape.")
        else:
            all_masks = []
            for m_i, seg_path in enumerate(input_files):
                _, seg_data = load_nifti_robust(seg_path)
                seg_data = to_display(fill_non_finite(seg_data, f'Mask {m_i + 1}', ceiling=thresholds[m_i]), layout)
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

    # Build list of slices to use
    valid_slices = [
        idx for idx in range(start_slice, end_slice + 1)
        if slice_has_content(brain_data_full[:, :, idx])
    ]
    if not valid_slices:
        # Empty image: show the slices as they are (one uniform color)
        print(f"[NeuroMontage] Warning: no slices with image content between indices {start_slice} and "
              f"{end_slice}: every slice is constant (e.g. all 0 or all NaN). Image shape: "
              f"{brain_data_full.shape}. Saving the slices as they are.")
        valid_slices = list(range(start_slice, end_slice + 1))
    
    # Apply user-specified slice_step first
    used_slices = valid_slices[::slice_step]

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
        # In alternate mode plan with slice pairs (one tile per slice, twice as tall)
        target_width, target_height, adaptive_step, final_display_count = \
            calculate_auto_resolution_and_sampling(
                len(used_slices) if alternate else display_count,
                slice_h=2 * slice_h if alternate else slice_h,
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
    
    # Translate the user row count into a forced layout row count
    force_rows = None
    if rows is not None:
        if alternate:
            # Alternate mode emits row pairs. Round an odd request down.
            pairs = rows // 2
            if rows % 2 != 0:
                print(f"[NeuroMontage] Warning: --rows must be even in alternate mode. Rounding down to {2 * pairs}.")
            if pairs < 1:
                print("[NeuroMontage] Warning: alternate mode needs at least 2 rows. Using 2 rows.")
                pairs = 1
            if pairs > len(used_slices):
                pairs = len(used_slices)
                print(f"[NeuroMontage] Warning: --rows exceeds slice count. Using {2 * pairs} rows.")
            force_rows = pairs
        else:
            force_rows = min(rows, display_count)
            if force_rows < rows:
                print(f"[NeuroMontage] Warning: --rows ({rows}) exceeds slice count. Using {force_rows} rows.")
        if resolution == 'auto':
            # Bound the forced layout by the maximum canvas, not the auto layout target.
            target_width, target_height = MAX_WIDTH, MAX_HEIGHT_3D

    # Preset resolutions: the frame wins over the minimum slice size
    is_preset = resolution != 'auto'
    min_slice_pixels = 0 if is_preset else MIN_SLICE_SIZE

    # Calculate optimal layout. In alternate mode each slice is shown twice,
    # one above the other (brain-only row + brain+overlay row), so lay out
    # pairs: one tile per slice, twice as tall. force_rows counts pairs there.
    if alternate:
        cols, pair_rows, scale, actual_width, actual_height = calculate_optimal_layout(
            len(used_slices), 2 * slice_h, slice_w, target_width, target_height,
            min_slice_pixels=min_slice_pixels, force_rows=force_rows, allow_upscale=is_preset
        )
        rows = 2 * pair_rows
    else:
        cols, rows, scale, actual_width, actual_height = calculate_optimal_layout(
            display_count, slice_h, slice_w, target_width, target_height,
            min_slice_pixels=min_slice_pixels, force_rows=force_rows, allow_upscale=is_preset
        )

    print(f"[NeuroMontage] Layout: {cols}×{rows} grid")
    small_side = int(min(slice_h, slice_w) * scale)
    if is_preset and small_side < MIN_SLICE_SIZE:
        print(f"[NeuroMontage] Warning: slices are {small_side}px to fit {resolution.upper()}. "
              f"Use --slice_step to show fewer, larger slices.")
    print(f"[NeuroMontage] Output dimensions: {actual_width}×{actual_height} (scale: {scale:.2f}x)")
    if resolution != 'auto':
        print(f"[NeuroMontage] Target was: {target_width}×{target_height} ({resolution.upper()})")
    
    # Apply uniform scale to slice dimensions
    scaled_slice_h = int(slice_h * scale)
    scaled_slice_w = int(slice_w * scale)
    
    # Create canvases
    canvas_h = rows * scaled_slice_h
    canvas_w = cols * scaled_slice_w
    
    # Gray values, or RGB (0-1) for color images
    color_shape = (3,) if brain_rgb is not None else ()
    base_canvas = np.zeros((canvas_h, canvas_w) + color_shape, dtype=np.float32)
    
    if has_segmentation:
        overlay_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        alpha_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        outline_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Robust normalization for brain
    norm_brain = calculate_robust_normalization(brain_data_full)
    
    if has_segmentation and not outline_mode:
        thr0 = thresholds[0]
        norm_overlay, cmap = calculate_overlay_colors(cum_data_full, thr0, colormap_name, log_scale)
    elif has_segmentation and outline_mode:
        contour_colors, contour_patterns = resolve_outline_styles(len(all_masks), outline_colors, line_styles)

    # Build the display sequence with explicit row/column positions
    display_sequence = []
    if alternate:
        current_row = 0
        for chunk_start in range(0, len(used_slices), cols):
            chunk_end = min(chunk_start + cols, len(used_slices))
            chunk_slices = used_slices[chunk_start:chunk_end]
            
            # First row: brain-only for this chunk
            for col_idx, slice_idx in enumerate(chunk_slices):
                display_sequence.append((slice_idx, False, current_row, col_idx))
            current_row += 1
            
            # Second row: brain+overlay for this chunk
            for col_idx, slice_idx in enumerate(chunk_slices):
                display_sequence.append((slice_idx, True, current_row, col_idx))
            current_row += 1
    else:
        for display_idx, slice_idx in enumerate(used_slices):
            row_i = display_idx // cols
            col_i = display_idx % cols
            display_sequence.append((slice_idx, has_segmentation, row_i, col_i))

    # Process each display slot
    for entry in display_sequence:
        slice_idx, show_overlay, row_i, col_i = entry
        
        if row_i >= rows:
            break
        
        y0, y1 = row_i * scaled_slice_h, (row_i + 1) * scaled_slice_h
        x0, x1 = col_i * scaled_slice_w, (col_i + 1) * scaled_slice_w

        # Get and scale structural slice
        if brain_rgb is not None:
            base_slice_norm = get_slice(brain_rgb, slice_idx) / 255.0
        else:
            base_slice_norm = norm_brain(get_slice(brain_data_full, slice_idx))
        
        if scale != 1.0:
            base_slice_scaled = resize(base_slice_norm, (scaled_slice_h, scaled_slice_w), 
                                      order=1 if scale < 1.0 else 0, anti_aliasing=scale < 1.0,
                                      preserve_range=True)
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
                # Outline mode - draw masks in order (solid first, dashed, dotted on top)
                for m_i, mask_vol in enumerate(all_masks):
                    seg_slice = get_slice(mask_vol, slice_idx)
                    thr_i = thresholds[m_i]
                    
                    if scale != 1.0:
                        seg_slice_scaled = resize(seg_slice, (scaled_slice_h, scaled_slice_w),
                                                order=0, anti_aliasing=False, preserve_range=True)
                    else:
                        seg_slice_scaled = seg_slice
                    
                    binary_slice = (seg_slice_scaled > thr_i).astype(np.uint8)
                    # 2-pixel thick line to survive JPEG compression
                    draw_contours(outline_canvas[y0:y1, x0:x1], binary_slice,
                                  contour_colors[m_i], contour_patterns[m_i])

    # Build final composite
    if brain_rgb is not None:
        rgb_base = base_canvas
    else:
        rgb_base = colormaps['gray'](base_canvas)[..., :3]
    if has_segmentation and not outline_mode:
        rgb_overlay = cmap(overlay_canvas)[..., :3]
        composite = (rgb_overlay * alpha_mask[..., None] +
                     rgb_base * (1.0 - alpha_mask[..., None]))
        composite_uint8 = (composite * 255).astype(np.uint8)
    elif has_segmentation and outline_mode:
        composite_uint8 = (rgb_base * 255).astype(np.uint8)
        mask_idx = np.any(outline_canvas > 0, axis=2)
        composite_uint8[mask_idx] = outline_canvas[mask_idx]
    else:
        composite_uint8 = (rgb_base * 255).astype(np.uint8)

    # Highlight slices with segmentation
    if highlight and has_segmentation:
        red_border = np.array([255, 0, 0], dtype=np.uint8)
        border_width = max(2, int(scale * 2))
        seg_volumes = all_masks if outline_mode else [cum_data_full]
        
        for entry in display_sequence:
            slice_idx, show_overlay, row_i, col_i = entry
            
            if row_i >= rows:
                break
            
            # Only highlight slices that show overlay
            if not show_overlay:
                continue
                
            if segmentation_in_slice(seg_volumes, thresholds, slice_idx):
                y0, y1 = row_i * scaled_slice_h, (row_i + 1) * scaled_slice_h
                x0, x1 = col_i * scaled_slice_w, (col_i + 1) * scaled_slice_w
                draw_border(composite_uint8[y0:y1, x0:x1], border_width, red_border)

    # Add orientation labels (L/R, or A/P for sagittal slices)
    left_label, right_label = generate_orientation_labels(layout)
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
    
    if pad + scaled_l_h < canvas_h and pad + scaled_l_w < canvas_w:
        composite_uint8[pad:pad + scaled_l_h, pad:pad + scaled_l_w, :] = \
            np.maximum(composite_uint8[pad:pad + scaled_l_h, pad:pad + scaled_l_w, :],
                      left_label_scaled[..., None] * 0.9)
    
    if pad + scaled_r_h < canvas_h and canvas_w - pad - scaled_r_w > 0:
        composite_uint8[pad:pad + scaled_r_h, -pad - scaled_r_w:-pad if pad > 0 else None, :] = \
            np.maximum(composite_uint8[pad:pad + scaled_r_h, -pad - scaled_r_w:-pad if pad > 0 else None, :],
                      right_label_scaled[..., None] * 0.9)

    # Preset resolution: center the mosaic on a black canvas of exactly that size
    if resolution != 'auto':
        out_h, out_w = target_height, target_width
        if canvas_h <= out_h and canvas_w <= out_w:
            framed = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            y_off = (out_h - canvas_h) // 2
            x_off = (out_w - canvas_w) // 2
            framed[y_off:y_off + canvas_h, x_off:x_off + canvas_w] = composite_uint8
            composite_uint8 = framed
        else:
            print(f"[NeuroMontage] Warning: mosaic ({canvas_w}×{canvas_h}) is larger than "
                  f"{resolution.upper()}. Not padded to {out_w}×{out_h}.")

    # Save - use format based on extension
    fine_lines = has_segmentation and outline_mode
    save_mosaic(Image.fromarray(composite_uint8), output_file, fine_lines=fine_lines)
    print(f"[NeuroMontage] Mosaic saved to {output_file}")
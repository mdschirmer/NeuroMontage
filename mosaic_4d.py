"""
4D super mosaic creation for NeuroMontage.
Each volume is displayed in a single row, with optional segmentation overlays.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from PIL import Image
from skimage.transform import resize

from layout_utils import (RESOLUTION_PRESETS, calculate_auto_resolution_and_sampling, 
                          calculate_optimal_layout)
from image_utils import (get_slice, find_contours_on_slice, generate_lr_labels, 
                         get_orientation, get_contour_colors, should_draw_pixel,
                         calculate_robust_normalization,
                         calculate_median_excluding_background, load_nifti_robust)

# Increase PIL's image size limit
Image.MAX_IMAGE_PIXELS = None


def create_super_mosaic_4d(
    brain_file,
    output_file,
    input_files=None,
    resolution='auto',
    thresholds=None,
    alpha=0.7,
    colormap_name='viridis',
    log_scale=False,
    outline=False,
    highlight=False,
    alternate=False
):
    """
    Create a "super mosaic" for 4D data where each volume is displayed in one row.
    All slices from a volume are arranged horizontally in that volume's row.
    Each row has an intensity indicator bar on the right showing where that volume's
    median falls relative to the range of medians across all volumes.
    
    Args:
        brain_file: Path to 4D brain NIfTI file
        output_file: Output file path (.jpg)
        input_files: Optional list of segmentation NIfTI files (3D or 4D)
        resolution: 'auto', 'hd', '2k', or '4k'
        thresholds: Threshold value(s) for segmentation
        alpha: Overlay transparency (0-1, for single-input overlay mode)
        colormap_name: Matplotlib colormap for overlay
        log_scale: Use logarithmic normalization for overlay
        outline: Force outline mode even with single input file
        highlight: Draw red border around slices containing segmentation
        alternate: Alternate rows between brain-only and brain+overlay
    
    Segmentation Modes:
        - No input_files: Brain-only visualization
        - Single 3D file (no --outline): Colormap overlay applied to all volumes
        - Single 4D file (no --outline): Per-volume colormap overlay
        - Multiple files or --outline: Colored contour outlines
        - 3D segmentation with 4D brain: Same mask applied to all volumes
        - 4D segmentation with 4D brain: Per-volume masks (must match volume count)
    """
    brain_nii, brain_data_4d = load_nifti_robust(brain_file)
    
    if len(brain_data_4d.shape) != 4:
        raise ValueError("create_super_mosaic_4d only supports 4D data.")
    
    slice_h = brain_data_4d.shape[0]
    slice_w = brain_data_4d.shape[1]
    n_slices_per_volume = brain_data_4d.shape[2]
    n_volumes = brain_data_4d.shape[3]
    
    print(f"[NeuroMontage] 4D volume: {n_volumes} volumes, {n_slices_per_volume} slices each ({slice_h}x{slice_w})")
    
    # Determine segmentation mode
    has_segmentation = input_files is not None and len(input_files) > 0
    
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
        
        # Load and validate segmentation data
        if not outline_mode:
            # Single overlay mode
            _, seg_data_full = load_nifti_robust(input_files[0])
            
            seg_is_4d = len(seg_data_full.shape) == 4
            
            if seg_is_4d:
                if seg_data_full.shape[:3] != brain_data_4d.shape[:3]:
                    raise ValueError("4D segmentation spatial dimensions must match brain volume.")
                if seg_data_full.shape[3] != n_volumes:
                    raise ValueError(f"4D segmentation has {seg_data_full.shape[3]} volumes but brain has {n_volumes}.")
                print(f"[NeuroMontage] 4D segmentation: per-volume overlay")
            else:
                if seg_data_full.shape != brain_data_4d.shape[:3]:
                    raise ValueError("3D segmentation dimensions must match brain volume spatial dimensions.")
                print(f"[NeuroMontage] 3D segmentation: same overlay for all volumes")
        else:
            # Outline mode - load all masks
            all_masks = []
            masks_are_4d = []
            
            for seg_path in input_files:
                _, seg_data = load_nifti_robust(seg_path)
                
                is_4d = len(seg_data.shape) == 4
                masks_are_4d.append(is_4d)
                
                if is_4d:
                    if seg_data.shape[:3] != brain_data_4d.shape[:3]:
                        raise ValueError(f"4D segmentation {seg_path} spatial dimensions must match brain volume.")
                    if seg_data.shape[3] != n_volumes:
                        raise ValueError(f"4D segmentation {seg_path} has {seg_data.shape[3]} volumes but brain has {n_volumes}.")
                else:
                    if seg_data.shape != brain_data_4d.shape[:3]:
                        raise ValueError(f"3D segmentation {seg_path} dimensions must match brain volume spatial dimensions.")
                
                all_masks.append(seg_data)
            
            contour_colors = get_contour_colors()
            num_colors = len(contour_colors)
            print(f"[NeuroMontage] Outline mode: {len(all_masks)} mask(s)")
    else:
        outline_mode = False
        single_file = False
    
    # Calculate rows needed (double if alternate mode with segmentation)
    if alternate and has_segmentation:
        display_rows = n_volumes * 2
        print(f"[NeuroMontage] Alternate mode: {n_volumes} volumes × 2 rows = {display_rows} total rows")
    else:
        display_rows = n_volumes
    
    # Determine target resolution and adaptive sampling
    if resolution == 'auto':
        total_slices = display_rows * n_slices_per_volume
        target_width, target_height, slice_step, actual_total_slices = \
            calculate_auto_resolution_and_sampling(
                total_slices,
                slice_h=slice_h,
                slice_w=slice_w,
                is_4d=True,
                n_volumes=display_rows,
                n_slices_per_volume=n_slices_per_volume
            )
        
        # Update slice count if adaptive sampling recommended
        sampled_slices_per_volume = n_slices_per_volume
        if slice_step > 1:
            sampled_slices_per_volume = len(range(0, n_slices_per_volume, slice_step))
            print(f"[NeuroMontage] Using {sampled_slices_per_volume} slices per volume")
        
        print(f"[NeuroMontage] Auto resolution: {target_width}×{target_height}")
    else:
        slice_step = 1
        sampled_slices_per_volume = n_slices_per_volume
        target_width, target_height = RESOLUTION_PRESETS[resolution]
        print(f"[NeuroMontage] Using preset {resolution.upper()} resolution: {target_width}×{target_height}")
    
    # Intensity bar parameters
    bar_width = 20
    gap = 15
    
    # Calculate scale to fit
    available_width = target_width - gap - bar_width
    
    if resolution == 'auto':
        scale_w = available_width / (sampled_slices_per_volume * slice_w)
        scale_h = target_height / (display_rows * slice_h)
        scale = min(scale_w, scale_h)
        scale = min(1.0, scale)  # Never upscale
    else:
        scale_w = available_width / (sampled_slices_per_volume * slice_w)
        scale_h = target_height / (display_rows * slice_h)
        scale = min(scale_w, scale_h)
    
    # Apply uniform scale
    scaled_slice_h = int(slice_h * scale)
    scaled_slice_w = int(slice_w * scale)
    
    # Calculate final canvas dimensions
    slices_width = scaled_slice_w * sampled_slices_per_volume
    canvas_width = slices_width + gap + bar_width
    canvas_height = scaled_slice_h * display_rows
    
    print(f"[NeuroMontage] Layout: {sampled_slices_per_volume} slices × {display_rows} rows (+ intensity indicators)")
    print(f"[NeuroMontage] Output dimensions: {canvas_width}×{canvas_height} (scale: {scale:.2f}x)")
    if resolution != 'auto':
        print(f"[NeuroMontage] Target was: {target_width}×{target_height} ({resolution.upper()})")
    
    # Get orientation
    orientation = get_orientation(brain_nii.affine)
    
    # Calculate volume medians for intensity indicators
    volume_medians = []
    for vol_idx in range(n_volumes):
        volume_data = brain_data_4d[:, :, :, vol_idx]
        threshold_5pct = np.percentile(np.unique(volume_data), 5)
        brain_voxels = volume_data[volume_data > threshold_5pct]
        if brain_voxels.size > 0:
            vol_median_original = np.median(brain_voxels)
        else:
            vol_median_original = np.median(volume_data)
        volume_medians.append(vol_median_original)
    
    median_min = min(volume_medians)
    median_max = max(volume_medians)
    print(f"[NeuroMontage] Median intensity range across volumes: {median_min:.1f} - {median_max:.1f}")
    
    # Prepare overlay normalization if needed
    if has_segmentation and not outline_mode:
        thr0 = thresholds[0]
        if log_scale:
            # Find positive values for log scaling
            if seg_is_4d:
                pos_vals = seg_data_full[seg_data_full > thr0]
            else:
                pos_vals = seg_data_full[seg_data_full > thr0]
            if pos_vals.size == 0:
                raise ValueError("No positive segmentation values above threshold for log scaling.")
            norm_overlay = LogNorm(vmin=pos_vals.min(), vmax=pos_vals.max())
        else:
            norm_overlay = calculate_robust_normalization(seg_data_full)
        cmap = plt.get_cmap(colormap_name)
    
    # Create RGB canvas
    canvas_rgb = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    
    # Build display sequence
    display_sequence = []
    if alternate and has_segmentation:
        for vol_idx in range(n_volumes):
            display_sequence.append((vol_idx, False))  # Brain-only row
            display_sequence.append((vol_idx, True))   # Brain+overlay row
    else:
        for vol_idx in range(n_volumes):
            display_sequence.append((vol_idx, has_segmentation))
    
    # Render each row
    for row_idx, (vol_idx, show_overlay) in enumerate(display_sequence):
        volume_data = brain_data_4d[:, :, :, vol_idx]
        
        # Per-volume normalization for brain
        vol_min = np.percentile(volume_data, 5)
        vol_max = np.percentile(volume_data, 99.99)
        if vol_max > vol_min:
            vol_norm = Normalize(vmin=vol_min, vmax=vol_max)
        else:
            vol_norm = Normalize(vmin=vol_min, vmax=vol_min + 1)
        
        if show_overlay:
            mode_str = "brain+overlay" if not outline_mode else "brain+outline"
        else:
            mode_str = "brain-only"
        print(f"[NeuroMontage] Row {row_idx + 1}: Vol {vol_idx + 1} ({mode_str}), median={volume_medians[vol_idx]:.1f}")
        
        # Get segmentation data for this volume
        if has_segmentation and show_overlay:
            if not outline_mode:
                if seg_is_4d:
                    seg_volume = seg_data_full[:, :, :, vol_idx]
                else:
                    seg_volume = seg_data_full
        
        # Calculate row position
        row_y_start = row_idx * scaled_slice_h
        row_y_end = row_y_start + scaled_slice_h
        
        # Process each slice
        for slice_count, slice_idx in enumerate(range(0, brain_data_4d.shape[2], slice_step)):
            if slice_count >= sampled_slices_per_volume:
                break
            
            # Get and normalize brain slice
            brain_slice = get_slice(volume_data, slice_idx)
            
            if scale != 1.0:
                brain_slice_scaled = resize(
                    brain_slice,
                    (scaled_slice_h, scaled_slice_w),
                    order=1,
                    anti_aliasing=True,
                    preserve_range=True
                )
            else:
                brain_slice_scaled = brain_slice
            
            brain_slice_norm = vol_norm(brain_slice_scaled)
            rgb_brain = plt.cm.gray(brain_slice_norm)[..., :3]
            
            # Calculate column position
            col_x_start = slice_count * scaled_slice_w
            col_x_end = col_x_start + scaled_slice_w
            
            # Apply segmentation if needed
            if has_segmentation and show_overlay:
                if not outline_mode:
                    # Colormap overlay mode
                    seg_slice = get_slice(seg_volume, slice_idx)
                    
                    if scale != 1.0:
                        seg_slice_scaled = resize(
                            seg_slice,
                            (scaled_slice_h, scaled_slice_w),
                            order=0,
                            anti_aliasing=False,
                            preserve_range=True
                        )
                    else:
                        seg_slice_scaled = seg_slice
                    
                    thr0 = thresholds[0]
                    alpha_mask = np.where(seg_slice_scaled > thr0, alpha, 0.0)
                    seg_slice_norm = norm_overlay(seg_slice_scaled)
                    rgb_overlay = cmap(seg_slice_norm)[..., :3]
                    
                    # Blend
                    rgb_slice = (rgb_overlay * alpha_mask[..., None] +
                                rgb_brain * (1.0 - alpha_mask[..., None]))
                else:
                    # Outline mode
                    rgb_slice = rgb_brain.copy()
                    
                    for m_i, mask_data in enumerate(all_masks):
                        # Get appropriate volume from mask
                        if masks_are_4d[m_i]:
                            mask_volume = mask_data[:, :, :, vol_idx]
                        else:
                            mask_volume = mask_data
                        
                        seg_slice = get_slice(mask_volume, slice_idx)
                        
                        if scale != 1.0:
                            seg_slice_scaled = resize(
                                seg_slice,
                                (scaled_slice_h, scaled_slice_w),
                                order=0,
                                anti_aliasing=False,
                                preserve_range=True
                            )
                        else:
                            seg_slice_scaled = seg_slice
                        
                        thr_i = thresholds[m_i]
                        binary_slice = (seg_slice_scaled > thr_i).astype(np.uint8)
                        contours = find_contours_on_slice(binary_slice)
                        color = contour_colors[m_i % num_colors] / 255.0
                        
                        for contour in contours:
                            for pt_idx, (r, c) in enumerate(contour):
                                if not should_draw_pixel(pt_idx, m_i):
                                    continue
                                rr = int(round(r))
                                cc = int(round(c))
                                # Draw 2-pixel thick line to survive JPEG compression
                                for dr in (0, 1):
                                    for dc in (0, 1):
                                        rrr = rr + dr
                                        ccc = cc + dc
                                        if 0 <= rrr < scaled_slice_h and 0 <= ccc < scaled_slice_w:
                                            rgb_slice[rrr, ccc, :] = color
            else:
                rgb_slice = rgb_brain
            
            # Place slice on canvas
            canvas_rgb[row_y_start:row_y_end, col_x_start:col_x_end, :] = rgb_slice
            
            # Highlight slices with segmentation
            if highlight and has_segmentation and show_overlay:
                has_seg_in_slice = False
                
                if not outline_mode:
                    if seg_is_4d:
                        seg_slice_check = seg_data_full[:, :, slice_idx, vol_idx]
                    else:
                        seg_slice_check = seg_data_full[:, :, slice_idx]
                    if np.any(seg_slice_check > thresholds[0]):
                        has_seg_in_slice = True
                else:
                    for m_i, mask_data in enumerate(all_masks):
                        if masks_are_4d[m_i]:
                            seg_slice_check = mask_data[:, :, slice_idx, vol_idx]
                        else:
                            seg_slice_check = mask_data[:, :, slice_idx]
                        if np.any(seg_slice_check > thresholds[m_i]):
                            has_seg_in_slice = True
                            break
                
                if has_seg_in_slice:
                    red_border = np.array([1.0, 0.0, 0.0])
                    border_width = max(2, int(scale * 2))
                    
                    for t in range(border_width):
                        canvas_rgb[row_y_start + t, col_x_start:col_x_end, :] = red_border
                        canvas_rgb[row_y_end - 1 - t, col_x_start:col_x_end, :] = red_border
                        canvas_rgb[row_y_start:row_y_end, col_x_start + t, :] = red_border
                        canvas_rgb[row_y_start:row_y_end, col_x_end - 1 - t, :] = red_border
        
        # Add L/R labels for this row
        add_lr_labels_to_row(
            canvas_rgb,
            row_y_start,
            row_y_end,
            slices_width,
            scaled_slice_w,
            sampled_slices_per_volume,
            orientation,
            scale
        )
    
    # Convert to uint8
    canvas_uint8 = (canvas_rgb * 255).astype(np.uint8)
    
    # Add intensity indicator bars
    bar_x_start = slices_width + gap
    bar_x_end = bar_x_start + bar_width
    
    for row_idx, (vol_idx, show_overlay) in enumerate(display_sequence):
        row_y_start = row_idx * scaled_slice_h
        row_y_end = row_y_start + scaled_slice_h
        
        # Create grayscale gradient bar
        gradient = np.linspace(0, 255, scaled_slice_h)[::-1, np.newaxis]
        gradient = np.tile(gradient, (1, bar_width))
        
        canvas_uint8[row_y_start:row_y_end, bar_x_start:bar_x_end, :] = gradient[..., np.newaxis]
        
        # Add orange indicator line
        vol_median_original = volume_medians[vol_idx]
        
        if median_max > median_min:
            normalized_position = (vol_median_original - median_min) / (median_max - median_min)
        else:
            normalized_position = 0.5
        
        normalized_position = np.clip(normalized_position, 0, 1)
        indicator_y = row_y_start + int((1 - normalized_position) * scaled_slice_h)
        indicator_y = np.clip(indicator_y, row_y_start + 1, row_y_end - 2)
        
        orange = np.array([255, 165, 0], dtype=np.uint8)
        line_thickness = 2
        for offset in range(-line_thickness, line_thickness + 1):
            y_pos = indicator_y + offset
            if row_y_start <= y_pos < row_y_end:
                canvas_uint8[y_pos, bar_x_start:bar_x_end, :] = orange
    
    # Add volume labels
    fig_dpi = 100
    fig_width = canvas_width / fig_dpi
    fig_height = canvas_height / fig_dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)
    ax.imshow(canvas_uint8)
    
    # Add y-axis labels
    y_ticks = []
    y_labels = []
    for row_idx, (vol_idx, show_overlay) in enumerate(display_sequence):
        y_center = (row_idx + 0.5) * scaled_slice_h
        y_ticks.append(y_center)
        
        if alternate and has_segmentation:
            if show_overlay:
                y_labels.append(f"Vol {vol_idx + 1} +seg")
            else:
                y_labels.append(f"Vol {vol_idx + 1}")
        else:
            y_labels.append(f"Vol {vol_idx + 1}")
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=max(8, int(scale * 8)), ha='right')
    ax.tick_params(axis='y', which='both', left=False, right=False)
    
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"[NeuroMontage] 4D mosaic saved to {output_file}")


def add_lr_labels_to_row(canvas, row_y_start, row_y_end, slices_width, 
                          scaled_slice_w, n_slices, orientation, scale):
    """
    Add L/R orientation labels to a single row.
    
    Args:
        canvas: The canvas array to modify (RGB float)
        row_y_start: Starting y-coordinate of the row
        row_y_end: Ending y-coordinate of the row
        slices_width: Width of the slice area (not including intensity bar)
        scaled_slice_w: Width of each scaled slice
        n_slices: Number of slices in the row
        orientation: 'L-R' or 'R-L'
        scale: Scaling factor used
    """
    left_label, right_label = generate_lr_labels()
    
    # Scale labels appropriately
    label_scale = min(1.0, scale * 1.5)
    scaled_l_h = int(left_label.shape[0] * label_scale)
    scaled_l_w = int(left_label.shape[1] * label_scale)
    scaled_r_h = int(right_label.shape[0] * label_scale)
    scaled_r_w = int(right_label.shape[1] * label_scale)
    
    if scaled_l_h > 0 and scaled_l_w > 0:
        left_label_scaled = resize(
            left_label, 
            (scaled_l_h, scaled_l_w), 
            order=0, 
            anti_aliasing=False, 
            preserve_range=True
        )
    else:
        return
    
    if scaled_r_h > 0 and scaled_r_w > 0:
        right_label_scaled = resize(
            right_label, 
            (scaled_r_h, scaled_r_w), 
            order=0, 
            anti_aliasing=False, 
            preserve_range=True
        )
    else:
        return
    
    # Padding from edges
    pad = max(3, int(scale * 5))
    label_intensity = 0.9
    
    # Calculate the position of the last slice in the row
    last_slice_x_end = n_slices * scaled_slice_w
    
    # Place labels based on orientation
    if orientation == 'L-R':
        # Left label on the left side
        if (row_y_start + pad + scaled_l_h <= row_y_end and 
            pad + scaled_l_w <= slices_width):
            label_region = canvas[row_y_start + pad:row_y_start + pad + scaled_l_h,
                                  pad:pad + scaled_l_w, :]
            label_mask = (left_label_scaled / 255.0 * label_intensity)[..., np.newaxis]
            canvas[row_y_start + pad:row_y_start + pad + scaled_l_h,
                   pad:pad + scaled_l_w, :] = np.maximum(label_region, label_mask)
        
        # Right label on the right side
        if (row_y_start + pad + scaled_r_h <= row_y_end and 
            last_slice_x_end - pad - scaled_r_w >= 0):
            label_region = canvas[row_y_start + pad:row_y_start + pad + scaled_r_h,
                                  last_slice_x_end - pad - scaled_r_w:last_slice_x_end - pad, :]
            label_mask = (right_label_scaled / 255.0 * label_intensity)[..., np.newaxis]
            canvas[row_y_start + pad:row_y_start + pad + scaled_r_h,
                   last_slice_x_end - pad - scaled_r_w:last_slice_x_end - pad, :] = np.maximum(label_region, label_mask)
    else:  # R-L orientation
        # Right label on the left side
        if (row_y_start + pad + scaled_r_h <= row_y_end and 
            pad + scaled_r_w <= slices_width):
            label_region = canvas[row_y_start + pad:row_y_start + pad + scaled_r_h,
                                  pad:pad + scaled_r_w, :]
            label_mask = (right_label_scaled / 255.0 * label_intensity)[..., np.newaxis]
            canvas[row_y_start + pad:row_y_start + pad + scaled_r_h,
                   pad:pad + scaled_r_w, :] = np.maximum(label_region, label_mask)
        
        # Left label on the right side
        if (row_y_start + pad + scaled_l_h <= row_y_end and 
            last_slice_x_end - pad - scaled_l_w >= 0):
            label_region = canvas[row_y_start + pad:row_y_start + pad + scaled_l_h,
                                  last_slice_x_end - pad - scaled_l_w:last_slice_x_end - pad, :]
            label_mask = (left_label_scaled / 255.0 * label_intensity)[..., np.newaxis]
            canvas[row_y_start + pad:row_y_start + pad + scaled_l_h,
                   last_slice_x_end - pad - scaled_l_w:last_slice_x_end - pad, :] = np.maximum(label_region, label_mask)
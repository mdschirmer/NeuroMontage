"""
4D super mosaic creation for NeuroMontage.
Each volume is displayed in a single row.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from skimage.transform import resize

from layout_utils import (RESOLUTION_PRESETS, calculate_auto_resolution_and_sampling, 
                          calculate_optimal_layout)
from image_utils import get_slice, generate_lr_labels, get_orientation, calculate_robust_normalization, calculate_median_excluding_background

# Increase PIL's image size limit
Image.MAX_IMAGE_PIXELS = None


def create_super_mosaic_4d(brain_file, output_file, resolution='auto'):
    """
    Create a "super mosaic" for 4D data where each volume is displayed in one row.
    All slices from a volume are arranged horizontally in that volume's row.
    Each row has an intensity indicator bar on the right showing where that volume's
    median falls relative to the range of medians across all volumes.
    """
    brain_nii = nib.load(brain_file)
    brain_data_4d = brain_nii.get_fdata()
    
    if len(brain_data_4d.shape) != 4:
        raise ValueError("create_super_mosaic_4d only supports 4D data.")
    
    slice_h = brain_data_4d.shape[0]
    slice_w = brain_data_4d.shape[1]
    n_slices_per_volume = brain_data_4d.shape[2]
    n_volumes = brain_data_4d.shape[3]
    
    print(f"[NeuroMontage] 4D volume: {n_volumes} volumes, {n_slices_per_volume} slices each ({slice_h}x{slice_w})")
    
    # Determine target resolution and adaptive sampling
    if resolution == 'auto':
        total_slices = n_volumes * n_slices_per_volume
        target_width, target_height, slice_step, actual_total_slices = \
            calculate_auto_resolution_and_sampling(
                total_slices,
                slice_h=slice_h,
                slice_w=slice_w,
                is_4d=True,
                n_volumes=n_volumes,
                n_slices_per_volume=n_slices_per_volume
            )
        
        # Update slice count if adaptive sampling recommended
        if slice_step > 1:
            n_slices_per_volume = len(range(0, n_slices_per_volume, slice_step))
            print(f"[NeuroMontage] Using {n_slices_per_volume} slices per volume")
        
        print(f"[NeuroMontage] Auto resolution: {target_width}×{target_height}")
    else:
        slice_step = 1  # No adaptive sampling for preset resolutions
        target_width, target_height = RESOLUTION_PRESETS[resolution]
        print(f"[NeuroMontage] Using preset {resolution.upper()} resolution: {target_width}×{target_height}")
    
    # Intensity bar parameters
    bar_width = 20  # Fixed width in pixels
    gap = 15  # Gap between slices and intensity bar
    
    # Calculate scale to fit:
    # - Width: all slices from one volume in a row + gap + bar
    # - Height: n_volumes rows
    available_width = target_width - gap - bar_width
    
    if resolution == 'auto':
        # For auto mode, ensure each slice has good visibility
        # Calculate scale needed to fit
        scale_w = available_width / (n_slices_per_volume * slice_w)
        scale_h = target_height / (n_volumes * slice_h)
        scale = min(scale_w, scale_h)
        
        # Never upscale - cap at 1.0x
        scale = min(1.0, scale)
    else:
        # For preset resolutions, fit within bounds (may downscale significantly)
        scale_w = available_width / (n_slices_per_volume * slice_w)
        scale_h = target_height / (n_volumes * slice_h)
        scale = min(scale_w, scale_h)
    
    # Apply uniform scale
    scaled_slice_h = int(slice_h * scale)
    scaled_slice_w = int(slice_w * scale)
    
    # Calculate final canvas dimensions (including intensity bar area)
    slices_width = scaled_slice_w * n_slices_per_volume
    canvas_width = slices_width + gap + bar_width
    canvas_height = scaled_slice_h * n_volumes
    
    print(f"[NeuroMontage] Layout: {n_slices_per_volume} slices × {n_volumes} rows (+ intensity indicators)")
    print(f"[NeuroMontage] Output dimensions: {canvas_width}×{canvas_height} (scale: {scale:.2f}x)")
    if resolution != 'auto':
        print(f"[NeuroMontage] Target was: {target_width}×{target_height} ({resolution.upper()})")
    
    # Create canvas
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    
    # Get orientation
    orientation = get_orientation(brain_nii.affine)
    
    # Store volume medians (BEFORE normalization) for indicator bars
    # We need to calculate these first to determine the range for the indicator scale
    volume_medians = []
    
    # First pass: calculate all medians
    for vol_idx in range(n_volumes):
        volume_data = brain_data_4d[:, :, :, vol_idx]
        
        # Calculate median from ORIGINAL data, excluding background (bottom 5%)
        threshold_5pct = np.percentile(np.unique(volume_data), 5)
        brain_voxels = volume_data[volume_data > threshold_5pct]
        if brain_voxels.size > 0:
            vol_median_original = np.median(brain_voxels)
        else:
            vol_median_original = np.median(volume_data)  # Fallback
        volume_medians.append(vol_median_original)
    
    # Determine the range for indicator bars (min to max of all medians)
    median_min = min(volume_medians)
    median_max = max(volume_medians)
    print(f"[NeuroMontage] Median intensity range across volumes: {median_min:.1f} - {median_max:.1f}")
    
    # Second pass: render volumes
    for vol_idx in range(n_volumes):
        volume_data = brain_data_4d[:, :, :, vol_idx]
        
        # Per-volume normalization for display only
        vol_min = np.percentile(volume_data, 5)
        vol_max = np.percentile(volume_data, 99.99)
        if vol_max > vol_min:
            vol_norm = Normalize(vmin=vol_min, vmax=vol_max)
        else:
            vol_norm = Normalize(vmin=vol_min, vmax=vol_min + 1)
        
        print(f"[NeuroMontage] Vol {vol_idx + 1}: median={volume_medians[vol_idx]:.1f}, display range=[{vol_min:.1f}, {vol_max:.1f}]")
        
        # Calculate row position for this volume
        row_y_start = vol_idx * scaled_slice_h
        row_y_end = row_y_start + scaled_slice_h
        
        # Process each slice in this volume (with adaptive sampling if needed)
        for slice_count, slice_idx in enumerate(range(0, brain_data_4d.shape[2], slice_step)):
            if slice_count >= n_slices_per_volume:
                break
            
            # Get slice
            image_slice = get_slice(volume_data, slice_idx)
            
            # Scale slice if needed
            if scale != 1.0:
                image_slice = resize(
                    image_slice, 
                    (scaled_slice_h, scaled_slice_w),
                    order=1,  # bilinear interpolation
                    anti_aliasing=True,
                    preserve_range=True
                )
            
            # Normalize per-volume for better visibility
            image_slice_norm = vol_norm(image_slice)
            
            # Calculate column position for this slice
            col_x_start = slice_count * scaled_slice_w
            col_x_end = col_x_start + scaled_slice_w
            
            # Place slice on canvas
            canvas[row_y_start:row_y_end, col_x_start:col_x_end] = image_slice_norm
        
        # Add L/R labels for this row
        add_lr_labels_to_row(
            canvas, 
            row_y_start, 
            row_y_end, 
            slices_width,  # Use slices width, not total canvas width
            scaled_slice_w,
            n_slices_per_volume,
            orientation, 
            scale
        )
    
    # Convert to RGB
    canvas_rgb = plt.cm.gray(canvas)[..., :3]
    canvas_uint8 = (canvas_rgb * 255).astype(np.uint8)
    
    # Add intensity indicator bars
    bar_x_start = slices_width + gap
    bar_x_end = bar_x_start + bar_width
    
    for vol_idx in range(n_volumes):
        row_y_start = vol_idx * scaled_slice_h
        row_y_end = row_y_start + scaled_slice_h
        
        # Create grayscale gradient bar representing MEDIAN intensity range
        # Bottom of bar = median_min (dark), Top of bar = median_max (bright)
        # Map median_min to 0 (black) and median_max to 255 (white)
        gradient = np.linspace(0, 255, scaled_slice_h)[::-1, np.newaxis]  # Reverse so top is bright
        gradient = np.tile(gradient, (1, bar_width))
        
        # Place gradient on canvas
        canvas_uint8[row_y_start:row_y_end, bar_x_start:bar_x_end, :] = gradient[..., np.newaxis]
        
        # Add orange horizontal line indicating this volume's median (from ORIGINAL data)
        vol_median_original = volume_medians[vol_idx]
        
        # Calculate position: where does this ORIGINAL median fall in the median range?
        if median_max > median_min:
            # normalized_position: 0 = median_min (bottom), 1 = median_max (top)
            normalized_position = (vol_median_original - median_min) / (median_max - median_min)
        else:
            normalized_position = 0.5
        
        # Clamp to valid range [0, 1]
        normalized_position = np.clip(normalized_position, 0, 1)
        
        # Calculate y position in pixels
        # normalized_position=1 should be at top (row_y_start)
        # normalized_position=0 should be at bottom (row_y_end)
        indicator_y = row_y_start + int((1 - normalized_position) * scaled_slice_h)
        
        # Clamp to row bounds with small margin
        indicator_y = np.clip(indicator_y, row_y_start + 1, row_y_end - 2)
        
        # Draw orange horizontal line (thicker for visibility)
        orange = np.array([255, 165, 0], dtype=np.uint8)
        line_thickness = 2
        for offset in range(-line_thickness, line_thickness + 1):
            y_pos = indicator_y + offset
            if row_y_start <= y_pos < row_y_end:
                canvas_uint8[y_pos, bar_x_start:bar_x_end, :] = orange
    
    # Add volume labels on the left side
    fig_dpi = 100
    fig_width = canvas_width / fig_dpi
    fig_height = canvas_height / fig_dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)
    ax.imshow(canvas_uint8)
    
    # Add y-axis labels for volumes
    y_ticks = []
    y_labels = []
    for vol_idx in range(n_volumes):
        y_center = (vol_idx + 0.5) * scaled_slice_h
        y_ticks.append(y_center)
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
        canvas: The canvas array to modify
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
            canvas[row_y_start + pad:row_y_start + pad + scaled_l_h,
                   pad:pad + scaled_l_w] = np.maximum(
                canvas[row_y_start + pad:row_y_start + pad + scaled_l_h,
                       pad:pad + scaled_l_w],
                left_label_scaled / 255.0 * label_intensity
            )
        
        # Right label on the right side (at the end of actual slices)
        if (row_y_start + pad + scaled_r_h <= row_y_end and 
            last_slice_x_end - pad - scaled_r_w >= 0):
            canvas[row_y_start + pad:row_y_start + pad + scaled_r_h,
                   last_slice_x_end - pad - scaled_r_w:last_slice_x_end - pad] = np.maximum(
                canvas[row_y_start + pad:row_y_start + pad + scaled_r_h,
                       last_slice_x_end - pad - scaled_r_w:last_slice_x_end - pad],
                right_label_scaled / 255.0 * label_intensity
            )
    else:  # R-L orientation
        # Right label on the left side
        if (row_y_start + pad + scaled_r_h <= row_y_end and 
            pad + scaled_r_w <= slices_width):
            canvas[row_y_start + pad:row_y_start + pad + scaled_r_h,
                   pad:pad + scaled_r_w] = np.maximum(
                canvas[row_y_start + pad:row_y_start + pad + scaled_r_h,
                       pad:pad + scaled_r_w],
                right_label_scaled / 255.0 * label_intensity
            )
        
        # Left label on the right side (at the end of actual slices)
        if (row_y_start + pad + scaled_l_h <= row_y_end and 
            last_slice_x_end - pad - scaled_l_w >= 0):
            canvas[row_y_start + pad:row_y_start + pad + scaled_l_h,
                   last_slice_x_end - pad - scaled_l_w:last_slice_x_end - pad] = np.maximum(
                canvas[row_y_start + pad:row_y_start + pad + scaled_l_h,
                       last_slice_x_end - pad - scaled_l_w:last_slice_x_end - pad],
                left_label_scaled / 255.0 * label_intensity
            )
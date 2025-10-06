"""
4D super mosaic creation for NeuroMontage.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.transform import resize

from layout_utils import RESOLUTION_PRESETS, determine_auto_resolution, calculate_optimal_layout
from image_utils import get_slice, generate_lr_labels, get_orientation


def render_volume_row(volume_data, canvas, volume_index, slice_r, slice_c, max_columns, 
                     rows_per_volume, norm=None, orientation='L-R'):
    """
    Render a single volume's slices in its allocated row space.
    """
    n_slices = volume_data.shape[2]
    
    # Calculate the starting row for this volume
    volume_start_row = volume_index * rows_per_volume
    
    # Generate small L/R labels
    left_label, right_label = generate_lr_labels()
    label_scale = 0.75
    scaled_l_h, scaled_l_w = int(left_label.shape[0] * label_scale), int(left_label.shape[1] * label_scale)
    scaled_r_h, scaled_r_w = int(right_label.shape[0] * label_scale), int(right_label.shape[1] * label_scale)
    
    from skimage.transform import resize as sk_resize
    left_label_small = (sk_resize(left_label, (scaled_l_h, scaled_l_w), order=0, anti_aliasing=False) * 255).astype(np.uint8)
    right_label_small = (sk_resize(right_label, (scaled_r_h, scaled_r_w), order=0, anti_aliasing=False) * 255).astype(np.uint8)
    
    rows_with_slices = set()
    
    for slice_idx in range(n_slices):
        image_slice = get_slice(volume_data, slice_idx)
        
        if norm:
            image_slice = norm(image_slice)
        
        row_within_volume = slice_idx // max_columns
        col_within_row = slice_idx % max_columns
        
        rows_with_slices.add(row_within_volume)
        
        absolute_row = volume_start_row + row_within_volume
        
        r_start = absolute_row * slice_r
        r_end = r_start + slice_r
        c_start = col_within_row * slice_c
        c_end = c_start + slice_c
        
        if (r_end <= canvas.shape[0] and c_end <= canvas.shape[1] and 
            row_within_volume < rows_per_volume):
            canvas[r_start:r_end, c_start:c_end] = image_slice
    
    # Add L/R labels
    pad = 3
    label_intensity = 0.9
    
    for row_within_volume in rows_with_slices:
        absolute_row = volume_start_row + row_within_volume
        r_start = absolute_row * slice_r
        
        slices_in_this_row = [i for i in range(n_slices) if i // max_columns == row_within_volume]
        if not slices_in_this_row:
            continue
        last_slice_in_row = max(slices_in_this_row)
        last_col_with_slice = last_slice_in_row % max_columns
        
        if orientation == 'L-R':
            if pad + scaled_l_h <= slice_r and pad + scaled_l_w <= slice_c:
                canvas[r_start + pad:r_start + pad + scaled_l_h, 
                       pad:pad + scaled_l_w] = np.maximum(
                    canvas[r_start + pad:r_start + pad + scaled_l_h, 
                           pad:pad + scaled_l_w],
                    left_label_small / 255.0 * label_intensity
                )
            
            r_col_start = last_col_with_slice * slice_c
            if r_col_start + slice_c - pad - scaled_r_w >= 0:
                canvas[r_start + pad:r_start + pad + scaled_r_h,
                       r_col_start + slice_c - pad - scaled_r_w:r_col_start + slice_c - pad] = np.maximum(
                    canvas[r_start + pad:r_start + pad + scaled_r_h,
                           r_col_start + slice_c - pad - scaled_r_w:r_col_start + slice_c - pad],
                    right_label_small / 255.0 * label_intensity
                )
        else:  # R-L
            if pad + scaled_r_h <= slice_r and pad + scaled_r_w <= slice_c:
                canvas[r_start + pad:r_start + pad + scaled_r_h,
                       pad:pad + scaled_r_w] = np.maximum(
                    canvas[r_start + pad:r_start + pad + scaled_r_h,
                           pad:pad + scaled_r_w],
                    right_label_small / 255.0 * label_intensity
                )
            
            l_col_start = last_col_with_slice * slice_c
            if l_col_start + slice_c - pad - scaled_l_w >= 0:
                canvas[r_start + pad:r_start + pad + scaled_l_h,
                       l_col_start + slice_c - pad - scaled_l_w:l_col_start + slice_c - pad] = np.maximum(
                    canvas[r_start + pad:r_start + pad + scaled_l_h,
                           l_col_start + slice_c - pad - scaled_l_w:l_col_start + slice_c - pad],
                    left_label_small / 255.0 * label_intensity
                )
    
    return canvas


def create_super_mosaic_4d(brain_file, output_file, resolution='auto'):
    """
    Create a "super mosaic" for 4D data where each volume gets its own row space.
    """
    brain_nii = nib.load(brain_file)
    brain_data_4d = brain_nii.get_fdata()
    
    if len(brain_data_4d.shape) != 4:
        raise ValueError("create_super_mosaic_4d only supports 4D data.")
    
    slice_r = brain_data_4d.shape[0]
    slice_c = brain_data_4d.shape[1]
    n_slices_per_volume = brain_data_4d.shape[2]
    n_volumes = brain_data_4d.shape[3]
    
    # Determine target resolution
    total_slices = n_volumes * n_slices_per_volume
    if resolution == 'auto':
        resolution = determine_auto_resolution(total_slices)
        print(f"[NeuroMontage] Auto-selected {resolution.upper()} resolution for {total_slices} total slices")
    
    target_width, target_height = RESOLUTION_PRESETS[resolution]
    
    # Calculate optimal layout for 4D
    cols, rows, scale, actual_width, actual_height = calculate_optimal_layout(
        total_slices, slice_r, slice_c, target_width, target_height
    )
    
    # Adjust for volume-based layout
    max_columns = min(n_slices_per_volume, cols)
    rows_per_volume = (n_slices_per_volume + max_columns - 1) // max_columns
    total_rows = n_volumes * rows_per_volume
    
    # Recalculate scale based on actual layout
    scale = min(target_width / (max_columns * slice_c), 
               target_height / (total_rows * slice_r))
    
    # Apply scale uniformly
    scaled_slice_r = int(slice_r * scale)
    scaled_slice_c = int(slice_c * scale)
    
    final_height = scaled_slice_r * total_rows
    final_width = scaled_slice_c * max_columns
    canvas = np.zeros((final_height, final_width))
    
    print(f"[NeuroMontage] 4D layout: {max_columns} columns × {total_rows} total rows")
    print(f"[NeuroMontage] Output dimensions: {final_width}×{final_height} (scale: {scale:.2f}x)")
    
    orientation = get_orientation(brain_nii.affine)
    
    global_norm = Normalize(vmin=np.percentile(brain_data_4d, 5), 
                           vmax=np.percentile(brain_data_4d, 99.99))
    
    for vol in range(n_volumes):
        volume_data = brain_data_4d[:, :, :, vol]
        
        if scale != 1.0:
            new_shape = (scaled_slice_c, scaled_slice_r, volume_data.shape[2])
            volume_data = resize(volume_data, new_shape, order=0, anti_aliasing=False, preserve_range=True)
        
        canvas = render_volume_row(volume_data, canvas, vol, scaled_slice_r, scaled_slice_c, 
                                 max_columns, rows_per_volume, norm=global_norm, 
                                 orientation=orientation)
    
    # Render and save
    fig_width = final_width / 100
    fig_height = final_height / 100
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(canvas, cmap='gray', vmin=0, vmax=1)
    
    # Add volume labels
    y_ticks = []
    y_labels = []
    for vol in range(n_volumes):
        volume_start_row = vol * rows_per_volume
        volume_end_row = volume_start_row + rows_per_volume - 1
        volume_center_row = (volume_start_row + volume_end_row) / 2
        
        y_pos = (volume_center_row + 0.5) * scaled_slice_r
        y_ticks.append(y_pos)
        y_labels.append(f"Vol {vol + 1}/{n_volumes}")
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8, ha='right')
    ax.tick_params(axis='y', which='both', left=False, right=False)
    
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, pad_inches=0, bbox_inches='tight')
    plt.close()
    
    print(f"[NeuroMontage] 4D Super mosaic saved to {output_file}")
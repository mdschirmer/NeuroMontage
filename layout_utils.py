"""
Layout and resolution calculation utilities for NeuroMontage.
"""

import numpy as np

# Resolution presets
RESOLUTION_PRESETS = {
    'hd': (1920, 1080),   # 16:9 - Standard HD
    '2k': (2560, 1440),   # 16:9 - QHD/2K  
    '4k': (3840, 2160),   # 16:9 - 4K UHD
}


def determine_auto_resolution(n_slices):
    """
    Automatically determine appropriate resolution based on slice count.
    
    Args:
        n_slices: Number of slices to display
    
    Returns:
        str: Resolution key ('hd', '2k', or '4k')
    """
    if n_slices <= 40:
        return 'hd'
    elif n_slices <= 100:
        return '2k'
    else:
        return '4k'


def calculate_optimal_layout(n_slices, slice_h, slice_w, target_width, target_height, 
                            min_slice_pixels=80, max_slice_pixels=400):
    """
    Calculate optimal grid layout to best fit target resolution.
    Uses uniform scaling to preserve slice aspect ratios.
    
    Args:
        n_slices: Number of slices to arrange
        slice_h: Height of each slice
        slice_w: Width of each slice
        target_width: Target output width
        target_height: Target output height
        min_slice_pixels: Minimum size for each slice in pixels
        max_slice_pixels: Maximum size for each slice in pixels
    
    Returns:
        tuple: (columns, rows, scale_factor, actual_width, actual_height)
    """
    target_aspect = target_width / target_height
    slice_aspect = slice_w / slice_h
    
    best_layout = None
    best_score = -float('inf')
    
    # Calculate reasonable column range
    min_cols = max(1, int(np.sqrt(n_slices * target_aspect / slice_aspect) * 0.5))
    max_cols = min(n_slices, int(np.sqrt(n_slices * target_aspect / slice_aspect) * 2))
    
    for cols in range(min_cols, max_cols + 1):
        rows = int(np.ceil(n_slices / cols))
        
        # Calculate uniform scale to fit in target resolution
        scale_w = target_width / (cols * slice_w)
        scale_h = target_height / (rows * slice_h)
        scale = min(scale_w, scale_h)  # Use smaller to ensure it fits
        
        # Check slice size constraints
        scaled_slice_size = min(slice_w, slice_h) * scale
        if scaled_slice_size < min_slice_pixels:
            continue  # Too small
        if scaled_slice_size > max_slice_pixels:
            scale = max_slice_pixels / min(slice_w, slice_h)
        
        # Calculate actual dimensions with uniform scale
        actual_width = int(cols * slice_w * scale)
        actual_height = int(rows * slice_h * scale)
        
        # Score based on resolution utilization and aspect ratio match
        area_utilization = (actual_width * actual_height) / (target_width * target_height)
        aspect_match = 1 - abs((actual_width/actual_height) - target_aspect) / target_aspect
        
        # Prefer layouts that fill more of the target area and match aspect ratio
        score = area_utilization * aspect_match
        
        # Slight preference for more square-ish individual slice arrangements
        squareness = 1 - abs(rows - cols) / max(rows, cols)
        score *= (0.8 + 0.2 * squareness)
        
        if score > best_score:
            best_score = score
            best_layout = (cols, rows, scale, actual_width, actual_height)
    
    if best_layout is None:
        # Fallback if no layout meets constraints
        cols = int(np.sqrt(n_slices * target_aspect / slice_aspect))
        rows = int(np.ceil(n_slices / cols))
        scale = min(target_width / (cols * slice_w), target_height / (rows * slice_h))
        actual_width = int(cols * slice_w * scale)
        actual_height = int(rows * slice_h * scale)
        best_layout = (cols, rows, scale, actual_width, actual_height)
    
    return best_layout
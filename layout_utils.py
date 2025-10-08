"""
Layout and resolution calculation utilities for NeuroMontage.
"""

import numpy as np

# Resolution presets (only used when explicitly specified)
RESOLUTION_PRESETS = {
    'hd': (1920, 1080),   # 16:9 - Standard HD
    '2k': (2560, 1440),   # 16:9 - QHD/2K  
    '4k': (3840, 2160),   # 16:9 - 4K UHD
}

# Practical limits for output dimensions
MAX_WIDTH = 4000   # Maximum width in pixels
MAX_HEIGHT = 4000  # Maximum height in pixels
MIN_SLICE_SIZE = 100  # Minimum acceptable size for smaller dimension (triggers sampling)


def calculate_auto_resolution_and_sampling(n_slices, slice_h, slice_w, is_4d=False,
                                          n_volumes=1, n_slices_per_volume=None):
    """
    Calculate output resolution AND recommend slice sampling to prevent file size explosion.
    
    This function determines:
    1. Whether all slices can fit with good quality
    2. If not, what sampling step to use (e.g., show every 2nd or 3rd slice)
    3. The optimal output dimensions
    
    Args:
        n_slices: Number of slices to display (before any sampling)
        slice_h: Height of each slice
        slice_w: Width of each slice  
        is_4d: Whether this is 4D data
        n_volumes: Number of volumes (for 4D data)
        n_slices_per_volume: Slices per volume (for 4D data)
    
    Returns:
        tuple: (target_width, target_height, recommended_step, actual_n_slices)
               where recommended_step=1 means use all slices, step=2 means every 2nd, etc.
    """
    recommended_step = 1
    actual_n_slices = n_slices
    
    # Determine starting scale factor
    # NEVER upscale (scale > 1.0) - always use native (1.0x) or downscale if needed
    target_scale = 1.0
    smaller_dim = min(slice_h, slice_w)
    
    if smaller_dim < MIN_SLICE_SIZE:
        print(f"[NeuroMontage] Warning: Native slice size is small ({smaller_dim}px).")
    
    if is_4d:
        # 4D layout: all slices from one volume in a row
        # Width constraint: n_slices_per_volume * slice_w * scale
        required_width = n_slices_per_volume * slice_w * target_scale
        required_height = n_volumes * slice_h * target_scale
        
        if required_width > MAX_WIDTH:
            # Need to either reduce scale or sample slices
            # First check if we can fit by downscaling
            scale_to_fit = MAX_WIDTH / (n_slices_per_volume * slice_w)
            
            if scale_to_fit < 0.5:
                # Downscaling too much (< 0.5x) - better to sample slices instead
                # Calculate how many slices fit at native or moderate downscale
                max_scale_for_sampling = 0.75  # Don't downscale more than 0.75x when sampling
                max_slices_at_scale = int(MAX_WIDTH / (slice_w * max_scale_for_sampling))
                
                if max_slices_at_scale < n_slices_per_volume:
                    # Need to sample slices
                    recommended_step = int(np.ceil(n_slices_per_volume / max_slices_at_scale))
                    actual_slices_per_volume = len(range(0, n_slices_per_volume, recommended_step))
                    actual_n_slices = actual_slices_per_volume * n_volumes
                    
                    print(f"[NeuroMontage] Too many slices for width: showing every {recommended_step} slice(s)")
                    print(f"[NeuroMontage] Reduced from {n_slices_per_volume} to {actual_slices_per_volume} slices per volume")
                    
                    required_width = actual_slices_per_volume * slice_w * max_scale_for_sampling
                    target_scale = max_scale_for_sampling
            else:
                # Can fit with reasonable downscaling (≥ 0.5x)
                target_scale = scale_to_fit
                required_width = MAX_WIDTH
                print(f"[NeuroMontage] Downscaling to {target_scale:.2f}x to fit width")
        
        # Check height constraint
        if required_height > MAX_HEIGHT:
            # Reduce scale to fit height
            scale_reduction = MAX_HEIGHT / required_height
            target_scale *= scale_reduction
            required_width = int(required_width * scale_reduction)
            required_height = MAX_HEIGHT
            print(f"[NeuroMontage] Height constraint: reduced scale to {target_scale:.2f}x")
        
        target_width = int(required_width)
        target_height = int(required_height)
        
    else:
        # 3D layout: flexible grid arrangement
        # Estimate grid dimensions (roughly square layout)
        aspect_ratio = slice_w / slice_h
        cols_estimate = int(np.sqrt(n_slices * aspect_ratio) * 1.2)
        rows_estimate = int(np.ceil(n_slices / cols_estimate))
        
        required_width = cols_estimate * slice_w * target_scale
        required_height = rows_estimate * slice_h * target_scale
        
        # Check if we exceed limits
        exceeds_width = required_width > MAX_WIDTH
        exceeds_height = required_height > MAX_HEIGHT
        
        if exceeds_width or exceeds_height:
            # Calculate how many slices we can fit at target quality
            max_slices_width = int((MAX_WIDTH / (slice_w * target_scale)) ** 2 / aspect_ratio)
            max_slices_height = int((MAX_HEIGHT / (slice_h * target_scale)) ** 2 * aspect_ratio)
            max_slices_at_target = min(max_slices_width, max_slices_height)
            
            if n_slices > max_slices_at_target:
                # Need to sample slices
                recommended_step = int(np.ceil(n_slices / max_slices_at_target))
                actual_n_slices = len(range(0, n_slices, recommended_step))
                
                print(f"[NeuroMontage] Dimension constraint: showing every {recommended_step} slice(s)")
                print(f"[NeuroMontage] Reduced from {n_slices} to {actual_n_slices} slices")
                
                # Recalculate with reduced slice count
                cols_estimate = int(np.sqrt(actual_n_slices * aspect_ratio) * 1.2)
                rows_estimate = int(np.ceil(actual_n_slices / cols_estimate))
                required_width = cols_estimate * slice_w * target_scale
                required_height = rows_estimate * slice_h * target_scale
        
        # Final bounds check - reduce scale if still too large
        if required_width > MAX_WIDTH or required_height > MAX_HEIGHT:
            scale_w = MAX_WIDTH / required_width if required_width > MAX_WIDTH else 1.0
            scale_h = MAX_HEIGHT / required_height if required_height > MAX_HEIGHT else 1.0
            scale_reduction = min(scale_w, scale_h)
            required_width = int(required_width * scale_reduction)
            required_height = int(required_height * scale_reduction)
            print(f"[NeuroMontage] Final scale adjustment: {target_scale * scale_reduction:.2f}x")
        
        target_width = int(required_width)
        target_height = int(required_height)
    
    # Ensure minimum dimensions
    target_width = max(800, target_width)
    target_height = max(600, target_height)
    
    return (target_width, target_height, recommended_step, actual_n_slices)


def calculate_optimal_layout(n_slices, slice_h, slice_w, target_width, target_height, 
                            min_slice_pixels=100, max_slice_pixels=None):
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
        max_slice_pixels: Maximum size for each slice in pixels (None = no limit)
    
    Returns:
        tuple: (columns, rows, scale_factor, actual_width, actual_height)
    """
    if target_width is None or target_height is None:
        target_width = slice_w * int(np.sqrt(n_slices) * 1.5)
        target_height = slice_h * int(np.sqrt(n_slices) * 1.5)
    
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
        
        # Check minimum slice size constraint
        scaled_slice_size = min(slice_w, slice_h) * scale
        if scaled_slice_size < min_slice_pixels:
            continue  # Too small
        
        # Apply maximum constraint if specified
        if max_slice_pixels is not None and scaled_slice_size > max_slice_pixels:
            scale = max_slice_pixels / min(slice_w, slice_h)
        
        # Calculate actual dimensions with uniform scale
        actual_width = int(cols * slice_w * scale)
        actual_height = int(rows * slice_h * scale)
        
        # Score based on resolution utilization and aspect ratio match
        area_utilization = (actual_width * actual_height) / (target_width * target_height)
        aspect_match = 1 - abs((actual_width/actual_height) - target_aspect) / target_aspect
        
        # Prefer layouts that fill more of the target area and match aspect ratio
        score = (area_utilization ** 1.5) * (aspect_match ** 0.8)
        
        # Slight preference for more square-ish grid arrangements
        squareness = 1 - abs(rows - cols) / max(rows, cols)
        score *= (0.85 + 0.15 * squareness)
        
        # Bonus for maintaining good slice detail
        scale_bonus = min(1.0, scale / 2.0)
        score *= (0.9 + 0.1 * scale_bonus)
        
        if score > best_score:
            best_score = score
            best_layout = (cols, rows, scale, actual_width, actual_height)
    
    if best_layout is None:
        # Fallback if no layout meets constraints
        cols = max(1, int(np.sqrt(n_slices * target_aspect / slice_aspect)))
        rows = int(np.ceil(n_slices / cols))
        scale = min(target_width / (cols * slice_w), target_height / (rows * slice_h))
        
        # Ensure minimum scale
        min_scale = min_slice_pixels / min(slice_w, slice_h)
        scale = max(scale, min_scale)
        
        actual_width = int(cols * slice_w * scale)
        actual_height = int(rows * slice_h * scale)
        best_layout = (cols, rows, scale, actual_width, actual_height)
    
    return best_layout
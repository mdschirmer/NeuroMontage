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
# 3D: constrain to 4K display (3840×2160)
# 4D: constrain width, but let height grow (one row per volume)
MAX_WIDTH = 3840   # Maximum width in pixels
MAX_HEIGHT_3D = 2160  # Maximum height for 3D mosaics
MIN_SLICE_SIZE = 100  # Minimum slice dimension (both 3D and 4D)


def max_upscale(slice_h, slice_w):
    """
    Whole-number upscale factor that brings the smaller slice side to MIN_SLICE_SIZE.
    Returns 1 (no upscaling) for slices that are already large enough.
    """
    return max(1, int(np.ceil(MIN_SLICE_SIZE / min(slice_h, slice_w))))


def limit_scale(scale, slice_h, slice_w):
    """
    Limit a fit-to-target scale factor.
    Downscaling is unchanged. Upscaling only uses whole-number factors (so
    nearest-neighbour resizing keeps every voxel the same size) and only up to
    max_upscale(), i.e. only for slices smaller than MIN_SLICE_SIZE.
    """
    if scale <= 1.0:
        return scale
    return float(min(int(scale), max_upscale(slice_h, slice_w)))


def calculate_auto_resolution_and_sampling(n_slices, slice_h, slice_w, is_4d=False,
                                          n_volumes=1, n_slices_per_volume=None, reserved_width=0):
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
        reserved_width: Width kept free next to the slices (for 4D data, e.g. intensity bars)
    
    Returns:
        tuple: (target_width, target_height, recommended_step, actual_n_slices)
               target_width excludes reserved_width
               where recommended_step=1 means use all slices, step=2 means every 2nd, etc.
    """
    recommended_step = 1
    actual_n_slices = n_slices
    
    # Determine starting scale factor
    # Upscale only small slices, by a whole-number factor, to reach MIN_SLICE_SIZE.
    # Plan the layout with the enlarged slice size.
    target_scale = 1.0
    smaller_dim = min(slice_h, slice_w)
    upscale = max_upscale(slice_h, slice_w)
    
    if upscale > 1:
        print(f"[NeuroMontage] Small slices ({smaller_dim}px): enlarging {upscale}x (no interpolation)")
        slice_h, slice_w = slice_h * upscale, slice_w * upscale
    
    if is_4d:
        max_width = MAX_WIDTH - reserved_width
        # 4D layout: all slices from one volume in a row
        # Width constraint: n_slices_per_volume * slice_w * scale
        required_width = n_slices_per_volume * slice_w * target_scale
        required_height = n_volumes * slice_h * target_scale
        
        if required_width > max_width:
            # Need to either reduce scale or sample slices
            # First check if we can fit by downscaling
            scale_to_fit = max_width / (n_slices_per_volume * slice_w)
            
            if scale_to_fit < 0.5:
                # Downscaling too much (< 0.5x) - better to sample slices instead
                # Calculate how many slices fit at native or moderate downscale
                max_scale_for_sampling = 0.75  # Don't downscale more than 0.75x when sampling
                max_slices_at_scale = int(max_width / (slice_w * max_scale_for_sampling))
                
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
                required_width = max_width
                print(f"[NeuroMontage] Downscaling to {target_scale:.2f}x to fit width")
        
        # Check height constraint - ensure each slice is at least MIN_SLICE_SIZE tall
        scaled_slice_h = slice_h * target_scale
        if scaled_slice_h < MIN_SLICE_SIZE:
            # Scale up to meet minimum slice height
            target_scale = MIN_SLICE_SIZE / slice_h
            required_width = int(n_slices_per_volume * slice_w * target_scale)
            required_height = int(n_volumes * slice_h * target_scale)
            print(f"[NeuroMontage] Ensuring minimum slice height ({MIN_SLICE_SIZE}px): scale {target_scale:.2f}x")
        else:
            required_height = int(n_volumes * slice_h * target_scale)
        
        target_width = int(required_width)
        target_height = int(required_height)
        
    else:
        # 3D layout: target 16:9 aspect ratio output
        TARGET_ASPECT = 16 / 9  # ~1.78
        
        # Calculate max slices that fit at native scale
        max_cols = int(MAX_WIDTH / slice_w)
        max_rows = int(MAX_HEIGHT_3D / slice_h)
        max_slices_at_native = max_cols * max_rows
        
        if n_slices > max_slices_at_native:
            # Need to sample slices
            recommended_step = int(np.ceil(n_slices / max_slices_at_native))
            actual_n_slices = len(range(0, n_slices, recommended_step))
            
            print(f"[NeuroMontage] Dimension constraint: showing every {recommended_step} slice(s)")
            print(f"[NeuroMontage] Reduced from {n_slices} to {actual_n_slices} slices")
        else:
            actual_n_slices = n_slices
        
        # Find layout closest to 16:9 aspect ratio
        best_cols = None
        best_rows = None
        best_aspect_diff = float('inf')
        
        for cols in range(1, actual_n_slices + 1):
            rows = int(np.ceil(actual_n_slices / cols))
            
            # Check if it fits
            width_needed = cols * slice_w
            height_needed = rows * slice_h
            
            if width_needed > MAX_WIDTH or height_needed > MAX_HEIGHT_3D:
                continue
            
            # Calculate output aspect ratio
            output_aspect = width_needed / height_needed
            aspect_diff = abs(output_aspect - TARGET_ASPECT)
            
            if aspect_diff < best_aspect_diff:
                best_aspect_diff = aspect_diff
                best_cols = cols
                best_rows = rows
        
        if best_cols is None:
            # Fallback: use max that fits
            best_cols = max_cols
            best_rows = int(np.ceil(actual_n_slices / best_cols))
        
        cols_estimate = best_cols
        rows_estimate = best_rows
        required_width = cols_estimate * slice_w
        required_height = rows_estimate * slice_h
        
        # Final bounds check - reduce scale if still too large
        if required_width > MAX_WIDTH or required_height > MAX_HEIGHT_3D:
            scale_w = MAX_WIDTH / required_width if required_width > MAX_WIDTH else 1.0
            scale_h = MAX_HEIGHT_3D / required_height if required_height > MAX_HEIGHT_3D else 1.0
            scale_reduction = min(scale_w, scale_h)
            required_width = int(required_width * scale_reduction)
            required_height = int(required_height * scale_reduction)
            target_scale = scale_reduction
            print(f"[NeuroMontage] Scaling to fit: {target_scale:.2f}x")
        
        target_width = int(required_width)
        target_height = int(required_height)
    
    return (target_width, target_height, recommended_step, actual_n_slices)


def calculate_optimal_layout(n_slices, slice_h, slice_w, target_width, target_height,
                            min_slice_pixels=100, max_slice_pixels=None, force_rows=None,
                            allow_upscale=False):
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
        force_rows: If set, use exactly this row count and skip the layout search
        allow_upscale: If True, scale freely to fill the target (preset resolutions).
                       If False, upscale only small slices (see limit_scale).

    Returns:
        tuple: (columns, rows, scale_factor, actual_width, actual_height)
    """
    def cap(scale):
        return scale if allow_upscale else limit_scale(scale, slice_h, slice_w)

    if force_rows is not None:
        # The user selects the row count. Derive the column count from it.
        rows = max(1, min(force_rows, n_slices))
        cols = int(np.ceil(n_slices / rows))
        # Remove empty trailing rows if the slice count does not divide evenly.
        rows = int(np.ceil(n_slices / cols))
        # Scale to fit the target dimensions.
        scale = cap(min(target_width / (cols * slice_w),
                        target_height / (rows * slice_h)))
        actual_width = int(cols * slice_w * scale)
        actual_height = int(rows * slice_h * scale)
        return (cols, rows, scale, actual_width, actual_height)

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
        scale = cap(min(scale_w, scale_h))  # Use smaller to ensure it fits
        
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
        
        # Ensure positive dimensions
        if actual_width <= 0 or actual_height <= 0:
            continue
        
        # Score based on resolution utilization and aspect ratio match
        area_utilization = (actual_width * actual_height) / (target_width * target_height)
        
        # Use absolute difference for aspect match to avoid negative values
        aspect_diff = abs((actual_width / actual_height) - target_aspect)
        aspect_match = 1.0 / (1.0 + aspect_diff / target_aspect)  # Always positive, max = 1.0
        
        # Prefer layouts that fill more of the target area and match aspect ratio
        # Use abs() and clamp to ensure positive values before exponentiation
        score = (max(0.0, area_utilization) ** 1.5) * (max(0.0, aspect_match) ** 0.8)
        
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
        
        # Ensure minimum scale (whole-number factor when upscaling)
        min_scale = min_slice_pixels / min(slice_w, slice_h)
        if min_scale > 1.0:
            min_scale = max_upscale(slice_h, slice_w)
        scale = cap(max(scale, min_scale))
        
        actual_width = int(cols * slice_w * scale)
        actual_height = int(rows * slice_h * scale)
        best_layout = (cols, rows, scale, actual_width, actual_height)
    
    return best_layout
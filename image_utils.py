"""
Image processing utilities for NeuroMontage.
"""

import numpy as np
import nibabel as nib
import skimage.measure
from matplotlib.colors import Normalize
from skimage.transform import resize


def load_nifti_robust(filepath):
    """
    Load a NIfTI file with robust error handling for unusual data types.
    
    Handles edge cases including:
    - Files with void/structured dtypes (RGB, complex data)
    - Files with unusual bit depths
    - Corrupted or non-standard headers
    
    Args:
        filepath: Path to the NIfTI file
    
    Returns:
        tuple: (nibabel image object, numpy array of image data)
    
    Raises:
        ValueError: If the file cannot be loaded after all fallback attempts
    """
    nii = nib.load(filepath)
    header = nii.header
    
    # Get info for diagnostics
    raw_dtype = header.get_data_dtype()
    shape = header.get_data_shape()
    
    # First attempt: standard get_fdata()
    try:
        data = nii.get_fdata()
        return nii, data
    except TypeError as e:
        if 'void' in str(e) or 'DType' in str(e):
            print(f"[NeuroMontage] Warning: Non-standard data type ({raw_dtype}) in {filepath}")
            print(f"[NeuroMontage] Attempting fallback loading methods...")
        else:
            raise
    
    # Second attempt: try loading raw data and converting
    try:
        dataobj = nii.dataobj
        
        # Check for structured/compound dtypes
        if hasattr(raw_dtype, 'subdtype') and raw_dtype.subdtype is not None:
            base_dtype, sub_shape = raw_dtype.subdtype
            print(f"[NeuroMontage] Structured dtype: base={base_dtype}, shape={sub_shape}")
            raw_data = np.asarray(dataobj)
            if raw_data.dtype.names:
                first_field = raw_data.dtype.names[0]
                data = raw_data[first_field].astype(np.float64)
            else:
                data = raw_data.view(base_dtype).astype(np.float64)
            print(f"[NeuroMontage] Successfully loaded with structured dtype handling")
            return nii, data
        
        # Try to interpret void dtype
        raw_data = np.asarray(dataobj)
        if raw_data.dtype == np.dtype('void') or str(raw_data.dtype).startswith('V'):
            nbytes = raw_data.nbytes
            expected_elements = np.prod(shape) if shape else 1
            bytes_per_element = nbytes // expected_elements if expected_elements > 0 else 1
            
            print(f"[NeuroMontage] Void dtype with {bytes_per_element} bytes/voxel")
            
            if bytes_per_element == 1:
                data = raw_data.view(np.uint8).reshape(shape).astype(np.float64)
            elif bytes_per_element == 2:
                data = raw_data.view(np.uint16).reshape(shape).astype(np.float64)
            elif bytes_per_element == 3:
                # RGB data
                rgb_data = raw_data.view(np.uint8).reshape(shape + (3,))
                data = np.mean(rgb_data, axis=-1)
                print("[NeuroMontage] Converted RGB data to grayscale")
            elif bytes_per_element == 4:
                try:
                    data = raw_data.view(np.float32).reshape(shape).astype(np.float64)
                except ValueError:
                    rgba_data = raw_data.view(np.uint8).reshape(shape + (4,))
                    data = np.mean(rgba_data[..., :3], axis=-1)
                    print("[NeuroMontage] Converted RGBA data to grayscale")
            elif bytes_per_element == 8:
                data = raw_data.view(np.float64).reshape(shape)
            else:
                raise ValueError(f"Cannot interpret void dtype with {bytes_per_element} bytes/voxel")
            
            print(f"[NeuroMontage] Successfully loaded via void dtype interpretation")
            return nii, data
        
        # Non-void but problematic - try direct conversion
        data = raw_data.astype(np.float64)
        print(f"[NeuroMontage] Successfully loaded via direct dtype conversion")
        return nii, data
            
    except Exception as e2:
        print(f"[NeuroMontage] Fallback method 1 failed: {e2}")
    
    # Third attempt: force loading with explicit float64
    try:
        data = np.asanyarray(nii.dataobj, dtype=np.float64)
        print(f"[NeuroMontage] Successfully loaded via explicit float64 conversion")
        return nii, data
    except Exception as e3:
        print(f"[NeuroMontage] Fallback method 2 failed: {e3}")
    
    # Fourth attempt: try Nifti1Image explicitly
    try:
        nii = nib.Nifti1Image.from_filename(filepath)
        data = np.asarray(nii.dataobj).astype(np.float64)
        print(f"[NeuroMontage] Successfully loaded via Nifti1Image")
        return nii, data
    except Exception as e4:
        print(f"[NeuroMontage] Fallback method 3 failed: {e4}")
    
    # If all else fails, give a detailed error message
    bitpix = header.get('bitpix', 'unknown') if hasattr(header, 'get') else 'unknown'
    raise ValueError(
        f"[NeuroMontage] Cannot load {filepath}. Unsupported data type.\n"
        f"Header info:\n"
        f"  - Data type: {raw_dtype}\n"
        f"  - Shape: {shape}\n"
        f"  - Bits per voxel: {bitpix}\n"
        f"Possible causes:\n"
        f"  - RGB or RGBA encoded image\n"
        f"  - Complex-valued data\n"
        f"  - Corrupted file or non-standard format\n"
        f"Try converting with: fslmaths {filepath} -odt float output.nii.gz"
    )


def get_slice(img_data, idx):
    """
    Return the idx-th axial slice (rotated 90° clockwise for a consistent orientation).
    If idx is out of bounds, use the last valid slice.
    """
    if idx >= img_data.shape[2]:
        idx = img_data.shape[2] - 1
    slice_2d = img_data[:, :, idx]
    return np.rot90(slice_2d, 1)


def find_contours_on_slice(binary_slice):
    """
    Given a 2D binary array `binary_slice` (values 0 or 1), find contours at threshold = 0.5.
    Returns a list of (N_i × 2) arrays of (row, col) floats.
    """
    return skimage.measure.find_contours(binary_slice, 0.5)


def generate_lr_labels():
    """
    Generate two 25×25 "L" and "R" binary masks (values 0 or 255) for orientation labels.
    """
    L = np.zeros((50, 50), dtype=np.uint8)
    L[5:45, 10:15] = 255
    L[40:45, 15:35] = 255

    R = np.zeros((50, 50), dtype=np.uint8)
    R[5:45, 10:15] = 255
    R[5:10, 15:27] = 255
    R[20:25, 15:27] = 255
    for i in range(8):
        R[5 + i : 10 + i, 27 + i : 31 + i] = 255
        R[20 - i : 25 - i, 27 + i : 31 + i] = 255
    R[25:27, 15:25] = 255
    for i in range(21):
        R[21 + i : 25 + i, 15 + i : 20 + i] = 255

    return L[::2, ::2], R[::2, ::2]


def get_orientation(affine):
    """
    Determine orientation (L-R or R-L) from NIfTI affine matrix.
    
    Args:
        affine: NIfTI affine matrix
    
    Returns:
        str: 'L-R' or 'R-L'
    """
    return 'R-L' if affine[0, 0] < 0 else 'L-R'


def get_contour_colors():
    """
    Return Okabe-Ito colorblind-friendly color palette for multiple segmentation masks.
    
    The Okabe-Ito palette is designed to be distinguishable by people with all types
    of color vision deficiency.
    
    Returns:
        list: List of RGB color arrays in Okabe-Ito palette order
    """
    return [
        np.array([230, 159, 0], dtype=np.uint8),    # orange
        np.array([86, 180, 233], dtype=np.uint8),   # sky blue
        np.array([0, 158, 115], dtype=np.uint8),    # bluish green
        np.array([240, 228, 66], dtype=np.uint8),   # yellow
        np.array([0, 114, 178], dtype=np.uint8),    # blue
        np.array([213, 94, 0], dtype=np.uint8),     # vermillion
        np.array([204, 121, 167], dtype=np.uint8),  # reddish purple
        np.array([0, 0, 0], dtype=np.uint8),        # black
    ]


def get_line_pattern(mask_index):
    """
    Return line pattern for a given mask index.
    
    Different masks get different line styles (solid, dashed, dotted) so 
    overlapping contours are all visible. Draw order matters: solid first 
    (bottom), then dashed, then dotted on top.
    
    Args:
        mask_index: Index of the segmentation mask (0, 1, 2, ...)
    
    Returns:
        tuple: (pattern_length, on_length) defining the dash pattern
               - solid: (1, 1) - all pixels drawn
               - dashed: (5, 3) - 3 on, 2 off (60% visible)
               - dotted: (3, 2) - 2 on, 1 off (67% visible)
    """
    patterns = [
        (1, 1),   # Mask 0: solid (draw every pixel)
        (5, 3),   # Mask 1: dashed (3 on, 2 off)
        (3, 2),   # Mask 2: dotted (2 on, 1 off)
        (7, 4),   # Mask 3: long dash (4 on, 3 off)
        (4, 2),   # Mask 4: short dash (2 on, 2 off)
        (9, 5),   # Mask 5: very long dash
        (6, 4),   # Mask 6: medium dash
        (4, 3),   # Mask 7: dense dash
    ]
    return patterns[mask_index % len(patterns)]


# Names of the palette colors and line patterns above, in the same order
CONTOUR_COLOR_NAMES = ['orange', 'skyblue', 'green', 'yellow', 'blue', 'vermillion', 'pink', 'black']
LINE_STYLE_NAMES = ['solid', 'dashed', 'dotted', 'long-dash', 'short-dash',
                    'very-long-dash', 'medium-dash', 'dense-dash']


def resolve_outline_styles(n_masks, colors=None, line_styles=None):
    """
    Color and line pattern for each mask in outline mode.
    
    Without colors/line_styles, mask i gets palette color i and line pattern i.
    With them, each mask gets the given color (palette name or #RRGGBB) and line
    style (name from LINE_STYLE_NAMES), so a caller can leave out a mask without
    the following masks changing their look.
    
    Args:
        n_masks: Number of masks
        colors: Optional list of color names / hex values, one per mask
        line_styles: Optional list of line style names, one per mask
    
    Returns:
        tuple: (list of RGB uint8 arrays, list of line pattern indexes)
    
    Raises:
        ValueError: if a list does not have one entry per mask, or a value is unknown
    """
    palette = get_contour_colors()
    if colors is None:
        rgb = [palette[i % len(palette)] for i in range(n_masks)]
    else:
        if len(colors) != n_masks:
            raise ValueError(f"--colors needs one color per mask ({n_masks}), got {len(colors)}")
        rgb = []
        for color in colors:
            name = color.strip().lower()
            if name in CONTOUR_COLOR_NAMES:
                rgb.append(palette[CONTOUR_COLOR_NAMES.index(name)])
            elif len(name) == 7 and name.startswith('#'):
                try:
                    rgb.append(np.array([int(name[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8))
                except ValueError:
                    raise ValueError(f"Invalid color '{color}' (use #RRGGBB)")
            else:
                raise ValueError(f"Unknown color '{color}'. Use #RRGGBB or one of: "
                                 f"{', '.join(CONTOUR_COLOR_NAMES)}")
    
    if line_styles is None:
        patterns = list(range(n_masks))
    else:
        if len(line_styles) != n_masks:
            raise ValueError(f"--line-styles needs one style per mask ({n_masks}), got {len(line_styles)}")
        patterns = []
        for style in line_styles:
            name = style.strip().lower()
            if name not in LINE_STYLE_NAMES:
                raise ValueError(f"Unknown line style '{style}'. Use one of: {', '.join(LINE_STYLE_NAMES)}")
            patterns.append(LINE_STYLE_NAMES.index(name))
    
    return rgb, patterns


def should_draw_pixel(contour_index, mask_index):
    """
    Determine if a pixel should be drawn based on its position along the contour
    and the mask's line pattern.
    
    Args:
        contour_index: Index of this point along the contour (0, 1, 2, ...)
        mask_index: Line pattern index (by default the mask's index)
    
    Returns:
        bool: True if this pixel should be drawn
    """
    pattern_length, on_length = get_line_pattern(mask_index)
    return (contour_index % pattern_length) < on_length


def calculate_robust_normalization(volume_data, lower_percentile=1, upper_percentile=99.99):
    """
    Calculate robust normalization bounds excluding background.
    
    This function excludes the bottom 5% of unique intensity values (typically background)
    and then calculates percentile-based normalization bounds from the remaining voxels.
    This approach is robust to outliers and handles volumes with large background regions.
    
    Args:
        volume_data: numpy array of intensity values (3D or 4D)
        lower_percentile: Lower percentile for normalization (default: 1)
        upper_percentile: Upper percentile for normalization (default: 99.99)
    
    Returns:
        Normalize: matplotlib Normalize object with robust vmin/vmax
    
    Example:
        >>> brain_data = nib.load('brain.nii.gz').get_fdata()
        >>> norm = calculate_robust_normalization(brain_data)
        >>> normalized_slice = norm(brain_data[:, :, 50])
    """
    # Exclude background by removing the bottom 5% of unique intensity values
    unique_values = np.unique(volume_data)
    threshold_5pct = np.percentile(unique_values, 5)
    brain_voxels = volume_data[volume_data > threshold_5pct]
    
    if brain_voxels.size > 0:
        # Calculate robust bounds from non-background voxels
        vmin = np.percentile(brain_voxels, lower_percentile)
        vmax = np.percentile(brain_voxels, upper_percentile)
    else:
        # Fallback if no voxels above threshold (shouldn't happen in practice)
        vmin = np.min(volume_data)
        vmax = np.max(volume_data)
    
    # Ensure vmax > vmin to avoid division by zero
    if vmax <= vmin:
        vmax = vmin + 1
    
    return Normalize(vmin=vmin, vmax=vmax)


def calculate_median_excluding_background(volume_data):
    """
    Calculate median intensity excluding background voxels.
    
    This function excludes the bottom 5% of unique intensity values (typically background)
    before calculating the median. Useful for intensity comparisons across volumes.
    
    Args:
        volume_data: numpy array of intensity values (3D)
    
    Returns:
        float: Median intensity of non-background voxels
    
    Example:
        >>> brain_data = nib.load('brain.nii.gz').get_fdata()
        >>> median = calculate_median_excluding_background(brain_data)
    """
    # Exclude background by removing the bottom 5% of unique intensity values
    unique_values = np.unique(volume_data)
    threshold_5pct = np.percentile(unique_values, 5)
    brain_voxels = volume_data[volume_data > threshold_5pct]
    
    if brain_voxels.size > 0:
        return np.median(brain_voxels)
    else:
        # Fallback if no voxels above threshold
        return np.median(volume_data)
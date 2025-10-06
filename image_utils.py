"""
Image processing utilities for NeuroMontage.
"""

import numpy as np
import skimage.measure
from matplotlib.colors import Normalize
from skimage.transform import resize


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
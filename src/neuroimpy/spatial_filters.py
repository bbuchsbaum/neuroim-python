"""Spatial filtering functions for neuroimaging data.

Direct translation of R's neuroim2 spatial filtering functions.
"""

import numpy as np
from scipy import ndimage
from typing import Union, Optional
from .neuro_vol import NeuroVol, DenseNeuroVol, LogicalNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec
from .kernel import Kernel, gaussian_kernel

def gaussian_blur(vol: NeuroVol, mask: Optional[LogicalNeuroVol] = None, 
                 sigma: float = 2, window: int = 1) -> DenseNeuroVol:
    """Apply Gaussian blur to a volumetric image.
    
    This function applies an isotropic discrete Gaussian kernel to smooth
    a volumetric image (3D brain MRI data). The blurring is performed within
    a specified image mask, with customizable kernel parameters.
    
    Parameters
    ----------
    vol : NeuroVol
        The image volume to be smoothed
    mask : LogicalNeuroVol, optional
        Image mask defining the region where blurring is applied.
        If not provided, the entire volume is processed.
    sigma : float
        Standard deviation of the Gaussian kernel (default is 2)
    window : int
        Number of voxels to include on each side of the center voxel.
        For example, window=1 results in a 3x3x3 kernel (default is 1)
        
    Returns
    -------
    DenseNeuroVol
        The smoothed image
        
    R Equivalent
    ------------
    neuroim2::gaussian_blur
    """
    if window < 1:
        raise ValueError("Window size must be at least 1")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    # Work with the original data when possible
    data = vol.data
    
    if mask is not None:
        # Create output array only when mask is provided
        output_data = data.copy()
        mask_indices = np.where(mask.data)
        # Apply gaussian filter only to masked region for efficiency
        blurred_data = ndimage.gaussian_filter(data, sigma=sigma, truncate=window)
        output_data[mask_indices] = blurred_data[mask_indices]
    else:
        # When no mask, directly apply filter (no copy needed)
        output_data = ndimage.gaussian_filter(data, sigma=sigma, truncate=window)

    return DenseNeuroVol(output_data, vol.space)

def box_blur(data: np.ndarray, mask_indices: np.ndarray, radius: int) -> np.ndarray:
    """Helper function for guided filter: applies box blur to the data."""
    kernel = np.ones((2*radius+1, 2*radius+1, 2*radius+1)) / ((2*radius+1)**3)
    blurred = ndimage.convolve(data, kernel, mode='constant', cval=0.0)
    return blurred[mask_indices]

def guided_filter(vol: NeuroVol, radius: int = 4, epsilon: float = 0.49) -> DenseNeuroVol:
    """Apply edge-preserving guided filter to a volumetric image.
    
    This function applies a guided filter to perform edge-preserving smoothing.
    The guided filter smooths the image while preserving edges, providing a
    balance between noise reduction and structural preservation.
    
    Parameters
    ----------
    vol : NeuroVol
        The image volume to be filtered
    radius : int
        Spatial radius of the filter (default is 4)
    epsilon : float
        Regularization parameter controlling the degree of smoothing
        and edge preservation (default is 0.49)
        
    Returns
    -------
    DenseNeuroVol
        The filtered image
        
    R Equivalent
    ------------
    neuroim2::guided_filter
    """
    if radius < 1:
        raise ValueError("Radius must be at least 1")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")
    
    data = vol.data
    mask_indices = np.where(data != 0)

    # Apply box blur to compute local means
    mean_I = box_blur(vol, mask_indices, radius)
    
    # Create temporary volumes for squared data
    vol_squared = DenseNeuroVol(data * data, vol.space)
    mean_II = box_blur(vol_squared, mask_indices, radius)

    var_I = mean_II - mean_I * mean_I
    mean_p = mean_I
    mean_Ip = mean_II

    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I

    # Create temporary volumes for a and b
    a_full = np.zeros_like(data)
    b_full = np.zeros_like(data)
    a_full[mask_indices] = a
    b_full[mask_indices] = b
    
    a_vol = DenseNeuroVol(a_full, vol.space)
    b_vol = DenseNeuroVol(b_full, vol.space)
    
    mean_a = box_blur(a_vol, mask_indices, radius)
    mean_b = box_blur(b_vol, mask_indices, radius)

    # Apply the guided filter formula
    output_data = data.copy()
    output_data[mask_indices] = mean_a * data[mask_indices] + mean_b

    return DenseNeuroVol(output_data, vol.space)

def bilateral_filter(vol: NeuroVol, mask: Optional[LogicalNeuroVol] = None, 
                     spatial_sigma: float = 2, intensity_sigma: float = 1, 
                     window: int = 1) -> DenseNeuroVol:
    """Apply bilateral filter to a volumetric image.
    
    This function smooths a volumetric image using a bilateral filter.
    The bilateral filter considers both spatial closeness and intensity
    similarity for smoothing.
    
    Parameters
    ----------
    vol : NeuroVol
        The image volume to be smoothed
    mask : LogicalNeuroVol, optional
        Image mask defining the region where filtering is applied.
        If not provided, the entire volume is considered.
    spatial_sigma : float
        Standard deviation of the spatial Gaussian kernel (default is 2)
    intensity_sigma : float
        Standard deviation of the intensity Gaussian kernel (default is 1)
    window : int
        Number of voxels around the center voxel to include on each side.
        For example, window=1 for a 3x3x3 kernel (default is 1)
        
    Returns
    -------
    DenseNeuroVol
        The smoothed image
        
    R Equivalent
    ------------
    neuroim2::bilateral_filter
    """
    if window < 1:
        raise ValueError("Window size must be at least 1")
    if spatial_sigma <= 0 or intensity_sigma <= 0:
        raise ValueError("Sigma values must be positive")

    data = vol.data
    if mask is not None:
        mask_indices = np.where(mask.data)
    else:
        mask_indices = np.where(np.ones_like(data, dtype=bool))

    # Create spatial and intensity kernels
    spatial_kernel = np.exp(-np.arange(-window, window+1)**2 / (2 * spatial_sigma**2))
    spatial_kernel = spatial_kernel[:, np.newaxis, np.newaxis] * spatial_kernel[np.newaxis, :, np.newaxis] * spatial_kernel[np.newaxis, np.newaxis, :]

    def bilateral_at_point(point):
        i, j, k = point
        local_region = data[max(0, i-window):i+window+1, 
                            max(0, j-window):j+window+1, 
                            max(0, k-window):k+window+1]
        
        intensity_diff = local_region - data[i, j, k]
        intensity_kernel = np.exp(-intensity_diff**2 / (2 * intensity_sigma**2))
        
        kernel = spatial_kernel[:local_region.shape[0], :local_region.shape[1], :local_region.shape[2]] * intensity_kernel
        
        weighted_sum = np.sum(local_region * kernel)
        normalization = np.sum(kernel)
        
        return weighted_sum / normalization

    # Initialize output - only copy if we have a mask
    if mask is not None:
        output_data = data.copy()
        for point in zip(*mask_indices):
            output_data[point] = bilateral_at_point(point)
    else:
        # Process entire volume
        output_data = np.empty_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    output_data[i, j, k] = bilateral_at_point((i, j, k))

    return DenseNeuroVol(output_data, vol.space)

def bilateral_filter_vec(vec: NeuroVec, mask: Optional[LogicalNeuroVol] = None, 
                         spatial_sigma: float = 2, intensity_sigma: float = 1, 
                         window: int = 1) -> DenseNeuroVec:
    """Apply bilateral filter to each volume of a NeuroVec.
    
    This function applies a bilateral filter to each volume of a NeuroVec object.
    
    Parameters
    ----------
    vec : NeuroVec
        The volumes to be filtered
    mask : LogicalNeuroVol, optional
        Binary mask specifying the region of interest.
        If not provided, the whole volume is considered.
    spatial_sigma : float
        Spatial sigma for the bilateral filter (default is 2)
    intensity_sigma : float
        Intensity sigma for the bilateral filter (default is 1)
    window : int
        Size of the window for the bilateral filter (default is 1)
        
    Returns
    -------
    DenseNeuroVec
        The filtered volumes
        
    R Equivalent
    ------------
    neuroim2::bilateral_filter_vec
    """
    # Extract individual volumes and filter
    filtered_vols = []
    for i in range(vec.shape[3]):
        vol = vec.vols(i)
        filtered_vol = bilateral_filter(vol, mask, spatial_sigma, intensity_sigma, window)
        filtered_vols.append(filtered_vol)
    
    # Stack filtered volumes
    filtered_data = np.stack([vol.data for vol in filtered_vols], axis=-1)
    return DenseNeuroVec(filtered_data, vec.space)



def box_blur(vol: NeuroVol, mask_indices: tuple, radius: int) -> np.ndarray:
    """Helper function for guided filter: applies box blur to the data.
    
    Parameters
    ----------
    vol : NeuroVol
        Volume to blur
    mask_indices : tuple
        Indices where to apply the blur
    radius : int
        Radius of the box filter
        
    Returns
    -------
    np.ndarray
        Blurred values at mask indices
        
    Notes
    -----
    This is used internally by guided_filter.
    """
    data = vol.data
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size ** 3)
    
    # Apply convolution
    blurred = ndimage.convolve(data, kernel, mode='constant', cval=0.0)
    
    # Return values at mask indices
    return blurred[mask_indices]
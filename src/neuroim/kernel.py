"""Kernel support for spatial filtering operations.

Defines spatial kernels used by neuroim filtering operations.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Any
from scipy.spatial.distance import euclidean

from .neuro_space import NeuroSpace
from .neuro_vol import SparseNeuroVol

class Kernel:
    """A class representing a filter kernel for spatial operations.

    This class encapsulates a spatial kernel with its weights, voxel offsets,
    and coordinates for use in filtering operations.

    Attributes
    ----------
    width : tuple
        The dimensions of the kernel (e.g., (3, 3, 3) for a 3x3x3 kernel)
    weights : np.ndarray
        The kernel weights (flattened)
    voxels : np.ndarray
        Matrix of relative voxel locations centered at (0,0,0)
    coords : np.ndarray
        Matrix of coordinate locations corresponding to voxels

    """

    def __init__(
        self,
        width: Tuple[int, ...],
        weights: np.ndarray,
        voxels: np.ndarray,
        coords: np.ndarray,
    ):
        """Initialize a Kernel object.

        Parameters
        ----------
        width : tuple
            Kernel dimensions
        weights : np.ndarray
            Kernel weights (1D array)
        voxels : np.ndarray
            Voxel offsets (N x ndim matrix)
        coords : np.ndarray
            Coordinate locations (N x ndim matrix)
        """
        self.width = tuple(width)
        self.weights = np.asarray(weights).flatten()
        self.voxels = np.asarray(voxels)
        self.coords = np.asarray(coords)

        # Validate dimensions
        if len(self.weights) != len(self.voxels):
            raise ValueError("weights and voxels must have same length")
        if self.voxels.shape[1] != len(self.width):
            raise ValueError("voxels dimensions must match kernel width dimensions")

    def get_voxels(self, center_voxel: Optional[np.ndarray] = None) -> np.ndarray:
        """Get voxel locations, optionally centered at a specific voxel.

        Parameters
        ----------
        center_voxel : np.ndarray, optional
            Absolute location of the center voxel

        Returns
        -------
        np.ndarray
            Voxel locations

        """
        if center_voxel is None:
            return self.voxels.copy()
        else:
            return self.voxels + np.asarray(center_voxel)

    def __repr__(self):
        return f"Kernel(width={self.width}, n_weights={len(self.weights)})"

def gaussian_kernel(
    vdim: Tuple[float, ...], kdim: Tuple[int, ...], sigma: float, normalize: bool = True
) -> Kernel:
    """Create a Gaussian kernel.

    Parameters
    ----------
    vdim : tuple
        Voxel dimensions (spacing)
    kdim : tuple
        Kernel dimensions in voxels
    sigma : float
        Standard deviation of Gaussian
    normalize : bool
        Whether to normalize weights to sum to 1

    Returns
    -------
    Kernel
        Gaussian kernel object

    """
    # Convert to arrays
    vdim = np.asarray(vdim)
    kdim = np.asarray(kdim)

    # Ensure odd dimensions
    kdim = np.array([k if k % 2 == 1 else k + 1 for k in kdim])

    # Half-width for each dimension
    hwidth = np.ceil(kdim / 2 - 1).astype(int)

    # Create grid
    grids = [np.arange(-hw, hw + 1) for hw in hwidth]
    voxel_ind = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(kdim), -1).T

    # Convert to coordinates
    cvoxel_ind = voxel_ind * vdim
    coords = np.where(
        cvoxel_ind == 0, 0, np.sign(cvoxel_ind) * (np.abs(cvoxel_ind) - 0.5)
    )

    # Calculate distances from center
    distances = np.sqrt(np.sum(coords**2, axis=1))

    # Gaussian weights
    weights = np.exp(-0.5 * (distances / sigma) ** 2)

    if normalize:
        weights = weights / np.sum(weights)

    return Kernel(width=tuple(kdim), weights=weights, voxels=voxel_ind, coords=coords)

def spherical_kernel(
    vdim: Tuple[float, ...], radius: float, weight_func: Optional[Callable] = None
) -> Kernel:
    """Create a spherical kernel.

    Parameters
    ----------
    vdim : tuple
        Voxel dimensions (spacing)
    radius : float
        Radius of sphere in real-world units
    weight_func : callable, optional
        Function to compute weights from distances
        If None, uses uniform weights

    Returns
    -------
    Kernel
        Spherical kernel object

    """
    # Convert to arrays
    vdim = np.asarray(vdim)

    # Compute kernel dimensions to encompass sphere
    kdim = np.ceil(2 * radius / vdim).astype(int)
    kdim = np.array([k if k % 2 == 1 else k + 1 for k in kdim])

    # Half-width for each dimension
    hwidth = np.ceil(kdim / 2 - 1).astype(int)

    # Create grid
    grids = [np.arange(-hw, hw + 1) for hw in hwidth]
    voxel_ind = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(kdim), -1).T

    # Convert to coordinates
    cvoxel_ind = voxel_ind * vdim
    coords = np.where(
        cvoxel_ind == 0, 0, np.sign(cvoxel_ind) * (np.abs(cvoxel_ind) - 0.5)
    )

    # Calculate distances from center
    distances = np.sqrt(np.sum(coords**2, axis=1))

    # Keep only voxels within radius
    mask = distances <= radius
    voxel_ind = voxel_ind[mask]
    coords = coords[mask]
    distances = distances[mask]

    # Compute weights
    if weight_func is None:
        weights = np.ones(len(distances))
    else:
        weights = weight_func(distances)

    # Normalize
    weights = weights / np.sum(weights)

    return Kernel(width=tuple(kdim), weights=weights, voxels=voxel_ind, coords=coords)

def box_kernel(kdim: Tuple[int, ...]) -> Kernel:
    """Create a box (uniform) kernel.

    Parameters
    ----------
    kdim : tuple
        Kernel dimensions in voxels

    Returns
    -------
    Kernel
        Box kernel object with uniform weights

    """
    # Ensure odd dimensions
    kdim = np.array([k if k % 2 == 1 else k + 1 for k in kdim])

    # Half-width for each dimension
    hwidth = np.ceil(kdim / 2 - 1).astype(int)

    # Create grid
    grids = [np.arange(-hw, hw + 1) for hw in hwidth]
    voxel_ind = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(kdim), -1).T

    # Uniform weights
    n_voxels = len(voxel_ind)
    weights = np.ones(n_voxels) / n_voxels

    # Dummy coords (not used for box kernel)
    coords = voxel_ind.astype(float)

    return Kernel(width=tuple(kdim), weights=weights, voxels=voxel_ind, coords=coords)

def embed_kernel(
    kernel: Kernel, space: NeuroSpace, center_voxel: np.ndarray, weight: float = 1.0
) -> SparseNeuroVol:
    """Embed a kernel in a NeuroSpace at a specific location.

    Parameters
    ----------
    kernel : Kernel
        The kernel to embed
    space : NeuroSpace
        The space to embed into
    center_voxel : np.ndarray
        The center voxel location
    weight : float
        Multiply kernel weights by this value

    Returns
    -------
    SparseNeuroVol
        Sparse volume with kernel embedded

    """
    # Get absolute voxel locations
    vox = np.floor(kernel.get_voxels(center_voxel)).astype(int)

    # Convert to indices
    indices = space.grid_to_index(vox)

    # Create sparse volume
    return SparseNeuroVol(data=kernel.weights * weight, space=space, indices=indices)

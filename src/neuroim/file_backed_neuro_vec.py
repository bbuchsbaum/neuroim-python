"""FileBackedNeuroVec - File-backed neuroimaging vector data.

This module provides a file-backed implementation of NeuroVec
that reads data from disk on demand, supporting various file formats.

Provides a Python-native lazy vector container for image series on disk.
"""

import numpy as np
from typing import Optional, List, Union
import os
from .neuro_vec import NeuroVec
from .neuro_space import NeuroSpace
from .neuro_vol import NeuroVol, DenseNeuroVol
from .io import read_vol

class FileBackedNeuroVec(NeuroVec):
    """A file-backed 4D neuroimaging vector.

    This class provides lazy loading of 4D neuroimaging data from
    a list of files, loading volumes on demand to minimize memory usage.

    Parameters
    ----------
    filenames : list of str
        List of file paths, one per volume
    space : NeuroSpace, optional
        The 4D spatial metadata. If not provided, will be inferred
        from the first file.
    cache_size : int, optional
        Number of volumes to cache in memory. Default is 1.

    """

    def __init__(
        self,
        filenames: List[str],
        space: Optional[NeuroSpace] = None,
        cache_size: int = 1,
    ):
        """Initialize FileBackedNeuroVec."""
        if not filenames:
            raise ValueError("filenames list cannot be empty")

        self.filenames = filenames
        self.n_volumes = len(filenames)

        # Read first volume to get dimensions
        first_vol = read_vol(filenames[0])
        vol_shape = first_vol.shape
        vol_space = first_vol.space

        # Create 4D space if not provided
        if space is None:
            # Create 4D axes with spatial dimensions first, then time
            # This matches R neuroim2 convention: (x, y, z, time)
            from .axis import AxisSet4D, NamedAxis

            time_axis = NamedAxis("t", 1)
            axes_4d = AxisSet4D(
                vol_space.axes.i, vol_space.axes.j, vol_space.axes.k, time_axis
            )

            space = NeuroSpace(
                dim=[int(d) for d in vol_shape] + [self.n_volumes],
                spacing=[float(s) for s in vol_space.spacing] + [1.0],
                origin=[float(o) for o in vol_space.origin] + [0.0],
                axes=axes_4d,
            )

        super().__init__(space)

        # Cache setup
        self.cache_size = max(1, cache_size)
        self._cache = {}  # Dict mapping volume index to data
        self._cache_order = []  # Track order for LRU eviction

        # Store volume metadata
        self.vol_shape = vol_shape
        self.vol_space = vol_space
        # Keep the historical public dtype of read-loaded file-backed vectors:
        # read_vol materializes analysis data as float64 even when the source
        # NIfTI payload is narrower.
        self.dtype = np.dtype(np.float64)

    def _load_volume(self, idx: int) -> np.ndarray:
        """Load a volume from disk with caching."""
        if idx < 0 or idx >= self.n_volumes:
            raise IndexError(f"Volume index {idx} out of range")

        # Check cache
        if idx in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            return self._cache[idx]

        # Load from disk
        vol = read_vol(self.filenames[idx])
        if vol.shape != self.vol_shape:
            raise ValueError(
                f"Volume {idx} has inconsistent shape: "
                f"{vol.shape} vs expected {self.vol_shape}"
            )

        # Add to cache
        self._cache[idx] = vol.data
        self._cache_order.append(idx)

        # Evict oldest if cache full
        if len(self._cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return vol.data

    def __getitem__(self, key):
        """Extract data using various indexing methods."""
        if isinstance(key, int):
            # Single time point - extract entire volume
            # Shape is (x, y, z, time) so extract [:, :, :, key]
            return self._load_volume(key)

        elif isinstance(key, slice):
            # Special case: single slice means time slice across all space
            # This maintains backward compatibility with fb_vec[1:4]
            indices = range(*key.indices(self.n_volumes))
            # Result shape should be (x, y, z, n_timepoints)
            result = np.zeros(self.vol_shape + (len(indices),), dtype=self.dtype)
            for i, idx in enumerate(indices):
                result[:, :, :, i] = self._load_volume(idx)
            return result

        elif isinstance(key, tuple):
            # Complex indexing
            if len(key) == 4:
                # Full 4D indexing (x, y, z, time)
                spatial_idx = key[:3]
                t_idx = key[3]

                if isinstance(t_idx, int):
                    # Single time point
                    vol_data = self._load_volume(t_idx)
                    return vol_data[spatial_idx]
                elif isinstance(t_idx, slice):
                    # Multiple time points
                    t_indices = range(*t_idx.indices(self.n_volumes))
                    # Get the shape of the spatial selection first
                    first_vol = self._load_volume(t_indices[0])
                    spatial_result = first_vol[spatial_idx]

                    # Create result array with time as last dimension
                    if isinstance(spatial_result, np.ndarray):
                        result_shape = spatial_result.shape + (len(t_indices),)
                    else:
                        # Scalar result from spatial indexing
                        result_shape = (len(t_indices),)

                    result = np.zeros(result_shape, dtype=self.dtype)

                    # Fill in the data
                    for i, t in enumerate(t_indices):
                        vol_data = self._load_volume(t)
                        if result.ndim == 1:
                            result[i] = vol_data[spatial_idx]
                        else:
                            result[..., i] = vol_data[spatial_idx]

                    return result
                else:
                    raise TypeError(f"Unsupported time index type: {type(t_idx)}")
            else:
                raise ValueError(f"Expected 4 indices, got {len(key)}")

        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

    def __setitem__(self, key, value):
        """Setting values not supported for file-backed data."""
        raise TypeError("Cannot modify read-only file-backed data")

    @property
    def data(self) -> np.ndarray:
        """Get all data (loads all volumes into memory)."""
        result = np.zeros(self.shape, dtype=self.dtype)
        for i in range(self.n_volumes):
            result[:, :, :, i] = self._load_volume(i)
        return result

    @property
    def values(self) -> np.ndarray:
        """Get all values (loads all volumes into memory)."""
        return self.data

    def series(self, x, y=None, z=None) -> np.ndarray:
        """Extract time series for voxel(s)."""
        if y is not None and z is not None:
            # Single voxel
            series = np.zeros(self.n_volumes, dtype=self.dtype)
            for t in range(self.n_volumes):
                vol_data = self._load_volume(t)
                series[t] = vol_data[x, y, z]
            return series
        elif isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 3:
                # Nx3 matrix of coordinates
                result = np.zeros((self.n_volumes, x.shape[0]), dtype=self.dtype)
                for t in range(self.n_volumes):
                    vol_data = self._load_volume(t)
                    for i, coord in enumerate(x):
                        # Check bounds
                        if (
                            0 <= coord[0] < self.vol_shape[0]
                            and 0 <= coord[1] < self.vol_shape[1]
                            and 0 <= coord[2] < self.vol_shape[2]
                        ):
                            result[t, i] = vol_data[coord[0], coord[1], coord[2]]
                        # else: leave as zeros
                return result
            elif x.ndim == 1:
                # Linear indices - convert to 3D coordinates
                coords = np.array(np.unravel_index(x, self.vol_shape, order="F")).T
                result = np.zeros((self.n_volumes, len(x)), dtype=self.dtype)
                for t in range(self.n_volumes):
                    vol_data = self._load_volume(t)
                    for i, coord in enumerate(coords):
                        if (
                            0 <= coord[0] < self.vol_shape[0]
                            and 0 <= coord[1] < self.vol_shape[1]
                            and 0 <= coord[2] < self.vol_shape[2]
                        ):
                            result[t, i] = vol_data[coord[0], coord[1], coord[2]]
                return result
        elif isinstance(x, int):
            # Single linear index - convert to 3D coordinates
            coords = np.unravel_index(x, self.vol_shape, order="F")
            series = np.zeros(self.n_volumes, dtype=self.dtype)
            for t in range(self.n_volumes):
                vol_data = self._load_volume(t)
                series[t] = vol_data[coords[0], coords[1], coords[2]]
            return series
        else:
            raise ValueError("Invalid input for series extraction")

    def sub_vector(self, indices: Union[slice, np.ndarray]) -> "FileBackedNeuroVec":
        """Extract subset of volumes."""
        if isinstance(indices, slice):
            indices = list(range(*indices.indices(self.n_volumes)))
        else:
            indices = list(indices)

        # Create new file list
        subset_files = [self.filenames[i] for i in indices]

        # Create new space
        subset_space = NeuroSpace(
            dim=list(self.vol_shape) + [len(indices)],
            spacing=self.space.spacing,
            origin=self.space.origin,
            axes=self.space.axes,
        )

        return FileBackedNeuroVec(subset_files, subset_space, self.cache_size)

    def vols(self, indices: Optional[np.ndarray] = None) -> List[NeuroVol]:
        """Extract volumes as a list of NeuroVol objects."""
        if indices is None:
            indices = range(self.n_volumes)

        vol_list = []
        for i in indices:
            vol_data = self._load_volume(i)
            vol = DenseNeuroVol(vol_data, self.vol_space)
            vol_list.append(vol)

        return vol_list

    def as_matrix(self) -> np.ndarray:
        """Convert to 2D matrix (loads all data into memory)."""
        # Reshape to (n_voxels, n_timepoints) following R convention
        return self.data.reshape(-1, self.n_volumes, order="F").T

    def as_dense(self) -> "DenseNeuroVec":
        """Convert to DenseNeuroVec (loads all data into memory)."""
        from .neuro_vec import DenseNeuroVec

        return DenseNeuroVec(self.data, self.space)

    def as_sparse(self, mask: Optional[np.ndarray] = None) -> "SparseNeuroVec":
        """Convert to SparseNeuroVec."""
        from .neuro_vec import SparseNeuroVec
        from .neuro_vol import LogicalNeuroVol

        if mask is None:
            # Use non-zero voxels from first volume
            first_vol = self._load_volume(0)
            mask = first_vol != 0

        # Create LogicalNeuroVol from mask
        mask_vol = LogicalNeuroVol(mask, self.vol_space)

        # Get mask indices
        mask_indices = np.where(mask.ravel(order="F"))[0]
        n_masked = len(mask_indices)

        # Extract masked data - shape should be (n_timepoints, n_masked) for SparseNeuroVec
        sparse_data = np.zeros((self.n_volumes, n_masked), dtype=self.dtype)
        for t in range(self.n_volumes):
            vol_data = self._load_volume(t).ravel(order="F")
            sparse_data[t, :] = vol_data[mask_indices]

        return SparseNeuroVec(data=sparse_data, space=self.space, mask=mask_vol)

    def _arithmetic_op(self, other, op):
        """Arithmetic operations (loads all data)."""
        # For arithmetic, we need to load all data
        self_data = self.data

        try:
            if np.isscalar(other):
                result_data = op(self_data, other)
            elif isinstance(other, FileBackedNeuroVec):
                if self.shape != other.shape:
                    raise ValueError("Incompatible shapes for arithmetic operation")
                other_data = other.data
                result_data = op(self_data, other_data)
            elif isinstance(other, NeuroVec):
                if self.shape != other.shape:
                    raise ValueError("Incompatible shapes for arithmetic operation")
                result_data = op(self_data, other.data)
            else:
                raise TypeError(f"Unsupported operand type: {type(other)}")
        except TypeError as e:
            # Catch numpy type errors and re-raise as TypeError
            if "Unsupported operand type" not in str(e):
                raise TypeError(f"Unsupported operand type: {type(other)}")
            else:
                raise

        # Return as DenseNeuroVec
        from .neuro_vec import DenseNeuroVec

        return DenseNeuroVec(result_data, self.space)

    def _comparison_op(self, other, op):
        """Comparison operations (loads all data)."""
        self_data = self.data

        if np.isscalar(other):
            return op(self_data, other)
        elif isinstance(other, FileBackedNeuroVec):
            if self.shape != other.shape:
                raise ValueError("Incompatible shapes for comparison")
            return op(self_data, other.data)
        elif isinstance(other, NeuroVec):
            if self.shape != other.shape:
                raise ValueError("Incompatible shapes for comparison")
            return op(self_data, other.data)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def clear_cache(self):
        """Clear the volume cache."""
        self._cache.clear()
        self._cache_order.clear()

    def __repr__(self):
        """String representation."""
        return (
            f"FileBackedNeuroVec\n"
            f"  Type      : FileBackedNeuroVec\n"
            f"  Dimension : {' X '.join(map(str, self.shape))}\n"
            f"  Spacing   : {' X '.join(map(str, self.space.spacing))}\n"
            f"  Origin    : {', '.join(map(str, self.space.origin))}\n"
            f"  N Files   : {self.n_volumes}\n"
            f"  Cache Size: {self.cache_size}"
        )

def file_backed_neurovec(
    filenames: List[str], mask: Optional[np.ndarray] = None
) -> FileBackedNeuroVec:
    """Create FileBackedNeuroVec from a list of files.

    Parameters
    ----------
    filenames : list of str
        List of file paths, one per volume
    mask : array-like, optional
        Binary mask (currently unused, for API compatibility)

    Returns
    -------
    FileBackedNeuroVec
        File-backed 4D vector

    """
    return FileBackedNeuroVec(filenames)

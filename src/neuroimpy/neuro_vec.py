"""4D Neuroimaging Vector Classes.

Direct translation of R's neuroim2 NeuroVec classes.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple, List, Optional
from scipy import sparse

from .neuro_space import NeuroSpace
from .neuro_vol import NeuroVol, DenseNeuroVol, LogicalNeuroVol
from .axis import drop_axis


class NeuroVec(ABC):
    """Abstract base class for 4D neuroimaging vectors.
    
    Direct translation of R's NeuroVec class.
    
    Parameters
    ----------
    space : NeuroSpace
        4D spatial metadata
        
    R Equivalent
    ------------
    neuroim2::NeuroVec
    """
    
    def __init__(self, space: NeuroSpace):
        if space.ndim != 4:
            raise ValueError("NeuroVec requires 4D space")
        self.space = space
    
    @abstractmethod
    def __getitem__(self, key):
        """Extract values using various indexing methods.
        
        R Equivalent
        ------------
        neuroim2::`[.NeuroVec`
        """
        pass
    
    @abstractmethod
    def __setitem__(self, key, value):
        """Set values using various indexing methods.
        
        R Equivalent
        ------------
        neuroim2::`[<-.NeuroVec`
        """
        pass
    
    def series_3d(self, x: int, y: int, z: int) -> np.ndarray:
        """Extract time series for a single voxel using 3D coordinates.
        
        Parameters
        ----------
        x, y, z : int
            The 3D coordinates of the voxel
            
        Returns
        -------
        np.ndarray
            Time series for the specified voxel
            
        R Equivalent
        ------------
        neuroim2::series(vec, c(x+1, y+1, z+1))  # R uses 1-based indexing
        """
        return self.series(x, y, z)
    
    @abstractmethod
    def series(self, x, y=None, z=None) -> np.ndarray:
        """Extract time series for voxel(s).
        
        Parameters
        ----------
        x : int or array-like
            X coordinate or Nx3 matrix of coordinates
        y : int, optional
            Y coordinate (if x is int)
        z : int, optional
            Z coordinate (if x is int)
            
        Returns
        -------
        np.ndarray
            Time series data
            
        R Equivalent
        ------------
        neuroim2::series
        """
        pass
    
    def series_roi(self, roi) -> np.ndarray:
        """Extract time series for all voxels in an ROI.
        
        Parameters
        ----------
        roi : ROIVol or ROICoords
            The region of interest
            
        Returns
        -------
        np.ndarray
            Time series matrix (time x voxels)
            
        R Equivalent
            
        neuroim2::series_roi
        """
        from .roi import ROIVol, ROICoords
        
        if isinstance(roi, ROIVol):
            # Extract coordinates from ROIVol - it's directly stored in roi.coords
            coords = roi.coords
        elif isinstance(roi, ROICoords):
            # Use coordinates directly
            coords = roi.coords
        else:
            raise TypeError(f"roi must be ROIVol or ROICoords, got {type(roi)}")
        
        # Use the series method with coordinate matrix
        return self.series(coords)
    
    @abstractmethod
    def as_sparse(self, mask=None) -> 'SparseNeuroVec':
        """Convert to sparse representation.
        
        R Equivalent
        ------------
        neuroim2::as.sparse
        """
        pass
    
    @abstractmethod
    def sub_vector(self, indices: Union[int, slice, np.ndarray]) -> 'NeuroVec':
        """Extract subset of volumes.
        
        R Equivalent
        ------------
        neuroim2::sub_vector
        """
        pass
    
    def vols(self, indices=None):
        """Extract volumes as list or single volume.
        
        R Equivalent
        ------------
        neuroim2::vols
        """
        if indices is None:
            indices = range(self.shape[3])
            return [self[..., i] for i in indices]
        elif isinstance(indices, int):
            return self[..., indices]
        else:
            return [self[..., i] for i in indices]
    
    def concat(self, *others: 'NeuroVec') -> 'NeuroVec':
        """Concatenate multiple NeuroVecs along time dimension.
        
        R Equivalent
        ------------
        neuroim2::concat
        """
        # Concatenate by converting all to dense and stacking
        all_vecs = [self] + list(others)
        for vec in all_vecs[1:]:
            if vec.shape[:3] != self.shape[:3]:
                raise ValueError("All NeuroVecs must have same spatial dimensions")
        # Default: convert to dense and delegate
        dense_self = self.as_dense() if not isinstance(self, DenseNeuroVec) else self
        dense_others = [
            v.as_dense() if not isinstance(v, DenseNeuroVec) else v
            for v in others
        ]
        return dense_self.concat(*dense_others)
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.space.ndim

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Shape of the 4D data."""
        return tuple(int(d) for d in self.space.dim)

    @property
    def dim(self) -> np.ndarray:
        """Dimensions of the 4D data.
        
        R Equivalent
        ------------
        neuroim2::dim
        """
        return self.space.dim
    
    @property
    def spacing(self) -> np.ndarray:
        """Voxel dimensions.
        
        R Equivalent
        ------------
        neuroim2::spacing
        """
        return self.space.spacing
    
    @property
    def origin(self) -> np.ndarray:
        """Origin coordinates.
        
        R Equivalent
        ------------
        neuroim2::origin
        """
        return self.space.origin
    
    @property
    def trans(self) -> np.ndarray:
        """Transformation matrix.
        
        R Equivalent
        ------------
        neuroim2::trans
        """
        return self.space.trans
    
    def __repr__(self):
        """String representation."""
        return (f"{self.__class__.__name__}\n"
                f"  Type      : {self.__class__.__name__}\n"
                f"  Dimension : {' X '.join(map(str, self.dim))}\n"
                f"  Spacing   : {' X '.join(map(str, self.spacing))}\n"
                f"  Origin    : {', '.join(map(str, self.origin))}")
    
    # Arithmetic operations
    def __add__(self, other):
        """Add two vectors or vector and scalar/volume."""
        return self._arithmetic_op(other, np.add)
    
    def __sub__(self, other):
        """Subtract two vectors or vector and scalar/volume."""
        return self._arithmetic_op(other, np.subtract)
    
    def __mul__(self, other):
        """Multiply two vectors or vector and scalar/volume."""
        return self._arithmetic_op(other, np.multiply)
    
    def __truediv__(self, other):
        """Divide two vectors or vector and scalar/volume."""
        return self._arithmetic_op(other, np.divide)

    def __radd__(self, other):
        """Handle scalar/array + vector via reversed dispatch."""
        return self._reverse_arithmetic_op(other, np.add)

    def __rsub__(self, other):
        """Handle scalar/array - vector via reversed dispatch."""
        return self._reverse_arithmetic_op(other, np.subtract)

    def __rmul__(self, other):
        """Handle scalar/array * vector via reversed dispatch."""
        return self._reverse_arithmetic_op(other, np.multiply)

    def __rtruediv__(self, other):
        """Handle scalar/array / vector via reversed dispatch."""
        return self._reverse_arithmetic_op(other, np.divide)

    def _reverse_arithmetic_op(self, other, op):
        """Perform reversed arithmetic when right-hand side is this object."""
        return self._arithmetic_op(other, lambda x, y: op(y, x))
    
    @abstractmethod
    def _arithmetic_op(self, other, op):
        """Perform arithmetic operation."""
        pass


class DenseNeuroVec(NeuroVec):
    """Dense 4D neuroimaging vector.
    
    Direct translation of R's DenseNeuroVec class.
    
    Parameters
    ----------
    data : array-like
        4D array or matrix of voxel values
    space : NeuroSpace
        4D spatial metadata
    label : str, optional
        Vector label
        
    R Equivalent
    ------------
    neuroim2::DenseNeuroVec
    """
    
    def __init__(self, data, space: NeuroSpace, label: str = ""):
        super().__init__(space)
        
        # Handle different input types
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # Matrix input (time x voxels or voxels x time)
                splen = np.prod(self.shape[:3])
                if data.shape[0] == splen:
                    # voxels x time -> reshape to 4D
                    data = data.T.reshape(self.shape, order='F')
                elif data.shape[1] == splen:
                    # time x voxels -> reshape to 4D
                    data = data.reshape(self.shape, order='F')
                else:
                    raise ValueError("Matrix dimensions do not match space dimensions")
            elif data.ndim == 1:
                # Vector input
                if data.size == np.prod(self.shape):
                    data = data.reshape(self.shape, order='F')
                else:
                    raise ValueError(f"Data size {data.size} doesn't match space size {np.prod(self.shape)}")
            elif data.ndim == 4:
                if data.shape != self.shape:
                    raise ValueError(f"Data shape {data.shape} doesn't match space shape {self.shape}")
            else:
                raise ValueError(f"Data must be 1D, 2D or 4D array, got {data.ndim}D")
        else:
            data = np.asarray(data)
            
        self.data = data
        self.label = label
    
    def __getitem__(self, key):
        """Extract values using various indexing methods."""
        if isinstance(key, tuple) and len(key) == 4:
            # Standard 4D indexing
            return self.data[key]
        elif isinstance(key, tuple) and len(key) == 3 and all(isinstance(k, (int, np.integer)) for k in key):
            # Get time series for single voxel
            return self.data[key[0], key[1], key[2], :]
        else:
            # Let numpy handle it
            result = self.data[key]
            # If we extracted a single volume, wrap it as NeuroVol
            if result.ndim == 3:
                vol_space = NeuroSpace(result.shape, 
                                     spacing=self.spacing[:3],
                                     origin=self.origin[:3],
                                     axes=drop_axis(self.space.axes, 3) if self.space.ndim == 4 else None,
                                     trans=self.trans[:4, :4] if self.space.ndim <= 4 else None)
                return DenseNeuroVol(result, vol_space)
            return result
    
    def __setitem__(self, key, value):
        """Set values using various indexing methods."""
        self.data[key] = value
    
    def series_3d(self, x: int, y: int, z: int) -> np.ndarray:
        """Extract time series for a single voxel using 3D coordinates.
        
        Parameters
        ----------
        x, y, z : int
            The 3D coordinates of the voxel
            
        Returns
        -------
        np.ndarray
            Time series for the specified voxel
            
        R Equivalent
        ------------
        neuroim2::series(vec, c(x+1, y+1, z+1))  # R uses 1-based indexing
        """
        return self.series(x, y, z)
    
    def series(self, x, y=None, z=None) -> np.ndarray:
        """Extract time series for voxel(s)."""
        if y is not None and z is not None:
            # Single voxel
            return self.data[x, y, z, :]
        elif isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 3:
                # Nx3 matrix of coordinates - vectorized version
                # Check bounds all at once
                valid_mask = (
                    (x[:, 0] >= 0) & (x[:, 0] < self.shape[0]) &
                    (x[:, 1] >= 0) & (x[:, 1] < self.shape[1]) &
                    (x[:, 2] >= 0) & (x[:, 2] < self.shape[2])
                )
                
                # Initialize result
                result = np.zeros((x.shape[0], self.shape[3]))
                
                # Extract data for valid coordinates using advanced indexing
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    valid_coords = x[valid_indices]
                    result[valid_indices] = self.data[
                        valid_coords[:, 0],
                        valid_coords[:, 1], 
                        valid_coords[:, 2], :
                    ]
                
                return result.T  # Return as time x voxels
            elif x.ndim == 1:
                # Linear indices - vectorized version
                # Convert to 3D indices
                coords = np.unravel_index(x, self.shape[:3], order='F')
                # Use advanced indexing to extract all at once
                result = self.data[coords[0], coords[1], coords[2], :]
                return result.T
        elif isinstance(x, int):
            # Single linear index
            coords = np.unravel_index(x, self.shape[:3], order='F')
            return self.data[coords[0], coords[1], coords[2], :]
        else:
            raise ValueError("Invalid input for series extraction")
    
    def as_sparse(self, mask=None) -> 'SparseNeuroVec':
        """Convert to sparse representation."""
        if mask is None:
            # Use all non-zero voxels
            mask_data = np.any(self.data != 0, axis=3)
            mask_space = NeuroSpace(self.shape[:3], 
                                  spacing=self.spacing[:3],
                                  origin=self.origin[:3],
                                  axes=drop_axis(self.space.axes, 3) if self.space.ndim == 4 else None)
            mask = LogicalNeuroVol(mask_data, mask_space)
        elif not isinstance(mask, LogicalNeuroVol):
            # Convert to LogicalNeuroVol
            mask_space = NeuroSpace(self.shape[:3],
                                  spacing=self.spacing[:3],
                                  origin=self.origin[:3],
                                  axes=drop_axis(self.space.axes, 3) if self.space.ndim == 4 else None)
            mask = LogicalNeuroVol(mask, mask_space)
        
        # Extract data for masked voxels
        mask_indices = np.where(mask.data.ravel(order='F'))[0]
        data_flat = self.data.reshape(-1, self.shape[3], order='F')
        sparse_data = data_flat[mask_indices, :]
        
        return SparseNeuroVec(sparse_data.T, self.space, mask, self.label)
    
    def sub_vector(self, indices: Union[int, slice, np.ndarray]) -> 'DenseNeuroVec':
        """Extract subset of volumes."""
        if isinstance(indices, int):
            indices = [indices]
        
        sub_data = self.data[..., indices]
        if sub_data.ndim == 3:
            sub_data = sub_data[..., np.newaxis]
            
        sub_space = NeuroSpace((*self.shape[:3], sub_data.shape[3]),
                             spacing=self.spacing,
                             origin=self.origin,
                             axes=self.space.axes)
        
        return DenseNeuroVec(sub_data, sub_space, self.label)
    
    def vectors(self, subset=None):
        """Extract per-voxel time series as a list of 1D arrays.

        Parameters
        ----------
        subset : array-like, optional
            Linear indices of voxels to extract. If None, all voxels.

        Returns
        -------
        list of np.ndarray
            List of time series arrays, one per voxel.
        """
        flat = self.data.reshape(-1, self.shape[3], order='F')
        if subset is not None:
            return [flat[i] for i in subset]
        return [flat[i] for i in range(flat.shape[0])]

    def concat(self, *others: 'DenseNeuroVec') -> 'DenseNeuroVec':
        """Concatenate multiple NeuroVecs along time dimension."""
        all_vecs = [self] + list(others)
        
        # Check spatial compatibility
        for vec in all_vecs[1:]:
            if vec.shape[:3] != self.shape[:3]:
                raise ValueError("All NeuroVecs must have same spatial dimensions")
            if not np.allclose(vec.spacing[:3], self.spacing[:3]):
                raise ValueError("All NeuroVecs must have same spacing")
        
        # Concatenate data
        all_data = [vec.data for vec in all_vecs]
        concat_data = np.concatenate(all_data, axis=3)
        
        # Create new space with combined time dimension
        concat_space = NeuroSpace((*self.shape[:3], concat_data.shape[3]),
                                spacing=self.spacing,
                                origin=self.origin,
                                axes=self.space.axes)
        
        return DenseNeuroVec(concat_data, concat_space, self.label)
    
    def _arithmetic_op(self, other, op):
        """Perform arithmetic operation."""
        if isinstance(other, (int, float, np.integer, np.floating)):
            # Scalar operation
            result_data = op(self.data, other)
            return DenseNeuroVec(result_data, self.space, self.label)
        elif isinstance(other, DenseNeuroVec):
            # Vector-vector operation
            if other.shape != self.shape:
                raise ValueError("NeuroVecs must have same shape for arithmetic")
            result_data = op(self.data, other.data)
            return DenseNeuroVec(result_data, self.space, self.label)
        elif isinstance(other, NeuroVol):
            # Vector-volume operation (broadcast volume across time)
            if other.shape != self.shape[:3]:
                raise ValueError("NeuroVol must match spatial dimensions of NeuroVec")
            vol_data = other.data[..., np.newaxis]  # Add time dimension
            result_data = op(self.data, vol_data)
            return DenseNeuroVec(result_data, self.space, self.label)
        elif isinstance(other, np.ndarray):
            # ndarray broadcasting (e.g. time weights, spatial weights)
            result_data = op(self.data, other)
            if result_data.shape == self.data.shape:
                return DenseNeuroVec(result_data, self.space, self.label)
            return result_data
        else:
            return NotImplemented
    
    def as_matrix(self) -> np.ndarray:
        """Convert to matrix (voxels x time).
        
        R Equivalent
        ------------
        neuroim2::as.matrix
        """
        return self.data.reshape(-1, self.shape[3], order='F')
    
    def scale_series(self, center: bool = True, scale: bool = True) -> 'DenseNeuroVec':
        """Scale (center and/or normalize) each time series.
        
        R Equivalent
        ------------
        neuroim2::scale_series
        """
        data = self.data.copy()
        
        if center:
            # Center each time series
            mean = np.mean(data, axis=3, keepdims=True)
            data = data - mean
            
        if scale:
            # Scale by standard deviation
            std = np.std(data, axis=3, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            data = data / std
            
        return DenseNeuroVec(data, self.space, self.label)


class SparseNeuroVec(NeuroVec):
    """Sparse 4D neuroimaging vector.
    
    Direct translation of R's SparseNeuroVec class.
    
    Parameters
    ----------
    data : np.ndarray
        2D array (time x masked_voxels)
    space : NeuroSpace
        4D spatial metadata
    mask : LogicalNeuroVol
        Mask defining which voxels are included
    label : str, optional
        Vector label
        
    R Equivalent
    ------------
    neuroim2::SparseNeuroVec
    """
    
    def __init__(self, data: np.ndarray, space: NeuroSpace, mask, label: str = ""):
        super().__init__(space)

        # Accept indices array and convert to LogicalNeuroVol
        if isinstance(mask, np.ndarray) and not mask.dtype == bool:
            # Integer indices array - convert to LogicalNeuroVol
            mask_data = np.zeros(self.shape[:3], dtype=bool)
            mask_flat = mask_data.ravel(order='F')
            mask_flat[mask] = True
            mask_data = mask_flat.reshape(self.shape[:3], order='F')
            from .neuro_vol import LogicalNeuroVol as LNV
            mask_space = NeuroSpace(self.shape[:3], spacing=space.spacing[:3], origin=space.origin[:3])
            mask = LNV(mask_data, mask_space)
        elif isinstance(mask, np.ndarray) and mask.dtype == bool:
            # Boolean array - convert to LogicalNeuroVol
            from .neuro_vol import LogicalNeuroVol as LNV
            mask_space = NeuroSpace(self.shape[:3], spacing=space.spacing[:3], origin=space.origin[:3])
            mask = LNV(mask, mask_space)
        elif not isinstance(mask, LogicalNeuroVol):
            raise TypeError("mask must be a LogicalNeuroVol, boolean array, or integer indices array")
            
        if mask.shape != self.shape[:3]:
            raise ValueError("Mask dimensions must match spatial dimensions of space")
        
        # Handle data dimensionality
        if data.ndim == 2:
            if data.shape[0] == self.shape[3] and data.shape[1] == mask.sum:
                # Correct orientation (time x voxels)
                pass
            elif data.shape[1] == self.shape[3] and data.shape[0] == mask.sum:
                # Need to transpose (voxels x time -> time x voxels)
                data = data.T
            else:
                raise ValueError(f"Data shape {data.shape} doesn't match mask cardinality {mask.sum} and time dimension {self.shape[3]}")
        else:
            raise ValueError("Data must be 2D array (time x masked_voxels)")
        
        self.data = data
        self.mask = mask
        self.label = label
        
        # Create lookup for fast indexing
        self._lookup = np.where(mask.data.ravel(order='F'))[0]
        self._inverse_lookup = np.full(np.prod(self.shape[:3]), -1, dtype=int)
        self._inverse_lookup[self._lookup] = np.arange(len(self._lookup))
    
    def __getitem__(self, key):
        """Extract values using various indexing methods."""
        # Create dense version for complex indexing
        dense_data = self._to_dense_array()
        result = dense_data[key]
        
        # If we extracted a single volume, wrap it as NeuroVol
        if result.ndim == 3:
            vol_space = NeuroSpace(result.shape,
                                 spacing=self.spacing[:3],
                                 origin=self.origin[:3],
                                 axes=drop_axis(self.space.axes, 3) if self.space.ndim == 4 else None)
            return DenseNeuroVol(result, vol_space)
        return result
    
    def __setitem__(self, key, value):
        """Set values using various indexing methods."""
        # For now, convert to dense, modify, convert back
        # This is inefficient but ensures correctness
        dense_data = self._to_dense_array()
        dense_data[key] = value
        
        # Extract updated sparse data
        data_flat = dense_data.reshape(-1, self.shape[3], order='F')
        self.data = data_flat[self._lookup, :].T
    
    def _to_dense_array(self) -> np.ndarray:
        """Convert sparse data to dense 4D array."""
        dense = np.zeros(self.shape, dtype=self.data.dtype, order='F')
        data_flat = dense.reshape(-1, self.shape[3], order='F')
        data_flat[self._lookup, :] = self.data.T
        return dense
    
    def series(self, x, y=None, z=None) -> np.ndarray:
        """Extract time series for voxel(s)."""
        if y is not None and z is not None:
            # Single voxel
            linear_idx = np.ravel_multi_index((x, y, z), self.shape[:3], order='F')
            sparse_idx = self._inverse_lookup[linear_idx]
            if sparse_idx == -1:
                return np.zeros(self.shape[3])
            return self.data[:, sparse_idx]
        elif isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 3:
                # Nx3 matrix of coordinates
                result = np.zeros((self.shape[3], x.shape[0]))
                for i, coord in enumerate(x):
                    linear_idx = np.ravel_multi_index(coord, self.shape[:3], order='F')
                    sparse_idx = self._inverse_lookup[linear_idx]
                    if sparse_idx != -1:
                        result[:, i] = self.data[:, sparse_idx]
                return result
            elif x.ndim == 1:
                # Linear indices
                result = np.zeros((self.shape[3], len(x)))
                for i, idx in enumerate(x):
                    sparse_idx = self._inverse_lookup[idx]
                    if sparse_idx != -1:
                        result[:, i] = self.data[:, sparse_idx]
                return result
        elif isinstance(x, int):
            # Single linear index
            sparse_idx = self._inverse_lookup[x]
            if sparse_idx == -1:
                return np.zeros(self.shape[3])
            return self.data[:, sparse_idx]
        else:
            raise ValueError("Invalid input for series extraction")
    
    def as_sparse(self, mask=None) -> 'SparseNeuroVec':
        """Already sparse, return self or apply new mask."""
        if mask is None:
            return self
        else:
            # Apply additional mask by converting to dense, then back to sparse with new mask
            dense = self.as_dense()
            return dense.as_sparse(mask)
    
    def as_dense(self) -> DenseNeuroVec:
        """Convert to dense representation."""
        dense_data = self._to_dense_array()
        return DenseNeuroVec(dense_data, self.space, self.label)
    
    def sub_vector(self, indices: Union[int, slice, np.ndarray]) -> 'SparseNeuroVec':
        """Extract subset of volumes."""
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(self.shape[3])))
        
        sub_data = self.data[indices, :]
        
        sub_space = NeuroSpace((*self.shape[:3], len(indices)),
                             spacing=self.spacing,
                             origin=self.origin,
                             axes=self.space.axes)
        
        return SparseNeuroVec(sub_data, sub_space, self.mask, self.label)
    
    def concat(self, *others: 'SparseNeuroVec') -> 'SparseNeuroVec':
        """Concatenate multiple SparseNeuroVecs along time dimension."""
        all_vecs = [self] + list(others)
        
        # Check compatibility
        for vec in all_vecs[1:]:
            if not np.array_equal(vec.mask.data, self.mask.data):
                raise ValueError("All SparseNeuroVecs must have same mask")
        
        # Concatenate data
        all_data = [vec.data for vec in all_vecs]
        concat_data = np.vstack(all_data)
        
        # Create new space
        concat_space = NeuroSpace((*self.shape[:3], concat_data.shape[0]),
                                spacing=self.spacing,
                                origin=self.origin,
                                axes=self.space.axes)
        
        return SparseNeuroVec(concat_data, concat_space, self.mask, self.label)
    
    def _arithmetic_op(self, other, op):
        """Perform arithmetic operation."""
        if isinstance(other, (int, float)):
            # Scalar operation
            result_data = op(self.data, other)
            return SparseNeuroVec(result_data, self.space, self.mask, self.label)
        elif isinstance(other, SparseNeuroVec):
            # Sparse-sparse operation
            if not np.array_equal(other.mask.data, self.mask.data):
                raise ValueError("SparseNeuroVecs must have same mask for arithmetic")
            result_data = op(self.data, other.data)
            return SparseNeuroVec(result_data, self.space, self.mask, self.label)
        elif isinstance(other, DenseNeuroVec):
            # Convert to dense for operation
            return self.as_dense()._arithmetic_op(other, op)
        elif isinstance(other, NeuroVol):
            # Vector-volume operation
            if other.shape != self.shape[:3]:
                raise ValueError("NeuroVol must match spatial dimensions of NeuroVec")
            # Extract volume values at mask locations
            vol_masked = other.values()[self._lookup]
            # Broadcast across time
            result_data = op(self.data, vol_masked[np.newaxis, :])
            return SparseNeuroVec(result_data, self.space, self.mask, self.label)
        else:
            return NotImplemented


def neurovecseq(vecs: List, label: str = "") -> NeuroVec:
    """Create NeuroVec from sequence of volumes or vectors.
    
    Parameters
    ----------
    vecs : list
        List of NeuroVol objects or DenseNeuroVec objects
    label : str, optional
        Label for the result
        
    Returns
    -------
    NeuroVec
        Combined 4D vector (DenseNeuroVec or SparseNeuroVec, depending on input).
        
    R Equivalent
    ------------
    neuroim2::NeuroVecSeq
    """
    if not vecs:
        raise ValueError("Empty vector list")
    
    first = vecs[0]
    
    if isinstance(first, NeuroVol):
        # List of volumes
        space_3d = first.space
        
        # Check all volumes have same space
        for vol in vecs[1:]:
            if vol.shape != first.shape:
                raise ValueError("All volumes must have same dimensions")
            if not np.allclose(vol.spacing, first.spacing):
                raise ValueError("All volumes must have same spacing")
        
        # Stack volumes
        data = np.stack([vol.data for vol in vecs], axis=3)
        
        # Create 4D space - add_dim takes (n, size) parameters
        space_4d = space_3d.add_dim(1, len(vecs))
        
        return DenseNeuroVec(data, space_4d, label)
    
    elif isinstance(first, DenseNeuroVec):
        # List of vectors - concatenate
        return first.concat(*vecs[1:])

    elif isinstance(first, SparseNeuroVec):
        # List of sparse vectors - concatenate with sparse semantics
        return first.concat(*vecs[1:])
    
    else:
        raise TypeError("Input must be list of NeuroVol or DenseNeuroVec objects")


def neurovec(data, space: NeuroSpace = None, mask=None, label: str = "") -> NeuroVec:
    """Factory function to create appropriate NeuroVec type.
    
    Parameters
    ----------
    data : array-like or list
        The image data
    space : NeuroSpace, optional
        4D spatial metadata
    mask : LogicalNeuroVol, optional
        Mask for sparse representation
    label : str, optional
        Vector label
        
    Returns
    -------
    NeuroVec
        DenseNeuroVec or SparseNeuroVec
        
    R Equivalent
    ------------
    neuroim2::NeuroVec
    """
    # Handle list of volumes
    if isinstance(data, list):
        return neurovecseq(data, label)
    
    # Convert data to array
    data = np.asarray(data)
    
    # Create default space if needed
    if space is None:
        if data.ndim == 4:
            space = NeuroSpace(data.shape)
        else:
            raise ValueError("Cannot infer space from non-4D data")
    
    # Create appropriate type
    if mask is None:
        return DenseNeuroVec(data, space, label)
    else:
        # For sparse, need to ensure mask is LogicalNeuroVol
        if not isinstance(mask, LogicalNeuroVol):
            mask_space = NeuroSpace(space.dim[:3],
                                  spacing=space.spacing[:3],
                                  origin=space.origin[:3],
                                  axes=drop_axis(space.axes, 3) if space.ndim == 4 else None)
            mask = LogicalNeuroVol(mask, mask_space)
        
        # If data is 4D, extract sparse representation
        if data.ndim == 4:
            mask_indices = np.where(mask.data.ravel(order='F'))[0]
            data_flat = data.reshape(-1, data.shape[3], order='F')
            sparse_data = data_flat[mask_indices, :].T
            return SparseNeuroVec(sparse_data, space, mask, label)
        else:
            return SparseNeuroVec(data, space, mask, label)

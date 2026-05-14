"""5D+ Neuroimaging HyperVector Classes.

NeuroHyperVec represents neuroimaging data with:
- 3 spatial dimensions (x, y, z) 
- 1 time dimension
- 1+ feature dimensions (e.g., frequency bands, model parameters, echo times)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple, List, Optional, Callable
import warnings

from .neuro_space import NeuroSpace
from .neuro_vec import NeuroVec, DenseNeuroVec
from .neuro_vol import NeuroVol, DenseNeuroVol, LogicalNeuroVol
from .neuro_vec import SparseNeuroVec


class NeuroHyperVec(ABC):
    """Abstract base class for 5D+ neuroimaging hypervectors.
    
    Represents data with spatial dimensions, time, and feature dimensions.
    
    Parameters
    ----------
    space : NeuroSpace
        5D+ spatial metadata
    label : str, optional
        Label for the hypervector
    """
    
    def __init__(self, space: NeuroSpace, label: str = ""):
        if space.ndim < 5:
            raise ValueError(f"NeuroHyperVec requires at least 5D space, got {space.ndim}D")
        self.space = space
        self.label = label
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the data array."""
        return tuple(self.space.dim)
    
    @property
    def dim(self) -> np.ndarray:
        """Dimensions of the data."""
        return self.space.dim
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return self.space.ndim
    
    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        """Shape of spatial dimensions (x, y, z)."""
        return tuple(self.space.dim[:3])
    
    @property
    def n_timepoints(self) -> int:
        """Number of time points."""
        return self.space.dim[3]
    
    @property
    def n_features(self) -> int:
        """Number of features (5th dimension)."""
        return self.space.dim[4]
    
    @property
    def spacing(self) -> np.ndarray:
        """Voxel spacing."""
        return self.space.spacing
    
    @property
    def origin(self) -> np.ndarray:
        """Origin coordinates."""
        return self.space.origin
    
    @abstractmethod
    def __getitem__(self, key):
        """Extract values using indexing."""
        pass
    
    @abstractmethod
    def series(self, coords: Union[List, np.ndarray], feature: Optional[int] = None) -> np.ndarray:
        """Extract time series for given coordinates.
        
        Parameters
        ----------
        coords : array-like
            Spatial coordinates (single voxel or multiple)
        feature : int, optional
            Specific feature index to extract
            
        Returns
        -------
        np.ndarray
            Time series data
        """
        pass
    
    @abstractmethod
    def mean_features(self) -> NeuroVec:
        """Average across feature dimension."""
        pass
    
    @abstractmethod
    def select_features(self, indices: List[int]) -> 'NeuroHyperVec':
        """Select subset of features."""
        pass
    
    @abstractmethod
    def apply_feature_func(self, func: Callable) -> NeuroVec:
        """Apply function across features at each voxel/time."""
        pass


class DenseNeuroHyperVec(NeuroHyperVec):
    """Dense 5D+ neuroimaging hypervector.
    
    Parameters
    ----------
    data : np.ndarray
        5D+ data array
    space : NeuroSpace
        5D+ spatial metadata
    label : str, optional
        Label for the hypervector
    """
    
    def __init__(self, data: np.ndarray, space: NeuroSpace, label: str = ""):
        super().__init__(space, label)
        
        # Convert and validate data
        self.data = np.asarray(data, order='C')
        
        if self.data.shape != self.shape:
            raise ValueError(f"Data shape {self.data.shape} doesn't match space shape {self.shape}")
    
    def __getitem__(self, key):
        """Extract values using indexing."""
        result = self.data[key]
        
        # If we extracted a 4D subset (removed feature dimension), return NeuroVec
        if isinstance(key, tuple) and len(key) == self.n_dims:
            # Check if we selected a single feature
            if isinstance(key[4], (int, np.integer)):
                # Create 4D space for the result
                space_4d = NeuroSpace(
                    dim=self.space.dim[:4],
                    spacing=self.space.spacing[:4],
                    origin=self.space.origin[:4]
                )
                return DenseNeuroVec(result, space_4d, self.label)
        
        return result
    
    def series(self, coords: Union[List, np.ndarray], feature: Optional[int] = None) -> np.ndarray:
        """Extract time series for given coordinates."""
        coords = np.atleast_2d(coords)
        
        if coords.shape[1] != 3:
            raise ValueError("Coordinates must be 3D (x, y, z)")
        
        if feature is not None:
            # Extract specific feature
            if coords.shape[0] == 1:
                # Single voxel
                return self.data[coords[0, 0], coords[0, 1], coords[0, 2], :, feature]
            else:
                # Multiple voxels
                result = np.zeros((self.n_timepoints, coords.shape[0]))
                for i, coord in enumerate(coords):
                    result[:, i] = self.data[coord[0], coord[1], coord[2], :, feature]
                return result
        else:
            # Extract all features
            if coords.shape[0] == 1:
                # Single voxel: time x features
                return self.data[coords[0, 0], coords[0, 1], coords[0, 2], :, :]
            else:
                # Multiple voxels: time x features x voxels
                result = np.zeros((self.n_timepoints, self.n_features, coords.shape[0]))
                for i, coord in enumerate(coords):
                    result[:, :, i] = self.data[coord[0], coord[1], coord[2], :, :]
                return result
    
    def mean_features(self) -> DenseNeuroVec:
        """Average across feature dimension."""
        # Average over the last dimension (features)
        mean_data = np.mean(self.data, axis=-1)
        
        # Create 4D space
        space_4d = NeuroSpace(
            dim=self.space.dim[:4],
            spacing=self.space.spacing[:4],
            origin=self.space.origin[:4]
        )
        
        return DenseNeuroVec(mean_data, space_4d, self.label)
    
    def std_features(self, time_idx: Optional[int] = None) -> Union[DenseNeuroVol, np.ndarray]:
        """Standard deviation across features.
        
        Parameters
        ----------
        time_idx : int, optional
            If provided, compute std at specific time point and return NeuroVol
            
        Returns
        -------
        DenseNeuroVol or np.ndarray
            3D volume if time_idx provided, otherwise 4D array
        """
        if time_idx is not None:
            # Std at specific time point
            std_data = np.std(self.data[:, :, :, time_idx, :], axis=-1)
            
            # Create 3D space
            space_3d = NeuroSpace(
                dim=self.space.dim[:3],
                spacing=self.space.spacing[:3],
                origin=self.space.origin[:3]
            )
            
            return DenseNeuroVol(std_data, space_3d, self.label)
        else:
            # Std across all time points
            return np.std(self.data, axis=-1)
    
    def select_features(self, indices: List[int]) -> 'DenseNeuroHyperVec':
        """Select subset of features."""
        indices = np.asarray(indices)
        
        # Extract selected features
        selected_data = self.data[:, :, :, :, indices]
        
        # Create new space with updated feature dimension
        new_dim = list(self.space.dim)
        new_dim[4] = len(indices)
        
        new_space = NeuroSpace(
            dim=new_dim,
            spacing=self.space.spacing,
            origin=self.space.origin
        )
        
        return DenseNeuroHyperVec(selected_data, new_space, self.label)
    
    def weighted_mean_features(self, weights: np.ndarray) -> DenseNeuroVec:
        """Compute weighted mean across features.
        
        Parameters
        ----------
        weights : np.ndarray
            Weights for each feature
            
        Returns
        -------
        DenseNeuroVec
            4D weighted average
        """
        if len(weights) != self.n_features:
            raise ValueError(f"Number of weights ({len(weights)}) must match number of features ({self.n_features})")
        
        # Normalize weights
        weights = np.asarray(weights)
        weights = weights / weights.sum()
        
        # Apply weights and sum
        weighted_data = np.zeros(self.shape[:4])
        for i, w in enumerate(weights):
            weighted_data += self.data[:, :, :, :, i] * w
        
        # Create 4D space
        space_4d = NeuroSpace(
            dim=self.space.dim[:4],
            spacing=self.space.spacing[:4],
            origin=self.space.origin[:4]
        )
        
        return DenseNeuroVec(weighted_data, space_4d, self.label)
    
    def apply_feature_func(self, func: Callable) -> DenseNeuroVec:
        """Apply function across features at each voxel/time."""
        # Initialize output
        output_shape = self.shape[:4]
        output_data = np.zeros(output_shape)
        
        # Apply function at each voxel/time
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    for t in range(self.shape[3]):
                        feature_vec = self.data[i, j, k, t, :]
                        output_data[i, j, k, t] = func(feature_vec)
        
        # Create 4D space
        space_4d = NeuroSpace(
            dim=self.space.dim[:4],
            spacing=self.space.spacing[:4],
            origin=self.space.origin[:4]
        )
        
        return DenseNeuroVec(output_data, space_4d, self.label)
    
    def squeeze_features(self) -> DenseNeuroVec:
        """Remove single feature dimension."""
        if self.n_features != 1:
            raise ValueError(f"Can only squeeze single feature dimension, got {self.n_features} features")
        
        # Remove last dimension
        squeezed_data = self.data.squeeze(axis=-1)
        
        # Create 4D space
        space_4d = NeuroSpace(
            dim=self.space.dim[:4],
            spacing=self.space.spacing[:4],
            origin=self.space.origin[:4]
        )
        
        return DenseNeuroVec(squeezed_data, space_4d, self.label)
    
    def mean_time_features(self) -> DenseNeuroVol:
        """Average across both time and feature dimensions."""
        # Average over time (axis 3) and features (axis 4)
        mean_data = np.mean(self.data, axis=(3, 4))
        
        # Create 3D space
        space_3d = NeuroSpace(
            dim=self.space.dim[:3],
            spacing=self.space.spacing[:3],
            origin=self.space.origin[:3]
        )
        
        return DenseNeuroVol(mean_data, space_3d, self.label)


class SparseNeuroHyperVec(NeuroHyperVec):
    """Sparse 5D+ neuroimaging hypervector.
    
    Parameters
    ----------
    data : np.ndarray
        Sparse data array with shape (n_features, n_timepoints, n_voxels)
    mask : LogicalNeuroVol
        3D mask indicating which voxels contain data
    space : NeuroSpace
        5D+ spatial metadata
    label : str, optional
        Label for the hypervector
    """
    
    def __init__(self, data: np.ndarray, mask: LogicalNeuroVol, space: NeuroSpace, label: str = ""):
        super().__init__(space, label)
        
        if mask.shape != self.spatial_shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match spatial shape {self.spatial_shape}")
        
        self.mask = mask
        self.data = np.asarray(data, order='C')
        
        # Validate data shape
        n_voxels = mask.sum
        expected_shape = (self.n_features, self.n_timepoints, n_voxels)
        if self.data.shape != expected_shape:
            raise ValueError(f"Data shape {self.data.shape} doesn't match expected shape {expected_shape}")
        
        # Create lookup for voxel indices
        self._indices = np.where(mask.data.ravel(order='F'))[0]
        self._lookup = np.full(np.prod(self.spatial_shape), -1, dtype=int)
        self._lookup[self._indices] = np.arange(len(self._indices))
    
    def __getitem__(self, key):
        """Extract values using indexing."""
        # Convert to dense for complex indexing
        return self.as_dense()[key]
    
    def as_dense(self) -> DenseNeuroHyperVec:
        """Convert to dense representation."""
        # Initialize dense array
        dense_data = np.zeros(self.shape)
        
        # Fill in sparse data
        mask_coords = np.column_stack(np.where(self.mask.data))
        for i, coord in enumerate(mask_coords):
            dense_data[coord[0], coord[1], coord[2], :, :] = self.data[:, :, i].T
        
        return DenseNeuroHyperVec(dense_data, self.space, self.label)
    
    def series(self, coords: Union[List, np.ndarray], feature: Optional[int] = None) -> np.ndarray:
        """Extract time series for given coordinates."""
        coords = np.atleast_2d(coords)
        
        if coords.shape[1] != 3:
            raise ValueError("Coordinates must be 3D (x, y, z)")
        
        result_list = []
        for coord in coords:
            # Get linear index
            linear_idx = np.ravel_multi_index(coord, self.spatial_shape, order='F')
            voxel_idx = self._lookup[linear_idx]
            
            if voxel_idx == -1:
                # Voxel not in mask
                if feature is not None:
                    result_list.append(np.zeros(self.n_timepoints))
                else:
                    result_list.append(np.zeros((self.n_timepoints, self.n_features)))
            else:
                if feature is not None:
                    result_list.append(self.data[feature, :, voxel_idx])
                else:
                    result_list.append(self.data[:, :, voxel_idx].T)
        
        if len(result_list) == 1:
            return result_list[0]
        else:
            if feature is not None:
                return np.column_stack(result_list)
            else:
                return np.stack(result_list, axis=2)
    
    def mean_features(self) -> SparseNeuroVec:
        """Average across feature dimension."""
        # Average over features (first dimension)
        mean_data = np.mean(self.data, axis=0)  # Shape: (n_timepoints, n_voxels)
        
        # Create 4D space
        space_4d = NeuroSpace(
            dim=self.space.dim[:4],
            spacing=self.space.spacing[:4],
            origin=self.space.origin[:4]
        )
        
        # Transpose to match SparseNeuroVec format (n_voxels, n_timepoints)
        return SparseNeuroVec(mean_data.T, space_4d, self.mask, self.label)
    
    def select_features(self, indices: List[int]) -> 'SparseNeuroHyperVec':
        """Select subset of features."""
        indices = np.asarray(indices)
        
        # Extract selected features
        selected_data = self.data[indices, :, :]
        
        # Create new space with updated feature dimension
        new_dim = list(self.space.dim)
        new_dim[4] = len(indices)
        
        new_space = NeuroSpace(
            dim=new_dim,
            spacing=self.space.spacing,
            origin=self.space.origin
        )
        
        return SparseNeuroHyperVec(selected_data, self.mask, new_space, self.label)
    
    def apply_feature_func(self, func: Callable) -> SparseNeuroVec:
        """Apply function across features at each voxel/time."""
        # Initialize output
        n_voxels = self.mask.sum
        output_data = np.zeros((n_voxels, self.n_timepoints))
        
        # Apply function at each voxel/time
        for v in range(n_voxels):
            for t in range(self.n_timepoints):
                feature_vec = self.data[:, t, v]
                output_data[v, t] = func(feature_vec)
        
        # Create 4D space
        space_4d = NeuroSpace(
            dim=self.space.dim[:4],
            spacing=self.space.spacing[:4],
            origin=self.space.origin[:4]
        )
        
        return SparseNeuroVec(output_data, space_4d, self.mask, self.label)


def concat_features(hypervecs: List[NeuroHyperVec]) -> DenseNeuroHyperVec:
    """Concatenate NeuroHyperVecs along feature dimension.
    
    Parameters
    ----------
    hypervecs : list of NeuroHyperVec
        Hypervectors to concatenate (must have same spatial/time dimensions)
        
    Returns
    -------
    DenseNeuroHyperVec
        Concatenated hypervector
    """
    if not hypervecs:
        raise ValueError("Empty list of hypervectors")
    
    first = hypervecs[0]
    
    # Check compatibility
    for hv in hypervecs[1:]:
        if hv.shape[:4] != first.shape[:4]:
            raise ValueError("All hypervectors must have same spatial and time dimensions")
    
    # Convert sparse to dense if needed
    dense_hvs = []
    for hv in hypervecs:
        if isinstance(hv, SparseNeuroHyperVec):
            dense_hvs.append(hv.as_dense())
        else:
            dense_hvs.append(hv)
    
    # Concatenate along feature dimension (axis 4)
    concat_data = np.concatenate([hv.data for hv in dense_hvs], axis=4)
    
    # Create new space with combined features
    total_features = sum(hv.n_features for hv in hypervecs)
    new_dim = list(first.space.dim)
    new_dim[4] = total_features
    
    new_space = NeuroSpace(
        dim=new_dim,
        spacing=first.space.spacing,
        origin=first.space.origin
    )
    
    return DenseNeuroHyperVec(concat_data, new_space)


# I/O functions
def write_neurohypervec(hvec: NeuroHyperVec, filename: str):
    """Write NeuroHyperVec to HDF5 file.
    
    Parameters
    ----------
    hvec : NeuroHyperVec
        Hypervector to save
    filename : str
        Output filename (should end with .h5)
    """
    import h5py
    
    with h5py.File(filename, 'w') as f:
        # Save data
        if isinstance(hvec, SparseNeuroHyperVec):
            f.create_dataset('data', data=hvec.data)
            f.create_dataset('mask', data=hvec.mask.data)
            f.attrs['type'] = 'sparse'
        else:
            f.create_dataset('data', data=hvec.data)
            f.attrs['type'] = 'dense'
        
        # Save metadata
        f.attrs['shape'] = hvec.shape
        f.attrs['spacing'] = hvec.spacing
        f.attrs['origin'] = hvec.origin
        f.attrs['label'] = hvec.label


def read_neurohypervec(filename: str) -> NeuroHyperVec:
    """Read NeuroHyperVec from HDF5 file.
    
    Parameters
    ----------
    filename : str
        Input filename
        
    Returns
    -------
    NeuroHyperVec
        Loaded hypervector
    """
    import h5py
    
    with h5py.File(filename, 'r') as f:
        # Load metadata
        shape = f.attrs['shape']
        spacing = f.attrs['spacing']
        origin = f.attrs['origin']
        label = f.attrs.get('label', '')
        hvec_type = f.attrs.get('type', 'dense')
        
        # Create space
        space = NeuroSpace(dim=shape, spacing=spacing, origin=origin)
        
        # Load data
        data = f['data'][:]
        
        if hvec_type == 'sparse':
            # Load mask
            mask_data = f['mask'][:]
            mask_space = NeuroSpace(dim=shape[:3], spacing=spacing[:3], origin=origin[:3])
            mask = LogicalNeuroVol(mask_data, mask_space)
            return SparseNeuroHyperVec(data, mask, space, label)
        else:
            return DenseNeuroHyperVec(data, space, label)


# Factory function
def NeuroHyperVec(data: Union[np.ndarray, str], space: NeuroSpace, 
                  mask: Optional[LogicalNeuroVol] = None, label: str = "") -> NeuroHyperVec:
    """Factory function to create appropriate NeuroHyperVec type.
    
    Parameters
    ----------
    data : array-like or str
        The hypervector data or filename for memory-mapped
    space : NeuroSpace
        5D+ spatial metadata
    mask : LogicalNeuroVol, optional
        Mask for sparse representation
    label : str, optional
        Hypervector label
        
    Returns
    -------
    NeuroHyperVec
        Appropriate hypervector type
    """
    data = np.asarray(data)

    if mask is not None:
        from .verify import assert_same_space

        assert_same_space(space, mask)
        # Sparse representation
        return SparseNeuroHyperVec(data, mask, space, label)
    else:
        # Dense representation
        return DenseNeuroHyperVec(data, space, label)

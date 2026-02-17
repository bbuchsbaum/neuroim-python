"""BigNeuroVec - Memory-mapped neuroimaging vector data.

This module provides a memory-mapped implementation of NeuroVec
that can handle very large 4D datasets efficiently by keeping data on disk.

Direct translation of R's neuroim2 BigNeuroVec class.
"""

import numpy as np
import tempfile
import os
from typing import Optional, Union, Tuple
from .neuro_vec import NeuroVec
from .neuro_space import NeuroSpace
from .neuro_vol import NeuroVol, DenseNeuroVol


class BigNeuroVec(NeuroVec):
    """A memory-mapped 4D neuroimaging vector.
    
    This class provides efficient handling of large 4D neuroimaging data
    by using memory-mapped arrays that remain on disk.
    
    Parameters
    ----------
    data : array-like
        The 4D data array. If a regular array, will be copied to a
        memory-mapped file. If already a memmap, will be used directly.
    space : NeuroSpace
        The 4D spatial metadata
    filename : str, optional
        Path to the memory-mapped file. If not provided, a temporary
        file will be created.
    mode : str, optional
        File mode for memory mapping. Default is 'r+' (read/write).
        Use 'r' for read-only access.
    
    R Equivalent
    ------------
    neuroim2::BigNeuroVec
    """
    
    def __init__(self, data: Optional[Union[np.ndarray, np.memmap, str]] = None, 
                 space: Optional[NeuroSpace] = None,
                 filename: Optional[str] = None,
                 mode: str = 'r+',
                 shape: Optional[Tuple[int, ...]] = None,
                 dtype: np.dtype = np.float32):
        """Initialize BigNeuroVec.
        
        Can be initialized in two ways:
        1. With data and space (original way)
        2. With filename, shape, and dtype (for creating empty files)
        """
        # Handle case where first argument is a string (filename)
        if isinstance(data, str) and space is None:
            # First argument is actually a filename
            filename = data
            data = None
        
        # Handle two different initialization patterns
        if data is not None and space is not None:
            # Original initialization with data
            super().__init__(space)
            
            # Validate data shape
            data = np.asarray(data, order='C')  # Ensure C-contiguous
            expected_shape = tuple(space.dim)
            if data.shape != expected_shape:
                raise ValueError(f"data shape {data.shape} does not match space dimensions {expected_shape}")
            
            # Handle filename
            if filename is None:
                # Create temporary file
                self._temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
                self.filename = self._temp_file.name
                self._temp_file.close()  # Must close before using with memmap
                self._owns_file = True
            else:
                self.filename = filename
                self._owns_file = False
                self._temp_file = None
            
            # Create or open memory-mapped array
            if isinstance(data, np.memmap) and hasattr(data, 'filename') and data.filename == self.filename:
                # Already a memmap for this file
                self._data = data
            else:
                # Always create new file with data
                # Create new file - ensure parent directory exists if specified
                dirname = os.path.dirname(self.filename)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
                # Create and populate memmap
                fp = np.memmap(self.filename, dtype=data.dtype, 
                               mode='w+', shape=data.shape, order='C')
                fp[:] = data
                fp.flush()
                del fp  # Close the file
                # Re-open in requested mode
                self._data = np.memmap(self.filename, dtype=data.dtype,
                                      mode=mode, shape=data.shape, order='C')
        
        elif filename is not None and shape is not None:
            # New initialization pattern for empty files
            # Create NeuroSpace from shape
            if len(shape) != 4:
                raise ValueError("BigNeuroVec requires 4D shape")
            
            # Create space with default spacing and origin
            space = NeuroSpace(
                dim=list(shape),
                spacing=[1.0] * 4,
                origin=[0.0] * 4
            )
            super().__init__(space)
            
            self.filename = filename
            self._owns_file = False
            self._temp_file = None
            
            # Create empty memory-mapped file
            dirname = os.path.dirname(self.filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
                
            # Create new memmap file
            self._data = np.memmap(self.filename, dtype=dtype,
                                  mode='w+', shape=shape, order='C')
            # Initialize with zeros
            self._data[:] = 0
            self._data.flush()
            
            # Re-open in requested mode
            del self._data
            self._data = np.memmap(self.filename, dtype=dtype,
                                  mode=mode, shape=shape, order='C')
        else:
            raise ValueError("BigNeuroVec requires either (data, space) or (filename, shape)")
            
        self.mode = mode
    
    @property
    def data(self) -> np.memmap:
        """Get the memory-mapped data array."""
        return self._data
    
    def __getitem__(self, key):
        """Extract data using standard indexing."""
        return self._data[key]
    
    def __setitem__(self, key, value):
        """Set data using standard indexing."""
        if self.mode == 'r':
            raise ValueError("Cannot modify read-only BigNeuroVec")
        self._data[key] = value
        self._data.flush()
    
    @property
    def values(self) -> np.ndarray:
        """Get all values (loads entire array into memory)."""
        return np.array(self._data)
    
    def series(self, x, y=None, z=None) -> np.ndarray:
        """Extract time series for voxel(s)."""
        if y is not None and z is not None:
            # Single voxel
            return self._data[x, y, z, :]
        elif isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 3:
                # Nx3 matrix of coordinates
                result = np.zeros((self.shape[3], x.shape[0]))
                for i, coord in enumerate(x):
                    # Check bounds
                    if (0 <= coord[0] < self.shape[0] and 
                        0 <= coord[1] < self.shape[1] and 
                        0 <= coord[2] < self.shape[2]):
                        result[:, i] = self._data[coord[0], coord[1], coord[2], :]
                    # else: leave as zeros
                return result
            elif x.ndim == 1:
                # Linear indices - convert to 3D coordinates
                coords = np.array(np.unravel_index(x, self.shape[:3], order='F')).T
                result = np.zeros((self.shape[3], len(x)))
                for i, coord in enumerate(coords):
                    if (0 <= coord[0] < self.shape[0] and
                        0 <= coord[1] < self.shape[1] and
                        0 <= coord[2] < self.shape[2]):
                        result[:, i] = self._data[coord[0], coord[1], coord[2], :]
                return result
        elif isinstance(x, int):
            # Single linear index - convert to 3D coordinates
            coords = np.unravel_index(x, self.shape[:3], order='F')
            return self._data[coords[0], coords[1], coords[2], :]
        else:
            raise ValueError("Invalid input for series extraction")
    
    def sub_vector(self, indices: Union[slice, np.ndarray]) -> 'BigNeuroVec':
        """Extract subset of volumes."""
        if isinstance(indices, slice):
            # Direct slicing is more efficient than converting to array
            subset_data = self._data[indices]
        else:
            indices = np.atleast_1d(indices)
            subset_data = self._data[indices]
        
        # Create new BigNeuroVec with subset data
        subset_shape = subset_data.shape
        subset_vec = BigNeuroVec(
            subset_data,
            NeuroSpace(dim=subset_shape, 
                      spacing=self.space.spacing,
                      origin=self.space.origin)
        )
        
        # Ensure data is flushed if it's memory-mapped
        if hasattr(subset_vec._data, 'flush'):
            subset_vec._data.flush()
        
        return subset_vec
    
    def vols(self, indices: Optional[np.ndarray] = None) -> list:
        """Extract volumes as a list of NeuroVol objects."""
        if indices is None:
            indices = range(self.shape[0])
        
        vol_list = []
        vol_space = NeuroSpace(dim=self.shape[1:], 
                              spacing=self.space.spacing[1:],
                              origin=self.space.origin[1:])
        
        for i in indices:
            vol = DenseNeuroVol(self._data[i], vol_space)
            vol_list.append(vol)
        
        return vol_list
    
    def as_matrix(self) -> np.ndarray:
        """Convert to 2D matrix (loads into memory)."""
        n_time = self.shape[0]
        n_voxels = np.prod(self.shape[1:])
        return self._data.reshape(n_time, n_voxels, order='F')
    
    def as_dense(self) -> 'DenseNeuroVec':
        """Convert to DenseNeuroVec (loads into memory).
        
        Note: This creates a copy of the data in memory. 
        For large datasets, consider using views or chunks instead.
        """
        from .neuro_vec import DenseNeuroVec
        # Use copy=False if data is already C-contiguous to avoid extra copy
        if self._data.flags['C_CONTIGUOUS']:
            return DenseNeuroVec(self._data, self.space)
        else:
            return DenseNeuroVec(np.ascontiguousarray(self._data), self.space)
    
    def as_sparse(self, mask: Optional[np.ndarray] = None) -> 'SparseNeuroVec':
        """Convert to SparseNeuroVec."""
        from .neuro_vec import SparseNeuroVec
        if mask is None:
            # Use non-zero voxels from first volume
            mask = self._data[0] != 0
        
        # Extract masked data
        mask_indices = np.where(mask.ravel(order='F'))[0]
        n_time = self.shape[0]
        sparse_data = np.zeros((n_time, len(mask_indices)))
        
        for t in range(n_time):
            vol_data = self._data[t].ravel(order='F')
            sparse_data[t] = vol_data[mask_indices]
        
        return SparseNeuroVec(
            data=sparse_data.ravel(order='F'),
            space=self.space,
            indices=np.tile(mask_indices, n_time),
            mask=np.repeat(np.arange(len(mask_indices)), n_time)
        )
    
    def _arithmetic_op(self, other, op):
        """Arithmetic operations."""
        if np.isscalar(other):
            result = BigNeuroVec(
                op(self._data, other),
                self.space
            )
        elif isinstance(other, BigNeuroVec):
            if self.shape != other.shape:
                raise ValueError("Incompatible shapes for arithmetic operation")
            result = BigNeuroVec(
                op(self._data, other._data),
                self.space
            )
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")
        
        return result
    
    def _comparison_op(self, other, op):
        """Comparison operations."""
        if np.isscalar(other):
            return op(self._data, other)
        elif isinstance(other, BigNeuroVec):
            if self.shape != other.shape:
                raise ValueError("Incompatible shapes for comparison")
            return op(self._data, other._data)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")
    
    def flush(self):
        """Flush memory-mapped data to disk."""
        self._data.flush()
    
    def process_chunks(self, func, chunk_size=100, axis=0):
        """Process data in chunks to avoid loading entire dataset into memory.
        
        Parameters
        ----------
        func : callable
            Function to apply to each chunk
        chunk_size : int
            Number of volumes per chunk
        axis : int
            Axis along which to chunk (default is time axis)
            
        Yields
        ------
        Results of func applied to each chunk
        """
        n_chunks = int(np.ceil(self.shape[axis] / chunk_size))
        
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, self.shape[axis])
            
            if axis == 0:
                chunk = self._data[start:end]
            else:
                # For other axes, use slicing
                slices = [slice(None)] * self.ndim
                slices[axis] = slice(start, end)
                chunk = self._data[tuple(slices)]
            
            yield func(chunk)
    
    def close(self):
        """Close the memory-mapped file."""
        if hasattr(self._data, '_mmap'):
            self._data._mmap.close()
    
    def __del__(self):
        """Clean up when object is destroyed."""
        try:
            self.close()
            if self._owns_file and os.path.exists(self.filename):
                os.unlink(self.filename)
        except:
            pass
    
    def __repr__(self):
        """String representation."""
        return (f"BigNeuroVec\n"
                f"  Type      : BigNeuroVec (memory-mapped)\n"
                f"  Dimension : {' X '.join(map(str, self.shape))}\n"
                f"  Spacing   : {' X '.join(map(str, self.space.spacing))}\n"
                f"  Origin    : {', '.join(map(str, self.space.origin))}\n"
                f"  Filename  : {self.filename}\n"
                f"  Mode      : {self.mode}")


def big_neurovecseq(vols: list, mask: Optional[np.ndarray] = None) -> BigNeuroVec:
    """Create BigNeuroVec from a sequence of volumes.
    
    Parameters
    ----------
    vols : list of NeuroVol
        List of 3D volumes to combine
    mask : array-like, optional
        Binary mask for the data
        
    Returns
    -------
    BigNeuroVec
        Memory-mapped 4D vector
        
    R Equivalent
    ------------
    neuroim2::BigNeuroVecSeq
    """
    if not vols:
        raise ValueError("vols list cannot be empty")
    
    # Get dimensions from first volume
    first_vol = vols[0]
    vol_shape = first_vol.shape
    n_vols = len(vols)
    
    # Create 4D space
    space_4d = NeuroSpace(
        dim=[n_vols] + list(vol_shape),
        spacing=[1] + list(first_vol.space.spacing),
        origin=[0] + list(first_vol.space.origin)
    )
    
    # Create BigNeuroVec
    vec = BigNeuroVec(
        np.zeros((n_vols,) + vol_shape, dtype=first_vol.data.dtype),
        space_4d
    )
    
    # Fill with volume data
    for i, vol in enumerate(vols):
        if vol.shape != vol_shape:
            raise ValueError(f"Volume {i} has inconsistent shape")
        vec._data[i] = vol.data
        
    vec.flush()
    
    return vec
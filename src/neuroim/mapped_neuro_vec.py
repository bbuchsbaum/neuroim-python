"""MappedNeuroVec - Mapped neuroimaging vector data.

This module provides a mapped implementation of NeuroVec that
applies transformations on-the-fly to underlying data.

Applies lazy transformations over vector-valued neuroimaging data.
"""

import numpy as np
from typing import Optional, Callable, Union, Tuple
from .neuro_vec import NeuroVec
from .neuro_space import NeuroSpace
from .neuro_vol import NeuroVol, DenseNeuroVol

class MappedNeuroVec(NeuroVec):
    """A mapped 4D neuroimaging vector.

    This class provides a view of neuroimaging data where values
    are transformed on-the-fly using a mapping function. The
    underlying data is not modified.

    Parameters
    ----------
    source : NeuroVec
        The source NeuroVec to map over
    map_fun : callable
        Function to apply to each value. Should accept and return
        arrays of the same shape.
    inverse_fun : callable, optional
        Inverse mapping function for setting values

    """

    def __init__(
        self,
        source: NeuroVec,
        map_fun: Callable[[np.ndarray], np.ndarray],
        inverse_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """Initialize MappedNeuroVec."""
        super().__init__(source.space)

        self.source = source
        self.map_fun = map_fun
        self.inverse_fun = inverse_fun

    def __getitem__(self, key):
        """Extract data with mapping applied."""
        # Get data from source
        source_data = self.source[key]

        # Apply mapping
        return self.map_fun(source_data)

    def __setitem__(self, key, value):
        """Set data using inverse mapping if available."""
        if self.inverse_fun is None:
            raise ValueError("Cannot set values without inverse mapping function")

        # Apply inverse mapping
        source_value = self.inverse_fun(value)

        # Set in source
        self.source[key] = source_value

    @property
    def data(self) -> np.ndarray:
        """Get all mapped data."""
        return self.map_fun(self.source.data)

    @property
    def values(self) -> np.ndarray:
        """Get all mapped values."""
        return self.data

    def series(self, x, y=None, z=None) -> np.ndarray:
        """Extract mapped time series for voxel(s)."""
        # Get series from source
        source_series = self.source.series(x, y, z)

        # Apply mapping based on series shape
        if source_series.ndim == 1:
            # Single voxel - apply directly
            return self.map_fun(source_series)
        else:
            # Multiple voxels - apply to each column
            result = np.zeros_like(source_series)
            for i in range(source_series.shape[1]):
                result[:, i] = self.map_fun(source_series[:, i])
            return result

    def sub_vector(self, indices: Union[slice, np.ndarray]) -> "MappedNeuroVec":
        """Extract subset of volumes as a new MappedNeuroVec."""
        source_subset = self.source.sub_vector(indices)
        return MappedNeuroVec(source_subset, self.map_fun, self.inverse_fun)

    def vols(self, indices: Optional[np.ndarray] = None) -> list:
        """Extract volumes as a list of NeuroVol objects."""
        source_vols = self.source.vols(indices)

        # Create mapped volumes
        mapped_vols = []
        for vol in source_vols:
            mapped_data = self.map_fun(vol.data)
            mapped_vol = DenseNeuroVol(mapped_data, vol.space)
            mapped_vols.append(mapped_vol)

        return mapped_vols

    def as_matrix(self) -> np.ndarray:
        """Convert to 2D matrix with mapping applied."""
        source_matrix = self.source.as_matrix()
        return self.map_fun(source_matrix)

    def as_dense(self) -> "DenseNeuroVec":
        """Convert to DenseNeuroVec with mapping applied."""
        from .neuro_vec import DenseNeuroVec

        return DenseNeuroVec(self.data, self.space)

    def as_sparse(self, mask: Optional[np.ndarray] = None) -> "SparseNeuroVec":
        """Convert to SparseNeuroVec with mapping applied."""
        # First convert source to sparse
        source_sparse = self.source.as_sparse(mask)

        # Apply mapping to sparse data
        from .neuro_vec import SparseNeuroVec

        mapped_data = self.map_fun(source_sparse.data)

        return SparseNeuroVec(
            data=mapped_data,
            space=source_sparse.space,
            indices=source_sparse.indices,
            mask=source_sparse.mask,
        )

    def _arithmetic_op(self, other, op):
        """Arithmetic operations on mapped data."""
        # Get mapped data
        self_data = self.data

        if np.isscalar(other):
            result_data = op(self_data, other)
        elif isinstance(other, MappedNeuroVec):
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

        # Return as DenseNeuroVec
        from .neuro_vec import DenseNeuroVec

        return DenseNeuroVec(result_data, self.space)

    def _comparison_op(self, other, op):
        """Comparison operations on mapped data."""
        self_data = self.data

        if np.isscalar(other):
            return op(self_data, other)
        elif isinstance(other, MappedNeuroVec):
            if self.shape != other.shape:
                raise ValueError("Incompatible shapes for comparison")
            return op(self_data, other.data)
        elif isinstance(other, NeuroVec):
            if self.shape != other.shape:
                raise ValueError("Incompatible shapes for comparison")
            return op(self_data, other.data)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def __repr__(self):
        """String representation."""
        return (
            f"MappedNeuroVec\n"
            f"  Type      : MappedNeuroVec\n"
            f"  Dimension : {' X '.join(map(str, self.shape))}\n"
            f"  Spacing   : {' X '.join(map(str, self.space.spacing))}\n"
            f"  Origin    : {', '.join(map(str, self.space.origin))}\n"
            f"  Map Func  : {self.map_fun.__name__ if hasattr(self.map_fun, '__name__') else 'custom'}\n"
            f"  Has Inverse: {self.inverse_fun is not None}"
        )

# Common mapping functions

def scale_mapper(scale: float, center: float = 0) -> Tuple[Callable, Callable]:
    """Create scale mapping functions.

    Parameters
    ----------
    scale : float
        Scale factor
    center : float
        Center point for scaling

    Returns
    -------
    tuple
        (forward_mapper, inverse_mapper)
    """

    def forward(x):
        return (x - center) * scale + center

    def inverse(x):
        return (x - center) / scale + center

    return forward, inverse

def log_mapper(base: float = np.e, offset: float = 0) -> Tuple[Callable, Callable]:
    """Create logarithmic mapping functions.

    Parameters
    ----------
    base : float
        Logarithm base
    offset : float
        Offset to add before log (to handle zeros)

    Returns
    -------
    tuple
        (forward_mapper, inverse_mapper)
    """

    def forward(x):
        return np.log(x + offset) / np.log(base)

    def inverse(x):
        return np.power(base, x) - offset

    return forward, inverse

def threshold_mapper(threshold: float, below_value: float = 0) -> Callable:
    """Create threshold mapping function.

    Parameters
    ----------
    threshold : float
        Threshold value
    below_value : float
        Value to use below threshold

    Returns
    -------
    callable
        Forward mapper (no inverse possible)
    """

    def forward(x):
        result = x.copy()
        result[x < threshold] = below_value
        return result

    return forward

def mapped_neurovecseq(
    vecs: list, map_fun: Callable, inverse_fun: Optional[Callable] = None
) -> MappedNeuroVec:
    """Create MappedNeuroVec from a sequence of vectors.

    Parameters
    ----------
    vecs : list of NeuroVec
        List of vectors to map
    map_fun : callable
        Mapping function
    inverse_fun : callable, optional
        Inverse mapping function

    Returns
    -------
    MappedNeuroVec
        Mapped vector sequence

    """
    if not vecs:
        raise ValueError("vecs list cannot be empty")

    # Concatenate vectors
    from .neuro_vec import neurovecseq

    source = neurovecseq(vecs)

    return MappedNeuroVec(source, map_fun, inverse_fun)

"""2D Neuroimaging Slice Classes.

2D slice containers for spatial neuroimaging data.
"""

import numpy as np
from typing import Tuple, Union, Optional
from .neuro_space import NeuroSpace
from .axis import AxisSet2D, NamedAxis, drop_axis


class NeuroSlice:
    """A 2D slice of neuroimaging data.

    Stores a 2D data slice with spatial metadata.

    Parameters
    ----------
    data : np.ndarray
        2D array containing the slice data
    space : NeuroSpace
        2D spatial metadata

    R Equivalent
    ------------
    neuroim2::NeuroSlice
    """

    def __init__(self, data: np.ndarray, space: NeuroSpace):
        if space.ndim != 2:
            raise ValueError("Space must be 2-dimensional for NeuroSlice")

        # Handle different data shapes like R does
        if data.ndim == 1:
            # Reshape 1D array to 2D
            if data.size != np.prod(space.dim):
                raise ValueError(
                    f"Data length {data.size} must match space dimensions {np.prod(space.dim)}"
                )
            data = data.reshape(tuple(space.dim), order="F")
        elif data.ndim == 2:
            if data.shape != tuple(space.dim):
                raise ValueError(
                    f"Data shape {data.shape} must match space dimensions {tuple(space.dim)}"
                )
        else:
            raise ValueError("Data must be 1D or 2D array")

        self.data = data
        self.space = space

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the 2D data."""
        return tuple(int(d) for d in self.data.shape)

    @property
    def dim(self) -> np.ndarray:
        """Dimensions of the slice.

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
    def axes(self):
        """Axes of the slice.

        R Equivalent
        ------------
        neuroim2::axes
        """
        return self.space.axes

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        """String representation matching R's show method."""
        val_range = np.nanmin(self.data), np.nanmax(self.data)
        return (
            f"NeuroSlice\n"
            f"  Type       : NeuroSlice\n"
            f"  Dimensions : {' x '.join(map(str, self.dim))}\n"
            f"  Spacing    : {' x '.join(f'{s:.2f}' for s in self.spacing)}\n"
            f"  Origin     : {' x '.join(f'{o:.2f}' for o in self.origin)}\n"
            f"  Value Range: [{val_range[0]:.2f}, {val_range[1]:.2f}]"
        )

    def _arithmetic_op(self, other, op):
        """Perform arithmetic operation."""
        if isinstance(other, (int, float)):
            # Scalar operation
            return NeuroSlice(op(self.data, other), self.space)
        elif isinstance(other, np.ndarray):
            # Array operation
            if other.shape == self.shape:
                return NeuroSlice(op(self.data, other), self.space)
            else:
                raise ValueError(
                    f"Array shape {other.shape} must match slice shape {self.shape}"
                )
        elif isinstance(other, NeuroSlice):
            # Slice-slice operation
            if not np.array_equal(self.space.dim, other.space.dim):
                raise ValueError("NeuroSlice objects must have same dimensions")
            return NeuroSlice(op(self.data, other.data), self.space)
        else:
            return NotImplemented

    def __add__(self, other):
        return self._arithmetic_op(other, np.add)

    def __sub__(self, other):
        return self._arithmetic_op(other, np.subtract)

    def __mul__(self, other):
        return self._arithmetic_op(other, np.multiply)

    def __truediv__(self, other):
        return self._arithmetic_op(other, np.divide)

    def __radd__(self, other):
        return self._reverse_arithmetic_op(other, np.add)

    def __rsub__(self, other):
        return self._reverse_arithmetic_op(other, np.subtract)

    def __rmul__(self, other):
        return self._reverse_arithmetic_op(other, np.multiply)

    def __rtruediv__(self, other):
        return self._reverse_arithmetic_op(other, np.divide)

    def _reverse_arithmetic_op(self, other, op):
        return self._arithmetic_op(other, lambda x, y: op(y, x))

    def grid_to_index(self, coords: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to linear indices.

        Parameters
        ----------
        coords : np.ndarray
            Grid coordinates, shape (n, 2) or (2,)

        Returns
        -------
        np.ndarray
            Linear indices (0-based)

        R Equivalent
        ------------
        neuroim2::grid_to_index
        """
        return self.space.grid_to_index(coords)

    def index_to_grid(self, idx: Union[int, np.ndarray]) -> np.ndarray:
        """Convert linear indices to grid coordinates.

        Parameters
        ----------
        idx : int or np.ndarray
            Linear indices (0-based)

        Returns
        -------
        np.ndarray
            Grid coordinates, shape (n, 2)

        R Equivalent
        ------------
        neuroim2::index_to_grid
        """
        return self.space.index_to_grid(idx)

    def values(self) -> np.ndarray:
        """Get the data values as a 1D array.

        R Equivalent
        ------------
        neuroim2::values
        """
        return self.data.ravel(order="F")


def neuroslice(
    data: Union[np.ndarray, list],
    space: NeuroSpace,
    indices: Optional[np.ndarray] = None,
) -> NeuroSlice:
    """Factory function to create a NeuroSlice.

    Parameters
    ----------
    data : array-like
        The slice data values
    space : NeuroSpace
        2D spatial metadata
    indices : np.ndarray, optional
        Linear indices for sparse initialization

    Returns
    -------
    NeuroSlice
        New NeuroSlice instance

    R Equivalent
    ------------
    neuroim2::NeuroSlice
    """
    data = np.asarray(data)

    if indices is None:
        # Dense mode
        return NeuroSlice(data, space)
    else:
        # Sparse mode - create zero matrix and fill at indices
        if space.ndim != 2:
            raise ValueError("Space must be 2-dimensional for NeuroSlice")

        slice_data = np.zeros(tuple(space.dim), dtype=np.float64)
        # Use ravel with Fortran order to get the right view
        slice_flat = slice_data.ravel(order="F")
        slice_flat[indices] = data
        # Reshape back maintaining the data
        slice_data = slice_flat.reshape(tuple(space.dim), order="F")
        return NeuroSlice(slice_data, space)


def slice(vol, zlevel: int, along: int) -> NeuroSlice:
    """Extract a 2D slice from a 3D volume.

    Parameters
    ----------
    vol : NeuroVol
        Volume to slice from
    zlevel : int
        Index of the slice (1-based in R, 0-based in Python)
    along : int
        Axis to slice along (1, 2, or 3 in R; 0, 1, or 2 in Python)

    Returns
    -------
    NeuroSlice
        2D slice at the specified position

    R Equivalent
    ------------
    neuroim2::slice
    """
    from .neuro_vol import NeuroVol

    if not isinstance(vol, NeuroVol):
        raise TypeError("vol must be a NeuroVol")

    # Convert from Python 0-based to validate
    if along < 0 or along > 2:
        raise ValueError("along must be 0, 1, or 2")
    if zlevel < 0 or zlevel >= vol.shape[along]:
        raise ValueError(f"zlevel {zlevel} out of bounds for axis {along}")

    # Extract slice data
    if along == 0:
        slice_data = vol.data[zlevel, :, :]
    elif along == 1:
        slice_data = vol.data[:, zlevel, :]
    else:  # along == 2
        slice_data = vol.data[:, :, zlevel]

    # Create slice space by dropping the sliced dimension
    slice_space = vol.space.drop_dim(along)

    return NeuroSlice(slice_data, slice_space)


def slices(vol):
    """Extract all 2D slices from a 3D volume.

    Parameters
    ----------
    vol : NeuroVol
        Volume to extract slices from

    Returns
    -------
    list
        List of NeuroSlice objects

    R Equivalent
    ------------
    neuroim2::slices
    """
    from .neuro_vol import NeuroVol

    if not isinstance(vol, NeuroVol):
        raise TypeError("vol must be a NeuroVol")

    nslices = vol.shape[2]
    return [slice(vol, i, 2) for i in range(nslices)]

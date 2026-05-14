"""Spatially aware containers for 3D neuroimaging volumes."""

from abc import ABC, abstractmethod
from typing import Any
from typing import Union, Tuple, Optional, List, Dict
import numpy as np
from scipy import sparse

from .neuro_space import NeuroSpace


class NeuroVol(ABC):
    """Abstract base class for volumetric neuroimaging data.

    Parameters
    ----------
    space : NeuroSpace
        The spatial metadata for the volume
    """

    def __init__(self, space: NeuroSpace):
        if not isinstance(space, NeuroSpace):
            raise TypeError("space must be a NeuroSpace object")
        self.space = space

    @classmethod
    def from_array(cls, data, space: NeuroSpace, coords=None) -> "DenseNeuroVol":
        """Create a dense volume from array data and a spatial contract."""
        if coords is None:
            return DenseNeuroVol(data, space)

        coords = np.asarray(coords, dtype=int)
        values = np.asarray(data)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an Nx3 coordinate matrix")
        if values.size != coords.shape[0]:
            raise ValueError(
                f"data has {values.size} values but coords has {coords.shape[0]} rows"
            )

        full = np.zeros(tuple(int(d) for d in space.dim[:3]), dtype=values.dtype)
        full[coords[:, 0], coords[:, 1], coords[:, 2]] = values.reshape(-1)
        return DenseNeuroVol(full, space)

    @classmethod
    def from_nibabel(cls, img: Any) -> "DenseNeuroVol":
        """Create a dense 3D volume from a nibabel SpatialImage-like object."""
        if not hasattr(img, "shape") or not hasattr(img, "affine"):
            raise TypeError("from_nibabel expects an image with shape and affine")
        data_obj = getattr(img, "dataobj", None)
        if data_obj is None and not hasattr(img, "get_fdata"):
            raise TypeError("from_nibabel expects an image with dataobj or get_fdata()")
        data = np.asanyarray(data_obj) if data_obj is not None else img.get_fdata()
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"NeuroVol.from_nibabel expects 3D data, got {data.ndim}D")

        vol = DenseNeuroVol(
            data,
            NeuroSpace.from_affine(
                img.affine,
                data.shape,
                header=getattr(img, "header", None),
            ),
        )
        _attach_nibabel_metadata(vol, img)
        return vol

    def to_nibabel(self, cls=None):
        """Convert this volume to a nibabel image.

        When ``self.provenance`` is a :class:`~neuroim.results.Receipt`, it
        is embedded as a NIfTI 'comment' header extension (ecode 6) carrying
        the marker prefix and JSON payload — so a clean-process round-trip
        via :func:`~neuroim.io.read_image` recovers the Receipt (PAIN-6 /
        Scenario 05).
        """
        import nibabel as nib

        img_cls = cls or nib.Nifti1Image
        data = self.as_dense().data
        header = getattr(self, "_nibabel_header", None)
        if header is not None:
            header = header.copy()
        img = img_cls(data, self.space.affine, header=header)
        _restore_nibabel_xforms(img, self)
        _embed_receipt_extension(img, getattr(self, "provenance", None))
        return img

    # Abstract methods that subclasses must implement
    @abstractmethod
    def __getitem__(self, key):
        """Extract values using various indexing methods."""
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        """Set values using various indexing methods."""
        pass

    @abstractmethod
    def values(self) -> np.ndarray:
        """Get underlying data values.        """
        pass

    @abstractmethod
    def as_dense(self) -> "DenseNeuroVol":
        """Convert to dense representation.        """
        pass

    @abstractmethod
    def as_sparse(self, mask=None) -> "SparseNeuroVol":
        """Convert to sparse representation.        """
        pass

    @abstractmethod
    def as_logical(self) -> "LogicalNeuroVol":
        """Convert to logical/binary representation.        """
        pass

    # ME-5: Pythonic aliases for the R-shaped ``as_*`` verbs.  These are the
    # canonical names; ``as_dense`` / ``as_sparse`` / ``as_logical`` remain as
    # compatibility shims.
    def to_dense(self) -> "DenseNeuroVol":
        """Pythonic alias for :meth:`as_dense`."""
        return self.as_dense()

    def to_sparse(self, mask=None) -> "SparseNeuroVol":
        """Pythonic alias for :meth:`as_sparse`."""
        return self.as_sparse(mask)

    def to_logical(self) -> "LogicalNeuroVol":
        """Pythonic alias for :meth:`as_logical`."""
        return self.as_logical()

    # Properties from NeuroSpace
    @property
    def dim(self) -> np.ndarray:
        """Dimensions of the volume.        """
        return self.space.dim

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the volume (Python style)."""
        return tuple(self.space.dim)

    @property
    def ndim(self) -> int:
        """Number of dimensions.        """
        return self.space.ndim

    @property
    def spacing(self) -> np.ndarray:
        """Voxel dimensions.        """
        return self.space.spacing

    @property
    def origin(self) -> np.ndarray:
        """Origin coordinates.        """
        return self.space.origin

    @property
    def axes(self):
        """Axis information.        """
        return self.space.axes

    @property
    def trans(self) -> np.ndarray:
        """Transformation matrix.        """
        return self.space.trans

    def bounds(self) -> np.ndarray:
        """Spatial bounds of the volume.        """
        return self.space.bounds()

    # Coordinate transformation methods (delegate to space)
    def coord_to_grid(self, coords: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid indices.        """
        return self.space.coord_to_grid(coords)

    def grid_to_coord(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid indices to world coordinates.        """
        return self.space.grid_to_coord(grid)

    def coord_to_index(self, coords: np.ndarray) -> np.ndarray:
        """Convert world coordinates to linear indices.        """
        return self.space.coord_to_index(coords)

    def index_to_coord(self, indices: Union[int, np.ndarray]) -> np.ndarray:
        """Convert linear indices to world coordinates.        """
        return self.space.index_to_coord(indices)

    def grid_to_index(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid indices to linear indices.        """
        return self.space.grid_to_index(grid)

    def index_to_grid(self, indices: Union[int, np.ndarray]) -> np.ndarray:
        """Convert linear indices to grid indices.        """
        return self.space.index_to_grid(indices)

    def get_orthogonal_slices(
        self, world_point: np.ndarray, slice_types: Optional[List[str]] = None
    ) -> Dict[str, "NeuroSlice"]:
        """Extract orthogonal slices at a given world-space point.

        Parameters
        ----------
        world_point : np.ndarray
            World-space coordinates (x, y, z) at which to extract slices
        slice_types : List[str], optional
            List of slice types to extract ('axial', 'sagittal', 'coronal').
            If None, all three slice types are extracted.

        Returns
        -------
        Dict[str, NeuroSlice]
            Dictionary mapping slice type names to NeuroSlice objects

        Examples
        --------
        >>> # Extract all orthogonal slices at center of volume
        >>> center = vol.space.centroid()
        >>> slices = vol.get_orthogonal_slices(center)
        >>>
        >>> # Extract only axial slice
        >>> slices = vol.get_orthogonal_slices(center, ['axial'])
        """
        from .orthogonal_slices import extract_orthogonal_slices

        return extract_orthogonal_slices(self, world_point, slice_types)

    # Utility methods
    def coords(self, real: bool = False) -> np.ndarray:
        """Get voxel coordinates.

        Parameters
        ----------
        real : bool
            If True, return world coordinates; if False, return grid coordinates        """
        # Create grid of all voxel coordinates
        grids = np.mgrid[0 : self.dim[0], 0 : self.dim[1], 0 : self.dim[2]]
        grid_coords = np.column_stack([g.ravel() for g in grids])

        if real:
            return self.grid_to_coord(grid_coords)
        else:
            return grid_coords

    def indices(self) -> np.ndarray:
        """Get linear indices for all voxels.        """
        return np.arange(np.prod(self.dim))

    # Arithmetic operations
    def __add__(self, other):
        """Add two volumes or volume and scalar."""
        return self._arithmetic_op(other, np.add)

    def __sub__(self, other):
        """Subtract two volumes or volume and scalar."""
        return self._arithmetic_op(other, np.subtract)

    def __mul__(self, other):
        """Multiply two volumes or volume and scalar."""
        return self._arithmetic_op(other, np.multiply)

    def __truediv__(self, other):
        """Divide two volumes or volume and scalar."""
        return self._arithmetic_op(other, np.divide)

    def __radd__(self, other):
        """Handle scalar/array + volume via reversed dispatch."""
        return self._reverse_arithmetic_op(other, np.add)

    def __rsub__(self, other):
        """Handle scalar/array - volume via reversed dispatch."""
        return self._reverse_arithmetic_op(other, np.subtract)

    def __rmul__(self, other):
        """Handle scalar/array * volume via reversed dispatch."""
        return self._reverse_arithmetic_op(other, np.multiply)

    def __rtruediv__(self, other):
        """Handle scalar/array / volume via reversed dispatch."""
        return self._reverse_arithmetic_op(other, np.divide)

    def _reverse_arithmetic_op(self, other, op):
        """Perform reversed arithmetic when right-hand side is this object."""
        return self._arithmetic_op(other, lambda x, y: op(y, x))

    @abstractmethod
    def _arithmetic_op(self, other, op):
        """Perform arithmetic operation."""
        pass

    # Comparison operations
    def __gt__(self, value):
        """Greater than comparison."""
        return self._comparison_op(value, np.greater)

    def __lt__(self, value):
        """Less than comparison."""
        return self._comparison_op(value, np.less)

    def __ge__(self, value):
        """Greater than or equal comparison."""
        return self._comparison_op(value, np.greater_equal)

    def __le__(self, value):
        """Less than or equal comparison."""
        return self._comparison_op(value, np.less_equal)

    def __eq__(self, value):
        """Equal comparison."""
        return self._comparison_op(value, np.equal)

    def __ne__(self, value):
        """Not equal comparison."""
        return self._comparison_op(value, np.not_equal)

    @abstractmethod
    def _comparison_op(self, value, op):
        """Perform comparison operation."""
        pass

    # Summary statistics
    def sum(self, na_rm: bool = False) -> float:
        """Sum of all values.        """
        values = self.values()
        if na_rm:
            return np.nansum(values)
        return np.sum(values)

    def mean(self, na_rm: bool = False) -> float:
        """Mean of all values.        """
        values = self.values()
        if na_rm:
            return np.nanmean(values)
        return np.mean(values)

    # Compatibility aliases retained as deprecated shims (see neuroim.compat for migration policy).
    def vol_mean(self, na_rm: bool = False) -> float:
        """Deprecated alias for :meth:`mean`; retained for migration only."""
        return self.mean(na_rm=na_rm)

    def min(self, na_rm: bool = False) -> float:
        """Minimum value.        """
        values = self.values()
        if na_rm:
            return np.nanmin(values)
        return np.min(values)

    def which_min(self):
        """Return linear index of minimum value (F-order indexing)."""
        return int(np.argmin(self.values()))

    def max(self, na_rm: bool = False) -> float:
        """Maximum value.        """
        values = self.values()
        if na_rm:
            return np.nanmax(values)
        return np.max(values)

    def which_max(self):
        """Return linear index of maximum value (F-order indexing)."""
        return int(np.argmax(self.values()))

    def vol_sd(self, na_rm: bool = False) -> float:
        """Alias for standard deviation for neuroim2 compatibility."""
        values = self.values()
        if na_rm:
            return np.nanstd(values)
        return float(np.std(values))

    def range(self, na_rm: bool = False) -> Tuple[float, float]:
        """Range of values.        """
        return (self.min(na_rm), self.max(na_rm))

    def __repr__(self):
        """String representation matching R's show method."""
        return (
            f"{self.__class__.__name__}\n"
            f"  Type      : {self.__class__.__name__}\n"
            f"  Dimension : {' X '.join(map(str, self.dim))}\n"
            f"  Spacing   : {' X '.join(map(str, self.spacing))}\n"
            f"  Origin    : {', '.join(map(str, self.origin))}\n"
            f"  Range     : [{self.min():.3f}, {self.max():.3f}]"
        )


class DenseNeuroVol(NeuroVol):
    """Dense 3D neuroimaging volume.

    Parameters
    ----------
    data : array-like
        3D array of voxel values
    space : NeuroSpace
        Spatial metadata
    label : str, optional
        Volume label
    indices : array-like, optional
        If provided, only these indices will be filled with data    """

    def __init__(
        self,
        data,
        space: NeuroSpace,
        label: str = "",
        indices=None,
        *,
        provenance=None,
    ):
        super().__init__(space)

        # Handle different input types
        if indices is not None:
            # Create volume and fill only specified indices
            flat_data = np.zeros(int(np.prod(self.shape)), dtype=np.asarray(data).dtype)
            flat_data[np.asarray(indices, dtype=int)] = data
            self.data = flat_data.reshape(self.shape, order="F")
        else:
            # Direct initialization
            data = np.asarray(data)

            # Coerce complex to real
            if np.issubdtype(data.dtype, np.complexfloating):
                data = data.real.astype(np.float64)

            # Handle 1D data
            if data.ndim == 1:
                if data.size == np.prod(self.shape):
                    self.data = data.reshape(
                        self.shape, order="F"
                    )  # Fortran order to match R
                else:
                    raise ValueError(
                        f"Data size {data.size} doesn't match space size {np.prod(self.shape)}"
                    )
            # Handle 2D data (matrix with single row or column)
            elif data.ndim == 2:
                if data.shape[0] == 1 or data.shape[1] == 1:
                    # Flatten and treat as 1D
                    flat_data = data.ravel()
                    if flat_data.size == np.prod(self.shape):
                        self.data = flat_data.reshape(self.shape, order="F")
                    else:
                        raise ValueError(
                            f"Data size {flat_data.size} doesn't match space size {np.prod(self.shape)}"
                        )
                else:
                    raise ValueError(
                        f"2D data must have single row or column, got shape {data.shape}"
                    )
            # Handle 3D data
            elif data.ndim == 3:
                if data.shape != self.shape:
                    raise ValueError(
                        f"Data shape {data.shape} doesn't match space shape {self.shape}"
                    )
                self.data = data
            else:
                raise ValueError(
                    f"Data must be 1D, 2D (single row/col), or 3D array, got {data.ndim}D"
                )

        self.label = label
        # Optional provenance Receipt for derived maps (temporal reductions,
        # future map-producing reducers).  ROI/searchlight typed results
        # already carry receipts via dedicated result objects; for volumes
        # produced inside neuroim the receipt rides directly on the volume.
        self.provenance = provenance

    def __getitem__(self, key):
        """Extract values using various indexing methods."""
        if isinstance(key, tuple) and len(key) == 3:
            # Standard 3D indexing
            return self.data[key]
        elif isinstance(key, (list, np.ndarray)):
            # Convert list to array if needed
            key = np.asarray(key)
            if key.dtype == bool:
                # Boolean indexing
                return self.data[key]
            elif key.ndim == 2 and key.shape[1] == 3:
                # Nx3 matrix of coordinates
                return self.data[key[:, 0], key[:, 1], key[:, 2]]
            elif key.ndim == 1:
                # Linear indices (Fortran order)
                return self.data.ravel(order="F")[key]
        elif isinstance(key, (int, slice)):
            # Single index or slice (Fortran order)
            return self.data.ravel(order="F")[key]
        elif isinstance(key, LogicalNeuroVol):
            # Logical indexing
            if key.space != self.space:
                raise ValueError("Mask must have same space as volume")
            return self.data[key.data]
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        """Set values using various indexing methods."""
        if isinstance(key, tuple) and len(key) == 3:
            self.data[key] = value
        elif isinstance(key, np.ndarray):
            if key.ndim == 2 and key.shape[1] == 3:
                self.data[key[:, 0], key[:, 1], key[:, 2]] = value
            elif key.ndim == 1:
                self.data.ravel(order="F")[key] = value
        elif isinstance(key, (int, slice)):
            self.data.ravel(order="F")[key] = value
        elif isinstance(key, LogicalNeuroVol):
            if key.space != self.space:
                raise ValueError("Mask must have same space as volume")
            self.data[key.data] = value
        else:
            self.data[key] = value

    def values(self) -> np.ndarray:
        """Get underlying data values."""
        return self.data.ravel(order="F")  # Fortran order to match R

    def as_dense(self) -> "DenseNeuroVol":
        """Already dense, return self."""
        return self

    def as_sparse(self, mask=None) -> "SparseNeuroVol":
        """Convert to sparse representation."""
        if mask is None:
            # Use non-zero values as mask
            indices = np.nonzero(self.data.ravel(order="F"))[0]
            values = self.data.ravel(order="F")[indices]
        elif isinstance(mask, LogicalNeuroVol):
            indices = np.where(mask.data.ravel(order="F"))[0]
            values = self.data.ravel(order="F")[indices]
        elif isinstance(mask, np.ndarray):
            if mask.dtype == bool:
                indices = np.where(mask.ravel(order="F"))[0]
            else:
                indices = mask
            values = self.data.ravel(order="F")[indices]
        else:
            raise TypeError("mask must be LogicalNeuroVol, boolean array, or indices")

        return SparseNeuroVol(values, self.space, indices=indices, label=self.label)

    def as_logical(self) -> "LogicalNeuroVol":
        """Convert to logical representation."""
        return LogicalNeuroVol(self.data != 0, self.space, label=self.label)

    def as_mask(self, indices=None) -> "LogicalNeuroVol":
        """Convert to mask.        """
        if indices is None:
            return self.as_logical()
        else:
            flat_mask = np.zeros(int(np.prod(self.shape)), dtype=bool)
            flat_mask[np.asarray(indices, dtype=int)] = True
            mask_data = flat_mask.reshape(self.shape, order="F")
            return LogicalNeuroVol(mask_data, self.space)

    def as_array(self) -> np.ndarray:
        """Convert to numpy array.        """
        return self.data

    def _arithmetic_op(self, other, op):
        """Perform arithmetic operation."""
        if isinstance(other, (int, float, np.ndarray)):
            return DenseNeuroVol(op(self.data, other), self.space, self.label)
        elif isinstance(other, DenseNeuroVol):
            if self.space != other.space:
                raise ValueError("Volumes must have same space")
            return DenseNeuroVol(op(self.data, other.data), self.space, self.label)
        elif isinstance(other, SparseNeuroVol):
            # Convert sparse to dense for operation
            other_dense = other.as_dense()
            return DenseNeuroVol(
                op(self.data, other_dense.data), self.space, self.label
            )
        else:
            return NotImplemented

    def _comparison_op(self, value, op):
        """Perform comparison operation."""
        if isinstance(value, (int, float, np.ndarray)):
            return LogicalNeuroVol(op(self.data, value), self.space)
        elif isinstance(value, DenseNeuroVol):
            if self.space != value.space:
                raise ValueError("Volumes must have same space")
            return LogicalNeuroVol(op(self.data, value.data), self.space)
        else:
            return NotImplemented

    def concat(self, *others: "DenseNeuroVol") -> "DenseNeuroVec":
        """Concatenate multiple volumes into a DenseNeuroVec along a new time dimension.

        Parameters
        ----------
        *others : DenseNeuroVol
            Additional volumes to concatenate.

        Returns
        -------
        DenseNeuroVec
            4D vector containing all volumes stacked along the time axis.
        """
        from .neuro_vec import DenseNeuroVec

        all_vols = [self] + list(others)
        for v in all_vols[1:]:
            if v.shape != self.shape:
                raise ValueError("All volumes must have same spatial dimensions")
        stacked = np.stack([v.data for v in all_vols], axis=-1)
        vec_space = NeuroSpace(
            (*self.shape, len(all_vols)),
            spacing=np.append(self.space.spacing, 1.0),
            origin=np.append(self.space.origin, 0.0),
        )
        return DenseNeuroVec(stacked, vec_space)


class SparseNeuroVol(NeuroVol):
    """Sparse 3D neuroimaging volume.
    Uses scipy.sparse for efficient storage.

    Parameters
    ----------
    data : array-like
        Non-zero values
    space : NeuroSpace
        Spatial metadata
    indices : array-like
        Linear indices of non-zero values
    label : str, optional
        Volume label    """

    def __init__(
        self, data, space: NeuroSpace, indices=None, label: str = "", mask=None
    ):
        super().__init__(space)

        # Handle mask parameter
        if mask is not None and indices is None:
            # Extract boolean array from LogicalNeuroVol if needed.
            # ndarray.data is a memoryview, not the underlying ndarray, so
            # prefer raw ndarray inputs and only fall back to .data when the
            # input is a NeuroVol-like container.
            if isinstance(mask, np.ndarray):
                mask_array = mask
            elif hasattr(mask, "data") and isinstance(mask.data, np.ndarray):
                mask_array = mask.data
            else:
                mask_array = np.asarray(mask)

            indices = np.where(mask_array.ravel(order="F"))[0]
        elif mask is None and indices is None:
            raise ValueError("Either 'indices' or 'mask' must be provided")
        elif mask is not None and indices is not None:
            raise ValueError("Cannot provide both 'mask' and 'indices'")

        data = np.asarray(data).ravel(order="F")
        indices = np.asarray(indices).ravel()

        if len(data) != len(indices):
            raise ValueError("data and indices must have same length")

        # Store as sparse vector (similar to R's sparseVector)
        size = np.prod(self.shape)
        self.sparse_data = sparse.csr_matrix(
            (data, (np.zeros(len(indices)), indices)), shape=(1, size)
        )
        self.label = label

    @property
    def indices(self) -> np.ndarray:
        """Get indices of non-zero values."""
        return self.sparse_data.indices

    @property
    def data(self) -> np.ndarray:
        """Get non-zero values."""
        return self.sparse_data.data

    def __getitem__(self, key):
        """Extract values using various indexing methods."""
        if isinstance(key, tuple) and len(key) == 3:
            # Convert 3D to linear index
            idx = self.space.grid_to_index(np.array([key]))[0]
            return self.sparse_data[0, idx]
        elif isinstance(key, np.ndarray):
            if key.ndim == 2 and key.shape[1] == 3:
                # Nx3 matrix of coordinates
                indices = self.space.grid_to_index(key)
                return self.sparse_data[0, indices].A.ravel()
            elif key.ndim == 1:
                # Linear indices
                return self.sparse_data[0, key].A.ravel()
        elif isinstance(key, (int, slice)):
            return self.sparse_data[0, key]
        else:
            # Convert to dense for complex indexing
            return self.as_dense()[key]

    def __setitem__(self, key, value):
        """Set values using various indexing methods."""
        # Note: Setting values in sparse array can be inefficient
        if isinstance(key, tuple) and len(key) == 3:
            idx = self.space.grid_to_index(np.array([key]))[0]
            self.sparse_data[0, idx] = value
        elif isinstance(key, int):
            self.sparse_data[0, key] = value
        elif isinstance(key, np.ndarray):
            if key.ndim == 2 and key.shape[1] == 3:
                # Nx3 coordinate array
                indices = self.space.grid_to_index(key)
                for idx, val in zip(indices, np.broadcast_to(value, len(indices))):
                    self.sparse_data[0, idx] = val
            elif key.ndim == 1:
                # Linear indices
                for idx, val in zip(key, np.broadcast_to(value, len(key))):
                    self.sparse_data[0, idx] = val
            elif key.dtype == bool:
                # Boolean mask
                linear_indices = np.where(key.ravel(order="F"))[0]
                vals = np.broadcast_to(value, len(linear_indices))
                for idx, val in zip(linear_indices, vals):
                    self.sparse_data[0, idx] = val
            else:
                raise TypeError(
                    f"Unsupported array index type for SparseNeuroVol: {key.dtype}"
                )
        elif isinstance(key, slice):
            indices = range(*key.indices(np.prod(self.shape)))
            vals = np.broadcast_to(value, len(indices))
            for idx, val in zip(indices, vals):
                self.sparse_data[0, idx] = val
        else:
            raise TypeError(f"Unsupported index type for SparseNeuroVol: {type(key)}")

    def values(self) -> np.ndarray:
        """Get all values (including zeros)."""
        return self.sparse_data.toarray().ravel()

    def as_dense(self) -> DenseNeuroVol:
        """Convert to dense representation."""
        dense_data = self.sparse_data.toarray().reshape(self.shape, order="F")
        return DenseNeuroVol(dense_data, self.space, self.label)

    def as_sparse(self, mask=None) -> "SparseNeuroVol":
        """Already sparse, optionally apply new mask."""
        if mask is None:
            return self
        else:
            # Apply additional mask
            dense = self.as_dense()
            return dense.as_sparse(mask)

    def as_logical(self) -> "LogicalNeuroVol":
        """Convert to logical representation."""
        return self.as_dense().as_logical()

    def as_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.as_dense().data

    def as_numeric(self) -> np.ndarray:
        """Get non-zero values only.        """
        return self.data

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return self.sparse_data.nnz

    def _arithmetic_op(self, other, op):
        """Perform arithmetic operation."""
        if isinstance(other, (int, float)):
            # Scalar operation preserves sparsity
            new_data = op(self.data, other)
            return SparseNeuroVol(new_data, self.space, self.indices, self.label)
        elif isinstance(other, SparseNeuroVol):
            # For sparse-sparse operations, we need to handle index union
            # Convert both operands to dense to align sparse index sets.
            dense1 = self.as_dense()
            dense2 = other.as_dense()
            result = dense1._arithmetic_op(dense2, op)
            return result
        else:
            # Convert to dense for operations with dense volumes
            dense = self.as_dense()
            result = dense._arithmetic_op(other, op)
            return result

    def _comparison_op(self, value, op):
        """Perform comparison operation."""
        if isinstance(value, (int, float)):
            # Create logical volume
            dense = self.as_dense()
            return dense._comparison_op(value, op)
        else:
            dense = self.as_dense()
            return dense._comparison_op(value, op)


class LogicalNeuroVol(DenseNeuroVol):
    """Logical/binary 3D neuroimaging volume.
    Extends DenseNeuroVol with boolean data.

    Parameters
    ----------
    data : array-like
        Boolean 3D array
    space : NeuroSpace
        Spatial metadata
    label : str, optional
        Volume label
    indices : array-like, optional
        If provided, only these indices will be set to True    """

    def __init__(self, data, space: NeuroSpace, label: str = "", indices=None):
        if indices is not None:
            # Create false volume and set specified indices to true
            flat_data = np.zeros(int(np.prod(space.dim)), dtype=bool)
            flat_data[np.asarray(indices, dtype=int)] = True
            bool_data = flat_data.reshape(space.dim, order="F")
            super().__init__(bool_data, space, label)
        else:
            # Ensure boolean type
            data = np.asarray(data, dtype=bool)
            super().__init__(data, space, label)

    def as_logical(self) -> "LogicalNeuroVol":
        """Already logical, return self."""
        return self

    def __and__(self, other):
        """Logical AND operation."""
        if isinstance(other, LogicalNeuroVol):
            if self.space != other.space:
                raise ValueError("Volumes must have same space")
            return LogicalNeuroVol(self.data & other.data, self.space)
        return NotImplemented

    def __or__(self, other):
        """Logical OR operation."""
        if isinstance(other, LogicalNeuroVol):
            if self.space != other.space:
                raise ValueError("Volumes must have same space")
            return LogicalNeuroVol(self.data | other.data, self.space)
        return NotImplemented

    def __xor__(self, other):
        """Logical XOR operation."""
        if isinstance(other, LogicalNeuroVol):
            if self.space != other.space:
                raise ValueError("Volumes must have same space")
            return LogicalNeuroVol(self.data ^ other.data, self.space)
        return NotImplemented

    def __invert__(self):
        """Logical NOT operation."""
        return LogicalNeuroVol(~self.data, self.space)

    @property
    def sum(self) -> int:
        """Number of True voxels."""
        return int(np.sum(self.data))

    def _arithmetic_op(self, other, op):
        """Arithmetic operations on logical volumes."""
        # Convert to numeric for arithmetic
        numeric_data = self.data.astype(float)
        if isinstance(other, (int, float)):
            result = op(numeric_data, other)
        elif isinstance(other, LogicalNeuroVol):
            if self.space != other.space:
                raise ValueError("Volumes must have same space")
            result = op(numeric_data, other.data.astype(float))
        else:
            return super()._arithmetic_op(other, op)

        # Return as DenseNeuroVol since result may not be boolean
        return DenseNeuroVol(result, self.space, self.label)


def neurovol(data, space: NeuroSpace, label: str = "", indices=None) -> NeuroVol:
    """Create a NeuroVol object.

    Factory function that creates the appropriate NeuroVol subclass
    based on the input data.

    Parameters
    ----------
    data : array-like
        Volume data
    space : NeuroSpace
        Spatial metadata
    label : str, optional
        Volume label
    indices : array-like, optional
        For sparse volumes, the indices of non-zero values

    Returns
    -------
    NeuroVol
        DenseNeuroVol, SparseNeuroVol, or LogicalNeuroVol    """
    if indices is not None:
        # Create sparse volume
        return SparseNeuroVol(data, space, indices, label)
    else:
        data = np.asarray(data)
        if data.dtype == bool:
            # Create logical volume
            return LogicalNeuroVol(data, space, label)
        else:
            # Create dense volume
            return DenseNeuroVol(data, space, label)


def _attach_nibabel_metadata(obj, img) -> None:
    header = getattr(img, "header", None)
    if header is not None:
        obj._nibabel_header = header.copy()
    if hasattr(img, "get_qform"):
        _, code = img.get_qform(coded=True)
        obj._nibabel_qform_code = int(code)
    if hasattr(img, "get_sform"):
        _, code = img.get_sform(coded=True)
        obj._nibabel_sform_code = int(code)
    receipt = _extract_receipt_extension(img)
    if receipt is not None:
        obj.provenance = receipt


def _restore_nibabel_xforms(img, obj) -> None:
    if hasattr(img, "set_qform") and hasattr(obj, "_nibabel_qform_code"):
        img.set_qform(obj.space.affine, code=obj._nibabel_qform_code)
    if hasattr(img, "set_sform") and hasattr(obj, "_nibabel_sform_code"):
        img.set_sform(obj.space.affine, code=obj._nibabel_sform_code)


def _embed_receipt_extension(img, receipt) -> None:
    """Attach a Receipt to ``img.header.extensions`` as a 'comment' extension.

    No-op when ``receipt`` is ``None`` or when the image header does not
    support extensions (non-NIfTI back ends).
    """
    if receipt is None:
        return
    header = getattr(img, "header", None)
    extensions = getattr(header, "extensions", None)
    if extensions is None:
        return
    try:
        from nibabel.nifti1 import Nifti1Extension
    except ImportError:  # pragma: no cover
        return
    payload = receipt.to_nifti_extension_bytes()
    extensions[:] = [ext for ext in extensions if not _is_receipt_extension(ext)]
    extensions.append(Nifti1Extension(6, payload))


def _is_receipt_extension(ext) -> bool:
    try:
        if int(ext.get_code()) != 6:
            return False
        content = bytes(ext.get_content())
    except Exception:  # pragma: no cover
        return False
    from .results import RECEIPT_NIFTI_PREFIX

    return content.rstrip(b"\x00").decode("utf-8", errors="replace").startswith(
        RECEIPT_NIFTI_PREFIX
    )


def _extract_receipt_extension(img):
    """Recover a :class:`~neuroim.results.Receipt` from a nibabel image's
    'comment' extension, or ``None`` if no marker extension is present.
    """
    header = getattr(img, "header", None)
    extensions = getattr(header, "extensions", None)
    if not extensions:
        return None
    from .results import Receipt

    for ext in extensions:
        try:
            if int(ext.get_code()) != 6:
                continue
            content = bytes(ext.get_content())
        except Exception:  # pragma: no cover
            continue
        receipt = Receipt.from_nifti_extension_bytes(content)
        if receipt is not None:
            return receipt
    return None

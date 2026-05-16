"""Spatially aware containers for 4D neuroimaging series."""

from abc import ABC, abstractmethod
import numpy as np
import warnings as _warnings
from typing import Any, Union, Tuple, List, Optional
from scipy import sparse

from .neuro_space import NeuroSpace
from .neuro_vol import (
    NeuroVol,
    DenseNeuroVol,
    LogicalNeuroVol,
    _attach_nibabel_metadata,
    _embed_receipt_extension,
    _restore_nibabel_xforms,
)
from .axis import drop_axis
from ._array_guard import refuse_array_conversion
from .exceptions import OutOfBoundsError, WorldOutOfBoundsError, InvalidArgumentError


def _readonly_array(value, *, dtype):
    arr = np.array(value, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def _empty_time_slice_space(dim, spacing, origin, axes):
    """Build the private space needed for NumPy-valid empty time slices."""
    ndim = len(dim)
    trans = np.eye(ndim + 1)
    trans[:ndim, :ndim] = np.diag(spacing)
    trans[:ndim, ndim] = origin

    space = object.__new__(NeuroSpace)
    object.__setattr__(space, "_frozen", False)
    object.__setattr__(space, "dim", _readonly_array(dim, dtype=int))
    object.__setattr__(space, "spacing", _readonly_array(spacing, dtype=float))
    object.__setattr__(space, "origin", _readonly_array(origin, dtype=float))
    object.__setattr__(space, "trans", _readonly_array(trans, dtype=float))
    object.__setattr__(
        space, "inverse", _readonly_array(np.linalg.inv(trans), dtype=float)
    )
    object.__setattr__(space, "axes", axes)
    object.__setattr__(space, "_frozen", True)
    return space


def _warn_legacy_method(name: str, replacement: str) -> None:
    _warnings.warn(
        f"NeuroVec.{name}() is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _call_without_legacy_warning(func, *args, **kwargs):
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", DeprecationWarning)
        return func(*args, **kwargs)


class NeuroVec(ABC):
    """Abstract base class for 4D neuroimaging vectors.

    Parameters
    ----------
    space : NeuroSpace
        4D spatial metadata
    """

    def __init__(self, space: NeuroSpace):
        if space.ndim != 4:
            raise ValueError("NeuroVec requires 4D space")
        self.space = space

    def __array__(self, *args, **kwargs):
        # .data returns the natural-rank 4-D ndarray; .as_matrix() would
        # hand back a (n_voxels, n_time) matricization, not what a caller
        # reaching for np.asarray(vec) expects.
        refuse_array_conversion(self, ".data")

    @classmethod
    def from_array(cls, data, space: NeuroSpace) -> "DenseNeuroVec":
        """Create a dense vector from array data and a spatial contract."""
        return DenseNeuroVec(data, space)

    @classmethod
    def from_nibabel(cls, img: Any, *, lazy: bool = False) -> "DenseNeuroVec":
        """Create a 4D vector from a nibabel SpatialImage-like object.

        ``lazy`` is accepted as part of the public adapter contract.  The
        current minimal implementation avoids ``get_fdata()`` and reads from
        ``dataobj`` directly when available.
        """
        if not hasattr(img, "shape") or not hasattr(img, "affine"):
            raise TypeError("from_nibabel expects an image with shape and affine")
        data_obj = getattr(img, "dataobj", None)
        if data_obj is None and not hasattr(img, "get_fdata"):
            raise TypeError("from_nibabel expects an image with dataobj or get_fdata()")
        data = np.asanyarray(data_obj) if data_obj is not None else img.get_fdata()
        if data.ndim == 3:
            data = data[..., np.newaxis]
        if data.ndim != 4:
            raise ValueError(
                f"NeuroVec.from_nibabel expects 3D or 4D data, got {data.ndim}D"
            )

        space = NeuroSpace.from_nibabel(img)
        if space.ndim == 3:
            space = space.add_dim(n=1, size=data.shape[3])

        vec = DenseNeuroVec(data, space)
        _attach_nibabel_metadata(vec, img)
        return vec

    def to_nibabel(self, cls=None):
        """Convert this vector to a nibabel image.

        When ``self.provenance`` is a :class:`~neuroim.results.Receipt`, it
        is embedded as a NIfTI 'comment' header extension (ecode 6, marker
        prefix ``neuroim/receipt/v1:``) — matching the NeuroVol path so a
        clean-process round-trip via :func:`~neuroim.io.read_image`
        recovers it on ``.provenance``.  See
        ``docs/spec/receipt-nifti-extension.md``.
        """
        import nibabel as nib

        img_cls = cls or nib.Nifti1Image
        data = self.to_dense().data if hasattr(self, "to_dense") else self.data
        header = getattr(self, "_nibabel_header", None)
        if header is not None:
            header = header.copy()
        img = img_cls(data, self.space.affine, header=header)
        _restore_nibabel_xforms(img, self)
        _embed_receipt_extension(img, getattr(self, "provenance", None))
        return img

    @abstractmethod
    def __getitem__(self, key):
        """Extract values using various indexing methods."""
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        """Set values using various indexing methods."""
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
            Time series for the specified voxel"""
        return self.series_at(x, y, z)

    def _validate_out_of_bounds_mode(self, out_of_bounds: str) -> None:
        if out_of_bounds not in {"raise", "zero"}:
            raise InvalidArgumentError("out_of_bounds must be 'raise' or 'zero'")

    def _spatial_shape(self) -> Tuple[int, int, int]:
        return tuple(int(d) for d in self.shape[:3])

    def _time_length(self) -> int:
        return int(self.shape[3])

    @property
    def spatial_space(self) -> NeuroSpace:
        """Return the 3-D spatial subspace for world/voxel queries."""
        return self.space if self.space.ndim == 3 else self.space.drop_dim(3)

    def _coords_valid_mask(self, coords: np.ndarray) -> np.ndarray:
        shape = self._spatial_shape()
        return (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < shape[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < shape[1])
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < shape[2])
        )

    def _indices_valid_mask(self, indices: np.ndarray) -> np.ndarray:
        total = int(np.prod(self._spatial_shape()))
        return (indices >= 0) & (indices < total)

    def _raise_oob(self, label: str, values: np.ndarray, valid_mask: np.ndarray) -> None:
        if np.all(valid_mask):
            return
        bad = values[~valid_mask]
        preview = bad[:5].tolist()
        suffix = "" if bad.shape[0] <= 5 else f" ... (+{bad.shape[0] - 5} more)"
        raise OutOfBoundsError(
            f"{label} out of bounds for spatial shape {self._spatial_shape()}: "
            f"{preview}{suffix}"
        )

    def series_at(
        self, x: int, y: int, z: int, *, out_of_bounds: str = "raise"
    ) -> np.ndarray:
        """Extract the time series at one voxel coordinate.

        ``out_of_bounds='raise'`` is the default safety contract.  Use
        ``out_of_bounds='zero'`` only for algorithms that intentionally need
        sparse/searchlight-style zero-fill semantics.
        """
        self._validate_out_of_bounds_mode(out_of_bounds)
        coord = np.asarray([[x, y, z]], dtype=int)
        valid = self._coords_valid_mask(coord)
        if not valid[0]:
            if out_of_bounds == "zero":
                return np.zeros(self._time_length())
            self._raise_oob("coordinate", coord, valid)

        impl = getattr(self, "_series", None)
        if impl is not None:
            return impl(x, y, z)
        return _call_without_legacy_warning(self.series, x, y, z)

    def series_at_coords(
        self, coords: np.ndarray, *, out_of_bounds: str = "raise"
    ) -> np.ndarray:
        """Extract time-by-voxel series at an ``N x 3`` coordinate matrix."""
        self._validate_out_of_bounds_mode(out_of_bounds)
        coords = np.asarray(coords, dtype=int)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an N x 3 coordinate matrix")
        valid = self._coords_valid_mask(coords)
        if not np.all(valid):
            if out_of_bounds == "zero":
                result = np.zeros((self._time_length(), coords.shape[0]))
                if np.any(valid):
                    result[:, valid] = self.series_at_coords(
                        coords[valid], out_of_bounds="raise"
                    )
                return result
            self._raise_oob("coordinates", coords, valid)

        impl = getattr(self, "_series", None)
        if impl is not None:
            return impl(coords)
        return _call_without_legacy_warning(self.series, coords)

    def series_at_indices(
        self, indices: np.ndarray, *, out_of_bounds: str = "raise"
    ) -> np.ndarray:
        """Extract time-by-voxel series at Fortran-order linear indices.

        The linear-index contract is column-major/Fortran-order, matching
        ``numpy.ravel_multi_index(..., order="F")`` and the storage contract
        documented for ``VoxelSeriesStore``.
        """
        self._validate_out_of_bounds_mode(out_of_bounds)
        indices = np.asarray(indices, dtype=int)
        scalar = indices.ndim == 0
        flat = np.atleast_1d(indices)
        valid = self._indices_valid_mask(flat)
        if not np.all(valid):
            if out_of_bounds == "zero":
                if scalar:
                    return np.zeros(self._time_length())
                result = np.zeros((self._time_length(), flat.shape[0]))
                if np.any(valid):
                    result[:, valid] = self.series_at_indices(
                        flat[valid], out_of_bounds="raise"
                    )
                return result
            self._raise_oob("linear indices", flat, valid)

        impl = getattr(self, "_series", None)
        if impl is not None:
            return impl(int(flat[0])) if scalar else impl(flat)
        return _call_without_legacy_warning(
            self.series, int(flat[0]) if scalar else flat
        )

    def _voxel_from_world(self, world_xyz: np.ndarray) -> np.ndarray:
        world = np.asarray(world_xyz, dtype=float)
        if world.shape != (3,):
            raise ValueError(f"world_xyz must have shape (3,); got {world.shape}")
        return np.asarray(self.space.world_to_grid(world), dtype=int)

    def _raise_world_oob(self, world_xyz: np.ndarray, voxel: np.ndarray) -> None:
        raise WorldOutOfBoundsError(
            f"world coord {tuple(float(x) for x in world_xyz)} mm maps to voxel "
            f"{tuple(int(v) for v in voxel)} which is outside the image grid "
            f"of shape {self._spatial_shape()}."
        )

    def series_at_world(
        self, world_xyz: np.ndarray, *, out_of_bounds: str = "raise"
    ) -> np.ndarray:
        """Extract one voxel time series at a 3-D world-coordinate seed."""
        voxel = self._voxel_from_world(world_xyz)
        try:
            return self.series_at(
                int(voxel[0]),
                int(voxel[1]),
                int(voxel[2]),
                out_of_bounds=out_of_bounds,
            )
        except IndexError as exc:
            if out_of_bounds == "zero":
                raise
            self._raise_world_oob(np.asarray(world_xyz, dtype=float), voxel)
            raise exc  # pragma: no cover - _raise_world_oob always raises

    def series_roi_world(
        self,
        center_xyz: np.ndarray,
        radius: float = 0.0,
        *,
        return_legacy: bool = False,
    ):
        """Extract a typed ROI time-series result around a world-coordinate seed."""
        from .roi import ROICoords

        center = self._voxel_from_world(center_xyz)
        if radius < 0:
            raise ValueError("radius must be non-negative")
        try:
            self.series_at(int(center[0]), int(center[1]), int(center[2]))
        except IndexError as exc:
            self._raise_world_oob(np.asarray(center_xyz, dtype=float), center)
            raise exc  # pragma: no cover - _raise_world_oob always raises

        if radius == 0:
            coords = center[None, :]
        else:
            spacing = np.asarray(self.spatial_space.spacing[:3], dtype=float)
            voxel_radius = np.ceil(radius / spacing).astype(int)
            lower = np.maximum(0, center - voxel_radius)
            upper = np.minimum(np.asarray(self._spatial_shape()), center + voxel_radius + 1)
            grids = np.meshgrid(
                range(lower[0], upper[0]),
                range(lower[1], upper[1]),
                range(lower[2], upper[2]),
                indexing="ij",
            )
            candidates = np.column_stack([g.ravel() for g in grids])
            distances = np.sqrt(np.sum(((candidates - center) * spacing) ** 2, axis=1))
            coords = candidates[distances <= radius]

        roi = ROICoords(coords, space=self.spatial_space)
        return self.series_roi(
            roi,
            return_legacy=return_legacy,
            _method_name="series_roi_world",
            _radius=float(radius),
        )

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
            Time series data"""
        pass

    def series_roi(
        self,
        roi,
        *,
        return_legacy: bool = False,
        _method_name: str = "series_roi",
        _radius: Optional[float] = None,
    ):
        """Extract time series for all voxels in an ROI.

        Parameters
        ----------
        roi : ROIVol or ROICoords
            The region of interest
        return_legacy : bool, optional
            Default ``False`` returns a typed
            :class:`~neuroim.results.ROIExtractionResult`.  Pass ``True`` for
            the historical time-by-voxel ndarray; this opt-in emits a
            :class:`DeprecationWarning` and will be removed in the next
            minor.

        Returns
        -------
        ROIExtractionResult or np.ndarray
            Typed result by default; bare time-by-voxel ndarray with
            ``return_legacy=True``.
        """
        import warnings as _warnings
        from .roi import ROIVol, ROICoords
        from .results import ROIExtractionResult, make_receipt
        from .verify import assert_same_space

        if return_legacy:
            _warnings.warn(
                "return_legacy=True is deprecated and will be removed in the "
                "next minor release; consume the typed ROIExtractionResult "
                "instead (.values, .coords, .provenance).",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(roi, ROIVol):
            # Extract coordinates from ROIVol - it's directly stored in roi.coords
            coords = roi.coords
        elif isinstance(roi, ROICoords):
            # Use coordinates directly
            coords = roi.coords
        else:
            raise TypeError(f"roi must be ROIVol or ROICoords, got {type(roi)}")

        assert_same_space(self.space, roi.space)

        # Use the series method with coordinate matrix
        values = self.series_at_coords(coords, out_of_bounds="zero")
        if return_legacy:
            return values

        coords = np.ascontiguousarray(coords, dtype=int)
        from .results import RoiOpParams, receipt_for

        receipt = receipt_for(
            self,
            mask=coords,
            n_voxels=int(coords.shape[0]),
            params=RoiOpParams(method_name=_method_name, radius=_radius),
        )

        # Provenance threading (ME-9): if self carries a Receipt
        # (e.g. from a prior concat / from_nibabel), compose it into the
        # ROI extraction's output Receipt so the full pipeline is
        # reconstructible.  Receipt.merge raises on input_space_hash
        # disagreement — that's the upstream-tamper / silent-mismatch
        # catch the mission claim depends on.
        upstream = getattr(self, "provenance", None)
        if upstream is not None:
            receipt = upstream.merge(
                receipt,
                method_name=f"{upstream.method_name}+{_method_name}",
            )

        return ROIExtractionResult(
            values=np.ascontiguousarray(values),
            coords=coords,
            space=roi.space,
            mask_hash=receipt.mask_hash,
            provenance=receipt,
        )

    def temporal_snr(self, *, mask=None):
        """Compute a masked 3-D temporal SNR map with provenance.

        The result is a :class:`~neuroim.neuro_vol.DenseNeuroVol` on this
        vector's spatial frame.  If ``mask`` is supplied, its space must be
        compatible with the vector's spatial space; voxels outside the mask
        and zero-variance voxels are set to zero.
        """
        from .neuro_vol import DenseNeuroVol
        from . import verify as _verify

        mask_data = None
        if mask is not None:
            _verify.assert_same_space(self, mask)
            mask_data = np.asarray(mask.data, dtype=bool)

        data = np.asarray(self.to_dense().data, dtype=np.float64)
        if data.ndim != 4:
            raise ValueError(f"expected 4D BOLD, got {data.ndim}D")

        spatial_shape = data.shape[:3]
        if mask_data is None:
            mask_data = np.ones(spatial_shape, dtype=bool)
        elif mask_data.shape != spatial_shape:
            raise ValueError(
                f"mask shape {mask_data.shape} does not match BOLD spatial shape {spatial_shape}"
            )

        mean = data.mean(axis=3)
        std = data.std(axis=3)
        tsnr = np.zeros(spatial_shape, dtype=np.float64)
        valid = mask_data & (std > 0)
        tsnr[valid] = mean[valid] / std[valid]

        spatial = self.spatial_space
        from .results import TemporalReductionParams, receipt_for

        receipt = receipt_for(
            spatial,
            mask=mask_data,
            n_voxels=int(np.count_nonzero(mask_data)),
            params=TemporalReductionParams(method_name="temporal_snr"),
        )
        upstream = getattr(self, "provenance", None)
        if upstream is not None:
            try:
                receipt = upstream.merge(
                    receipt,
                    method_name=f"{upstream.method_name}+temporal_snr",
                )
            except ValueError:
                pass
        return DenseNeuroVol(tsnr, spatial, label="temporal_snr", provenance=receipt)

    def parcel_means(self, atlas, *, label: str = ""):
        """Extract per-parcel mean BOLD time series from an atlas.

        ``atlas`` may be a typed :class:`neuroim.atlas.VolumetricAtlas`, an
        integer-labelled :class:`DenseNeuroVol`, or an already-built
        :class:`ClusteredNeuroVol`.  The returned
        :class:`ClusteredNeuroVec` stores a ``(n_time, n_clusters)`` matrix,
        sorted by ascending cluster id, and carries a provenance
        :class:`Receipt` recording the input space and atlas label payload.
        """
        from .clustered_neuro_vec import ClusteredNeuroVec
        from .clustered_neuro_vol import ClusteredNeuroVol
        from .results import RoiOpParams, receipt_for
        from .verify import assert_same_space

        atlas_provenance = None
        if hasattr(atlas, "to_clustered_vol") and hasattr(atlas, "label_image"):
            cvol = atlas.to_clustered_vol()
            atlas_payload = np.asarray(atlas.label_image.data, dtype=np.int32)
            atlas_provenance = getattr(atlas, "provenance", None)
        elif isinstance(atlas, ClusteredNeuroVol):
            cvol = atlas
            atlas_payload = cvol.as_dense().data
            atlas_provenance = getattr(cvol, "atlas_provenance", None)
        elif isinstance(atlas, DenseNeuroVol):
            atlas_payload = np.asarray(atlas.data, dtype=np.int32)
            mask = LogicalNeuroVol(atlas_payload > 0, atlas.space)
            cvol = ClusteredNeuroVol(mask, atlas_payload)
        else:
            raise TypeError(
                "atlas must be a DenseNeuroVol or ClusteredNeuroVol, "
                f"got {type(atlas).__name__}"
            )

        assert_same_space(self, cvol)

        data = np.asarray(self.to_dense().data, dtype=np.float64)
        if data.ndim != 4:
            raise ValueError(f"parcel_means expects 4D data, got {data.ndim}D")
        nx, ny, nz, nt = data.shape
        flat = data.reshape(nx * ny * nz, nt, order="F")

        cluster_ids = np.sort(np.array(list(cvol.cluster_map.keys())))
        ts = np.empty((nt, cluster_ids.size), dtype=np.float64)
        for col, cid in enumerate(cluster_ids):
            indices = cvol.cluster_map[int(cid)]
            ts[:, col] = flat[indices, :].mean(axis=0)

        receipt = receipt_for(
            self,
            mask=np.asarray(atlas_payload, dtype=np.int32),
            n_voxels=int(cvol.num_clusters()),
            params=RoiOpParams(method_name="parcel_means"),
            upstream=self,
        )
        return ClusteredNeuroVec(
            cvol,
            ts,
            label=label,
            provenance=receipt,
            atlas_provenance=atlas_provenance,
        )

    @abstractmethod
    def as_sparse(self, mask=None) -> "SparseNeuroVec":
        """Convert to sparse representation."""
        pass

    def to_dense(self) -> "DenseNeuroVec":
        """Convert to dense representation."""
        return _call_without_legacy_warning(self.as_dense)

    def to_sparse(self, mask=None) -> "SparseNeuroVec":
        """Convert to sparse representation."""
        return _call_without_legacy_warning(self.as_sparse, mask)

    @abstractmethod
    def sub_vector(self, indices: Union[int, slice, np.ndarray]) -> "NeuroVec":
        """Extract subset of volumes."""
        pass

    def subvolumes(self, indices: Union[int, slice, np.ndarray]) -> "NeuroVec":
        """Extract a subset of volumes along the time axis."""
        return _call_without_legacy_warning(self.sub_vector, indices)

    def vols(self, indices=None):
        """Extract volumes as list or single volume."""
        if indices is None:
            indices = range(self.shape[3])
            return [self[..., i] for i in indices]
        elif isinstance(indices, int):
            return self[..., indices]
        else:
            return [self[..., i] for i in indices]

    def concat(self, *others: "NeuroVec") -> "NeuroVec":
        """Concatenate multiple NeuroVecs along time dimension."""
        from .verify import assert_same_space

        # Concatenate by converting all to dense and stacking
        all_vecs = [self] + list(others)
        for vec in all_vecs[1:]:
            assert_same_space(self.space, vec.space)
        # Default: convert to dense and delegate
        dense_self = self.to_dense() if not isinstance(self, DenseNeuroVec) else self
        dense_others = [
            v.to_dense() if not isinstance(v, DenseNeuroVec) else v for v in others
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
        """Dimensions of the 4D data."""
        return self.space.dim

    @property
    def dtype(self) -> np.dtype:
        """Data dtype without forcing full materialization when possible."""
        if "_dtype" in self.__dict__:
            return self.__dict__["_dtype"]
        if "source" in self.__dict__:
            return np.dtype(self.__dict__["source"].dtype)
        if "_data" in self.__dict__:
            return np.dtype(self.__dict__["_data"].dtype)
        if "data" in self.__dict__:
            return np.dtype(self.__dict__["data"].dtype)
        data = getattr(self, "data", None)
        if data is not None:
            return np.dtype(data.dtype)
        raise AttributeError(f"{self.__class__.__name__} cannot report dtype")

    @dtype.setter
    def dtype(self, value) -> None:
        self.__dict__["_dtype"] = np.dtype(value)

    @property
    def store(self):
        """Voxel-series storage adapter for this vector."""
        from .storage import NeuroVecStoreAdapter

        return NeuroVecStoreAdapter(self)

    @property
    def spacing(self) -> np.ndarray:
        """Voxel dimensions."""
        return self.space.spacing

    @property
    def origin(self) -> np.ndarray:
        """Origin coordinates."""
        return self.space.origin

    @property
    def trans(self) -> np.ndarray:
        """Transformation matrix."""
        return self.space.trans

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}\n"
            f"  Type      : {self.__class__.__name__}\n"
            f"  Dimension : {' X '.join(map(str, self.dim))}\n"
            f"  Spacing   : {' X '.join(map(str, self.spacing))}\n"
            f"  Origin    : {', '.join(map(str, self.origin))}"
        )

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

    Parameters
    ----------
    data : array-like
        4D array or matrix of voxel values
    space : NeuroSpace
        4D spatial metadata
    label : str, optional
        Vector label"""

    def __init__(self, data, space: NeuroSpace, label: str = ""):
        super().__init__(space)

        # Handle different input types
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # Matrix input (time x voxels or voxels x time)
                splen = np.prod(self.shape[:3])
                if data.shape[0] == splen:
                    # voxels x time -> reshape to 4D
                    data = data.T.reshape(self.shape, order="F")
                elif data.shape[1] == splen:
                    # time x voxels -> reshape to 4D
                    data = data.reshape(self.shape, order="F")
                else:
                    raise ValueError("Matrix dimensions do not match space dimensions")
            elif data.ndim == 1:
                # Vector input
                if data.size == np.prod(self.shape):
                    data = data.reshape(self.shape, order="F")
                else:
                    raise ValueError(
                        f"Data size {data.size} doesn't match space size {np.prod(self.shape)}"
                    )
            elif data.ndim == 4:
                if data.shape != self.shape:
                    raise ValueError(
                        f"Data shape {data.shape} doesn't match space shape {self.shape}"
                    )
            else:
                raise ValueError(f"Data must be 1D, 2D or 4D array, got {data.ndim}D")
        else:
            data = np.asarray(data)

        self.data = data
        self.label = label

    def as_dense(self) -> "DenseNeuroVec":
        """Deprecated alias for :meth:`to_dense`."""
        _warn_legacy_method("as_dense", "to_dense()")
        return self

    def __getitem__(self, key):
        """Extract values using various indexing methods."""
        if isinstance(key, tuple) and len(key) == 4:
            # Standard 4D indexing
            return self.data[key]
        elif (
            isinstance(key, tuple)
            and len(key) == 3
            and all(isinstance(k, (int, np.integer)) for k in key)
        ):
            # Get time series for single voxel
            return self.data[key[0], key[1], key[2], :]
        else:
            # Let numpy handle it
            result = self.data[key]
            # If we extracted a single volume, wrap it as NeuroVol
            if result.ndim == 3:
                vol_space = NeuroSpace(
                    result.shape,
                    spacing=self.spacing[:3],
                    origin=self.origin[:3],
                    axes=(
                        drop_axis(self.space.axes, 3) if self.space.ndim == 4 else None
                    ),
                    trans=self.trans[:4, :4] if self.space.ndim <= 4 else None,
                )
                return DenseNeuroVol(result, vol_space)
            # PAIN-13: a pure time-axis selection (spatial dims unchanged) must
            # remain a typed NeuroVec so downstream callers do not need to
            # re-wrap.  PAIN-14: attach a TemporalSliceParams Receipt that
            # records the slice indices and chains any upstream provenance,
            # so a downstream temporal_snr / series_roi composes the lineage.
            if (
                result.ndim == 4
                and tuple(result.shape[:3]) == tuple(self.data.shape[:3])
            ):
                from .axis import AxisSet4D, NamedAxis
                from .results import TemporalSliceParams, receipt_for

                spatial = self.spatial_space
                t_axis = NamedAxis("t", int(result.shape[3]))
                axes_4d = AxisSet4D(
                    spatial.axes.i, spatial.axes.j, spatial.axes.k, t_axis
                )
                new_dim = [int(d) for d in result.shape]
                new_spacing = [float(s) for s in spatial.spacing] + [1.0]
                new_origin = [float(o) for o in spatial.origin] + [0.0]
                if new_dim[3] == 0:
                    new_space = _empty_time_slice_space(
                        new_dim, new_spacing, new_origin, axes_4d
                    )
                else:
                    new_space = NeuroSpace(
                        dim=new_dim,
                        spacing=new_spacing,
                        origin=new_origin,
                        axes=axes_4d,
                    )
                sliced = DenseNeuroVec(result, new_space)

                # Extract the (start, stop, step) the caller passed on the
                # time axis, if any.  Handles ``vec[..., :N]`` (key is a
                # 2-tuple) and the explicit ``vec[:, :, :, slice]`` form.
                t_slice: Optional[slice] = None
                if isinstance(key, tuple):
                    if len(key) > 0 and key[-1] is Ellipsis:
                        t_slice = None
                    elif len(key) >= 1 and isinstance(key[-1], slice):
                        t_slice = key[-1]
                start = t_slice.start if t_slice is not None else None
                stop = t_slice.stop if t_slice is not None else None
                step = t_slice.step if t_slice is not None else None

                # Encode the slice in the method_name so a chained Receipt
                # reads e.g. "temporal_slice(start=10,stop=None,step=None)+
                # temporal_snr" — same shape as resample_vec(order=1,...).
                pieces = ["temporal_slice("]
                pieces.append(f"start={start},stop={stop},step={step}")
                pieces.append(")")
                slice_method_name = "".join(pieces)

                sliced.provenance = receipt_for(
                    new_space,
                    n_voxels=int(np.prod(new_space.dim[:3])),
                    params=TemporalSliceParams(
                        method_name=slice_method_name,
                        start=start,
                        stop=stop,
                        step=step,
                    ),
                    upstream=self,
                )
                return sliced
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
            Time series for the specified voxel"""
        return self.series_at(x, y, z)

    def series(self, x, y=None, z=None) -> np.ndarray:
        """Deprecated dispatcher for voxel time-series extraction."""
        _warn_legacy_method(
            "series",
            "series_at(), series_at_coords(), or series_at_indices()",
        )
        return self._series(x, y, z)

    def _series(self, x, y=None, z=None) -> np.ndarray:
        """Extract time series for voxel(s)."""
        if y is not None and z is not None:
            # Single voxel
            return self.data[x, y, z, :]
        elif isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 3:
                # Nx3 matrix of coordinates - vectorized version
                # Check bounds all at once
                valid_mask = (
                    (x[:, 0] >= 0)
                    & (x[:, 0] < self.shape[0])
                    & (x[:, 1] >= 0)
                    & (x[:, 1] < self.shape[1])
                    & (x[:, 2] >= 0)
                    & (x[:, 2] < self.shape[2])
                )

                # Initialize result
                result = np.zeros((x.shape[0], self.shape[3]))

                # Extract data for valid coordinates using advanced indexing
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    valid_coords = x[valid_indices]
                    result[valid_indices] = self.data[
                        valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2], :
                    ]

                return result.T  # Return as time x voxels
            elif x.ndim == 1:
                # Linear indices - vectorized version
                # Convert to 3D indices
                coords = np.unravel_index(x, self.shape[:3], order="F")
                # Use advanced indexing to extract all at once
                result = self.data[coords[0], coords[1], coords[2], :]
                return result.T
        elif isinstance(x, int):
            # Single linear index
            coords = np.unravel_index(x, self.shape[:3], order="F")
            return self.data[coords[0], coords[1], coords[2], :]
        else:
            raise ValueError("Invalid input for series extraction")

    def as_sparse(self, mask=None) -> "SparseNeuroVec":
        """Deprecated alias for :meth:`to_sparse`."""
        _warn_legacy_method("as_sparse", "to_sparse()")
        return self._as_sparse(mask)

    def _as_sparse(self, mask=None) -> "SparseNeuroVec":
        """Convert to sparse representation."""
        if mask is None:
            # Use all non-zero voxels
            mask_data = np.any(self.data != 0, axis=3)
            mask_space = NeuroSpace(
                self.shape[:3],
                spacing=self.spacing[:3],
                origin=self.origin[:3],
                axes=drop_axis(self.space.axes, 3) if self.space.ndim == 4 else None,
            )
            mask = LogicalNeuroVol(mask_data, mask_space)
        elif isinstance(mask, LogicalNeuroVol):
            from .verify import assert_same_space

            assert_same_space(self, mask)
        elif not isinstance(mask, LogicalNeuroVol):
            # Convert to LogicalNeuroVol
            mask_space = NeuroSpace(
                self.shape[:3],
                spacing=self.spacing[:3],
                origin=self.origin[:3],
                axes=drop_axis(self.space.axes, 3) if self.space.ndim == 4 else None,
            )
            mask = LogicalNeuroVol(mask, mask_space)

        # Extract data for masked voxels
        mask_indices = np.where(mask.data.ravel(order="F"))[0]
        data_flat = self.data.reshape(-1, self.shape[3], order="F")
        sparse_data = data_flat[mask_indices, :]

        return SparseNeuroVec(sparse_data.T, self.space, mask, self.label)

    def sub_vector(self, indices: Union[int, slice, np.ndarray]) -> "DenseNeuroVec":
        """Deprecated alias for :meth:`subvolumes`."""
        _warn_legacy_method("sub_vector", "subvolumes()")
        return self._sub_vector(indices)

    def _sub_vector(self, indices: Union[int, slice, np.ndarray]) -> "DenseNeuroVec":
        """Extract subset of volumes."""
        if isinstance(indices, int):
            indices = [indices]

        sub_data = self.data[..., indices]
        if sub_data.ndim == 3:
            sub_data = sub_data[..., np.newaxis]

        sub_space = NeuroSpace(
            (*self.shape[:3], sub_data.shape[3]),
            spacing=self.spacing,
            origin=self.origin,
            axes=self.space.axes,
        )

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
        flat = self.data.reshape(-1, self.shape[3], order="F")
        if subset is not None:
            return [flat[i] for i in subset]
        return [flat[i] for i in range(flat.shape[0])]

    def concat(self, *others: "DenseNeuroVec") -> "DenseNeuroVec":
        """Concatenate multiple NeuroVecs along time dimension."""
        from .verify import assert_same_space

        all_vecs = [self] + list(others)

        # Check spatial compatibility
        for vec in all_vecs[1:]:
            assert_same_space(self.space, vec.space)

        # Concatenate data
        all_data = [vec.data for vec in all_vecs]
        concat_data = np.concatenate(all_data, axis=3)

        # Create new space with combined time dimension
        concat_space = NeuroSpace(
            (*self.shape[:3], concat_data.shape[3]),
            spacing=self.spacing,
            origin=self.origin,
            axes=self.space.axes,
            trans=self.space.trans,
        )

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
        """Convert to matrix (voxels x time)."""
        return self.data.reshape(-1, self.shape[3], order="F")

    def scale_series(self, center: bool = True, scale: bool = True) -> "DenseNeuroVec":
        """Scale (center and/or normalize) each time series."""
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

    Parameters
    ----------
    data : np.ndarray
        2D array (time x masked_voxels)
    space : NeuroSpace
        4D spatial metadata
    mask : LogicalNeuroVol
        Mask defining which voxels are included
    label : str, optional
        Vector label"""

    def __init__(self, data: np.ndarray, space: NeuroSpace, mask, label: str = ""):
        super().__init__(space)

        # Accept indices array and convert to LogicalNeuroVol
        if isinstance(mask, np.ndarray) and not mask.dtype == bool:
            # Integer indices array - convert to LogicalNeuroVol
            mask_data = np.zeros(self.shape[:3], dtype=bool)
            mask_flat = mask_data.ravel(order="F")
            mask_flat[mask] = True
            mask_data = mask_flat.reshape(self.shape[:3], order="F")
            from .neuro_vol import LogicalNeuroVol as LNV

            mask_space = NeuroSpace(
                self.shape[:3], spacing=space.spacing[:3], origin=space.origin[:3]
            )
            mask = LNV(mask_data, mask_space)
        elif isinstance(mask, np.ndarray) and mask.dtype == bool:
            # Boolean array - convert to LogicalNeuroVol
            from .neuro_vol import LogicalNeuroVol as LNV

            mask_space = NeuroSpace(
                self.shape[:3], spacing=space.spacing[:3], origin=space.origin[:3]
            )
            mask = LNV(mask, mask_space)
        elif not isinstance(mask, LogicalNeuroVol):
            raise TypeError(
                "mask must be a LogicalNeuroVol, boolean array, or integer indices array"
            )

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
                raise ValueError(
                    f"Data shape {data.shape} doesn't match mask cardinality {mask.sum} and time dimension {self.shape[3]}"
                )
        else:
            raise ValueError("Data must be 2D array (time x masked_voxels)")

        self.data = data
        self.mask = mask
        self.label = label

        # Create lookup for fast indexing
        self._lookup = np.where(mask.data.ravel(order="F"))[0]
        self._inverse_lookup = np.full(np.prod(self.shape[:3]), -1, dtype=int)
        self._inverse_lookup[self._lookup] = np.arange(len(self._lookup))

    def __getitem__(self, key):
        """Extract values using various indexing methods."""
        # Create dense version for complex indexing
        dense_data = self._to_dense_array()
        result = dense_data[key]

        # If we extracted a single volume, wrap it as NeuroVol
        if result.ndim == 3:
            vol_space = NeuroSpace(
                result.shape,
                spacing=self.spacing[:3],
                origin=self.origin[:3],
                axes=drop_axis(self.space.axes, 3) if self.space.ndim == 4 else None,
            )
            return DenseNeuroVol(result, vol_space)
        return result

    def __setitem__(self, key, value):
        """Set values using various indexing methods."""
        # For now, convert to dense, modify, convert back
        # This is inefficient but ensures correctness
        dense_data = self._to_dense_array()
        dense_data[key] = value

        # Extract updated sparse data
        data_flat = dense_data.reshape(-1, self.shape[3], order="F")
        self.data = data_flat[self._lookup, :].T

    def _to_dense_array(self) -> np.ndarray:
        """Convert sparse data to dense 4D array."""
        dense = np.zeros(self.shape, dtype=self.data.dtype, order="F")
        data_flat = dense.reshape(-1, self.shape[3], order="F")
        data_flat[self._lookup, :] = self.data.T
        return dense

    def series(self, x, y=None, z=None) -> np.ndarray:
        """Deprecated dispatcher for voxel time-series extraction."""
        _warn_legacy_method(
            "series",
            "series_at(), series_at_coords(), or series_at_indices()",
        )
        return self._series(x, y, z)

    def _series(self, x, y=None, z=None) -> np.ndarray:
        """Extract time series for voxel(s)."""
        if y is not None and z is not None:
            # Single voxel
            linear_idx = np.ravel_multi_index((x, y, z), self.shape[:3], order="F")
            sparse_idx = self._inverse_lookup[linear_idx]
            if sparse_idx == -1:
                return np.zeros(self.shape[3])
            return self.data[:, sparse_idx]
        elif isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 3:
                # Nx3 matrix of coordinates
                result = np.zeros((self.shape[3], x.shape[0]))
                for i, coord in enumerate(x):
                    if not (
                        0 <= coord[0] < self.shape[0]
                        and 0 <= coord[1] < self.shape[1]
                        and 0 <= coord[2] < self.shape[2]
                    ):
                        continue
                    linear_idx = np.ravel_multi_index(coord, self.shape[:3], order="F")
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

    def as_sparse(self, mask=None) -> "SparseNeuroVec":
        """Deprecated alias for :meth:`to_sparse`."""
        _warn_legacy_method("as_sparse", "to_sparse()")
        return self._as_sparse(mask)

    def _as_sparse(self, mask=None) -> "SparseNeuroVec":
        """Already sparse, return self or apply new mask."""
        if mask is None:
            return self

        if isinstance(mask, LogicalNeuroVol):
            from .verify import assert_same_space

            assert_same_space(self, mask)

        # Apply additional mask by converting to dense, then back to sparse with new mask.
        dense = self.to_dense()
        return dense.to_sparse(mask)

    def as_dense(self) -> DenseNeuroVec:
        """Deprecated alias for :meth:`to_dense`."""
        _warn_legacy_method("as_dense", "to_dense()")
        return self._as_dense()

    def _as_dense(self) -> DenseNeuroVec:
        """Convert to dense representation."""
        dense_data = self._to_dense_array()
        return DenseNeuroVec(dense_data, self.space, self.label)

    def sub_vector(self, indices: Union[int, slice, np.ndarray]) -> "SparseNeuroVec":
        """Deprecated alias for :meth:`subvolumes`."""
        _warn_legacy_method("sub_vector", "subvolumes()")
        return self._sub_vector(indices)

    def _sub_vector(self, indices: Union[int, slice, np.ndarray]) -> "SparseNeuroVec":
        """Extract subset of volumes."""
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(self.shape[3])))

        sub_data = self.data[indices, :]

        sub_space = NeuroSpace(
            (*self.shape[:3], len(indices)),
            spacing=self.spacing,
            origin=self.origin,
            axes=self.space.axes,
        )

        return SparseNeuroVec(sub_data, sub_space, self.mask, self.label)

    def concat(self, *others: "SparseNeuroVec") -> "SparseNeuroVec":
        """Concatenate multiple SparseNeuroVecs along time dimension."""
        from .verify import assert_same_space

        all_vecs = [self] + list(others)

        # Check compatibility
        for vec in all_vecs[1:]:
            assert_same_space(self.space, vec.space)
            if not np.array_equal(vec.mask.data, self.mask.data):
                raise ValueError("All SparseNeuroVecs must have same mask")

        # Concatenate data
        all_data = [vec.data for vec in all_vecs]
        concat_data = np.vstack(all_data)

        # Create new space
        concat_space = NeuroSpace(
            (*self.shape[:3], concat_data.shape[0]),
            spacing=self.spacing,
            origin=self.origin,
            axes=self.space.axes,
            trans=self.space.trans,
        )

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
            return self.to_dense()._arithmetic_op(other, op)
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
        Combined 4D vector (DenseNeuroVec or SparseNeuroVec, depending on input)."""
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
        result = first.concat(*vecs[1:])
        if label:
            result.label = label
        return result

    elif isinstance(first, SparseNeuroVec):
        # List of sparse vectors - concatenate with sparse semantics
        result = first.concat(*vecs[1:])
        if label:
            result.label = label
        return result

    else:
        raise TypeError(
            "Input must be list of NeuroVol, DenseNeuroVec, or SparseNeuroVec objects"
        )


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
        DenseNeuroVec or SparseNeuroVec"""
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
            mask_space = NeuroSpace(
                space.dim[:3],
                spacing=space.spacing[:3],
                origin=space.origin[:3],
                axes=drop_axis(space.axes, 3) if space.ndim == 4 else None,
            )
            mask = LogicalNeuroVol(mask, mask_space)

        # If data is 4D, extract sparse representation
        if data.ndim == 4:
            mask_indices = np.where(mask.data.ravel(order="F"))[0]
            data_flat = data.reshape(-1, data.shape[3], order="F")
            sparse_data = data_flat[mask_indices, :].T
            return SparseNeuroVec(sparse_data, space, mask, label)
        else:
            return SparseNeuroVec(data, space, mask, label)

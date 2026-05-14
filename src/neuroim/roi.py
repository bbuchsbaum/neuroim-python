"""Region of Interest (ROI) Classes.

Includes ROI, ROICoords, ROIVol, ROIVec, and ROIVolWindow.
"""

import warnings

import numpy as np
from typing import Union, Tuple, Optional, List
from abc import ABC, abstractmethod
from .neuro_space import NeuroSpace
from .typing import MaskLike, NeuroVolLike

_VolumeInput = Union[NeuroVolLike, MaskLike]

_LEGACY_DEPRECATION_MSG = (
    "return_legacy=True is deprecated and will be removed in the next minor "
    "release; consume the typed ROIExtractionResult instead (.values, "
    ".coords, .provenance)."
)


def _warn_legacy_return(return_legacy: bool) -> None:
    if return_legacy:
        warnings.warn(_LEGACY_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)

class ROI(ABC):
    """Abstract base class for Region of Interest (ROI) objects.
    
    """
    pass

class ROICoords(ROI):
    """A class representing a region of interest (ROI) in a brain image,
    defined by a set of coordinates.
    
    This class stores the geometric space of the image and the coordinates 
    of the voxels within the ROI.
    
    Parameters
    ----------
    coords : np.ndarray
        Matrix with 3 columns representing (i, j, k) coordinates.
        Each row represents a coordinate.
    space : NeuroSpace, optional
        NeuroSpace object defining the spatial reference. If not provided,
        creates a default space based on coordinate bounds.
        
    """
    
    def __init__(self, coords: np.ndarray, space: Optional[NeuroSpace] = None):
        coords = np.atleast_2d(coords)
        if coords.shape[1] != 3:
            raise ValueError("coords must be a matrix with 3 columns (i,j,k)")
        
        self.coords = coords
        
        # If no space provided, create default space like R does
        if space is None:
            # Calculate bounds from coordinates
            if len(coords) > 0:
                max_coords = coords.max(axis=0) + 1
                from .neuro_space import NeuroSpace
                space = NeuroSpace(dim=max_coords.astype(int))
            else:
                # For empty coords, create minimal space
                from .neuro_space import NeuroSpace
                space = NeuroSpace(dim=[1, 1, 1])
        
        self.space = space
    
    @property
    def dim(self) -> Tuple[int, int]:
        """Dimensions of the coordinate matrix.
        
        """
        return self.coords.shape
    
    def __len__(self) -> int:
        """Number of coordinates in the ROI.
        
        """
        return self.coords.shape[0]
    
    def __getitem__(self, i):
        """Extract subset of coordinates.
        
        """
        if isinstance(i, (int, np.integer)):
            # Single row
            return ROICoords(self.coords[i:i+1, :], self.space)
        elif isinstance(i, (list, np.ndarray, slice)):
            # Multiple rows
            return ROICoords(self.coords[i, :], self.space)
        else:
            raise TypeError(f"Invalid index type: {type(i)}")
    
    def __repr__(self):
        """String representation matching R's show method."""
        return f"ROICoords\n  Dimension      : {self.dim[0]} x {self.dim[1]}"
    
    def indices(self) -> np.ndarray:
        """Get linear indices for the coordinates.
        
        Returns
        -------
        np.ndarray
            Linear indices (0-based)
            
        """
        return self.space.grid_to_index(self.coords)

class ROIVol(ROICoords):
    """Class representing a volumetric region of interest (ROI) in a brain image,
    defined by a set of coordinates and associated data values.
    
    Parameters
    ----------
    data : np.ndarray
        Numeric vector of values corresponding to each coordinate
    space : NeuroSpace
        NeuroSpace object defining the spatial reference
    coords : np.ndarray
        Matrix with 3 columns representing (i,j,k) coordinates
        
    """
    
    def __init__(self, data: np.ndarray, space: NeuroSpace, coords: np.ndarray):
        super().__init__(coords, space)
        
        data = np.asarray(data).ravel()
        if len(data) != len(self.coords):
            raise ValueError(f"length of data ({len(data)}) must equal number of coordinates ({len(self.coords)})")
        
        self.data = data
    
    def __getitem__(self, key):
        """Extract data or coordinates based on indexing.
        
        """
        if isinstance(key, tuple):
            i, j = key
            if isinstance(i, (int, np.integer, list, np.ndarray, slice)):
                if isinstance(j, (int, np.integer, list, np.ndarray, slice)):
                    # Extract coords[i, j]
                    return self.coords[i, j]
                else:
                    # Extract data[i] 
                    return ROIVol(self.data[i], self.space, self.coords[i, :])
            elif isinstance(i, np.ndarray) and i.shape[1] == 3:
                # New coordinates provided
                if j is None:
                    return ROIVol(self.data, self.space, i)
                else:
                    # Extract columns from new coords
                    return i[:, j]
            elif isinstance(i, ROICoords):
                if j is None:
                    # Use coords from ROICoords
                    return self[i.coords]
                else:
                    # Extract columns from ROICoords
                    return i.coords[:, j]
        else:
            # Single index
            if isinstance(key, (int, np.integer, list, np.ndarray, slice)):
                # Extract data subset
                return ROIVol(self.data[key], self.space, self.coords[key, :])
            elif key is None:
                # Return all data
                return self.data
            else:
                raise TypeError(f"Invalid index type: {type(key)}")
    
    def __repr__(self):
        """String representation matching R's show method."""
        val_range = (np.nanmin(self.data), np.nanmax(self.data))
        return (f"\n ROIVol Object \n\n"
                f"Properties:\n"  
                f"  Dimensions:  {self.dim[0]} x {self.dim[1]}\n"
                f"  ROI Points:  {len(self):,}\n"
                f"  Value Range: [{val_range[0]:.2f}, {val_range[1]:.2f}]")
    
    def as_numeric(self) -> np.ndarray:
        """Convert to numeric array.
        
        """
        return self.data
    
    def as_sparse(self):
        """Convert to SparseNeuroVol.
        
        """
        from .neuro_vol import SparseNeuroVol
        return SparseNeuroVol(data=self.data, space=self.space, 
                              indices=self.indices())
    
    def as_logical(self):
        """Convert to LogicalNeuroVol.
        
        """
        from .neuro_vol import LogicalNeuroVol
        mask = np.zeros(self.space.dim, dtype=bool, order='F')
        mask.ravel(order='F')[self.indices()] = True
        return LogicalNeuroVol(mask, self.space)
    
    def get_coords(self, real: bool = False) -> np.ndarray:
        """Get coordinates.
        
        Parameters
        ----------
        real : bool
            If True, return real-world coordinates.
            If False, return grid coordinates.
            
        """
        if real:
            # Convert to real coordinates like R does
            # R code: t(cbind(x@coords - 0.5, rep(1, nrow(x@coords))))
            homog_coords = np.column_stack([self.coords - 0.5, 
                                           np.ones(len(self.coords))])
            # Apply transformation
            real_coords = homog_coords @ self.space.trans.T
            return real_coords[:, :3]
        else:
            return self.coords

class ROIVec(ROICoords):
    """Class representing a vector-valued volumetric region of interest (ROI) 
    in a brain image.
    
    Parameters
    ----------
    data : np.ndarray
        Matrix where each column corresponds to a coordinate.
        Shape should be (n_timepoints, n_coords).
    space : NeuroSpace
        NeuroSpace object defining the spatial reference
    coords : np.ndarray
        Matrix with 3 columns representing (i,j,k) coordinates
        
    """
    
    def __init__(self, data: np.ndarray, space: NeuroSpace, coords: np.ndarray):
        super().__init__(coords, space)
        
        data = np.atleast_2d(data)
        if data.shape[1] != len(coords):
            raise ValueError(f"ncol(data) {data.shape[1]} must equal nrow(coords) {len(coords)}")
        
        self.data = data
    
    def __getitem__(self, idx):
        """Extract data columns.
        
        """
        if isinstance(idx, tuple):
            # Handle both row and column indexing
            row_idx, col_idx = idx
            return self.data[row_idx, col_idx]
        else:
            # Single index - return entire column
            return self.data[:, idx]
    
    def __setitem__(self, idx, value):
        """Set data columns."""
        if isinstance(idx, tuple):
            # Handle both row and column indexing
            row_idx, col_idx = idx
            self.data[row_idx, col_idx] = value
        else:
            # Single index - set entire column
            self.data[:, idx] = value

class ROIVolWindow(ROIVol):
    """A spatially windowed volumetric region of interest (ROI) in a brain image,
    derived from a larger parent ROI.
    
    Parameters
    ----------
    data : np.ndarray
        Numeric vector of values  
    space : NeuroSpace
        NeuroSpace object defining the spatial reference
    coords : np.ndarray
        Matrix with 3 columns representing (i,j,k) coordinates
    parent_index : int
        Opaque 1D index of the center voxel in parent space
        using the parent space indexing convention.
    center_index : int
        Zero-based row in ``coords`` of the center voxel.
        
    """
    
    def __init__(self, data: np.ndarray, space: NeuroSpace, coords: np.ndarray,
                 parent_index: int, center_index: int):
        super().__init__(data, space, coords)
        self.parent_index = int(parent_index)
        self.center_index = int(center_index)

    @property
    def parent_grid(self) -> np.ndarray:
        """Parent voxel grid coordinate for this window."""
        return self.space.index_to_grid(np.array([self.parent_index], dtype=int))[0]
    
    def __repr__(self):
        """String representation."""
        return (f"ROIVolWindow(n_voxels={len(self)}, "
                f"parent_index={self.parent_index}, "
                f"center_index={self.center_index})")


def _roi_coords_and_space(roi):
    if isinstance(roi, (ROIVol, ROICoords)):
        return np.ascontiguousarray(roi.coords, dtype=int), roi.space
    raise TypeError(f"roi must be ROIVol or ROICoords, got {type(roi)}")


def _roi_extraction_result(values, coords, space, input_space, method_name):
    from .results import ROIExtractionResult, RoiOpParams, receipt_for

    receipt = receipt_for(
        input_space,
        mask=coords,
        n_voxels=int(coords.shape[0]),
        params=RoiOpParams(method_name=method_name),
    )
    return ROIExtractionResult(
        values=np.ascontiguousarray(values),
        coords=np.ascontiguousarray(coords, dtype=int),
        space=space,
        mask_hash=receipt.mask_hash,
        provenance=receipt,
    )


def values_roi(x, roi, *, return_legacy: bool = False):
    """Extract volume values for all voxels in an ROI.

    Default: returns an :class:`~neuroim.results.ROIExtractionResult` with
    values, coords, space, and a provenance :class:`~neuroim.results.Receipt`.
    Pass ``return_legacy=True`` for the historical bare-ndarray projection;
    this opt-in emits a :class:`DeprecationWarning` and will be removed in
    the next minor.
    """
    _warn_legacy_return(return_legacy)
    coords, roi_space = _roi_coords_and_space(roi)
    input_space = getattr(x, "space", None)
    if input_space is not None:
        from .verify import assert_same_space

        assert_same_space(input_space, roi_space)

    data = x.as_dense().data if hasattr(x, "as_dense") else getattr(x, "data", None)
    if data is None:
        raise TypeError("x must provide volume data or as_dense().data")

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"values_roi expects 3D data, got {data.ndim}D")

    values = np.zeros(coords.shape[0], dtype=data.dtype)
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < data.shape[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < data.shape[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < data.shape[2])
    )
    if np.any(valid):
        vc = coords[valid]
        values[valid] = data[vc[:, 0], vc[:, 1], vc[:, 2]]

    if return_legacy:
        return values

    return _roi_extraction_result(
        values,
        coords,
        roi_space,
        input_space if input_space is not None else roi_space,
        "values_roi",
    )


def series_roi(x, roi, *, return_legacy: bool = False):
    """Extract time series for all voxels in an ROI.

    Default: typed :class:`~neuroim.results.ROIExtractionResult`.  Pass
    ``return_legacy=True`` for the historical time-by-voxel ndarray; this
    opt-in emits a :class:`DeprecationWarning` and will be removed in the
    next minor.
    """
    if hasattr(x, "series_roi"):
        return x.series_roi(roi, return_legacy=return_legacy)
    return values_roi(x, roi, return_legacy=return_legacy)


def _volume_data(bvol: _VolumeInput) -> np.ndarray:
    if not hasattr(bvol, "space") or not hasattr(bvol, "data"):
        raise TypeError("bvol must satisfy NeuroVolLike: .space and 3D .data")
    data = np.asarray(bvol.data)
    if data.ndim != 3 and hasattr(bvol, "as_dense"):
        data = np.asarray(bvol.as_dense().data)
    if data.ndim != 3:
        raise ValueError(f"bvol data must be 3D, got {data.ndim}D")
    return data


def _volume_shape(bvol: _VolumeInput) -> Tuple[int, int, int]:
    return tuple(int(d) for d in _volume_data(bvol).shape)


def _volume_spacing(bvol: _VolumeInput) -> np.ndarray:
    spacing = getattr(bvol, "spacing", None)
    if spacing is None:
        spacing = bvol.space.spacing
    return np.asarray(spacing, dtype=float)


def _volume_value_at(bvol: _VolumeInput, coord: np.ndarray):
    key = tuple(int(c) for c in coord)
    data = np.asarray(getattr(bvol, "data", None))
    if data.ndim == 3:
        return data[key]
    if hasattr(bvol, "__getitem__"):
        value = bvol[key]
        value = np.asarray(value)
        return value.item() if value.shape == () else value.ravel()[0]
    return _volume_data(bvol)[key]

# Factory functions matching R's constructors

def roicoords(coords: np.ndarray) -> ROICoords:
    """Create ROI Coordinates Object.
    
    Creates an ROICoords object from a matrix of coordinates
    representing points in 3D space.
    
    Parameters
    ----------
    coords : np.ndarray
        Matrix with 3 columns representing (i, j, k) coordinates
        
    Returns
    -------
    ROICoords
        ROICoords object
        
    Examples
    --------
    >>> coords = np.array([[1,2,3], [4,5,6]])
    >>> roi_coords = ROICoords(coords)
        
    """
    coords = np.atleast_2d(coords)
    if coords.shape[1] != 3:
        raise ValueError("coords must be a matrix with 3 columns (i,j,k)")
    
    return ROICoords(coords)

def roivol(space: NeuroSpace, coords: np.ndarray, data: np.ndarray) -> ROIVol:
    """Create ROI Volume Object.
    
    Creates an ROIVol object representing a set of values
    at specific 3D coordinates within a spatial reference system.
    
    Parameters
    ----------
    space : NeuroSpace
        NeuroSpace object defining the spatial reference
    coords : np.ndarray
        Matrix with 3 columns representing (i,j,k) coordinates
    data : np.ndarray
        Numeric vector of values corresponding to each coordinate
        
    Returns
    -------
    ROIVol
        ROIVol object
        
    Examples
    --------
    >>> from neuroim import NeuroSpace
    >>> space = NeuroSpace([64,64,64])
    >>> coords = np.array([[1,2,3], [4,5,6]])
    >>> data = np.array([1.5, 2.5])
    >>> roi_vol = ROIVol(space, coords, data)

    """
    coords = np.atleast_2d(coords)
    if coords.shape[1] != 3:
        raise ValueError("coords must be a matrix with 3 columns (i,j,k)")

    data = np.asarray(data)
    if len(data) != len(coords):
        raise ValueError("length of data must match number of coordinates")

    return ROIVol(data, space, coords)


# ROI construction functions


def square_roi(
    bvol: _VolumeInput,
    centroid: Union[List, np.ndarray],
    surround: int,
    fill: Optional[float] = None,
    nonzero: bool = False,
    fixdim: int = 3,
) -> ROIVol:
    """Construct a square ROI (2D) from a brain volume.

    Parameters
    ----------
    bvol : NeuroVol
        The brain volume to extract ROI from
    centroid : array-like
        3D coordinates (i,j,k) of the ROI center (0-based)
    surround : int
        Number of voxels to include around center (radius)
    fill : float, optional
        Value to assign to ROI voxels. If None, use original values.
    nonzero : bool
        If True, keep only nonzero voxels
    fixdim : int
        Fixed dimension (0, 1, or 2).
        
    Returns
    -------
    ROIVol
        Square ROI object
        
    """
    data_source = _volume_data(bvol)
    
    centroid = np.asarray(centroid).astype(int)
    if len(centroid) != 3:
        raise ValueError("square_roi: centroid must have length of 3 (i,j,k coordinates)")
    
    # Convert fixdim from Python 0-based if needed
    if fixdim not in [0, 1, 2]:
        raise ValueError("fixdim must be 0, 1, or 2")
    
    # Calculate bounds
    dims = list(data_source.shape)
    lower = np.maximum(0, centroid - surround)
    upper = np.minimum(dims, centroid + surround + 1)
    
    # Generate coordinates for the square in 2D
    if fixdim == 0:
        # Fix i dimension
        j_coords, k_coords = np.meshgrid(range(lower[1], upper[1]),
                                         range(lower[2], upper[2]),
                                         indexing='ij')
        i_coords = np.full_like(j_coords, centroid[0])
    elif fixdim == 1:
        # Fix j dimension  
        i_coords, k_coords = np.meshgrid(range(lower[0], upper[0]),
                                         range(lower[2], upper[2]),
                                         indexing='ij')
        j_coords = np.full_like(i_coords, centroid[1])
    else:  # fixdim == 2
        # Fix k dimension
        i_coords, j_coords = np.meshgrid(range(lower[0], upper[0]),
                                         range(lower[1], upper[1]),
                                         indexing='ij')
        k_coords = np.full_like(i_coords, centroid[2])
    
    # Create coordinate matrix
    coords = np.column_stack([i_coords.ravel(), 
                              j_coords.ravel(), 
                              k_coords.ravel()])
    
    # Extract data values
    if fill is not None:
        data = np.full(len(coords), fill)
    else:
        # Get values from volume
        data = np.array([_volume_value_at(bvol, coords[i])
                         for i in range(len(coords))])
    
    # Filter nonzero if requested
    if nonzero:
        mask = data != 0
        coords = coords[mask]
        data = data[mask]
    
    return ROIVol(data, bvol.space, coords)

def cuboid_roi(bvol: _VolumeInput, centroid: Union[List, np.ndarray], surround: Union[int, List, np.ndarray],
               fill: Optional[float] = None, nonzero: bool = False) -> ROIVol:
    """Construct a cuboid (3D box) ROI from a brain volume.
    
    Parameters
    ----------
    bvol : NeuroVol
        The brain volume to extract ROI from
    centroid : array-like
        3D coordinates (i,j,k) of the ROI center (0-based)
    surround : int or array-like
        Number of voxels around center. If int, same for all dims.
        If array, specify for each dimension.
    fill : float, optional
        Value to assign to ROI voxels. If None, use original values.
    nonzero : bool
        If True, keep only nonzero voxels
        
    Returns
    -------
    ROIVol
        Cuboid ROI object
        
    """
    data_source = _volume_data(bvol)
    
    centroid = np.asarray(centroid).astype(int)
    if len(centroid) != 3:
        raise ValueError("cuboid_roi: centroid must have length of 3 (i,j,k coordinates)")
    
    # Handle surround parameter
    if np.isscalar(surround):
        surround = np.array([surround, surround, surround])
    else:
        surround = np.asarray(surround)
        if len(surround) != 3:
            raise ValueError("surround must be scalar or length 3")
    
    # Calculate bounds
    dims = np.array(data_source.shape)
    lower = np.maximum(0, centroid - surround)
    upper = np.minimum(dims, centroid + surround + 1)
    
    # Generate coordinates
    i_coords, j_coords, k_coords = np.meshgrid(
        range(lower[0], upper[0]),
        range(lower[1], upper[1]),
        range(lower[2], upper[2]),
        indexing='ij'
    )
    
    coords = np.column_stack([i_coords.ravel(), 
                              j_coords.ravel(), 
                              k_coords.ravel()])
    
    # Extract data values
    if fill is not None:
        data = np.full(len(coords), fill)
    else:
        # Get values from volume
        data = np.array([_volume_value_at(bvol, coords[i])
                         for i in range(len(coords))])
    
    # Filter nonzero if requested
    if nonzero:
        mask = data != 0
        coords = coords[mask]
        data = data[mask]
    
    return ROIVol(data, bvol.space, coords)

def spherical_roi(bvol: _VolumeInput, centroid: Union[List, np.ndarray], radius: float,
                  fill: Optional[float] = None, nonzero: bool = False) -> ROIVol:
    """Construct a spherical ROI from a brain volume.
    
    Parameters
    ----------
    bvol : NeuroVol
        The brain volume to extract ROI from
    centroid : array-like
        3D coordinates (i,j,k) of the ROI center (0-based)
    radius : float
        Radius of sphere in voxel units
    fill : float, optional
        Value to assign to ROI voxels. If None, use original values.
    nonzero : bool
        If True, keep only nonzero voxels
        
    Returns
    -------
    ROIVol
        Spherical ROI object
        
    """
    data_source = _volume_data(bvol)
    
    centroid = np.asarray(centroid)
    if len(centroid) != 3:
        raise ValueError("centroid must have length 3")

    if radius <= 0:
        raise ValueError("radius must be positive")
    
    # Calculate bounding box for sphere in physical space using spacing
    spacing = _volume_spacing(bvol)
    shape = np.asarray(data_source.shape)
    if radius < float(np.min(spacing)):
        raise ValueError("radius is too small; must be at least one voxel dimension")
    if np.any(centroid < 0) or np.any(centroid >= shape):
        raise ValueError("centroid coordinates must be within volume bounds")
    voxel_radius = np.ceil(radius / spacing).astype(int)
    lower = np.maximum(0, np.floor(centroid - voxel_radius).astype(int))
    upper = np.minimum(shape, np.ceil(centroid + voxel_radius + 1).astype(int))
    
    # Generate candidate coordinates
    i_coords, j_coords, k_coords = np.meshgrid(
        range(lower[0], upper[0]),
        range(lower[1], upper[1]),
        range(lower[2], upper[2]),
        indexing='ij'
    )
    
    coords = np.column_stack([i_coords.ravel(), 
                              j_coords.ravel(), 
                              k_coords.ravel()])
    
    # Calculate distances from centroid in physical units (mm)
    distances = np.sqrt(np.sum(((coords - centroid) * spacing)**2, axis=1))
    
    # Keep only points within radius
    mask = distances <= radius
    coords = coords[mask]
    
    source_data = None
    if nonzero:
        source_data = np.array([_volume_value_at(bvol, coords[i])
                                for i in range(len(coords))])

    # Extract data values
    if fill is not None:
        data = np.full(len(coords), fill)
    else:
        # Get values from volume
        data = source_data
        if data is None:
            data = np.array([_volume_value_at(bvol, coords[i])
                             for i in range(len(coords))])
    
    # Filter nonzero if requested
    if nonzero:
        nonzero_mask = source_data != 0
        coords = coords[nonzero_mask]
        data = data[nonzero_mask]
    
    return ROIVol(data, bvol.space, coords)

def spherical_roi_set(bvol: _VolumeInput, centroids: np.ndarray, radius: float,
                      fill: Optional[Union[float, List[float]]] = None, 
                      nonzero: bool = False) -> List[ROIVol]:
    """Create multiple spherical ROIs efficiently.
    
    Parameters
    ----------
    bvol : NeuroVol
        The brain volume to extract ROIs from
    centroids : np.ndarray
        Matrix where each row is a 3D centroid coordinate (0-based)
    radius : float
        Radius of spheres in voxel units
    fill : float or list, optional
        Value(s) to assign to ROI voxels. If list, must match number of centroids.
    nonzero : bool
        If True, keep only nonzero voxels
        
    Returns
    -------
    list of ROIVol
        List of spherical ROI objects
        
    """
    centroids = np.atleast_2d(centroids)
    if centroids.shape[1] != 3:
        raise ValueError("centroids must have 3 columns")
    
    # Handle fill values
    if fill is not None:
        if np.isscalar(fill):
            fill_values = [fill] * len(centroids)
        else:
            fill_values = list(fill)
            if len(fill_values) != len(centroids):
                raise ValueError("fill must be scalar or match number of centroids")
    else:
        fill_values = [None] * len(centroids)
    
    # Create ROIs
    result_list = []
    for i, (centroid, fill_val) in enumerate(zip(centroids, fill_values)):
        roi = spherical_roi(bvol, centroid, radius, fill=fill_val, nonzero=nonzero)
        result_list.append(roi)
    
    return result_list

def cube_roi(bvol: _VolumeInput, centroid: Union[List, np.ndarray], width: int,
             fill: Optional[float] = None, nonzero: bool = False) -> ROIVol:
    """Construct a cubic ROI from a brain volume.

    Convenience wrapper around :func:`cuboid_roi` with equal surround
    in all three dimensions.

    Parameters
    ----------
    bvol : NeuroVol
        The brain volume to extract ROI from
    centroid : array-like
        3D coordinates (i,j,k) of the ROI center (0-based)
    width : int
        Half-width of the cube in voxels (surround distance)
    fill : float, optional
        Value to assign to ROI voxels. If None, use original values.
    nonzero : bool
        If True, keep only nonzero voxels

    Returns
    -------
    ROIVol
        Cubic ROI object
    """
    return cuboid_roi(bvol, centroid, surround=width, fill=fill, nonzero=nonzero)

def ellipsoid_roi(bvol: _VolumeInput, centroid: Union[List, np.ndarray],
                  radii: Union[List, np.ndarray],
                  fill: Optional[float] = None,
                  nonzero: bool = False) -> ROIVol:
    """Construct an ellipsoid ROI from a brain volume.

    Parameters
    ----------
    bvol : NeuroVol
        The brain volume to extract ROI from
    centroid : array-like
        3D coordinates (i,j,k) of the ROI center (0-based)
    radii : array-like
        Radii along each axis (i, j, k) in voxel units
    fill : float, optional
        Value to assign to ROI voxels. If None, use original values.
    nonzero : bool
        If True, keep only nonzero voxels

    Returns
    -------
    ROIVol
        Ellipsoid ROI object
    """
    data_source = _volume_data(bvol)

    centroid = np.asarray(centroid, dtype=float)
    if len(centroid) != 3:
        raise ValueError("centroid must have length 3")

    radii = np.asarray(radii, dtype=float)
    if len(radii) != 3:
        raise ValueError("radii must have length 3")
    if np.any(radii <= 0):
        raise ValueError("radii must be positive")

    # Bounding box based on the largest radius per axis
    lower = np.maximum(0, np.floor(centroid - radii).astype(int))
    upper = np.minimum(data_source.shape, np.ceil(centroid + radii + 1).astype(int))

    # Generate candidate coordinates
    i_coords, j_coords, k_coords = np.meshgrid(
        range(lower[0], upper[0]),
        range(lower[1], upper[1]),
        range(lower[2], upper[2]),
        indexing='ij'
    )

    coords = np.column_stack([i_coords.ravel(),
                               j_coords.ravel(),
                               k_coords.ravel()])

    # Ellipsoid equation: sum((x - c)^2 / r^2) <= 1
    scaled = (coords - centroid) / radii
    distances_sq = np.sum(scaled ** 2, axis=1)
    mask = distances_sq <= 1.0
    coords = coords[mask]

    # Extract data values
    if fill is not None:
        data = np.full(len(coords), fill)
    else:
        data = np.array([_volume_value_at(bvol, coords[i])
                         for i in range(len(coords))])

    if nonzero:
        nz = data != 0
        coords = coords[nz]
        data = data[nz]

    return ROIVol(data, bvol.space, coords)

def patch_set(bvol: _VolumeInput, centroids: np.ndarray, radius: float,
              shape: str = "sphere",
              fill: Optional[Union[float, List[float]]] = None,
              nonzero: bool = False) -> List[ROIVol]:
    """Create a set of ROI patches with a specified shape.

    Parameters
    ----------
    bvol : NeuroVol
        The brain volume to extract patches from
    centroids : np.ndarray
        Matrix where each row is a 3D centroid coordinate (0-based)
    radius : float
        Radius (or half-width) in voxel units
    shape : str
        Shape of each patch: ``"sphere"`` (default) or ``"cube"``
    fill : float or list, optional
        Value(s) to assign. If list, must match number of centroids.
    nonzero : bool
        If True, keep only nonzero voxels

    Returns
    -------
    list of ROIVol
        List of ROI patch objects
    """
    centroids = np.atleast_2d(centroids)
    if centroids.shape[1] != 3:
        raise ValueError("centroids must have 3 columns")

    shape = shape.lower()
    if shape not in ("sphere", "cube"):
        raise ValueError(f"shape must be 'sphere' or 'cube', got '{shape}'")

    # Normalise fill values
    if fill is not None:
        if np.isscalar(fill):
            fill_values = [fill] * len(centroids)
        else:
            fill_values = list(fill)
            if len(fill_values) != len(centroids):
                raise ValueError("fill must be scalar or match number of centroids")
    else:
        fill_values = [None] * len(centroids)

    result = []
    for centroid, fill_val in zip(centroids, fill_values):
        if shape == "sphere":
            roi = spherical_roi(bvol, centroid, radius, fill=fill_val, nonzero=nonzero)
        else:
            roi = cube_roi(bvol, centroid, int(radius), fill=fill_val, nonzero=nonzero)
        result.append(roi)

    return result

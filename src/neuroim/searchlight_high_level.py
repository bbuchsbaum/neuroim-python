"""High-level searchlight functions that apply methods.

Includes the standard searchlight as well as resampled and
cluster-based searchlight variants.
"""

import warnings

import numpy as np
from typing import Union, Optional, Callable, Dict
from joblib import Parallel, delayed
from .neuro_vol import NeuroVol, LogicalNeuroVol, DenseNeuroVol
from .neuro_vec import NeuroVec
from .results import SearchlightResult, make_receipt
from .searchlight import searchlight_iterator
from .clustered_neuro_vol import ClusteredNeuroVol
from .typing import MaskLike, NeuroVecLike, NeuroVolLike


_LEGACY_DEPRECATION_MSG = (
    "return_legacy=True is deprecated and will be removed in the next minor "
    "release; consume the typed result object instead (.values, "
    ".map_to_volume(), .provenance)."
)


def _warn_legacy_return(return_legacy: bool) -> None:
    if return_legacy:
        warnings.warn(_LEGACY_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)


def _coords_from_parent_index(space, parent_index: int) -> tuple[int, ...]:
    """Return coordinate tuple for an opaque parent index."""
    return tuple(space.index_to_grid(np.array([parent_index], dtype=int))[0].astype(int))


def searchlight(mask: MaskLike,
                radius: float,
                method,
                combiner: str = "mean",
                data: Optional[Union[NeuroVolLike, NeuroVecLike]] = None,
                eager: bool = False,
                nonzero: bool = False,
                cores: int = 0,
                return_legacy: bool = False):
    """Apply a searchlight analysis with a given method function.
    
    This function performs searchlight analysis by applying a method function
    to each searchlight sphere and combining the results.
    
    Parameters
    ----------
    mask : NeuroVol or LogicalNeuroVol
        A NeuroVol object representing the brain mask
    radius : float
        The radius (in mm) of the spherical searchlight
    method : callable
        Function to apply to data within each searchlight. Should accept
        a numpy array and return a scalar value.
    combiner : str, optional
        How to combine results ("mean", "median", etc.). Default is "mean".
    data : NeuroVol or NeuroVec, optional
        Data to extract from each searchlight. If not provided, uses the mask.
    eager : bool, optional
        Whether to eagerly compute the searchlight ROIs. Default is False.
    nonzero : bool, optional
        Whether to include only coordinates with nonzero values in the
        supplied mask. Default is False
    cores : int, optional
        Number of cores to use for parallel computation. Default is 0,
        which uses a single core.
        
    Returns
    -------
    SearchlightResult or NeuroVol
        Default: a :class:`~neuroim.results.SearchlightResult` carrying
        values, centers, space, radius, shape, and a
        :class:`~neuroim.results.Receipt`.  Pass ``return_legacy=True`` for
        the historical :class:`~neuroim.neuro_vol.DenseNeuroVol` projection;
        this opt-in emits a ``DeprecationWarning`` and will be removed in
        the next minor.
    """
    _warn_legacy_return(return_legacy)

    # Use mask as data if not provided
    if data is None:
        data = mask
        
    # Convert to LogicalNeuroVol if needed
    if not isinstance(mask, LogicalNeuroVol):
        mask = mask.as_logical()
    
    # Create result volume with same space as mask
    result_data = np.full(mask.data.shape, np.nan, dtype=np.float64)
    
    # Get searchlight iterator
    searchlights = searchlight_iterator(mask, radius, eager=eager, nonzero=nonzero, cores=cores)
    
    # Define processing function for a single searchlight
    def process_searchlight(sl):
        """Process a single searchlight and return (center_idx, result)."""
        # Extract data from the searchlight region
        if hasattr(data, 'series'):
            # For NeuroVec, extract time series
            sl_data = data.series(sl.coords)
        else:
            # For NeuroVol, extract values
            sl_indices = mask.space.grid_to_index(sl.coords)
            sl_data = data.data.ravel(order="F")[sl_indices]
        
        # Apply method
        if sl_data.size > 0:
            result = method(sl_data)
        else:
            result = np.nan
            
        return sl.parent_index, result
    
    if cores > 1 and eager:
        # Parallel processing when searchlights are eagerly computed
        results = Parallel(n_jobs=cores, prefer="threads")(
            delayed(process_searchlight)(sl) for sl in searchlights
        )
        # Store results
        for center_idx, result in results:
            result_data[_coords_from_parent_index(mask.space, center_idx)] = result
    else:
        # Sequential processing
        for sl in searchlights:
            center_idx, result = process_searchlight(sl)
            result_data[_coords_from_parent_index(mask.space, center_idx)] = result
    
    legacy_vol = DenseNeuroVol(result_data, mask.space)
    if return_legacy:
        return legacy_vol

    finite_mask = np.isfinite(result_data)
    if finite_mask.any():
        center_indices = np.argwhere(finite_mask)
        values = result_data[finite_mask]
    else:
        center_indices = np.zeros((0, 3), dtype=int)
        values = np.zeros(0, dtype=np.float64)

    receipt = make_receipt(
        input_space=getattr(data, "space", mask.space),
        mask_data=mask.data,
        radius=float(radius),
        n_voxels=int(center_indices.shape[0]),
        method_name=getattr(method, "__name__", repr(method)),
        seed=None,
        source_affine=mask.space.trans,
    )
    return SearchlightResult(
        values=np.ascontiguousarray(values),
        centers=center_indices,
        space=mask.space,
        radius=float(radius),
        shape="sphere",
        provenance=receipt,
        method_name=getattr(method, "__name__", repr(method)),
    )


def resampled_searchlight(
    vec: NeuroVecLike,
    radius: float,
    fun: Callable,
    n_resamples: int = 100,
    mask: Optional[MaskLike] = None,
    seed: Optional[int] = None,
) -> DenseNeuroVol:
    """Searchlight with bootstrap resampling of time points.

    For each searchlight sphere, the time-point axis is bootstrap-resampled
    ``n_resamples`` times.  *fun* is applied to the resampled data each
    time, and the results are averaged across resamples.

    Parameters
    ----------
    vec : NeuroVec
        A 4-D neuroimaging vector (e.g. ``DenseNeuroVec``).
    radius : float
        Searchlight sphere radius in mm.
    fun : callable
        Function applied to the data within each searchlight.  Receives a
        2-D array of shape ``(n_timepoints, n_voxels)`` and should return a
        scalar.
    n_resamples : int
        Number of bootstrap resamples per searchlight centre.
    mask : NeuroVol or LogicalNeuroVol, optional
        Brain mask.  If *None*, a mask is derived from the first volume of
        *vec* (all non-zero voxels).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DenseNeuroVol
        Volume of averaged statistics, one value per searchlight centre.
    """
    rng = np.random.default_rng(seed)
    n_time = vec.shape[3]

    # Derive mask
    if mask is None:
        first_vol = vec[..., 0]
        if isinstance(first_vol, NeuroVol):
            mask = first_vol.as_logical()
        else:
            from .neuro_space import NeuroSpace as _NS
            mask_space = _NS(vec.shape[:3], spacing=vec.spacing[:3], origin=vec.origin[:3])
            mask = LogicalNeuroVol(first_vol != 0, mask_space)
    elif not isinstance(mask, LogicalNeuroVol):
        mask = mask.as_logical()

    searchlights = searchlight_iterator(mask, radius, eager=False, nonzero=True)

    result_data = np.full(mask.data.shape, np.nan, dtype=np.float64)

    for sl in searchlights:
        coords = sl.coords
        accum = 0.0
        for _ in range(n_resamples):
            idx = rng.choice(n_time, size=n_time, replace=True)
            # Extract resampled time series matrix
            ts = vec.series(coords)  # shape (n_time, n_voxels)
            resampled = ts[idx, :] if ts.ndim == 2 else ts[idx]
            accum += fun(resampled)
        center_idx = _coords_from_parent_index(mask.space, sl.parent_index)
        result_data[center_idx] = accum / n_resamples

    return DenseNeuroVol(result_data, mask.space)


def cluster_searchlight_series(
    vec: NeuroVecLike,
    cvol: ClusteredNeuroVol,
    fun: Callable,
    mask: Optional[MaskLike] = None,
) -> Dict[int, object]:
    """Searchlight using cluster neighbourhoods instead of spheres.

    For each cluster defined in *cvol*, the time series of all voxels in
    the cluster are extracted from *vec*, *fun* is applied, and the result
    is stored keyed by cluster id.

    Parameters
    ----------
    vec : NeuroVec
        A 4-D neuroimaging vector.
    cvol : ClusteredNeuroVol
        Clustered volume whose clusters define neighbourhoods.
    fun : callable
        Function applied to each cluster's data.  Receives a 2-D array of
        shape ``(n_timepoints, n_voxels_in_cluster)`` and may return an
        arbitrary result.
    mask : NeuroVol or LogicalNeuroVol, optional
        Additional mask to restrict computation.  Currently unused but
        accepted for API consistency with other searchlight functions.

    Returns
    -------
    dict
        Mapping from integer cluster id to the value returned by *fun*.
    """
    results: Dict[int, object] = {}

    for cluster_id, indices_in_mask in cvol.cluster_map.items():
        # indices_in_mask are positions within the mask's flat array.
        # Convert to 3-D grid coordinates so we can call vec.series().
        grid_coords = cvol.space.index_to_grid(indices_in_mask)
        ts = vec.series(grid_coords)  # (n_time, n_voxels)
        results[cluster_id] = fun(ts)

    return results

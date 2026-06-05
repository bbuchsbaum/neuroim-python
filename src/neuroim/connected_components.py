"""Connected components functionality for neuroimaging data.

This module provides functions for identifying and labeling connected
components in 3D neuroimaging data, supporting various connectivity patterns.

Connected-component utilities for labeled 3D neuroimaging data.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import label, maximum_filter
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from dataclasses import dataclass

if TYPE_CHECKING:  # pragma: no cover - type-checkers only
    import pandas as pd

# pandas is an OPTIONAL dependency: only the cluster-table projection
# uses it and it is not declared in pyproject. A top-level `import
# pandas` here made `import neuroim` crash on a clean install. Import it
# lazily at the one runtime use site instead, mirroring results.py
# SearchlightResult.to_dataframe.

from .neuro_vol import NeuroVol, LogicalNeuroVol, DenseNeuroVol
from .neuro_space import NeuroSpace
from .clustered_neuro_vol import ClusteredNeuroVol
from .results import OpParams, Receipt, receipt_for


@dataclass
class ConnCompResult:
    """Result of connected components analysis.

    The ``cluster_table`` columns are documented at ``_compute_cluster_table``
    -- the canonical schema is ``index, x, y, z, N, Area, value`` for the
    centroid-flavoured stats, followed by ``peak_x_mm, peak_y_mm,
    peak_z_mm, peak_value`` for the per-cluster local maximum (signed value
    retained from the original input, so two-tailed maps keep the sign).
    """

    index: ClusteredNeuroVol
    size: NeuroVol
    voxels: List[np.ndarray]
    cluster_table: Optional[pd.DataFrame] = None
    local_maxima: Optional[np.ndarray] = None
    provenance: Optional[Receipt] = None

def conn_comp(
    x: NeuroVol,
    threshold: float = 0,
    cluster_table: bool = True,
    local_maxima: bool = True,
    local_maxima_dist: float = 15,
    connect: str = "26-connect",
    *,
    mask: Optional[LogicalNeuroVol] = None,
    two_tailed: bool = False,
) -> ConnCompResult:
    """Find connected components in an image.

    Identifies and labels spatially connected regions in neuroimaging data,
    supporting both binary masks and thresholded volumes.

    Parameters
    ----------
    x : NeuroVol
        The image object.
    threshold : float, optional
        Threshold defining lower intensity bound for the binary mask.
        Default is 0.  See ``two_tailed`` for two-sided thresholding.
    cluster_table : bool, optional
        Whether to return the cluster statistics table.  Default True.
    local_maxima : bool, optional
        Whether to compute local maxima within clusters.  Default True.
    local_maxima_dist : float, optional
        Minimum distance between local maxima in mm.  Default 15.
    connect : str, optional
        Connectivity pattern: "26-connect", "18-connect", or "6-connect".
        Default "26-connect".
    mask : LogicalNeuroVol, optional (keyword only)
        Restrict clustering to voxels where ``mask`` is True.  When
        supplied, ``verify.assert_same_space(x, mask)`` is invoked first
        so a foreign-affine mask raises before any clustering.
    two_tailed : bool, optional (keyword only)
        When True, threshold the absolute value (``|x.data| > threshold``)
        so both positive- and negative-tail clusters survive.  ``peak_value``
        in the cluster table is read back from the original signed data so
        the sign of each cluster is retained.  Default False (one-tailed
        ``x.data > threshold``, the legacy behaviour).

    Returns
    -------
    ConnCompResult
        ``index`` (``ClusteredNeuroVol``), ``size`` (``NeuroVol``),
        ``voxels``, ``cluster_table``, ``local_maxima``, and
        ``provenance`` (populated with ``method_name="conn_comp"`` and the
        input space + mask hashes).
    """
    if mask is not None:
        from .verify import assert_same_space

        assert_same_space(x, mask)

    # Apply threshold to create binary mask.  LogicalNeuroVol input is
    # already a mask -- no threshold or two-tailed semantics apply.
    if isinstance(x, LogicalNeuroVol):
        mask_data = np.asarray(x.data, dtype=bool)
    elif two_tailed:
        mask_data = np.abs(np.asarray(x.data)) > threshold
    else:
        mask_data = np.asarray(x.data) > threshold

    # Optional brain-mask restriction.
    if mask is not None:
        mask_data = mask_data & np.asarray(mask.data, dtype=bool)

    # Get connectivity structure
    structure = _get_structure(connect)

    # Find connected components
    labeled_array, num_features = label(mask_data, structure=structure)

    # Create size array
    size_array = np.zeros_like(labeled_array, dtype=int)
    voxels_list = []

    # Compute sizes and collect voxel coordinates
    for i in range(1, num_features + 1):
        cluster_mask = labeled_array == i
        size = int(np.sum(cluster_mask))
        size_array[cluster_mask] = size

        # Get voxel coordinates (grid indices)
        coords = np.column_stack(np.where(cluster_mask))
        voxels_list.append(coords)

    # Create ClusteredNeuroVol for index
    mask_vol = LogicalNeuroVol(mask_data, x.space)
    cluster_labels = labeled_array[mask_data]
    index_vol = ClusteredNeuroVol(mask_vol, cluster_labels)

    # Create NeuroVol for sizes
    size_vol = DenseNeuroVol(size_array.astype(float), x.space)

    # Receipt -- populated regardless of whether clustering produced any
    # features, so a downstream consumer can still inspect the threshold
    # and mask that produced an empty result.  Goes through the structural
    # ``receipt_for`` path so the call site cannot silently omit fields
    # an op should always record (the bd-01KRKRZPDC6V5CZF7SH0C9KEDD
    # contract).
    mask_payload = (
        np.asarray(mask.data, dtype=bool) if mask is not None else None
    )
    receipt = receipt_for(
        x,
        mask=mask_payload,
        n_voxels=int(np.sum(mask_data)),
        params=OpParams(method_name="conn_comp", radius=float(threshold)),
    )

    result = ConnCompResult(
        index=index_vol,
        size=size_vol,
        voxels=voxels_list,
        provenance=receipt,
    )

    # Compute cluster table if requested.  The table projection is the only
    # pandas-dependent step; pandas is an OPTIONAL dependency, so on a stock
    # install it may be absent.  Degrade gracefully (warn + leave the table
    # ``None``) rather than crashing the whole call — the cluster index,
    # sizes, voxel lists and provenance are all still returned.
    if cluster_table and num_features > 0:
        try:
            result.cluster_table = _compute_cluster_table(
                x, labeled_array, voxels_list, x.space
            )
        except ModuleNotFoundError as exc:
            if exc.name != "pandas":
                raise
            warnings.warn(
                "conn_comp(cluster_table=True) needs pandas to build the "
                "cluster table, but pandas is not installed. Returning "
                "cluster_table=None. Install pandas (`pip install pandas`) "
                "or pass cluster_table=False to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Find local maxima if requested
    if local_maxima and num_features > 0:
        result.local_maxima = _find_local_maxima(
            x, labeled_array, local_maxima_dist, x.space
        )

    return result

def conn_comp_3D(
    mask: Union[np.ndarray, LogicalNeuroVol], connect: str = "26-connect"
) -> Dict[str, np.ndarray]:
    """Extract connected components from a 3D binary mask.

    Identifies and labels connected components in a 3D binary mask using
    a two-pass algorithm. The function supports different connectivity
    constraints and returns both component indices and their sizes.

    Parameters
    ----------
    mask : np.ndarray or LogicalNeuroVol
        A 3D logical array or LogicalNeuroVol representing the binary mask
    connect : str, optional
        Connectivity constraint: "26-connect", "18-connect", or "6-connect".
        Default is "26-connect".

    Returns
    -------
    dict
        Dictionary with two components:
        - 'index': 3D array of integers with cluster indices (0 = background)
        - 'size': 3D array of integers with cluster sizes (0 = background)

    """
    # Extract mask data if LogicalNeuroVol
    if isinstance(mask, LogicalNeuroVol):
        mask_data = mask.data
    else:
        mask_data = mask

    if mask_data.ndim != 3:
        raise ValueError("Mask must be 3D")

    if mask_data.dtype != bool:
        raise ValueError("Mask must be boolean/logical")

    # Get connectivity structure
    structure = _get_structure(connect)

    # Find connected components
    labeled_array, num_features = label(mask_data, structure=structure)

    # Create size array
    size_array = np.zeros_like(labeled_array, dtype=int)

    # Compute sizes for each component
    for i in range(1, num_features + 1):
        component_mask = labeled_array == i
        component_size = np.sum(component_mask)
        size_array[component_mask] = component_size

    return {"index": labeled_array, "size": size_array}

def _get_structure(connectivity: str) -> np.ndarray:
    """Get the structuring element for the specified connectivity.

    Parameters
    ----------
    connectivity : str
        Connectivity pattern: "6-connect", "18-connect", or "26-connect"

    Returns
    -------
    np.ndarray
        3x3x3 boolean array defining the connectivity structure
    """
    if connectivity == "6-connect":
        # Only face-adjacent voxels
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, 1] = True
        structure[0, 1, 1] = True
        structure[2, 1, 1] = True
        structure[1, 0, 1] = True
        structure[1, 2, 1] = True
        structure[1, 1, 0] = True
        structure[1, 1, 2] = True
        return structure

    elif connectivity == "18-connect":
        # Face and edge-adjacent voxels
        structure = np.zeros((3, 3, 3), dtype=bool)
        # Add center
        structure[1, 1, 1] = True
        # Add faces (6)
        structure[0, 1, 1] = True
        structure[2, 1, 1] = True
        structure[1, 0, 1] = True
        structure[1, 2, 1] = True
        structure[1, 1, 0] = True
        structure[1, 1, 2] = True
        # Add edges (12)
        structure[0, 0, 1] = True
        structure[0, 2, 1] = True
        structure[2, 0, 1] = True
        structure[2, 2, 1] = True
        structure[0, 1, 0] = True
        structure[0, 1, 2] = True
        structure[2, 1, 0] = True
        structure[2, 1, 2] = True
        structure[1, 0, 0] = True
        structure[1, 0, 2] = True
        structure[1, 2, 0] = True
        structure[1, 2, 2] = True
        return structure

    elif connectivity == "26-connect":
        # All neighbors in 3x3x3 cube
        return np.ones((3, 3, 3), dtype=bool)

    else:
        raise ValueError(
            f"Invalid connectivity: {connectivity}. "
            "Choose from '6-connect', '18-connect', or '26-connect'."
        )

def _compute_cluster_table(
    vol: NeuroVol,
    labeled_array: np.ndarray,
    voxels_list: List[np.ndarray],
    space: NeuroSpace,
) -> pd.DataFrame:
    """Compute cluster statistics table.

    Parameters
    ----------
    vol : NeuroVol
        Original volume
    labeled_array : np.ndarray
        Labeled cluster array
    voxels_list : List[np.ndarray]
        List of voxel coordinates for each cluster
    space : NeuroSpace
        Spatial reference

    Returns
    -------
    pd.DataFrame
        Table with columns: index, x, y, z, N, Area, value
    """
    import pandas as pd  # optional dep; lazy so `import neuroim` never needs it

    rows = []
    data = np.asarray(vol.data)

    for i, voxels in enumerate(voxels_list, start=1):
        if len(voxels) == 0:
            continue

        # Center of mass in grid coordinates, then to world mm.
        center_grid = np.mean(voxels, axis=0)
        center_world = space.grid_to_coord(center_grid.reshape(1, -1))[0]

        n_voxels = len(voxels)
        voxel_volume = float(np.prod(space.spacing[:3]))
        area = n_voxels * voxel_volume

        # Per-cluster mean and per-cluster peak (largest |value|, retain
        # sign from original data so two-tailed maps don't lose direction).
        cluster_mask = labeled_array == i
        cluster_values = data[cluster_mask]
        mean_value = float(np.mean(cluster_values))
        peak_local_idx = int(np.argmax(np.abs(cluster_values)))
        peak_value = float(cluster_values[peak_local_idx])
        peak_ijk = voxels[peak_local_idx]
        peak_world = space.grid_to_coord(
            np.asarray(peak_ijk, dtype=float).reshape(1, -1)
        )[0]

        rows.append(
            {
                "index": i,
                "x": float(center_world[0]),
                "y": float(center_world[1]),
                "z": float(center_world[2]),
                "N": n_voxels,
                "Area": area,
                "value": mean_value,
                "peak_x_mm": float(peak_world[0]),
                "peak_y_mm": float(peak_world[1]),
                "peak_z_mm": float(peak_world[2]),
                "peak_value": peak_value,
            }
        )

    return pd.DataFrame(rows)

def _find_local_maxima(
    vol: NeuroVol, labeled_array: np.ndarray, min_distance: float, space: NeuroSpace
) -> np.ndarray:
    """Find local maxima within clusters.

    Parameters
    ----------
    vol : NeuroVol
        Original volume
    labeled_array : np.ndarray
        Labeled cluster array
    min_distance : float
        Minimum distance between maxima in mm
    space : NeuroSpace
        Spatial reference

    Returns
    -------
    np.ndarray
        Array with columns: index, x, y, z, value
    """
    # Convert minimum distance to voxels
    min_dist_voxels = min_distance / np.mean(space.spacing)

    maxima_list = []

    # Process each cluster
    num_clusters = np.max(labeled_array)
    for i in range(1, num_clusters + 1):
        cluster_mask = labeled_array == i

        # Extract cluster data
        cluster_data = np.where(cluster_mask, vol.data, -np.inf)

        # Find local maxima using maximum filter
        local_max = maximum_filter(cluster_data, size=3)
        maxima_mask = (cluster_data == local_max) & cluster_mask

        # Get maxima coordinates
        maxima_coords = np.column_stack(np.where(maxima_mask))

        if len(maxima_coords) == 0:
            continue

        # Get values at maxima
        maxima_values = vol.data[maxima_mask]

        # Sort by value (descending)
        # Handle boolean data
        if maxima_values.dtype == bool:
            # For boolean data, no need to sort
            sort_idx = np.arange(len(maxima_values))
        else:
            sort_idx = np.argsort(-maxima_values)
        maxima_coords = maxima_coords[sort_idx]
        maxima_values = maxima_values[sort_idx]

        # Apply minimum distance constraint
        keep_mask = np.ones(len(maxima_coords), dtype=bool)
        for j in range(len(maxima_coords)):
            if not keep_mask[j]:
                continue
            # Remove nearby maxima
            for k in range(j + 1, len(maxima_coords)):
                if keep_mask[k]:
                    dist = np.linalg.norm(maxima_coords[j] - maxima_coords[k])
                    if dist < min_dist_voxels:
                        keep_mask[k] = False

        # Keep only filtered maxima
        maxima_coords = maxima_coords[keep_mask]
        maxima_values = maxima_values[keep_mask]

        # Convert to world coordinates
        for coord, value in zip(maxima_coords, maxima_values):
            world_coord = space.grid_to_coord(coord.reshape(1, -1))[0]
            maxima_list.append(
                [i, world_coord[0], world_coord[1], world_coord[2], value]
            )

    if maxima_list:
        return np.array(maxima_list)
    else:
        return np.empty((0, 5))

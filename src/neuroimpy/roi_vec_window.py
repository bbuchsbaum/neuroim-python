"""ROIVecWindow - Windowed ROI for 4D data (time series within a spatial ROI).

Provides access to time-series data extracted from a spatial neighbourhood,
analogous to the relationship between ROIVolWindow and ROIVol but for
vector-valued (4D) data.

Direct translation of R's neuroim2 ROIVecWindow concept.
"""

import numpy as np
from typing import Optional

from .neuro_space import NeuroSpace


class ROIVecWindow:
    """Windowed ROI for 4D data (time series within a spatial ROI).

    Stores a compact (n_timepoints x n_voxels) data matrix together with the
    voxel coordinates, and provides convenient accessors for individual voxel
    time series and summary statistics.

    Parameters
    ----------
    space : NeuroSpace
        The spatial metadata for the parent volume.
    coords : np.ndarray
        N x 3 array of (i, j, k) voxel coordinates.
    data : np.ndarray
        (n_timepoints x n_voxels) array of time-series data.
    parent_index : int, optional
        Linear index of the centre voxel in the parent volume (default 0).
    center_index : int, optional
        Position of the centre voxel within *coords* (default 0).

    Examples
    --------
    >>> from neuroimpy import NeuroSpace
    >>> space = NeuroSpace([64, 64, 64])
    >>> coords = np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]])
    >>> data = np.random.randn(100, 3)
    >>> win = ROIVecWindow(space, coords, data)
    >>> win.num_voxels
    3
    >>> win.num_timepoints
    100
    >>> win.time_series(0).shape
    (100,)

    R Equivalent
    ------------
    neuroim2::ROIVecWindow
    """

    def __init__(
        self,
        space: NeuroSpace,
        coords: np.ndarray,
        data: np.ndarray,
        parent_index: int = 0,
        center_index: int = 0,
    ):
        if not isinstance(space, NeuroSpace):
            raise TypeError("space must be a NeuroSpace object")

        coords = np.atleast_2d(np.asarray(coords))
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an N x 3 array")

        data = np.atleast_2d(np.asarray(data, dtype=float))
        if data.shape[1] != coords.shape[0]:
            raise ValueError(
                f"data columns ({data.shape[1]}) must equal number of "
                f"coordinates ({coords.shape[0]})"
            )

        self.space = space
        self.coords = coords
        self.data = data
        self.parent_index = int(parent_index)
        self.center_index = int(center_index)

    @property
    def parent_grid(self) -> np.ndarray:
        """Parent voxel grid coordinate for this window."""
        return self.space.index_to_grid(np.array([self.parent_index], dtype=int))[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def time_series(self, voxel_idx: int) -> np.ndarray:
        """Return the time series for a single voxel.

        Parameters
        ----------
        voxel_idx : int
            Column index into the data matrix (0-based).

        Returns
        -------
        np.ndarray
            1D array of length *num_timepoints*.
        """
        return self.data[:, voxel_idx]

    def mean_series(self) -> np.ndarray:
        """Return the mean time series averaged across all voxels.

        Returns
        -------
        np.ndarray
            1D array of length *num_timepoints*.
        """
        return np.mean(self.data, axis=1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_voxels(self) -> int:
        """Number of voxels in the window."""
        return self.coords.shape[0]

    @property
    def num_timepoints(self) -> int:
        """Number of time points in the data."""
        return self.data.shape[0]

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of voxels."""
        return self.num_voxels

    def __repr__(self) -> str:
        return (
            f"ROIVecWindow(n_voxels={self.num_voxels}, "
            f"n_timepoints={self.num_timepoints}, "
            f"parent_index={self.parent_index}, "
            f"center_index={self.center_index})"
        )

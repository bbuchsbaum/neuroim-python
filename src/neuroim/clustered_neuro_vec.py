"""ClusteredNeuroVec - 4D data with cluster assignments and shared time-series per cluster.

Represents 4D data using cluster assignments and shared time series.
"""

import numpy as np
from typing import Dict, Optional

from .neuro_space import NeuroSpace
from .clustered_neuro_vol import ClusteredNeuroVol


class ClusteredNeuroVec:
    """4D neuroimaging data with cluster assignments and shared time-series per cluster.

    Each voxel belongs to a cluster (defined by a ``ClusteredNeuroVol``), and
    all voxels within a cluster share a single representative time-series
    stored in ``ts``.

    Parameters
    ----------
    cvol : ClusteredNeuroVol
        The spatial cluster volume defining cluster membership.
    ts : np.ndarray
        Time-series matrix of shape ``(n_time, n_clusters)``.  Column *k*
        holds the representative time-series for cluster *k* (using the
        unique cluster IDs sorted in ascending order).
    label : str, optional
        Descriptive label for this object.

    Attributes
    ----------
    cvol : ClusteredNeuroVol
    ts : np.ndarray
    cl_map : np.ndarray
        1-D integer array of cluster IDs (one per masked voxel), taken
        from ``cvol.clusters``.
    label : str

    R Equivalent
    ------------
    neuroim2::ClusteredNeuroVec
    """

    def __init__(
        self,
        cvol: ClusteredNeuroVol,
        ts: np.ndarray,
        label: str = "",
    ):
        if not isinstance(cvol, ClusteredNeuroVol):
            raise TypeError("cvol must be a ClusteredNeuroVol")

        ts = np.asarray(ts)
        if ts.ndim != 2:
            raise ValueError(f"ts must be 2-D (time x clusters), got {ts.ndim}-D")

        n_clusters = cvol.num_clusters()
        if ts.shape[1] != n_clusters:
            raise ValueError(
                f"ts has {ts.shape[1]} columns but cvol has {n_clusters} clusters"
            )

        self.cvol = cvol
        self.ts = ts
        self.cl_map = cvol.clusters
        self.label = label

        # sorted unique cluster IDs for column mapping
        self._cluster_ids = np.sort(np.array(list(cvol.cluster_map.keys())))
        self._id_to_col: Dict[int, int] = {
            int(cid): col for col, cid in enumerate(self._cluster_ids)
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def cluster_timeseries(self, cluster_id: int) -> np.ndarray:
        """Return the representative time-series for a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster identifier.

        Returns
        -------
        np.ndarray
            1-D array of length ``n_time``.
        """
        col = self._id_to_col.get(cluster_id)
        if col is None:
            raise KeyError(f"Cluster ID {cluster_id} not found")
        return self.ts[:, col]

    def voxel_cluster(self, voxel_index: int) -> int:
        """Return the cluster ID assigned to a voxel.

        Parameters
        ----------
        voxel_index : int
            Linear index into the masked voxel array (``cl_map``).

        Returns
        -------
        int
        """
        return int(self.cl_map[voxel_index])

    def voxel_timeseries(self, voxel_index: int) -> np.ndarray:
        """Return the time-series for a voxel (its cluster's representative).

        Parameters
        ----------
        voxel_index : int
            Linear index into the masked voxel array.

        Returns
        -------
        np.ndarray
            1-D array of length ``n_time``.
        """
        cid = self.voxel_cluster(voxel_index)
        return self.cluster_timeseries(cid)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_clusters(self):
        """Iterate over ``(cluster_id, timeseries)`` pairs.

        Yields
        ------
        tuple of (int, np.ndarray)
        """
        for cid in self._cluster_ids:
            yield int(cid), self.cluster_timeseries(int(cid))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_time(self) -> int:
        """Number of time points."""
        return self.ts.shape[0]

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return self.ts.shape[1]

    @property
    def cluster_ids(self) -> np.ndarray:
        """Sorted array of unique cluster IDs."""
        return self._cluster_ids.copy()

    @property
    def space(self) -> NeuroSpace:
        """The spatial metadata from the underlying ClusteredNeuroVol."""
        return self.cvol.space

    @property
    def shape(self):
        """Shape of the underlying 3D space."""
        return self.cvol.shape

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"ClusteredNeuroVec\n"
            f"  Num Clusters : {self.n_clusters}\n"
            f"  Num Time     : {self.n_time}\n"
            f"  Dimension    : {' X '.join(map(str, self.cvol.space.dim))}\n"
            f"  Spacing      : {' X '.join(map(str, self.cvol.space.spacing))}\n"
            f"  Label        : {self.label!r}"
        )

"""ClusteredNeuroVec - 4D data with cluster assignments and shared time-series per cluster.

Represents 4D data using cluster assignments and shared time series.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .neuro_space import NeuroSpace
from .clustered_neuro_vol import ClusteredNeuroVol
from .results import ParcelContrastParams, Receipt, chain_receipt, receipt_for


@dataclass(frozen=True)
class ParcelEffectResult:
    """Per-cluster scalar effects with owned projection back to volume space."""

    labels: np.ndarray
    effects: np.ndarray
    cvol: ClusteredNeuroVol
    provenance: Receipt
    atlas_provenance: Optional[Any] = None
    positive_name: str = "task"
    negative_name: str = "rest"

    @property
    def values(self) -> np.ndarray:
        """Alias for the numeric parcel effects."""
        return self.effects

    @property
    def winning_label(self) -> int:
        """Cluster id with the largest positive effect."""
        return int(self.labels[int(np.argmax(self.effects))])

    def map_to_volume(
        self, *, dtype: Any = np.float64, fill: float = 0.0
    ):
        """Project parcel effects into a dense 3-D volume.

        Voxel values are filled by cluster id, preserving the underlying
        ``ClusteredNeuroVol`` spatial frame and carrying this result's Receipt.
        """
        from .neuro_vol import DenseNeuroVol

        labels = np.asarray(self.labels, dtype=np.int32)
        effects = np.asarray(self.effects, dtype=dtype)
        if labels.ndim != 1 or effects.ndim != 1:
            raise ValueError("labels and effects must be 1-D")
        if labels.size != effects.size:
            raise ValueError(
                f"labels has {labels.size} entries but effects has {effects.size}"
            )

        flat = np.full(int(np.prod(self.cvol.shape)), fill, dtype=dtype)
        for label, effect in zip(labels, effects):
            indices = self.cvol.cluster_map.get(int(label))
            if indices is None:
                raise KeyError(f"Cluster ID {int(label)} not found")
            flat[indices] = effect

        out = flat.reshape(self.cvol.shape, order="F")
        vol = DenseNeuroVol(out, self.cvol.space, label="parcel_effect")
        vol.provenance = self.provenance
        return vol


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

    """

    def __init__(
        self,
        cvol: ClusteredNeuroVol,
        ts: np.ndarray,
        label: str = "",
        provenance: Optional[Receipt] = None,
        atlas_provenance: Optional[Any] = None,
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
        self.provenance = provenance
        self.atlas_provenance = atlas_provenance

        # sorted unique cluster IDs for column mapping
        self._cluster_ids = np.sort(np.array(list(cvol.cluster_map.keys())))
        self._id_to_col: Dict[int, int] = {
            int(cid): col for col, cid in enumerate(self._cluster_ids)
        }

    @classmethod
    def from_neurovec(
        cls,
        vec,
        cvol: ClusteredNeuroVol,
        *,
        label: str = "",
    ) -> "ClusteredNeuroVec":
        """Extract per-cluster mean time series from a NeuroVec.

        This factory mirrors :meth:`neuroim.NeuroVec.parcel_means` for users
        who discover the clustered container first.
        """
        return vec.parcel_means(cvol, label=label)

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

    def contrast(
        self,
        condition,
        *,
        positive_name: str = "task",
        negative_name: str = "rest",
    ) -> ParcelEffectResult:
        """Compute a positive-minus-negative mean contrast per cluster.

        ``condition`` is a boolean vector aligned to the time axis. ``True``
        samples form the positive condition; ``False`` samples form the
        negative condition. The returned result owns label/effect alignment,
        volume projection, atlas provenance, and chained Receipt metadata.
        """
        condition = np.asarray(condition, dtype=bool)
        if condition.shape != (self.n_time,):
            raise ValueError(
                f"condition shape {condition.shape} != time axis {(self.n_time,)}"
            )
        if not condition.any() or condition.all():
            raise ValueError("condition must contain both positive and negative samples")

        effects = self.ts[condition].mean(axis=0) - self.ts[~condition].mean(axis=0)
        method_name = f"contrast[{positive_name}-{negative_name}]"
        params = ParcelContrastParams(
            method_name=method_name,
            positive_name=positive_name,
            negative_name=negative_name,
        )
        if isinstance(self.provenance, Receipt):
            receipt = chain_receipt(self, params=params, n_voxels=self.n_clusters)
        else:
            receipt = receipt_for(
                self.cvol,
                mask=self.cvol.as_dense().data,
                n_voxels=self.n_clusters,
                params=params,
            )
        return ParcelEffectResult(
            labels=self.cluster_ids,
            effects=effects,
            cvol=self.cvol,
            provenance=receipt,
            atlas_provenance=self.atlas_provenance,
            positive_name=positive_name,
            negative_name=negative_name,
        )

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

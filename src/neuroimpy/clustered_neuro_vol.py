import numpy as np
from typing import List, Dict
from .neuro_vol import NeuroVol, LogicalNeuroVol, SparseNeuroVol, DenseNeuroVol
from .neuro_space import NeuroSpace

class ClusteredNeuroVol(NeuroVol):
    """
    A class representing a clustered volumetric neuroimaging data.
    """

    def __init__(self, mask: LogicalNeuroVol, clusters: np.ndarray, label_map: Dict[str, int] = None):
        """
        Initialize a ClusteredNeuroVol.

        :param mask: A LogicalNeuroVol representing the spatial domain of the clusters.
        :param clusters: An array of cluster labels for each voxel in the mask.
        :param label_map: A dictionary mapping cluster names to cluster numbers.
        """
        super().__init__(mask.space)
        self.mask = mask
        clusters_array = np.asarray(clusters)
        mask_indices = np.where(mask.data.ravel(order="F"))[0]
        if clusters_array.shape == self.shape:
            clusters = clusters_array.ravel(order="F")[mask_indices]
        else:
            clusters = clusters_array.ravel(order="F")
            if clusters.size != len(mask_indices):
                raise ValueError(
                    f"clusters length ({clusters.size}) must match number of mask voxels ({len(mask_indices)})"
                )
        self._mask_indices = mask_indices.astype(int)
        self.clusters = clusters
        self.label_map = label_map or {}
        self.cluster_map = self._create_cluster_map()

    def _create_cluster_map(self) -> Dict[int, np.ndarray]:
        """
        Create a mapping from cluster IDs to the set of 1D spatial indices belonging to that cluster.
        """
        unique_clusters = np.unique(self.clusters)
        return {
            cluster: self._mask_indices[self.clusters == cluster].astype(int)
            for cluster in unique_clusters
        }

    def __getitem__(self, key):
        if isinstance(key, str):
            # If key is a string, assume it's a cluster label
            cluster_id = self.label_map.get(key)
            if cluster_id is None:
                raise KeyError(f"Cluster label '{key}' not found in label map")
            return self.cluster_map[cluster_id]
        elif isinstance(key, int):
            # If key is an integer, assume it's a cluster ID
            return self.cluster_map.get(key, np.array([]))
        else:
            # Otherwise, delegate to the mask's __getitem__
            return self.mask[key]

    def __setitem__(self, key, value):
        raise TypeError("ClusteredNeuroVol is read-only and does not support item assignment")

    def get_cluster_mask(self, cluster_id: int) -> LogicalNeuroVol:
        """
        Get a binary mask for a specific cluster.

        :param cluster_id: The ID of the cluster.
        :return: A LogicalNeuroVol representing the mask of the specified cluster.
        """
        cluster_mask = np.zeros(self.shape, dtype=bool)
        cluster_mask[np.unravel_index(self.cluster_map[cluster_id], self.shape, order="F")] = True
        return LogicalNeuroVol(cluster_mask, self.space)

    def get_cluster_data(self, data: NeuroVol, cluster_id: int) -> np.ndarray:
        """
        Get the data for a specific cluster from a NeuroVol.

        :param data: A NeuroVol containing the data to extract.
        :param cluster_id: The ID of the cluster.
        :return: An array of data values for the specified cluster.
        """
        if not isinstance(data, NeuroVol) or data.space != self.space:
            raise ValueError("Data must be a NeuroVol with matching space")
        return data[np.unravel_index(self.cluster_map[cluster_id], self.shape, order="F")]

    def to_sparse(self) -> SparseNeuroVol:
        """
        Convert the clustered volume to a SparseNeuroVol.

        :return: A SparseNeuroVol representation of the clustered volume.
        """
        return SparseNeuroVol(self.clusters, self.space, mask=self.mask)

    def cluster_sizes(self) -> Dict[int, int]:
        """
        Get the size (number of voxels) of each cluster.

        :return: A dictionary mapping cluster IDs to their sizes.
        """
        return {cluster_id: len(indices) for cluster_id, indices in self.cluster_map.items()}

    def num_clusters(self) -> int:
        """
        Get the number of clusters.

        :return: The number of unique clusters.
        """
        return len(self.cluster_map)

    def cluster_centers(self) -> Dict[int, np.ndarray]:
        """
        Calculate the center of mass for each cluster.

        :return: A dictionary mapping cluster IDs to their center coordinates.
        """
        centers = {}
        for cluster_id, indices in self.cluster_map.items():
            coords = np.array(np.unravel_index(indices, self.shape, order="F")).T
            center = np.mean(coords, axis=0)
            centers[cluster_id] = center
        return centers

    def split_clusters(self, data: NeuroVol) -> Dict[int, np.ndarray]:
        """
        Split a NeuroVol by clusters, extracting data for each cluster.

        :param data: A NeuroVol containing the data to split.
        :return: A dictionary mapping cluster IDs to their data arrays.
        """
        if not isinstance(data, NeuroVol) or data.space != self.space:
            raise ValueError("Data must be a NeuroVol with matching space")

        result = {}
        data_flat = data.data.reshape(-1, order="F")
        for cluster_id, indices in self.cluster_map.items():
            result[cluster_id] = data_flat[indices]
        return result

    @property
    def values(self) -> np.ndarray:
        """Get the cluster label values."""
        return self.clusters

    def as_sparse(self) -> SparseNeuroVol:
        """Convert to SparseNeuroVol."""
        return self.to_sparse()
    
    def as_dense(self) -> DenseNeuroVol:
        """Convert to DenseNeuroVol."""
        # Create dense representation
        dense_data = np.zeros(self.shape, dtype=self.clusters.dtype)
        dense_data.reshape(-1, order="F")[self._mask_indices] = self.clusters
        return DenseNeuroVol(dense_data, self.space)
    
    def as_logical(self) -> LogicalNeuroVol:
        """Convert to LogicalNeuroVol."""
        return self.mask
    
    def _arithmetic_op(self, other, op):
        """Arithmetic operations delegate to dense representation."""
        return self.as_dense()._arithmetic_op(other, op)

    def _comparison_op(self, other, op):
        """Comparison operations delegate to dense representation."""
        return self.as_dense()._comparison_op(other, op)

    def __repr__(self):
        return (f"ClusteredNeuroVol(\n"
                f"  Num clusters   : {self.num_clusters()}\n"
                f"  Dimension      : {self.shape}\n"
                f"  Spacing        : {' X '.join(map(str, self.space.spacing))}\n"
                f"  Origin         : {' X '.join(map(str, self.space.origin))}\n"
                f"  Axes           : {' '.join(ax.axis for ax in self.space.axes)}")

def clustered_neuro_vol(mask: LogicalNeuroVol, clusters: np.ndarray, label_map: Dict[str, int] = None) -> ClusteredNeuroVol:
    """
    Factory function to create a ClusteredNeuroVol instance.

    :param mask: A LogicalNeuroVol representing the spatial domain of the clusters.
    :param clusters: An array of cluster labels for each voxel in the mask.
    :param label_map: A dictionary mapping cluster names to cluster numbers.
    :return: A new ClusteredNeuroVol instance.
    """
    return ClusteredNeuroVol(mask, clusters, label_map)

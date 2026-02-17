"""NeuroBucket - A sequence of labeled NeuroVol objects forming a named 4D representation.

Direct translation of R's neuroim2 NeuroBucket concept.
"""

import numpy as np
from typing import List, Optional, Union

from .neuro_space import NeuroSpace
from .neuro_vol import NeuroVol


class NeuroBucket:
    """A sequence of labeled NeuroVol objects forming a named 4D representation.

    NeuroBucket holds a list of named 3D volumes that share the same spatial
    geometry.  It supports iteration, integer/label indexing, slicing, and
    ``len()``.

    Parameters
    ----------
    labels : list of str
        Names for each volume.
    data : list of NeuroVol
        The volumes.  Must all share the same 3D space.
    space : NeuroSpace
        The common 3D spatial metadata.

    R Equivalent
    ------------
    neuroim2::NeuroBucket
    """

    def __init__(self, labels: List[str], data: List[NeuroVol], space: NeuroSpace):
        if len(labels) != len(data):
            raise ValueError(
                f"labels length ({len(labels)}) must match data length ({len(data)})"
            )
        if len(data) == 0:
            raise ValueError("data must contain at least one volume")

        # Validate all volumes share the same space dimensions
        for i, vol in enumerate(data):
            if vol.shape != tuple(space.dim):
                raise ValueError(
                    f"Volume {i} shape {vol.shape} does not match space {tuple(space.dim)}"
                )

        self.labels = list(labels)
        self.data = list(data)
        self.space = space
        # Build label -> index lookup
        self._label_index = {lab: idx for idx, lab in enumerate(self.labels)}

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of volumes in the bucket."""
        return len(self.data)

    def __iter__(self):
        """Iterate over (label, volume) pairs."""
        return zip(self.labels, self.data)

    def __contains__(self, label: str) -> bool:
        """Check if a label exists in the bucket."""
        return label in self._label_index

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key) -> Union[NeuroVol, 'NeuroBucket']:
        """Index by integer position, label string, or slice.

        Parameters
        ----------
        key : int, str, or slice
            - int: positional index returning a single NeuroVol.
            - str: label name returning the corresponding NeuroVol.
            - slice: returns a new NeuroBucket with the selected subset.

        Returns
        -------
        NeuroVol or NeuroBucket
        """
        if isinstance(key, str):
            idx = self._label_index.get(key)
            if idx is None:
                raise KeyError(f"Label '{key}' not found in bucket")
            return self.data[idx]
        elif isinstance(key, (int, np.integer)):
            return self.data[key]
        elif isinstance(key, slice):
            sub_labels = self.labels[key]
            sub_data = self.data[key]
            return NeuroBucket(sub_labels, sub_data, self.space)
        else:
            raise TypeError(f"Invalid key type {type(key)}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        """Shape of the underlying 3D space."""
        return tuple(self.space.dim)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (always 3)."""
        return self.space.ndim

    @property
    def spacing(self) -> np.ndarray:
        """Voxel spacing."""
        return self.space.spacing

    @property
    def origin(self) -> np.ndarray:
        """Origin coordinates."""
        return self.space.origin

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"NeuroBucket\n"
            f"  Num Volumes : {len(self)}\n"
            f"  Dimension   : {' X '.join(map(str, self.space.dim))}\n"
            f"  Spacing     : {' X '.join(map(str, self.spacing))}\n"
            f"  Origin      : {', '.join(map(str, self.origin))}\n"
            f"  Labels      : {', '.join(self.labels[:5])}"
            + (", ..." if len(self.labels) > 5 else "")
        )

"""IndexLookupVol - Maps 1D grid indices to sparse table indices.

Provides efficient bidirectional mapping between linear grid indices
and compact table indices for sparse neuroimaging volumes.

Provides a compact lookup volume for sparse voxel-table indexing.
"""

import numpy as np
from typing import Union

from .neuro_space import NeuroSpace


class IndexLookupVol:
    """Maps 1D grid indices to sparse table indices for efficient sparse lookups.

    Given a set of active grid indices (e.g., from a brain mask), this class
    builds a bidirectional mapping so that grid indices can be quickly
    translated to compact 0-based table indices and vice versa.

    Parameters
    ----------
    space : NeuroSpace
        The spatial metadata for the volume.
    indices : np.ndarray
        1D array of active grid indices (linear indices into the volume).

    Examples
    --------
    >>> from neuroim import NeuroSpace
    >>> space = NeuroSpace([10, 10, 10])
    >>> active = np.array([5, 20, 100, 500])
    >>> lut = IndexLookupVol(space, active)
    >>> lut.lookup_index(20)
    1
    >>> lut.grid_to_table(np.array([5, 500]))
    array([0, 3])

    R Equivalent
    ------------
    neuroim2::IndexLookupVol
    """

    def __init__(self, space: NeuroSpace, indices: np.ndarray):
        if not isinstance(space, NeuroSpace):
            raise TypeError("space must be a NeuroSpace object")

        indices = np.asarray(indices, dtype=np.intp).ravel()

        vol_size = int(np.prod(space.dim))
        if len(indices) > 0 and (np.any(indices < 0) or np.any(indices >= vol_size)):
            raise ValueError(
                f"indices must be in range [0, {vol_size}), "
                f"got range [{indices.min()}, {indices.max()}]"
            )

        self.space = space
        self._grid_indices = indices  # table_idx -> grid_idx

        # Build grid_idx -> table_idx lookup.
        # Use a dense array when the index range is manageable,
        # otherwise fall back to a dictionary.
        if vol_size <= 10_000_000:  # ~10M entries, ~80 MB for int64
            self._lookup = np.full(vol_size, -1, dtype=np.intp)
            self._lookup[indices] = np.arange(len(indices), dtype=np.intp)
            self._use_dict = False
        else:
            self._lookup = {int(g): int(t) for t, g in enumerate(indices)}
            self._use_dict = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_index(self, grid_idx: int) -> int:
        """Look up the table index for a single grid index.

        Parameters
        ----------
        grid_idx : int
            Linear index in the full volume grid.

        Returns
        -------
        int
            Corresponding table index (0-based position in the active set).

        Raises
        ------
        KeyError
            If *grid_idx* is not among the active indices.
        """
        if self._use_dict:
            try:
                return self._lookup[int(grid_idx)]
            except KeyError:
                raise KeyError(f"Grid index {grid_idx} not found in lookup table")
        else:
            idx = int(grid_idx)
            if idx < 0 or idx >= len(self._lookup) or self._lookup[idx] == -1:
                raise KeyError(f"Grid index {grid_idx} not found in lookup table")
            return int(self._lookup[idx])

    def grid_to_table(self, indices: np.ndarray) -> np.ndarray:
        """Convert an array of grid indices to table indices (batch).

        Parameters
        ----------
        indices : np.ndarray
            1D array of grid indices.

        Returns
        -------
        np.ndarray
            1D array of table indices.

        Raises
        ------
        KeyError
            If any grid index is not among the active indices.
        """
        indices = np.asarray(indices, dtype=np.intp).ravel()

        if self._use_dict:
            result = np.empty(len(indices), dtype=np.intp)
            for i, g in enumerate(indices):
                try:
                    result[i] = self._lookup[int(g)]
                except KeyError:
                    raise KeyError(f"Grid index {g} not found in lookup table")
            return result
        else:
            result = self._lookup[indices]
            bad = result == -1
            if np.any(bad):
                first_bad = indices[bad][0]
                raise KeyError(f"Grid index {first_bad} not found in lookup table")
            return result

    def table_to_grid(self, indices: np.ndarray) -> np.ndarray:
        """Convert an array of table indices back to grid indices (reverse).

        Parameters
        ----------
        indices : np.ndarray
            1D array of table indices.

        Returns
        -------
        np.ndarray
            1D array of grid indices.

        Raises
        ------
        IndexError
            If any table index is out of range.
        """
        indices = np.asarray(indices, dtype=np.intp).ravel()
        if len(indices) > 0 and (
            np.any(indices < 0) or np.any(indices >= len(self._grid_indices))
        ):
            raise IndexError(f"Table index out of range [0, {len(self._grid_indices)})")
        return self._grid_indices[indices]

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __contains__(self, grid_idx) -> bool:
        """Check whether a grid index is in the active set."""
        if self._use_dict:
            return int(grid_idx) in self._lookup
        else:
            idx = int(grid_idx)
            return 0 <= idx < len(self._lookup) and self._lookup[idx] != -1

    def __len__(self) -> int:
        """Number of active indices."""
        return len(self._grid_indices)

    def __repr__(self) -> str:
        vol_size = int(np.prod(self.space.dim))
        return (
            f"IndexLookupVol\n"
            f"  Space       : {' x '.join(map(str, self.space.dim))}\n"
            f"  Active      : {len(self)} / {vol_size}\n"
            f"  Density     : {len(self) / vol_size * 100:.1f}%"
        )

"""Voxel-series storage protocol and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy as np


IndexLike = Union[int, np.ndarray]


@runtime_checkable
class VoxelSeriesStore(Protocol):
    """Storage contract for vector-valued voxel series."""

    @property
    def shape(self) -> tuple[int, int, int, int]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def series(
        self, x: IndexLike, y: Optional[int] = None, z: Optional[int] = None
    ) -> np.ndarray: ...

    def volume(self, i: int) -> Any: ...

    def as_matrix(self) -> np.ndarray: ...


@runtime_checkable
class WritableVoxelSeriesStore(VoxelSeriesStore, Protocol):
    """Voxel-series store that also supports assignment."""

    def __setitem__(self, key: Any, value: Any) -> None: ...


@dataclass(frozen=True)
class NeuroVecStoreAdapter:
    """Adapter exposing a NeuroVec through the VoxelSeriesStore protocol."""

    vec: Any

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.vec.shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.vec.dtype)

    def series(
        self, x: IndexLike, y: Optional[int] = None, z: Optional[int] = None
    ) -> np.ndarray:
        return self.vec.series(x, y, z)

    def volume(self, i: int) -> Any:
        return self.vec[:, :, :, i]

    def as_matrix(self) -> np.ndarray:
        if hasattr(self.vec, "as_matrix"):
            return self.vec.as_matrix()
        data = self.vec.data
        return data.reshape((-1, data.shape[-1]), order="F").T

    def __getitem__(self, key: Any) -> Any:
        return self.vec[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.vec[key] = value


DenseStore = NeuroVecStoreAdapter
SparseStore = NeuroVecStoreAdapter
BigStore = NeuroVecStoreAdapter
FileBackedStore = NeuroVecStoreAdapter
MappedStore = NeuroVecStoreAdapter


__all__ = [
    "VoxelSeriesStore",
    "WritableVoxelSeriesStore",
    "NeuroVecStoreAdapter",
    "DenseStore",
    "SparseStore",
    "BigStore",
    "FileBackedStore",
    "MappedStore",
]

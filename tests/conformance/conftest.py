"""Shared fixture for the VoxelSeriesStore conformance matrix.

Builds one deterministic 8x8x4x10 voxel array and provides factories that wrap
it into each of the five NeuroVec backends.  Tests in this directory parametrize
over the factories so every backend runs the same conformance operations on
identical data.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest

from neuroim import (
    BigNeuroVec,
    DenseNeuroVec,
    FileBackedNeuroVec,
    MappedNeuroVec,
    NeuroSpace,
    NeuroVec,
    SparseNeuroVec,
)
from neuroim.neuro_vol import DenseNeuroVol


SHAPE: Tuple[int, int, int, int] = (8, 8, 4, 10)


@pytest.fixture(scope="module")
def fixture_array() -> np.ndarray:
    """Deterministic float32 4D array shared across every backend."""
    rng = np.random.default_rng(seed=20260514)
    return rng.standard_normal(size=SHAPE).astype(np.float32)


@pytest.fixture(scope="module")
def fixture_space() -> NeuroSpace:
    """4D NeuroSpace with mm-shaped spacing."""
    return NeuroSpace(dim=SHAPE, spacing=(2.0, 2.0, 3.0, 1.0))


def _make_dense(arr: np.ndarray, space: NeuroSpace) -> DenseNeuroVec:
    return DenseNeuroVec(arr.copy(), space)


def _make_sparse(arr: np.ndarray, space: NeuroSpace) -> SparseNeuroVec:
    full_mask = np.ones(arr.shape[:3], dtype=bool)
    return DenseNeuroVec(arr.copy(), space).as_sparse(full_mask)


def _make_big(arr: np.ndarray, space: NeuroSpace) -> BigNeuroVec:
    return BigNeuroVec(arr.astype(np.float32, copy=True), space)


def _make_file_backed(arr: np.ndarray, space: NeuroSpace, tmp_path: Path) -> FileBackedNeuroVec:
    import nibabel as nib

    affine = np.asarray(space.trans[:4, :4], dtype=float)
    filenames = []
    for t in range(arr.shape[-1]):
        path = tmp_path / f"vol_{t:03d}.nii.gz"
        nib.save(nib.Nifti1Image(arr[..., t].astype(np.float32), affine), str(path))
        filenames.append(str(path))
    return FileBackedNeuroVec(filenames)


def _make_mapped(arr: np.ndarray, space: NeuroSpace) -> MappedNeuroVec:
    source = DenseNeuroVec(arr.copy(), space)
    return MappedNeuroVec(source, map_fun=lambda x: x, inverse_fun=lambda x: x)


BACKEND_FACTORIES = {
    "dense": _make_dense,
    "sparse": _make_sparse,
    "big": _make_big,
    "file_backed": _make_file_backed,
    "mapped": _make_mapped,
}


@pytest.fixture(
    params=list(BACKEND_FACTORIES.keys()),
    ids=lambda name: name,
)
def backend(
    request,
    fixture_array: np.ndarray,
    fixture_space: NeuroSpace,
    tmp_path_factory: pytest.TempPathFactory,
) -> NeuroVec:
    """Yield a freshly-built backend instance for each backend name."""
    name = request.param
    factory: Callable = BACKEND_FACTORIES[name]
    if name == "file_backed":
        tmp_dir = tmp_path_factory.mktemp(f"fb_{name}")
        return factory(fixture_array, fixture_space, tmp_dir)
    return factory(fixture_array, fixture_space)


@pytest.fixture(scope="module")
def fixture_volume_at_zero(fixture_array: np.ndarray, fixture_space: NeuroSpace) -> DenseNeuroVol:
    """The volume at t=0 from the canonical fixture, used for ground-truth."""
    vol_space = NeuroSpace(
        dim=fixture_array.shape[:3],
        spacing=fixture_space.spacing[:3],
        origin=fixture_space.origin[:3],
    )
    return DenseNeuroVol(fixture_array[..., 0].copy(), vol_space)

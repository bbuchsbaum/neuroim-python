"""Regression tests for concat spatial-contract enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vec import DenseNeuroVec
from neuroim.operations import concat


def _vec(space: NeuroSpace, value: float = 1.0) -> DenseNeuroVec:
    data = np.full(tuple(space.dim), value, dtype=np.float64)
    return DenseNeuroVec(data, space)


def _flipped_space() -> NeuroSpace:
    affine = np.eye(4)
    affine[0, 0] = -2.0
    affine[0, 3] = 12.0
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0
    return NeuroSpace((4, 4, 4, 2), trans=affine)


def _rotated_space() -> NeuroSpace:
    affine = np.eye(4)
    affine[0, 1] = -2.0
    affine[1, 0] = 2.0
    affine[2, 2] = 2.0
    affine[:3, 3] = [10.0, -4.0, 2.0]
    return NeuroSpace((4, 4, 4, 2), trans=affine)


def test_neurovec_concat_rejects_affine_mismatch():
    base = _vec(NeuroSpace((4, 4, 4, 2), spacing=(2, 2, 2, 1)), 1.0)
    flipped = _vec(_flipped_space(), 2.0)

    with pytest.raises(ValueError, match="spatial contract mismatch"):
        base.concat(flipped)


def test_operations_concat_rejects_affine_mismatch():
    base = _vec(NeuroSpace((4, 4, 4, 2), spacing=(2, 2, 2, 1)), 1.0)
    flipped = _vec(_flipped_space(), 2.0)

    with pytest.raises(ValueError, match="spatial contract mismatch"):
        concat(base, flipped)


def test_neurovec_concat_still_accepts_matching_spaces():
    space = _rotated_space()
    first = _vec(space, 1.0)
    second = _vec(space, 2.0)

    out = first.concat(second)

    assert out.shape == (4, 4, 4, 4)
    np.testing.assert_allclose(out.space.affine, space.affine)
    np.testing.assert_array_equal(out.data[..., :2], first.data)
    np.testing.assert_array_equal(out.data[..., 2:], second.data)

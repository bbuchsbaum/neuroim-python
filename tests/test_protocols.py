"""Structural Protocol acceptance test (WP-6).

The headline acceptance criterion is that a *hand-rolled* class satisfying
``NeuroVecLike`` -- with no inheritance from the ``NeuroVec`` ABC -- can
flow through ``searchlight()`` and produce a ``SearchlightResult``.
"""

from __future__ import annotations

import numpy as np

from neuroim import searchlight_apply
from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vol import LogicalNeuroVol
from neuroim.results import SearchlightResult
from neuroim.protocols import (
    MaskLike,
    NeuroVecLike,
    NeuroVolLike,
    SupportsDense,
    SupportsSparse,
)


# --- Hand-rolled, ABC-free Protocol citizens --------------------------------


class _PlainVec:
    """A minimal 4-D voxel-series container.

    Carries only the documented Protocol surface: ``space``, ``shape``, and
    ``series(coords) -> ndarray``.  Notably does NOT inherit from
    :class:`neuroim.neuro_vec.NeuroVec`.
    """

    def __init__(self, data: np.ndarray, space: NeuroSpace):
        if data.ndim != 4:
            raise ValueError("PlainVec expects 4-D data")
        self._data = data
        self.space = space
        self.shape = data.shape  # 4-tuple

    def series(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=int)
        return self._data[coords[:, 0], coords[:, 1], coords[:, 2], :].T


class _PlainVol:
    """A minimal 3-D volume satisfying ``NeuroVolLike``."""

    def __init__(self, data: np.ndarray, space: NeuroSpace):
        if data.ndim != 3:
            raise ValueError("PlainVol expects 3-D data")
        self.data = data
        self.space = space
        self.shape = data.shape


# --- Protocol runtime conformance ------------------------------------------


def test_dense_subclasses_satisfy_structural_protocols():
    from neuroim.neuro_vol import DenseNeuroVol

    space3d = NeuroSpace((4, 4, 4))
    vol = DenseNeuroVol(np.zeros((4, 4, 4)), space3d)
    assert isinstance(vol, NeuroVolLike)
    assert isinstance(vol, SupportsDense)
    assert isinstance(vol, SupportsSparse)


def test_logical_volume_satisfies_mask_like():
    space3d = NeuroSpace((4, 4, 4))
    mask = LogicalNeuroVol(np.ones((4, 4, 4), dtype=bool), space3d)
    assert isinstance(mask, MaskLike)


def test_hand_rolled_vec_satisfies_neuro_vec_like():
    space4d = NeuroSpace((3, 3, 3, 5))
    vec = _PlainVec(np.zeros((3, 3, 3, 5)), space4d)
    assert isinstance(vec, NeuroVecLike)


def test_hand_rolled_vol_satisfies_neuro_vol_like():
    space3d = NeuroSpace((3, 3, 3))
    vol = _PlainVol(np.zeros((3, 3, 3)), space3d)
    assert isinstance(vol, NeuroVolLike)


# --- Acceptance: a hand-rolled NeuroVecLike flows through searchlight ------


def test_searchlight_accepts_protocol_only_neuro_vec_like():
    """The headline acceptance criterion: no NeuroVec inheritance required."""
    rng = np.random.default_rng(7)
    spatial = (6, 6, 4)
    n_time = 4

    data = rng.standard_normal(spatial + (n_time,)).astype(np.float64)
    space4d = NeuroSpace((*spatial, n_time))
    from neuroim.neuro_vec import NeuroVec

    vec = _PlainVec(data, space4d)
    assert isinstance(vec, NeuroVecLike)
    assert not isinstance(vec, NeuroVec)  # sanity: no ABC inheritance

    mask_array = np.zeros(spatial, dtype=bool)
    mask_array[2:4, 2:4, 1:3] = True
    space3d = NeuroSpace(spatial)
    mask = LogicalNeuroVol(mask_array, space3d)

    result = searchlight_apply(
        mask,
        radius=2.0,
        method=lambda x: float(np.mean(x)),
        data=vec,
        return_legacy=False,
    )

    assert isinstance(result, SearchlightResult)
    assert result.centers.ndim == 2 and result.centers.shape[1] == 3
    assert result.values.shape[0] == result.centers.shape[0]
    assert result.space is mask.space
    assert result.provenance.method_name in {"<lambda>", "lambda"}

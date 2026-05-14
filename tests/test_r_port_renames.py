"""ME-5: Pythonic aliases for R-shaped ``as_*`` conversion verbs.

The R-port surface uses ``as_dense`` / ``as_sparse`` / ``as_logical``.  The
Pythonic equivalents — ``to_dense`` / ``to_sparse`` / ``to_logical`` —
are the canonical names going forward; the old names remain as shims
during the deprecation cycle.

This test asserts:
  - The new methods exist on both ``NeuroVec`` and ``NeuroVol`` hierarchies.
  - They return the same object identity / data as the legacy ``as_*``
    methods on a representative concrete subclass.
"""

from __future__ import annotations

import numpy as np

from neuroim import (
    DenseNeuroVec,
    DenseNeuroVol,
    NeuroSpace,
    SparseNeuroVec,
    SparseNeuroVol,
)


def _make_dense_vol():
    space = NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    data = np.random.default_rng(0).standard_normal((4, 4, 4)).astype(np.float64)
    return DenseNeuroVol(data, space)


def _make_dense_vec():
    space = NeuroSpace(dim=(4, 4, 4, 5), spacing=(2.0, 2.0, 2.0, 1.0))
    data = np.random.default_rng(0).standard_normal((4, 4, 4, 5)).astype(np.float64)
    return DenseNeuroVec(data, space)


# ----------------------------------------------------------------------
# NeuroVol
# ----------------------------------------------------------------------


def test_neurovol_to_dense_alias_exists_and_matches_as_dense():
    vol = _make_dense_vol()
    np.testing.assert_array_equal(vol.to_dense().data, vol.as_dense().data)


def test_neurovol_to_sparse_alias_exists_and_matches_as_sparse():
    vol = _make_dense_vol()
    np.testing.assert_array_equal(
        vol.to_sparse().data, vol.as_sparse().data
    )


def test_neurovol_to_logical_alias_exists_and_matches_as_logical():
    vol = _make_dense_vol()
    np.testing.assert_array_equal(
        np.asarray(vol.to_logical().data),
        np.asarray(vol.as_logical().data),
    )


# ----------------------------------------------------------------------
# NeuroVec
# ----------------------------------------------------------------------


def test_neurovec_to_dense_alias_matches_as_dense():
    vec = _make_dense_vec()
    np.testing.assert_array_equal(vec.to_dense().data, vec.as_dense().data)


def test_neurovec_to_sparse_alias_matches_as_sparse():
    vec = _make_dense_vec()
    np.testing.assert_array_equal(
        vec.to_sparse().data, vec.as_sparse().data
    )

"""Graph filters, searchlight, verify, extensions and simulation contracts
from the second correctness audit."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neuroim import DenseNeuroVol, LogicalNeuroVol, NeuroSpace
from neuroim.clustered_neuro_vol import ClusteredNeuroVol
from neuroim.graph_filter import cgb_filter, cgb_make_graph, cgb_smooth_loro
from neuroim.neuro_vec import DenseNeuroVec
from neuroim.nifti_extension import NiftiExtension, get_afni_attribute
from neuroim.searchlight_high_level import searchlight
from neuroim.simulation import simulate_fmri
from neuroim.verify import assert_same_mask


def test_graph_nodes_follow_linear_index_order():
    shape = (3, 2, 1)
    g = cgb_make_graph(NeuroSpace(shape), mask=np.ones(shape, bool))
    vals = np.zeros(shape)
    vals[2, 0, 0] = 100.0
    smoothed = cgb_filter(vals.ravel(order="F"), g).reshape(shape, order="F")
    # (2,0,0) has neighbours (1,0,0) and (2,1,0); each of those sees the spike
    # among its own neighbours, so the mean over them is non-zero there only.
    assert smoothed[1, 0, 0] > 0 and smoothed[2, 1, 0] > 0
    assert smoothed[0, 1, 0] == 0


def test_smooth_loro_uses_only_other_runs_original_data():
    d = np.random.default_rng(0).normal(size=(3, 3, 3, 6))
    vec = DenseNeuroVec(d, NeuroSpace(d.shape))
    runs = np.array([0, 0, 1, 1, 2, 2])
    out = np.asarray(cgb_smooth_loro(vec, 1e-3, runs).data)
    np.testing.assert_allclose(out[..., 0], d[..., 2:].mean(3), atol=1e-6)
    np.testing.assert_allclose(out[..., 4], d[..., :4].mean(3), atol=1e-6)


def test_clustered_vol_accepts_numpy_integer_ids():
    m = np.zeros((4, 4, 3), bool)
    m[1:3, 1:3, :] = True
    cv = ClusteredNeuroVol(
        LogicalNeuroVol(m, NeuroSpace(m.shape)), np.repeat([1, 2, 3], 4)
    )
    for key in cv.cluster_map:
        np.testing.assert_array_equal(cv[np.int64(key)], cv[int(key)])


def test_searchlight_keeps_non_finite_centres():
    sp = NeuroSpace((4, 4, 4))
    mask = LogicalNeuroVol(np.ones((4, 4, 4), bool), sp)
    vol = DenseNeuroVol(np.arange(64.0).reshape(4, 4, 4), sp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = searchlight(
            mask, 1.0, lambda x: np.inf if x.min() == 0 else x.sum(), data=vol
        )
    assert len(res.values) == 64
    assert np.isinf(np.asarray(res.map_to_volume().data)[0, 0, 0])


def test_assert_same_mask_checks_space_and_ignores_dtype():
    m = np.zeros((6, 6, 6), bool)
    m[1:5, 1:5, 1:5] = True
    sp = NeuroSpace((6, 6, 6), spacing=(2, 2, 2))
    shifted = NeuroSpace((6, 6, 6), spacing=(2, 2, 2), origin=(0, 0, 2))
    mask = LogicalNeuroVol(m, sp)
    assert_same_mask(mask, m.astype(np.uint8))
    with pytest.raises(ValueError):
        assert_same_mask(mask, LogicalNeuroVol(m, shifted))
    m2 = m.copy()
    m2[1, 1, 1] = False
    with pytest.raises(ValueError):
        assert_same_mask(mask, LogicalNeuroVol(m2, sp))


def test_afni_extension_attributes_are_typed():
    xml = b"""<?xml version='1.0' ?>
<AFNI_attributes ni_form="ni_group" >
<AFNI_atr ni_type="String" ni_dimen="1" atr_name="BRICK_LABS" >
 "Full_Fstat~Coef~"
</AFNI_atr>
<AFNI_atr ni_type="float" ni_dimen="3" atr_name="DELTA" >
 -3 -3 3
</AFNI_atr>
<AFNI_atr ni_type="int" ni_dimen="2" atr_name="DATASET_RANK" >
 3 2
</AFNI_atr>
</AFNI_attributes>
\x00\x00"""
    ext = NiftiExtension(4, xml)
    np.testing.assert_array_equal(get_afni_attribute(ext, "DELTA"), [-3.0, -3.0, 3.0])
    rank = get_afni_attribute(ext, "DATASET_RANK")
    assert rank.dtype.kind == "i" and list(rank) == [3, 2]
    assert get_afni_attribute(ext, "BRICK_LABS") == "Full_Fstat~Coef~"
    assert get_afni_attribute(ext, "MISSING") is None


def test_simulate_fmri_small_grid_and_zero_baseline():
    vec = simulate_fmri(
        NeuroSpace((3, 1, 4)),
        n_timepoints=30,
        n_regions=2,
        noise_sd=0.0,
        signal_sd=1.0,
        seed=0,
    )
    assert vec.shape == (3, 1, 4, 30)
    # Without noise, voxels outside any region and every voxel's first
    # (pre-response) sample sit at the zero baseline.
    np.testing.assert_allclose(np.asarray(vec.data)[..., 0], 0.0)

"""Plotting correctness carried over from neuroim2's figure redesign.

Overlay scales come from the whole map, graded opacity keeps supra-threshold
voxels visible, backgrounds are windowed over the head, ``plot_neuro_vol``
thresholds like neuroim2, and plane names resolve through the affine.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from neuroim import DenseNeuroVol, NeuroSpace  # noqa: E402
from neuroim import plotting as P  # noqa: E402
from neuroim.orthogonal_slices import (  # noqa: E402
    extract_axial_slice,
    get_world_bounds_for_slice,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _pair():
    shape = (20, 20, 20)
    sp = NeuroSpace(shape)
    bg = DenseNeuroVol(np.ones(shape) * 100.0, sp)
    ov = np.zeros(shape)
    ov[5, 5, 3] = 3.5
    ov[10, 10, 15] = 18.0
    ov[11, 10, 15] = 9.0
    return bg, DenseNeuroVol(ov, sp)


def test_overlay_scale_independent_of_displayed_slices():
    bg, ov = _pair()
    limits = []
    for z in ([3], [15], [3, 15]):
        fig, axes = P.plot_overlay(bg, ov, threshold=2.5, zlevels=z, colorbar=True)
        cbar_ax = fig.get_axes()[-1]
        limits.append(cbar_ax.get_ylim())
    assert limits[0] == limits[1] == limits[2]


def test_overlay_scale_contains_threshold_and_starts_at_zero():
    lim, cap, signed, symmetric = P._overlay_scale(
        np.r_[np.zeros(50), np.linspace(3.5, 11.8, 40)], threshold=3.0
    )
    assert lim[0] == 0.0 and lim[1] >= 11.8
    assert not signed and not symmetric
    lim, cap, *_ = P._overlay_scale(np.r_[np.zeros(10), [1.0, 1.5]], threshold=3.0)
    assert lim[1] > 3.0  # extended so the threshold is visible


@pytest.mark.parametrize("mode", ["ramp", "proportional", "soft"])
def test_graded_alpha_floor_for_supra_threshold(mode):
    vals = np.array([0.0, 2.9, 3.01, 4.0, 12.0])
    alpha = P._alpha_for_overlay(vals, mode=mode, threshold=3.0, cap=12.0)
    assert alpha[0] == 0 and alpha[1] == 0
    assert np.all(alpha[2:] >= 0.6 - 1e-12)
    assert alpha[-1] == pytest.approx(1.0)


def test_binary_alpha_hides_zeros_without_threshold():
    alpha = P._alpha_for_overlay(
        np.array([0.0, 1.0]), mode="binary", threshold=None, cap=1
    )
    assert list(alpha) == [0.0, 1.0]


def test_soft_alpha_curve_shared_across_panels():
    shape = (12, 12, 12)
    sp = NeuroSpace(shape)
    ov = np.zeros(shape)
    ov[..., 3] = np.linspace(0.1, 1.0, 144).reshape(12, 12)
    ov[..., 9] = np.linspace(1.0, 10.0, 144).reshape(12, 12)
    ov[0, 0, 3] = ov[0, 0, 9] = 0.9
    bg = DenseNeuroVol(np.ones(shape), sp)
    fig, axes = P.plot_overlay(
        bg, DenseNeuroVol(ov, sp), zlevels=[3, 9], alpha_mode="soft"
    )
    # Overlays are RGBA images; the alpha channel holds per-voxel opacity.
    a3, a9 = (np.asarray(ax.images[-1].get_array())[..., 3] for ax in axes)
    # The same value gets the same opacity in both panels.
    v3 = P._display_slice(DenseNeuroVol(ov, sp), axis=2, index=3).data
    v9 = P._display_slice(DenseNeuroVol(ov, sp), axis=2, index=9).data
    assert a3[np.isclose(v3, 0.9)][0] == pytest.approx(a9[np.isclose(v9, 0.9)][0])


def test_background_window_does_not_clip_skull_stripped_brain():
    shape = (40, 40, 40)
    g = np.indices(shape)
    r = np.sqrt(((g - 20) ** 2).sum(0))
    rng = np.random.default_rng(0)
    data = np.where(r < 9, 600 + rng.normal(0, 30, shape), 0.0)
    lo, hi = P._background_display_limits(data.ravel())
    brain = data[r < 9]
    assert np.mean(brain > hi) < 0.01


def test_plot_neuro_vol_thresh_hides_inside_range():
    data = np.arange(20.0).reshape(20, 1, 1) * np.ones((1, 4, 4))
    vol = DenseNeuroVol(data, NeuroSpace(data.shape))
    fig = P.plot_neuro_vol(vol, zlevels=[0], thresh=(0, 10), axis=1, crop=False)
    img = [im for ax in fig.get_axes() for im in ax.images][-1]
    shown = np.ma.compressed(img.get_array())
    assert shown.min() > 10


def _asr_vol():
    a = np.zeros((4, 4))
    a[3, 3] = 1
    a[0, 2] = a[1, 0] = a[2, 1] = 1  # native axes run A, S, R
    shape = (6, 7, 8)
    return DenseNeuroVol(
        np.random.default_rng(1).random(shape), NeuroSpace(shape, trans=a)
    )


def test_plane_names_follow_affine():
    vol = _asr_vol()
    axial = P._normalize_axis("axial", vol=vol)
    assert P._PLANE_BY_AXIS[int(P._spatial_orientation(vol.space)[axial, 0])] == "Axial"
    assert P._normalize_axis(2, vol=vol) == 2  # integers stay native
    s = extract_axial_slice(vol, np.array([1.0, 2.0, 3.0]))
    assert s.shape == (6, 8)
    fig, axes = P.plot_montage(vol, axis="axial", zlevels=[2])
    assert axes[0].get_title() == "z = 2 mm"


def test_world_bounds_for_slice_uses_plane_axis():
    vol = _asr_vol()
    lo, hi = get_world_bounds_for_slice(vol, "axial", slice_index=2)
    assert lo[2] == hi[2] == 2.0
    assert np.all(lo <= hi)

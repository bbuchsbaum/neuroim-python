"""Filter contracts carried over from neuroim2 (#20, #22, 43bcbcb).

Masked Gaussian smoothing is insulated to the mask by default, and the 3-D
bilateral filter uses millimetre distances, in-mask neighbours only and an
intensity bandwidth of ``intensity_sigma * range_scale``.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroim import DenseNeuroVec, DenseNeuroVol, LogicalNeuroVol, NeuroSpace
from neuroim.spatial_filters import (
    bilateral_filter,
    bilateral_filter_vec,
    gaussian_blur,
)


def _half_mask(shape=(10, 10, 10)):
    m = np.zeros(shape, dtype=bool)
    m[2:8, 2:8, 2:5] = True
    return m


def test_masked_gaussian_ignores_nan_outside_mask():
    sp = NeuroSpace((10, 10, 10), spacing=(2.0, 2.0, 2.0))
    m = _half_mask()
    data = np.where(m, 1.0, np.nan)
    out = gaussian_blur(DenseNeuroVol(data, sp), mask=LogicalNeuroVol(m, sp), sigma=1)
    assert not np.isnan(out.data[m]).any()
    np.testing.assert_allclose(out.data[m], 1.0)  # constant stays constant
    assert np.isnan(out.data[~m]).all()  # outside the mask is untouched


def test_masked_gaussian_4d_is_insulated_per_volume():
    sp = NeuroSpace((10, 10, 10, 3))
    m = _half_mask()
    data = np.where(m[..., None], 5.0, 1000.0) * np.ones((1, 1, 1, 3))
    out = gaussian_blur(
        DenseNeuroVec(data, sp), mask=LogicalNeuroVol(m, NeuroSpace((10, 10, 10)))
    )
    np.testing.assert_allclose(out.data[m], 5.0)


def test_masked_gaussian_legacy_mode_kept():
    sp = NeuroSpace((10, 10, 10))
    m = _half_mask()
    data = np.where(m, 1.0, 0.0)
    out = gaussian_blur(
        DenseNeuroVol(data, sp), mask=LogicalNeuroVol(m, sp), sigma=1, normalize=False
    )
    assert out.data[m].min() < 0.9  # zeros outside the mask bleed in


def _oracle(d, m, ss, isg, w, sp):
    out = d.copy()
    n = d.shape
    for i, j, k in np.argwhere(m):
        num = den = 0.0
        for a in range(-w, w + 1):
            for b in range(-w, w + 1):
                for c in range(-w, w + 1):
                    x, y, z = i + a, j + b, k + c
                    inside = 0 <= x < n[0] and 0 <= y < n[1] and 0 <= z < n[2]
                    if not inside or not m[x, y, z]:
                        continue
                    dist2 = (a * sp[0]) ** 2 + (b * sp[1]) ** 2 + (c * sp[2]) ** 2
                    wt = np.exp(-dist2 / (2 * ss**2))
                    wt *= np.exp(-((d[x, y, z] - d[i, j, k]) ** 2) / (2 * isg**2))
                    num += wt * d[x, y, z]
                    den += wt
        out[i, j, k] = num / den
    return out


@pytest.mark.parametrize("spacing,window", [((1, 1, 1), 1), ((1.5, 2, 3), 2)])
def test_bilateral_matches_bruteforce(spacing, window):
    d = np.random.default_rng(3).normal(size=(6, 6, 6))
    m = np.ones(d.shape, dtype=bool)
    m[0, :, :2] = False
    sp = NeuroSpace(d.shape, spacing=spacing)
    out = bilateral_filter(
        DenseNeuroVol(d, sp),
        mask=LogicalNeuroVol(m, sp),
        spatial_sigma=2.0,
        intensity_sigma=0.7,
        window=window,
        range_scale=1.0,
    )
    np.testing.assert_allclose(out.data, _oracle(d, m, 2.0, 0.7, window, spacing))


def test_bilateral_excludes_out_of_mask_neighbours():
    sp = NeuroSpace((6, 6, 6))
    m = np.zeros((6, 6, 6), dtype=bool)
    m[:3] = True
    out = bilateral_filter(
        DenseNeuroVol(np.where(m, 1.0, np.nan), sp),
        mask=LogicalNeuroVol(m, sp),
        intensity_sigma=50,
    )
    np.testing.assert_allclose(out.data[m], 1.0)


def test_bilateral_auto_range_scale_is_scale_equivariant():
    d = np.random.default_rng(4).normal(size=(5, 5, 5))
    sp = NeuroSpace(d.shape)
    a = bilateral_filter(DenseNeuroVol(d, sp)).data
    b = bilateral_filter(DenseNeuroVol(d * 100.0, sp)).data
    np.testing.assert_allclose(b, a * 100.0)


def test_bilateral_vec_fixed_range_scale_and_validation():
    d = np.random.default_rng(5).normal(size=(4, 4, 4, 2))
    vec = DenseNeuroVec(d, NeuroSpace(d.shape))
    out = bilateral_filter_vec(vec, range_scale=2.0)
    first = bilateral_filter(
        DenseNeuroVol(d[..., 0], NeuroSpace(d.shape[:3])), range_scale=2.0
    )
    np.testing.assert_allclose(out.data[..., 0], first.data)
    with pytest.raises(ValueError, match="range_scale"):
        bilateral_filter_vec(vec, range_scale=0)


# --- sigma is in millimetres (neuroim2 contract) -----------------------------


def _delivered_sd_mm(spacing, **kwargs):
    shape = (41, 41, 41)
    delta = np.zeros(shape)
    delta[20, 20, 20] = 1.0
    out = gaussian_blur(
        DenseNeuroVol(delta, NeuroSpace(shape, spacing=spacing)), **kwargs
    )
    sds = []
    for axis in range(3):
        marginal = out.data.sum(axis=tuple(a for a in range(3) if a != axis))
        grid = np.arange(marginal.size)
        w = marginal / marginal.sum()
        mean = (grid * w).sum()
        sds.append(np.sqrt(((grid - mean) ** 2 * w).sum()) * spacing[axis])
    return np.array(sds)


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (2.0, 2.0, 3.5)])
def test_sigma_is_millimetres(spacing):
    np.testing.assert_allclose(_delivered_sd_mm(spacing, sigma=3.0), 3.0, rtol=0.02)
    # The default is 2 mm (sub-voxel on 3.5 mm voxels, so sampling costs ~2%).
    np.testing.assert_allclose(_delivered_sd_mm(spacing), 2.0, rtol=0.03)


def test_sigma_and_fwhm_are_exclusive():
    vol = DenseNeuroVol(np.zeros((5, 5, 5)), NeuroSpace((5, 5, 5)))
    with pytest.raises(ValueError, match="not both"):
        gaussian_blur(vol, sigma=2.0, fwhm_mm=6.0)

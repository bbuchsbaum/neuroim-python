"""Impulse-response contracts for Gaussian smoothing support and units."""

from __future__ import annotations

import numpy as np
import pytest

import neuroim as ni
import neuroim.spatial_filters as spatial_filters

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _impulse(shape, spacing=(1.0, 1.0, 1.0)):
    data = np.zeros(shape, dtype=float)
    center = tuple(s // 2 for s in shape)
    data[center] = 1.0
    return ni.DenseNeuroVol(data, ni.NeuroSpace(shape, spacing=spacing)), center


def _profile_fwhm(data, center, axis, spacing=1.0):
    selector = list(center)
    selector[axis] = slice(None)
    profile = np.array(data[tuple(selector)], dtype=float, copy=True)
    profile /= profile.sum()
    offsets = (np.arange(profile.size) - center[axis]) * spacing
    sigma = np.sqrt(np.sum(profile * offsets**2))
    return float(FWHM_PER_SIGMA * sigma)


def test_default_legacy_sigma_delivers_requested_voxel_fwhm():
    vol, center = _impulse((65, 65, 65))

    observed = ni.gaussian_blur(vol, sigma=2.0)

    delivered = _profile_fwhm(observed.data, center, axis=0)
    assert delivered == pytest.approx(2.0 * FWHM_PER_SIGMA, abs=0.01)


def test_explicit_window_remains_a_voxel_half_width():
    vol, center = _impulse((33, 33, 33))

    observed = ni.gaussian_blur(vol, sigma=2.0, window=1)

    profile = observed.data[:, center[1], center[2]]
    support = np.flatnonzero(profile > 0)
    np.testing.assert_array_equal(support, [center[0] - 1, center[0], center[0] + 1])


def test_fwhm_mm_is_isotropic_on_anisotropic_voxels():
    spacing = (1.0, 2.0, 4.0)
    vol, center = _impulse((65, 35, 19), spacing=spacing)

    observed = ni.gaussian_blur(vol, fwhm_mm=8.0)

    delivered = [
        _profile_fwhm(observed.data, center, axis, spacing[axis]) for axis in range(3)
    ]
    assert delivered == pytest.approx([8.0, 8.0, 8.0], abs=0.12)


def test_4d_smoothing_uses_scipy_110_signature_and_never_smooths_time(
    monkeypatch,
):
    shape = (17, 17, 17, 2)
    data = np.zeros(shape, dtype=float)
    data[8, 8, 8, 0] = 1.0
    data[8, 8, 8, 1] = 10.0
    vec = ni.DenseNeuroVec(data, ni.NeuroSpace(shape))
    scipy_filter = spatial_filters.ndimage.gaussian_filter

    def scipy_110_filter(
        input,
        sigma,
        order=0,
        output=None,
        mode="reflect",
        cval=0.0,
        truncate=4.0,
        *,
        radius=None,
    ):
        return scipy_filter(
            input,
            sigma,
            order=order,
            output=output,
            mode=mode,
            cval=cval,
            truncate=truncate,
            radius=radius,
        )

    monkeypatch.setattr(spatial_filters.ndimage, "gaussian_filter", scipy_110_filter)

    observed = ni.gaussian_blur(vec, sigma=1.5, window=3)

    np.testing.assert_allclose(observed.data.sum(axis=(0, 1, 2)), [1.0, 10.0])
    np.testing.assert_allclose(observed.data[..., 1], 10.0 * observed.data[..., 0])


@pytest.mark.parametrize(
    "name,value",
    [("sigma", np.nan), ("sigma", np.inf), ("fwhm_mm", np.nan), ("fwhm_mm", np.inf)],
)
def test_nonfinite_width_parameters_are_rejected(name, value):
    vol, _ = _impulse((9, 9, 9))

    with pytest.raises(ValueError, match=name):
        ni.gaussian_blur(vol, **{name: value})


@pytest.mark.parametrize("window", [0, -1, 1.5, np.nan, np.inf, [2]])
def test_window_must_be_one_positive_integer_or_none(window):
    vol, _ = _impulse((9, 9, 9))

    with pytest.raises(ValueError, match="window"):
        ni.gaussian_blur(vol, window=window)


@pytest.mark.parametrize("truncate", [0, -1, np.nan, np.inf, [4]])
def test_truncate_must_be_one_positive_finite_scalar(truncate):
    vol, _ = _impulse((9, 9, 9))

    with pytest.raises(ValueError, match="truncate"):
        ni.gaussian_blur(vol, truncate=truncate)

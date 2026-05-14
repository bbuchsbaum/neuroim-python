"""Sanity tests for the realistic BOLD fixture (ME-8)."""

import numpy as np
import pytest

from fixtures.realistic_bold import RealisticBOLD, make_realistic_bold, to_nibabel


def test_default_shape_and_dtype():
    fx = make_realistic_bold()
    assert isinstance(fx, RealisticBOLD)
    assert np.asarray(fx.bold.data).shape == (32, 32, 24, 40)
    assert np.asarray(fx.bold.data).dtype == np.float64
    assert np.asarray(fx.mask.data).shape == (32, 32, 24)
    assert fx.regressor.shape == (40,)
    assert fx.target_roi_centers.shape == (27, 3)


def test_mask_is_brain_shaped_and_covers_meaningful_volume():
    fx = make_realistic_bold()
    mask = np.asarray(fx.mask.data)
    coverage = int(mask.sum())
    total = int(np.prod(mask.shape))
    assert 0.2 * total < coverage < 0.7 * total, f"mask covers {coverage}/{total}"


def test_fixture_is_deterministic_for_same_seed():
    a = make_realistic_bold(seed=42)
    b = make_realistic_bold(seed=42)
    np.testing.assert_array_equal(np.asarray(a.bold.data), np.asarray(b.bold.data))
    np.testing.assert_array_equal(np.asarray(a.mask.data), np.asarray(b.mask.data))
    np.testing.assert_array_equal(a.regressor, b.regressor)


def test_seed_changes_data_but_not_geometry():
    a = make_realistic_bold(seed=1)
    b = make_realistic_bold(seed=2)
    assert not np.array_equal(np.asarray(a.bold.data), np.asarray(b.bold.data))
    np.testing.assert_array_equal(np.asarray(a.mask.data), np.asarray(b.mask.data))


def test_rotated_mask_has_flipped_affine_but_same_data():
    fx = make_realistic_bold()
    aff = fx.mask.space.trans
    rot = fx.rotated_mask.space.trans
    np.testing.assert_allclose(rot[:, 0], -aff[:, 0])
    np.testing.assert_array_equal(
        np.asarray(fx.mask.data),
        np.asarray(fx.rotated_mask.data),
    )


def test_target_roi_carries_signal_correlating_with_regressor():
    fx = make_realistic_bold()
    coords = fx.target_roi_centers
    data = np.asarray(fx.bold.data)
    target_ts = data[coords[:, 0], coords[:, 1], coords[:, 2], :].mean(axis=0)
    target_ts = target_ts - target_ts.mean()
    norm = float(np.linalg.norm(target_ts)) or 1.0
    target_ts /= norm
    r = float(np.dot(target_ts, fx.regressor))
    assert r > 0.5, f"expected r>0.5 in target ROI, got {r:.3f}"


def test_background_voxel_has_weak_correlation_with_regressor():
    fx = make_realistic_bold()
    data = np.asarray(fx.bold.data)
    bg_ts = data[2, 2, 2, :]
    bg_ts = bg_ts - bg_ts.mean()
    norm = float(np.linalg.norm(bg_ts)) or 1.0
    bg_ts /= norm
    r = float(np.dot(bg_ts, fx.regressor))
    assert abs(r) < 0.5, f"background should have weak correlation, got |r|={abs(r):.3f}"


def test_to_nibabel_round_trip():
    fx = make_realistic_bold()
    bold_img, mask_img = to_nibabel(fx)
    assert bold_img.shape == np.asarray(fx.bold.data).shape
    assert mask_img.shape == np.asarray(fx.mask.data).shape
    np.testing.assert_array_equal(bold_img.get_fdata(), np.asarray(fx.bold.data))
    np.testing.assert_array_equal(
        mask_img.get_fdata().astype(bool),
        np.asarray(fx.mask.data),
    )

"""Acceptance test for Scenario 12 -- Gaussian spatial smoothing at FWHM = 6 mm.

Six assertions (all passing after PAIN-1/2/3/4 land):

1. **Numeric parity (happy path)** -- baseline (scipy + manual FWHM->voxel
   conversion) and the neuroim rewrite (``gaussian_blur(vol, fwhm_mm=6)``)
   agree to machine precision on the matched-space fixture.

2. **PAIN-1 closed -- same-space gate** -- an LR-flipped mask raises
   ``ValueError("spatial contract mismatch")`` from the
   ``gaussian_blur`` call site, not from a manual helper.

3. **Anisotropy attack on the legacy voxel-sigma path** -- using
   ``gaussian_blur(vol, sigma=2)`` on the 3 x 3 x 3.5 mm fixture produces
   *anisotropic* mm-space smoothing, proving the unit pitfall is real
   and motivating ``fwhm_mm=`` as the recommended path.

4. **PAIN-2 closed -- isotropic-mm-space smoothing via fwhm_mm** -- on the
   same anisotropic fixture, ``gaussian_blur(vol, fwhm_mm=6)`` produces
   per-axis FWHM that match in mm units (within the discretisation
   tolerance), confirming the spacing-aware conversion does its job.

5. **PAIN-3 closed -- provenance Receipt** -- the smoothed
   ``DenseNeuroVol`` carries ``.provenance`` with
   ``method_name == "gaussian_blur"`` and ``radius == fwhm_mm``.

6. **PAIN-4 closed -- public namespace** -- ``gaussian_blur`` is in
   ``neuroim.__all__``.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.results import Receipt

from fixtures.realistic_bold import make_realistic_bold


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "scenarios"
    / "12_gaussian_smoothing"
)
baseline = _load_module("scenario12_baseline", _SCENARIO_DIR / "baseline_nibabel.py")
rewrite = _load_module("scenario12_rewrite", _SCENARIO_DIR / "neuroim_version.py")


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


@pytest.fixture(scope="module")
def stat_vol(fixture):
    """3-D stat map (t=0 slice of the BOLD) on the fixture's spatial space."""
    arr = np.ascontiguousarray(fixture.bold.data[..., 0], dtype=np.float64)
    return ni.DenseNeuroVol(arr, fixture.mask.space)


@pytest.fixture(scope="module")
def stat_nifti(stat_vol):
    affine = np.asarray(stat_vol.space.trans, dtype=float)[:4, :4]
    return nib.Nifti1Image(np.asarray(stat_vol.data, dtype=np.float64), affine)


def test_baseline_and_neuroim_agree_at_fwhm_6mm(stat_vol, stat_nifti):
    """Both paths produce the same array at FWHM = 6 mm."""
    base = baseline.smooth_fwhm_mm(stat_nifti, fwhm_mm=6.0)
    rew = rewrite.smooth_fwhm_mm(stat_vol, fwhm_mm=6.0)
    np.testing.assert_allclose(
        np.asarray(base.dataobj),
        np.asarray(rew.data),
        rtol=1e-10,
        atol=1e-10,
    )


def test_neuroim_rejects_mismatched_affine_mask(stat_vol, fixture):
    """PAIN-1 closed: the gate lives inside ``gaussian_blur`` itself."""
    rotated_mask = fixture.rotated_mask
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        rewrite.smooth_fwhm_mm(stat_vol, fwhm_mm=6.0, mask=rotated_mask)


def _axis_spread_voxels(arr: np.ndarray, axis: int) -> float:
    """Standard deviation of ``arr`` along ``axis`` after marginal-summing
    over the other axes.  Reports spread in voxel units."""
    other_axes = tuple(i for i in range(arr.ndim) if i != axis)
    marginal = arr.sum(axis=other_axes)
    grid = np.arange(marginal.size, dtype=float)
    weight = marginal / marginal.sum()
    center = float(np.sum(grid * weight))
    var = float(np.sum((grid - center) ** 2 * weight))
    return math.sqrt(var)


def test_legacy_voxel_sigma_is_anisotropic_in_mm_on_anisotropic_voxels(fixture):
    """PAIN-2 motivation: scalar voxel-sigma silently produces anisotropic
    mm-space smoothing on a 3 x 3 x 3.5 mm fixture.

    Smooth a centred delta with ``sigma=2`` voxels.  The output is
    voxel-isotropic by construction, but the *mm-space* spread differs
    along z because z's voxel size differs from x/y.  The ratio of the
    z-spread to the x-spread should track the spacing ratio.
    """
    spacing = np.asarray(fixture.bold.space.spacing[:3], dtype=float)
    assert spacing[2] != spacing[0], (
        f"fixture is not anisotropic: spacing={spacing!r}"
    )

    nx, ny, nz = 32, 32, 24
    delta = np.zeros((nx, ny, nz), dtype=np.float64)
    delta[nx // 2, ny // 2, nz // 2] = 1.0
    vol = ni.DenseNeuroVol(delta, fixture.mask.space)

    smoothed = np.asarray(rewrite.smooth_via_voxel_sigma(vol, 2.0).data)

    spread_x_vox = _axis_spread_voxels(smoothed, 0)
    spread_z_vox = _axis_spread_voxels(smoothed, 2)
    # Voxel-isotropic, as expected for a scalar sigma.
    np.testing.assert_allclose(spread_x_vox, spread_z_vox, rtol=0.05)

    # mm-space spread differs by the voxel-size ratio: this is the bug.
    spread_x_mm = spread_x_vox * spacing[0]
    spread_z_mm = spread_z_vox * spacing[2]
    expected_ratio = spacing[2] / spacing[0]
    actual_ratio = spread_z_mm / spread_x_mm
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.05)
    # Sanity: the two spreads really do disagree in mm.
    assert not math.isclose(spread_x_mm, spread_z_mm, rel_tol=0.05)


def test_fwhm_mm_path_is_isotropic_in_mm_on_anisotropic_voxels(fixture):
    """PAIN-2 closed: ``fwhm_mm=`` produces mm-isotropic smoothing.

    Same fixture as above, but now using ``fwhm_mm=`` so the per-axis
    voxel-sigma is derived from ``vol.space.spacing``.  mm-space spread
    must match across axes within discretisation tolerance.
    """
    spacing = np.asarray(fixture.bold.space.spacing[:3], dtype=float)
    nx, ny, nz = 32, 32, 24
    delta = np.zeros((nx, ny, nz), dtype=np.float64)
    delta[nx // 2, ny // 2, nz // 2] = 1.0
    vol = ni.DenseNeuroVol(delta, fixture.mask.space)

    smoothed = np.asarray(
        rewrite.smooth_fwhm_mm(vol, fwhm_mm=6.0).data
    )
    spread_x_mm = _axis_spread_voxels(smoothed, 0) * spacing[0]
    spread_z_mm = _axis_spread_voxels(smoothed, 2) * spacing[2]
    # Allow a small tolerance for the integer-voxel discretisation of the
    # Gaussian on the smaller z extent (24 voxels at 3.5 mm).
    np.testing.assert_allclose(spread_x_mm, spread_z_mm, rtol=0.05)


def test_smoothed_output_carries_provenance(stat_vol):
    """PAIN-3 closed: output ``DenseNeuroVol`` has ``.provenance``."""
    smoothed = ni.gaussian_blur(stat_vol, fwhm_mm=6.0)
    assert isinstance(smoothed.provenance, Receipt)
    assert smoothed.provenance.method_name == "gaussian_blur"
    assert smoothed.provenance.radius == pytest.approx(6.0)


def test_gaussian_blur_in_public_namespace():
    """PAIN-4 closed: ``gaussian_blur`` is in ``neuroim.__all__``."""
    assert "gaussian_blur" in ni.__all__

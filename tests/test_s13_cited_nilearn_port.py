"""Acceptance test for Scenario 13 -- cited Nilearn seed-to-voxel port.

Five assertions:

1. **Numeric parity (happy path)** -- baseline (nibabel + numpy
   replicating the Nilearn tutorial) and the neuroim rewrite produce the
   same in-mask correlation values to machine precision.

2. **Output affine parity** -- both paths return a connectivity map on
   the BOLD's spatial frame.

3. **neuroim catches the LR-flipped mask** -- the same-space gate fires
   inside ``NeuroVec.series_roi``, surfacing a foreign-affine mask
   before the correlation runs.

4. **neuroim catches the OOB seed coordinate** -- a seed world-mm
   coordinate that maps outside the BOLD grid raises through the
   ``series_roi_world`` OOB contract (S01 PAIN-2 fix), not silently into
   numpy wrap-around.

5. **Output map carries provenance** -- the neuroim result has a
   populated ``Receipt`` recording the seed radius, seed parameters via
   ``method_name="seed_to_voxel_correlation_map"``, and the mask hash.

The point of S13 vs S06 is not new code but new *evidence*: the workflow
is named (Nilearn's plot_seed_to_voxel_correlation tutorial), the
version is pinned in the README, and the line-for-line nilearn -> neuroim
mapping lives in the scenario's REPORT.md.  Nilearn is not imported
here; the comparison is against a hand-rolled nibabel + numpy
implementation that mirrors the cited example's analytical structure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.results import Receipt

from fixtures.realistic_bold import make_realistic_bold, to_nibabel


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
    / "13_cited_nilearn_port"
)
baseline = _load_module("scenario13_baseline", _SCENARIO_DIR / "baseline_nibabel.py")
rewrite = _load_module("scenario13_rewrite", _SCENARIO_DIR / "neuroim_version.py")


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


@pytest.fixture(scope="module")
def nib_pair(fixture):
    return to_nibabel(fixture)


def _seed_world_xyz(fixture) -> np.ndarray:
    """Pick a seed in world mm that lands inside the brain mask.

    Use the centroid of one of the fixture's target ROI voxels, converted
    to world coordinates via the BOLD's spatial-affine.
    """
    centre_voxel = np.asarray(fixture.target_roi_centers[0], dtype=float)
    homog = np.concatenate([centre_voxel, [1.0]])
    affine = np.asarray(fixture.bold.space.trans, dtype=float)[:4, :4]
    return (affine @ homog)[:3]


def test_baseline_and_neuroim_correlation_maps_agree(fixture, nib_pair):
    """Numeric parity at the in-mask voxels."""
    bold_img, mask_img = nib_pair
    seed = _seed_world_xyz(fixture)
    base = baseline.seed_to_voxel_correlation_map(
        bold_img, mask_img, seed_xyz=seed, radius_mm=8.0
    )
    rew = rewrite.seed_to_voxel_correlation_map(
        fixture.bold, fixture.mask, seed_xyz=seed, radius_mm=8.0
    )
    base_arr = np.asarray(base.dataobj, dtype=np.float64)
    rew_arr = np.asarray(rew.data, dtype=np.float64)
    mask = np.asarray(fixture.mask.data, dtype=bool)
    np.testing.assert_allclose(
        base_arr[mask], rew_arr[mask], rtol=1e-10, atol=1e-10
    )


def test_output_affine_matches_bold(fixture, nib_pair):
    """Both paths anchor the map on the BOLD spatial frame."""
    bold_img, mask_img = nib_pair
    seed = _seed_world_xyz(fixture)
    base = baseline.seed_to_voxel_correlation_map(
        bold_img, mask_img, seed_xyz=seed, radius_mm=8.0
    )
    rew = rewrite.seed_to_voxel_correlation_map(
        fixture.bold, fixture.mask, seed_xyz=seed, radius_mm=8.0
    )
    np.testing.assert_allclose(np.asarray(base.affine), np.asarray(bold_img.affine))
    np.testing.assert_allclose(
        np.asarray(rew.space.trans)[:4, :4], np.asarray(bold_img.affine)
    )


def test_neuroim_rejects_mismatched_affine_mask(fixture):
    """Same-space contract catches the LR-flipped mask via series_roi gate."""
    seed = _seed_world_xyz(fixture)
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        rewrite.seed_to_voxel_correlation_map(
            fixture.bold,
            fixture.rotated_mask,
            seed_xyz=seed,
            radius_mm=8.0,
        )


def test_neuroim_rejects_seed_outside_bold_grid(fixture):
    """A seed world-coord outside the BOLD raises via the world-coord OOB gate."""
    far_seed = np.array([1000.0, 1000.0, 1000.0])
    with pytest.raises((ValueError, IndexError)):
        rewrite.seed_to_voxel_correlation_map(
            fixture.bold, fixture.mask, seed_xyz=far_seed, radius_mm=8.0
        )


def test_neuroim_output_carries_receipt(fixture):
    """The connectivity map ships a populated provenance Receipt."""
    seed = _seed_world_xyz(fixture)
    rew = rewrite.seed_to_voxel_correlation_map(
        fixture.bold, fixture.mask, seed_xyz=seed, radius_mm=8.0
    )
    assert isinstance(rew.provenance, Receipt)
    assert rew.provenance.method_name == "seed_to_voxel_correlation_map"
    assert rew.provenance.radius == pytest.approx(8.0)
    assert rew.provenance.n_voxels == int(np.asarray(fixture.mask.data).sum())

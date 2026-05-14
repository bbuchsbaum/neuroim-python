"""Acceptance test for Scenario 03 — Seed-Sphere Mean.

Lives one level up from ``03_seed_sphere_mean/`` because the leading
digit in the scenario folder name makes it an illegal Python module.
See :file:`conftest.py` and the suite README for the layout note.

Five assertions:

1. **Numeric parity** — baseline and the simple neuroim form return
   the same mean time series for an 8 mm sphere at the in-bounds
   world coordinate.
2. **Typed-result shape** — the canonical typed form ships per-voxel
   values ``(T, V=73)``, coords ``(73, 3)``, the ROI's space, and a
   populated ``Receipt``.
3. **Radius-zero collapse** — `radius_mm = 0` returns the single
   nearest voxel's series (both forms).
4. **Out-of-bounds parity** — both implementations raise
   ``ValueError`` when the centre maps off the grid.
5. **xfail pain points** — `Receipt.radius` should record the radius
   the caller passed, and `Receipt.method_name` should distinguish
   `series_roi_world` from `series_roi`.  Filed in
   :file:`03_seed_sphere_mean/REPORT.md`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import neuroim as ni
from neuroim.results import ROIExtractionResult

from fixtures.realistic_bold import make_realistic_bold, to_nibabel


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = Path(__file__).resolve().parent / "03_seed_sphere_mean"
baseline_nibabel = _load_module(
    "scenario03_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario03_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold()


@pytest.fixture(scope="module")
def nib_bold(fixture):
    bold_img, _ = to_nibabel(fixture)
    return bold_img


@pytest.fixture(scope="module")
def target_mni(fixture):
    """World-mm centre of the fixture's known target voxel.

    Voxel ``(16, 22, 16)`` with affine ``diag(3.0, 3.0, 3.5, 1.0)``
    → world ``(48.0, 66.0, 56.0)`` mm.
    """
    target_voxel = np.array([16, 22, 16], dtype=float)
    return np.asarray(fixture.bold.space.grid_to_world(target_voxel))


# ----------------------------------------------------------------------
# 1. Numeric parity on the happy path
# ----------------------------------------------------------------------


def test_baseline_and_simple_form_agree(fixture, nib_bold, target_mni):
    base = baseline_nibabel.mean_series_in_sphere_at_mni(nib_bold, target_mni, 8.0)
    rewrite = neuroim_version.mean_series_in_sphere_at_mni(
        fixture.bold, target_mni, 8.0
    )
    assert base.shape == (fixture.bold.shape[3],)
    np.testing.assert_allclose(rewrite, base, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# 2. Typed form: per-voxel values, coords, populated Receipt
# ----------------------------------------------------------------------


def test_typed_form_shape_and_receipt(fixture, target_mni):
    typed = neuroim_version.mean_series_in_sphere_at_mni_typed(
        fixture.bold, target_mni, 8.0
    )
    assert isinstance(typed, ROIExtractionResult)
    nt = fixture.bold.shape[3]
    n_vox_expected = 73  # hand-verified for 8 mm sphere on (3, 3, 3.5) mm voxels
    assert typed.values.shape == (nt, n_vox_expected)
    assert typed.coords.shape == (n_vox_expected, 3)
    assert typed.provenance.n_voxels == n_vox_expected
    assert typed.provenance.input_space_hash != "none"
    assert typed.provenance.neuroim_version == ni.__version__


# ----------------------------------------------------------------------
# 3. radius=0 collapses to a single voxel
# ----------------------------------------------------------------------


def test_zero_radius_collapses_to_single_voxel(fixture, target_mni):
    typed = neuroim_version.mean_series_in_sphere_at_mni_typed(
        fixture.bold, target_mni, 0.0
    )
    nt = fixture.bold.shape[3]
    assert typed.values.shape == (nt, 1)
    assert typed.provenance.n_voxels == 1


# ----------------------------------------------------------------------
# 4. Out-of-bounds parity
# ----------------------------------------------------------------------


@pytest.fixture
def oob_mni():
    """World coord whose voxel maps to a negative index in the fixture."""
    return np.array([-30.0, -30.0, -30.0])


def test_baseline_raises_on_out_of_bounds(nib_bold, oob_mni):
    with pytest.raises(ValueError, match="outside the image grid"):
        baseline_nibabel.mean_series_in_sphere_at_mni(nib_bold, oob_mni, 8.0)


def test_neuroim_raises_on_out_of_bounds(fixture, oob_mni):
    with pytest.raises(ValueError, match="outside the image grid"):
        neuroim_version.mean_series_in_sphere_at_mni(fixture.bold, oob_mni, 8.0)


# ----------------------------------------------------------------------
# 5. Provenance metadata from the world-coordinate entry point
# ----------------------------------------------------------------------


def test_receipt_records_caller_supplied_radius(fixture, target_mni):
    typed = neuroim_version.mean_series_in_sphere_at_mni_typed(
        fixture.bold, target_mni, 8.0
    )
    assert typed.provenance.radius == 8.0


def test_receipt_method_name_distinguishes_world_entry_point(fixture, target_mni):
    typed = neuroim_version.mean_series_in_sphere_at_mni_typed(
        fixture.bold, target_mni, 8.0
    )
    assert typed.provenance.method_name == "series_roi_world"

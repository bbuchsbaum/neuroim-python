"""Acceptance test for Scenario 01 — MNI spotlight.

Lives here (one level up from ``01_mni_spotlight/``) because the leading
digit in the scenario folder name makes it an illegal Python module —
pytest can't collect tests inside such a folder under the default
``prepend`` import mode.  The two implementations are loaded by path
via :mod:`importlib` so the scenario folder stays self-contained.

Three assertions:

1. **Numeric parity** — baseline and the simple neuroim form return the
   same time series at the in-bounds world coordinate.
2. **Typed-result parity** — the canonical typed form matches the
   baseline numerically *and* ships a populated Receipt.
3. **Out-of-bounds parity** — every implementation raises
   ``ValueError`` on a world coordinate that maps off the grid.  The
   neuroim rewrite gets this from ``series_at_world`` rather than a
   user-written bounds check.
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


_SCENARIO_DIR = Path(__file__).resolve().parent / "01_mni_spotlight"
baseline_nibabel = _load_module(
    "scenario01_baseline_nibabel", _SCENARIO_DIR / "baseline_nibabel.py"
)
neuroim_version = _load_module(
    "scenario01_neuroim_version", _SCENARIO_DIR / "neuroim_version.py"
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

    Fixture target voxel ``(16, 22, 16)`` with affine
    ``diag(3.0, 3.0, 3.5, 1.0)`` → world ``(48.0, 66.0, 56.0)`` mm.

    ``bold.space`` is 4-D; spatial 3-D grid queries are accepted and
    routed through the spatial affine.
    """
    target_voxel = np.array([16, 22, 16], dtype=float)
    return np.asarray(fixture.bold.space.grid_to_world(target_voxel))


# ----------------------------------------------------------------------
# 1. Numeric parity on the happy path
# ----------------------------------------------------------------------


def test_baseline_and_simple_form_agree(fixture, nib_bold, target_mni):
    base = baseline_nibabel.series_at_mni(nib_bold, target_mni)
    rewrite = neuroim_version.series_at_mni(fixture.bold, target_mni)
    assert base.shape == (fixture.bold.shape[3],)
    np.testing.assert_allclose(rewrite, base, rtol=0, atol=0)


# ----------------------------------------------------------------------
# 2. Typed form: numeric parity + populated Receipt
# ----------------------------------------------------------------------


def test_typed_form_matches_baseline_and_carries_receipt(
    fixture, nib_bold, target_mni
):
    base = baseline_nibabel.series_at_mni(nib_bold, target_mni)
    typed = neuroim_version.series_at_mni_typed(fixture.bold, target_mni)

    assert isinstance(typed, ROIExtractionResult)
    assert typed.values.shape == (fixture.bold.shape[3], 1)
    np.testing.assert_allclose(typed.values[:, 0], base, rtol=0, atol=0)

    receipt = typed.provenance
    assert receipt.method_name == "series_roi_world"
    assert receipt.input_space_hash != "none"
    assert receipt.n_voxels == 1
    assert receipt.neuroim_version == ni.__version__


# ----------------------------------------------------------------------
# 3. Out-of-bounds parity
# ----------------------------------------------------------------------


@pytest.fixture
def oob_mni():
    """A world-mm coord that maps to a negative voxel index in the fixture.

    The fixture's affine has positive diagonals and origin at 0, so any
    sufficiently-negative world coord round-trips to a negative voxel
    index, which numpy would silently wrap.  Both implementations must
    catch that.
    """
    return np.array([-30.0, -30.0, -30.0])


def test_baseline_raises_on_out_of_bounds(nib_bold, oob_mni):
    with pytest.raises(ValueError, match="outside the image grid"):
        baseline_nibabel.series_at_mni(nib_bold, oob_mni)


def test_neuroim_simple_raises_on_out_of_bounds(fixture, oob_mni):
    with pytest.raises(ValueError, match="outside the image grid"):
        neuroim_version.series_at_mni(fixture.bold, oob_mni)


def test_neuroim_typed_raises_on_out_of_bounds(fixture, oob_mni):
    with pytest.raises(ValueError, match="outside the image grid"):
        neuroim_version.series_at_mni_typed(fixture.bold, oob_mni)

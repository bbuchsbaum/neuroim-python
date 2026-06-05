"""Acceptance test for Scenario 20 -- parcellated functional-connectivity matrix.

The deliverable is the ``N x N`` parcel-to-parcel Pearson connectome built from
per-parcel mean BOLD time series -- the canonical Nilearn
``NiftiLabelsMasker`` -> ``ConnectivityMeasure`` pipeline, one step past
Scenario 11's parcel time series.

Checks:

1. **Numeric parity (happy path)** -- baseline (nibabel + numpy) and the
   neuroim rewrite produce the same ``(N, N)`` connectome on a matched-space
   atlas + BOLD, and the matrix is a valid correlation matrix (symmetric,
   unit diagonal, in ``[-1, 1]``).

2. **Same-space gate** -- an LR-flipped atlas (same dims, different affine)
   raises ``ValueError`` through ``assert_same_space`` inside
   ``parcel_means``, before any correlation is computed.  The mission claim
   under test at the connectome boundary.

3. **Baseline silently accepts the mismatched-affine atlas** -- the same
   nibabel pipeline returns a wrong-but-plausible connectome on the flipped
   atlas.  Evidence the bug class is real without neuroim's contract layer.

4. **Provenance** -- the connectome carries the ``parcel_means`` Receipt, so
   the matrix is traceable to the extraction (and its input space) that
   produced it.

5. **Public-surface accessor parity** -- the ``(n_time, n_parcels)`` matrix
   built from ``cluster_ids`` + ``cluster_timeseries`` matches the typed
   ``ClusteredNeuroVec.ts`` payload.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.results import Receipt

from fixtures.realistic_bold import (
    make_atlas,
    make_realistic_bold,
    make_rotated_atlas,
    to_nibabel,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses defined under ``from __future__
    # import annotations`` can resolve their own module during class build.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_SCENARIO_DIR = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "scenarios"
    / "20_parcellated_connectome"
)
baseline = _load_module("scenario20_baseline", _SCENARIO_DIR / "baseline_nibabel.py")
rewrite = _load_module("scenario20_rewrite", _SCENARIO_DIR / "neuroim_version.py")


@pytest.fixture(scope="module")
def fixture():
    return make_realistic_bold(seed=20)


@pytest.fixture(scope="module")
def atlas(fixture):
    return make_atlas(fixture)


@pytest.fixture(scope="module")
def nib_bold(fixture):
    bold_img, _ = to_nibabel(fixture)
    return bold_img


def _atlas_to_nifti(atlas_vol) -> nib.Nifti1Image:
    """Project an integer-labelled DenseNeuroVol atlas to nibabel."""
    affine = np.asarray(atlas_vol.space.trans, dtype=float)[:4, :4]
    return nib.Nifti1Image(np.asarray(atlas_vol.data, dtype=np.int32), affine)


# ----------------------------------------------------------------------
# Happy path: numeric parity + valid correlation matrix
# ----------------------------------------------------------------------
def test_baseline_and_neuroim_connectomes_agree(fixture, atlas, nib_bold):
    """Same atlas + BOLD -> identical (N, N) connectome from both paths."""
    base_labels, base_conn = baseline.parcel_connectome(
        nib_bold, _atlas_to_nifti(atlas)
    )
    result = rewrite.parcel_connectome(fixture.bold, atlas)

    n = base_labels.size
    assert base_conn.shape == (n, n)
    assert result.matrix.shape == (n, n)
    np.testing.assert_array_equal(result.labels, base_labels.astype(np.int64))
    np.testing.assert_allclose(result.matrix, base_conn, rtol=1e-10, atol=1e-10)


def test_connectome_is_a_valid_correlation_matrix(fixture, atlas):
    """The connectome is symmetric, unit-diagonal, and bounded in [-1, 1]."""
    result = rewrite.parcel_connectome(fixture.bold, atlas)
    m = result.matrix
    np.testing.assert_allclose(m, m.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.diag(m), 1.0, rtol=1e-10, atol=1e-10)
    assert m.min() >= -1.0 - 1e-9 and m.max() <= 1.0 + 1e-9


# ----------------------------------------------------------------------
# Same-space gate (mission-bearing)
# ----------------------------------------------------------------------
def test_neuroim_rejects_mismatched_affine_atlas(fixture, atlas):
    """An LR-flipped atlas must raise through ``assert_same_space``.

    The gate lives inside ``parcel_means``, so it fires before any
    correlation is computed -- no wrong-but-plausible connectome escapes.
    """
    rotated = make_rotated_atlas(atlas)
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        rewrite.parcel_connectome(fixture.bold, rotated)


def test_baseline_silently_accepts_mismatched_affine_atlas(fixture, atlas, nib_bold):
    """The bug class is real: nibabel + numpy returns a plausible connectome.

    The baseline overlays atlas labels onto the BOLD *by voxel index* and
    never consults the affine, so an LR-flipped atlas (same dims, different
    affine) sails through with no error.  In fact the bare-array path is so
    affine-blind that the flipped-atlas connectome is byte-identical to the
    matched one -- the spatial-frame disagreement leaves no trace at all.

    A correct pipeline would resample the atlas into the BOLD frame using the
    two affines before overlaying; the bare-array path skips that silently.
    neuroim's ``parcel_means`` refuses the frame mismatch up front
    (``test_neuroim_rejects_mismatched_affine_atlas``).
    """
    rotated = make_rotated_atlas(atlas)
    labels, conn = baseline.parcel_connectome(nib_bold, _atlas_to_nifti(rotated))
    n = labels.size
    assert conn.shape == (n, n)  # no error raised: the mismatch is silent
    _, matched_conn = baseline.parcel_connectome(nib_bold, _atlas_to_nifti(atlas))
    # Affine-blind: the wrong-frame atlas produces the *same* numbers.
    np.testing.assert_array_equal(conn, matched_conn)


# ----------------------------------------------------------------------
# Provenance + public-surface accessor
# ----------------------------------------------------------------------
def test_connectome_carries_parcel_means_provenance(fixture, atlas):
    """The connectome inherits the extraction's typed Receipt."""
    result = rewrite.parcel_connectome(fixture.bold, atlas)
    assert isinstance(result.provenance, Receipt)
    assert result.provenance.method_name == "parcel_means"
    assert result.provenance.n_voxels == result.labels.size


def test_public_accessor_matches_typed_matrix(fixture, atlas):
    """``cluster_ids`` + ``cluster_timeseries`` reproduce ``ClusteredNeuroVec.ts``."""
    parcels = fixture.bold.parcel_means(atlas)
    public = rewrite.parcel_timeseries_matrix(parcels)  # (n_time, n_parcels)
    assert public.shape == (parcels.n_time, parcels.n_clusters)
    np.testing.assert_allclose(public, parcels.ts, rtol=1e-12, atol=1e-12)


def test_clustered_classes_are_publicly_exported():
    """The atlas-workflow classes are discoverable via the curated public API."""
    assert "ClusteredNeuroVec" in ni.__all__
    assert "ClusteredNeuroVol" in ni.__all__

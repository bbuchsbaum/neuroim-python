"""Stress test: an *ergonomic* seed-based connectivity mini-pipeline.

This test approaches ``neuroim`` the way a competent-but-new user would:
it reads only the curated public surface (``ni.*`` / what ``dir(ni)``
advertises and what the README documents) and runs a complete, natural
fMRI workflow end to end::

    4D BOLD  ->  spatial mask  ->  seed ROI  ->  seed time series
             ->  voxelwise correlation map (NeuroVol)
             ->  spatial smoothing  ->  threshold + connected components
             ->  write to NIfTI with provenance  ->  read back & verify
             ->  quick orthographic plot

The workflow test (:func:`test_seed_connectivity_workflow_completes`)
walks the whole pipeline using the *ergonomic* form of each step.

Five ergonomic snags were originally discovered while exercising this
workflow; all five have since been fixed, and each is now pinned below as
a focused **regression guard** so the ergonomic path cannot silently
regress:

PAIN-1  ``NeuroSpace(dim=<4-tuple>, spacing=<3-tuple>)`` — the natural call
        for a 4D series whose time axis has no spatial spacing — used to
        raise a cryptic numpy broadcast ``ValueError``. The constructor now
        pads trailing (e.g. temporal) axes with unit spacing / zero origin,
        and an over-long vector raises a clear ``InvalidSpaceError``.

PAIN-2  ``NeuroVec`` had no ``.mean()`` despite the README documenting
        ``mean_vol = fmri.mean(axis=3)``. ``NeuroVec.mean(axis=-1|3)`` now
        returns a provenance-carrying 3D ``DenseNeuroVol``.

PAIN-3  ``ni.gaussian_blur`` rejected a 4D ``NeuroVec``. It now smooths a
        time series spatially (no temporal blur) and returns a
        ``DenseNeuroVec`` with a ``gaussian_blur`` Receipt.

PAIN-4  ``ni.conn_comp(vol, threshold=...)`` used to crash with
        ``ModuleNotFoundError: No module named 'pandas'`` on a stock install
        (the ``cluster_table=True`` default lazily imports pandas, which is
        only a ``dev`` extra). It now degrades gracefully: with pandas the
        table is a ``DataFrame``; without pandas the call succeeds, warns,
        and leaves ``cluster_table=None``.

PAIN-5  ``ni.write_vol`` / ``ni.write_vec`` dropped provenance (wrote zero
        NIfTI extensions). Both now route through ``to_nibabel``, so a
        Receipt survives ``write_vol`` -> ``read_vol`` just as it does via
        ``to_nibabel``.
"""

from __future__ import annotations

import sys
from importlib.metadata import requires

import matplotlib
import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.exceptions import InvalidSpaceError
from neuroim.verify import receipt_of

from fixtures.realistic_bold import make_realistic_bold

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def bundle():
    """A deterministic 4D BOLD + 3D mask with a planted seed signal."""
    return make_realistic_bold(seed=0xC0FFEE)


# ----------------------------------------------------------------------
# Workflow — the full ergonomic pipeline, end to end.
# ----------------------------------------------------------------------
def test_seed_connectivity_workflow_completes(bundle, tmp_path):
    bold, mask = bundle.bold, bundle.mask

    # --- spaces line up. `spatial_space` bridges a 4D vec to its 3D frame.
    assert bold.spatial_space.compatible_with(mask.space)

    # --- seed ROI over a known active region of the fixture.
    seed_center = tuple(int(c) for c in bundle.target_roi_centers.mean(0).round())
    roi = ni.spherical_roi(mask, centroid=seed_center, radius=4.0)
    assert len(roi.coords) > 0

    # --- seed time series: typed extraction -> reduce across voxels.
    extraction = bold.series_roi(roi)
    assert extraction.values.shape[0] == bold.shape[-1]
    seed = extraction.values.mean(axis=1)  # (n_time,)
    assert seed.shape == (bold.shape[-1],)

    # --- mean-over-time volume via the first-class API (PAIN-2).
    mean_vol = bold.mean(axis=3)
    assert mean_vol.shape == mask.shape
    assert mean_vol.provenance.method_name == "mean"

    # --- voxelwise Pearson r of the seed against every in-mask voxel.
    arr = np.asarray(bold.data, dtype=np.float64)
    m = np.asarray(mask.data, dtype=bool)
    ts = arr[m]
    s = (seed - seed.mean()) / (seed.std() + 1e-12)
    tsz = (ts - ts.mean(1, keepdims=True)) / (ts.std(1, keepdims=True) + 1e-12)
    r = (tsz @ s) / s.shape[0]
    rmap = np.zeros(mask.shape, dtype=np.float32)
    rmap[m] = r.astype(np.float32)
    conn = ni.NeuroVol.from_array(rmap, space=mask.space)
    # The planted signal must produce a strong positive cluster.
    assert float(np.nanmax(r)) > 0.4

    # --- spatial smoothing works on both a 3D map and the 4D series (PAIN-3).
    conn_sm = ni.gaussian_blur(conn, fwhm_mm=4.0)
    assert conn_sm.shape == conn.shape
    bold_sm = ni.gaussian_blur(bold, fwhm_mm=4.0)
    assert bold_sm.shape == bold.shape
    assert bold_sm.provenance.method_name == "gaussian_blur"

    # --- threshold + connected components, with the cluster table (PAIN-4).
    cc = ni.conn_comp(conn, threshold=0.3)
    assert len(cc.voxels) >= 1
    assert cc.cluster_table is not None  # pandas present in the dev/CI env
    assert len(cc.cluster_table) == len(cc.voxels)

    # --- write / read round-trip that *preserves provenance* (PAIN-5).
    conn_with_prov = ni.gaussian_blur(conn, fwhm_mm=4.0)
    out_path = tmp_path / "seed_connectivity.nii.gz"
    ni.write_vol(conn_with_prov, str(out_path))
    back = ni.read_vol(str(out_path))
    assert np.allclose(
        np.asarray(back.data), np.asarray(conn_with_prov.data), atol=1e-4
    )
    assert back.space.compatible_with(conn.space)
    rc = receipt_of(back)
    assert rc is not None and rc.method_name == "gaussian_blur"

    # --- quick orthographic plot (smoke).
    fig = ni.plot_ortho(conn_sm)
    assert fig is not None


# ----------------------------------------------------------------------
# PAIN-1 — natural 4D-dim / 3D-spacing constructor call.
# ----------------------------------------------------------------------
def test_pain1_neurospace_accepts_spatial_spacing_for_4d():
    # A 4D series naturally carries only 3 spatial spacings; the time axis
    # has none. The trailing axis is padded with unit spacing.
    space = ni.NeuroSpace(dim=(20, 24, 18, 60), spacing=(3.0, 3.0, 3.5))
    assert tuple(np.asarray(space.spacing)) == (3.0, 3.0, 3.5, 1.0)
    assert np.allclose(np.diag(np.asarray(space.trans))[:4], (3.0, 3.0, 3.5, 1.0))

    # An over-long spacing is a genuine mistake -> a clear, typed error.
    with pytest.raises(InvalidSpaceError):
        ni.NeuroSpace(dim=(4, 4, 4), spacing=(1.0, 1.0, 1.0, 1.0))


# ----------------------------------------------------------------------
# PAIN-2 — first-class NeuroVec.mean over time (README's documented form).
# ----------------------------------------------------------------------
def test_pain2_neurovec_mean_over_time(bundle):
    bold = bundle.bold
    mean_vol = bold.mean(axis=3)
    assert mean_vol.shape == bundle.mask.shape
    assert mean_vol.provenance.method_name == "mean"
    np.testing.assert_allclose(
        np.asarray(mean_vol.data),
        np.asarray(bold.data, dtype=np.float64).mean(axis=3),
        rtol=1e-6,
    )
    # axis=-1 is equivalent; a non-time axis is rejected with guidance.
    assert bold.mean(axis=-1).shape == bundle.mask.shape
    with pytest.raises(ValueError):
        bold.mean(axis=0)


# ----------------------------------------------------------------------
# PAIN-3 — Gaussian smoothing of a 4D series is first-class & spatial-only.
# ----------------------------------------------------------------------
def test_pain3_gaussian_blur_accepts_4d_vec(bundle):
    bold = bundle.bold
    smoothed = ni.gaussian_blur(bold, fwhm_mm=6.0)
    assert isinstance(smoothed, ni.DenseNeuroVec)
    assert smoothed.shape == bold.shape
    assert smoothed.provenance.method_name == "gaussian_blur"

    # Spatial-only: a constant-in-space, varying-in-time signal keeps its
    # exact temporal profile (no blurring across time).
    data = np.asarray(bold.data, dtype=np.float64)
    sm = np.asarray(smoothed.data, dtype=np.float64)
    flat_mean_in, flat_mean_out = data.mean(axis=(0, 1, 2)), sm.mean(axis=(0, 1, 2))
    np.testing.assert_allclose(flat_mean_in, flat_mean_out, rtol=1e-6)

    # A mask restricts the write region; outside-mask voxels are untouched.
    masked = ni.gaussian_blur(bold, sigma=1.5, mask=bundle.mask)
    outside = ~np.asarray(bundle.mask.data, dtype=bool)
    np.testing.assert_array_equal(np.asarray(masked.data)[outside], data[outside])


# ----------------------------------------------------------------------
# PAIN-4 — conn_comp degrades gracefully without the optional pandas dep.
#
# pandas ships only in the `dev` extra, so a stock `pip install neuroim`
# does not get it. We reproduce the stock condition deterministically by
# blocking the import and assert the natural call still succeeds.
# ----------------------------------------------------------------------
def test_pain4_conn_comp_graceful_without_pandas(bundle):
    # pandas is (still) not a declared *runtime* dependency.
    runtime_deps = [r for r in (requires("neuroim") or []) if "extra ==" not in r]
    assert not any(d.lower().startswith("pandas") for d in runtime_deps)

    vol = ni.NeuroVol.from_array(
        np.asarray(bundle.mask.data, dtype=np.float32), space=bundle.mask.space
    )

    # With pandas available, the default call builds a real DataFrame.
    import pandas as pd

    with_pandas = ni.conn_comp(vol, threshold=0.5)
    assert isinstance(with_pandas.cluster_table, pd.DataFrame)

    # Now simulate a stock install: the natural call must NOT crash.
    saved = sys.modules.get("pandas", "<<absent>>")
    sys.modules["pandas"] = None  # makes `import pandas` raise ImportError
    try:
        with pytest.warns(RuntimeWarning, match="pandas"):
            result = ni.conn_comp(vol, threshold=0.5)  # default cluster_table=True
        assert result.cluster_table is None
        # The structural outputs are still fully populated.
        assert len(result.voxels) >= 1
        assert result.provenance is not None
    finally:
        if saved == "<<absent>>":
            del sys.modules["pandas"]
        else:
            sys.modules["pandas"] = saved


# ----------------------------------------------------------------------
# PAIN-5 — write_vol / write_vec persist the provenance receipt.
# ----------------------------------------------------------------------
def test_pain5_write_vol_persists_provenance(bundle, tmp_path):
    blurred = ni.gaussian_blur(bundle.bold.vols()[0], fwhm_mm=6.0)
    assert blurred.provenance is not None

    out_path = tmp_path / "blurred.nii.gz"
    ni.write_vol(blurred, str(out_path))

    # The receipt is embedded as a NIfTI extension on disk ...
    on_disk = nib.load(str(out_path))
    assert len(on_disk.header.extensions) == 1

    # ... and `read_vol` recovers it intact.
    recovered = receipt_of(ni.read_vol(str(out_path)))
    assert recovered is not None and recovered.method_name == "gaussian_blur"


def test_pain5_write_vec_persists_provenance(bundle, tmp_path):
    blurred = ni.gaussian_blur(bundle.bold, fwhm_mm=6.0)
    assert blurred.provenance is not None

    out_path = tmp_path / "blurred_vec.nii.gz"
    ni.write_vec(blurred, str(out_path))

    on_disk = nib.load(str(out_path))
    assert len(on_disk.header.extensions) == 1
    recovered = receipt_of(ni.read_vec(str(out_path)))
    assert recovered is not None and recovered.method_name == "gaussian_blur"

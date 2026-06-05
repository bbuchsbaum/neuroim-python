"""Stress test: an *ergonomic* seed-based connectivity mini-pipeline.

This test approaches ``neuroim`` the way a competent-but-new user would:
it reads only the curated public surface (``ni.*`` / what ``dir(ni)``
advertises and what the README documents) and tries to run a complete,
natural fMRI workflow end to end::

    4D BOLD  ->  spatial mask  ->  seed ROI  ->  seed time series
             ->  voxelwise correlation map (NeuroVol)
             ->  spatial smoothing  ->  threshold + connected components
             ->  write to NIfTI with provenance  ->  read back & verify
             ->  quick orthographic plot

The happy-path test (:func:`test_seed_connectivity_workflow_completes`)
walks the whole pipeline and *passes* — but every spot where ergonomics
forced a detour onto a numpy escape hatch or a non-obvious workaround is
flagged inline with a ``PAIN-N`` marker.

The five ``PAIN-N`` findings discovered while exercising this workflow are
pinned below as focused tests. Four are ``xfail(strict=True)`` so they flip
to **XPASS** (and fail the suite, demanding this file be updated) the moment
the gap closes — mirroring the convention already used by the ``s11``-``s19``
scenario tests. PAIN-4 is environment-sensitive (pandas is a *dev*-only
optional dep, so it is present in CI but absent for a stock
``pip install neuroim``) and is therefore pinned as a deterministic,
pandas-blocking documentation test instead.

PAIN-1  ``NeuroSpace(dim=<4-tuple>, spacing=<3-tuple>)`` — the natural call
        for a 4D series whose time axis has no spatial spacing — raises a
        cryptic ``ValueError: could not broadcast input array from shape
        (3,3) into shape (4,4)`` instead of accepting the 3-length spacing
        (or raising a clear ``InvalidSpaceError``).

PAIN-2  ``NeuroVec`` has no ``.mean()`` — yet ``README.md`` documents
        ``mean_vol = fmri.mean(axis=3)  # Mean across time``. The single
        most common time reduction has no method on the public surface; the
        user must drop to ``np.asarray(vec.data).mean(axis=3)``.

PAIN-3  ``ni.gaussian_blur`` rejects a 4D ``NeuroVec`` with the array-level
        message ``Data must be 1D, 2D ... got 4D``. Gaussian smoothing of a
        4D BOLD series — the canonical preprocessing step — has no
        first-class API (only ``bilateral_filter_vec`` / ``_4d`` exist), and
        the error names array dims rather than a remedy.

PAIN-4  ``ni.conn_comp(vol, threshold=...)`` crashes by default with
        ``ModuleNotFoundError: No module named 'pandas'`` on a stock install,
        because the ``cluster_table=True`` default lazily imports pandas,
        which is declared only under ``[project.optional-dependencies].dev``.

PAIN-5  ``ni.write_vol`` silently drops provenance: a volume carrying a
        populated Receipt writes **zero** NIfTI extensions to disk, so
        ``read_vol`` recovers no receipt — even though ``to_nibabel`` *does*
        embed it and ``read_vol`` *does* recover it from a ``to_nibabel``
        file. The curated top-level write path is inconsistent with the
        library's headline provenance contract.
"""

from __future__ import annotations

import sys
from importlib.metadata import requires

import matplotlib
import nibabel as nib
import numpy as np
import pytest

import neuroim as ni
from neuroim.verify import receipt_of

from fixtures.realistic_bold import make_realistic_bold

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def bundle():
    """A deterministic 4D BOLD + 3D mask with a planted seed signal."""
    return make_realistic_bold(seed=0xC0FFEE)


# ----------------------------------------------------------------------
# Happy path — the full ergonomic workflow, end to end.
# ----------------------------------------------------------------------
def test_seed_connectivity_workflow_completes(bundle, tmp_path):
    bold, mask = bundle.bold, bundle.mask

    # --- spaces line up. `spatial_space` is the (non-curated but
    #     discoverable) bridge from a 4D vec to its 3D spatial contract.
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

    # --- mean-over-time volume.
    #     PAIN-2: `bold.mean(axis=3)` does not exist; drop to numpy.
    mean_data = np.asarray(bold.data, dtype=np.float64).mean(axis=3)
    mean_vol = ni.NeuroVol.from_array(mean_data.astype(np.float32), space=mask.space)
    assert mean_vol.shape == mask.shape

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

    # --- spatial smoothing of the 3D statistic map (works on a NeuroVol).
    #     PAIN-3: smoothing the *4D* series directly is not possible here.
    conn_sm = ni.gaussian_blur(conn, fwhm_mm=4.0)
    assert conn_sm.shape == conn.shape

    # --- threshold + connected components on the (unsmoothed) statistic map.
    #     PAIN-4: must pass cluster_table=False to avoid an undeclared
    #     pandas dependency on a stock install.
    cc = ni.conn_comp(conn, threshold=0.3, cluster_table=False)
    assert len(cc.voxels) >= 1

    # --- write / read round-trip via the curated top-level I/O.
    out_path = tmp_path / "seed_connectivity.nii.gz"
    ni.write_vol(conn, str(out_path))
    back = ni.read_vol(str(out_path))
    assert np.allclose(np.asarray(back.data), np.asarray(conn.data), atol=1e-4)
    assert back.space.compatible_with(conn.space)

    # --- provenance survives the file boundary via the documented
    #     `to_nibabel` path (PAIN-5 covers the `write_vol` gap separately).
    prov_path = tmp_path / "blurred.nii.gz"
    nib.save(conn_sm.to_nibabel(), str(prov_path))
    recovered = ni.read_vol(str(prov_path))
    rc = receipt_of(recovered)
    assert rc is not None and rc.method_name == "gaussian_blur"

    # --- quick orthographic plot (smoke).
    fig = ni.plot_ortho(conn_sm)
    assert fig is not None


# ----------------------------------------------------------------------
# PAIN-1 — natural 4D-dim / 3D-spacing constructor call.
# ----------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="PAIN-1: NeuroSpace(4D dim, 3D spacing) should accept the spatial "
    "spacing (or raise a clear InvalidSpaceError), not a numpy broadcast error.",
)
def test_pain1_neurospace_accepts_spatial_spacing_for_4d():
    # A user building a 4D series naturally gives 3 spatial spacings; the
    # time axis has none. This should succeed.
    space = ni.NeuroSpace(dim=(20, 24, 18, 60), spacing=(3.0, 3.0, 3.5))
    assert tuple(np.asarray(space.spacing)[:3]) == (3.0, 3.0, 3.5)


# ----------------------------------------------------------------------
# PAIN-2 — documented-but-missing NeuroVec.mean.
# ----------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="PAIN-2: README documents `fmri.mean(axis=3)`; NeuroVec has no "
    "`mean`, so the most common time reduction has no public method.",
)
def test_pain2_neurovec_mean_over_time(bundle):
    mean_vol = bundle.bold.mean(axis=3)  # AttributeError today
    assert mean_vol.shape == bundle.mask.shape


# ----------------------------------------------------------------------
# PAIN-3 — Gaussian smoothing of a 4D series.
# ----------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="PAIN-3: gaussian_blur has no 4D/NeuroVec path; spatial smoothing "
    "of a BOLD time series — the canonical preprocessing step — is not "
    "first-class and fails with an array-dimension error.",
)
def test_pain3_gaussian_blur_accepts_4d_vec(bundle):
    smoothed = ni.gaussian_blur(bundle.bold, fwhm_mm=6.0)  # ValueError today
    assert smoothed.shape == bundle.bold.shape


# ----------------------------------------------------------------------
# PAIN-4 — conn_comp's default needs an undeclared optional dependency.
#
# pandas ships in the `dev` extra, so it IS importable in CI; a stock
# `pip install neuroim` does not get it. We reproduce the stock condition
# deterministically by blocking the import, then assert two things:
#   (a) the default call (cluster_table=True) hard-crashes, and
#   (b) the workaround (cluster_table=False) works without pandas.
# Both are bugs-as-documented, so this test PASSES (it pins current
# behaviour); when conn_comp stops requiring pandas by default, part (a)
# will start failing and force this file to be revisited.
# ----------------------------------------------------------------------
def test_pain4_conn_comp_default_requires_undeclared_pandas(bundle):
    # pandas is not declared as a runtime dependency (only under [dev]).
    runtime_deps = [r for r in (requires("neuroim") or []) if "extra ==" not in r]
    assert not any(d.lower().startswith("pandas") for d in runtime_deps)

    vol = ni.NeuroVol.from_array(
        np.asarray(bundle.mask.data, dtype=np.float32), space=bundle.mask.space
    )

    saved = sys.modules.get("pandas", "<<absent>>")
    sys.modules["pandas"] = None  # makes `import pandas` raise ImportError
    try:
        # (a) the natural call fails on a stock install.
        with pytest.raises(ModuleNotFoundError):
            ni.conn_comp(vol, threshold=0.5)
        # (b) the non-obvious workaround succeeds.
        cc = ni.conn_comp(vol, threshold=0.5, cluster_table=False)
        assert len(cc.voxels) >= 1
    finally:
        if saved == "<<absent>>":
            del sys.modules["pandas"]
        else:
            sys.modules["pandas"] = saved


# ----------------------------------------------------------------------
# PAIN-5 — write_vol drops the provenance receipt.
# ----------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="PAIN-5: write_vol writes 0 NIfTI extensions, so the provenance "
    "Receipt is lost — inconsistent with to_nibabel, which embeds it.",
)
def test_pain5_write_vol_persists_provenance(bundle, tmp_path):
    blurred = ni.gaussian_blur(bundle.bold.vols()[0], fwhm_mm=6.0)
    assert blurred.provenance is not None  # source carries a receipt

    out_path = tmp_path / "blurred.nii.gz"
    ni.write_vol(blurred, str(out_path))

    # Today: zero extensions on disk -> receipt is lost.
    on_disk = nib.load(str(out_path))
    assert len(on_disk.header.extensions) == 1

    recovered = receipt_of(ni.read_vol(str(out_path)))
    assert recovered is not None and recovered.method_name == "gaussian_blur"

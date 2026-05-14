"""ME-9: Compose Receipt provenance across pipeline ops.

Verifies that provenance threads forward through:
  concat -> series_roi
  concat -> searchlight

so that a downstream result's Receipt records the pipeline that produced it.
Without composition, receipts are decoration even after ME-2 ships the
verifier.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroim import (
    DenseNeuroVec,
    LogicalNeuroVol,
    NeuroSpace,
    concat,
    searchlight_apply,
    spherical_roi,
)
from neuroim.results import ROIExtractionResult, Receipt, SearchlightResult


def _make_vec(seed, *, time=3, space=None):
    if space is None:
        space = NeuroSpace(dim=(4, 4, 4, time), spacing=(2.0, 2.0, 2.0, 1.0))
    rng = np.random.default_rng(seed=seed)
    data = rng.standard_normal((4, 4, 4, time)).astype(np.float32)
    return DenseNeuroVec(data, space)


def test_concat_attaches_provenance():
    a = _make_vec(seed=1, time=3)
    b = _make_vec(seed=2, time=4)
    out = concat(a, b)
    assert hasattr(out, "provenance")
    assert isinstance(out.provenance, Receipt)
    assert out.provenance.method_name == "concat"


def test_concat_then_series_roi_chains_provenance():
    a = _make_vec(seed=1, time=3)
    b = _make_vec(seed=2, time=4)
    cat = concat(a, b)
    mask = LogicalNeuroVol(
        np.ones((4, 4, 4), dtype=bool),
        NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0)),
    )
    roi = spherical_roi(mask, centroid=(2, 2, 2), radius=4.0)
    res = cat.series_roi(roi)
    assert isinstance(res, ROIExtractionResult)
    # method_name should record the chain
    assert "concat" in res.provenance.method_name
    assert "series_roi" in res.provenance.method_name


def test_concat_then_searchlight_chains_provenance():
    a = _make_vec(seed=1, time=3)
    b = _make_vec(seed=2, time=4)
    cat = concat(a, b)
    mask = LogicalNeuroVol(
        np.ones((4, 4, 4), dtype=bool),
        NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0)),
    )
    res = searchlight_apply(
        mask, radius=4.0, method=lambda arr: float(np.asarray(arr).mean()), data=cat
    )
    assert isinstance(res, SearchlightResult)
    assert "concat" in res.provenance.method_name
    assert "searchlight" in res.provenance.method_name


def test_series_roi_without_upstream_has_no_chain_prefix():
    """When the upstream NeuroVec has no provenance, series_roi's output
    Receipt records only its own method_name — confirming chain semantics."""
    vec = _make_vec(seed=0, time=3)
    mask = LogicalNeuroVol(
        np.ones((4, 4, 4), dtype=bool),
        NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0)),
    )
    roi = spherical_roi(mask, centroid=(2, 2, 2), radius=4.0)
    res = vec.series_roi(roi)
    assert res.provenance.method_name == "series_roi"


def test_concat_then_series_roi_with_mismatched_space_raises():
    """When the concat output has been bound to a different space than the
    mask used downstream, the threaded provenance surfaces the mismatch.

    This is the canonical "silent space mismatch" the mission promises to
    catch — the provenance chain makes it falsifiable.
    """
    a = _make_vec(seed=1, time=3)
    b = _make_vec(seed=2, time=4)
    cat = concat(a, b)

    # Manually rewrite the upstream Receipt's input_space_hash to simulate
    # a tampered or stale pipeline.
    from dataclasses import replace
    cat.provenance = replace(cat.provenance, input_space_hash="tampered")

    mask = LogicalNeuroVol(
        np.ones((4, 4, 4), dtype=bool),
        NeuroSpace(dim=(4, 4, 4), spacing=(2.0, 2.0, 2.0)),
    )
    roi = spherical_roi(mask, centroid=(2, 2, 2), radius=4.0)
    with pytest.raises(ValueError, match="input_space_hash"):
        cat.series_roi(roi)

"""Acceptance test for Scenario 18 -- cluster-table from a thresholded stat map.

Six assertions:

1. **Numeric parity (happy path)** -- baseline (scipy + numpy) and the
   neuroim rewrite produce the same cluster table (same number of
   clusters; same voxel counts; same signed peak values; same world-mm
   peak coordinates) on the synthesized two-cluster stat map.

2. **Two-tailed coverage** -- both paths surface the negative cluster
   (z ~ -3.2) as well as the positive one (z ~ +3.5).  Documents the
   bug class S18 PAIN-4 surfaces: ``conn_comp`` is one-tailed by
   default, so the rewrite has to absolutize the stat map before
   calling.

3. **Same-space gate** -- LR-flipped mask raises ``ValueError`` through
   ``assert_same_space`` at the call site (PAIN-3).

4-6. **Strict xfails** -- one per remaining PAIN (1, 2, 4, 5).  Each
   xfail flips to XPASS when the API gap closes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import neuroim as ni


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
    / "18_cluster_table_post_stats"
)
baseline = _load_module("scenario18_baseline", _SCENARIO_DIR / "baseline_nibabel.py")
rewrite = _load_module("scenario18_rewrite", _SCENARIO_DIR / "neuroim_version.py")


@pytest.fixture(scope="module")
def fixture():
    stat_img, mask_img = baseline.synthesize_stat_map()
    stat_vol, mask_vol = rewrite.typed_from_nibabel(stat_img, mask_img)
    return stat_img, mask_img, stat_vol, mask_vol


# ----------------------------------------------------------------------
# Happy path: numeric parity
# ----------------------------------------------------------------------


def test_baseline_and_neuroim_cluster_tables_agree(fixture):
    """Same input stat-map + mask -> identical cluster tables."""
    stat_img, mask_img, stat_vol, mask_vol = fixture
    base = baseline.cluster_table_from_stat_map(
        stat_img, mask_img, threshold=2.3, min_extent=5
    )
    rew = rewrite.cluster_table_from_stat_map(
        stat_vol, mask_vol, threshold=2.3, min_extent=5
    )
    rew_table = rew.table

    # Same set of clusters
    assert len(base) == len(rew_table)
    # Sort both by descending voxel count (the rewrites already do this)
    # and compare row-by-row.
    for col in ("n_voxels", "peak_value", "peak_x_mm", "peak_y_mm", "peak_z_mm"):
        np.testing.assert_allclose(
            base[col].to_numpy(),
            rew_table[col].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"column {col!r} disagrees between baseline and rewrite",
        )


def test_both_paths_surface_negative_cluster(fixture):
    """Two-tailed: the |z| > 2.3 sweep must catch the negative cluster.

    The fixture embeds a positive cluster (z ~ +3.5) AND a negative
    cluster (z ~ -3.2).  A correct cluster table contains both, with
    opposite-signed ``peak_value`` columns.
    """
    stat_img, mask_img, stat_vol, mask_vol = fixture
    base = baseline.cluster_table_from_stat_map(
        stat_img, mask_img, threshold=2.3, min_extent=5
    )
    rew = rewrite.cluster_table_from_stat_map(
        stat_vol, mask_vol, threshold=2.3, min_extent=5
    ).table
    for table in (base, rew):
        assert len(table) >= 2, f"missing the negative cluster: {table!r}"
        peaks = table["peak_value"].to_numpy()
        assert (peaks > 0).any()
        assert (peaks < 0).any()


def test_neuroim_rejects_mismatched_affine_mask(fixture):
    """LR-flipped mask trips ``assert_same_space`` at the call site."""
    _, _, stat_vol, _ = fixture
    flipped = np.asarray(stat_vol.space.trans, dtype=float).copy()
    flipped[:, 0] = -flipped[:, 0]
    rotated_space = ni.NeuroSpace.from_affine(flipped, stat_vol.shape)
    mask_data = np.ones(stat_vol.shape, dtype=bool)
    rotated_mask = ni.LogicalNeuroVol(mask_data, rotated_space)
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        rewrite.cluster_table_from_stat_map(
            stat_vol, rotated_mask, threshold=2.3, min_extent=5
        )


# ----------------------------------------------------------------------
# PAIN gates (strict xfails -- flip to XPASS when each lands)
# ----------------------------------------------------------------------


def test_conn_comp_in_public_namespace():
    """PAIN-1 closed: ``conn_comp`` is part of the curated public API."""
    assert "conn_comp" in ni.__all__


def test_conn_comp_result_carries_provenance(fixture):
    """PAIN-2 closed: ``ConnCompResult`` ships a populated Receipt."""
    _, _, stat_vol, _ = fixture
    from neuroim.connected_components import conn_comp
    from neuroim.results import Receipt

    result = conn_comp(stat_vol, threshold=2.3)
    assert isinstance(result.provenance, Receipt)
    assert result.provenance.method_name == "conn_comp"
    assert result.provenance.radius == pytest.approx(2.3)


def test_conn_comp_accepts_mask_parameter_with_same_space_gate(fixture):
    """PAIN-3 closed: ``mask=`` is a keyword-only parameter and an
    LR-flipped mask raises through ``assert_same_space``.
    """
    import inspect

    _, _, stat_vol, _ = fixture
    from neuroim.connected_components import conn_comp

    sig = inspect.signature(conn_comp)
    assert "mask" in sig.parameters

    flipped = np.asarray(stat_vol.space.trans, dtype=float).copy()
    flipped[:, 0] = -flipped[:, 0]
    rotated_space = ni.NeuroSpace.from_affine(flipped, stat_vol.shape)
    rotated_mask = ni.LogicalNeuroVol(
        np.ones(stat_vol.shape, dtype=bool), rotated_space
    )
    with pytest.raises(ValueError, match="spatial contract mismatch"):
        conn_comp(stat_vol, threshold=2.3, mask=rotated_mask)


def test_conn_comp_supports_two_tailed(fixture):
    """PAIN-4 closed: ``two_tailed=True`` picks up the negative cluster
    directly through the public API.
    """
    import inspect

    _, _, stat_vol, _ = fixture
    from neuroim.connected_components import conn_comp

    sig = inspect.signature(conn_comp)
    assert "two_tailed" in sig.parameters

    result = conn_comp(stat_vol, threshold=2.3, two_tailed=True)
    table = result.cluster_table
    assert table is not None and len(table) >= 2
    peaks = table["peak_value"].to_numpy()
    assert (peaks > 0).any()
    assert (peaks < 0).any()


def test_cluster_table_reports_world_mm_peaks(fixture):
    """PAIN-5 closed: peak columns in world mm are always present."""
    _, _, stat_vol, _ = fixture
    from neuroim.connected_components import conn_comp

    result = conn_comp(stat_vol, threshold=2.3, two_tailed=True)
    cols = set(result.cluster_table.columns)
    assert {"peak_x_mm", "peak_y_mm", "peak_z_mm", "peak_value"} <= cols

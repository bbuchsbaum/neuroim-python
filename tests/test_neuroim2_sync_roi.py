"""ROI and searchlight contracts carried over from neuroim2 (Aug 2026).

Window builders truncate fractional centres, sphere sizes survive hostile
radii, spaces reject non-finite spacing, random searchlights partition the
mask, clustered searchlights accept ClusteredNeuroVol, and conn_comp's
vectorised bookkeeping matches the per-cluster definition.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuroim import DenseNeuroVol, LogicalNeuroVol, NeuroSpace
from neuroim.clustered_neuro_vol import ClusteredNeuroVol
from neuroim.connected_components import conn_comp
from neuroim.exceptions import InvalidSpaceError
from neuroim.roi import ellipsoid_roi, spherical_roi
from neuroim.searchlight import (
    clustered_searchlight,
    random_searchlight,
    searchlight_iterator,
)


def _coords(roi):
    return sorted(map(tuple, np.asarray(roi.coords).tolist()))


def test_fractional_centroid_matches_its_voxel():
    sp = NeuroSpace((9, 9, 9))
    assert _coords(spherical_roi(sp, (4.6, 4, 4), 1.0)) == _coords(
        spherical_roi(sp, (4, 4, 4), 1.0)
    )
    assert _coords(ellipsoid_roi(sp, (4.6, 4.2, 4), (1.0, 1.0, 1.0))) == _coords(
        ellipsoid_roi(sp, (4, 4, 4), (1.0, 1.0, 1.0))
    )


@pytest.mark.parametrize("radius", [1e18, 1e19, 1e300])
def test_huge_radius_covers_volume(radius):
    sp = NeuroSpace((5, 5, 5))
    assert len(spherical_roi(sp, (2, 2, 2), radius).coords) == 125


def test_huge_radius_searchlight_window():
    mask = LogicalNeuroVol(np.ones((3, 3, 3), dtype=bool), NeuroSpace((3, 3, 3)))
    assert len(searchlight_iterator(mask, 1e19)[0].coords) == 27


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_spacing_rejected(bad):
    with pytest.raises(InvalidSpaceError):
        NeuroSpace((3, 3, 3), spacing=(bad, 1.0, 1.0))


def test_random_searchlight_partitions_mask():
    np.random.seed(0)
    m = np.zeros((8, 8, 8), dtype=bool)
    m[1:7, 1:7, 1:7] = True
    mask = LogicalNeuroVol(m, NeuroSpace(m.shape))
    windows = random_searchlight(mask, 2.0)
    members = np.concatenate([w.coords for w in windows])
    keys = [tuple(c) for c in members.tolist()]
    assert len(keys) == len(set(keys)) == int(m.sum())
    for w in windows:
        center = np.asarray(w.coords)[w.center_index]
        assert m[tuple(center)]


def test_clustered_searchlight_accepts_clustered_volume():
    sp = NeuroSpace((4, 4, 2))
    m = np.zeros((4, 4, 2), dtype=bool)
    m[:2] = True
    labels = np.where(np.indices((4, 4, 2))[1][m] < 2, 1, 2)
    cvol = ClusteredNeuroVol(LogicalNeuroVol(m, sp), labels)
    full = LogicalNeuroVol(np.ones((4, 4, 2), dtype=bool), sp)
    regions = list(clustered_searchlight(full, cvol=cvol))
    assert sorted(len(r.coords) for r in regions) == [8, 8]


def test_cluster_map_groups_mask_indices():
    sp = NeuroSpace((5, 4, 3))
    m = np.random.default_rng(2).random((5, 4, 3)) > 0.3
    labels = np.random.default_rng(3).integers(1, 6, int(m.sum()))
    cvol = ClusteredNeuroVol(LogicalNeuroVol(m, sp), labels)
    idx = np.flatnonzero(m.ravel(order="F"))
    for k, members in cvol.cluster_map.items():
        np.testing.assert_array_equal(members, idx[labels == k])


def test_conn_comp_bookkeeping_matches_definition():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(12, 10, 8))
    vol = DenseNeuroVol(data, NeuroSpace(data.shape))
    res = conn_comp(vol, threshold=0.5, connect="26-connect")
    from scipy.ndimage import generate_binary_structure, label

    lab, n = label(data > 0.5, structure=generate_binary_structure(3, 3))
    assert len(res.voxels) == n
    for i, vox in enumerate(res.voxels, start=1):
        np.testing.assert_array_equal(vox, np.column_stack(np.where(lab == i)))
        assert np.all(res.size.data[lab == i] == (lab == i).sum())
    assert np.all(res.size.data[lab == 0] == 0)

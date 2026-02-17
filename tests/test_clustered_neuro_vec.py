"""Test suite for ClusteredNeuroVec class."""

import pytest
import numpy as np
from neuroimpy import NeuroSpace, LogicalNeuroVol, ClusteredNeuroVol
from neuroimpy.clustered_neuro_vec import ClusteredNeuroVec


@pytest.fixture
def cvol():
    """Create a small ClusteredNeuroVol with 3 clusters."""
    space = NeuroSpace((4, 4, 4), spacing=(1, 1, 1))
    mask_data = np.ones((4, 4, 4), dtype=bool)
    mask = LogicalNeuroVol(mask_data, space)
    # 64 voxels, assign to 3 clusters
    clusters = np.array([0, 1, 2] * 21 + [0])  # length 64
    return ClusteredNeuroVol(mask, clusters)


@pytest.fixture
def cvec(cvol):
    """Create a ClusteredNeuroVec with 5 time points and 3 clusters."""
    rng = np.random.default_rng(7)
    n_time = 5
    n_clusters = cvol.num_clusters()
    ts = rng.random((n_time, n_clusters))
    return ClusteredNeuroVec(cvol, ts, label="test_vec")


class TestClusteredNeuroVecCreation:

    def test_basic(self, cvec, cvol):
        assert isinstance(cvec, ClusteredNeuroVec)
        assert cvec.n_time == 5
        assert cvec.n_clusters == 3
        assert cvec.label == "test_vec"
        assert cvec.cvol is cvol

    def test_bad_cvol_type(self):
        with pytest.raises(TypeError):
            ClusteredNeuroVec("not_a_cvol", np.zeros((5, 3)))

    def test_ts_wrong_ndim(self, cvol):
        with pytest.raises(ValueError, match="2-D"):
            ClusteredNeuroVec(cvol, np.zeros(10))

    def test_ts_wrong_n_clusters(self, cvol):
        with pytest.raises(ValueError, match="columns"):
            ClusteredNeuroVec(cvol, np.zeros((5, 99)))


class TestClusteredNeuroVecTimeseries:

    def test_cluster_timeseries(self, cvec):
        ts0 = cvec.cluster_timeseries(0)
        assert ts0.shape == (5,)
        assert np.array_equal(ts0, cvec.ts[:, cvec._id_to_col[0]])

    def test_cluster_timeseries_invalid(self, cvec):
        with pytest.raises(KeyError):
            cvec.cluster_timeseries(999)

    def test_voxel_cluster(self, cvec):
        cid = cvec.voxel_cluster(0)
        assert cid == cvec.cl_map[0]

    def test_voxel_timeseries(self, cvec):
        ts = cvec.voxel_timeseries(0)
        expected_cid = cvec.voxel_cluster(0)
        expected_ts = cvec.cluster_timeseries(expected_cid)
        assert np.array_equal(ts, expected_ts)


class TestClusteredNeuroVecIteration:

    def test_iter_clusters(self, cvec):
        pairs = list(cvec.iter_clusters())
        assert len(pairs) == 3
        ids_seen = set()
        for cid, ts in pairs:
            assert isinstance(cid, int)
            assert ts.shape == (5,)
            ids_seen.add(cid)
        assert ids_seen == {0, 1, 2}


class TestClusteredNeuroVecProperties:

    def test_cluster_ids(self, cvec):
        ids = cvec.cluster_ids
        assert np.array_equal(ids, [0, 1, 2])

    def test_space(self, cvec, cvol):
        assert cvec.space is cvol.space

    def test_shape(self, cvec):
        assert cvec.shape == (4, 4, 4)

    def test_repr(self, cvec):
        r = repr(cvec)
        assert "ClusteredNeuroVec" in r
        assert "3" in r  # num clusters
        assert "5" in r  # num time

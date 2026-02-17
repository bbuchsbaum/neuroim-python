"""Test suite for NeuroBucket class."""

import pytest
import numpy as np
from neuroimpy import NeuroSpace, DenseNeuroVol
from neuroimpy.neuro_bucket import NeuroBucket


@pytest.fixture
def space3d():
    return NeuroSpace((4, 4, 4), spacing=(2, 2, 2))


@pytest.fixture
def volumes(space3d):
    rng = np.random.default_rng(42)
    vols = [DenseNeuroVol(rng.random((4, 4, 4)), space3d) for _ in range(3)]
    return vols


@pytest.fixture
def bucket(volumes, space3d):
    labels = ["alpha", "beta", "gamma"]
    return NeuroBucket(labels, volumes, space3d)


class TestNeuroBucketCreation:

    def test_basic_creation(self, bucket):
        assert isinstance(bucket, NeuroBucket)
        assert len(bucket) == 3

    def test_mismatched_labels_raises(self, volumes, space3d):
        with pytest.raises(ValueError, match="labels length"):
            NeuroBucket(["a", "b"], volumes, space3d)

    def test_empty_data_raises(self, space3d):
        with pytest.raises(ValueError, match="at least one"):
            NeuroBucket([], [], space3d)

    def test_shape_mismatch_raises(self, space3d):
        wrong_space = NeuroSpace((8, 8, 8))
        vol = DenseNeuroVol(np.zeros((8, 8, 8)), wrong_space)
        with pytest.raises(ValueError, match="does not match"):
            NeuroBucket(["x"], [vol], space3d)


class TestNeuroBucketIndexing:

    def test_int_index(self, bucket, volumes):
        vol = bucket[0]
        assert isinstance(vol, DenseNeuroVol)
        assert np.array_equal(vol.data, volumes[0].data)

    def test_negative_int_index(self, bucket, volumes):
        vol = bucket[-1]
        assert np.array_equal(vol.data, volumes[-1].data)

    def test_str_index(self, bucket, volumes):
        vol = bucket["beta"]
        assert np.array_equal(vol.data, volumes[1].data)

    def test_str_index_missing_raises(self, bucket):
        with pytest.raises(KeyError, match="not found"):
            bucket["missing"]

    def test_slice_returns_bucket(self, bucket):
        sub = bucket[:2]
        assert isinstance(sub, NeuroBucket)
        assert len(sub) == 2
        assert sub.labels == ["alpha", "beta"]

    def test_invalid_key_type_raises(self, bucket):
        with pytest.raises(TypeError):
            bucket[3.14]


class TestNeuroBucketIteration:

    def test_iter_yields_pairs(self, bucket, volumes):
        pairs = list(bucket)
        assert len(pairs) == 3
        for (label, vol), expected_label in zip(pairs, ["alpha", "beta", "gamma"]):
            assert label == expected_label
            assert isinstance(vol, DenseNeuroVol)

    def test_contains(self, bucket):
        assert "alpha" in bucket
        assert "missing" not in bucket


class TestNeuroBucketProperties:

    def test_shape(self, bucket):
        assert bucket.shape == (4, 4, 4)

    def test_ndim(self, bucket):
        assert bucket.ndim == 3

    def test_spacing(self, bucket):
        assert np.array_equal(bucket.spacing, [2, 2, 2])

    def test_repr(self, bucket):
        r = repr(bucket)
        assert "NeuroBucket" in r
        assert "alpha" in r

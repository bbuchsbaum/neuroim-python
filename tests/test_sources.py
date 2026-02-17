"""Tests for Source factory classes (lazy loaders)."""

import pytest
import numpy as np
import tempfile
import os

from neuroimpy.sources import (
    FileSource, NeuroVolSource, NeuroVecSource,
    SparseNeuroVecSource, MappedNeuroVecSource,
)

# nibabel is required for these tests
nib = pytest.importorskip("nibabel")


def _make_nifti_3d(tmp_path, shape=(10, 10, 10), fname="vol.nii.gz"):
    """Create a temporary 3-D NIfTI file and return its path."""
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = os.path.join(str(tmp_path), fname)
    nib.save(img, path)
    return path, data


def _make_nifti_4d(tmp_path, shape=(10, 10, 10, 5), fname="vec.nii.gz"):
    """Create a temporary 4-D NIfTI file and return its path."""
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = os.path.join(str(tmp_path), fname)
    nib.save(img, path)
    return path, data


# ---- FileSource base ----

class TestFileSource:

    def test_repr_not_loaded(self, tmp_path):
        path, _ = _make_nifti_3d(tmp_path)
        src = FileSource(path)
        assert "not loaded" in repr(src)

    def test_meta_reads_header(self, tmp_path):
        path, _ = _make_nifti_3d(tmp_path)
        src = FileSource(path)
        meta = src.meta
        assert "dim" in meta  # read_header returns a dict with 'dim'

    def test_meta_uses_provided(self, tmp_path):
        path, _ = _make_nifti_3d(tmp_path)
        custom_meta = {"custom": True}
        src = FileSource(path, meta_info=custom_meta)
        assert src.meta is custom_meta

    def test_load_raises(self, tmp_path):
        path, _ = _make_nifti_3d(tmp_path)
        src = FileSource(path)
        with pytest.raises(NotImplementedError):
            src.load()


# ---- NeuroVolSource ----

class TestNeuroVolSource:

    def test_lazy_load(self, tmp_path):
        path, data = _make_nifti_3d(tmp_path)
        src = NeuroVolSource(path)
        assert src._loaded is None
        vol = src.load()
        assert src._loaded is not None
        assert vol.shape == data.shape

    def test_property_triggers_load(self, tmp_path):
        path, data = _make_nifti_3d(tmp_path)
        src = NeuroVolSource(path)
        assert src._loaded is None
        d = src.data
        assert src._loaded is not None
        assert d.shape == data.shape

    def test_space_property(self, tmp_path):
        path, _ = _make_nifti_3d(tmp_path)
        src = NeuroVolSource(path)
        sp = src.space
        assert hasattr(sp, "dim")

    def test_repr_loaded(self, tmp_path):
        path, _ = _make_nifti_3d(tmp_path)
        src = NeuroVolSource(path)
        src.load()
        assert "loaded" in repr(src)
        assert "not loaded" not in repr(src)

    def test_index_parameter(self, tmp_path):
        """NeuroVolSource with index extracts the correct volume from 4-D."""
        path, data = _make_nifti_4d(tmp_path, shape=(8, 8, 8, 3))
        src = NeuroVolSource(path, index=2)
        vol = src.load()
        np.testing.assert_allclose(vol.data, data[:, :, :, 2], atol=1e-5)


# ---- NeuroVecSource ----

class TestNeuroVecSource:

    def test_lazy_load(self, tmp_path):
        path, data = _make_nifti_4d(tmp_path)
        src = NeuroVecSource(path)
        assert src._loaded is None
        vec = src.load()
        assert vec.data.shape == data.shape

    def test_property_triggers_load(self, tmp_path):
        path, _ = _make_nifti_4d(tmp_path)
        src = NeuroVecSource(path)
        _ = src.data  # triggers load
        assert src._loaded is not None

    def test_indices_subset(self, tmp_path):
        path, data = _make_nifti_4d(tmp_path, shape=(8, 8, 8, 6))
        src = NeuroVecSource(path, indices=[0, 2, 4])
        vec = src.load()
        assert vec.data.shape[3] == 3


# ---- SparseNeuroVecSource ----

class TestSparseNeuroVecSource:

    def test_lazy_load_with_mask(self, tmp_path):
        from neuroimpy import NeuroSpace, LogicalNeuroVol
        path, data = _make_nifti_4d(tmp_path, shape=(8, 8, 8, 4))
        mask_data = np.zeros((8, 8, 8), dtype=bool)
        mask_data[2:6, 2:6, 2:6] = True
        space3d = NeuroSpace([8, 8, 8])
        mask = LogicalNeuroVol(mask_data, space3d)

        src = SparseNeuroVecSource(path, mask=mask)
        assert src._loaded is None
        vec = src.load()
        assert src._loaded is not None
        # SparseNeuroVec has mask attribute
        assert hasattr(vec, "mask")


# ---- MappedNeuroVecSource ----

class TestMappedNeuroVecSource:

    def test_lazy_load_with_map(self, tmp_path):
        path, _ = _make_nifti_4d(tmp_path, shape=(6, 6, 6, 3))
        src = MappedNeuroVecSource(path, map_fun=lambda x: x * 2.0)
        assert src._loaded is None
        mvec = src.load()
        assert src._loaded is not None
        # The mapped vec should exist
        assert hasattr(mvec, "map_fun")

    def test_data_property(self, tmp_path):
        path, data = _make_nifti_4d(tmp_path, shape=(6, 6, 6, 3))
        src = MappedNeuroVecSource(path, map_fun=lambda x: x * 0.0)
        d = src.data
        np.testing.assert_allclose(d, 0.0, atol=1e-7)

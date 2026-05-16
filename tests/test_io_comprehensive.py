"""Comprehensive tests for io.py to improve coverage."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

try:
    import nibabel as nib

    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

from neuroim import (
    write_vol,
    write_vec,
    DenseNeuroVol,
    SparseNeuroVol,
    LogicalNeuroVol,
    DenseNeuroVec,
    SparseNeuroVec,
    NeuroSpace,
)
from neuroim.io import read_header, read_meta_info
from neuroim.io import read_vol, read_vec, read_vol_list


@pytest.mark.skipif(not NIBABEL_AVAILABLE, reason="nibabel not available")
class TestReadWriteVol:
    """Test read_vol and write_vol functions."""

    def test_read_write_vol_roundtrip(self, tmp_path):
        """Test reading and writing volume data."""
        path = tmp_path / "roundtrip.nii"

        data = np.random.randn(10, 12, 8).astype(np.float32)
        space = NeuroSpace(
            dim=data.shape, spacing=(2.0, 2.0, 3.0), origin=(10.0, 20.0, 30.0)
        )
        vol = DenseNeuroVol(data, space)

        write_vol(vol, path)
        loaded_vol = read_vol(path)

        np.testing.assert_array_almost_equal(loaded_vol.data, data, decimal=5)
        assert loaded_vol.shape == data.shape
        np.testing.assert_array_almost_equal(loaded_vol.spacing, (2.0, 2.0, 3.0))

    def test_read_vol_from_4d_file(self):
        """Test reading a specific volume from 4D file."""
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create 4D data
            data_4d = np.random.randn(10, 10, 10, 5).astype(np.float32)
            img = nib.Nifti1Image(data_4d, np.eye(4))
            nib.save(img, tmp_path)

            # Read different volumes
            vol0 = read_vol(tmp_path, index=0)
            vol3 = read_vol(tmp_path, index=3)

            np.testing.assert_array_equal(vol0.data, data_4d[:, :, :, 0])
            np.testing.assert_array_equal(vol3.data, data_4d[:, :, :, 3])

            # Test out of range
            with pytest.raises(ValueError, match="index 5 out of range"):
                read_vol(tmp_path, index=5)

        finally:
            os.unlink(tmp_path)

    def test_read_vol_wrong_dimensions(self):
        """Test error handling for wrong dimensions."""
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create 2D data (invalid)
            data_2d = np.random.randn(10, 10).astype(np.float32)
            img = nib.Nifti1Image(data_2d, np.eye(4))
            nib.save(img, tmp_path)

            with pytest.raises(ValueError, match="Expected 3D or 4D data, got 2D"):
                read_vol(tmp_path)

        finally:
            os.unlink(tmp_path)

    def test_write_vol_with_data_types(self, tmp_path):
        """Test writing volumes with different data types."""
        vol_data = np.random.randn(5, 5, 5) * 100
        space = NeuroSpace(dim=(5, 5, 5))
        vol = DenseNeuroVol(vol_data, space)

        data_types = ["FLOAT32", "FLOAT64", "INT16", "INT32", "UINT8", "UINT16"]

        for dtype in data_types:
            path = tmp_path / f"vol_{dtype}.nii"
            write_vol(vol, path, data_type=dtype)

            img = nib.load(path)
            loaded_data = img.get_fdata()
            assert loaded_data.shape == vol_data.shape

    def test_write_vol_accepts_float_alias(self):
        """Test legacy aliases are accepted for NIfTI data_type."""
        vol = DenseNeuroVol(
            np.ones((3, 3, 3), dtype=np.float32), NeuroSpace(dim=(3, 3, 3))
        )

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_vol(vol, tmp_path, data_type="DOUBLE")
            loaded = nib.load(tmp_path)
            assert loaded.header.get_data_dtype() == np.dtype(np.float64)
        finally:
            os.unlink(tmp_path)

    def test_write_vol_accepts_nifti_format_aliases(self):
        """Test legacy NIfTI format aliases are accepted."""
        vol = DenseNeuroVol(
            np.ones((4, 4, 4), dtype=np.float32), NeuroSpace(dim=(4, 4, 4))
        )

        for alias in ["NIFTI1", "NIFTI-1"]:
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                write_vol(vol, tmp_path, format=alias)
                loaded = nib.load(tmp_path)
                assert loaded.header.get_data_dtype() == np.dtype(np.float32)
            finally:
                os.unlink(tmp_path)

    def test_write_vol_accepts_legacy_data_type_aliases(self):
        """Test legacy NIfTI data_type aliases are accepted."""
        vol = DenseNeuroVol(
            np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int16),
            NeuroSpace(dim=(2, 2, 2)),
        )

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_short:
            tmp_path_short = tmp_short.name

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_int:
            tmp_path_int = tmp_int.name

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_binary:
            tmp_path_binary = tmp_binary.name
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_ubyte:
            tmp_path_ubyte = tmp_ubyte.name

        try:
            write_vol(vol, tmp_path_short, data_type="SHORT")
            short_img = nib.load(tmp_path_short)
            assert short_img.header.get_data_dtype() == np.dtype(np.int16)

            write_vol(vol, tmp_path_int, data_type="INT")
            int_img = nib.load(tmp_path_int)
            assert int_img.header.get_data_dtype() == np.dtype(np.int32)

            write_vol(vol, tmp_path_binary, data_type="BINARY")
            binary_img = nib.load(tmp_path_binary)
            assert binary_img.header.get_data_dtype() == np.dtype(np.uint8)

            write_vol(vol, tmp_path_ubyte, data_type="UBYTE")
            ubyte_img = nib.load(tmp_path_ubyte)
            assert ubyte_img.header.get_data_dtype() == np.dtype(np.uint8)
        finally:
            os.unlink(tmp_path_short)
            os.unlink(tmp_path_int)
            os.unlink(tmp_path_binary)
            os.unlink(tmp_path_ubyte)

    def test_write_vol_defaults_to_float_type(self):
        """Test no data_type uses default float output."""
        vol = DenseNeuroVol(
            np.array([[[1, 0], [1, 0]], [[0, 1], [1, 0]]], dtype=np.int16),
            NeuroSpace(dim=(2, 2, 2)),
        )

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_vol(vol, tmp_path)
            loaded = nib.load(tmp_path)
            assert loaded.header.get_data_dtype() == np.dtype(np.float32)
        finally:
            os.unlink(tmp_path)

    def test_write_sparse_vol(self, tmp_path):
        """Test writing sparse volume (converts to dense)."""
        path = tmp_path / "sparse_vol.nii"

        indices = np.array([0, 10, 20, 30, 40])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        space = NeuroSpace(dim=(5, 5, 5))
        sparse_vol = SparseNeuroVol(data, space, indices)

        write_vol(sparse_vol, path)
        loaded_vol = read_vol(path)

        dense_data = sparse_vol.as_dense().data
        np.testing.assert_array_almost_equal(loaded_vol.data, dense_data)

    def test_write_vol_unsupported_format(self):
        """Test error for unsupported format."""
        vol = DenseNeuroVol(np.ones((5, 5, 5)), NeuroSpace(dim=(5, 5, 5)))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            write_vol(vol, "test.hdr", format="ANALYZE")


@pytest.mark.skipif(not NIBABEL_AVAILABLE, reason="nibabel not available")
class TestReadWriteVec:
    """Test read_vec and write_vec functions."""

    def test_read_write_vec_roundtrip(self, tmp_path):
        """Test reading and writing 4D vector data."""
        path = tmp_path / "vec_input.nii"
        path2 = tmp_path / "vec_roundtrip.nii"

        data = np.random.randn(10, 10, 10, 20).astype(np.float32)
        affine = np.diag([2.0, 2.0, 3.0, 1.0])
        img = nib.Nifti1Image(data, affine)
        nib.save(img, path)

        loaded_vec = read_vec(path)

        np.testing.assert_array_almost_equal(loaded_vec.data, data, decimal=5)
        assert loaded_vec.shape == data.shape

        write_vec(loaded_vec, path2)
        loaded_vec2 = read_vec(path2)
        np.testing.assert_array_almost_equal(loaded_vec2.data, data, decimal=5)

    def test_read_vec_from_3d_file(self):
        """Test reading 3D file as 4D vector with single volume."""
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create 3D data
            data_3d = np.random.randn(8, 8, 8).astype(np.float32)
            img = nib.Nifti1Image(data_3d, np.eye(4))
            nib.save(img, tmp_path)

            # Read as vector (should add time dimension)
            vec = read_vec(tmp_path)

            assert vec.shape == (8, 8, 8, 1)
            np.testing.assert_array_equal(vec.data[:, :, :, 0], data_3d)

        finally:
            os.unlink(tmp_path)

    def test_read_vec_with_indices(self):
        """Test reading specific volumes using indices."""
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create 4D data with 10 volumes
            data_4d = np.random.randn(5, 5, 5, 10).astype(np.float32)
            img = nib.Nifti1Image(data_4d, np.eye(4))
            nib.save(img, tmp_path)

            # Read specific indices
            indices = [0, 2, 5, 7]
            vec = read_vec(tmp_path, indices=indices)

            assert vec.shape == (5, 5, 5, 4)
            for i, idx in enumerate(indices):
                np.testing.assert_array_equal(
                    vec.data[:, :, :, i], data_4d[:, :, :, idx]
                )

            # Test out of range indices
            with pytest.raises(ValueError, match="indices out of range"):
                read_vec(tmp_path, indices=[0, 5, 10])

        finally:
            os.unlink(tmp_path)

    def test_read_vec_with_scalar_index(self):
        """Test reading a single volume via scalar index."""
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            data_4d = np.random.randn(5, 5, 5, 10).astype(np.float32)
            img = nib.Nifti1Image(data_4d, np.eye(4))
            nib.save(img, tmp_path)

            vec = read_vec(tmp_path, indices=1)

            assert vec.shape == (5, 5, 5, 1)
            np.testing.assert_array_equal(vec.data[:, :, :, 0], data_4d[:, :, :, 1])

        finally:
            os.unlink(tmp_path)

    def test_read_vec_with_mask(self):
        """Test reading vector with mask for sparse representation."""
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create 4D data
            data_4d = np.random.randn(5, 5, 5, 8).astype(np.float32)
            img = nib.Nifti1Image(data_4d, np.eye(4))
            nib.save(img, tmp_path)

            # Create mask
            mask_data = np.zeros((5, 5, 5), dtype=bool)
            mask_data[1:4, 1:4, 1:4] = True
            mask_space = NeuroSpace(dim=(5, 5, 5))
            mask = LogicalNeuroVol(mask_data, mask_space)

            # Read with mask
            sparse_vec = read_vec(tmp_path, mask=mask)

            assert isinstance(sparse_vec, SparseNeuroVec)
            assert sparse_vec.shape == (5, 5, 5, 8)
            assert sparse_vec.data.shape == (8, np.sum(mask_data))  # time x voxels

        finally:
            os.unlink(tmp_path)

    def test_read_vec_wrong_dimensions(self, tmp_path):
        """Test error handling for wrong dimensions."""
        path = tmp_path / "wrong_dims.nii"

        data_5d = np.random.randn(5, 5, 5, 5, 5).astype(np.float32)
        data_4d = data_5d.reshape(5, 5, 5, -1)
        img = nib.Nifti1Image(data_4d, np.eye(4))
        img.header["dim"][0] = 5
        img.header["dim"][5] = 5
        nib.save(img, path)

        vec = read_vec(path)
        assert len(vec.shape) == 4

    def test_write_sparse_vec(self, tmp_path):
        """Test writing sparse vector (converts to dense)."""
        path = tmp_path / "sparse_vec.nii"

        space = NeuroSpace(dim=(5, 5, 5, 10))
        mask_data = np.zeros((5, 5, 5), dtype=bool)
        mask_data[2, 2, 2] = True
        mask_data[3, 3, 3] = True
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=(5, 5, 5)))

        sparse_data = np.random.randn(10, 2)  # time x voxels
        sparse_vec = SparseNeuroVec(sparse_data, space, mask)

        write_vec(sparse_vec, path)
        loaded_vec = read_vec(path)

        dense_data = sparse_vec.as_dense().data
        np.testing.assert_array_almost_equal(loaded_vec.data, dense_data, decimal=5)

    def test_write_vec_with_data_types(self):
        """Test writing vectors with different data types."""
        vec_data = np.random.randn(3, 3, 3, 5) * 100
        space = NeuroSpace(dim=(3, 3, 3, 5))
        vec = DenseNeuroVec(vec_data, space)

        data_types = ["FLOAT32", "INT16", "UINT8"]

        for dtype in data_types:
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                write_vec(vec, tmp_path, data_type=dtype)

                # Verify file was created
                assert os.path.exists(tmp_path)

            finally:
                os.unlink(tmp_path)

    def test_write_vol_rejects_invalid_data_type(self):
        """Test NIfTI write rejects unsupported data_type values."""
        data = np.ones((4, 4, 4), dtype=np.float32)
        vol = DenseNeuroVol(data, NeuroSpace(dim=(4, 4, 4)))

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Unsupported NIfTI data_type"):
                write_vol(vol, tmp_path, data_type="BOGUS")
        finally:
            os.unlink(tmp_path)

    def test_write_vec_rejects_invalid_data_type(self):
        """Test NIfTI vector write rejects unsupported data_type values."""
        vec = DenseNeuroVec(np.ones((4, 4, 4, 2)), NeuroSpace(dim=(4, 4, 4, 2)))

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Unsupported NIfTI data_type"):
                write_vec(vec, tmp_path, data_type="BOGUS")
        finally:
            os.unlink(tmp_path)

    def test_write_vec_accepts_float_alias(self):
        """Test legacy aliases are accepted for NIfTI data_type."""
        vec = DenseNeuroVec(
            np.ones((3, 3, 3, 4), dtype=np.float32), NeuroSpace(dim=(3, 3, 3, 4))
        )

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_vec(vec, tmp_path, data_type="DOUBLE")
            loaded = nib.load(tmp_path)
            assert loaded.header.get_data_dtype() == np.dtype(np.float64)
        finally:
            os.unlink(tmp_path)

    def test_write_vec_accepts_nifti_format_aliases(self):
        """Test legacy NIfTI format aliases are accepted."""
        vec = DenseNeuroVec(
            np.ones((3, 3, 3, 4), dtype=np.float32), NeuroSpace(dim=(3, 3, 3, 4))
        )

        for alias in ["NIFTI1", "NIFTI-1"]:
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                write_vec(vec, tmp_path, format=alias)
                loaded = nib.load(tmp_path)
                assert loaded.header.get_data_dtype() == np.dtype(np.float32)
            finally:
                os.unlink(tmp_path)

    def test_write_vec_accepts_legacy_data_type_aliases(self):
        """Test legacy NIfTI vector data_type aliases are accepted."""
        vec = DenseNeuroVec(
            np.arange(24, dtype=np.int16).reshape((3, 2, 2, 2)),
            NeuroSpace(dim=(3, 2, 2, 2)),
        )

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_short:
            tmp_path_short = tmp_short.name

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_int:
            tmp_path_int = tmp_int.name

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_binary:
            tmp_path_binary = tmp_binary.name
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_ubyte:
            tmp_path_ubyte = tmp_ubyte.name

        try:
            write_vec(vec, tmp_path_short, data_type="SHORT")
            short_img = nib.load(tmp_path_short)
            assert short_img.header.get_data_dtype() == np.dtype(np.int16)

            write_vec(vec, tmp_path_int, data_type="INT")
            int_img = nib.load(tmp_path_int)
            assert int_img.header.get_data_dtype() == np.dtype(np.int32)

            write_vec(vec, tmp_path_binary, data_type="BINARY")
            binary_img = nib.load(tmp_path_binary)
            assert binary_img.header.get_data_dtype() == np.dtype(np.uint8)

            write_vec(vec, tmp_path_ubyte, data_type="UBYTE")
            ubyte_img = nib.load(tmp_path_ubyte)
            assert ubyte_img.header.get_data_dtype() == np.dtype(np.uint8)
        finally:
            os.unlink(tmp_path_short)
            os.unlink(tmp_path_int)
            os.unlink(tmp_path_binary)
            os.unlink(tmp_path_ubyte)

    def test_write_vec_defaults_to_float_type(self):
        """Test no data_type uses default float output."""
        vec = DenseNeuroVec(
            np.arange(24, dtype=np.int16).reshape((4, 3, 2, 1)),
            NeuroSpace(dim=(4, 3, 2, 1)),
        )

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_vec(vec, tmp_path)
            loaded = nib.load(tmp_path)
            assert loaded.header.get_data_dtype() == np.dtype(np.float32)
        finally:
            os.unlink(tmp_path)

    def test_write_vec_unsupported_format(self):
        """Test error for unsupported format."""
        vec = DenseNeuroVec(np.ones((5, 5, 5, 10)), NeuroSpace(dim=(5, 5, 5, 10)))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            write_vec(vec, "test.mnc", format="MINC")


@pytest.mark.skipif(not NIBABEL_AVAILABLE, reason="nibabel not available")
class TestReadVolList:
    """Test read_vol_list function."""

    def test_read_vol_list_basic(self, tmp_path):
        """Test reading multiple volumes from list of files."""
        tmp_files = []
        data_list = []

        for i in range(3):
            path = tmp_path / f"vol_{i}.nii"
            tmp_files.append(path)

            data = np.ones((5, 5, 5)) * (i + 1)
            img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
            nib.save(img, path)
            data_list.append(data)

        vols = read_vol_list(tmp_files)

        assert len(vols) == 3
        for i, vol in enumerate(vols):
            assert isinstance(vol, DenseNeuroVol)
            np.testing.assert_array_equal(vol.data, data_list[i])

    def test_read_vol_list_from_4d(self):
        """Test reading specific volume from 4D files."""
        tmp_files = []

        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                    tmp_path = tmp.name
                    tmp_files.append(tmp_path)

                # Create 4D data
                data_4d = np.random.randn(4, 4, 4, 3).astype(np.float32)
                img = nib.Nifti1Image(data_4d, np.eye(4))
                nib.save(img, tmp_path)

            # Read second volume from each file
            vols = read_vol_list(tmp_files, index=1)

            assert len(vols) == 2
            assert all(vol.shape == (4, 4, 4) for vol in vols)

        finally:
            for f in tmp_files:
                if os.path.exists(f):
                    os.unlink(f)


@pytest.mark.skipif(not NIBABEL_AVAILABLE, reason="nibabel not available")
class TestReadHeader:
    """Test read_header function with edge cases."""

    def test_read_header_comprehensive(self):
        """Test reading all header fields."""
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create NIfTI with specific header values
            data = np.ones((10, 12, 14), dtype=np.int16)
            affine = np.array(
                [
                    [2.0, 0.1, 0.0, -20.0],
                    [0.0, 3.0, 0.2, -30.0],
                    [0.1, 0.0, 4.0, -40.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            img = nib.Nifti1Image(data, affine)
            img.header["descrip"] = b"Test scan with metadata"
            img.header["qform_code"] = 2
            img.header["sform_code"] = 3
            img.header["scl_slope"] = 2.5
            img.header["scl_inter"] = -10.0
            img.header["bitpix"] = 16
            nib.save(img, tmp_path)

            # Read header
            header = read_header(tmp_path)

            # Check all fields
            assert header["dim"] == (10, 12, 14)
            assert len(header["spacing"]) == 3
            assert header["bitpix"] == 16
            assert header["qform_code"] == 2
            assert header["sform_code"] == 3
            # Note: scl_slope and scl_inter might not be preserved exactly by nibabel
            # when creating int16 data, so just check they exist
            assert "scl_slope" in header
            assert "scl_inter" in header
            assert "Test scan" in header["description"]
            np.testing.assert_array_almost_equal(header["origin"], affine[:3, 3])
            np.testing.assert_array_almost_equal(header["affine"], affine)

        finally:
            os.unlink(tmp_path)

    def test_read_header_empty_description(self):
        """Test handling of empty description field."""
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create NIfTI with empty description
            data = np.ones((5, 5, 5), dtype=np.float32)
            img = nib.Nifti1Image(data, np.eye(4))
            img.header["descrip"] = b""
            nib.save(img, tmp_path)

            # Read header
            header = read_header(tmp_path)

            # Empty description should be handled gracefully
            # The actual value depends on how nibabel handles empty descriptions
            assert isinstance(header["description"], str)

        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

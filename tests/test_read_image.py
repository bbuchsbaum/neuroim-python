"""Tests for read_image() dispatch and load_data() convenience function."""

import numpy as np
import nibabel as nib
import pytest
import gzip
from pathlib import Path

from neuroimpy.io import read_image, load_data, read_vol, read_vec
from neuroimpy.sources import NeuroVolSource, NeuroVecSource
from neuroimpy.neuro_vol import DenseNeuroVol
from neuroimpy.neuro_vec import DenseNeuroVec, SparseNeuroVec
from neuroimpy import NeuroSpace, LogicalNeuroVol


@pytest.fixture
def tmp_3d_nifti(tmp_path):
    """Create a temporary 3D NIfTI file."""
    data = np.random.rand(5, 6, 7).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "vol3d.nii.gz"
    nib.save(img, str(path))
    return path, data


@pytest.fixture
def tmp_4d_nifti(tmp_path):
    """Create a temporary 4D NIfTI file."""
    data = np.random.rand(5, 6, 7, 10).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "vec4d.nii.gz"
    nib.save(img, str(path))
    return path, data


def _write_afni_pair(
    out_dir: Path,
    stem: str,
    data: np.ndarray,
    *,
    gzip_data: bool = False,
) -> Path:
    if data.ndim == 3:
        dims = data.shape
        nvols = 1
    else:
        dims = data.shape[:3]
        nvols = data.shape[3]

    head = out_dir / f"{stem}.HEAD"
    brik = out_dir / f"{stem}.BRIK"
    if gzip_data:
        brik = out_dir / f"{stem}.BRIK.gz"

    labels = "~" + "~".join([f"vol{i}" for i in range(nvols)]) + "~"
    head_txt = (
        "type = integer-attribute\n"
        "name = DATASET_DIMENSIONS\n"
        "count = 5\n"
        f"{dims[0]} {dims[1]} {dims[2]} 1 0\n\n"
        "type = integer-attribute\n"
        "name = DATASET_RANK\n"
        f"count = 2\n"
        f"3 {nvols}\n\n"
        "type = float-attribute\n"
        "name = DELTA\n"
        "count = 3\n"
        "1.0 1.0 1.0\n\n"
        "type = float-attribute\n"
        "name = ORIGIN\n"
        "count = 3\n"
        "0.0 0.0 0.0\n\n"
        "type = float-attribute\n"
        "name = IJK_TO_DICOM\n"
        "count = 12\n"
        "1.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 1.0 0.0\n\n"
        "type = integer-attribute\n"
        "name = BRICK_TYPES\n"
        f"count = {nvols}\n"
        + "3 " * (nvols - 1) + "3\n\n"
        "type = float-attribute\n"
        "name = BRICK_FLOAT_FACS\n"
        f"count = {nvols}\n"
        + "1.0 " * (nvols - 1) + "1.0\n\n"
        "type = string-attribute\n"
        "name = BYTEORDER_STRING\n"
        "count = 10\n"
        "~LSB_FIRST~\n\n"
        "type = string-attribute\n"
        "name = BRICK_LABS\n"
        f"count = {len(labels)}\n"
        f"{labels}\n"
    )

    head.write_text(head_txt, encoding="utf-8")
    raw = np.asarray(data, dtype=np.float32).ravel(order="F").tobytes()
    if gzip_data:
        with gzip.open(brik, "wb") as f:
            f.write(raw)
    else:
        brik.write_bytes(raw)
    return head


class TestReadImage:
    """Tests for read_image() auto-dispatch."""

    def test_3d_returns_vol(self, tmp_3d_nifti):
        path, data = tmp_3d_nifti
        result = read_image(path)
        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (5, 6, 7)
        np.testing.assert_allclose(result.data, data, atol=1e-6)

    def test_4d_returns_vec(self, tmp_4d_nifti):
        path, data = tmp_4d_nifti
        result = read_image(path)
        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (5, 6, 7, 10)
        np.testing.assert_allclose(result.data, data, atol=1e-6)

    def test_4d_singleton_time_dimension_returns_vol(self, tmp_path):
        data = np.random.rand(5, 6, 7, 1).astype(np.float32)
        path = tmp_path / "vec_singleton.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

        result = read_image(path)

        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (5, 6, 7)
        np.testing.assert_allclose(result.data, data[..., 0], atol=1e-6)

    def test_3d_kwargs_forwarded(self, tmp_4d_nifti):
        """read_image on a 4D file passes kwargs like indices to read_vec."""
        path, data = tmp_4d_nifti
        result = read_image(path, indices=[0, 1, 2])
        assert isinstance(result, DenseNeuroVec)
        assert result.shape[3] == 3

    def test_read_image_index_is_forwarded_for_single_4d(self, tmp_4d_nifti):
        path, data = tmp_4d_nifti
        result = read_image(path, index=2)

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (5, 6, 7, 1)
        np.testing.assert_allclose(result.data[..., 0], data[..., 2], atol=1e-6)

    def test_type_vol_forces_volume(self, tmp_4d_nifti):
        path, data = tmp_4d_nifti
        result = read_image(path, type="vol", index=1)

        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (5, 6, 7)
        np.testing.assert_allclose(result.data, data[..., 1], atol=1e-6)

    def test_type_vec_forces_vector_and_index(self, tmp_4d_nifti):
        path, data = tmp_4d_nifti
        result = read_image(path, type="vec", index=1)

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (5, 6, 7, 1)
        np.testing.assert_allclose(result.data[..., 0], data[..., 1], atol=1e-6)

    def test_type_vec_3d_returns_vector(self, tmp_3d_nifti):
        path, data = tmp_3d_nifti
        result = read_image(path, type="vec")

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (5, 6, 7, 1)
        np.testing.assert_allclose(result.data[..., 0], data, atol=1e-6)

    def test_type_vec_singleton_list(self, tmp_3d_nifti):
        path, data = tmp_3d_nifti
        result = read_image([path], type="vec", index=0)

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (5, 6, 7, 1)
        np.testing.assert_allclose(result.data[..., 0], data, atol=1e-6)

    def test_type_vol_rejects_multiple_files(self, tmp_3d_nifti):
        path, _ = tmp_3d_nifti
        extra = np.random.rand(5, 6, 7).astype(np.float32)
        path2 = path.with_name("second.nii.gz")
        nib.save(nib.Nifti1Image(extra, np.eye(4)), str(path2))

        with pytest.raises(ValueError, match="type='vol' expects a single file_name"):
            read_image([path, path2], type="vol")

    def test_type_vol_accepts_singleton_list(self, tmp_3d_nifti):
        path, data = tmp_3d_nifti
        path_list = [path]

        result = read_image(path_list, type="vol")

        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (5, 6, 7)
        np.testing.assert_allclose(result.data, data, atol=1e-6)

    def test_indexing_3d_image_raises_in_read_image(self, tmp_3d_nifti):
        path, _ = tmp_3d_nifti
        with pytest.raises(ValueError, match="index 1 invalid"):
            read_image(path, index=1)

    def test_list_with_index_does_not_error(self, tmp_path):
        data1 = np.random.rand(4, 3, 2).astype(np.float32)
        data2 = np.random.rand(4, 3, 2).astype(np.float32)
        path1 = tmp_path / "vol1.nii.gz"
        path2 = tmp_path / "vol2.nii.gz"
        nib.save(nib.Nifti1Image(data1, np.eye(4)), str(path1))
        nib.save(nib.Nifti1Image(data2, np.eye(4)), str(path2))

        result = read_image([path1, path2], index=1)

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (4, 3, 2, 2)
        assert np.allclose(result.data[..., 0], data1)
        assert np.allclose(result.data[..., 1], data2)

    def test_string_path(self, tmp_3d_nifti):
        path, _ = tmp_3d_nifti
        result = read_image(str(path))
        assert isinstance(result, DenseNeuroVol)

    def test_pathlib_path(self, tmp_3d_nifti):
        path, _ = tmp_3d_nifti
        result = read_image(path)
        assert isinstance(result, DenseNeuroVol)

    def test_afni_path(self, tmp_path):
        data = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
        head = _write_afni_pair(tmp_path, "vol+orig", data)

        result = read_image(head)

        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (4, 3, 2)
        np.testing.assert_allclose(result.data, data, atol=1e-6)

    def test_afni_4d_singleton_time_dimension_returns_vol(self, tmp_path):
        data = np.arange(4 * 3 * 2 * 1, dtype=np.float32).reshape((4, 3, 2, 1), order="F")
        head = _write_afni_pair(tmp_path, "vol4d_singleton+orig", data)

        result = read_image(head)

        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (4, 3, 2)
        np.testing.assert_allclose(result.data, data[..., 0], atol=1e-6)

    def test_afni_4d_auto_dispatch_returns_vec(self, tmp_path):
        data = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
        head = _write_afni_pair(tmp_path, "vol4d+orig", data)

        result = read_image(head)

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (4, 3, 2, 2)
        np.testing.assert_allclose(result.data, data, atol=1e-6)

    def test_multiple_files_vec(self, tmp_path):
        data1 = np.random.rand(4, 3, 2).astype(np.float32)
        data2 = np.random.rand(4, 3, 2).astype(np.float32)
        path1 = tmp_path / "vol1.nii.gz"
        path2 = tmp_path / "vol2.nii.gz"
        nib.save(nib.Nifti1Image(data1, np.eye(4)), str(path1))
        nib.save(nib.Nifti1Image(data2, np.eye(4)), str(path2))

        result = read_image([path1, path2])

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (4, 3, 2, 2)
        assert np.allclose(result.data[..., 0], data1)
        assert np.allclose(result.data[..., 1], data2)

    def test_multiple_files_vec_index_ignored(self, tmp_path):
        base = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
        data1 = base + 0.0
        data2 = base + 100.0
        path1 = tmp_path / "vol1.nii.gz"
        path2 = tmp_path / "vol2.nii.gz"
        nib.save(nib.Nifti1Image(data1, np.eye(4)), str(path1))
        nib.save(nib.Nifti1Image(data2, np.eye(4)), str(path2))

        result = read_image([path1, path2], type="vec", index=1)

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (4, 3, 2, 4)
        np.testing.assert_allclose(result.data[..., 0], data1[..., 0], atol=1e-6)
        np.testing.assert_allclose(result.data[..., 1], data1[..., 1], atol=1e-6)
        np.testing.assert_allclose(result.data[..., 2], data2[..., 0], atol=1e-6)
        np.testing.assert_allclose(result.data[..., 3], data2[..., 1], atol=1e-6)

    def test_multiple_files_vec_indices_explicit(self, tmp_path):
        data1 = np.random.rand(4, 3, 2, 3).astype(np.float32)
        data2 = np.random.rand(4, 3, 2, 3).astype(np.float32)
        path1 = tmp_path / "vol1.nii.gz"
        path2 = tmp_path / "vol2.nii.gz"
        nib.save(nib.Nifti1Image(data1, np.eye(4)), str(path1))
        nib.save(nib.Nifti1Image(data2, np.eye(4)), str(path2))

        result = read_image([path1, path2], type="vec", indices=[1])

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (4, 3, 2, 2)
        assert np.allclose(result.data[..., 0], data1[..., 1], atol=1e-6)
        assert np.allclose(result.data[..., 1], data2[..., 1], atol=1e-6)

    def test_multiple_files_vec_with_mask(self, tmp_path):
        data1 = np.random.rand(4, 3, 2, 2).astype(np.float32)
        data2 = np.random.rand(4, 3, 2, 2).astype(np.float32)
        path1 = tmp_path / "vol1.nii.gz"
        path2 = tmp_path / "vol2.nii.gz"
        nib.save(nib.Nifti1Image(data1, np.eye(4)), str(path1))
        nib.save(nib.Nifti1Image(data2, np.eye(4)), str(path2))

        mask_data = np.zeros((4, 3, 2), dtype=bool)
        mask_data[0, 0, 0] = True
        mask = LogicalNeuroVol(mask_data, NeuroSpace(dim=(4, 3, 2)))

        result = read_image([path1, path2], type="vec", mask=mask, indices=[0, 1])

        assert result.shape == (4, 3, 2, 4)
        assert isinstance(result, SparseNeuroVec)
        dense = result.as_dense()
        assert np.allclose(dense.data[0, 0, 0, :2], data1[0, 0, 0, :])
        assert np.allclose(dense.data[0, 0, 0, 2:], data2[0, 0, 0, :])

    def test_multiple_files_afni_vec(self, tmp_path):
        data1 = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
        data2 = (np.arange(4 * 3 * 2, dtype=np.float32) + 100).reshape((4, 3, 2), order="F")
        head1 = _write_afni_pair(tmp_path, "vol1+orig", data1)
        head2 = _write_afni_pair(tmp_path, "vol2+orig", data2)

        result = read_image([head1, head2])

        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (4, 3, 2, 2)
        np.testing.assert_allclose(result.data[..., 0], data1)
        np.testing.assert_allclose(result.data[..., 1], data2)


class TestLoadData:
    """Tests for load_data() convenience wrapper."""

    def test_load_vol_source(self, tmp_3d_nifti):
        path, data = tmp_3d_nifti
        source = NeuroVolSource(path)
        result = load_data(source)
        assert isinstance(result, DenseNeuroVol)
        assert result.shape == (5, 6, 7)
        np.testing.assert_allclose(result.data, data, atol=1e-6)

    def test_load_vec_source(self, tmp_4d_nifti):
        path, data = tmp_4d_nifti
        source = NeuroVecSource(path)
        result = load_data(source)
        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (5, 6, 7, 10)

    def test_load_caches(self, tmp_3d_nifti):
        """Calling load_data twice returns the same cached object."""
        path, _ = tmp_3d_nifti
        source = NeuroVolSource(path)
        r1 = load_data(source)
        r2 = load_data(source)
        # Second call should return the cached _loaded attribute
        assert r2 is source._loaded

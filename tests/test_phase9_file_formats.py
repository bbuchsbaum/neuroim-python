"""Tests for Phase 9: File Format Support."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import shutil

from neuroimpy import NeuroSpace, DenseNeuroVol

# Try to import file format functions (may not be available without nibabel)
try:
    from neuroimpy.file_format import (
        FileFormat, NIFTIFormat, AFNIFormat,
        NIFTI, NIFTI_GZ, NIFTI_PAIR, NIFTI_PAIR_GZ, AFNI, AFNI_GZ,
        find_descriptor
    )
    from neuroimpy.nifti_utils import (
        create_nifti_header, as_nifti_header,
        matrix_to_quatern, quatern_to_matrix
    )
    HAS_FILE_FORMAT = True
except ImportError:
    HAS_FILE_FORMAT = False


@pytest.mark.skipif(not HAS_FILE_FORMAT, reason="File format support not available (nibabel required)")
class TestFileFormatConstants:
    """Test file format constants."""
    
    def test_nifti_format(self):
        """Test NIFTI format constant."""
        assert NIFTI.file_format == "NIFTI"
        assert NIFTI.header_extension == "nii"
        assert NIFTI.data_extension == "nii"
        assert NIFTI.header_encoding == "raw"
        assert NIFTI.data_encoding == "raw"
    
    def test_nifti_gz_format(self):
        """Test NIFTI_GZ format constant."""
        assert NIFTI_GZ.file_format == "NIFTI"
        assert NIFTI_GZ.header_extension == "nii.gz"
        assert NIFTI_GZ.data_extension == "nii.gz"
        assert NIFTI_GZ.header_encoding == "gzip"
        assert NIFTI_GZ.data_encoding == "gzip"
    
    def test_nifti_pair_format(self):
        """Test NIFTI_PAIR format constant."""
        assert NIFTI_PAIR.file_format == "NIFTI"
        assert NIFTI_PAIR.header_extension == "hdr"
        assert NIFTI_PAIR.data_extension == "img"
        assert NIFTI_PAIR.header_encoding == "raw"
        assert NIFTI_PAIR.data_encoding == "raw"
    
    def test_nifti_pair_gz_format(self):
        """Test NIFTI_PAIR_GZ format constant."""
        assert NIFTI_PAIR_GZ.file_format == "NIFTI"
        assert NIFTI_PAIR_GZ.header_extension == "hdr.gz"
        assert NIFTI_PAIR_GZ.data_extension == "img.gz"
        assert NIFTI_PAIR_GZ.header_encoding == "gzip"
        assert NIFTI_PAIR_GZ.data_encoding == "gzip"
    
    def test_afni_format(self):
        """Test AFNI format constant."""
        assert AFNI.file_format == "AFNI"
        assert AFNI.header_extension == "HEAD"
        assert AFNI.data_extension == "BRIK"
        assert AFNI.header_encoding == "raw"
        assert AFNI.data_encoding == "raw"
    
    def test_afni_gz_format(self):
        """Test AFNI_GZ format constant."""
        assert AFNI_GZ.file_format == "AFNI"
        assert AFNI_GZ.header_extension == "HEAD"
        assert AFNI_GZ.data_extension == "BRIK.gz"
        assert AFNI_GZ.header_encoding == "gzip"
        assert AFNI_GZ.data_encoding == "gzip"


@pytest.mark.skipif(not HAS_FILE_FORMAT, reason="File format support not available (nibabel required)")
class TestFileFormatMethods:
    """Test FileFormat methods."""
    
    def setup_method(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_header_file_matches(self):
        """Test header file matching."""
        assert NIFTI.header_file_matches("test.nii")
        assert not NIFTI.header_file_matches("test.img")
        assert NIFTI_GZ.header_file_matches("test.nii.gz")
        assert NIFTI_PAIR.header_file_matches("test.hdr")
        assert not NIFTI_PAIR.header_file_matches("test.nii")
    
    def test_data_file_matches(self):
        """Test data file matching."""
        assert NIFTI.data_file_matches("test.nii")
        assert not NIFTI.data_file_matches("test.hdr")
        assert NIFTI_PAIR.data_file_matches("test.img")
        assert not NIFTI_PAIR.data_file_matches("test.nii")
    
    def test_strip_extension(self):
        """Test stripping file extensions."""
        assert NIFTI.strip_extension("test.nii") == "test"
        assert NIFTI_GZ.strip_extension("test.nii.gz") == "test"
        assert NIFTI_PAIR.strip_extension("test.hdr") == "test"
        assert NIFTI_PAIR.strip_extension("test.img") == "test"
        
        with pytest.raises(ValueError):
            NIFTI.strip_extension("test.txt")
    
    def test_header_file(self):
        """Test getting header file name."""
        assert NIFTI.header_file("test.nii") == "test.nii"
        assert NIFTI_PAIR.header_file("test.img") == "test.hdr"
        assert NIFTI_PAIR.header_file("test.hdr") == "test.hdr"
        
        with pytest.raises(ValueError):
            NIFTI.header_file("test.txt")
    
    def test_data_file(self):
        """Test getting data file name."""
        assert NIFTI.data_file("test.nii") == "test.nii"
        assert NIFTI_PAIR.data_file("test.hdr") == "test.img"
        assert NIFTI_PAIR.data_file("test.img") == "test.img"
        
        with pytest.raises(ValueError):
            NIFTI.data_file("test.txt")
    
    def test_file_matches_with_files(self):
        """Test file matching with actual files."""
        # Create paired files
        hdr_path = os.path.join(self.temp_dir, "test.hdr")
        img_path = os.path.join(self.temp_dir, "test.img")
        
        # Test when both files exist
        Path(hdr_path).touch()
        Path(img_path).touch()
        
        assert NIFTI_PAIR.file_matches(hdr_path)
        assert NIFTI_PAIR.file_matches(img_path)
        
        # Test when only one file exists
        os.remove(img_path)
        assert not NIFTI_PAIR.file_matches(hdr_path)
        
        # Test single file format
        nii_path = os.path.join(self.temp_dir, "test.nii")
        Path(nii_path).touch()
        assert NIFTI.file_matches(nii_path)


@pytest.mark.skipif(not HAS_FILE_FORMAT, reason="File format support not available (nibabel required)")
class TestFindDescriptor:
    """Test find_descriptor function."""
    
    def setup_method(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_find_descriptor_single_file(self):
        """Test finding descriptor for single file formats."""
        # Create test files
        nii_path = os.path.join(self.temp_dir, "test.nii")
        Path(nii_path).touch()
        
        desc = find_descriptor(nii_path)
        assert desc is not None
        assert desc.file_format == "NIFTI"
        assert desc.header_extension == "nii"
    
    def test_find_descriptor_paired_files(self):
        """Test finding descriptor for paired file formats."""
        # Create paired files
        hdr_path = os.path.join(self.temp_dir, "test.hdr")
        img_path = os.path.join(self.temp_dir, "test.img")
        Path(hdr_path).touch()
        Path(img_path).touch()
        
        desc = find_descriptor(hdr_path)
        assert desc is not None
        assert desc.file_format == "NIFTI"
        assert desc.header_extension == "hdr"
        assert desc.data_extension == "img"
    
    def test_find_descriptor_no_match(self):
        """Test when no format matches."""
        txt_path = os.path.join(self.temp_dir, "test.txt")
        Path(txt_path).touch()
        
        desc = find_descriptor(txt_path)
        assert desc is None


@pytest.mark.skipif(not HAS_FILE_FORMAT, reason="File format support not available (nibabel required)")
class TestQuaternionFunctions:
    """Test quaternion conversion functions."""
    
    def test_matrix_to_quatern_identity(self):
        """Test converting identity matrix to quaternion."""
        mat = np.eye(4)
        result = matrix_to_quatern(mat)
        
        assert 'quaternion' in result
        assert 'qfac' in result
        assert len(result['quaternion']) == 3
        assert result['qfac'] == 1.0
        
        # For identity matrix, quaternion should be close to [0, 0, 0]
        assert np.allclose(result['quaternion'], [0, 0, 0], atol=1e-10)
    
    def test_matrix_to_quatern_rotation(self):
        """Test converting rotation matrix to quaternion."""
        # 90-degree rotation around z-axis
        angle = np.pi / 2
        mat = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        result = matrix_to_quatern(mat)
        assert result['qfac'] == 1.0
        
        # Should have non-zero quaternion components
        assert not np.allclose(result['quaternion'], [0, 0, 0])
    
    def test_matrix_to_quatern_negative_determinant(self):
        """Test matrix with negative determinant."""
        # Reflection matrix
        mat = np.diag([1, 1, -1, 1])
        
        result = matrix_to_quatern(mat)
        assert result['qfac'] == -1.0
    
    def test_quatern_to_matrix_identity(self):
        """Test converting identity quaternion to matrix."""
        # Identity quaternion [0, 0, 0] with implicit a=1
        mat = quatern_to_matrix([0, 0, 0], qfac=1.0)
        
        # Should be identity rotation
        assert np.allclose(mat[:3, :3], np.eye(3))
    
    def test_quatern_to_matrix_with_offset(self):
        """Test quaternion to matrix with translation."""
        qoffset = [10, 20, 30]
        mat = quatern_to_matrix([0, 0, 0], qfac=1.0, qoffset=qoffset)
        
        assert np.allclose(mat[:3, 3], qoffset)
    
    def test_quaternion_round_trip(self):
        """Test converting matrix to quaternion and back."""
        # Create a random rotation matrix
        # Using QR decomposition to get a proper rotation
        random_mat = np.random.randn(3, 3)
        q, r = np.linalg.qr(random_mat)
        # Ensure proper rotation (det = 1)
        if np.linalg.det(q) < 0:
            q[:, -1] = -q[:, -1]
        
        mat = np.eye(4)
        mat[:3, :3] = q
        mat[:3, 3] = [5, 10, 15]  # Add translation
        
        # Convert to quaternion and back
        quat_info = matrix_to_quatern(mat)
        mat_reconstructed = quatern_to_matrix(
            quat_info['quaternion'], 
            quat_info['qfac'],
            mat[:3, 3]
        )
        
        # Should be close to original (within numerical precision)
        assert np.allclose(mat, mat_reconstructed, atol=1e-10)


@pytest.mark.skipif(not HAS_FILE_FORMAT, reason="File format support not available (nibabel required)")
class TestNIfTIHeader:
    """Test NIfTI header functions."""
    
    def test_create_nifti_header_single_file(self):
        """Test creating NIfTI header for single file."""
        header = create_nifti_header(one_file=True)
        
        assert header['sizeof_hdr'] == 348
        assert header['magic'] == 'n+1'
        assert header['vox_offset'] == 352.0
        assert header['dim'][0] == 3  # Default 3D
        assert header['pixdim'][0] == 1.0
    
    def test_create_nifti_header_paired_files(self):
        """Test creating NIfTI header for paired files."""
        header = create_nifti_header(one_file=False)
        
        assert header['magic'] == 'ni1'
        assert header['vox_offset'] == 0.0
    
    def test_create_nifti_header_with_filename(self):
        """Test creating header with filename."""
        filename = "test_volume.nii"
        header = create_nifti_header(file_name=filename)
        
        assert filename in header['descrip']
    
    def test_as_nifti_header(self):
        """Test creating NIfTI header from NeuroVol."""
        # Create a test volume
        space = NeuroSpace(
            dim=[64, 64, 32],
            spacing=[2.0, 2.0, 3.0],
            origin=[10, 20, 30]
        )
        vol = DenseNeuroVol(np.zeros((64, 64, 32)), space)
        
        header = as_nifti_header(vol, "test.nii")
        
        # Check dimensions
        assert header['dim'][0] == 3
        assert header['dim'][1] == 64
        assert header['dim'][2] == 64
        assert header['dim'][3] == 32
        
        # Check voxel dimensions
        assert header['pixdim'][1] == 2.0
        assert header['pixdim'][2] == 2.0
        assert header['pixdim'][3] == 3.0
        
        # Check data type
        assert header['datatype'] == 16  # FLOAT32
        assert header['bitpix'] == 32
        
        # Check coordinate info
        assert header['qform_code'] == 1
        assert header['sform_code'] == 1
        
        # Check offsets
        assert header['qoffset_x'] == 10
        assert header['qoffset_y'] == 20
        assert header['qoffset_z'] == 30
    
    def test_as_nifti_header_data_types(self):
        """Test different data types."""
        space = NeuroSpace(dim=[10, 10, 10])
        vol = DenseNeuroVol(np.zeros((10, 10, 10)), space)
        
        # Test different data types
        for dtype_str, expected_code in [
            ("FLOAT", 16),
            ("DOUBLE", 64),
            ("INT16", 4),
            ("UINT8", 2),
            ("BINARY", 2),
            ("UBYTE", 2),
        ]:
            header = as_nifti_header(vol, "test.nii", data_type=dtype_str)
            assert header['datatype'] == expected_code


@pytest.mark.skipif(not HAS_FILE_FORMAT, reason="File format support not available (nibabel required)")
class TestAFNIFormat:
    """Test AFNI format metadata parsing."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def _write_afni_pair(self, stem: str, data: np.ndarray, float_facs=None):
        head = os.path.join(self.temp_dir, f"{stem}.HEAD")
        brik = os.path.join(self.temp_dir, f"{stem}.BRIK")
        if data.ndim == 3:
            dims = data.shape
            nvols = 1
        else:
            dims = data.shape[:3]
            nvols = data.shape[3]
        if float_facs is None:
            float_facs = [1.0] * nvols
        brick_types = " ".join(["3"] * nvols)
        facs = " ".join(str(x) for x in float_facs)
        labels = "~" + "~".join([f"vol{i}" for i in range(nvols)]) + "~"
        head_txt = (
            "type = integer-attribute\n"
            "name = DATASET_DIMENSIONS\n"
            "count = 5\n"
            f"{dims[0]} {dims[1]} {dims[2]} 1 0\n\n"
            "type = integer-attribute\n"
            "name = DATASET_RANK\n"
            "count = 2\n"
            f"3 {nvols}\n\n"
            "type = float-attribute\n"
            "name = DELTA\n"
            "count = 3\n"
            "2.0 -2.0 3.0\n\n"
            "type = float-attribute\n"
            "name = ORIGIN\n"
            "count = 3\n"
            "10.0 20.0 30.0\n\n"
            "type = float-attribute\n"
            "name = IJK_TO_DICOM\n"
            "count = 12\n"
            "2.0 0.0 0.0 10.0 0.0 -2.0 0.0 20.0 0.0 0.0 3.0 30.0\n\n"
            "type = integer-attribute\n"
            "name = BRICK_TYPES\n"
            f"count = {nvols}\n"
            f"{brick_types}\n\n"
            "type = float-attribute\n"
            "name = BRICK_FLOAT_FACS\n"
            f"count = {nvols}\n"
            f"{facs}\n\n"
            "type = string-attribute\n"
            "name = BYTEORDER_STRING\n"
            "count = 10\n"
            "~LSB_FIRST~\n\n"
            "type = string-attribute\n"
            "name = BRICK_LABS\n"
            f"count = {len(labels)}\n"
            f"{labels}\n"
        )
        with open(head, "w", encoding="utf-8") as f:
            f.write(head_txt)
        np.asarray(data, dtype=np.float32).ravel(order="F").tofile(brik)
        return head

    def test_afni_read_meta_info(self):
        data = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
        head = self._write_afni_pair("mini+orig", data, float_facs=[1.0, 2.0])

        meta = AFNI.read_meta_info(head)

        assert meta.data_file.endswith(".BRIK")
        assert meta.dims == (4, 3, 2, 2)
        assert meta.spacing == (2.0, 2.0, 3.0)
        assert meta.origin == (10.0, 20.0, 30.0)
        assert meta.data_type == "FLOAT32"
        np.testing.assert_array_equal(np.asarray(meta.slope), np.asarray([1.0, 2.0]))

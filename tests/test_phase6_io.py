"""Tests for Phase 6: File I/O and Formats.

Direct translation of R's neuroim2 tests for file format and I/O functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

try:
    from neuroim import (
        # File formats
        FileFormat, NIFTIFormat, AFNIFormat,
        NIFTI, NIFTI_GZ, NIFTI_PAIR, NIFTI_PAIR_GZ, AFNI, AFNI_GZ,
        find_descriptor,
        # Metadata
        MetaInfo, FileMetaInfo, NIFTIMetaInfo, AFNIMetaInfo,
        # Binary I/O
        BinaryReader, BinaryWriter, ColumnReader,
        # I/O functions
        read_header, read_meta_info,
        # Supporting classes
        DenseNeuroVol, NeuroSpace, find_anatomy_3d
    )
    NIBABEL_AVAILABLE = True
except ImportError:
    # Import components that don't require nibabel
    from neuroim.file_format import (
        FileFormat, NIFTIFormat, AFNIFormat,
        NIFTI, NIFTI_GZ, NIFTI_PAIR, NIFTI_PAIR_GZ, AFNI, AFNI_GZ,
        find_descriptor
    )
    from neuroim.meta_info import MetaInfo, FileMetaInfo, NIFTIMetaInfo, AFNIMetaInfo
    from neuroim.binary_io import BinaryReader, BinaryWriter, ColumnReader
    from neuroim import DenseNeuroVol, NeuroSpace, find_anatomy_3d
    NIBABEL_AVAILABLE = False


class TestFileFormat:
    """Test FileFormat class functionality."""
    
    def test_nifti_format_attributes(self):
        """Test NIFTI format has correct attributes."""
        assert NIFTI.file_format == "NIFTI"
        assert NIFTI.header_encoding == "raw"
        assert NIFTI.header_extension == "nii"
        assert NIFTI.data_encoding == "raw"
        assert NIFTI.data_extension == "nii"
    
    def test_nifti_gz_format_attributes(self):
        """Test NIFTI_GZ format has correct attributes."""
        assert NIFTI_GZ.file_format == "NIFTI"
        assert NIFTI_GZ.header_encoding == "gzip"
        assert NIFTI_GZ.header_extension == "nii.gz"
        assert NIFTI_GZ.data_encoding == "gzip"
        assert NIFTI_GZ.data_extension == "nii.gz"
    
    def test_nifti_pair_format_attributes(self):
        """Test NIFTI_PAIR format has correct attributes."""
        assert NIFTI_PAIR.file_format == "NIFTI"
        assert NIFTI_PAIR.header_encoding == "raw"
        assert NIFTI_PAIR.header_extension == "hdr"
        assert NIFTI_PAIR.data_encoding == "raw"
        assert NIFTI_PAIR.data_extension == "img"
    
    def test_file_matching(self):
        """Test file format matching."""
        # Single file NIfTI
        assert NIFTI.header_file_matches("brain.nii")
        assert NIFTI.data_file_matches("brain.nii")
        assert not NIFTI.header_file_matches("brain.nii.gz")
        
        # Compressed NIfTI
        assert NIFTI_GZ.header_file_matches("brain.nii.gz")
        assert NIFTI_GZ.data_file_matches("brain.nii.gz")
        assert not NIFTI_GZ.header_file_matches("brain.nii")
        
        # Paired NIfTI
        assert NIFTI_PAIR.header_file_matches("brain.hdr")
        assert NIFTI_PAIR.data_file_matches("brain.img")
        assert not NIFTI_PAIR.header_file_matches("brain.img")
        assert not NIFTI_PAIR.data_file_matches("brain.hdr")
    
    def test_strip_extension(self):
        """Test extension stripping."""
        assert NIFTI.strip_extension("brain.nii") == "brain"
        assert NIFTI_GZ.strip_extension("brain.nii.gz") == "brain"
        assert NIFTI_PAIR.strip_extension("brain.hdr") == "brain"
        assert NIFTI_PAIR.strip_extension("brain.img") == "brain"
        
        # Test with paths
        assert NIFTI.strip_extension("/path/to/brain.nii") == "/path/to/brain"
        assert NIFTI_GZ.strip_extension("./data/brain.nii.gz") == "./data/brain"
    
    def test_header_and_data_file_derivation(self):
        """Test deriving header and data file names."""
        # Single file format
        assert NIFTI.header_file("brain.nii") == "brain.nii"
        assert NIFTI.data_file("brain.nii") == "brain.nii"
        
        # Paired format
        assert NIFTI_PAIR.header_file("brain.hdr") == "brain.hdr"
        assert NIFTI_PAIR.header_file("brain.img") == "brain.hdr"
        assert NIFTI_PAIR.data_file("brain.hdr") == "brain.img"
        assert NIFTI_PAIR.data_file("brain.img") == "brain.img"
    
    def test_file_matches_with_missing_pair(self):
        """Test file_matches when paired file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only header file
            hdr_file = Path(tmpdir) / "brain.hdr"
            hdr_file.touch()
            
            # Should return False because .img is missing
            assert not NIFTI_PAIR.file_matches(str(hdr_file))
            
            # Create data file
            img_file = Path(tmpdir) / "brain.img"
            img_file.touch()
            
            # Now should return True
            assert NIFTI_PAIR.file_matches(str(hdr_file))
            assert NIFTI_PAIR.file_matches(str(img_file))
    
    def test_find_descriptor(self):
        """Test finding appropriate descriptor for files."""
        assert find_descriptor("brain.nii") is None  # File doesn't exist
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            nii_file = Path(tmpdir) / "brain.nii"
            nii_file.touch()
            
            nii_gz_file = Path(tmpdir) / "brain2.nii.gz"
            nii_gz_file.touch()
            
            # Test descriptor finding
            desc = find_descriptor(str(nii_file))
            assert desc is not None
            assert desc.header_extension == "nii"
            
            desc_gz = find_descriptor(str(nii_gz_file))
            assert desc_gz is not None
            assert desc_gz.header_extension == "nii.gz"


class TestMetaInfo:
    """Test MetaInfo classes."""
    
    def test_meta_info_creation(self):
        """Test creating MetaInfo object."""
        spatial_axes = find_anatomy_3d("RAS")
        from neuroim.axis import AxisSet
        
        meta = MetaInfo(
            data_type="FLOAT32",
            dims=(64, 64, 32),
            spatial_axes=spatial_axes,
            additional_axes=AxisSet(0),
            spacing=(3.0, 3.0, 4.0),
            origin=(0.0, 0.0, 0.0),
            label=["scan1"]
        )
        
        assert meta.data_type == "FLOAT32"
        assert meta.dims == (64, 64, 32)
        assert meta.spacing == (3.0, 3.0, 4.0)
        assert meta.origin == (0.0, 0.0, 0.0)
        assert meta.label == ["scan1"]
        assert meta.ndim == 3
        assert meta.nvols == 1
    
    def test_file_meta_info_creation(self):
        """Test creating FileMetaInfo object."""
        file_meta = FileMetaInfo(
            header_file="brain.nii",
            data_file="brain.nii",
            descriptor=NIFTI,
            data_type="FLOAT32",
            dims=(64, 64, 32),
            spacing=(3.0, 3.0, 4.0),
            origin=(0.0, 0.0, 0.0),
            endian="little",
            data_offset=352,
            bytes_per_element=4,
            intercept=0.0,
            slope=1.0
        )
        
        assert file_meta.header_file == "brain.nii"
        assert file_meta.data_file == "brain.nii"
        assert file_meta.descriptor == NIFTI
        assert file_meta.endian == "little"
        assert file_meta.data_offset == 352
        assert file_meta.bytes_per_element == 4
        assert file_meta.byte_order == '<'
        
        # Test numpy dtype
        dtype = file_meta.get_data_dtype()
        assert dtype == np.dtype('<f4')
    
    def test_nifti_meta_info(self):
        """Test NIFTIMetaInfo specific features."""
        nifti_header = {
            'qform_code': 1,
            'sform_code': 2,
            'descrip': b'Test description\x00\x00'
        }
        
        nifti_meta = NIFTIMetaInfo(
            nifti_header=nifti_header,
            header_file="brain.nii",
            data_file="brain.nii",
            descriptor=NIFTI,
            data_type="FLOAT32",
            dims=(64, 64, 32),
            spacing=(3.0, 3.0, 4.0),
            origin=(0.0, 0.0, 0.0)
        )
        
        assert nifti_meta.qform_code == 1
        assert nifti_meta.sform_code == 2
        assert nifti_meta.descrip == "Test description"


class TestBinaryIO:
    """Test binary I/O classes."""
    
    def test_binary_writer_reader_roundtrip(self):
        """Test writing and reading binary data."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write data
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
            
            with BinaryWriter(tmp_path, data_type="float32") as writer:
                writer.write_elements(data)
            
            # Read data back
            with BinaryReader(tmp_path, data_type="float32") as reader:
                read_data = reader.read_elements(5)
            
            np.testing.assert_array_equal(data, read_data)
            
        finally:
            os.unlink(tmp_path)
    
    def test_binary_io_with_offset(self):
        """Test binary I/O with byte offset."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write with offset
            offset = 100
            data = np.array([10, 20, 30, 40], dtype=np.int32)
            
            with BinaryWriter(tmp_path, byte_offset=offset, 
                            data_type="int32", bytes_per_element=4) as writer:
                writer.write_elements(data)
            
            # Read with offset
            with BinaryReader(tmp_path, byte_offset=offset,
                            data_type="int32", bytes_per_element=4) as reader:
                read_data = reader.read_elements(4)
            
            np.testing.assert_array_equal(data, read_data)
            
        finally:
            os.unlink(tmp_path)
    
    def test_binary_endianness(self):
        """Test binary I/O with different endianness."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            data = np.array([1.0, 2.0, 3.0], dtype='>f4')  # Big endian
            
            # Write big endian
            with BinaryWriter(tmp_path, data_type="float32", endian="big") as writer:
                writer.write_elements(data)
            
            # Read as big endian
            with BinaryReader(tmp_path, data_type="float32", endian="big") as reader:
                read_data = reader.read_elements(3)
            
            np.testing.assert_array_almost_equal(data, read_data)
            
        finally:
            os.unlink(tmp_path)
    
    def test_column_reader(self):
        """Test ColumnReader for column-oriented data."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create column data (3 rows, 4 columns)
            data = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]], dtype=np.float32)
            
            # Write in column-major order
            with BinaryWriter(tmp_path, data_type="float32") as writer:
                writer.write_elements(data.T.ravel())  # Transpose for column-major
            
            # Read columns
            with ColumnReader(tmp_path, n_cols=4, n_rows=3, data_type="float32") as reader:
                # Read all columns
                all_cols = reader.read_all_columns()
                np.testing.assert_array_equal(all_cols, data)
                
            # Read individual column
            with ColumnReader(tmp_path, n_cols=4, n_rows=3, data_type="float32") as reader:
                col1 = reader.read_column(1)  # Second column
                np.testing.assert_array_equal(col1, data[:, 1])
                
        finally:
            os.unlink(tmp_path)


class TestIOFunctions:
    """Test high-level I/O functions."""
    
    def test_read_header_integration(self):
        """Test read_header with actual NIfTI file."""
        if not NIBABEL_AVAILABLE:
            pytest.skip("nibabel not available")
            
        # This test requires nibabel and would normally use a test NIfTI file
        # For now, we'll skip if nibabel can't create a test file
        try:
            import nibabel as nib
            
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Create test NIfTI
                data = np.random.randn(10, 10, 10).astype(np.float32)
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, tmp_path)
                
                # Read header
                header_info = read_header(tmp_path)
                
                assert header_info['dim'] == (10, 10, 10)
                assert len(header_info['spacing']) >= 3
                assert 'affine' in header_info
                assert 'datatype' in header_info
                
            finally:
                os.unlink(tmp_path)
                
        except ImportError:
            pytest.skip("nibabel not available")
    
    def test_read_meta_info_integration(self):
        """Test read_meta_info with NIfTI file."""
        if not NIBABEL_AVAILABLE:
            pytest.skip("nibabel not available")
            
        try:
            import nibabel as nib
            
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Create test NIfTI with specific properties
                data = np.ones((20, 30, 15), dtype=np.float32)
                affine = np.diag([2.0, 3.0, 4.0, 1.0])
                affine[:3, 3] = [10.0, 20.0, 30.0]  # Set origin
                
                img = nib.Nifti1Image(data, affine)
                img.header['descrip'] = b'Test scan'
                img.header['scl_slope'] = 2.0
                img.header['scl_inter'] = 10.0
                nib.save(img, tmp_path)
                
                # Read metadata
                meta = read_meta_info(tmp_path)
                
                assert isinstance(meta, NIFTIMetaInfo)
                assert meta.dims == (20, 30, 15)
                assert meta.spacing == (2.0, 3.0, 4.0)
                assert np.allclose(meta.origin, [10.0, 20.0, 30.0])
                assert meta.slope == 2.0
                assert meta.intercept == 10.0
                
            finally:
                os.unlink(tmp_path)
                
        except ImportError:
            pytest.skip("nibabel not available")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
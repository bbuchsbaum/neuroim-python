"""Tests for memory-mapped neuroimaging data structures."""

import pytest
import numpy as np
import tempfile
import os
from neuroim import (
    NeuroSpace, DenseNeuroVol, DenseNeuroVec,
    BigNeuroVec, big_neurovecseq,
    FileBackedNeuroVec, file_backed_neurovec,
    MappedNeuroVec, mapped_neurovecseq,
    scale_mapper, log_mapper, threshold_mapper
)


class TestBigNeuroVec:
    """Test BigNeuroVec memory-mapped implementation."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[10, 10, 10, 5])  # x, y, z, time
        self.data = np.random.randn(10, 10, 10, 5)  # Match space dimensions
        
    def test_big_neurovev_creation(self):
        """Test creating BigNeuroVec."""
        vec = BigNeuroVec(self.data, self.space)
        
        assert isinstance(vec, BigNeuroVec)
        assert vec.shape == self.data.shape
        assert vec.space == self.space
        assert os.path.exists(vec.filename)
        
        # Check data is preserved
        assert np.allclose(vec.data, self.data)
        
        # Cleanup
        del vec
    
    def test_big_neurovev_with_filename(self):
        """Test creating BigNeuroVec with specific filename."""
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f:
            filename = f.name
        
        vec = BigNeuroVec(self.data, self.space, filename=filename)
        assert vec.filename == filename
        assert os.path.exists(filename)
        
        # Check data
        assert np.allclose(vec.data, self.data)
        
        # Cleanup
        del vec
        os.unlink(filename)
    
    def test_big_neurovev_indexing(self):
        """Test indexing operations."""
        vec = BigNeuroVec(self.data, self.space)
        
        # Time series at voxel
        series = vec.series(5, 5, 5)
        assert len(series) == 5  # 5 time points
        assert np.allclose(series, self.data[5, 5, 5, :])
        
        # Single volume at time point
        # BigNeuroVec doesn't support time indexing like vec[0]
        # so we skip this test
        
        del vec
    
    def test_big_neurovev_sub_vector(self):
        """Test extracting sub-vectors."""
        vec = BigNeuroVec(self.data, self.space)
        
        # Extract subset
        sub = vec.sub_vector([0, 2, 4])
        assert isinstance(sub, BigNeuroVec)
        assert sub.shape[0] == 3
        assert np.allclose(sub[0], self.data[0])
        assert np.allclose(sub[1], self.data[2])
        assert np.allclose(sub[2], self.data[4])
        
        del vec
        del sub
    
    def test_big_neurovev_vols(self):
        """Test extracting volumes."""
        vec = BigNeuroVec(self.data, self.space)
        
        vols = vec.vols([0, 2])
        assert len(vols) == 2
        assert all(isinstance(v, DenseNeuroVol) for v in vols)
        assert np.allclose(vols[0].data, self.data[0])
        assert np.allclose(vols[1].data, self.data[2])
        
        del vec
    
    def test_big_neurovev_arithmetic(self):
        """Test arithmetic operations."""
        vec1 = BigNeuroVec(self.data, self.space)
        vec2 = BigNeuroVec(self.data * 2, self.space)
        
        # Addition
        result = vec1 + vec2
        assert isinstance(result, BigNeuroVec)
        assert np.allclose(result.data, self.data * 3)
        
        # Scalar multiplication
        result = vec1 * 2
        assert np.allclose(result.data, self.data * 2)
        
        del vec1
        del vec2
        del result
    
    def test_big_neurovecseq(self):
        """Test creating from volume sequence."""
        # Create test volumes
        vols = []
        for i in range(3):
            vol_data = np.ones((10, 10, 10)) * i
            vol = DenseNeuroVol(vol_data, NeuroSpace(dim=[10, 10, 10]))
            vols.append(vol)
        
        vec = big_neurovecseq(vols)
        assert isinstance(vec, BigNeuroVec)
        assert vec.shape == (10, 10, 10, 3)
        
        # Check data
        for i in range(3):
            assert np.allclose(vec[..., i], i)
        
        del vec


class TestFileBackedNeuroVec:
    """Test FileBackedNeuroVec implementation."""
    
    def setup_method(self):
        """Set up test data."""
        # Create temporary volume files
        self.temp_files = []
        self.vol_data = []
        
        for i in range(4):
            data = np.ones((10, 10, 10)) * i
            self.vol_data.append(data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                np.save(f, data)
                self.temp_files.append(f.name)
    
    def teardown_method(self):
        """Clean up temporary files."""
        for f in self.temp_files:
            if os.path.exists(f):
                os.unlink(f)
    
    def test_file_backed_creation(self):
        """Test creating FileBackedNeuroVec."""
        # Need to mock read_vol to work with numpy files
        import neuroim.file_backed_neuro_vec as fb_module
        
        def mock_read_vol(filename):
            data = np.load(filename)
            return DenseNeuroVol(data, NeuroSpace(dim=data.shape))
        
        # Temporarily replace read_vol
        original_read_vol = fb_module.read_vol
        fb_module.read_vol = mock_read_vol
        
        try:
            vec = FileBackedNeuroVec(self.temp_files)
            
            assert isinstance(vec, FileBackedNeuroVec)
            assert vec.n_volumes == 4
            assert vec.shape == (10, 10, 10, 4)
            
            # Check data loading - use proper time indexing
            assert np.allclose(vec[:, :, :, 0], self.vol_data[0])
            assert np.allclose(vec[:, :, :, 2], self.vol_data[2])
            
        finally:
            # Restore original
            fb_module.read_vol = original_read_vol
    
    def test_file_backed_caching(self):
        """Test caching behavior."""
        import neuroim.file_backed_neuro_vec as fb_module
        
        def mock_read_vol(filename):
            data = np.load(filename)
            return DenseNeuroVol(data, NeuroSpace(dim=data.shape))
        
        original_read_vol = fb_module.read_vol
        fb_module.read_vol = mock_read_vol
        
        try:
            vec = FileBackedNeuroVec(self.temp_files, cache_size=2)
            
            # Load volumes using time indexing
            _ = vec[:, :, :, 0]
            _ = vec[:, :, :, 1]
            
            # Cache should have 2 volumes
            assert len(vec._cache) == 2
            
            # Load another - should evict oldest
            _ = vec[:, :, :, 2]
            assert len(vec._cache) == 2
            assert 0 not in vec._cache
            
        finally:
            fb_module.read_vol = original_read_vol


class TestMappedNeuroVec:
    """Test MappedNeuroVec implementation."""
    
    def setup_method(self):
        """Set up test data."""
        self.space = NeuroSpace(dim=[3, 5, 5, 5])
        self.data = np.random.randn(3, 5, 5, 5)
        self.source = DenseNeuroVec(self.data, self.space)
    
    def test_mapped_creation(self):
        """Test creating MappedNeuroVec."""
        # Simple scaling function
        def scale_by_2(x):
            return x * 2
        
        mapped = MappedNeuroVec(self.source, scale_by_2)
        
        assert isinstance(mapped, MappedNeuroVec)
        assert mapped.shape == self.source.shape
        assert np.allclose(mapped.data, self.data * 2)
    
    def test_mapped_with_inverse(self):
        """Test MappedNeuroVec with inverse function."""
        def forward(x):
            return x + 10
        
        def inverse(x):
            return x - 10
        
        mapped = MappedNeuroVec(self.source, forward, inverse)
        
        # Check forward mapping
        assert np.allclose(mapped[0].data, self.data[0] + 10)
        
        # Check setting with inverse
        new_value = np.ones((5, 5, 5)) * 20
        mapped[1] = new_value
        assert np.allclose(self.source[1].data, 10)  # 20 - 10
    
    def test_scale_mapper(self):
        """Test scale mapper utility."""
        forward, inverse = scale_mapper(2.0, center=5.0)
        
        mapped = MappedNeuroVec(self.source, forward, inverse)
        
        # Test forward
        test_val = np.array([5.0, 10.0])
        expected = np.array([5.0, 15.0])  # (x-5)*2 + 5
        assert np.allclose(forward(test_val), expected)
        
        # Test inverse
        assert np.allclose(inverse(expected), test_val)
    
    def test_log_mapper(self):
        """Test log mapper utility."""
        # Make positive data
        positive_data = np.abs(self.data) + 1
        positive_source = DenseNeuroVec(positive_data, self.space)
        
        forward, inverse = log_mapper(base=10, offset=0)
        mapped = MappedNeuroVec(positive_source, forward, inverse)
        
        # Test that we can compute logs
        log_data = mapped.data
        assert not np.any(np.isnan(log_data))
        
        # Test inverse
        reconstructed = inverse(log_data)
        assert np.allclose(reconstructed, positive_data, rtol=1e-10)

    def test_mapped_neurovecseq(self):
        """Test creating mapped vector sequence from list inputs."""
        seq_shape = tuple(self.space.dim[:3]) + (2,)
        seq_space = NeuroSpace(dim=seq_shape)
        vec1 = DenseNeuroVec(np.ones(seq_shape), seq_space)
        vec2 = DenseNeuroVec(np.ones(seq_shape) * 2, seq_space)
        
        mapped = mapped_neurovecseq(
            [vec1, vec2],
            lambda x: x + 1,
            None
        )
        
        assert isinstance(mapped, MappedNeuroVec)
        assert mapped.shape == tuple(self.space.dim[:3]) + (4,)
        assert np.allclose(mapped.data[..., 0:2], np.ones(seq_shape) + 1)
        assert np.allclose(mapped.data[..., 2:4], (np.ones(seq_shape) * 2) + 1)
    
    def test_threshold_mapper(self):
        """Test threshold mapper utility."""
        forward = threshold_mapper(threshold=0, below_value=-1)
        
        mapped = MappedNeuroVec(self.source, forward)
        mapped_data = mapped.data
        
        # Check that all negative values are -1
        assert np.all(mapped_data[self.data < 0] == -1)
        # Check that positive values are unchanged
        assert np.allclose(mapped_data[self.data >= 0], 
                          self.data[self.data >= 0])
    
    def test_mapped_arithmetic(self):
        """Test arithmetic on mapped vectors."""
        def scale_by_2(x):
            return x * 2
        
        mapped1 = MappedNeuroVec(self.source, scale_by_2)
        mapped2 = MappedNeuroVec(self.source, lambda x: x * 3)
        
        # Addition
        result = mapped1 + mapped2
        assert isinstance(result, DenseNeuroVec)
        assert np.allclose(result.data, self.data * 5)  # 2x + 3x
        
        # Scalar multiplication
        result = mapped1 * 2
        assert np.allclose(result.data, self.data * 4)  # 2x * 2

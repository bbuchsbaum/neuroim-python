"""
Test suite for neurovecseq functionality.

This module tests the neurovecseq function and related sequence operations,
corresponding to the R neuroim2 test-vecseq.R tests.
"""

import pytest
import numpy as np
from neuroim import (
    NeuroSpace, DenseNeuroVol, DenseNeuroVec, SparseNeuroVec,
    LogicalNeuroVol, MappedNeuroVec, neurovec, neurovecseq, read_vec, read_vol
)
from neuroim.io import read_vol as io_read_vol, read_vec as io_read_vec
import tempfile
import nibabel as nib


class TestNeuroVecSeq:
    """Test cases for neurovecseq functionality."""
    
    @pytest.fixture
    def create_test_vols(self):
        """Create test volumes for neurovecseq testing."""
        space = NeuroSpace((10, 10, 10), spacing=(1, 1, 1))
        
        # Create 3 test volumes with different patterns
        vol1 = DenseNeuroVol(np.ones((10, 10, 10)), space)
        vol2 = DenseNeuroVol(np.ones((10, 10, 10)) * 2, space)
        vol3 = DenseNeuroVol(np.ones((10, 10, 10)) * 3, space)
        
        return vol1, vol2, vol3, space
    
    @pytest.fixture
    def create_test_vecs(self):
        """Create test vectors for neurovecseq testing."""
        space_4d = NeuroSpace((10, 10, 10, 4), spacing=(1, 1, 1, 1))
        
        # Create 3 test vectors
        vec1 = DenseNeuroVec(np.random.rand(10, 10, 10, 4), space_4d)
        vec2 = DenseNeuroVec(np.random.rand(10, 10, 10, 4) * 2, space_4d)
        vec3 = DenseNeuroVec(np.random.rand(10, 10, 10, 4) * 3, space_4d)
        
        return vec1, vec2, vec3, space_4d
    
    def test_neurovecseq_from_volumes(self, create_test_vols):
        """Test creating neurovecseq from list of volumes."""
        vol1, vol2, vol3, space = create_test_vols
        
        # Create neurovecseq
        vec_seq = neurovecseq([vol1, vol2, vol3])
        
        # Check properties
        assert isinstance(vec_seq, DenseNeuroVec)
        assert vec_seq.shape == (10, 10, 10, 3)
        assert vec_seq.ndim == 4
        
        # Verify data - each volume should be a separate timepoint
        assert np.allclose(vec_seq.data[:, :, :, 0], 1)
        assert np.allclose(vec_seq.data[:, :, :, 1], 2)
        assert np.allclose(vec_seq.data[:, :, :, 2], 3)
    
    def test_neurovecseq_from_vectors(self, create_test_vecs):
        """Test creating neurovecseq from list of vectors (concatenation)."""
        vec1, vec2, vec3, space_4d = create_test_vecs
        
        # Create neurovecseq - should concatenate
        vec_seq = neurovecseq([vec1, vec2, vec3])
        
        # Check properties
        assert isinstance(vec_seq, DenseNeuroVec)
        assert vec_seq.shape == (10, 10, 10, 12)  # 4 + 4 + 4
        
        # Verify concatenation
        assert np.allclose(vec_seq.data[:, :, :, :4], vec1.data)
        assert np.allclose(vec_seq.data[:, :, :, 4:8], vec2.data)
        assert np.allclose(vec_seq.data[:, :, :, 8:12], vec3.data)
    
    def test_neurovecseq_linear_indexing(self, create_test_vols):
        """Test linear indexing of neurovecseq result."""
        vol1, vol2, vol3, space = create_test_vols
        
        vec_seq = neurovecseq([vol1, vol2, vol3])
        cvec = vol1.concat(vol2, vol3)  # Direct concatenation for comparison
        
        # Test flat data access via .data.ravel()
        vec_flat = vec_seq.data.ravel(order='F')
        cvec_flat = cvec.data.ravel(order='F')

        # Test multiple random indices (flat indexing)
        np.random.seed(42)
        for _ in range(20):
            idx = np.random.choice(vec_flat.size, 50)
            np.testing.assert_array_equal(vec_flat[idx], cvec_flat[idx])
    
    def test_neurovecseq_array_indexing(self, create_test_vols):
        """Test array indexing of neurovecseq result."""
        vol1, vol2, vol3, space = create_test_vols
        
        vec_seq = neurovecseq([vol1, vol2, vol3])
        cvec = vol1.concat(vol2, vol3)
        
        # Test various array indexing patterns
        np.testing.assert_array_equal(vec_seq[0, :, :, :], cvec[0, :, :, :])
        np.testing.assert_array_equal(vec_seq[0, 1, :, :], cvec[0, 1, :, :])
        np.testing.assert_array_equal(vec_seq[0, 1, 2, :], cvec[0, 1, 2, :])
        np.testing.assert_array_equal(vec_seq[0, 1, 2, 0], cvec[0, 1, 2, 0])
        np.testing.assert_array_equal(vec_seq[0:2, 1, 2, 0], cvec[0:2, 1, 2, 0])
        np.testing.assert_array_equal(vec_seq[0:2, 1, 2, 0:2], cvec[0:2, 1, 2, 0:2])
        np.testing.assert_array_equal(vec_seq[0:2, 1, :, 0:2], cvec[0:2, 1, :, 0:2])
        np.testing.assert_array_equal(vec_seq[0:2, :, :, 0:2], cvec[0:2, :, :, 0:2])
        np.testing.assert_array_equal(vec_seq[:, :, :, 0:2], cvec[:, :, :, 0:2])
    
    def test_extract_vectors(self, create_test_vols):
        """Test extracting individual vectors from neurovecseq result."""
        vol1, vol2, vol3, space = create_test_vols
        
        vec_seq = neurovecseq([vol1, vol2, vol3])
        cvec = vol1.concat(vol2, vol3)
        
        # Extract vectors (voxel time series)
        v1 = vec_seq.vectors()
        v2 = cvec.vectors()
        
        # Check random vectors match
        np.random.seed(42)
        check_ind = np.random.choice(len(v1), 100)
        for i in check_ind:
            np.testing.assert_array_equal(v1[i], v2[i])
    
    def test_extract_volumes(self, create_test_vols):
        """Test extracting volumes from neurovecseq result."""
        vol1, vol2, vol3, space = create_test_vols
        
        vec_seq = neurovecseq([vol1, vol2, vol3])
        cvec = vol1.concat(vol2, vol3)
        
        # Extract volumes
        vols1 = vec_seq.vols()
        vols2 = cvec.vols()
        
        # Check all volumes match
        for i in range(len(vols1)):
            np.testing.assert_array_equal(vols1[i].data, vols2[i].data)
    
    def test_map_over_volumes(self, create_test_vols):
        """Test mapping operations over volumes from neurovecseq."""
        vol1, vol2, vol3, space = create_test_vols
        
        vec_seq = neurovecseq([vol1, vol2, vol3])
        cvec = vol1.concat(vol2, vol3)
        
        # Map identity function over volumes and concatenate
        vols = vec_seq.vols()
        out = neurovecseq([vol for vol in vols])
        
        # Should reconstruct the same data
        np.testing.assert_array_equal(vec_seq.data, out.data)
    
    def test_subset_neurovecseq(self, create_test_vecs):
        """Test subsetting operations on neurovecseq result."""
        vec1, vec2, vec3, space_4d = create_test_vecs
        
        # Create a larger vector by concatenating
        vec_full = vec1.concat(vec2, vec3)
        
        # Extract subset (first 4 volumes)
        vec_subset = vec_full.sub_vector(range(4))
        
        # Should match the first vector
        np.testing.assert_array_equal(vec_subset.data, vec1.data)
    
    def test_sparse_neurovecseq(self):
        """Test neurovecseq with sparse vectors."""
        # Create test data with mask
        space = NeuroSpace((10, 10, 10), spacing=(1, 1, 1))
        
        # Create mask - only center voxels active
        mask_data = np.zeros((10, 10, 10), dtype=bool)
        mask_data[3:7, 3:7, 3:7] = True
        mask = LogicalNeuroVol(mask_data, space)
        
        # Create sparse vectors
        n_active = np.sum(mask_data)
        sparse_data1 = np.random.rand(n_active, 4)
        sparse_data2 = np.random.rand(n_active, 4) * 2
        sparse_data3 = np.random.rand(n_active, 4) * 3
        
        space_4d = space.add_dim(1, 4)
        indices = np.where(mask_data.flatten())[0]
        
        vec1 = SparseNeuroVec(sparse_data1, space_4d, indices)
        vec2 = SparseNeuroVec(sparse_data2, space_4d, indices)
        vec3 = SparseNeuroVec(sparse_data3, space_4d, indices)
        
        # Concatenate sparse vectors
        concat_sparse = vec1.concat(vec2, vec3)
        
        # Check dimensions
        assert concat_sparse.shape == (10, 10, 10, 12)
        assert concat_sparse.data.shape[1] == n_active
    
    def test_mapped_neurovecseq(self):
        """Test using DenseNeuroVec loaded from file in neurovecseq-like operations."""
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            # Create test data
            data = np.random.rand(10, 10, 10, 3).astype(np.float32)

            # Save as NIfTI
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, tmp.name)

            # Load via read_vec
            dvec = io_read_vec(tmp.name)

            # Test basic operations
            assert dvec.shape == (10, 10, 10, 3)

            # Extract volumes
            vols = dvec.vols()
            assert len(vols) == 3

            # Extract vectors
            vecs = dvec.vectors()
            assert len(vecs) == 1000  # 10*10*10
    
    def test_series_with_sparse_neurovecseq(self):
        """Test series extraction from sparse neurovecseq result."""
        # Create sparse test data
        space = NeuroSpace((8, 8, 8), spacing=(1, 1, 1))
        
        # Create mask
        mask_data = np.zeros((8, 8, 8), dtype=bool)
        mask_data[2:6, 2:6, 2:6] = True
        mask = LogicalNeuroVol(mask_data, space)
        
        # Create sparse vectors
        n_active = np.sum(mask_data)
        sparse_data1 = np.random.rand(n_active, 4)
        sparse_data2 = np.random.rand(n_active, 4)
        sparse_data3 = np.random.rand(n_active, 4)
        
        space_4d = space.add_dim(1, 4)
        indices = np.where(mask_data.flatten())[0]
        
        s1 = SparseNeuroVec(sparse_data1, space_4d, indices)
        s2 = SparseNeuroVec(sparse_data2, space_4d, indices) 
        s3 = SparseNeuroVec(sparse_data3, space_4d, indices)
        
        # Concatenate to form sequence
        seq_sparse = s1.concat(s2, s3)
        
        # Test series extraction with random indices
        np.random.seed(123)
        idx = np.random.choice(8**3, 25)
        
        # Extract series - returns (time x voxels) like R
        mat_seq = seq_sparse.series(idx)

        # Check dimensions: (time, n_voxels)
        assert mat_seq.shape == (12, 25)  # 12 timepoints, 25 voxels

        # Test single voxel extraction
        vox = int(idx[0])
        vec_single = seq_sparse.series(vox)

        assert isinstance(vec_single, np.ndarray)
        assert vec_single.shape == (12,)
    
    def test_empty_list_error(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="Empty vector list"):
            neurovecseq([])
    
    def test_mismatched_dimensions_error(self, create_test_vols):
        """Test that mismatched dimensions raise error."""
        vol1, vol2, vol3, space = create_test_vols
        
        # Create volume with different dimensions
        space_diff = NeuroSpace((8, 8, 8), spacing=(1, 1, 1))
        vol_diff = DenseNeuroVol(np.ones((8, 8, 8)), space_diff)
        
        with pytest.raises(ValueError, match="All volumes must have same dimensions"):
            neurovecseq([vol1, vol_diff])
    
    def test_mismatched_spacing_error(self, create_test_vols):
        """Test that mismatched spacing raises error."""
        vol1, vol2, vol3, space = create_test_vols
        
        # Create volume with different spacing
        space_diff = NeuroSpace((10, 10, 10), spacing=(2, 2, 2))
        vol_diff = DenseNeuroVol(np.ones((10, 10, 10)), space_diff)
        
        with pytest.raises(ValueError, match="All volumes must have same spacing"):
            neurovecseq([vol1, vol_diff])
    
    def test_with_label(self, create_test_vols):
        """Test neurovecseq with label parameter."""
        vol1, vol2, vol3, space = create_test_vols
        
        vec_seq = neurovecseq([vol1, vol2, vol3], label="test_sequence")
        
        assert vec_seq.label == "test_sequence"

    def test_with_label_for_vector_lists(self, create_test_vecs):
        """Test neurovecseq and neurovec propagate labels for vector inputs."""
        vec1, vec2, vec3, _ = create_test_vecs

        seq = neurovecseq([vec1, vec2, vec3], label="vec_sequence")
        assert seq.label == "vec_sequence"

        combo = [vec1, vec2]
        stacked = neurovec(combo, label="neuro_vec")
        assert stacked.label == "neuro_vec"

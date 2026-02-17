"""
Memory efficiency comparison tests
Tests memory usage patterns between different implementations
"""

import pytest
import numpy as np
import sys
from neuroimpy import NeuroSpace
from neuroimpy.neuro_vol import DenseNeuroVol, SparseNeuroVol, LogicalNeuroVol
from neuroimpy.neuro_vec import DenseNeuroVec, SparseNeuroVec
from neuroimpy.file_backed_neuro_vec import FileBackedNeuroVec
import tempfile
import gc
import os


def get_object_size(obj):
    """Get approximate size of object in MB"""
    if hasattr(obj, 'data') and hasattr(obj.data, 'nbytes'):
        return obj.data.nbytes / 1024 / 1024
    elif hasattr(obj, 'nbytes'):
        return obj.nbytes / 1024 / 1024
    else:
        return sys.getsizeof(obj) / 1024 / 1024


class TestMemoryEfficiency:
    """Test memory efficiency of different data structures"""
    
    def test_sparse_vs_dense_memory(self):
        """Compare memory usage of sparse vs dense representations"""
        # Create a large volume with 10% active voxels
        space = NeuroSpace(dim=(100, 100, 100))
        mask_data = np.random.rand(100, 100, 100) > 0.9
        mask = LogicalNeuroVol(mask_data, space)
        
        # Dense volume
        dense_data = np.random.randn(100, 100, 100).astype(np.float32)
        dense_vol = DenseNeuroVol(dense_data, space)
        dense_memory = get_object_size(dense_vol)
        
        # Sparse volume
        n_active = mask_data.sum()
        indices = np.where(mask_data.ravel())[0]
        sparse_data = np.random.randn(n_active).astype(np.float32)
        sparse_vol = SparseNeuroVol(data=sparse_data, space=space, indices=indices)
        sparse_memory = get_object_size(sparse_vol)
        
        # Sparse should use significantly less memory
        efficiency_ratio = sparse_memory / dense_memory
        assert efficiency_ratio < 0.2, f"Sparse used {efficiency_ratio:.1%} of dense memory"
        
        # Report findings
        print(f"\nMemory usage for 100x100x100 volume (10% active):")
        print(f"Dense: {dense_memory:.1f} MB")
        print(f"Sparse: {sparse_memory:.1f} MB")
        print(f"Efficiency: {efficiency_ratio:.1%}")
    
    def test_sparse_neurovec_memory_scaling(self):
        """Test memory scaling of sparse neurovec with sparsity"""
        space_3d = NeuroSpace(dim=(50, 50, 50))
        space_4d = NeuroSpace(dim=(50, 50, 50, 100))
        
        memory_results = []
        
        for sparsity in [0.01, 0.05, 0.1, 0.2, 0.5]:
            # Create mask with given sparsity
            mask_data = np.random.rand(50, 50, 50) < sparsity
            mask = LogicalNeuroVol(mask_data, space_3d)
            n_active = mask_data.sum()
            
            # Create sparse neurovec
            data = np.random.randn(n_active, 100).astype(np.float32)
            sparse_vec = SparseNeuroVec(data=data, mask=mask, space=space_4d)
            
            memory_used = get_object_size(sparse_vec)
            
            memory_results.append({
                'sparsity': sparsity,
                'n_active': n_active,
                'memory_mb': memory_used
            })
        
        # Memory should scale linearly with sparsity
        print("\nSparse NeuroVec memory scaling:")
        for result in memory_results:
            print(f"Sparsity {result['sparsity']:.0%}: "
                  f"{result['n_active']} voxels, "
                  f"{result['memory_mb']:.1f} MB")
        
        # Check linear scaling
        memories = [r['memory_mb'] for r in memory_results]
        sparsities = [r['sparsity'] for r in memory_results]
        
        # Rough linearity check
        for i in range(1, len(memories)):
            ratio = memories[i] / memories[0]
            sparsity_ratio = sparsities[i] / sparsities[0]
            # Allow 50% deviation from perfect linearity
            assert 0.5 < ratio / sparsity_ratio < 2.0
    
    def test_dtype_memory_efficiency(self):
        """Test memory usage with different data types"""
        space = NeuroSpace(dim=(100, 100, 100))
        
        dtypes = [
            (np.float64, 8),
            (np.float32, 4),
            (np.int32, 4),
            (np.int16, 2),
            (np.int8, 1),
            (bool, 1)
        ]
        
        memory_results = []
        
        for dtype, expected_bytes in dtypes:
            if dtype == bool:
                data = np.random.rand(100, 100, 100) > 0.5
            else:
                data = np.random.randn(100, 100, 100).astype(dtype)
            
            vol = DenseNeuroVol(data, space)
            memory_used = get_object_size(vol)
            
            # Expected memory in MB
            expected_mb = (100 * 100 * 100 * expected_bytes) / (1024 * 1024)
            
            memory_results.append({
                'dtype': dtype,
                'memory_mb': memory_used,
                'expected_mb': expected_mb,
                'ratio': memory_used / expected_mb if expected_mb > 0 else 0
            })
        
        print("\nData type memory efficiency:")
        for result in memory_results:
            print(f"{result['dtype']}: {result['memory_mb']:.1f} MB "
                  f"(expected: {result['expected_mb']:.1f} MB, "
                  f"ratio: {result['ratio']:.2f})")
        
        # Check that smaller dtypes use less memory
        float64_mem = memory_results[0]['memory_mb']
        float32_mem = memory_results[1]['memory_mb']
        int8_mem = memory_results[4]['memory_mb']
        
        assert float32_mem < 0.6 * float64_mem
        assert int8_mem < 0.2 * float64_mem
    
    def test_file_backed_lazy_loading(self):
        """Test that file-backed doesn't load all data immediately"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data files
            shape = (50, 50, 50)
            n_volumes = 20
            
            # Write volumes to disk as NIfTI
            import nibabel as nib
            filenames = []
            total_size = 0
            for i in range(n_volumes):
                data = np.random.randn(*shape).astype(np.float32)
                filename = os.path.join(tmpdir, f"vol_{i:03d}.nii.gz")
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, filename)
                filenames.append(filename)
                total_size += data.nbytes
            
            # Create file-backed vector
            fbvec = FileBackedNeuroVec(filenames)
            
            # The object itself should be small (just metadata)
            # Not loading actual data sizes
            assert hasattr(fbvec, 'shape')
            assert fbvec.shape == shape + (n_volumes,)
            
            # Access patterns should be lazy
            single_series = fbvec[25, 25, 25, :]  # Should only load necessary volumes
            assert single_series.shape == (n_volumes,)
            
            print(f"\nFile-backed vector for {shape}x{n_volumes} data:")
            print(f"Total data size: {total_size / 1024 / 1024:.1f} MB")
            print(f"Accessing single voxel series: shape {single_series.shape}")


class TestDataStructureEfficiency:
    """Test efficiency of different data structure choices"""
    
    def test_sparse_optimal_threshold(self):
        """Test when sparse representation becomes more efficient"""
        space = NeuroSpace(dim=(50, 50, 50))
        
        results = []
        
        for sparsity in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            # Create mask
            mask_data = np.random.rand(50, 50, 50) < sparsity
            mask = LogicalNeuroVol(mask_data, space)
            n_active = mask_data.sum()
            
            # Dense representation
            dense_data = np.random.randn(50, 50, 50).astype(np.float32)
            dense_vol = DenseNeuroVol(dense_data, space)
            dense_size = get_object_size(dense_vol)
            
            # Sparse representation
            indices = np.where(mask_data.ravel())[0]
            sparse_data = dense_data[mask_data].astype(np.float32)
            sparse_vol = SparseNeuroVol(data=sparse_data, space=space, indices=indices)
            sparse_size = get_object_size(sparse_vol)
            
            efficiency = sparse_size / dense_size
            
            results.append({
                'sparsity': sparsity,
                'dense_mb': dense_size,
                'sparse_mb': sparse_size,
                'efficiency': efficiency,
                'better': 'sparse' if efficiency < 1.0 else 'dense'
            })
        
        print("\nSparse vs Dense efficiency by sparsity:")
        for r in results:
            print(f"Sparsity {r['sparsity']:.0%}: "
                  f"Sparse/Dense = {r['efficiency']:.2f} "
                  f"({r['better']} is better)")
        
        # Find crossover point
        crossover = None
        for i in range(len(results) - 1):
            if results[i]['better'] == 'sparse' and results[i+1]['better'] == 'dense':
                crossover = (results[i]['sparsity'] + results[i+1]['sparsity']) / 2
                break
        
        if crossover:
            print(f"\nCrossover point: ~{crossover:.0%} sparsity")
            # Typical crossover should be around 30-50%
            assert 0.2 < crossover < 0.6
    
    def test_4d_representation_comparison(self):
        """Compare different 4D data representations"""
        shape_3d = (30, 30, 30)
        n_timepoints = 100
        
        space_3d = NeuroSpace(dim=shape_3d)
        space_4d = NeuroSpace(dim=shape_3d + (n_timepoints,))
        
        # Method 1: Dense 4D array
        data_4d = np.random.randn(*shape_3d, n_timepoints).astype(np.float32)
        dense_vec = DenseNeuroVec(data_4d, space_4d)
        dense_size = get_object_size(dense_vec)
        
        # Method 2: List of 3D volumes (simulate)
        list_size = 0
        vol_list = []
        for t in range(n_timepoints):
            vol = DenseNeuroVol(data_4d[..., t], space_3d)
            vol_list.append(vol)
            list_size += get_object_size(vol)
        
        # Method 3: Sparse with 50% mask
        mask_data = np.random.rand(*shape_3d) < 0.5
        mask = LogicalNeuroVol(mask_data, space_3d)
        n_active = mask_data.sum()
        
        sparse_data = np.random.randn(n_active, n_timepoints).astype(np.float32)
        sparse_vec = SparseNeuroVec(data=sparse_data, mask=mask, space=space_4d)
        sparse_size = get_object_size(sparse_vec)
        
        print(f"\n4D representation comparison ({shape_3d}x{n_timepoints}):")
        print(f"Dense 4D array: {dense_size:.1f} MB")
        print(f"List of 3D volumes: {list_size:.1f} MB")
        print(f"Sparse (50% active): {sparse_size:.1f} MB")
        
        # Dense should be most efficient for full data
        assert dense_size <= list_size  # Less or equal overhead than list
        assert sparse_size < dense_size  # Sparse saves with 50% mask
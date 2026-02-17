#!/usr/bin/env python
"""Simple test to verify notebook code works."""

import sys
import traceback

# This module is a CLI-style utility script and should not be collected by pytest.
__test__ = False

def test_image_volumes():
    """Test key code from image_volumes notebook."""
    print("\nTesting image_volumes notebook code...")
    try:
        import neuroimpy as pn
        import numpy as np
        
        # Create example volume
        space_3d = pn.NeuroSpace(dim=(64, 64, 25), spacing=(3.5, 3.5, 5.0), 
                                 origin=(-110.5, -88.9342, -42.75))
        example_data = np.random.randn(64, 64, 25)
        example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())
        
        # Create volume
        vol = pn.DenseNeuroVol(example_data, space_3d)
        print(f"✓ Created volume: {vol.shape}")
        
        # Test arithmetic
        vol2 = vol + vol
        assert np.sum(vol2.data) == 2 * np.sum(vol.data)
        print("✓ Arithmetic operations work")
        
        # Test logical conversion
        vol_binary = vol.as_logical()
        print(f"✓ Logical conversion works: {type(vol_binary)}")
        
        # Test sparse conversion
        sparse_vol = vol.as_sparse()
        print(f"✓ Sparse conversion works: {type(sparse_vol)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

def test_neuro_vectors():
    """Test key code from neuro_vectors notebook."""
    print("\nTesting neuro_vectors notebook code...")
    try:
        import neuroimpy as pn
        import numpy as np
        
        # Create 4D space
        space_4d = pn.NeuroSpace(dim=(64, 64, 25, 100), 
                                 spacing=(3.5, 3.5, 5.0, 2.0),
                                 origin=(-110.5, -88.9342, -42.75, 0.0))
        
        # Create 4D data
        np.random.seed(42)
        example_4d_data = np.random.randn(64, 64, 25, 100)
        
        # Create NeuroVec
        vec = pn.DenseNeuroVec(example_4d_data, space_4d)
        print(f"✓ Created 4D vector: {vec.shape}")
        
        # Test subsetting
        vec_subset = vec.sub_vector(slice(0, 6))
        print(f"✓ Subset extraction works: {vec_subset.shape}")
        
        # Test time series extraction
        ts = vec.series(10, 10, 10)
        print(f"✓ Time series extraction works: {ts.shape}")
        
        # Test concatenation
        vec1 = pn.DenseNeuroVec(np.random.randn(10, 10, 10, 50), 
                                pn.NeuroSpace(dim=(10, 10, 10, 50)))
        vec2 = pn.DenseNeuroVec(np.random.randn(10, 10, 10, 30), 
                                pn.NeuroSpace(dim=(10, 10, 10, 30)))
        combined = vec1.concat(vec2)
        print(f"✓ Concatenation works: {combined.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

def test_regions_of_interest():
    """Test key code from regions_of_interest notebook."""
    print("\nTesting regions_of_interest notebook code...")
    try:
        import neuroimpy as pn
        import numpy as np
        
        # Create volume
        space_3d = pn.NeuroSpace(dim=(64, 64, 25), spacing=(3.5, 3.5, 5.0), 
                                 origin=(-110.5, -88.9342, -42.75))
        example_data = np.random.randn(64, 64, 25)
        example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())
        vol = pn.DenseNeuroVol(example_data, space_3d)
        
        # Create spherical ROI
        sphere = pn.spherical_roi(vol, [20, 20, 10], radius=5, fill=100)
        print(f"✓ Created spherical ROI with {len(sphere)} voxels")
        
        # Convert to sparse
        sparsevol = sphere.as_sparse()
        print(f"✓ ROI to sparse conversion works")
        
        # Create other shapes
        square = pn.square_roi(vol, centroid=[30, 30, 10], 
                               surround=3, fixdim=2, fill=1)
        print(f"✓ Square ROI created with {len(square)} voxels")
        
        cube = pn.cuboid_roi(vol, centroid=[30, 30, 12], 
                             surround=5, fill=1)
        print(f"✓ Cuboid ROI created with {len(cube)} voxels")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

def test_pipelines():
    """Test key code from pipelines notebook."""
    print("\nTesting pipelines notebook code...")
    try:
        import neuroimpy as pn
        import numpy as np
        
        # Create volume
        space_3d = pn.NeuroSpace(dim=(64, 64, 25), spacing=(3.5, 3.5, 5.0), 
                                 origin=(-110.5, -88.9342, -42.75))
        example_data = np.random.randn(64, 64, 25)
        example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())
        vol = pn.DenseNeuroVol(example_data, space_3d)
        
        # Create volume with high-value regions
        vol2 = vol.as_dense()
        vol2.data[20:30, 20:30, 10:15] = 0.9
        vol2.data[40:50, 40:50, 15:20] = 0.85
        
        # Test connected components
        comp = pn.conn_comp(vol2, threshold=0.8)
        print(f"✓ Found {comp.index.num_clusters} connected components")
        
        # Test split clusters
        from neuroimpy import split_clusters
        cluster_rois = split_clusters(vol2, comp.index)
        print(f"✓ Split into {len(cluster_rois)} cluster ROIs")
        
        # Test searchlight
        from neuroimpy import searchlight
        mask = vol.data > 0.2
        mask_vol = pn.LogicalNeuroVol(mask, vol.space)
        rois = list(searchlight(mask_vol, radius=5))[:10]
        print(f"✓ Created {len(rois)} searchlight ROIs")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing notebook code execution...")
    print("=" * 60)
    
    tests = [
        ("image_volumes", test_image_volumes),
        ("neuro_vectors", test_neuro_vectors),
        ("regions_of_interest", test_regions_of_interest),
        ("pipelines", test_pipelines),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

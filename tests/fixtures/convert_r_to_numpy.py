#!/usr/bin/env python
"""
Convert R fixtures to NumPy format for easier testing.

This script converts R .rds files to NumPy .npz files that can be loaded
directly in Python tests without requiring R dependencies.

Usage:
    python convert_r_to_numpy.py

Requires:
    pip install pyreadr numpy
"""

import numpy as np
from pathlib import Path
import json

try:
    import pyreadr
except ImportError:
    pyreadr = None

def convert_rds_to_numpy():
    """Convert all .rds files in r_outputs/ to .npz files in numpy_fixtures/"""
    
    r_dir = Path("r_outputs")
    np_dir = Path("numpy_fixtures")
    np_dir.mkdir(exist_ok=True)
    
    conversions = {
        # NeuroVol arithmetic results
        "vol_add_result.rds": "vol_add_result.npy",
        "vol_mult_result.rds": "vol_mult_result.npy",
        "vol_div_result.rds": "vol_div_result.npy",
        
        # Series extraction results
        "vec_series_single.rds": "vec_series_single.npy",
        "vec_series_multi.rds": "vec_series_multi.npy",
        "sparse_vec_series.rds": "sparse_vec_series.npy",
        
        # ROI coordinates
        "roi_sphere_coords.rds": "roi_sphere_coords.npy",
        "roi_cube_coords.rds": "roi_cube_coords.npy",
        
        # Connected components
        "conn_comp_mask.rds": "conn_comp_mask.npy",
        "conn_comp_clusters.rds": "conn_comp_clusters.npy",
        
        # Other results
        "searchlight_result.rds": "searchlight_result.npy",
        "partition_result.rds": "partition_result.npy",
    }
    
    metadata = {}
    
    for rds_file, npy_file in conversions.items():
        rds_path = r_dir / rds_file
        npy_path = np_dir / npy_file
        
        if rds_path.exists():
            try:
                # Load R data
                result = pyreadr.read_r(str(rds_path))
                data = result[None]  # pyreadr returns dict
                
                # Convert to numpy array and save
                np_array = np.asarray(data)
                np.save(npy_path, np_array)
                
                metadata[npy_file] = {
                    "shape": np_array.shape,
                    "dtype": str(np_array.dtype),
                    "source": rds_file
                }
                
                print(f"Converted {rds_file} -> {npy_file}")
                print(f"  Shape: {np_array.shape}, dtype: {np_array.dtype}")
                
            except Exception as e:
                print(f"Error converting {rds_file}: {e}")
        else:
            print(f"Skipping {rds_file} (not found)")
    
    # Save metadata
    with open(np_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nConversion complete. Files saved to {np_dir}/")


def create_test_fixtures_manually():
    """
    Create test fixtures manually when R is not available.
    This ensures tests can run without R dependencies.
    """
    
    np_dir = Path("numpy_fixtures")
    np_dir.mkdir(exist_ok=True)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # NeuroVol test data (matching R's 1:1000)
    test_data = np.arange(1, 1001).reshape(10, 10, 10, order='F')
    
    # Arithmetic operations
    np.save(np_dir / "vol_add_result.npy", test_data + 10)
    np.save(np_dir / "vol_mult_result.npy", test_data * 2)
    np.save(np_dir / "vol_div_result.npy", test_data / 2)
    
    # Create 4D test data
    vec_data = np.random.randn(10, 10, 10, 20)
    
    # Single voxel time series (R: c(5,5,5) -> Python [4,4,4])
    single_ts = vec_data[4, 4, 4, :]
    np.save(np_dir / "vec_series_single.npy", single_ts)
    
    # Multiple voxel time series
    coords = [(2, 2, 2), (6, 6, 6), (4, 5, 6)]  # 0-based
    multi_ts = np.array([vec_data[x, y, z, :] for x, y, z in coords])
    np.save(np_dir / "vec_series_multi.npy", multi_ts.T)  # Match R's column format
    
    # ROI coordinates (example for sphere with radius 3)
    # This is simplified - actual sphere calculation would be more complex
    sphere_coords = []
    center = np.array([4, 4, 4])  # 0-based
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if np.linalg.norm([i, j, k] - center) <= 3:
                    sphere_coords.append([i, j, k])
    np.save(np_dir / "roi_sphere_coords.npy", np.array(sphere_coords))
    
    # Cubic ROI (5x5x5 around center)
    cube_coords = []
    for i in range(2, 7):  # [2,3,4,5,6] in 0-based
        for j in range(2, 7):
            for k in range(2, 7):
                cube_coords.append([i, j, k])
    np.save(np_dir / "roi_cube_coords.npy", np.array(cube_coords))
    
    # Save metadata
    metadata = {
        "created_by": "create_test_fixtures_manually",
        "numpy_version": np.__version__,
        "seed": 42,
        "indexing": "0-based (Python)",
        "notes": "Manual fixtures for testing without R"
    }
    
    with open(np_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Manual test fixtures created successfully!")


if __name__ == "__main__":
    # Try to convert from R first
    if pyreadr is not None:
        try:
            convert_rds_to_numpy()
        except Exception as e:
            print(f"Error converting R fixtures: {e}")
            print("Creating manual test fixtures instead...")
            create_test_fixtures_manually()
    else:
        print("pyreadr not available. Creating manual test fixtures...")
        create_test_fixtures_manually()
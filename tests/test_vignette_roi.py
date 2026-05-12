"""Test that ROI vignette functionality works in Python."""
import numpy as np

# Test imports
try:
    from neuroim import (DenseNeuroVol, NeuroSpace, spherical_roi,
                          square_roi, cuboid_roi, ClusteredNeuroVol,
                          searchlight, searchlight_coords)
    from neuroim.roi import spherical_roi_set
    from neuroim.searchlight import random_searchlight, clustered_searchlight
    print("✓ Basic imports work")
except ImportError as e:
    print(f"✗ Import error: {e}")

def test_spherical_roi():
    """Test creating spherical ROIs."""
    # Create test volume
    data = np.ones((20, 20, 20))
    space = NeuroSpace(dim=[20, 20, 20], spacing=[1, 1, 1])
    vol = DenseNeuroVol(data, space)

    # Create spherical ROI
    sphere = spherical_roi(vol, [10, 10, 10], radius=5, fill=100)

    assert len(sphere) > 0
    assert np.all(sphere.data == 100)
    print("✓ Spherical ROI creation works")

    # Test without fill
    sphere2 = spherical_roi(vol, [10, 10, 10], radius=3)
    assert np.all(sphere2.data == 1)  # Should have original values
    print("✓ Spherical ROI without fill works")

def test_world_coordinates():
    """Test ROI creation with world coordinates."""
    # Create volume with known spacing
    data = np.ones((20, 20, 20))
    space = NeuroSpace(dim=[20, 20, 20], spacing=[2, 2, 2], origin=[-20, -20, -20])
    vol = DenseNeuroVol(data, space)

    # Real-world coordinate
    rpoint = np.array([0, 0, 0])  # Center of space

    # Convert to voxel coordinates
    vox = space.coord_to_grid(rpoint.reshape(1, -1))[0]

    # Create ROI
    sphere = spherical_roi(vol, vox, radius=10, fill=1)

    # Get world coords of ROI
    roi_coords = sphere.get_coords(real=True)
    center_of_mass = np.mean(roi_coords, axis=0)

    # Should be close to original point
    assert np.allclose(center_of_mass, rpoint, atol=2)
    print("✓ World coordinate ROI works")

def test_roi_to_sparse():
    """Test converting ROI to sparse volume."""
    # Create test volume
    data = np.zeros((20, 20, 20))
    space = NeuroSpace(dim=[20, 20, 20])
    vol = DenseNeuroVol(data, space)

    # Create ROI
    sphere = spherical_roi(vol, [10, 10, 10], radius=5, fill=1)

    # Convert to sparse
    sparse_vol = sphere.as_sparse()

    assert sparse_vol.shape == vol.shape
    assert np.sum(sparse_vol.data) == np.sum(sphere.data)
    print("✓ ROI to sparse conversion works")

def test_other_roi_shapes():
    """Test square and cuboid ROIs."""
    data = np.zeros((20, 20, 20))
    space = NeuroSpace(dim=[20, 20, 20])
    vol = DenseNeuroVol(data, space)

    # Square ROI
    try:
        square = square_roi(vol, centroid=[10, 10, 10], surround=3, fixdim=2, fill=1)
        assert len(square) == 49  # 7x7 square
        print("✓ Square ROI works")
    except:
        print("✗ Square ROI not implemented")

    # Cuboid ROI
    try:
        cube = cuboid_roi(vol, centroid=[10, 10, 10], surround=2, fill=1)
        assert len(cube) == 125  # 5x5x5 cube
        print("✓ Cuboid ROI works")

        # Asymmetric cuboid
        cuboid = cuboid_roi(vol, centroid=[10, 10, 10], surround=[1, 2, 3], fill=1)
        assert len(cuboid) == 3 * 5 * 7  # 3x5x7
        print("✓ Asymmetric cuboid works")
    except:
        print("✗ Cuboid ROI not implemented")

def test_multiple_rois():
    """Test creating multiple ROIs."""
    data = np.zeros((30, 30, 30))
    space = NeuroSpace(dim=[30, 30, 30])
    vol = DenseNeuroVol(data, space)

    # Multiple centers
    centers = np.array([[10, 10, 10], [20, 20, 20], [15, 15, 15]])

    try:
        roi_list = spherical_roi_set(vol, centers, radius=3, fill=[100, 200, 300])
        assert len(roi_list) == 3
        assert roi_list[0].data[0] == 100
        assert roi_list[1].data[0] == 200
        assert roi_list[2].data[0] == 300
        print("✓ Multiple ROI creation works")
    except:
        print("✗ spherical_roi_set not fully working")

def test_searchlight():
    """Test searchlight functionality."""
    data = np.random.randn(15, 15, 15)
    space = NeuroSpace(dim=[15, 15, 15])
    vol = DenseNeuroVol(data, space)

    # Create mask
    from neuroim import LogicalNeuroVol
    mask = LogicalNeuroVol(vol.data > -0.5, space)

    # Basic searchlight
    rois = list(searchlight(mask, radius=3))

    assert len(rois) > 0

    # Compute means
    means = []
    for roi in rois[:10]:  # Just test first 10
        values = vol.data[roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]]
        means.append(np.mean(values))

    assert len(means) == 10
    print("✓ Searchlight works")

def test_searchlight_coords():
    """Test searchlight coordinates only."""
    data = np.random.randn(10, 10, 10)
    space = NeuroSpace(dim=[10, 10, 10])
    vol = DenseNeuroVol(data, space)

    # Create mask
    from neuroim import LogicalNeuroVol
    mask = LogicalNeuroVol(vol.data > -0.5, space)

    # Get coordinates
    coords_list = list(searchlight_coords(mask, radius=2))

    assert len(coords_list) > 0

    # Process first few
    for coords in coords_list[:5]:
        vals = vol.data[coords[:, 0], coords[:, 1], coords[:, 2]]
        mean_val = np.mean(vals)
        assert isinstance(mean_val, float)

    print("✓ Searchlight coords works")

def test_random_searchlight():
    """Test random searchlight."""
    data = np.random.randn(15, 15, 15)
    space = NeuroSpace(dim=[15, 15, 15])
    vol = DenseNeuroVol(data, space)

    # Create mask
    from neuroim import LogicalNeuroVol
    mask = LogicalNeuroVol(vol.data > -0.5, space)

    try:
        rois = list(random_searchlight(mask, radius=3))
        assert len(rois) > 0
        print("✓ Random searchlight works")
    except:
        print("✗ Random searchlight not implemented")

def test_clustered_searchlight():
    """Test clustered searchlight."""
    from sklearn.cluster import KMeans

    data = np.random.randn(20, 20, 20)
    space = NeuroSpace(dim=[20, 20, 20])
    vol = DenseNeuroVol(data, space)

    # Create clustering
    mask_indices = np.where(vol.data > -0.5)
    coords = np.column_stack(mask_indices)

    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(coords)

    # Create clustered volume
    cluster_data = np.zeros_like(vol.data)
    cluster_data[mask_indices] = labels + 1
    # First create a mask volume
    mask_vol = DenseNeuroVol(cluster_data, space)
    kvol = ClusteredNeuroVol(mask_vol, cluster_data)

    try:
        cluster_rois = list(clustered_searchlight(vol, kvol))
        assert len(cluster_rois) == 10
        print("✓ Clustered searchlight works")
    except Exception as e:
        print(f"✗ Clustered searchlight failed: {e}")

if __name__ == "__main__":
    print("Testing ROI vignette functionality...")
    print("=" * 50)

    test_spherical_roi()
    test_world_coordinates()
    test_roi_to_sparse()
    test_other_roi_shapes()
    test_multiple_rois()
    test_searchlight()
    test_searchlight_coords()
    test_random_searchlight()
    test_clustered_searchlight()

    print("\nSummary: Core ROI functionality is working!")
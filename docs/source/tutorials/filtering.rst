Spatial Filtering
=================

Spatial filtering is used to smooth neuroimaging data, reduce noise, and enhance signal-to-noise ratio. neuroimpy provides several filtering methods with different properties.

Gaussian Blur
-------------

The most common spatial filter uses a Gaussian kernel:

.. code-block:: python

    import neuroimpy
    import numpy as np

    # Load a volume
    vol = neuroimpy.read_vol("structural.nii.gz")

    # Apply Gaussian blur with sigma=2.0mm
    smoothed = neuroimpy.gaussian_blur(vol, sigma=2.0)

    # Save the result
    neuroimpy.write_vol(smoothed, "smoothed_structural.nii.gz")

    # Use different sigma values in each dimension
    anisotropic_smoothed = neuroimpy.gaussian_blur(
        vol,
        sigma=[2.0, 2.0, 3.0]  # Less smoothing in z-direction
    )

Bilateral Filter
----------------

Bilateral filtering preserves edges while smoothing homogeneous regions:

.. code-block:: python

    # Bilateral filter with spatial and intensity parameters
    bilateral = neuroimpy.bilateral_filter(
        vol,
        spatial_sigma=2.0,  # Spatial neighborhood size
        intensity_sigma=0.1  # Intensity similarity threshold
    )

    # Good for structural images where you want to preserve boundaries
    neuroimpy.write_vol(bilateral, "bilateral_filtered.nii.gz")

The bilateral filter is particularly useful for structural MRI where you want to reduce noise within tissue types while preserving boundaries between gray matter, white matter, and CSF.

Guided Filter
-------------

Guided filtering uses a guidance image to determine filtering weights:

.. code-block:: python

    # Load functional and structural images
    functional = neuroimpy.read_vol("functional.nii.gz")
    structural = neuroimpy.read_vol("structural_aligned.nii.gz")

    # Use structural image to guide filtering of functional data
    guided = neuroimpy.guided_filter(
        input_vol=functional,
        guide_vol=structural,
        radius=5,       # Neighborhood radius
        epsilon=0.01    # Regularization parameter
    )

    neuroimpy.write_vol(guided, "guided_filtered.nii.gz")

Guided filtering is excellent when you have a high-quality structural image aligned to functional data and want to smooth the functional data while respecting anatomical boundaries.

Graph-Based Smoothing
---------------------

Constrained graph-based (CGB) smoothing respects anatomical or functional boundaries:

.. code-block:: python

    # Load data and a parcellation atlas
    vol = neuroimpy.read_vol("functional.nii.gz")
    parcellation = neuroimpy.read_vol("atlas.nii.gz")

    # Smooth within parcels, not across boundaries
    graph_smoothed = neuroimpy.cgb_smooth(
        vol,
        labels=parcellation,
        sigma=2.0,
        iterations=3
    )

    neuroimpy.write_vol(graph_smoothed, "cgb_smoothed.nii.gz")

This approach constructs a graph where edges connect voxels within the same parcel, ensuring smoothing doesn't blur across functionally or anatomically distinct regions.

Custom Kernels
--------------

Create custom convolution kernels for specialized filtering:

.. code-block:: python

    # Create a Gaussian kernel
    kernel = neuroimpy.gaussian_kernel(
        sigma=2.0,
        size=7  # Kernel will be 7x7x7
    )

    # Apply custom kernel
    result = neuroimpy.convolve(vol, kernel)

    # Spherical kernel (uniform weights within a sphere)
    sphere_kernel = neuroimpy.spherical_kernel(radius=3.0)
    sphere_smoothed = neuroimpy.convolve(vol, sphere_kernel)

    # Box kernel (uniform weights in a cuboid)
    box_kernel = neuroimpy.box_kernel(size=[3, 3, 3])
    box_smoothed = neuroimpy.convolve(vol, box_kernel)

Filtering 4D Data
-----------------

Apply spatial filters to 4D time series data:

.. code-block:: python

    # Load 4D fMRI data
    vec = neuroimpy.read_vec("fmri.nii.gz")

    # Smooth each volume independently
    smoothed_vec = neuroimpy.gaussian_blur_vec(
        vec,
        sigma=2.5
    )

    # Or apply volume-by-volume
    smoothed_vols = []
    for i in range(vec.shape[3]):
        vol = vec[..., i]
        smoothed_vol = neuroimpy.gaussian_blur(vol, sigma=2.5)
        smoothed_vols.append(smoothed_vol)

    # Reconstruct as NeuroVec
    smoothed_vec = neuroimpy.concat_vols(smoothed_vols)
    neuroimpy.write_vec(smoothed_vec, "smoothed_fmri.nii.gz")

Choosing Filter Parameters
---------------------------

**Gaussian blur:**
- Small sigma (1-2mm): minimal smoothing, preserves fine detail
- Medium sigma (3-5mm): standard fMRI preprocessing
- Large sigma (6-10mm): heavy smoothing for group analysis

**Bilateral filter:**
- `spatial_sigma`: similar to Gaussian sigma
- `intensity_sigma`: smaller values preserve more edges (0.05-0.2 typical)

**Guided filter:**
- `radius`: neighborhood size (3-7 typical)
- `epsilon`: controls edge preservation (0.001-0.1, larger = more smoothing)

**CGB smoothing:**
- `sigma`: within-region smoothing strength
- `iterations`: number of smoothing passes (1-5 typical)

Comparing Filters
-----------------

Visual comparison of different filtering approaches:

.. code-block:: python

    import matplotlib.pyplot as plt

    # Apply different filters
    gaussian = neuroimpy.gaussian_blur(vol, sigma=2.0)
    bilateral = neuroimpy.bilateral_filter(vol, spatial_sigma=2.0, intensity_sigma=0.1)

    # Extract a slice for visualization
    slice_idx = vol.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(vol.data[:, :, slice_idx], cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(gaussian.data[:, :, slice_idx], cmap='gray')
    axes[1].set_title('Gaussian')
    axes[2].imshow(bilateral.data[:, :, slice_idx], cmap='gray')
    axes[2].set_title('Bilateral')

    plt.tight_layout()
    plt.savefig('filter_comparison.png')

Visualization and Plotting
==========================

neuroim provides convenient functions for visualizing neuroimaging data using matplotlib.

Single Volume Display
----------------------

Display a single slice from a 3D volume:

.. code-block:: python

    import neuroim
    import matplotlib.pyplot as plt

    # Load a structural image
    vol = neuroim.read_vol("structural.nii.gz")

    # Plot a single slice
    fig, ax = neuroim.plot_neuro_vol(
        vol,
        slice_index=32,       # Which slice to show
        axis='z',             # Slice orientation ('x', 'y', or 'z')
        cmap='gray',          # Colormap
        title="Structural MRI",
        colorbar=True
    )

    plt.savefig("structural_slice.png")

Orthogonal Views
----------------

Display three orthogonal slices simultaneously:

.. code-block:: python

    # Plot orthogonal views at a specific coordinate
    fig = neuroim.plot_ortho(
        vol,
        coords=[32, 32, 16],  # x, y, z coordinates
        cmap='gray',
        title="Orthogonal Views"
    )

    plt.savefig("ortho_views.png")

    # Plot at the center of mass of a statistical map
    stat_vol = neuroim.read_vol("tstat.nii.gz")
    com = stat_vol.center_of_mass()

    fig = neuroim.plot_ortho(
        stat_vol,
        coords=com,
        cmap='RdBu_r',
        title="Statistical Map at Peak"
    )

Slice Montage
-------------

Display multiple slices in a grid:

.. code-block:: python

    # Create a montage of axial slices
    fig = neuroim.plot_montage(
        vol,
        axis='z',
        n_slices=12,          # Number of slices to show
        start_slice=5,        # First slice index
        step=2,               # Skip every other slice
        cols=4,               # Columns in grid
        cmap='gray',
        title="Axial Montage"
    )

    plt.savefig("axial_montage.png")

    # Automatic slice selection (evenly spaced)
    fig = neuroim.plot_montage(
        vol,
        axis='z',
        n_slices=16,
        cols=4,
        cmap='gray'
    )

Statistical Overlay
-------------------

Overlay statistical maps on anatomical images:

.. code-block:: python

    # Load anatomical and statistical images
    anat = neuroim.read_vol("T1.nii.gz")
    stat = neuroim.read_vol("tstat.nii.gz")

    # Plot overlay with thresholding
    fig = neuroim.plot_overlay(
        background=anat,
        overlay=stat,
        slice_index=25,
        axis='z',
        bg_cmap='gray',
        overlay_cmap='hot',
        threshold=2.5,        # Only show |stat| > 2.5
        alpha=0.7,            # Overlay transparency
        title="Activation Map"
    )

    plt.savefig("activation_overlay.png")

    # Two-sided threshold for positive and negative effects
    fig = neuroim.plot_overlay(
        background=anat,
        overlay=stat,
        slice_index=25,
        axis='z',
        bg_cmap='gray',
        overlay_cmap='RdBu_r',    # Diverging colormap
        threshold=(-3, 3),         # Separate thresholds
        symmetric=True             # Center colormap at zero
    )

Custom Colormapping
-------------------

Create custom color schemes:

.. code-block:: python

    # Map volume values to RGB colors
    rgb_array = neuroim.map_to_colors(
        vol,
        cmap='viridis',
        vmin=None,            # Auto-scale or specify
        vmax=None
    )

    print(rgb_array.shape)    # (x, y, z, 3) RGB values

    # Resolve colormap from string or matplotlib colormap
    cmap = neuroim.resolve_cmap('hot')

    # Use custom normalization
    from matplotlib.colors import PowerNorm

    rgb_array = neuroim.map_to_colors(
        vol,
        cmap=cmap,
        norm=PowerNorm(gamma=0.5)  # Nonlinear scaling
    )

Plotting 4D Data
----------------

Visualize time series data:

.. code-block:: python

    # Load 4D fMRI data
    vec = neuroim.read_vec("fmri.nii.gz")

    # Plot mean across time
    mean_vol = vec.mean(axis=3)
    fig, ax = neuroim.plot_neuro_vol(
        mean_vol,
        slice_index=16,
        axis='z',
        cmap='gray',
        title="Mean fMRI Signal"
    )

    # Plot standard deviation across time
    std_vol = vec.std(axis=3)
    fig, ax = neuroim.plot_neuro_vol(
        std_vol,
        slice_index=16,
        axis='z',
        cmap='hot',
        title="Temporal Standard Deviation"
    )

    # Animate through time
    fig, ax = plt.subplots()

    for t in range(0, vec.shape[3], 5):
        vol_t = vec[..., t]
        ax.clear()
        ax.imshow(vol_t.data[:, :, 16], cmap='gray')
        ax.set_title(f"Time point {t}")
        plt.pause(0.1)

ROI Visualization
-----------------

Highlight regions of interest:

.. code-block:: python

    # Create an ROI
    center = [32, 32, 16]
    roi = neuroim.spherical_roi(vol, center, radius=8.0)

    # Create a mask volume
    roi_vol = neuroim.LogicalNeuroVol(
        roi.as_mask(),
        vol.space
    )

    # Overlay ROI on anatomical image
    fig = neuroim.plot_overlay(
        background=vol,
        overlay=roi_vol.astype(float),
        slice_index=16,
        axis='z',
        bg_cmap='gray',
        overlay_cmap='Reds',
        alpha=0.5,
        title="ROI Overlay"
    )

Advanced Plotting
-----------------

Customize plots with matplotlib:

.. code-block:: python

    # Create custom figure layout
    fig = plt.figure(figsize=(15, 10))

    # Anatomical image
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(anat.data[:, :, 25], cmap='gray')
    ax1.set_title('Anatomical')
    ax1.axis('off')

    # Statistical map
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(stat.data[:, :, 25], cmap='RdBu_r', vmin=-5, vmax=5)
    ax2.set_title('T-statistic')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)

    # Overlay
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(anat.data[:, :, 25], cmap='gray')
    mask = np.abs(stat.data[:, :, 25]) > 2.5
    im3 = ax3.imshow(
        np.ma.masked_where(~mask, stat.data[:, :, 25]),
        cmap='RdBu_r',
        alpha=0.7,
        vmin=-5,
        vmax=5
    )
    ax3.set_title('Overlay')
    ax3.axis('off')

    # Time series from ROI
    ax4 = plt.subplot(2, 1, 2)
    ts = vec.series(32, 32, 16)
    ax4.plot(ts)
    ax4.set_xlabel('Time point')
    ax4.set_ylabel('Signal')
    ax4.set_title('Time Series')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("comprehensive_figure.png", dpi=300)

Glass Brain Visualization
-------------------------

Create glass brain projections:

.. code-block:: python

    # Maximum intensity projection
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sagittal projection
    axes[0].imshow(np.max(stat.data, axis=0).T, cmap='hot', origin='lower')
    axes[0].set_title('Sagittal')
    axes[0].axis('off')

    # Coronal projection
    axes[1].imshow(np.max(stat.data, axis=1).T, cmap='hot', origin='lower')
    axes[1].set_title('Coronal')
    axes[1].axis('off')

    # Axial projection
    axes[2].imshow(np.max(stat.data, axis=2), cmap='hot', origin='lower')
    axes[2].set_title('Axial')
    axes[2].axis('off')

    plt.suptitle('Glass Brain Projections')
    plt.savefig("glass_brain.png")

Visualization and Plotting
==========================

neuroim plotting is matplotlib-native and does not require nilearn. The helpers
work with plain 3D arrays, but they add the most value with ``NeuroVol`` objects:
slices are displayed with spatial orientation and world-coordinate extents when
the volume carries a ``NeuroSpace``.

Single Volume Display
---------------------

Display a small slice montage from a 3D volume:

.. code-block:: python

    import neuroim
    import matplotlib.pyplot as plt

    vol = neuroim.read_vol("structural.nii.gz")

    fig = neuroim.plot_neuro_vol(
        vol,
        zlevels=[12, 20, 28, 36],
        axis="z",
        cmap="gray",
        range="robust",
        ncol=4,
        title="Structural MRI",
        colorbar=True,
    )

    fig.savefig("structural_montage.png", dpi=150)
    plt.close(fig)

``zlevels`` are Python slice indices. Use ``axis="x"``, ``"y"``, or ``"z"``
for sagittal, coronal, or axial slicing. The compatibility argument
``along=1``/``2``/``3`` is also accepted for neuroim2-style calls.

Orthogonal Views
----------------

Display three orthogonal slices at a voxel or world coordinate:

.. code-block:: python

    fig, axes = neuroim.plot_ortho(
        vol,
        coords=(32, 32, 16),
        coord_space="voxel",
        cmap="gray",
        crosshair=True,
        title="Orthogonal Views",
    )

    fig.savefig("ortho_views.png", dpi=150)
    plt.close(fig)

World coordinates are accepted for spatial volumes:

.. code-block:: python

    fig, axes = neuroim.plot_ortho(
        vol,
        coords=(12.0, -18.0, 32.0),
        coord_space="world",
        cmap="gray",
        crosshair=True,
    )

Slice Montage
-------------

Use ``plot_montage`` when you need a regular grid of slices and access to the
created axes:

.. code-block:: python

    fig, axes = neuroim.plot_montage(
        vol,
        axis="z",
        zlevels=range(8, 56, 4),
        ncols=4,
        cmap="gray",
        range="robust",
        colorbar=True,
        title="Axial Montage",
    )

    fig.savefig("axial_montage.png", dpi=150)
    plt.close(fig)

Statistical Overlay
-------------------

Overlay statistical maps on anatomical images. With ``zlevels=None`` the helper
returns the historical three-plane orthogonal view; with ``zlevels`` it returns a
neuroim2-style slice panel grid.

.. code-block:: python

    anat = neuroim.read_vol("T1.nii.gz")
    stat = neuroim.read_vol("tstat.nii.gz")

    fig, axes = neuroim.plot_overlay(
        background=anat,
        overlay=stat,
        zlevels=[18, 24, 30, 36, 42, 48],
        axis="z",
        ncol=3,
        bg_cmap="gray",
        ov_cmap="blue-red",
        ov_thresh=2.5,
        ov_alpha_mode="soft",
        ov_symmetric=True,
        title="Activation Map",
        colorbar=True,
    )

    fig.savefig("activation_overlay.png", dpi=150)
    plt.close(fig)

For a coordinate-centered overlay:

.. code-block:: python

    fig, axes = neuroim.plot_overlay(
        anat,
        stat,
        coords=(12.0, -18.0, 32.0),
        coord_space="world",
        threshold=2.5,
        overlay_cmap="blue-red",
    )

Registration QC
---------------

The registration QC helpers validate that inputs occupy the same image grid
before plotting:

.. code-block:: python

    fig, axes = neuroim.plot_checkerboard(
        fixed,
        moving,
        zlevels=[16, 24, 32, 40],
        axis="z",
        tile=16,
        ncol=4,
        title="Checkerboard QC",
    )

    fig, axes = neuroim.plot_edge_overlay(
        fixed,
        fixed_edges,
        moving_edges,
        zlevels=[16, 24, 32, 40],
        axis="z",
        ncol=4,
        title="Edge Overlay QC",
    )

Custom Colormapping
-------------------

Use ``resolve_cmap`` and ``map_to_colors`` for lower-level color mapping:

.. code-block:: python

    data = vol.as_array()
    rgba = neuroim.map_to_colors(
        data,
        cmap=["#1f2937", "#f97316", "#fef3c7"],
        vmin=0.0,
        vmax=1000.0,
        alpha=0.85,
    )

    cmap = neuroim.resolve_cmap("blue-red")

Matplotlib Customization
------------------------

Most plotting helpers return ``(fig, axes)``. The lower-level matplotlib objects
remain available for additional annotations or layout adjustments:

.. code-block:: python

    fig, axes = neuroim.plot_montage(
        stat,
        zlevels=[18, 24, 30],
        ncols=3,
        cmap="blue-red",
        range=(-5, 5),
    )

    for ax in axes[:3]:
        ax.set_facecolor("black")

    fig.savefig("custom_stat_map.png", dpi=150)
    plt.close(fig)

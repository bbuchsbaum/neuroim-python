"""Tests for visualization utilities (plot_ortho, plot_montage, plot_overlay,
map_to_colors, resolve_cmap)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytest

from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vol import DenseNeuroVol
from neuroim.plotting import (
    plot_ortho,
    plot_montage,
    plot_overlay,
    map_to_colors,
    resolve_cmap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vol_3d():
    """Simple 20x20x20 volume with gradient data."""
    x, y, z = np.mgrid[0:20, 0:20, 0:20]
    return (np.sin(x / 5.0) * np.cos(y / 5.0) * np.sin(z / 5.0)).astype(np.float64)


@pytest.fixture
def overlay_3d(vol_3d):
    """Overlay volume with a central blob."""
    ov = np.zeros_like(vol_3d)
    ov[8:12, 8:12, 8:12] = 3.0
    return ov


# ---------------------------------------------------------------------------
# resolve_cmap
# ---------------------------------------------------------------------------

class TestResolveCmap:
    def test_standard_name(self):
        cmap = resolve_cmap("viridis")
        assert isinstance(cmap, mcolors.Colormap)
        assert cmap.name == "viridis"

    def test_gray(self):
        cmap = resolve_cmap("gray")
        assert isinstance(cmap, mcolors.Colormap)

    def test_alias_neuroimaging(self):
        cmap = resolve_cmap("neuroimaging")
        assert isinstance(cmap, mcolors.Colormap)
        assert cmap.name == "gray"

    def test_alias_activation(self):
        cmap = resolve_cmap("activation")
        assert isinstance(cmap, mcolors.Colormap)
        assert cmap.name == "hot"

    def test_alias_diverging(self):
        cmap = resolve_cmap("diverging")
        assert isinstance(cmap, mcolors.Colormap)
        assert cmap.name == "RdBu_r"

    def test_alias_stat(self):
        cmap = resolve_cmap("stat")
        assert isinstance(cmap, mcolors.Colormap)
        assert cmap.name == "RdYlBu_r"

    def test_invalid_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            resolve_cmap("totally_invalid_cmap_name_xyz")


# ---------------------------------------------------------------------------
# map_to_colors
# ---------------------------------------------------------------------------

class TestMapToColors:
    def test_shape_1d(self):
        data = np.array([0.0, 0.5, 1.0])
        rgba = map_to_colors(data)
        assert rgba.shape == (3, 4)

    def test_shape_2d(self):
        data = np.random.rand(5, 7)
        rgba = map_to_colors(data)
        assert rgba.shape == (5, 7, 4)

    def test_shape_3d(self):
        data = np.random.rand(4, 5, 6)
        rgba = map_to_colors(data)
        assert rgba.shape == (4, 5, 6, 4)

    def test_values_in_range(self):
        data = np.linspace(-10, 10, 100)
        rgba = map_to_colors(data)
        assert rgba.min() >= 0.0
        assert rgba.max() <= 1.0

    def test_custom_vmin_vmax(self):
        data = np.array([0.0, 5.0, 10.0])
        rgba_default = map_to_colors(data)
        rgba_custom = map_to_colors(data, vmin=0.0, vmax=10.0)
        # With matching vmin/vmax the results should be the same
        np.testing.assert_array_almost_equal(rgba_default, rgba_custom)

    def test_custom_cmap(self):
        data = np.array([0.0, 1.0])
        rgba_hot = map_to_colors(data, cmap="hot")
        rgba_cool = map_to_colors(data, cmap="cool")
        # Different colormaps should give different colours
        assert not np.allclose(rgba_hot, rgba_cool)

    def test_constant_data(self):
        data = np.ones((3, 3)) * 5.0
        rgba = map_to_colors(data)
        assert rgba.shape == (3, 3, 4)
        # All pixels should be the same colour
        assert np.all(rgba == rgba[0, 0])

    def test_nan_handling(self):
        data = np.array([0.0, np.nan, 1.0])
        rgba = map_to_colors(data)
        assert rgba.shape == (3, 4)
        # NaN should map to transparent (alpha=0) in most colormaps
        # At minimum, shape should be correct and no crash


# ---------------------------------------------------------------------------
# plot_ortho
# ---------------------------------------------------------------------------

class TestPlotOrtho:
    def test_returns_fig_and_axes(self, vol_3d):
        fig, axes = plot_ortho(vol_3d)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3
        plt.close(fig)

    def test_default_center_coords(self, vol_3d):
        fig, axes = plot_ortho(vol_3d)
        # Should not raise
        assert axes[0].get_title() == "Axial"
        assert axes[1].get_title() == "Sagittal"
        assert axes[2].get_title() == "Coronal"
        plt.close(fig)

    def test_custom_coords(self, vol_3d):
        fig, axes = plot_ortho(vol_3d, coords=(5, 5, 5))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title(self, vol_3d):
        fig, axes = plot_ortho(vol_3d, title="Test Title")
        assert fig._suptitle.get_text() == "Test Title"
        plt.close(fig)

    def test_custom_cmap(self, vol_3d):
        fig, axes = plot_ortho(vol_3d, cmap="hot")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, vol_3d):
        fig, axes = plot_ortho(vol_3d, figsize=(8, 3))
        assert abs(fig.get_figwidth() - 8) < 1
        plt.close(fig)

    def test_preexisting_axes(self, vol_3d):
        ext_fig, ext_axes = plt.subplots(1, 3)
        fig, axes = plot_ortho(vol_3d, axes=ext_axes)
        assert fig is ext_fig
        plt.close(fig)

    def test_with_neurovol_like_object(self):
        """Test that objects with a .data attribute work."""

        class FakeVol:
            def __init__(self):
                self.data = np.random.rand(10, 10, 10)

        fig, axes = plot_ortho(FakeVol())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_montage
# ---------------------------------------------------------------------------

class TestPlotMontage:
    def test_returns_fig_and_axes(self, vol_3d):
        fig, axes = plot_montage(vol_3d)
        assert isinstance(fig, plt.Figure)
        assert len(axes) >= 16
        plt.close(fig)

    def test_custom_n_slices(self, vol_3d):
        fig, axes = plot_montage(vol_3d, n_slices=8)
        # 8 slices, 4 cols -> 2 rows -> 8 axes
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_axis(self, vol_3d):
        for ax_idx in (0, 1, 2):
            fig, axes = plot_montage(vol_3d, axis=ax_idx, n_slices=4)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_ncols(self, vol_3d):
        fig, axes = plot_montage(vol_3d, n_slices=9, ncols=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title(self, vol_3d):
        fig, axes = plot_montage(vol_3d, title="Montage")
        assert fig._suptitle.get_text() == "Montage"
        plt.close(fig)

    def test_custom_figsize(self, vol_3d):
        fig, axes = plot_montage(vol_3d, figsize=(16, 8), n_slices=4)
        assert abs(fig.get_figwidth() - 16) < 1
        plt.close(fig)

    def test_preexisting_axes(self, vol_3d):
        ext_fig, ext_axes = plt.subplots(2, 4)
        fig, axes = plot_montage(vol_3d, n_slices=8, axes=ext_axes)
        assert fig is ext_fig
        plt.close(fig)

    def test_neurovol_respects_flipped_world_axis(self):
        data = np.zeros((4, 5, 2), dtype=float)
        data[0, 0, 0] = 1.0
        data[3, 4, 0] = 2.0
        affine = np.eye(4)
        affine[0, 0] = -1.0
        affine[0, 3] = 3.0
        vol = DenseNeuroVol(data, NeuroSpace((4, 5, 2), trans=affine))

        fig, axes = plot_montage(vol, zlevels=[0], ncols=1, range="data")
        try:
            shown = np.asarray(axes[0].images[0].get_array())
            np.testing.assert_array_equal(np.argwhere(shown == 1.0), [[0, 3]])
            np.testing.assert_array_equal(np.argwhere(shown == 2.0), [[4, 0]])
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# plot_overlay
# ---------------------------------------------------------------------------

class TestPlotOverlay:
    def test_returns_fig_and_axes(self, vol_3d, overlay_3d):
        fig, axes = plot_overlay(vol_3d, overlay_3d)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3
        plt.close(fig)

    def test_with_threshold(self, vol_3d, overlay_3d):
        fig, axes = plot_overlay(vol_3d, overlay_3d, threshold=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_colormaps(self, vol_3d, overlay_3d):
        fig, axes = plot_overlay(
            vol_3d, overlay_3d, base_cmap="bone", overlay_cmap="jet"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_alpha(self, vol_3d, overlay_3d):
        fig, axes = plot_overlay(vol_3d, overlay_3d, alpha=0.3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_coords(self, vol_3d, overlay_3d):
        fig, axes = plot_overlay(vol_3d, overlay_3d, coords=(10, 10, 10))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_preexisting_axes(self, vol_3d, overlay_3d):
        ext_fig, ext_axes = plt.subplots(1, 3)
        fig, axes = plot_overlay(vol_3d, overlay_3d, axes=ext_axes)
        assert fig is ext_fig
        plt.close(fig)

    def test_overlay_titles(self, vol_3d, overlay_3d):
        fig, axes = plot_overlay(vol_3d, overlay_3d)
        assert axes[0].get_title() == "Axial"
        assert axes[1].get_title() == "Sagittal"
        assert axes[2].get_title() == "Coronal"
        plt.close(fig)

    def test_montage_overlay_aliases_and_alpha_map(self):
        sp = NeuroSpace((5, 5, 3))
        bg = DenseNeuroVol(np.arange(75, dtype=float).reshape(5, 5, 3), sp)
        ov_data = np.zeros((5, 5, 3), dtype=float)
        ov_data[2, 2, 1] = 5.0
        ov = DenseNeuroVol(ov_data, sp)

        fig, axes = plot_overlay(
            background=bg,
            overlay=ov,
            zlevels=[1],
            ncol=1,
            bg_cmap="grays",
            ov_cmap="hot",
            ov_alpha_mode="proportional",
            ov_thresh=1.0,
        )
        try:
            assert len(axes) == 1
            assert axes[0].get_title() == "z = 1"
            assert len(axes[0].images) == 2
            rgba = np.asarray(axes[0].images[1].get_array())
            assert rgba.shape[-1] == 4
            assert rgba[..., 3].max() == pytest.approx(0.5)
            assert np.count_nonzero(rgba[..., 3]) == 1
        finally:
            plt.close(fig)

    def test_overlay_accepts_world_coordinates_for_ortho(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        sp = NeuroSpace((5, 5, 5), trans=affine)
        bg = DenseNeuroVol(np.zeros((5, 5, 5), dtype=float), sp)
        ov_data = np.zeros((5, 5, 5), dtype=float)
        ov_data[2, 2, 2] = 4.0
        ov = DenseNeuroVol(ov_data, sp)

        fig, axes = plot_overlay(
            bg,
            ov,
            coords=(4.0, 4.0, 4.0),
            coord_space="world",
            threshold=1.0,
        )
        try:
            assert [ax.get_title() for ax in axes] == ["Axial", "Sagittal", "Coronal"]
            assert all(len(ax.images) == 2 for ax in axes)
        finally:
            plt.close(fig)

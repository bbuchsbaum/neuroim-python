"""
Tests for plotting functionality.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

import neuroimpy as pn
from neuroimpy.plotting import plot_neuro_vol
from neuroimpy.neuro_space import NeuroSpace
from neuroimpy.neuro_vol import DenseNeuroVol

# Use non-interactive backend for testing
matplotlib.use('Agg')


class TestPlotNeuroVol:
    """Test plotting functionality for NeuroVol objects."""
    
    @pytest.fixture
    def sample_vol(self):
        """Create a sample NeuroVol for testing."""
        space = NeuroSpace(dim=(20, 20, 20), spacing=(1, 1, 1))
        # Create data with some structure for visual testing
        x, y, z = np.mgrid[0:20, 0:20, 0:20]
        data = np.sin(x/5) * np.cos(y/5) * np.sin(z/5)
        return DenseNeuroVol(data, space)
    
    def test_basic_plot(self, sample_vol):
        """Test basic plotting functionality."""
        fig = plot_neuro_vol(sample_vol)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0  # Should have at least one axis
        
        # Check that slices were created
        axes = fig.get_axes()
        # Should have 6 slice plots + 1 colorbar by default
        assert len([ax for ax in axes if ax.get_title()]) >= 6
        
        plt.close(fig)
    
    def test_custom_zlevels(self, sample_vol):
        """Test plotting with custom z-levels."""
        zlevels = [5, 10, 15]
        fig = plot_neuro_vol(sample_vol, zlevels=zlevels)
        
        # Check that correct number of slices were created
        axes = fig.get_axes()
        slice_axes = [ax for ax in axes if ax.get_title().startswith('Slice')]
        assert len(slice_axes) == 3
        
        # Check slice indices in titles
        titles = [ax.get_title() for ax in slice_axes]
        assert 'Slice 5' in titles
        assert 'Slice 10' in titles
        assert 'Slice 15' in titles
        
        plt.close(fig)
    
    def test_intensity_range(self, sample_vol):
        """Test plotting with custom intensity range."""
        irange = (-0.5, 0.5)
        fig = plot_neuro_vol(sample_vol, irange=irange)
        
        # Check that images have correct vmin/vmax
        for ax in fig.get_axes():
            for im in ax.get_images():
                if hasattr(im, 'get_clim'):
                    vmin, vmax = im.get_clim()
                    assert vmin == irange[0]
                    assert vmax == irange[1]
        
        plt.close(fig)
    
    def test_colormap(self, sample_vol):
        """Test plotting with different colormaps."""
        fig = plot_neuro_vol(sample_vol, cmap='hot')
        
        # Check that at least one image uses the specified colormap
        found_hot_cmap = False
        for ax in fig.get_axes():
            for im in ax.get_images():
                if hasattr(im, 'get_cmap'):
                    if im.get_cmap().name == 'hot':
                        found_hot_cmap = True
        
        assert found_hot_cmap
        plt.close(fig)
    
    def test_thresholding(self, sample_vol):
        """Test plotting with thresholding."""
        thresh = (-0.2, 0.2)
        fig = plot_neuro_vol(sample_vol, thresh=thresh)
        
        # Thresholding should mask values outside range
        # This is hard to test directly, but we can verify the plot completes
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_alpha_transparency(self, sample_vol):
        """Test plotting with alpha transparency."""
        fig = plot_neuro_vol(sample_vol, alpha=0.5)
        
        # Check that images have correct alpha
        for ax in fig.get_axes():
            for im in ax.get_images():
                if hasattr(im, 'get_alpha'):
                    alpha = im.get_alpha()
                    if alpha is not None:
                        assert alpha == 0.5
        
        plt.close(fig)
    
    def test_background_volume(self, sample_vol):
        """Test plotting with background volume."""
        # Create a background volume
        bg_space = NeuroSpace(dim=(20, 20, 20))
        bg_data = np.ones((20, 20, 20)) * 0.5
        bg_vol = DenseNeuroVol(bg_data, bg_space)
        
        fig = plot_neuro_vol(sample_vol, bgvol=bg_vol, bgcmap='bone')
        
        # Should have two images per slice (background + overlay)
        for ax in fig.get_axes():
            if ax.get_title().startswith('Slice'):
                assert len(ax.get_images()) >= 2
        
        plt.close(fig)
    
    def test_figure_size(self, sample_vol):
        """Test plotting with custom figure size."""
        figsize = (16, 10)
        fig = plot_neuro_vol(sample_vol, figsize=figsize)
        
        # Check figure size (may not be exact due to DPI)
        assert abs(fig.get_figwidth() - figsize[0]) < 1
        assert abs(fig.get_figheight() - figsize[1]) < 1
        
        plt.close(fig)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Very small volume
        small_space = NeuroSpace(dim=(3, 3, 3))
        small_vol = DenseNeuroVol(np.ones((3, 3, 3)), small_space)
        
        fig = plot_neuro_vol(small_vol)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Single slice volume
        single_space = NeuroSpace(dim=(10, 10, 1))
        single_vol = DenseNeuroVol(np.ones((10, 10, 1)), single_space)
        
        fig = plot_neuro_vol(single_vol)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_method_added_to_neurovol(self, sample_vol):
        """Test that plot method is added to NeuroVol class."""
        # The last line of plotting.py adds the method
        assert hasattr(sample_vol, 'plot')
        
        # Test using the method
        fig = sample_vol.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_slice_extraction(self, sample_vol):
        """Test that slice extraction works correctly."""
        # Test direct data extraction since slice method doesn't exist
        fig = plot_neuro_vol(sample_vol, zlevels=[5, 10])
        
        # Verify that figure was created with correct slices
        axes = fig.get_axes()
        slice_axes = [ax for ax in axes if ax.get_title().startswith('Slice')]
        
        # Should have 2 slices
        assert len(slice_axes) == 2
        
        # Check slice titles
        titles = [ax.get_title() for ax in slice_axes]
        assert 'Slice 5' in titles
        assert 'Slice 10' in titles
        
        # Verify that data was displayed
        for ax in slice_axes:
            assert len(ax.get_images()) > 0
            
        plt.close(fig)
    
    def test_colorbar_creation(self, sample_vol):
        """Test that colorbar is created correctly."""
        fig = plot_neuro_vol(sample_vol)
        
        # Find colorbar axes
        colorbars = [ax for ax in fig.get_axes() if hasattr(ax, 'collections')]
        
        # Should have at least one colorbar
        assert len(fig.get_axes()) > 6  # 6 slices + colorbar
        
        plt.close(fig)


def test_matplotlib_backend():
    """Test that matplotlib backend is set correctly for testing."""
    assert matplotlib.get_backend().lower() == 'agg'


def test_plot_integration():
    """Integration test with full pipeline."""
    # Create a volume with interesting structure
    space = NeuroSpace(dim=(30, 30, 30), spacing=(2, 2, 2))
    
    # Create a sphere in the center
    x, y, z = np.ogrid[:30, :30, :30]
    center = [15, 15, 15]
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    sphere_data = (r < 10).astype(float)
    
    vol = DenseNeuroVol(sphere_data, space)
    
    # Test plotting with various options
    fig = plot_neuro_vol(
        vol,
        cmap='viridis',
        zlevels=[10, 15, 20],
        irange=(0, 1),
        alpha=0.8,
        figsize=(10, 6)
    )
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) > 3  # At least 3 slices + colorbar
    
    plt.close(fig)
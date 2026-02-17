"""Tests for simulation utilities."""

import pytest
import numpy as np
from neuroimpy import NeuroSpace
from neuroimpy.simulation import simulate_fmri, prepare_confounds, make_time_weights
from neuroimpy.neuro_vec import DenseNeuroVec


class TestSimulateFmri:
    """Tests for simulate_fmri."""

    def test_basic_output_shape(self):
        space = NeuroSpace(dim=[10, 10, 10], spacing=[2, 2, 2])
        result = simulate_fmri(space, n_timepoints=20, seed=42)
        assert isinstance(result, DenseNeuroVec)
        assert result.shape == (10, 10, 10, 20)

    def test_seed_reproducibility(self):
        space = NeuroSpace(dim=[8, 8, 8])
        a = simulate_fmri(space, n_timepoints=15, seed=123)
        b = simulate_fmri(space, n_timepoints=15, seed=123)
        np.testing.assert_array_equal(a.data, b.data)

    def test_different_seeds_differ(self):
        space = NeuroSpace(dim=[8, 8, 8])
        a = simulate_fmri(space, n_timepoints=15, seed=1)
        b = simulate_fmri(space, n_timepoints=15, seed=2)
        assert not np.array_equal(a.data, b.data)

    def test_signal_exceeds_noise(self):
        space = NeuroSpace(dim=[12, 12, 12])
        result = simulate_fmri(space, n_timepoints=50, noise_sd=0.1, signal_sd=5.0, seed=0)
        # With strong signal, max absolute value should be notably larger
        # than what pure noise of sd=0.1 would give
        assert np.max(np.abs(result.data)) > 1.0

    def test_no_regions(self):
        space = NeuroSpace(dim=[6, 6, 6])
        result = simulate_fmri(space, n_timepoints=10, n_regions=0, noise_sd=1.0, seed=7)
        assert result.shape == (6, 6, 6, 10)
        # Pure noise -- standard deviation should be close to 1
        assert 0.5 < np.std(result.data) < 2.0

    def test_spacing_propagated(self):
        space = NeuroSpace(dim=[8, 8, 8], spacing=[3.0, 3.0, 3.0])
        result = simulate_fmri(space, n_timepoints=5, tr=1.5, seed=0)
        np.testing.assert_allclose(result.spacing[:3], [3.0, 3.0, 3.0])
        np.testing.assert_allclose(result.spacing[3], 1.5)

    def test_single_timepoint(self):
        space = NeuroSpace(dim=[5, 5, 5])
        result = simulate_fmri(space, n_timepoints=1, seed=0)
        assert result.shape == (5, 5, 5, 1)


class TestPrepareConfounds:
    """Tests for prepare_confounds."""

    def test_basic_no_extras(self):
        mp = np.random.randn(100, 6)
        result = prepare_confounds(mp, include_derivatives=False, include_squared=False)
        assert result.shape == (100, 6)
        np.testing.assert_array_equal(result, mp)

    def test_with_derivatives(self):
        mp = np.random.randn(50, 6)
        result = prepare_confounds(mp, include_derivatives=True, include_squared=False)
        assert result.shape == (50, 12)

    def test_with_squared(self):
        mp = np.random.randn(50, 6)
        result = prepare_confounds(mp, include_derivatives=False, include_squared=True)
        assert result.shape == (50, 12)  # 6 original + 6 squared

    def test_with_both(self):
        mp = np.random.randn(50, 6)
        result = prepare_confounds(mp, include_derivatives=True, include_squared=True)
        # 6 original + 6 deriv = 12, then 12 + 12 squared = 24
        assert result.shape == (50, 24)

    def test_derivative_values(self):
        mp = np.arange(30, dtype=float).reshape(5, 6)
        result = prepare_confounds(mp, include_derivatives=True, include_squared=False)
        # First row derivative should be 0 (zero-padded)
        np.testing.assert_array_equal(result[0, 6:], [0, 0, 0, 0, 0, 0])
        # Second row derivative should be the diff
        np.testing.assert_array_equal(result[1, 6:], mp[1] - mp[0])

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="n_timepoints, 6"):
            prepare_confounds(np.ones((10, 3)))

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            prepare_confounds(np.ones(6))


class TestMakeTimeWeights:
    """Tests for make_time_weights."""

    def test_exponential_sums_to_one(self):
        w = make_time_weights(100, method="exponential", decay=0.05)
        assert w.shape == (100,)
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_exponential_decreasing(self):
        w = make_time_weights(50, method="exponential", decay=0.1)
        assert np.all(np.diff(w) <= 0)

    def test_linear_sums_to_one(self):
        w = make_time_weights(20, method="linear")
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_linear_increasing(self):
        w = make_time_weights(30, method="linear")
        assert np.all(np.diff(w) >= 0)

    def test_uniform_equal(self):
        w = make_time_weights(10, method="uniform")
        np.testing.assert_allclose(w, np.full(10, 0.1))

    def test_uniform_sums_to_one(self):
        w = make_time_weights(7, method="uniform")
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            make_time_weights(10, method="quadratic")

    def test_single_timepoint(self):
        for method in ("exponential", "linear", "uniform"):
            w = make_time_weights(1, method=method)
            np.testing.assert_allclose(w, [1.0])

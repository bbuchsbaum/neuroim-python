"""Tests for IndexLookupVol."""

import numpy as np
import pytest

from neuroimpy.neuro_space import NeuroSpace
from neuroimpy.index_lookup_vol import IndexLookupVol


@pytest.fixture
def space():
    return NeuroSpace([10, 10, 10])


@pytest.fixture
def active_indices():
    return np.array([5, 20, 100, 500, 999])


@pytest.fixture
def lut(space, active_indices):
    return IndexLookupVol(space, active_indices)


class TestInit:
    def test_basic_construction(self, space, active_indices):
        lut = IndexLookupVol(space, active_indices)
        assert len(lut) == 5

    def test_invalid_space(self, active_indices):
        with pytest.raises(TypeError, match="space must be a NeuroSpace"):
            IndexLookupVol("not_a_space", active_indices)

    def test_out_of_range_indices(self, space):
        with pytest.raises(ValueError, match="indices must be in range"):
            IndexLookupVol(space, np.array([1000]))

    def test_negative_indices(self, space):
        with pytest.raises(ValueError, match="indices must be in range"):
            IndexLookupVol(space, np.array([-1, 5]))

    def test_empty_indices(self, space):
        lut = IndexLookupVol(space, np.array([], dtype=int))
        assert len(lut) == 0


class TestLookupIndex:
    def test_lookup_existing(self, lut):
        assert lut.lookup_index(5) == 0
        assert lut.lookup_index(20) == 1
        assert lut.lookup_index(100) == 2
        assert lut.lookup_index(500) == 3
        assert lut.lookup_index(999) == 4

    def test_lookup_missing(self, lut):
        with pytest.raises(KeyError):
            lut.lookup_index(6)

    def test_lookup_out_of_range(self, lut):
        with pytest.raises(KeyError):
            lut.lookup_index(1000)


class TestGridToTable:
    def test_single_index(self, lut):
        result = lut.grid_to_table(np.array([5]))
        np.testing.assert_array_equal(result, [0])

    def test_multiple_indices(self, lut):
        result = lut.grid_to_table(np.array([5, 500, 999]))
        np.testing.assert_array_equal(result, [0, 3, 4])

    def test_all_indices(self, lut, active_indices):
        result = lut.grid_to_table(active_indices)
        np.testing.assert_array_equal(result, np.arange(5))

    def test_missing_raises(self, lut):
        with pytest.raises(KeyError):
            lut.grid_to_table(np.array([5, 7]))


class TestTableToGrid:
    def test_single_index(self, lut):
        result = lut.table_to_grid(np.array([0]))
        np.testing.assert_array_equal(result, [5])

    def test_all_indices(self, lut, active_indices):
        result = lut.table_to_grid(np.arange(5))
        np.testing.assert_array_equal(result, active_indices)

    def test_out_of_range_raises(self, lut):
        with pytest.raises(IndexError):
            lut.table_to_grid(np.array([5]))

    def test_negative_raises(self, lut):
        with pytest.raises(IndexError):
            lut.table_to_grid(np.array([-1]))


class TestRoundTrip:
    def test_grid_table_grid(self, lut, active_indices):
        table = lut.grid_to_table(active_indices)
        recovered = lut.table_to_grid(table)
        np.testing.assert_array_equal(recovered, active_indices)

    def test_table_grid_table(self, lut):
        table_ids = np.array([0, 2, 4])
        grid = lut.table_to_grid(table_ids)
        recovered = lut.grid_to_table(grid)
        np.testing.assert_array_equal(recovered, table_ids)


class TestContains:
    def test_contains_active(self, lut):
        assert 5 in lut
        assert 500 in lut

    def test_not_contains_inactive(self, lut):
        assert 6 not in lut
        assert 0 not in lut


class TestRepr:
    def test_repr(self, lut):
        r = repr(lut)
        assert "IndexLookupVol" in r
        assert "10 x 10 x 10" in r
        assert "5 / 1000" in r

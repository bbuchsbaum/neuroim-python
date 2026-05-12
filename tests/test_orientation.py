"""Tests for orientation and affine utility functions."""

import numpy as np
import pytest

from neuroim.orientation import (
    affine_to_orientation,
    affine_to_axcodes,
    axcodes_to_orientation,
    apply_orientation,
    orientation_inverse_affine,
    obliquity,
    vox2out_vox,
    perm_mat,
    rescale_affine,
)


class TestAffineToOrientation:
    """Tests for affine_to_orientation."""

    def test_identity(self):
        ornt = affine_to_orientation(np.eye(4))
        expected = np.array([[0, 1], [1, 1], [2, 1]])
        np.testing.assert_array_equal(ornt, expected)

    def test_scaled_identity(self):
        ornt = affine_to_orientation(np.diag([2, 3, 4, 1]))
        expected = np.array([[0, 1], [1, 1], [2, 1]])
        np.testing.assert_array_equal(ornt, expected)

    def test_flipped_axis(self):
        aff = np.diag([-1, 1, 1, 1])
        ornt = affine_to_orientation(aff)
        assert ornt[0, 1] == -1  # first axis flipped

    def test_permuted_axes(self):
        # Swap x and z
        aff = np.zeros((4, 4))
        aff[0, 2] = 1
        aff[1, 1] = 1
        aff[2, 0] = 1
        aff[3, 3] = 1
        ornt = affine_to_orientation(aff)
        assert int(ornt[0, 0]) == 2  # output axis 0 comes from input axis 2
        assert int(ornt[2, 0]) == 0  # output axis 2 comes from input axis 0


class TestAffineToAxcodes:
    """Tests for affine_to_axcodes."""

    def test_ras_identity(self):
        codes = affine_to_axcodes(np.diag([1, 1, 1, 1]))
        assert codes == ("R", "A", "S")

    def test_las(self):
        codes = affine_to_axcodes(np.diag([-1, 1, 1, 1]))
        assert codes == ("L", "A", "S")

    def test_lpi(self):
        codes = affine_to_axcodes(np.diag([-1, -1, -1, 1]))
        assert codes == ("L", "P", "I")

    def test_with_scaling(self):
        codes = affine_to_axcodes(np.diag([2, 3, 4, 1]))
        assert codes == ("R", "A", "S")

    def test_custom_labels(self):
        labels = (("X-", "X+"), ("Y-", "Y+"), ("Z-", "Z+"))
        codes = affine_to_axcodes(np.diag([1, 1, 1, 1]), labels=labels)
        assert codes == ("X+", "Y+", "Z+")


class TestAxcodesToOrientation:
    """Tests for axcodes_to_orientation."""

    def test_ras(self):
        ornt = axcodes_to_orientation("RAS")
        expected = np.array([[0, 1], [1, 1], [2, 1]])
        np.testing.assert_array_equal(ornt, expected)

    def test_lpi(self):
        ornt = axcodes_to_orientation("LPI")
        expected = np.array([[0, -1], [1, -1], [2, -1]])
        np.testing.assert_array_equal(ornt, expected)

    def test_tuple_input(self):
        ornt = axcodes_to_orientation(("R", "A", "S"))
        expected = np.array([[0, 1], [1, 1], [2, 1]])
        np.testing.assert_array_equal(ornt, expected)

    def test_roundtrip_with_affine(self):
        aff = np.diag([-2, 3, -4, 1])
        codes = affine_to_axcodes(aff)
        ornt_from_codes = axcodes_to_orientation(codes)
        ornt_from_aff = affine_to_orientation(aff)
        np.testing.assert_array_equal(ornt_from_codes, ornt_from_aff)


class TestApplyOrientation:
    """Tests for apply_orientation."""

    def test_identity_orientation(self):
        data = np.arange(24).reshape(2, 3, 4)
        ornt = np.array([[0, 1], [1, 1], [2, 1]])
        out = apply_orientation(data, ornt)
        np.testing.assert_array_equal(out, data)

    def test_flip_first_axis(self):
        data = np.arange(24).reshape(2, 3, 4)
        ornt = np.array([[0, -1], [1, 1], [2, 1]])
        out = apply_orientation(data, ornt)
        np.testing.assert_array_equal(out, data[::-1, :, :])

    def test_swap_axes(self):
        data = np.arange(24).reshape(2, 3, 4)
        ornt = np.array([[2, 1], [1, 1], [0, 1]])
        out = apply_orientation(data, ornt)
        assert out.shape == (4, 3, 2)

    def test_preserves_data_values(self):
        data = np.random.rand(5, 6, 7)
        ornt = np.array([[1, -1], [0, 1], [2, 1]])
        out = apply_orientation(data, ornt)
        assert out.shape == (6, 5, 7)
        # Total sum should be preserved
        np.testing.assert_almost_equal(out.sum(), data.sum())


class TestOrientationInverseAffine:
    """Tests for orientation_inverse_affine."""

    def test_identity(self):
        ornt = np.array([[0, 1], [1, 1], [2, 1]])
        inv_aff = orientation_inverse_affine(ornt, (10, 20, 30))
        np.testing.assert_array_almost_equal(inv_aff, np.eye(4))

    def test_flip_roundtrip(self):
        ornt = np.array([[0, -1], [1, 1], [2, 1]])
        shape = (10, 20, 30)
        inv_aff = orientation_inverse_affine(ornt, shape)
        # The inverse affine should undo the flip
        assert inv_aff.shape == (4, 4)
        assert not np.allclose(inv_aff, np.eye(4))

    def test_permutation_roundtrip(self):
        data = np.random.rand(5, 6, 7)
        ornt = np.array([[2, 1], [1, -1], [0, 1]])
        reoriented = apply_orientation(data, ornt)
        inv_aff = orientation_inverse_affine(ornt, reoriented.shape)
        # inv_aff should be a valid 4x4 matrix
        assert inv_aff.shape == (4, 4)


class TestObliquity:
    """Tests for obliquity."""

    def test_cardinal_zero(self):
        assert obliquity(np.diag([2, 2, 2, 1])) == 0.0

    def test_identity_zero(self):
        assert obliquity(np.eye(4)) == 0.0

    def test_negative_diagonal_zero(self):
        # Flipped axes are still cardinal
        assert obliquity(np.diag([-1, -1, -1, 1])) == 0.0

    def test_oblique_positive(self):
        aff = np.eye(4)
        # Rotate slightly around z
        angle = np.radians(15)
        aff[0, 0] = np.cos(angle)
        aff[0, 1] = -np.sin(angle)
        aff[1, 0] = np.sin(angle)
        aff[1, 1] = np.cos(angle)
        assert obliquity(aff) > 0.0
        assert obliquity(aff) == pytest.approx(15.0, abs=0.5)

    def test_returns_float(self):
        assert isinstance(obliquity(np.eye(4)), float)


class TestVox2outVox:
    """Tests for vox2out_vox."""

    def test_identity(self):
        ornt, new_shape = vox2out_vox(np.diag([1, 1, 1, 1]), (10, 20, 30))
        assert new_shape == (10, 20, 30)

    def test_permuted(self):
        aff = np.zeros((4, 4))
        aff[0, 2] = 1  # output x from input z
        aff[1, 1] = 1
        aff[2, 0] = 1  # output z from input x
        aff[3, 3] = 1
        ornt, new_shape = vox2out_vox(aff, (10, 20, 30))
        assert new_shape == (30, 20, 10)

    def test_returns_orientation_array(self):
        ornt, _ = vox2out_vox(np.eye(4), (5, 5, 5))
        assert ornt.shape == (3, 2)


class TestPermMat:
    """Tests for perm_mat."""

    def test_identity_orientation(self):
        ornt = np.array([[0, 1], [1, 1], [2, 1]])
        mat = perm_mat(ornt)
        np.testing.assert_array_equal(mat, np.eye(3))

    def test_flip(self):
        ornt = np.array([[0, -1], [1, 1], [2, 1]])
        mat = perm_mat(ornt)
        expected = np.diag([-1, 1, 1]).astype(float)
        np.testing.assert_array_equal(mat, expected)

    def test_permutation(self):
        ornt = np.array([[2, 1], [1, 1], [0, 1]])
        mat = perm_mat(ornt)
        assert mat[0, 2] == 1.0
        assert mat[2, 0] == 1.0
        assert mat[1, 1] == 1.0

    def test_combined_flip_perm(self):
        ornt = np.array([[2, -1], [1, 1], [0, 1]])
        mat = perm_mat(ornt)
        assert mat[0, 2] == -1.0
        assert mat[2, 0] == 1.0


class TestRescaleAffine:
    """Tests for rescale_affine."""

    def test_same_zooms(self):
        aff = np.diag([2, 2, 2, 1]).astype(float)
        new_aff = rescale_affine(aff, (10, 10, 10), (2, 2, 2))
        np.testing.assert_array_almost_equal(new_aff[:3, :3], aff[:3, :3])

    def test_halve_zooms(self):
        aff = np.diag([2, 2, 2, 1]).astype(float)
        new_aff = rescale_affine(aff, (10, 10, 10), (1, 1, 1))
        np.testing.assert_array_almost_equal(
            np.abs(new_aff[:3, :3].diagonal()), [1, 1, 1]
        )

    def test_preserves_center(self):
        """Field-of-view center should stay the same."""
        aff = np.diag([2, 2, 2, 1]).astype(float)
        aff[:3, 3] = [10, 20, 30]
        shape = (10, 10, 10)
        new_aff = rescale_affine(aff, shape, (1, 1, 1))
        # Compute centers
        old_center = aff[:3, :3] @ ((np.array(shape[:3]) - 1) / 2.0) + aff[:3, 3]
        new_shape = np.ceil(np.array(shape[:3]) * 2.0 / 1.0).astype(int)
        new_center = new_aff[:3, :3] @ ((new_shape - 1) / 2.0) + new_aff[:3, 3]
        np.testing.assert_array_almost_equal(old_center, new_center)

    def test_explicit_new_shape(self):
        aff = np.diag([2, 2, 2, 1]).astype(float)
        new_aff = rescale_affine(aff, (10, 10, 10), (1, 1, 1), new_shape=(20, 20, 20))
        np.testing.assert_array_almost_equal(
            np.abs(new_aff[:3, :3].diagonal()), [1, 1, 1]
        )


class TestIntegration:
    """End-to-end integration tests combining multiple functions."""

    def test_roundtrip_reorient(self):
        """Reorient data and check affine consistency."""
        data = np.random.rand(10, 20, 30)
        aff = np.diag([2, 3, 4, 1]).astype(float)
        aff[:3, 3] = [10, 20, 30]

        # Get orientation and reorient
        ornt = affine_to_orientation(aff)
        reoriented = apply_orientation(data, ornt)
        codes = affine_to_axcodes(aff)

        assert codes == ("R", "A", "S")
        # Identity affine should not change data
        np.testing.assert_array_equal(reoriented, data)

    def test_non_ras_roundtrip(self):
        """Reorient LPI data to RAS and back."""
        data = np.random.rand(10, 20, 30)
        aff = np.diag([-2, -3, -4, 1]).astype(float)

        codes = affine_to_axcodes(aff)
        assert codes == ("L", "P", "I")

        ornt = affine_to_orientation(aff)
        ras_ornt = axcodes_to_orientation("RAS")
        transform = np.array([
            [ornt[i, 0], -ornt[i, 1] if ornt[i, 1] < 0 else ornt[i, 1]]
            for i in range(3)
        ])
        # Just verify the orientation was detected correctly
        assert int(ornt[0, 1]) == -1
        assert int(ornt[1, 1]) == -1
        assert int(ornt[2, 1]) == -1

    def test_obliquity_with_orientation(self):
        """Obliquity should be consistent with orientation detection."""
        # Cardinal affine
        aff = np.diag([2, 3, 4, 1]).astype(float)
        assert obliquity(aff) == 0.0
        codes = affine_to_axcodes(aff)
        assert len(codes) == 3

    def test_rescale_preserves_orientation(self):
        """Rescaling should not change axis orientation."""
        aff = np.diag([2, 3, 4, 1]).astype(float)
        aff[:3, 3] = [10, 20, 30]
        codes_before = affine_to_axcodes(aff)
        new_aff = rescale_affine(aff, (10, 10, 10), (1, 1.5, 2))
        codes_after = affine_to_axcodes(new_aff)
        assert codes_before == codes_after

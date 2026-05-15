"""Tests for neuroim2-style top-level compatibility helpers."""

import numpy as np
import pytest

import neuroim as ni
import neuroim.compat as compat


def test_compat_geometry_wrappers_delegate_to_space_methods():
    sp = ni.NeuroSpace((3, 4, 5), spacing=(2, 3, 4), origin=(10, 20, 30))
    vol = ni.DenseNeuroVol(np.arange(60).reshape((3, 4, 5), order="F"), sp)

    assert compat.space(vol) is sp
    np.testing.assert_array_equal(compat.spacing(vol), sp.spacing)
    np.testing.assert_array_equal(compat.origin(vol), sp.origin)
    assert compat.ndim(vol) == 3
    np.testing.assert_array_equal(
        compat.coord_to_grid(vol, np.array([[10, 20, 30]])), [[0, 0, 0]]
    )
    np.testing.assert_array_equal(compat.grid_to_index(vol, np.array([[1, 0, 0]])), [1])
    np.testing.assert_array_equal(compat.index_to_grid(vol, np.array([1])), [[1, 0, 0]])
    np.testing.assert_array_equal(compat.values(vol), vol.values())
    np.testing.assert_array_equal(compat.coords(vol)[:3], vol.coords()[:3])


def test_compat_vector_wrappers_delegate_to_methods():
    sp = ni.NeuroSpace((2, 2, 2, 3))
    data = np.arange(24, dtype=float).reshape((2, 2, 2, 3), order="F")
    vec = ni.DenseNeuroVec(data, sp)
    roi = ni.ROICoords(np.array([[0, 0, 0], [1, 0, 0]]), sp.drop_dim(3))

    np.testing.assert_array_equal(
        compat.series(vec, np.array([[0, 0, 0]])).ravel(),
        vec.series(np.array([[0, 0, 0]])).ravel(),
    )
    np.testing.assert_array_equal(compat.series_roi(vec, roi), vec.series_roi(roi).values)
    assert len(compat.vols(vec)) == 3
    assert compat.sub_vector(vec, [0, 2]).shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(
        compat.temporal_access(vec, [0, 2]),
        vec.data.reshape(-1, 3, order="F").T[[0, 2], :],
    )


def test_file_format_wrappers_delegate_to_descriptor_methods():
    assert compat.header_file_matches(ni.AFNI, "sample+orig.HEAD")
    assert compat.data_file_matches(ni.AFNI, "sample+orig.BRIK")
    assert compat.header_file(ni.AFNI, "sample+orig.BRIK") == "sample+orig.HEAD"
    assert compat.data_file(ni.AFNI, "sample+orig.HEAD") == "sample+orig.BRIK"
    assert compat.strip_extension(ni.AFNI, "sample+orig.HEAD") == "sample+orig"


def test_affine_orientation_utility_parity_helpers():
    aff = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pts = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    np.testing.assert_array_equal(
        ni.apply_affine(aff, pts), np.array([[10, 20, 30], [12, 26, 42]], dtype=float)
    )
    np.testing.assert_array_equal(ni.voxel_sizes(aff), [2, 3, 4])

    expanded = ni.append_diag(aff, [5], [7])
    assert expanded.shape == (5, 5)
    assert expanded[3, 3] == 5
    assert expanded[3, 4] == 7

    ornt = ni.axcodes_to_orientation("RAS")
    assert ni.orientation_to_axcodes(ornt) == ("R", "A", "S")
    np.testing.assert_array_equal(
        ni.orientation_transform(ornt, ornt),
        np.array([[0, 1], [1, 1], [2, 1]], dtype=float),
    )


def test_neurovecseq_alias_matches_factory():
    sp = ni.NeuroSpace((2, 2, 2))
    vols = [ni.DenseNeuroVol(np.full((2, 2, 2), i), sp) for i in range(2)]
    vec = compat.NeuroVecSeq(vols)
    assert isinstance(vec, ni.DenseNeuroVec)
    assert vec.shape == (2, 2, 2, 2)
    from neuroim.neuro_vec import neurovecseq

    vec2 = neurovecseq(vols)
    assert isinstance(vec2, ni.DenseNeuroVec)
    assert vec2.shape == vec.shape


def test_slice_to_volume_affine_matches_neuroim2_contract():
    aff = compat.slice_to_volume_affine(
        index=3, axis=3, shape=(10, 8, 6), index_base="R"
    )
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(aff, expected)
    np.testing.assert_array_equal(compat.slice2volume(3, 3, shape=(10, 8, 6)), expected)


def test_nifti_extension_parsing_helpers():
    ext = ni.NiftiExtension(6, b"hello\x00")
    assert ni.ecode_name(6) == "comment"
    assert ni.ecode_name(999) == "unknown"
    assert ni.parse_extension(ext) == "hello"


def test_searchlight_shape_callbacks_return_coordinate_matrices():
    mask = ni.LogicalNeuroVol(np.ones((5, 5, 5), dtype=bool), ni.NeuroSpace((5, 5, 5)))
    center = np.array([2, 2, 2])

    ellipsoid = ni.ellipsoid_shape(scales=(1, 1, 1))
    cube = ni.cube_shape()
    blobby = ni.blobby_shape(drop=0.0)

    for shape_fun in [ellipsoid, cube, blobby]:
        coords = shape_fun(mask, center, radius=1.5, iter=1, nonzero=False)
        assert isinstance(coords, np.ndarray)
        assert coords.shape[1] == 3
        assert np.any(np.all(coords == center, axis=1))


def test_mask_clip_automask_and_mmap_helpers(tmp_path):
    sp = ni.NeuroSpace((8, 8, 8, 2))
    data = np.ones((8, 8, 8, 2), dtype=float)
    data[:2, :, :, :] = 10
    vec = ni.DenseNeuroVec(data, sp)
    mask = ni.LogicalNeuroVol(np.ones((8, 8, 8), dtype=bool), sp.drop_dim(3))
    mask.data[0, :, :] = False

    masked = compat.apply_mask(vec, mask)
    assert isinstance(masked, ni.DenseNeuroVec)
    assert np.all(masked.data[0, :, :, :] == 0)

    level = compat.clip_level(vec)
    assert isinstance(level, float)
    auto = compat.automask(vec)
    assert isinstance(auto, ni.LogicalNeuroVol)
    assert auto.shape == (8, 8, 8)

    mmap_file = tmp_path / "vec.dat"
    mapped = compat.as_mmap(vec, file=mmap_file)
    assert isinstance(mapped, ni.BigNeuroVec)
    assert mapped.shape == vec.shape


def test_output_aligned_space_and_plot_style_aliases():
    aff = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    out = compat.output_aligned_space(
        {"shape": (4, 5, 6), "affine": aff}, voxel_sizes=(2, 2, 2)
    )
    assert isinstance(out, ni.NeuroSpace)
    np.testing.assert_array_equal(out.dim, [4, 5, 6])
    np.testing.assert_array_equal(out.origin, [10, 20, 30])

    assert compat.scale_fill_neuro("gray")["cmap"] == "gray"
    assert compat.theme_neuro(panel="clean")["panel"] == "clean"
    assert compat.annotate_orientation("axial")["args"] == ("axial",)


def test_literal_neuroim2_export_aliases_live_in_compat_namespace():
    expected = [
        "as_array",
        "as_dense",
        "as_mask",
        "as_matrix",
        "as_sparse",
        "bilateral_filter_4d",
        "createNIfTIHeader",
        "findAnatomy3D",
        "mapToColors",
        "matrixToQuatern",
        "quaternToMatrix",
        "read_hyper_vec",
        "AbstractSparseNeuroVec",
        "ArrayLike3D",
        "ArrayLike4D",
        "ArrayLike5D",
        "NeuroObj",
        "numericOrMatrix",
    ]
    missing = [name for name in expected if not hasattr(compat, name)]
    assert missing == []

    assert compat.as_array is not None
    assert compat.as_dense is not None
    assert compat.createNIfTIHeader is ni.create_nifti_header
    assert compat.findAnatomy3D is ni.find_anatomy_3d
    assert compat.mapToColors is ni.map_to_colors
    assert compat.matrixToQuatern is ni.matrix_to_quatern
    assert compat.quaternToMatrix is ni.quatern_to_matrix
    assert compat.read_hyper_vec is ni.read_neurohypervec


def test_dotted_r_aliases_are_not_top_level_attributes():
    for name in ["as.array", "as.dense", "as.mask", "as.matrix", "as.sparse", "None"]:
        assert not hasattr(ni, name)


def test_neuroim2_virtual_classes_are_structural_protocols():
    sp = ni.NeuroSpace((3, 4, 5))
    vol = ni.DenseNeuroVol(np.zeros((3, 4, 5)), sp)
    vec = ni.DenseNeuroVec(np.zeros((3, 4, 5, 2)), sp.add_dim(size=2))

    assert isinstance(vol, compat.NeuroObj)
    assert isinstance(vol, compat.ArrayLike3D)
    assert isinstance(vec, compat.NeuroObj)
    assert isinstance(vec, compat.ArrayLike4D)


def test_automask_keeps_largest_component_and_fills_internal_holes():
    sp = ni.NeuroSpace((10, 10, 10))
    arr = np.zeros((10, 10, 10), dtype=float)
    arr[1:8, 1:8, 1:8] = 100.0
    arr[3, 3, 3] = 0.0
    arr[9, 9, 9] = 100.0
    vol = ni.DenseNeuroVol(arr, sp)

    mask = compat.automask(vol, gradual=False, peels=0, connect="26-connect")

    assert mask.data[3, 3, 3]
    assert not mask.data[9, 9, 9]
    assert int(mask.data.sum()) == 343


def test_clip_level_gradual_returns_volume_thresholds():
    sp = ni.NeuroSpace((8, 8, 8))
    arr = np.linspace(1.0, 1000.0, 512).reshape((8, 8, 8), order="F")
    vol = ni.DenseNeuroVol(arr, sp)

    gradual = compat.clip_level(vol, gradual=True)

    assert isinstance(gradual, ni.DenseNeuroVol)
    assert gradual.shape == vol.shape
    assert np.all(np.isfinite(gradual.data))
    assert np.any(gradual.data > 0)


def test_output_aligned_space_accepts_scalar_voxel_size_and_deoblique_space():
    aff = np.array(
        [
            [2.0, 0.5, 0.0, 10.0],
            [0.0, 3.0, 0.25, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    sp = ni.NeuroSpace((4, 5, 6), trans=aff)

    out = compat.output_aligned_space(sp, voxel_sizes=2.0)
    deob = compat.deoblique(sp)

    assert isinstance(out, ni.NeuroSpace)
    assert isinstance(deob, ni.NeuroSpace)
    np.testing.assert_array_equal(out.spacing, [2.0, 2.0, 2.0])
    assert np.all(np.diag(deob.trans)[:3] > 0)


def test_bilateral_filter_4d_resolves_to_temporal_filter():
    assert ni.bilateral_filter_4d is not ni.bilateral_filter_vec


def test_bilateral_filter_4d_single_mask_voxel_with_zero_temporal_window_is_identity():
    sp = ni.NeuroSpace((3, 3, 3, 2))
    data = np.arange(54, dtype=float).reshape((3, 3, 3, 2), order="F")
    vec = ni.DenseNeuroVec(data, sp)
    mask = ni.LogicalNeuroVol(np.zeros((3, 3, 3), dtype=bool), sp.drop_dim(3))
    mask.data[1, 1, 1] = True

    out = ni.bilateral_filter_4d(
        vec,
        mask=mask,
        spatial_window=1,
        temporal_window=0,
        spatial_sigma=1,
        intensity_sigma=1,
        temporal_sigma=1,
    )

    assert isinstance(out, ni.DenseNeuroVec)
    np.testing.assert_allclose(out.data, data)


def test_bilateral_filter_4d_preserves_constant_arrays_without_nans():
    sp = ni.NeuroSpace((3, 3, 3, 4))
    data = np.full((3, 3, 3, 4), 5.0)
    vec = ni.DenseNeuroVec(data, sp)

    out = ni.bilateral_filter_4d(
        vec,
        spatial_window=1,
        temporal_window=1,
        spatial_sigma=1,
        intensity_sigma=1,
        temporal_sigma=1,
    )

    assert np.all(np.isfinite(out.data))
    np.testing.assert_allclose(out.data, data)


def test_bilateral_filter_4d_uses_temporal_neighbors_and_preserves_mask_exterior():
    sp = ni.NeuroSpace((1, 1, 1, 3))
    data = np.array([[[[0.0, 10.0, 0.0]]]])
    vec = ni.DenseNeuroVec(data, sp)

    out = ni.bilateral_filter_4d(
        vec,
        spatial_window=1,
        temporal_window=1,
        spatial_sigma=1,
        intensity_sigma=100,
        temporal_sigma=1,
        range_scale=1,
    )

    assert 0.0 < out.data[0, 0, 0, 0] < 10.0
    assert out.data[0, 0, 0, 1] < 10.0


def test_bilateral_filter_4d_rejects_invalid_temporal_sigma():
    sp = ni.NeuroSpace((3, 3, 3, 2))
    vec = ni.DenseNeuroVec(np.zeros((3, 3, 3, 2)), sp)

    with pytest.raises(ValueError):
        ni.bilateral_filter_4d(vec, temporal_sigma=0)

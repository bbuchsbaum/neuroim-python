"""Cross-language validation tests using rpy2 against R neuroim2.

These tests are optional and are skipped unless:
1. `rpy2` is installed, and
2. R package `neuroim2` is available in the active R library.
"""

from pathlib import Path

import numpy as np
import pytest

import nibabel as nib

import neuroim as pn

try:
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import conversion as ro_conversion
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except Exception:
    HAS_RPY2 = False
    ro = None
    numpy2ri = None
    ro_conversion = None
    localconverter = None

HAS_NEUROIM2 = False
if HAS_RPY2:
    HAS_NEUROIM2 = bool(ro.r('as.logical(requireNamespace("neuroim2", quietly=TRUE))')[0])

pytestmark = pytest.mark.skipif(
    not (HAS_RPY2 and HAS_NEUROIM2),
    reason="requires rpy2 and R package 'neuroim2'",
)


def _r_read_vol_array(file_path: str, index_1based: int = 1) -> np.ndarray:
    ro.globalenv["file_path"] = file_path
    ro.globalenv["index_1based"] = int(index_1based)
    with localconverter(ro.default_converter + numpy2ri.converter):
        arr = ro_conversion.get_conversion().rpy2py(
            ro.r("as.array(neuroim2::read_vol(file_path, index=index_1based))")
        )
    return np.asarray(arr)


def _r_read_vec_array(file_path: str, indices_1based: np.ndarray) -> np.ndarray:
    ro.globalenv["file_path"] = file_path
    ro.globalenv["indices_1based"] = ro.IntVector(indices_1based.astype(int).tolist())
    with localconverter(ro.default_converter + numpy2ri.converter):
        arr = ro_conversion.get_conversion().rpy2py(
            ro.r("as.array(neuroim2::read_vec(file_path, indices=indices_1based))")
        )
    return np.asarray(arr)


def _r_read_header_vectors(file_path: str):
    ro.globalenv["file_path"] = file_path
    with localconverter(ro.default_converter + numpy2ri.converter):
        dim = ro_conversion.get_conversion().rpy2py(
            ro.r("as.integer(dim(neuroim2::read_header(file_path)))")
        )
        spacing = ro_conversion.get_conversion().rpy2py(
            ro.r(
                "local({"
                "  h <- neuroim2::read_header(file_path);"
                "  tryCatch(as.numeric(neuroim2::spacing(h)),"
                "           error=function(e) as.numeric(methods::slot(h, 'spacing')))"
                "})"
            )
        )
        origin = ro_conversion.get_conversion().rpy2py(
            ro.r(
                "local({"
                "  h <- neuroim2::read_header(file_path);"
                "  tryCatch(as.numeric(neuroim2::origin(h)),"
                "           error=function(e) as.numeric(methods::slot(h, 'origin')))"
                "})"
            )
        )
    return np.asarray(dim, dtype=int), np.asarray(spacing, dtype=float), np.asarray(origin, dtype=float)


def _r_bilateral_filter_4d_array(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    spatial_sigma: float,
    intensity_sigma: float,
    temporal_sigma: float,
    spatial_window: int,
    temporal_window: int,
    temporal_spacing: float,
    range_scale: float,
) -> np.ndarray:
    ro.globalenv["bf4d_vals"] = ro.FloatVector(np.asarray(data).ravel(order="F"))
    ro.globalenv["bf4d_dims"] = ro.IntVector([int(x) for x in data.shape])
    ro.globalenv["bf4d_mask_vals"] = ro.BoolVector(np.asarray(mask, dtype=bool).ravel(order="F"))
    ro.globalenv["bf4d_spatial_sigma"] = float(spatial_sigma)
    ro.globalenv["bf4d_intensity_sigma"] = float(intensity_sigma)
    ro.globalenv["bf4d_temporal_sigma"] = float(temporal_sigma)
    ro.globalenv["bf4d_spatial_window"] = int(spatial_window)
    ro.globalenv["bf4d_temporal_window"] = int(temporal_window)
    ro.globalenv["bf4d_temporal_spacing"] = float(temporal_spacing)
    ro.globalenv["bf4d_range_scale"] = float(range_scale)
    with localconverter(ro.default_converter + numpy2ri.converter):
        arr = ro_conversion.get_conversion().rpy2py(
            ro.r(
                "local({"
                "  arr <- array(bf4d_vals, dim=bf4d_dims);"
                "  sp <- neuroim2::NeuroSpace(bf4d_dims);"
                "  vec <- neuroim2::DenseNeuroVec(arr, sp);"
                "  mask <- neuroim2::LogicalNeuroVol("
                "    array(bf4d_mask_vals, dim=bf4d_dims[1:3]),"
                "    neuroim2::NeuroSpace(bf4d_dims[1:3])"
                "  );"
                "  as.array(neuroim2::bilateral_filter_4d("
                "    vec, mask,"
                "    spatial_sigma=bf4d_spatial_sigma,"
                "    intensity_sigma=bf4d_intensity_sigma,"
                "    temporal_sigma=bf4d_temporal_sigma,"
                "    spatial_window=bf4d_spatial_window,"
                "    temporal_window=bf4d_temporal_window,"
                "    temporal_spacing=bf4d_temporal_spacing,"
                "    range_scale=bf4d_range_scale"
                "  ))"
                "})"
            )
        )
    return np.asarray(arr)


def _r_clip_level_outputs(data: np.ndarray):
    ro.globalenv["clip_vals"] = ro.FloatVector(np.asarray(data).ravel(order="F"))
    ro.globalenv["clip_dims"] = ro.IntVector([int(x) for x in data.shape])
    ro.r(
        "local({"
        "  arr <- array(clip_vals, dim=clip_dims);"
        "  sp <- neuroim2::NeuroSpace(clip_dims);"
        "  vol <- neuroim2::DenseNeuroVol(arr, sp);"
        "  clip_scalar <<- neuroim2::clip_level(vol);"
        "  clip_gradual <<- as.array(neuroim2::clip_level(vol, gradual=TRUE));"
        "  clip_mask <<- as.array(neuroim2::automask(vol, gradual=FALSE, peels=0L));"
        "})"
    )
    with localconverter(ro.default_converter + numpy2ri.converter):
        scalar = float(ro.r("clip_scalar")[0])
        gradual = ro_conversion.get_conversion().rpy2py(ro.r("clip_gradual"))
        mask = ro_conversion.get_conversion().rpy2py(ro.r("clip_mask"))
    return scalar, np.asarray(gradual), np.asarray(mask, dtype=bool)


def _r_output_aligned_space(shape, affine, voxel_sizes):
    ro.globalenv["oas_shape"] = ro.IntVector([int(x) for x in shape])
    ro.globalenv["oas_affine"] = ro.r.matrix(
        ro.FloatVector(np.asarray(affine).ravel(order="F")),
        nrow=4,
        ncol=4,
    )
    ro.globalenv["oas_voxel_sizes"] = ro.FloatVector(np.atleast_1d(voxel_sizes).astype(float))
    ro.r(
        "local({"
        "  o <- neuroim2::output_aligned_space("
        "    list(shape=oas_shape, affine=oas_affine),"
        "    voxel_sizes=oas_voxel_sizes"
        "  );"
        "  oas_out_shape <<- o$shape;"
        "  oas_out_affine <<- o$affine;"
        "})"
    )
    with localconverter(ro.default_converter + numpy2ri.converter):
        out_shape = ro_conversion.get_conversion().rpy2py(ro.r("oas_out_shape"))
        out_affine = ro_conversion.get_conversion().rpy2py(ro.r("oas_out_affine"))
    return np.asarray(out_shape, dtype=int), np.asarray(out_affine, dtype=float)


def _r_deoblique_space(shape, affine):
    ro.globalenv["deob_shape"] = ro.IntVector([int(x) for x in shape])
    ro.globalenv["deob_affine"] = ro.r.matrix(
        ro.FloatVector(np.asarray(affine).ravel(order="F")),
        nrow=4,
        ncol=4,
    )
    ro.r(
        "local({"
        "  sp <- neuroim2::NeuroSpace(deob_shape, trans=deob_affine);"
        "  d <- neuroim2::deoblique(sp);"
        "  deob_dim <<- dim(d);"
        "  deob_spacing <<- neuroim2::spacing(d);"
        "  deob_origin <<- neuroim2::origin(d);"
        "  deob_trans <<- neuroim2::trans(d);"
        "})"
    )
    with localconverter(ro.default_converter + numpy2ri.converter):
        dim = ro_conversion.get_conversion().rpy2py(ro.r("deob_dim"))
        spacing = ro_conversion.get_conversion().rpy2py(ro.r("deob_spacing"))
        origin = ro_conversion.get_conversion().rpy2py(ro.r("deob_origin"))
        trans = ro_conversion.get_conversion().rpy2py(ro.r("deob_trans"))
    return (
        np.asarray(dim, dtype=int),
        np.asarray(spacing, dtype=float),
        np.asarray(origin, dtype=float),
        np.asarray(trans, dtype=float),
    )


def _write_afni_pair(
    out_dir: Path,
    stem: str,
    data: np.ndarray,
    *,
    float_facs=None,
    brick_type_code: int = 3,
) -> Path:
    if data.ndim == 3:
        dims = data.shape
        nvols = 1
    else:
        dims = data.shape[:3]
        nvols = data.shape[3]

    if float_facs is None:
        float_facs = [1.0] * nvols

    brick_types = " ".join([str(brick_type_code)] * nvols)
    facs = " ".join(str(x) for x in float_facs)
    labels = "~" + "~".join([f"vol{i}" for i in range(nvols)]) + "~"

    head = out_dir / f"{stem}.HEAD"
    brik = out_dir / f"{stem}.BRIK"

    # R neuroim2's AFNI parser expects each block to start immediately after
    # a blank line and misses the very first block otherwise, so prepend
    # an initial blank line.
    head_txt = (
        "\n"
        "type = integer-attribute\n"
        "name = DATASET_DIMENSIONS\n"
        "count = 5\n"
        f"{dims[0]} {dims[1]} {dims[2]} 1 0\n\n"
        "type = integer-attribute\n"
        "name = DATASET_RANK\n"
        "count = 2\n"
        f"3 {nvols}\n\n"
        "type = float-attribute\n"
        "name = DELTA\n"
        "count = 3\n"
        "2.0 -2.0 3.0\n\n"
        "type = float-attribute\n"
        "name = IJK_TO_DICOM\n"
        "count = 12\n"
        "2.0 0.0 0.0 10.0 0.0 -2.0 0.0 20.0 0.0 0.0 3.0 30.0\n\n"
        "type = float-attribute\n"
        "name = ORIGIN\n"
        "count = 3\n"
        "10.0 20.0 30.0\n\n"
        "type = integer-attribute\n"
        "name = BRICK_TYPES\n"
        f"count = {nvols}\n"
        f"{brick_types}\n\n"
        "type = float-attribute\n"
        "name = BRICK_FLOAT_FACS\n"
        f"count = {nvols}\n"
        f"{facs}\n\n"
        "type = string-attribute\n"
        "name = BYTEORDER_STRING\n"
        "count = 9\n"
        "LSB_FIRST\n\n"
        "type = string-attribute\n"
        "name = BRICK_LABS\n"
        f"count = {len(labels)}\n"
        f"{labels}\n"
    )
    head.write_text(head_txt, encoding="utf-8")
    dtype_map = {
        0: np.uint8,
        1: np.int16,
        3: np.float32,
    }
    if brick_type_code not in dtype_map:
        raise ValueError(f"Unsupported brick_type_code: {brick_type_code}")
    np.asarray(data, dtype=dtype_map[brick_type_code]).ravel(order="F").tofile(brik)
    return head


def test_rpy_cross_nifti_read_vol(tmp_path):
    data = np.random.RandomState(42).randn(8, 7, 6).astype(np.float32)
    nii = tmp_path / "x.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(nii))

    py_vol = pn.read_vol(nii)
    r_arr = _r_read_vol_array(str(nii))
    np.testing.assert_allclose(py_vol.data, r_arr)


def test_rpy_cross_nifti_read_vec_indices(tmp_path):
    data4d = np.random.RandomState(7).randn(6, 5, 4, 4).astype(np.float32)
    nii = tmp_path / "vec.nii"
    nib.save(nib.Nifti1Image(data4d, np.eye(4)), str(nii))

    py_vec = pn.read_vec(nii, indices=[0, 2])
    r_arr = _r_read_vec_array(str(nii), np.array([1, 3], dtype=int))
    np.testing.assert_allclose(py_vec.data, r_arr)


def test_rpy_cross_afni_header_and_vol(tmp_path):
    base = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
    head = _write_afni_pair(tmp_path, "mini+orig", base, float_facs=[1.0, 2.0])

    py_hdr = pn.read_header(head)
    r_dim, r_spacing, r_origin = _r_read_header_vectors(str(head))

    py_dim = np.asarray(py_hdr["dim"], dtype=int)
    np.testing.assert_array_equal(py_dim[:3], r_dim[:3])
    if py_dim.size >= 4 and r_dim.size >= 5:
        assert py_dim[3] == r_dim[4]
    np.testing.assert_allclose(np.asarray(py_hdr["spacing"])[:3], r_spacing[:3])
    np.testing.assert_allclose(np.asarray(py_hdr["origin"])[:3], r_origin[:3])

    py_vol = pn.read_vol(head, index=0)
    r_arr = _r_read_vol_array(str(head), index_1based=1)
    np.testing.assert_allclose(py_vol.data, r_arr)


def test_rpy_cross_afni_scaling_first_brick(tmp_path):
    base = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
    scale = 2.5
    head = _write_afni_pair(tmp_path, "scaled+orig", base, float_facs=[scale])

    py_vol = pn.read_vol(head)
    r_arr = _r_read_vol_array(str(head), index_1based=1)

    np.testing.assert_allclose(py_vol.data, base * scale)
    np.testing.assert_allclose(py_vol.data, r_arr)


def test_rpy_cross_bilateral_filter_4d_temporal_semantics():
    data = np.arange(3 * 3 * 2 * 3, dtype=float).reshape((3, 3, 2, 3), order="F")
    data[1, 1, 1, 1] = 100.0
    mask_data = np.ones((3, 3, 2), dtype=bool)
    mask_data[0, 0, 0] = False
    sp = pn.NeuroSpace(data.shape)
    vec = pn.DenseNeuroVec(data, sp)
    mask = pn.LogicalNeuroVol(mask_data, sp.drop_dim(3))

    kwargs = dict(
        spatial_sigma=1.25,
        intensity_sigma=0.75,
        temporal_sigma=1.1,
        spatial_window=1,
        temporal_window=1,
        temporal_spacing=1.4,
        range_scale=25.0,
    )
    py_out = pn.bilateral_filter_4d(vec, mask, **kwargs)
    r_arr = _r_bilateral_filter_4d_array(data, mask_data, **kwargs)

    np.testing.assert_allclose(py_out.data, r_arr, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(py_out.data[0, 0, 0, :], data[0, 0, 0, :])


def test_rpy_cross_clip_level_and_automask_semantics():
    data = np.arange(1, 513, dtype=float).reshape((8, 8, 8), order="F")
    sp = pn.NeuroSpace(data.shape)
    vol = pn.DenseNeuroVol(data, sp)

    r_scalar, r_gradual, r_mask = _r_clip_level_outputs(data)

    assert pn.compat.clip_level(vol) == r_scalar
    np.testing.assert_allclose(pn.compat.clip_level(vol, gradual=True).data, r_gradual)
    np.testing.assert_array_equal(
        pn.compat.automask(vol, gradual=False, peels=0).data,
        r_mask,
    )


def test_rpy_cross_output_aligned_space_and_deoblique_space():
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    shape = (4, 5, 6)

    r_shape, r_affine = _r_output_aligned_space(shape, affine, voxel_sizes=2.0)
    py_space = pn.compat.output_aligned_space(
        {"shape": shape, "affine": affine}, voxel_sizes=2.0
    )
    np.testing.assert_array_equal(py_space.dim, r_shape)
    np.testing.assert_allclose(py_space.trans, r_affine)

    r_dim, r_spacing, r_origin, r_trans = _r_deoblique_space(shape, affine)
    py_deob = pn.compat.deoblique(pn.NeuroSpace(shape, trans=affine))
    np.testing.assert_array_equal(py_deob.dim, r_dim)
    np.testing.assert_allclose(py_deob.spacing, r_spacing)
    np.testing.assert_allclose(py_deob.origin, r_origin)
    np.testing.assert_allclose(py_deob.trans, r_trans)


@pytest.mark.parametrize(
    ("brick_type_code", "storage_dtype"),
    [(0, np.uint8), (1, np.int16)],
)
def test_rpy_cross_afni_integer_brick_types(tmp_path, brick_type_code, storage_dtype):
    base = np.arange(4 * 3 * 2, dtype=np.float64).reshape((4, 3, 2), order="F")
    scale = 2.0
    head = _write_afni_pair(
        tmp_path,
        f"int_type_{brick_type_code}+orig",
        base,
        float_facs=[scale],
        brick_type_code=brick_type_code,
    )

    py_vol = pn.read_vol(head)
    r_arr = _r_read_vol_array(str(head), index_1based=1)
    expected = np.asarray(base, dtype=storage_dtype).astype(np.float64) * scale

    np.testing.assert_allclose(py_vol.data, expected)
    np.testing.assert_allclose(py_vol.data, r_arr)

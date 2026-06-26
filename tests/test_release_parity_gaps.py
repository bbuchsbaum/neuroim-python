"""Release-hardening tests ported from high-value neuroim2 parity gaps."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neuroim as ni
from neuroim.orientation import rescale_affine


def test_downsample_accepts_neuroim2_spacing_for_4d_and_updates_affine():
    data = np.arange(4 * 6 * 8 * 2, dtype=float).reshape(4, 6, 8, 2)
    affine = np.array(
        [
            [0, 2, 0, -20],
            [-2, 0, 0, 10],
            [0, 0, 2, 30],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    vec = ni.DenseNeuroVec(
        data,
        ni.NeuroSpace((4, 6, 8, 2), spacing=(2, 2, 2, 1), trans=affine),
    )

    out = ni.downsample(vec, spacing=(4, 4, 4))

    assert isinstance(out, ni.DenseNeuroVec)
    assert out.shape == (2, 3, 4, 2)
    np.testing.assert_allclose(out.spacing[:3], (4, 4, 4))
    expected = rescale_affine(affine, (4, 6, 8), (4, 4, 4), (2, 3, 4))
    np.testing.assert_allclose(out.space.trans[:3, :3], expected[:3, :3])
    np.testing.assert_allclose(out.space.trans[:3, -1], expected[:3, 3])
    np.testing.assert_allclose(
        out.data[0, 0, 0, :],
        data[:2, :2, :2, :].mean(axis=(0, 1, 2)),
    )


def test_downsample_accepts_ratio_factor_outdim_warning_and_method_validation():
    vol = ni.DenseNeuroVol(
        np.ones((8, 8, 4), dtype=float),
        ni.NeuroSpace((8, 8, 4), spacing=(1, 1, 2)),
    )

    ratio = ni.downsample(vol, factor=(0.5, 0.5, 0.5))
    assert ratio.shape == (4, 4, 2)
    np.testing.assert_allclose(ratio.spacing, (2, 2, 4))

    with pytest.warns(UserWarning, match="aspect ratio"):
        ni.downsample(vol, outdim=(4, 4, 1))

    with pytest.raises(ValueError, match="Only 'box'"):
        ni.downsample(vol, factor=0.5, method="lanczos")

    with pytest.raises(ValueError, match="Exactly one"):
        ni.downsample(vol)


def test_resample_clustered_volume_preserves_labels_and_forces_nearest_neighbor():
    sp = ni.NeuroSpace((4, 4, 4))
    mask_data = np.zeros((4, 4, 4), dtype=bool)
    mask_data[:2, :2, :2] = True
    mask_data[2:, 2:, 2:] = True
    mask = ni.LogicalNeuroVol(mask_data, sp)
    clusters = np.repeat([1, 2], mask_data.sum() // 2)
    cvol = ni.ClusteredNeuroVol(mask, clusters, label_map={"regionA": 1, "regionB": 2})
    target = ni.DenseNeuroVol(np.zeros((4, 4, 4), dtype=float), sp)

    with pytest.warns(UserWarning, match="nearest-neighbor"):
        out = ni.resample(cvol, target, interpolation=3)

    assert isinstance(out, ni.ClusteredNeuroVol)
    assert out.space == sp
    assert set(np.unique(out.clusters)) == {1, 2}
    assert out.label_map == cvol.label_map


def test_registration_qc_plots_validate_grids_and_return_slice_panels():
    sp = ni.NeuroSpace((5, 6, 4))
    bg = ni.DenseNeuroVol(np.arange(5 * 6 * 4).reshape(5, 6, 4), sp)
    ov = ni.DenseNeuroVol(np.arange(5 * 6 * 4, 0, -1).reshape(5, 6, 4), sp)
    edges = ni.DenseNeuroVol(np.zeros((5, 6, 4)), sp)

    fig, axes = ni.plot_checkerboard(
        bg,
        ov,
        zlevels=[1, 3],
        tile=2,
        ncol=2,
        title="Checker",
        draw=False,
    )
    try:
        assert len(axes) == 2
        assert axes[0].get_title() == "z = 1"
        assert fig._suptitle.get_text() == "Checker"
    finally:
        plt.close(fig)

    fig, axes = ni.plot_edge_overlay(bg, edges, edges, zlevels=[1], ncol=1, draw=False)
    try:
        edge_layer = axes[0].images[1].get_array()
        assert edge_layer.shape[-1] == 4
        assert np.allclose(edge_layer[..., 3], 0)
    finally:
        plt.close(fig)

    shifted = ni.DenseNeuroVol(
        np.zeros((5, 6, 4)),
        ni.NeuroSpace((5, 6, 4), spacing=(2, 2, 2)),
    )
    with pytest.raises(ValueError, match="same NeuroSpace grid"):
        ni.plot_checkerboard(bg, shifted, zlevels=[1], draw=False)
    with pytest.raises(ValueError, match="`zlevels`"):
        ni.plot_checkerboard(bg, ov, zlevels=[], draw=False)
    with pytest.raises(ValueError, match="`ncol`"):
        ni.plot_edge_overlay(bg, edges, edges, zlevels=[1], ncol=0, draw=False)


def test_read_header_exposes_neuroim2_style_fields(tmp_path):
    nib = pytest.importorskip("nibabel")

    data = np.zeros((3, 4, 5, 2), dtype=np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = 1.5
    path = tmp_path / "header.nii"
    nib.save(img, path)

    header = ni.io.read_header(path)

    required = {
        "dim",
        "pixdim",
        "spacing",
        "origin",
        "trans",
        "qform",
        "sform",
        "intent_code",
        "intent_name",
        "descrip",
        "data_type",
        "bitpix",
        "scl_slope",
        "scl_inter",
        "cal_min",
        "cal_max",
        "TR",
        "raw",
    }
    assert required.issubset(header)
    assert header["dim"] == (3, 4, 5, 2)
    np.testing.assert_allclose(header["spacing"][:3], (2, 3, 4))
    assert header["qform"]["matrix"].shape == (4, 4)
    assert header["sform"]["matrix"].shape == (4, 4)
    assert header["TR"] == pytest.approx(1.5)


def test_hypervec_hdf5_roundtrip_via_r_compatible_alias(tmp_path):
    data = np.arange(2 * 3 * 4 * 2 * 3, dtype=float).reshape(2, 3, 4, 2, 3)
    hvec = ni.DenseNeuroHyperVec(
        data,
        ni.NeuroSpace((2, 3, 4, 2, 3), spacing=(1, 1, 2, 1, 1)),
        label="features",
    )
    path = tmp_path / "hypervec.h5"

    ni.write_neurohypervec(hvec, path)
    out = ni.compat.read_hyper_vec(path)

    assert isinstance(out, ni.DenseNeuroHyperVec)
    assert out.shape == hvec.shape
    assert out.label == "features"
    np.testing.assert_allclose(out.spacing, hvec.spacing)
    np.testing.assert_array_equal(out.data, data)

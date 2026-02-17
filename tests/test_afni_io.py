"""Tests for AFNI HEAD/BRIK support in io.py and file_format.py."""

import gzip
from pathlib import Path

import numpy as np

from neuroimpy import read_meta_info, read_header, read_vol, read_vec, write_vol, write_vec
from neuroimpy import DenseNeuroVol, DenseNeuroVec, NeuroSpace
from neuroimpy.meta_info import AFNIMetaInfo


def _write_afni_pair(
    out_dir: Path,
    stem: str,
    data: np.ndarray,
    *,
    float_facs=None,
    gzip_data: bool = False,
) -> Path:
    if data.ndim == 3:
        dims = data.shape
        nvols = 1
    else:
        dims = data.shape[:3]
        nvols = data.shape[3]

    if float_facs is None:
        float_facs = [1.0] * nvols

    brick_types = " ".join(["3"] * nvols)
    facs = " ".join(str(x) for x in float_facs)
    labels = "~" + "~".join([f"vol{i}" for i in range(nvols)]) + "~"

    head = out_dir / f"{stem}.HEAD"
    brik = out_dir / f"{stem}.BRIK"
    if gzip_data:
        brik = out_dir / f"{stem}.BRIK.gz"

    head_txt = (
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
        "count = 10\n"
        "~LSB_FIRST~\n\n"
        "type = string-attribute\n"
        "name = BRICK_LABS\n"
        f"count = {len(labels)}\n"
        f"{labels}\n"
    )

    head.write_text(head_txt, encoding="utf-8")
    raw = np.asarray(data, dtype=np.float32).ravel(order="F").tobytes()
    if gzip_data:
        with gzip.open(brik, "wb") as f:
            f.write(raw)
    else:
        brik.write_bytes(raw)
    return head


def test_read_meta_info_afni(tmp_path):
    data = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
    head = _write_afni_pair(tmp_path, "mini+orig", data, float_facs=[1.0, 2.0])

    meta = read_meta_info(head)
    assert isinstance(meta, AFNIMetaInfo)
    assert meta.dims == (4, 3, 2, 2)
    assert meta.spacing == (2.0, 2.0, 3.0)
    assert meta.origin == (10.0, 20.0, 30.0)
    np.testing.assert_array_equal(np.asarray(meta.slope), np.asarray([1.0, 2.0]))


def test_read_header_afni(tmp_path):
    data = np.ones((4, 3, 2), dtype=np.float32)
    head = _write_afni_pair(tmp_path, "header_only+orig", data)

    hdr = read_header(head)
    assert hdr["dim"] == (4, 3, 2)
    assert tuple(hdr["spacing"]) == (2.0, 2.0, 3.0)
    assert tuple(hdr["origin"]) == (10.0, 20.0, 30.0)
    assert "afni_header" in hdr


def test_read_vol_afni_4d_index_and_scaling(tmp_path):
    base = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
    head = _write_afni_pair(tmp_path, "vol+orig", base, float_facs=[1.0, 2.0])

    vol0 = read_vol(head, index=0)
    vol1 = read_vol(head, index=1)

    np.testing.assert_allclose(vol0.data, base[..., 0] * 1.0)
    np.testing.assert_allclose(vol1.data, base[..., 1] * 2.0)


def test_read_vec_afni_indices_and_gzip(tmp_path):
    base = np.arange(4 * 3 * 2 * 3, dtype=np.float32).reshape((4, 3, 2, 3), order="F")
    head = _write_afni_pair(tmp_path, "vec+orig", base, float_facs=[1.0, 2.0, 3.0], gzip_data=True)

    vec = read_vec(head, indices=[0, 2])
    expected = np.stack([base[..., 0] * 1.0, base[..., 2] * 3.0], axis=3)
    np.testing.assert_allclose(vec.data, expected)
    assert vec.shape == expected.shape


def test_write_read_vol_afni_roundtrip(tmp_path):
    data = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2), order="F")
    space = NeuroSpace(dim=data.shape, spacing=(2.0, 2.0, 3.0), origin=(10.0, 20.0, 30.0))
    vol = DenseNeuroVol(data, space)
    head = tmp_path / "wr_vol+orig.HEAD"

    write_vol(vol, head, format="AFNI")
    loaded = read_vol(head)
    hdr = read_header(head)

    np.testing.assert_allclose(loaded.data, data)
    assert tuple(hdr["dim"][:3]) == data.shape
    np.testing.assert_allclose(np.asarray(hdr["spacing"])[:3], np.asarray([2.0, 2.0, 3.0]))


def test_write_read_vol_afni_defaults_to_compatible_dtype(tmp_path):
    data = np.arange(4 * 3 * 2, dtype=np.int16).reshape((4, 3, 2), order="F")
    space = NeuroSpace(dim=data.shape, spacing=(2.0, 2.0, 3.0), origin=(10.0, 20.0, 30.0))
    vol = DenseNeuroVol(data, space)
    head = tmp_path / "wr_vol_default_afni+orig.HEAD"

    write_vol(vol, head, format="AFNI")
    loaded = read_vol(head)
    hdr = read_header(head)

    assert loaded.shape == data.shape
    np.testing.assert_allclose(loaded.data, data.astype(np.float32))
    assert tuple(hdr["dim"][:3]) == data.shape


def test_write_read_vec_afni_gz_roundtrip(tmp_path):
    data = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape((4, 3, 2, 2), order="F")
    space = NeuroSpace(dim=data.shape, spacing=(2.0, 2.0, 3.0, 1.0), origin=(10.0, 20.0, 30.0, 0.0))
    vec = DenseNeuroVec(data, space)
    head = tmp_path / "wr_vec+orig.HEAD"

    write_vec(vec, head, format="AFNI_GZ")
    loaded = read_vec(head)
    vol1 = read_vol(head, index=1)

    np.testing.assert_allclose(loaded.data, data)
    np.testing.assert_allclose(vol1.data, data[..., 1])

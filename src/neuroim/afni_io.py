"""AFNI header parsing helpers.

This module provides a lightweight AFNI ``.HEAD`` parser compatible with the
attribute block format used by AFNI BRIK/HEAD datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Dict, List, Union, Any
import gzip
import numpy as np


def _parse_int_attribute(lines: List[str]) -> List[int]:
    toks = " ".join(lines).strip().split()
    return [int(tok) for tok in toks]


def _parse_float_attribute(lines: List[str]) -> List[float]:
    toks = " ".join(lines).strip().split()
    return [float(tok) for tok in toks]


def _parse_string_attribute(lines: List[str]) -> List[str]:
    # AFNI writes string attributes as 'val1~val2~ (a leading single quote,
    # then ~-separated values); older/other writers use ~val1~val2~ or a bare
    # token (e.g. MSB_FIRST). Drop the quote so 'MSB_FIRST is recognised.
    text = "\n".join(lines).strip()
    if not text:
        return []

    if text.startswith("'"):
        text = text[1:]
    if text.startswith("~"):
        text = text[1:]
    if text.endswith("~"):
        text = text[:-1]

    values = [part for part in text.split("~") if part != ""]
    if values:
        return values

    return [text]


def _parse_element(block: List[str]) -> Dict[str, Any]:
    if len(block) < 4:
        raise ValueError("Invalid AFNI attribute block: expected at least 4 lines")

    atype = block[0].split("=", 1)[1].strip()
    name = block[1].split("=", 1)[1].strip()
    count = int(block[2].split("=", 1)[1].strip())
    content_lines = block[3:]

    if atype == "integer-attribute":
        content = _parse_int_attribute(content_lines)
    elif atype == "float-attribute":
        content = _parse_float_attribute(content_lines)
    elif atype == "string-attribute":
        content = _parse_string_attribute(content_lines)
    else:
        raise ValueError(f"Unrecognized AFNI attribute type: {atype}")

    return {"type": atype, "name": name, "count": count, "content": content}


def read_afni_header(file_name: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Read and parse an AFNI ``.HEAD`` file.

    Parameters
    ----------
    file_name : str or Path
        Path to an AFNI header file.

    Returns
    -------
    dict
        Mapping from attribute name to parsed attribute dictionary containing
        ``type``, ``name``, ``count``, and ``content``.
    """
    pth = Path(file_name)
    lines = pth.read_text(encoding="utf-8").splitlines()

    header: Dict[str, Dict[str, Any]] = {}
    block: List[str] = []
    for line in lines:
        if line.strip() == "":
            if block:
                parsed = _parse_element(block)
                header[parsed["name"]] = parsed
                block = []
            continue
        block.append(line)

    if block:
        parsed = _parse_element(block)
        header[parsed["name"]] = parsed

    return header


def parse_niml_element(element: str) -> Dict[str, Any]:
    """Parse a NIML element header.

    This is the Python counterpart of neuroim2's internal
    ``parse_niml_element()`` helper.  It handles the compact AFNI-style tag
    representation used by sparse data and index-list NIML blocks.

    Parameters
    ----------
    element : str
        Header text without angle brackets.

    Returns
    -------
    dict
        A mapping with ``label`` and ``attr`` keys.
    """
    text = element.replace("\n", "").replace('"', "").replace("/", "").strip()
    parts = [part for part in text.split(" ") if part and part != ">"]
    if not parts:
        return {"label": "", "attr": None}

    attrs: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        attrs[key] = val

    return {"label": parts[0], "attr": attrs or None}


def _read_char(fconn: BinaryIO) -> str:
    ch = fconn.read(1)
    if ch == b"":
        return ""
    return ch.decode("latin1")


def parse_niml_header(fconn: BinaryIO) -> Dict[str, Any]:
    """Read and parse the next NIML header from a binary stream.

    Parameters
    ----------
    fconn : binary file object
        Open file-like object positioned before a NIML tag.

    Returns
    -------
    dict
        Parsed element with ``label`` and ``attr`` keys.
    """
    chars: List[str] = []
    state = "BEGIN"
    while True:
        ch = _read_char(fconn)
        if ch == "":
            break
        if ch == "<" and state == "BEGIN":
            state = "HEADER"
        elif ch == ">" and state == "HEADER":
            break
        elif state == "HEADER":
            chars.append(ch)

    return parse_niml_element("".join(chars))


def _niml_dtype(meta: Dict[str, str]) -> tuple[int, str]:
    dtype = str(meta.get("ni_type", "float"))
    parts = dtype.split("*", 1)
    if len(parts) == 2:
        nvols = int(parts[0])
        typ = parts[1]
    else:
        nvols = 1
        typ = parts[0]

    if typ not in {"int", "float", "double"}:
        raise ValueError(f"Unrecognized NIML ni_type: {typ}")
    return nvols, typ


def read_niml_data(fconn: BinaryIO, meta: Dict[str, str]) -> np.ndarray:
    """Read a NIML sparse-data payload from a binary stream.

    Parameters
    ----------
    fconn : binary file object
        Open file-like object positioned immediately after the opening tag.
    meta : dict
        Parsed NIML attributes including ``ni_type`` and ``ni_dimen``.

    Returns
    -------
    ndarray
        Matrix with shape ``(nvols, ni_dimen)`` matching neuroim2's
        column-major matrix construction.
    """
    nvols, typ = _niml_dtype(meta)
    nels = int(meta["ni_dimen"])
    nvals = nvols * nels
    form = meta.get("ni_form")

    if form == "binary.lsbfirst":
        # NIML doubles are 8 bytes (neuroim2's readBin(size = 4) misreads them)
        dtype = np.dtype({"int": "<i4", "float": "<f4", "double": "<f8"}[typ])
        values = np.frombuffer(fconn.read(nvals * dtype.itemsize), dtype=dtype, count=nvals)
        if values.size != nvals:
            raise ValueError(f"NIML data block is truncated: expected {nvals}, got {values.size}")
    elif form == "binary.msbfirst":
        dtype = np.dtype({"int": ">i4", "float": ">f4", "double": ">f8"}[typ])
        values = np.frombuffer(fconn.read(nvals * dtype.itemsize), dtype=dtype, count=nvals)
        if values.size != nvals:
            raise ValueError(f"NIML data block is truncated: expected {nvals}, got {values.size}")
    else:
        payload = bytearray()
        while True:
            ch = fconn.read(1)
            if ch == b"":
                break
            if ch == b"<":
                fconn.seek(-1, 1)
                break
            payload.extend(ch)
        tokens = payload.decode("utf-8", errors="replace").split()
        if len(tokens) < nvals:
            raise ValueError(f"NIML data block is truncated: expected {nvals}, got {len(tokens)}")
        if typ == "int":
            values = np.asarray([int(tok) for tok in tokens[:nvals]], dtype=np.int32)
        else:
            values = np.asarray([float(tok) for tok in tokens[:nvals]], dtype=np.float64)

    return np.asarray(values).reshape((nvols, nels), order="F")


def _consume_niml_close_tag(fconn: BinaryIO) -> None:
    while True:
        ch = _read_char(fconn)
        if ch == "":
            return
        if ch == "<":
            while True:
                ch = _read_char(fconn)
                if ch in {"", ">"}:
                    return


def parse_niml_next(fconn: BinaryIO) -> Dict[str, Any]:
    """Parse the next NIML element from a binary stream."""
    header = parse_niml_header(fconn)
    if header.get("attr") is not None and header.get("label") in {"SPARSE_DATA", "INDEX_LIST"}:
        header["data"] = read_niml_data(fconn, header["attr"])
    _consume_niml_close_tag(fconn)
    return header


def parse_niml_file(file_name: Union[str, Path], maxels: int = 10000) -> List[Dict[str, Any]]:
    """Parse a NIML file into a list of element dictionaries.

    This mirrors neuroim2's lightweight ``parse_niml_file()`` behavior.  It is
    intentionally a structural parser rather than a full AFNI object model.
    Sparse data and index-list elements receive a ``data`` matrix.
    """
    path = Path(file_name)
    out: List[Dict[str, Any]] = []
    with path.open("rb") as fconn:
        file_size = path.stat().st_size
        out.append(parse_niml_header(fconn))
        while fconn.tell() < file_size and len(out) < maxels:
            out.append(parse_niml_next(fconn))
    return out


# AFNI BRICK_TYPES codes -> numpy dtypes (byte order applied by the reader)
AFNI_BRICK_DTYPES = {0: "u1", 1: "i2", 2: "i4", 3: "f4", 4: "f8"}

# RAS (NIfTI/neuroim) <-> DICOM/RAI (AFNI): x and y change sign.
_RAS_TO_DICOM = np.diag([-1.0, -1.0, 1.0])


def _content(header: Dict[str, Any], name: str):
    entry = header.get(name)
    return None if entry is None else entry.get("content")


def afni_affine(header: Dict[str, Any]) -> np.ndarray:
    """4x4 RAS voxel-to-world affine of an AFNI header.

    AFNI geometry is in DICOM (RAI) coordinates; flipping x and y gives RAS,
    as nibabel's AFNI reader does. Uses ``IJK_TO_DICOM_REAL`` (obliquity),
    else ``IJK_TO_DICOM``, else ``ORIENT_SPECIFIC`` + ``DELTA`` + ``ORIGIN``.
    """
    for name in ("IJK_TO_DICOM_REAL", "IJK_TO_DICOM"):
        vals = _content(header, name)
        if vals is not None and len(vals) >= 12:
            mat = np.asarray(vals[:12], dtype=float).reshape(3, 4)
            if np.all(np.isfinite(mat)) and np.linalg.matrix_rank(mat[:, :3]) == 3:
                break
    else:
        orient = _content(header, "ORIENT_SPECIFIC")
        delta = _content(header, "DELTA")
        origin = _content(header, "ORIGIN")
        if orient is None or delta is None or origin is None:
            raise ValueError(
                "AFNI header has neither IJK_TO_DICOM(_REAL) nor "
                "ORIENT_SPECIFIC/DELTA/ORIGIN"
            )
        mat = np.zeros((3, 4))
        for i, code in enumerate(orient[:3]):
            axis = int(code) // 2  # 0,1: x (R/L)  2,3: y (A/P)  4,5: z (I/S)
            mat[axis, i] = float(delta[i])
            mat[axis, 3] = float(origin[i])
    affine = np.eye(4)
    affine[:3, :] = _RAS_TO_DICOM @ mat
    return affine


def afni_tr_seconds(header: Dict[str, Any]) -> float | None:
    """Time step from TAXIS_FLOATS (converted from ms when TAXIS_NUMS says so)."""
    floats = _content(header, "TAXIS_FLOATS")
    if floats is None or len(floats) < 2 or float(floats[1]) <= 0:
        return None
    tr = float(floats[1])
    nums = _content(header, "TAXIS_NUMS")
    if nums is not None and len(nums) >= 3 and int(nums[2]) == 77001:  # msec
        tr /= 1000.0
    return tr


# ORIENT_SPECIFIC code for (DICOM axis, step > 0). DICOM +x = Left,
# +y = Posterior, +z = Superior; codes 0 R2L, 1 L2R, 2 P2A, 3 A2P, 4 I2S, 5 S2I.
_ORIENT_CODE = {
    (0, True): 0, (0, False): 1,
    (1, True): 3, (1, False): 2,
    (2, True): 4, (2, False): 5,
}


def _afni_geometry(affine: np.ndarray):
    """AFNI geometry attributes for a RAS affine.

    Returns ``(IJK_TO_DICOM_REAL, IJK_TO_DICOM, ORIENT_SPECIFIC, DELTA,
    ORIGIN)``; the cardinal IJK_TO_DICOM/DELTA/ORIGIN approximate an oblique
    grid by its nearest axis-aligned one (voxel sizes preserved), while
    IJK_TO_DICOM_REAL keeps the exact mapping.
    """
    real = _RAS_TO_DICOM @ np.asarray(affine, dtype=float)[:3, :]
    card = np.zeros((3, 4))
    orient, delta, origin = [], [], []
    used = set()
    for i in range(3):
        axis = int(np.argmax(np.abs(real[:, i])))
        if axis in used:
            raise ValueError("affine does not map voxel axes to distinct world axes")
        used.add(axis)
        positive = bool(real[axis, i] > 0)
        size = float(np.linalg.norm(real[:, i])) * (1.0 if positive else -1.0)
        orient.append(_ORIENT_CODE[(axis, positive)])
        delta.append(size)
        origin.append(float(real[axis, 3]))
        card[axis, i] = size
        card[axis, 3] = real[axis, 3]
    return real, card, orient, delta, origin


def _afni_dtype_info(dtype: np.dtype) -> tuple[int, np.dtype]:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.uint8):
        return 0, dt
    if dt == np.dtype(np.int16):
        return 1, dt
    # Everything else is stored as float, the type every AFNI reader supports.
    return 3, np.dtype(np.float32)


def _afni_string(name: str, values: List[str]) -> str:
    """A string-attribute block in AFNI's own syntax: 'val1~val2~."""
    body = "'" + "~".join(values) + "~"
    # AFNI separates blocks with a blank line *before* each one.
    return (
        "\ntype = string-attribute\n"
        f"name = {name}\n"
        f"count = {len(body) - 1}\n"
        f"{body}\n"
    )


def _afni_numbers(kind: str, name: str, values) -> str:
    vals = list(values)
    fmt = (lambda v: str(int(v))) if kind == "integer" else (lambda v: repr(float(v)))
    return (
        f"\ntype = {kind}-attribute\n"
        f"name = {name}\n"
        f"count = {len(vals)}\n"
        f"{' '.join(fmt(v) for v in vals)}\n"
    )


def _quantize_bricks(arr: np.ndarray, dtype: np.dtype):
    """Integer storage with per-brick BRICK_FLOAT_FACS, never truncating.

    Bricks whose values are already integers in range are stored as-is
    (factor 0, meaning 1); others are scaled to the type's range.
    """
    info = np.iinfo(dtype)
    bricks = arr[..., None] if arr.ndim == 3 else arr
    out = np.empty(bricks.shape, dtype=dtype)
    facs = []
    for b in range(bricks.shape[3]):
        vals = np.asarray(bricks[..., b], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError("integer AFNI output requires finite data values")
        lo, hi = float(vals.min()), float(vals.max())
        if np.all(vals == np.round(vals)) and lo >= info.min and hi <= info.max:
            out[..., b] = vals.astype(dtype)
            facs.append(0.0)
            continue
        if lo < 0 and info.min == 0:
            raise ValueError(
                f"{dtype} AFNI output cannot hold negative values; use FLOAT32 or INT16"
            )
        peak = max(abs(lo), abs(hi))
        fac = peak / info.max if peak > 0 else 1.0
        out[..., b] = np.clip(np.rint(vals / fac), info.min, info.max).astype(dtype)
        facs.append(fac)
    return (out[..., 0] if arr.ndim == 3 else out), facs


def write_afni_pair(
    file_name: Union[str, Path],
    data: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    data_encoding: str = "raw",
    data_type: str | None = None,
    affine: np.ndarray | None = None,
    tr: float | None = None,
) -> tuple[str, str]:
    """Write AFNI HEAD/BRIK pair.

    Parameters
    ----------
    file_name : str or Path
        Output path hint. May be ``.HEAD``, ``.BRIK``, ``.BRIK.gz``, or stem.
    data : ndarray
        3D or 4D array to write.
    spacing, origin : tuple
        Axis-aligned RAS geometry, used only when ``affine`` is None.
    data_encoding : {"raw","gzip"}
        Data payload encoding. ``gzip`` writes ``.BRIK.gz``.
    data_type : str, optional
        Requested storage type: ``FLOAT32``, ``INT16`` or ``UINT8``. Integer
        storage of non-integer data uses per-brick scale factors
        (``BRICK_FLOAT_FACS``) instead of truncating.
    affine : ndarray, optional
        4x4 RAS voxel-to-world affine; written as AFNI DICOM geometry
        (``IJK_TO_DICOM_REAL`` exactly, plus the cardinal
        ``IJK_TO_DICOM``/``ORIENT_SPECIFIC``/``DELTA``/``ORIGIN``).
    tr : float, optional
        Time step in seconds for 4-D data (``TAXIS_*``).

    Returns
    -------
    (header_file, data_file) : tuple[str, str]
    """
    arr = np.asarray(data)
    if arr.ndim not in (3, 4):
        raise ValueError(f"AFNI write supports 3D/4D arrays, got {arr.ndim}D")

    out = str(file_name)
    if out.endswith(".HEAD"):
        stem = out[:-5]
    elif out.endswith(".BRIK.gz"):
        stem = out[:-8]
    elif out.endswith(".BRIK"):
        stem = out[:-5]
    else:
        stem = out

    head_file = f"{stem}.HEAD"
    brik_file = f"{stem}.BRIK.gz" if data_encoding == "gzip" else f"{stem}.BRIK"

    if arr.ndim == 3:
        dims = arr.shape
        nvols = 1
    else:
        dims = arr.shape[:3]
        nvols = arr.shape[3]

    if data_type is not None:
        dt_req = str(data_type).upper()
        dtype_lookup = {
            "FLOAT32": np.float32,
            "FLOAT": np.float32,
            "INT16": np.int16,
            "SHORT": np.int16,
            "UINT8": np.uint8,
            "BYTE": np.uint8,
        }
        if dt_req not in dtype_lookup:
            raise ValueError(f"Unsupported AFNI data_type: {data_type}")
        arr_dtype = np.dtype(dtype_lookup[dt_req])
    else:
        arr_dtype = arr.dtype

    brick_type, storage_dtype = _afni_dtype_info(arr_dtype)
    if np.issubdtype(storage_dtype, np.integer):
        arr_write, facs = _quantize_bricks(arr, storage_dtype)
    else:
        arr_write = np.asarray(arr, dtype=storage_dtype)
        facs = [0.0] * nvols
        if np.issubdtype(arr.dtype, np.integer) and not np.array_equal(
            arr_write.astype(arr.dtype), arr
        ):
            import warnings

            warnings.warn(
                "integer values beyond float32 precision were rounded when "
                "writing AFNI (store them as INT16 with scaling or use NIfTI)",
                UserWarning,
                stacklevel=2,
            )

    if affine is None:
        affine = np.eye(4)
        affine[:3, :3] = np.diag([float(v) for v in spacing[:3]])
        affine[:3, 3] = [float(v) for v in origin[:3]]
    real, card, orient, delta, origin_dicom = _afni_geometry(affine)

    byte_order = "LSB_FIRST" if np.little_endian else "MSB_FIRST"
    head_txt = (
        _afni_string("TYPESTRING", ["3DIM_HEAD_ANAT"])
        # SCENE_DATA: view (0 = +orig), anatomy type, type-string index
        + _afni_numbers("integer", "SCENE_DATA", [0, 0, 0])
        + _afni_numbers("integer", "DATASET_DIMENSIONS", [*dims, 0, 0])
        + _afni_numbers("integer", "DATASET_RANK", [3, nvols])
        + _afni_numbers("integer", "ORIENT_SPECIFIC", orient)
        + _afni_numbers("float", "DELTA", delta)
        + _afni_numbers("float", "ORIGIN", origin_dicom)
        + _afni_numbers("float", "IJK_TO_DICOM", card.ravel())
        + _afni_numbers("float", "IJK_TO_DICOM_REAL", real.ravel())
        + _afni_numbers("integer", "BRICK_TYPES", [brick_type] * nvols)
        + _afni_numbers("float", "BRICK_FLOAT_FACS", facs)
        + _afni_string("BYTEORDER_STRING", [byte_order])
        + _afni_string("BRICK_LABS", [f"vol{i}" for i in range(nvols)])
    )
    if nvols > 1 and tr is not None and tr > 0:
        # TAXIS_NUMS: ntt, nz (no slice offsets), units (77002 = seconds)
        head_txt += _afni_numbers("integer", "TAXIS_NUMS", [nvols, 0, 77002])
        head_txt += _afni_numbers(
            "float", "TAXIS_FLOATS", [0.0, float(tr), 0.0, 0.0, 0.0]
        )
    Path(head_file).write_text(head_txt, encoding="utf-8")

    raw = arr_write.ravel(order="F").tobytes()
    if data_encoding == "gzip":
        with gzip.open(brik_file, "wb") as f:
            f.write(raw)
    else:
        Path(brik_file).write_bytes(raw)

    return head_file, brik_file

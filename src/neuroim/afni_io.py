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
    # AFNI string attributes are typically tilde-delimited: ~val1~val2~...
    # but some inputs may provide a single bare token (e.g. MSB_FIRST).
    text = "\n".join(lines).strip()
    if not text:
        return []

    if text.startswith("~") and text.endswith("~"):
        text = text[1:-1]

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
        if typ == "int":
            dtype = np.dtype("<i4")
        else:
            # neuroim2 reads binary float/double NIML payloads with size = 4.
            dtype = np.dtype("<f4")
        values = np.frombuffer(fconn.read(nvals * dtype.itemsize), dtype=dtype, count=nvals)
        if values.size != nvals:
            raise ValueError(f"NIML data block is truncated: expected {nvals}, got {values.size}")
    elif form == "binary.msbfirst":
        if typ == "int":
            dtype = np.dtype(">i4")
        else:
            dtype = np.dtype(">f4")
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


def _afni_dtype_info(dtype: np.dtype) -> tuple[int, np.dtype]:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.uint8):
        return 0, dt
    if dt == np.dtype(np.int16):
        return 1, dt
    # AFNI reader in neuroim currently supports FLOAT32 as general fallback.
    return 3, np.dtype(np.float32)


def write_afni_pair(
    file_name: Union[str, Path],
    data: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    data_encoding: str = "raw",
    data_type: str | None = None,
) -> tuple[str, str]:
    """Write AFNI HEAD/BRIK pair.

    Parameters
    ----------
    file_name : str or Path
        Output path hint. May be ``.HEAD``, ``.BRIK``, ``.BRIK.gz``, or stem.
    data : ndarray
        3D or 4D array to write.
    spacing : tuple
        Spatial spacing for x/y/z.
    origin : tuple
        Spatial origin for x/y/z.
    data_encoding : {"raw","gzip"}
        Data payload encoding. ``gzip`` writes ``.BRIK.gz``.
    data_type : str, optional
        Requested storage type. Supports ``FLOAT32``, ``INT16``, ``UINT8``.

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
            "INT16": np.int16,
            "UINT8": np.uint8,
        }
        if dt_req not in dtype_lookup:
            raise ValueError(f"Unsupported AFNI data_type: {data_type}")
        arr_dtype = np.dtype(dtype_lookup[dt_req])
    else:
        arr_dtype = arr.dtype

    brick_type, storage_dtype = _afni_dtype_info(arr_dtype)
    arr_write = np.asarray(arr, dtype=storage_dtype)

    facs = " ".join(["1.0"] * nvols)
    brick_types = " ".join([str(brick_type)] * nvols)
    labels = "~" + "~".join([f"vol{i}" for i in range(nvols)]) + "~"
    byte_order = "LSB_FIRST" if np.little_endian else "MSB_FIRST"

    sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))

    # Keep y negative to match common AFNI conventions; readers use abs(DELTA).
    dy = -abs(sy)
    ijk_to_dicom = (
        f"{sx} 0.0 0.0 {ox} "
        f"0.0 {dy} 0.0 {oy} "
        f"0.0 0.0 {sz} {oz}"
    )

    head_txt = (
        "type = integer-attribute\n"
        "name = DATASET_DIMENSIONS\n"
        "count = 5\n"
        f"{int(dims[0])} {int(dims[1])} {int(dims[2])} 1 0\n\n"
        "type = integer-attribute\n"
        "name = DATASET_RANK\n"
        "count = 2\n"
        f"3 {int(nvols)}\n\n"
        "type = float-attribute\n"
        "name = DELTA\n"
        "count = 3\n"
        f"{sx} {dy} {sz}\n\n"
        "type = float-attribute\n"
        "name = IJK_TO_DICOM\n"
        "count = 12\n"
        f"{ijk_to_dicom}\n\n"
        "type = float-attribute\n"
        "name = ORIGIN\n"
        "count = 3\n"
        f"{ox} {oy} {oz}\n\n"
        "type = integer-attribute\n"
        "name = BRICK_TYPES\n"
        f"count = {int(nvols)}\n"
        f"{brick_types}\n\n"
        "type = float-attribute\n"
        "name = BRICK_FLOAT_FACS\n"
        f"count = {int(nvols)}\n"
        f"{facs}\n\n"
        "type = string-attribute\n"
        "name = BYTEORDER_STRING\n"
        f"count = {len(byte_order) + 2}\n"
        f"~{byte_order}~\n\n"
        "type = string-attribute\n"
        "name = BRICK_LABS\n"
        f"count = {len(labels)}\n"
        f"{labels}\n"
    )
    Path(head_file).write_text(head_txt, encoding="utf-8")

    raw = arr_write.ravel(order="F").tobytes()
    if data_encoding == "gzip":
        with gzip.open(brik_file, "wb") as f:
            f.write(raw)
    else:
        Path(brik_file).write_bytes(raw)

    return head_file, brik_file

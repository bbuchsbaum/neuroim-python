"""Binary I/O support for neuroimaging data.

This module provides low-level binary reading and writing capabilities
for neuroimaging file formats.

Binary I/O helpers for neuroimaging file formats.
"""

from typing import Union, BinaryIO, Optional
from pathlib import Path
import numpy as np
import struct

_FIXED_TYPES = {
    "float32": "f4", "float": "f4", "single": "f4",
    "float64": "f8", "double": "f8",
    "int8": "i1", "int16": "i2", "short": "i2", "int32": "i4", "int64": "i8",
    "long": "i8", "uint8": "u1", "byte": "u1", "uint16": "u2", "uint32": "u4",
    "uint64": "u8",
}


def _binary_dtype(
    data_type: str, bytes_per_element: int, endian: str, signed: bool = True
) -> np.dtype:
    """Resolve numpy and R storage names to a dtype, checking the element size.

    R-style generic names take their width from ``bytes_per_element``
    (``"integer"``/``"int"``: 1, 2, 4 or 8 bytes; ``"numeric"``: 4 or 8).
    Unknown names, or a name whose width disagrees with
    ``bytes_per_element``, raise instead of silently reading float32.
    """
    name = str(data_type).lower()
    size = int(bytes_per_element)
    if name in ("integer", "int"):
        if size not in (1, 2, 4, 8):
            raise ValueError(f"no {size}-byte integer type")
        code = ("i" if signed else "u") + str(size)
    elif name == "numeric":
        if size not in (4, 8):
            raise ValueError(f"no {size}-byte floating type")
        code = "f" + str(size)
    elif name in _FIXED_TYPES:
        code = _FIXED_TYPES[name]
        if not signed and code[0] == "i":
            code = "u" + code[1:]
    else:
        raise ValueError(f"Unsupported binary data_type: {data_type!r}")
    dtype = np.dtype(("<" if endian == "little" else ">") + code)
    if dtype.itemsize != size:
        raise ValueError(
            f"data_type {data_type!r} is {dtype.itemsize} bytes per element, "
            f"but bytes_per_element={size}"
        )
    return dtype


class BinaryReader:
    """Class supporting reading of bulk binary data from a connection.

    Attributes
    ----------
    input : BinaryIO
        The binary input connection
    byte_offset : int
        Number of bytes to skip at start of input
    data_type : str
        Data type of the binary elements
    bytes_per_element : int
        Number of bytes in each data element
    endian : str
        Endianness ("little" or "big")
    signed : bool
        Whether the data type is signed

    """

    def __init__(
        self,
        input_connection: Union[BinaryIO, str, Path],
        byte_offset: int = 0,
        data_type: str = "float32",
        bytes_per_element: int = 4,
        endian: str = "little",
        signed: bool = True,
    ):

        # Handle file path input
        if isinstance(input_connection, (str, Path)):
            self.input = open(input_connection, "rb")
            self._owns_file = True
        else:
            self.input = input_connection
            self._owns_file = False

        self.byte_offset = byte_offset
        self.data_type = data_type
        self.bytes_per_element = bytes_per_element
        self.endian = endian
        self.signed = signed

        # Skip to offset
        if byte_offset > 0:
            self.input.seek(byte_offset)

    def read_elements(self, n: int) -> np.ndarray:
        """Read n elements from the binary stream.

        Parameters
        ----------
        n : int
            Number of elements to read

        Returns
        -------
        np.ndarray
            Array of read elements
        """
        # Determine numpy dtype
        dtype = self._get_numpy_dtype()

        # Read binary data
        bytes_to_read = n * self.bytes_per_element
        data = self.input.read(bytes_to_read)

        if len(data) < bytes_to_read:
            raise EOFError(f"Expected {bytes_to_read} bytes, got {len(data)}")

        # Convert to numpy array
        return np.frombuffer(data, dtype=dtype)

    def read_all(self) -> np.ndarray:
        """Read all remaining elements from the stream."""
        # Read all remaining data
        data = self.input.read()

        # Determine how many complete elements we have
        n_elements = len(data) // self.bytes_per_element

        if n_elements == 0:
            return np.array([], dtype=self._get_numpy_dtype())

        # Truncate to complete elements
        data = data[: n_elements * self.bytes_per_element]

        # Convert to numpy array
        dtype = self._get_numpy_dtype()
        return np.frombuffer(data, dtype=dtype)

    def _get_numpy_dtype(self) -> np.dtype:
        """Get numpy dtype from attributes."""
        return _binary_dtype(
            self.data_type,
            self.bytes_per_element,
            self.endian,
            getattr(self, "signed", True),
        )

    def close(self):
        """Close the connection if we own it."""
        if self._owns_file and hasattr(self.input, "close"):
            self.input.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class BinaryWriter:
    """Class supporting writing of bulk binary data to a connection.

    Attributes
    ----------
    output : BinaryIO
        The binary output connection
    byte_offset : int
        Number of bytes to skip at start of output
    data_type : str
        Data type of the binary elements
    bytes_per_element : int
        Number of bytes in each data element
    endian : str
        Endianness ("little" or "big")

    """

    def __init__(
        self,
        output_connection: Union[BinaryIO, str, Path],
        byte_offset: int = 0,
        data_type: str = "float32",
        bytes_per_element: int = 4,
        endian: str = "little",
    ):

        # Handle file path input
        if isinstance(output_connection, (str, Path)):
            self.output = open(output_connection, "wb")
            self._owns_file = True
        else:
            self.output = output_connection
            self._owns_file = False

        self.byte_offset = byte_offset
        self.data_type = data_type
        self.bytes_per_element = bytes_per_element
        self.endian = endian

        # Write padding bytes if offset specified
        if byte_offset > 0:
            self.output.write(b"\x00" * byte_offset)

    def write_elements(self, data: np.ndarray):
        """Write elements to the binary stream.

        Parameters
        ----------
        data : np.ndarray
            Array of elements to write
        """
        # Ensure data is numpy array
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        # Get expected dtype
        expected_dtype = self._get_numpy_dtype()

        # Convert if necessary
        if data.dtype != expected_dtype:
            data = data.astype(expected_dtype)

        # Write to stream
        self.output.write(data.tobytes())

    def _get_numpy_dtype(self) -> np.dtype:
        """Get numpy dtype from attributes."""
        return _binary_dtype(
            self.data_type,
            self.bytes_per_element,
            self.endian,
            getattr(self, "signed", True),
        )

    def close(self):
        """Close the connection if we own it."""
        if self._owns_file and hasattr(self.output, "close"):
            self.output.flush()
            self.output.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class ColumnReader:
    """Reader for column-oriented binary data.

    This class reads binary data organized in columns, where each column
    represents a different variable or time point.

    Attributes
    ----------
    reader : BinaryReader
        Underlying binary reader
    n_cols : int
        Number of columns
    n_rows : int
        Number of rows per column

    Notes
    -----
    This is useful for reading time series data where each column
    represents a different time point.
    """

    def __init__(
        self,
        input_connection: Union[BinaryIO, str, Path],
        n_cols: int,
        n_rows: Optional[int] = None,
        **kwargs,
    ):
        self.reader = BinaryReader(input_connection, **kwargs)
        self.n_cols = n_cols
        self.n_rows = n_rows

    def read_column(self, col_index: int) -> np.ndarray:
        """Read a single column.

        Parameters
        ----------
        col_index : int
            Column index (0-based)

        Returns
        -------
        np.ndarray
            Column data
        """
        if self.n_rows is None:
            raise ValueError("n_rows must be specified to read individual columns")

        if col_index < 0 or col_index >= self.n_cols:
            raise ValueError(
                f"Column index {col_index} out of range [0, {self.n_cols})"
            )

        # Seek to column start
        offset = (
            self.reader.byte_offset
            + col_index * self.n_rows * self.reader.bytes_per_element
        )
        self.reader.input.seek(offset)

        # Read column data
        return self.reader.read_elements(self.n_rows)

    def read_all_columns(self) -> np.ndarray:
        """Read all columns into a 2D array.

        Returns
        -------
        np.ndarray
            2D array with shape (n_rows, n_cols)
        """
        # Read all data from the start of the matrix, wherever an earlier
        # read_column() left the stream.
        self.reader.input.seek(self.reader.byte_offset)
        data = self.reader.read_all()

        if self.n_rows is None:
            # Infer n_rows from data size
            if len(data) % self.n_cols != 0:
                raise ValueError(
                    f"Data size {len(data)} not divisible by n_cols {self.n_cols}"
                )
            self.n_rows = len(data) // self.n_cols

        # Reshape to 2D
        return data.reshape(self.n_rows, self.n_cols, order="F")  # Column-major order

    def close(self):
        """Close the underlying reader."""
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

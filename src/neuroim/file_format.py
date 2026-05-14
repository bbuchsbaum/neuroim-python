"""File format support for neuroimaging data.

This module provides classes for handling various neuroimaging file formats,
including NIfTI and AFNI formats. It uses nibabel as the backend for robust
file I/O operations while maintaining API compatibility with R's neuroim2.

File-format descriptors used by neuroim image I/O.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Any, List
import re
import numpy as np

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


class FileFormat(ABC):
    """Abstract base class for neuroimaging file formats.

    This class represents a neuroimaging file format descriptor, containing
    information about the file format, encoding, and extensions for both header
    and data components.

    Attributes
    ----------
    file_format : str
        Name of the file format (e.g., "NIFTI", "AFNI")
    header_encoding : str
        File encoding of the header file ("raw" or "gzip")
    header_extension : str
        File extension for the header file
    data_encoding : str
        File encoding for the data file ("raw" or "gzip")
    data_extension : str
        File extension for the data file

    R Equivalent
    ------------
    neuroim2::FileFormat
    """

    def __init__(
        self,
        file_format: str,
        header_encoding: str,
        header_extension: str,
        data_encoding: str,
        data_extension: str,
    ):
        self.file_format = file_format
        self.header_encoding = header_encoding
        self.header_extension = header_extension
        self.data_encoding = data_encoding
        self.data_extension = data_extension

    def file_matches(self, file_name: Union[str, Path]) -> bool:
        """Check if a file matches this format and both header/data files exist.

        Parameters
        ----------
        file_name : str or Path
            File name to validate

        Returns
        -------
        bool
            True if file matches format and paired file exists

        R Equivalent
        ------------
        neuroim2::file_matches
        """
        file_name = str(file_name)

        if self.header_file_matches(file_name):
            # Get corresponding data file
            data_file = self.strip_extension(file_name) + "." + self.data_extension
            return Path(data_file).exists()
        elif self.data_file_matches(file_name):
            # Get corresponding header file
            header_file = self.strip_extension(file_name) + "." + self.header_extension
            return Path(header_file).exists()
        else:
            return False

    def header_file_matches(self, file_name: Union[str, Path]) -> bool:
        """Check if file name matches header format.

        Parameters
        ----------
        file_name : str or Path
            File name to check

        Returns
        -------
        bool
            True if file matches header format

        R Equivalent
        ------------
        neuroim2::header_file_matches
        """
        file_name = str(file_name)
        pattern = re.escape("." + self.header_extension) + "$"
        return bool(re.search(pattern, file_name))

    def data_file_matches(self, file_name: Union[str, Path]) -> bool:
        """Check if file name matches data format.

        Parameters
        ----------
        file_name : str or Path
            File name to check

        Returns
        -------
        bool
            True if file matches data format

        R Equivalent
        ------------
        neuroim2::data_file_matches
        """
        file_name = str(file_name)
        pattern = re.escape("." + self.data_extension) + "$"
        return bool(re.search(pattern, file_name))

    def header_file(self, file_name: Union[str, Path]) -> str:
        """Get header file name from any file name.

        Parameters
        ----------
        file_name : str or Path
            Input file name

        Returns
        -------
        str
            Header file name

        Raises
        ------
        ValueError
            If cannot derive header file name

        R Equivalent
        ------------
        neuroim2::header_file
        """
        file_name = str(file_name)

        if self.header_file_matches(file_name):
            return file_name
        elif self.data_file_matches(file_name):
            return self.strip_extension(file_name) + "." + self.header_extension
        else:
            raise ValueError(f"Could not derive header file name from: {file_name}")

    def data_file(self, file_name: Union[str, Path]) -> str:
        """Get data file name from any file name.

        Parameters
        ----------
        file_name : str or Path
            Input file name

        Returns
        -------
        str
            Data file name

        Raises
        ------
        ValueError
            If cannot derive data file name

        R Equivalent
        ------------
        neuroim2::data_file
        """
        file_name = str(file_name)

        if self.data_file_matches(file_name):
            return file_name
        elif self.header_file_matches(file_name):
            return self.strip_extension(file_name) + "." + self.data_extension
        else:
            raise ValueError(f"Could not derive data file name from: {file_name}")

    def strip_extension(self, file_name: Union[str, Path]) -> str:
        """Remove file extension based on format.

        Parameters
        ----------
        file_name : str or Path
            File name to strip

        Returns
        -------
        str
            File name without extension

        Raises
        ------
        ValueError
            If file doesn't match format

        R Equivalent
        ------------
        neuroim2::strip_extension
        """
        file_name = str(file_name)

        if self.header_file_matches(file_name):
            # Remove header extension
            pattern = re.escape("." + self.header_extension) + "$"
            return re.sub(pattern, "", file_name)
        elif self.data_file_matches(file_name):
            # Remove data extension
            pattern = re.escape("." + self.data_extension) + "$"
            return re.sub(pattern, "", file_name)
        else:
            raise ValueError(f"File does not match format: {file_name}")

    @abstractmethod
    def read_meta_info(self, file_name: Union[str, Path]) -> "FileMetaInfo":
        """Read metadata from file.

        Parameters
        ----------
        file_name : str or Path
            File to read metadata from

        Returns
        -------
        FileMetaInfo
            Metadata object

        R Equivalent
        ------------
        neuroim2::read_meta_info
        """
        pass


class NIFTIFormat(FileFormat):
    """NIfTI file format support.

    R Equivalent
    ------------
    neuroim2::NIFTIFormat
    """

    def read_meta_info(self, file_name: Union[str, Path]) -> "NIFTIMetaInfo":
        """Read NIfTI metadata using nibabel."""
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required to read NIfTI files")

        from .meta_info import NIFTIMetaInfo

        header_file = self.header_file(file_name)

        # Use nibabel to read the header
        img = nib.load(header_file)
        header = img.header

        # Extract slope and intercept from dataobj if available
        slope = 1.0
        intercept = 0.0
        if hasattr(img.dataobj, "slope") and img.dataobj.slope is not None:
            slope = float(img.dataobj.slope)
        elif not np.isnan(header["scl_slope"]) and float(header["scl_slope"]) != 0.0:
            slope = float(header["scl_slope"])

        if hasattr(img.dataobj, "inter") and img.dataobj.inter is not None:
            intercept = float(img.dataobj.inter)
        elif not np.isnan(header["scl_inter"]):
            intercept = float(header["scl_inter"])

        # Extract metadata
        meta_info = NIFTIMetaInfo(
            header_file=header_file,
            data_file=self.data_file(file_name),
            descriptor=self,
            data_type=str(header.get_data_dtype()),
            dims=img.shape,
            spacing=header.get_zooms(),
            origin=img.affine[:3, 3],
            endian="<" if header.endianness == "<" else ">",
            data_offset=int(header["vox_offset"]),
            bytes_per_element=header.get_data_dtype().itemsize,
            intercept=intercept,
            slope=slope,
            nifti_header=dict(header),
        )

        return meta_info


class AFNIFormat(FileFormat):
    """AFNI file format support.

    R Equivalent
    ------------
    neuroim2::AFNIFormat
    """

    def read_meta_info(self, file_name: Union[str, Path]) -> "AFNIMetaInfo":
        """Read AFNI metadata from ``.HEAD`` and paired ``.BRIK``."""
        from .afni_io import read_afni_header
        from .meta_info import AFNIMetaInfo

        header_file = self.header_file(file_name)
        afni_header = read_afni_header(header_file)
        afni_header["file_name"] = header_file

        ijk_to_dicom = afni_header.get("IJK_TO_DICOM", {}).get("content")
        try:
            ijk_to_dicom_values = np.asarray(ijk_to_dicom, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("Invalid IJK_TO_DICOM transformation in AFNI header")

        if ijk_to_dicom_values.size < 12 or not np.all(
            np.isfinite(ijk_to_dicom_values[:12])
        ):
            raise ValueError("Invalid IJK_TO_DICOM transformation in AFNI header")

        if "DATASET_DIMENSIONS" not in afni_header:
            raise ValueError("Missing DATASET_DIMENSIONS in AFNI header")
        dims_raw = [int(x) for x in afni_header["DATASET_DIMENSIONS"]["content"]]
        if len(dims_raw) < 3 or any(d <= 0 for d in dims_raw[:3]):
            raise ValueError("AFNI dataset must have at least 3 dimensions")
        dims: List[int] = dims_raw[:3]

        dataset_rank = afni_header.get("DATASET_RANK", {}).get("content", [3, 1])
        try:
            nvols = int(dataset_rank[1]) if len(dataset_rank) > 1 else 1
        except (TypeError, ValueError):
            raise ValueError("Invalid DATASET_RANK in AFNI header")
        if nvols > 1:
            dims.append(nvols)

        brick_labs = afni_header.get("BRICK_LABS", {}).get("content")
        if brick_labs is None:
            labels = [f"#{i}" for i in range(nvols)]
        else:
            labels = [str(x) for x in brick_labs]

        if "BRICK_TYPES" not in afni_header:
            raise ValueError("Missing BRICK_TYPES in AFNI header")
        brick_type_raw = afni_header["BRICK_TYPES"].get("content", [3])
        if not brick_type_raw:
            raise ValueError("Missing BRICK_TYPES in AFNI header")
        try:
            brick_type = int(brick_type_raw[0])
        except (TypeError, ValueError):
            raise ValueError("Invalid BRICK_TYPES in AFNI header")
        data_type_map = {0: "UINT8", 1: "INT16", 3: "FLOAT32"}
        bpe_map = {0: 1, 1: 2, 3: 4}
        if brick_type not in data_type_map:
            raise ValueError(f"Unsupported BRICK_TYPE in AFNI header: {brick_type}")
        data_type = data_type_map[brick_type]
        bytes_per_element = bpe_map[brick_type]

        byte_order = afni_header.get("BYTEORDER_STRING", {}).get(
            "content", ["LSB_FIRST"]
        )
        byte_order_val = (
            byte_order[0]
            if isinstance(byte_order, list) and byte_order
            else "LSB_FIRST"
        )
        endian = "big" if byte_order_val == "MSB_FIRST" else "little"

        spacing_vals = afni_header.get("DELTA", {}).get("content", [1.0, 1.0, 1.0])
        spacing = tuple(abs(float(x)) for x in spacing_vals[:3])

        origin_vals = afni_header.get("ORIGIN", {}).get("content", [0.0, 0.0, 0.0])
        origin = tuple(float(x) for x in origin_vals[:3])

        float_facs = afni_header.get("BRICK_FLOAT_FACS", {}).get("content", [0.0])
        slope = np.asarray(float_facs, dtype=float)
        slope[slope == 0] = 1.0
        if slope.size == 1:
            slope_out: Any = float(slope[0])
        else:
            slope_out = slope

        return AFNIMetaInfo(
            header_file=header_file,
            data_file=self.data_file(file_name),
            descriptor=self,
            data_type=data_type,
            dims=tuple(dims),
            spacing=spacing,
            origin=origin,
            endian=endian,
            data_offset=0,
            bytes_per_element=int(bytes_per_element),
            intercept=0.0,
            slope=slope_out,
            header=afni_header,
            label=labels,
            afni_header=afni_header,
        )


# Format constants matching R implementation
NIFTI = NIFTIFormat(
    file_format="NIFTI",
    header_encoding="raw",
    header_extension="nii",
    data_encoding="raw",
    data_extension="nii",
)

NIFTI_GZ = NIFTIFormat(
    file_format="NIFTI",
    header_encoding="gzip",
    header_extension="nii.gz",
    data_encoding="gzip",
    data_extension="nii.gz",
)

NIFTI_PAIR = NIFTIFormat(
    file_format="NIFTI",
    header_encoding="raw",
    header_extension="hdr",
    data_encoding="raw",
    data_extension="img",
)

NIFTI_PAIR_GZ = NIFTIFormat(
    file_format="NIFTI",
    header_encoding="gzip",
    header_extension="hdr.gz",
    data_encoding="gzip",
    data_extension="img.gz",
)

AFNI = AFNIFormat(
    file_format="AFNI",
    header_encoding="raw",
    header_extension="HEAD",
    data_encoding="raw",
    data_extension="BRIK",
)

AFNI_GZ = AFNIFormat(
    file_format="AFNI",
    header_encoding="gzip",
    header_extension="HEAD",
    data_encoding="gzip",
    data_extension="BRIK.gz",
)


def find_descriptor(file_name: Union[str, Path]) -> Optional[FileFormat]:
    """Find the appropriate file format descriptor for a file.

    Parameters
    ----------
    file_name : str or Path
        File name to check

    Returns
    -------
    FileFormat or None
        Matching format descriptor or None if no match

    R Equivalent
    ------------
    neuroim2::find_descriptor (internal function)
    """
    formats = [NIFTI, NIFTI_GZ, NIFTI_PAIR, NIFTI_PAIR_GZ, AFNI, AFNI_GZ]

    for fmt in formats:
        if fmt.file_matches(file_name):
            return fmt

    return None

"""Metadata information classes for neuroimaging data.

This module provides classes for storing and managing metadata associated with
neuroimaging files, including spatial dimensions, data types, and format-specific
header information.

Direct translation of R's neuroim2 metadata classes.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .axis import AxisSet, AxisSet3D
from .file_format import FileFormat


class MetaInfo:
    """Base class for neuroimaging metadata.
    
    This class encapsulates meta information for neuroimaging data types,
    including spatial and temporal characteristics, data type, and labeling.
    
    Attributes
    ----------
    data_type : str
        Data type code (e.g., "FLOAT", "INT")
    dims : tuple
        Image dimensions
    spatial_axes : AxisSet3D
        Image axes for spatial dimensions (x, y, z)
    additional_axes : AxisSet
        Axes for dimensions beyond spatial (e.g., time, color band)
    spacing : tuple
        Voxel dimensions in real-world units
    origin : tuple
        Coordinate origin
    label : List[str]
        Name(s) of images or data series
        
    R Equivalent
    ------------
    neuroim2::MetaInfo
    """
    
    def __init__(self, data_type: str, dims: tuple, spatial_axes: AxisSet3D,
                 additional_axes: AxisSet, spacing: tuple, origin: tuple,
                 label: Optional[List[str]] = None):
        self.data_type = data_type
        self.dims = tuple(dims)
        self.spatial_axes = spatial_axes
        self.additional_axes = additional_axes
        self.spacing = tuple(spacing)
        self.origin = tuple(origin)
        self.label = label or []
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dims)
    
    @property
    def nvols(self) -> int:
        """Number of volumes (for 4D data)."""
        if self.ndim >= 4:
            return self.dims[3]
        return 1


class FileMetaInfo(MetaInfo):
    """Extended metadata for file-based neuroimaging data.
    
    This class extends MetaInfo to include file-specific metadata for
    neuroimaging data files.
    
    Attributes
    ----------
    header_file : str
        Name of the file containing meta information
    data_file : str
        Name of the file containing image data
    descriptor : FileFormat
        Image file format descriptor
    endian : str
        Byte order of data ('little' or 'big')
    data_offset : int
        Number of bytes preceding the start of image data
    bytes_per_element : int
        Number of bytes per data element
    intercept : float or array
        Constant values added to image data (one per sub-image)
    slope : float or array
        Multipliers for image data (one per sub-image)
    header : dict
        Format-specific attributes
        
    R Equivalent
    ------------
    neuroim2::FileMetaInfo
    """
    
    def __init__(self, header_file: str, data_file: str, descriptor: FileFormat,
                 data_type: str, dims: tuple, spacing: tuple, origin: tuple,
                 endian: str = "little", data_offset: int = 0,
                 bytes_per_element: int = 4, intercept: float = 0.0,
                 slope: float = 1.0, header: Optional[Dict[str, Any]] = None,
                 spatial_axes: Optional[AxisSet3D] = None,
                 additional_axes: Optional[AxisSet] = None,
                 label: Optional[List[str]] = None):
        
        # Create default axes if not provided
        if spatial_axes is None:
            from .axis import find_anatomy_3d
            spatial_axes = find_anatomy_3d("RAS")  # Default to RAS
        
        if additional_axes is None:
            from .axis import AxisSet1D, TimeAxis
            if len(dims) > 3:
                additional_axes = AxisSet1D(TimeAxis)
            else:
                additional_axes = AxisSet(0)  # Empty for 3D
        
        super().__init__(
            data_type=data_type,
            dims=dims,
            spatial_axes=spatial_axes,
            additional_axes=additional_axes,
            spacing=spacing,
            origin=origin,
            label=label
        )
        
        self.header_file = header_file
        self.data_file = data_file
        self.descriptor = descriptor
        self.endian = endian
        self.data_offset = data_offset
        self.bytes_per_element = bytes_per_element
        self.intercept = intercept
        self.slope = slope
        self.header = header or {}
    
    @property
    def byte_order(self) -> str:
        """Get numpy-style byte order character."""
        return '<' if self.endian == "little" else '>'
    
    def get_data_dtype(self) -> np.dtype:
        """Get numpy dtype for the data."""
        # Map data_type strings to numpy dtypes
        dtype_map = {
            "FLOAT32": np.float32,
            "FLOAT64": np.float64,
            "FLOAT": np.float32,
            "DOUBLE": np.float64,
            "BYTE": np.uint8,
            "BINARY": np.uint8,
            "SHORT": np.int16,
            "INT8": np.int8,
            "INT16": np.int16,
            "INT32": np.int32,
            "INT64": np.int64,
            "UINT8": np.uint8,
            "UINT16": np.uint16,
            "UINT32": np.uint32,
            "UINT64": np.uint64,
        }
        
        base_dtype = dtype_map.get(self.data_type.upper(), np.float32)
        # Create numpy dtype instance first
        if not isinstance(base_dtype, np.dtype):
            base_dtype = np.dtype(base_dtype)
        # Add byte order
        return np.dtype(f"{self.byte_order}{base_dtype.str[1:]}")


class NIFTIMetaInfo(FileMetaInfo):
    """NIfTI-specific metadata.
    
    This class extends FileMetaInfo with NIfTI-specific metadata.
    
    Attributes
    ----------
    nifti_header : dict
        Attributes specific to the NIfTI file format
        
    R Equivalent
    ------------
    neuroim2::NIFTIMetaInfo
    """
    
    def __init__(self, nifti_header: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.nifti_header = nifti_header
    
    @property
    def qform_code(self) -> int:
        """Get qform code from NIfTI header."""
        return self.nifti_header.get('qform_code', 0)
    
    @property
    def sform_code(self) -> int:
        """Get sform code from NIfTI header."""
        return self.nifti_header.get('sform_code', 0)
    
    @property
    def descrip(self) -> str:
        """Get description from NIfTI header."""
        desc = self.nifti_header.get('descrip', b'')
        if isinstance(desc, bytes):
            return desc.decode('utf-8').strip('\x00')
        return str(desc).strip()


class AFNIMetaInfo(FileMetaInfo):
    """AFNI-specific metadata.
    
    This class extends FileMetaInfo with AFNI-specific metadata.
    
    Attributes
    ----------
    afni_header : dict
        Attributes specific to the AFNI file format
        
    R Equivalent
    ------------
    neuroim2::AFNIMetaInfo
    """
    
    def __init__(self, afni_header: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.afni_header = afni_header

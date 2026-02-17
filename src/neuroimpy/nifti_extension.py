"""NIfTI extension support.

Provides classes for reading, inspecting, and parsing NIfTI header
extensions (e.g. AFNI, CIFTI, DICOM comment blocks).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Any
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Extension code registry
# ---------------------------------------------------------------------------

NiftiExtensionCodes: Dict[int, str] = {
    0: "unknown",
    2: "dicom",
    4: "afni",
    6: "comment",
    8: "xcede",
    10: "jimdiminfo",
    12: "workflow_fwds",
    14: "freesurfer",
    16: "pypickle",
    18: "mind_ident",
    20: "b_value",
    22: "spherical_direction",
    24: "dt_component",
    26: "shc_degreeorder",
    28: "voxbo",
    30: "caret",
    32: "cifti",
    34: "variable_frame_timing",
    38: "eval",
    40: "matlab",
    42: "quantiphyse",
    44: "mrs",
}


# ---------------------------------------------------------------------------
# NiftiExtension
# ---------------------------------------------------------------------------

@dataclass
class NiftiExtension:
    """A single NIfTI header extension.

    Parameters
    ----------
    code : int
        Numeric extension code (see :data:`NiftiExtensionCodes`).
    content : bytes
        Raw extension content.
    """

    code: int
    content: bytes

    @property
    def name(self) -> str:
        """Human-readable name for the extension code."""
        return NiftiExtensionCodes.get(self.code, "unknown")

    def __repr__(self) -> str:
        return (
            f"NiftiExtension(code={self.code}, name='{self.name}', "
            f"size={len(self.content)})"
        )


# ---------------------------------------------------------------------------
# NiftiExtensionList
# ---------------------------------------------------------------------------

class NiftiExtensionList:
    """Ordered collection of :class:`NiftiExtension` objects.

    Supports iteration, indexing, and code-based queries.
    """

    def __init__(self, extensions: Optional[List[NiftiExtension]] = None):
        self._extensions: List[NiftiExtension] = list(extensions or [])

    # -- container protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._extensions)

    def __iter__(self) -> Iterator[NiftiExtension]:
        return iter(self._extensions)

    def __getitem__(self, index):
        return self._extensions[index]

    # -- query helpers -------------------------------------------------------

    def get_codes(self) -> List[int]:
        """Return list of extension codes present."""
        return [ext.code for ext in self._extensions]

    def count(self) -> int:
        """Number of extensions."""
        return len(self._extensions)

    def __repr__(self) -> str:
        return f"NiftiExtensionList(count={self.count()}, codes={self.get_codes()})"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_extensions(img) -> NiftiExtensionList:
    """Extract extensions from a nibabel NIfTI image.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        A loaded NIfTI image (must expose ``header.extensions``).

    Returns
    -------
    NiftiExtensionList
        Parsed extensions.
    """
    header = img.header
    extensions = getattr(header, "extensions", None)
    if extensions is None:
        return NiftiExtensionList()

    items: List[NiftiExtension] = []
    for ext in extensions:
        code = int(ext.get_code())
        content = bytes(ext.get_content())
        items.append(NiftiExtension(code=code, content=content))

    return NiftiExtensionList(items)


def parse_afni_extension(ext: NiftiExtension) -> Dict[str, Any]:
    """Parse an AFNI extension's XML content into a dict.

    Parameters
    ----------
    ext : NiftiExtension
        An extension with ``code == 4`` (AFNI).

    Returns
    -------
    dict
        Mapping of attribute names to their text values.

    Raises
    ------
    ValueError
        If the extension code is not 4 (AFNI).
    """
    if ext.code != 4:
        raise ValueError(
            f"Expected AFNI extension (code 4), got code {ext.code}"
        )

    text = ext.content.decode("utf-8", errors="replace").rstrip("\x00")
    root = ET.fromstring(text)

    result: Dict[str, Any] = {}
    for attr_elem in root.iter():
        name = attr_elem.get("ni_dimen") or attr_elem.get("atr_name") or attr_elem.tag
        # Prefer atr_name attribute when present (standard AFNI XML)
        atr_name = attr_elem.get("atr_name")
        if atr_name:
            name = atr_name
        if attr_elem.text and attr_elem.text.strip():
            result[name] = attr_elem.text.strip()

    return result


def get_afni_attribute(ext: NiftiExtension, name: str) -> Optional[str]:
    """Get a named attribute from an AFNI extension.

    Parameters
    ----------
    ext : NiftiExtension
        An AFNI extension (code 4).
    name : str
        Attribute name to look up.

    Returns
    -------
    str or None
        The attribute value, or ``None`` if not found.
    """
    parsed = parse_afni_extension(ext)
    return parsed.get(name)


def list_afni_attributes(ext: NiftiExtension) -> List[str]:
    """List all attribute names in an AFNI extension.

    Parameters
    ----------
    ext : NiftiExtension
        An AFNI extension (code 4).

    Returns
    -------
    list of str
        Attribute names.
    """
    parsed = parse_afni_extension(ext)
    return list(parsed.keys())

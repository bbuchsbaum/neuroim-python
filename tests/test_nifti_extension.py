"""Tests for NIfTI extension support."""

import numpy as np
import nibabel as nib
import pytest

from neuroim.nifti_extension import (
    NiftiExtensionCodes,
    NiftiExtension,
    NiftiExtensionList,
    parse_extensions,
    parse_afni_extension,
    get_afni_attribute,
    list_afni_attributes,
)


# ---------------------------------------------------------------------------
# NiftiExtensionCodes
# ---------------------------------------------------------------------------

class TestNiftiExtensionCodes:
    def test_known_codes(self):
        assert NiftiExtensionCodes[0] == "unknown"
        assert NiftiExtensionCodes[2] == "dicom"
        assert NiftiExtensionCodes[4] == "afni"
        assert NiftiExtensionCodes[6] == "comment"
        assert NiftiExtensionCodes[32] == "cifti"

    def test_is_dict(self):
        assert isinstance(NiftiExtensionCodes, dict)
        assert len(NiftiExtensionCodes) > 5


# ---------------------------------------------------------------------------
# NiftiExtension
# ---------------------------------------------------------------------------

class TestNiftiExtension:
    def test_basic_creation(self):
        ext = NiftiExtension(code=4, content=b"hello")
        assert ext.code == 4
        assert ext.content == b"hello"

    def test_name_known(self):
        ext = NiftiExtension(code=4, content=b"")
        assert ext.name == "afni"

    def test_name_unknown(self):
        ext = NiftiExtension(code=999, content=b"")
        assert ext.name == "unknown"

    def test_repr(self):
        ext = NiftiExtension(code=6, content=b"abc")
        r = repr(ext)
        assert "comment" in r
        assert "size=3" in r


# ---------------------------------------------------------------------------
# NiftiExtensionList
# ---------------------------------------------------------------------------

class TestNiftiExtensionList:
    def test_empty(self):
        lst = NiftiExtensionList()
        assert len(lst) == 0
        assert lst.count() == 0
        assert lst.get_codes() == []

    def test_with_extensions(self):
        exts = [
            NiftiExtension(code=4, content=b"a"),
            NiftiExtension(code=6, content=b"b"),
        ]
        lst = NiftiExtensionList(exts)
        assert len(lst) == 2
        assert lst.count() == 2
        assert lst.get_codes() == [4, 6]

    def test_getitem(self):
        exts = [
            NiftiExtension(code=4, content=b"a"),
            NiftiExtension(code=6, content=b"b"),
        ]
        lst = NiftiExtensionList(exts)
        assert lst[0].code == 4
        assert lst[1].code == 6

    def test_iter(self):
        exts = [
            NiftiExtension(code=4, content=b"x"),
            NiftiExtension(code=6, content=b"y"),
        ]
        lst = NiftiExtensionList(exts)
        codes = [e.code for e in lst]
        assert codes == [4, 6]

    def test_repr(self):
        lst = NiftiExtensionList([NiftiExtension(code=4, content=b"")])
        assert "count=1" in repr(lst)


# ---------------------------------------------------------------------------
# parse_extensions
# ---------------------------------------------------------------------------

class TestParseExtensions:
    def test_no_extensions(self, tmp_path):
        """A plain NIfTI file has no extensions."""
        data = np.zeros((3, 3, 3), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "plain.nii"
        nib.save(img, str(path))
        img2 = nib.load(str(path))
        result = parse_extensions(img2)
        assert isinstance(result, NiftiExtensionList)
        assert result.count() == 0

    def test_with_comment_extension(self, tmp_path):
        """Add a comment extension and verify round-trip."""
        data = np.zeros((3, 3, 3), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        comment = b"test comment"
        ext = nib.nifti1.Nifti1Extension(6, comment)
        img.header.extensions.append(ext)
        path = tmp_path / "ext.nii"
        nib.save(img, str(path))

        img2 = nib.load(str(path))
        result = parse_extensions(img2)
        assert result.count() == 1
        assert result[0].code == 6
        assert result[0].name == "comment"
        assert b"test comment" in result[0].content


# ---------------------------------------------------------------------------
# AFNI extension parsing
# ---------------------------------------------------------------------------

SAMPLE_AFNI_XML = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<AFNI_attributes>
  <AFNI_atr atr_name="HISTORY_NOTE">test history</AFNI_atr>
  <AFNI_atr atr_name="IDCODE_STRING">ABC_123</AFNI_atr>
  <AFNI_atr atr_name="SCENE_DATA">0 11 1</AFNI_atr>
</AFNI_attributes>
"""


class TestAfniParsing:
    def _make_afni_ext(self):
        return NiftiExtension(code=4, content=SAMPLE_AFNI_XML)

    def test_parse_afni_extension(self):
        ext = self._make_afni_ext()
        parsed = parse_afni_extension(ext)
        assert isinstance(parsed, dict)
        assert "HISTORY_NOTE" in parsed
        assert parsed["HISTORY_NOTE"] == "test history"
        assert parsed["IDCODE_STRING"] == "ABC_123"

    def test_parse_afni_wrong_code(self):
        ext = NiftiExtension(code=6, content=b"<x/>")
        with pytest.raises(ValueError, match="code 4"):
            parse_afni_extension(ext)

    def test_get_afni_attribute(self):
        ext = self._make_afni_ext()
        assert get_afni_attribute(ext, "HISTORY_NOTE") == "test history"
        assert get_afni_attribute(ext, "nonexistent") is None

    def test_list_afni_attributes(self):
        ext = self._make_afni_ext()
        names = list_afni_attributes(ext)
        assert "HISTORY_NOTE" in names
        assert "IDCODE_STRING" in names
        assert "SCENE_DATA" in names

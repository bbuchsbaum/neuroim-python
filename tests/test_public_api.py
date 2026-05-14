import neuroim as ni
import pytest


def test_public_api_budget_and_required_names():
    assert len(ni.__all__) <= 40

    required = {
        "NeuroSpace",
        "NeuroVol",
        "DenseNeuroVol",
        "SparseNeuroVol",
        "LogicalNeuroVol",
        "NeuroVec",
        "DenseNeuroVec",
        "SparseNeuroVec",
        "BigNeuroVec",
        "FileBackedNeuroVec",
        "MappedNeuroVec",
        "NeuroHyperVec",
        "ROICoords",
        "ROIVol",
        "spherical_roi",
        "searchlight",
        "searchlight_apply",
        "read_image",
        "read_vol",
        "write_vol",
        "read_vec",
        "write_vec",
        "from_nibabel",
        "resample",
        "reorient",
        "ConnCompResult",
        "SearchlightResult",
        "ROIExtractionResult",
        "Receipt",
        "compat",
    }
    assert required <= set(ni.__all__)


def test_public_api_has_no_r_shaped_export_names():
    forbidden = {
        "as.array",
        "as.dense",
        "as.mask",
        "as.matrix",
        "as.sparse",
        "None",
        "NeuroVecSeq",
        "createNIfTIHeader",
        "findAnatomy3D",
        "mapToColors",
        "matrixToQuatern",
        "quaternToMatrix",
        "read_hyper_vec",
    }
    assert forbidden.isdisjoint(ni.__all__)
    assert forbidden.isdisjoint(dir(ni))


def test_deprecated_top_level_import_shim_warns():
    namespace = {}
    with pytest.warns(DeprecationWarning, match="neuroim.compat"):
        exec("from neuroim import createNIfTIHeader", {}, namespace)

    assert namespace["createNIfTIHeader"] is ni.compat.createNIfTIHeader


def test_star_import_uses_curated_all_namespace():
    namespace = {}
    exec("from neuroim import *", {}, namespace)

    assert set(namespace) == set(ni.__all__)
    assert "NeuroSpace" in namespace
    assert "space" not in namespace
    assert "values" not in namespace
    assert "indices" not in namespace
    assert "as_array" not in namespace

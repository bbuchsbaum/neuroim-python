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


def test_dir_tracks_curated_public_surface():
    public_dir = {name for name in dir(ni) if not name.startswith("_")}
    allowed = set(ni.__all__)

    assert public_dir <= allowed
    assert len(public_dir) <= len(ni.__all__) + 5

    # Non-canonical helpers should stay discoverable through their submodules,
    # not through the interactive package root.
    assert "AxisSet" not in public_dir
    assert "FileFormat" not in public_dir
    assert "gaussian_blur" not in public_dir
    assert "neurovol" not in public_dir


# ME-4 additional guard: the prior `from .module import *` patterns leaked
# typing helpers (`Any`, `Optional`, `Union`), abstract bases (`ABC`,
# `abstractmethod`), and numpy (`np`) into the top-level namespace.  This
# is distinct from the R-shaped export blocklist above and catches
# implementation-detail re-exports that survive any future restructure.

_IMPLEMENTATION_LEAK_BLOCKLIST = frozenset({
    # typing
    "Any", "Optional", "Union", "List", "Tuple", "Dict", "Set",
    "Iterator", "Iterable", "Callable", "Sequence", "Mapping",
    "TYPE_CHECKING", "Generic", "TypeVar", "Protocol", "NewType",
    # ABCs / dataclass machinery
    "ABC", "abstractmethod", "dataclass", "field", "replace",
    # third-party shorthand
    "np", "numpy", "sparse", "pd", "pandas", "plt", "matplotlib",
    "nib", "nibabel",
    # stdlib pollution (NOTE: ``io`` is intentionally omitted — ``neuroim.io``
    # is the I/O subpackage)
    "os", "sys", "Path", "warnings", "logging", "logger", "hashlib",
})


def test_no_implementation_detail_leakage_at_top_level():
    leaked = sorted(set(dir(ni)) & _IMPLEMENTATION_LEAK_BLOCKLIST)
    assert not leaked, (
        f"Implementation-detail symbols leaked into neuroim namespace: "
        f"{leaked}. Likely cause: a `from .module import *` or an unaliased "
        f"`import <name>` at module load. Use `import <name> as _<name>` or "
        f"replace `*` with an explicit name list."
    )

"""Structural-provenance gate (MISSION.md rule 4).

Production code outside ``neuroim/results.py`` must build receipts through
:func:`neuroim.results.receipt_for` (with a typed :class:`OpParams`
subclass), not by hand-constructing :class:`Receipt` or by calling the
field-by-field :func:`make_receipt` shim.

Why a test instead of a convention: the alternative was lost when
``NeuroVol.to_nibabel``'s receipt-embed call site was silently stripped
in a parallel refactor (see ``examples/scenarios/05_receipt_io_boundary``
RESTORE 2026-05-14T18:something).  Convention-only contracts erode
under normal maintenance.  This test fails the build if a new
result-producing op skips the structural path.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Tuple

import pytest


# results.py owns the construction primitives.  Everything else must go
# through ``receipt_for``.
_RESULTS_FILE = "results.py"

# Operation-family classes that the structural path exposes.  Used to
# spot-check the call sites carry typed params (not raw kwargs).
_KNOWN_OPPARAMS = {
    "OpParams",
    "RoiOpParams",
    "SearchlightParams",
    "TemporalReductionParams",
    "SpatialFilterParams",
    "ParcelContrastParams",
    "ConcatParams",
}


def _src_neuroim_files() -> Iterable[Path]:
    root = Path(__file__).resolve().parent.parent / "src" / "neuroim"
    return sorted(p for p in root.rglob("*.py") if not p.name.startswith("_"))


def _direct_receipt_calls(file_path: Path) -> list[Tuple[int, str]]:
    """Return ``(lineno, source)`` for any ``Receipt(...)`` construction
    that is not the ``return Receipt(...)`` inside ``make_receipt``."""
    tree = ast.parse(file_path.read_text())
    hits: list[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Receipt":
            hits.append((node.lineno, ast.unparse(node)[:160]))
        elif isinstance(func, ast.Attribute) and func.attr == "Receipt":
            hits.append((node.lineno, ast.unparse(node)[:160]))
    return hits


def _make_receipt_callers(file_path: Path) -> list[Tuple[int, str]]:
    """Return ``(lineno, source)`` for direct ``make_receipt(...)`` calls
    outside ``results.py`` — these should migrate to ``receipt_for``."""
    tree = ast.parse(file_path.read_text())
    hits: list[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "make_receipt":
            hits.append((node.lineno, ast.unparse(node)[:160]))
    return hits


def test_only_results_module_constructs_receipt_directly():
    """Direct ``Receipt(...)`` construction only inside results.py.

    A new result-producing op that hand-builds a Receipt is precisely
    the regression S03 PAIN-1/2 surfaced (radius / method_name fields
    silently omitted).  The structural path makes the field set a
    property of the OpParams subclass, not of the call site.
    """
    offenders: dict[str, list[Tuple[int, str]]] = {}
    for path in _src_neuroim_files():
        if path.name == _RESULTS_FILE:
            continue
        direct = _direct_receipt_calls(path)
        if direct:
            offenders[str(path.relative_to(path.parents[2]))] = direct
    assert offenders == {}, (
        "Direct Receipt(...) construction found outside results.py:\n"
        + "\n".join(
            f"  {file}:{lineno}: {src}"
            for file, hits in offenders.items()
            for lineno, src in hits
        )
        + "\n\nUse neuroim.results.receipt_for(...) + an OpParams subclass instead."
    )


def test_production_callers_use_receipt_for_not_make_receipt():
    """In-tree result-producing ops must route through ``receipt_for``.

    ``make_receipt`` remains as the low-level primitive ``receipt_for``
    wraps; using it directly bypasses the typed-OpParams gate.
    """
    offenders: dict[str, list[Tuple[int, str]]] = {}
    for path in _src_neuroim_files():
        if path.name == _RESULTS_FILE:
            continue
        ad_hoc = _make_receipt_callers(path)
        if ad_hoc:
            offenders[str(path.relative_to(path.parents[2]))] = ad_hoc
    assert offenders == {}, (
        "Direct make_receipt(...) callers found outside results.py:\n"
        + "\n".join(
            f"  {file}:{lineno}: {src}"
            for file, hits in offenders.items()
            for lineno, src in hits
        )
        + "\n\nMigrate to receipt_for(carrier, mask=..., n_voxels=..., params=XxxParams(...))."
    )


def test_known_opparams_subclasses_are_importable():
    """Every advertised OpParams subclass exists and is a frozen dataclass."""
    import dataclasses

    from neuroim import results

    for name in _KNOWN_OPPARAMS:
        cls = getattr(results, name, None)
        assert cls is not None, f"{name} is not exported by neuroim.results"
        assert dataclasses.is_dataclass(cls), f"{name} is not a dataclass"
        assert cls.__dataclass_params__.frozen, f"{name} must be frozen"


def test_searchlight_params_requires_radius():
    """``SearchlightParams.radius`` is required — the S03 PAIN-1 regression."""
    from neuroim.results import SearchlightParams

    SearchlightParams(method_name="ok", radius=4.5)
    with pytest.raises(ValueError, match="radius"):
        SearchlightParams(method_name="ok")

"""Tests for the deprecated install aliases ``neuroimpy`` and ``pyneuroim``.

Per D10 in consensus sticky post-01KRKFEWY2: ``neuroim`` is canonical; the
two alias packages are pure re-export shims that emit a ``DeprecationWarning``
on import and are scheduled for removal after 0.3.

These tests guard:
- The shims emit DeprecationWarning on import.
- Every public symbol on the alias resolves to the same object as on ``neuroim``.
- The alias does not introduce any symbol that ``neuroim`` does not have.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


def _reimport(name: str):
    """Force a fresh import to trigger the module-level DeprecationWarning."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.parametrize("alias", ["neuroimpy", "pyneuroim"])
def test_alias_emits_deprecation_warning(alias):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reimport(alias)
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any(alias in m and "neuroim" in m for m in msgs), (
        f"expected DeprecationWarning naming {alias!r} and neuroim; got {msgs}"
    )


@pytest.mark.parametrize("alias", ["neuroimpy", "pyneuroim"])
def test_alias_reexports_canonical_symbols(alias):
    import neuroim

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mod = importlib.import_module(alias)

    # Spot-check a handful of canonical public symbols if present.
    for name in ("NeuroSpace", "NeuroVol", "NeuroVec", "read_image"):
        if hasattr(neuroim, name):
            assert hasattr(mod, name), f"{alias} missing {name}"
            assert getattr(mod, name) is getattr(neuroim, name), (
                f"{alias}.{name} is not the same object as neuroim.{name}"
            )


@pytest.mark.parametrize("alias", ["neuroimpy", "pyneuroim"])
def test_alias_does_not_introduce_extra_public_symbols(alias):
    import neuroim

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mod = importlib.import_module(alias)

    neuroim_public = {n for n in dir(neuroim) if not n.startswith("_")}
    alias_public = {n for n in dir(mod) if not n.startswith("_")}
    extra = alias_public - neuroim_public
    assert not extra, f"{alias} introduces symbols not in neuroim: {sorted(extra)}"

"""Package-root compatibility shims for historical import names."""

import importlib
import sys
from pathlib import Path

import pytest

import neuroim


def _fresh_alias_import(name):
    sys.modules.pop(name, None)
    with pytest.warns(DeprecationWarning, match="import neuroim instead"):
        return importlib.import_module(name)


@pytest.mark.parametrize("alias_name", ["neuroimpy", "pyneuroim"])
def test_deprecated_alias_roots_reexport_canonical_symbols(alias_name):
    alias = _fresh_alias_import(alias_name)

    assert alias.NeuroSpace is neuroim.NeuroSpace
    assert alias.DenseNeuroVol is neuroim.DenseNeuroVol
    assert alias.__version__ == neuroim.__version__


@pytest.mark.parametrize("alias_name", ["neuroimpy", "pyneuroim"])
def test_alias_roots_do_not_expose_symbols_outside_neuroim(alias_name):
    alias = _fresh_alias_import(alias_name)

    alias_public = {name for name in dir(alias) if not name.startswith("_")}
    neuroim_public = {name for name in dir(neuroim) if not name.startswith("_")}

    assert alias_public - neuroim_public == set()


def test_alias_root_shims_are_identical():
    root = Path(__file__).resolve().parents[1]
    neuroimpy_init = root / "src" / "neuroimpy" / "__init__.py"
    pyneuroim_init = root / "src" / "pyneuroim" / "__init__.py"

    assert neuroimpy_init.read_text() == pyneuroim_init.read_text()


def test_deprecated_package_names_declared_in_pyproject():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()

    assert "[tool.neuroim.deprecated-package-names]" in text
    assert "neuroimpy" in text
    assert "pyneuroim" in text
    assert "removal planned after 0.3" in text

"""Regression: the package must not ship a module named ``typing``.

A package-local ``neuroim/typing.py`` shadows the standard-library
``typing`` when the package directory itself is early on ``sys.path``
(editable installs, some tooling layouts): ``from typing import ...``
then resolves to the package module and import fails.

The fix was renaming that module to ``neuroim.protocols``. These tests
pin the property so it cannot silently regress.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "src" / "neuroim"


def test_no_package_local_typing_module():
    """``neuroim.typing`` must not be importable as a submodule."""
    assert (PKG_DIR / "typing.py").exists() is False, (
        "src/neuroim/typing.py reintroduced — it shadows stdlib typing"
    )
    assert importlib.util.find_spec("neuroim.typing") is None


def test_protocols_module_carries_the_surface():
    """The renamed module exposes the coordinate + Protocol surface."""
    from neuroim import protocols

    for name in (
        "VoxelCoord",
        "WorldCoord",
        "voxel_coord",
        "world_coord",
        "NeuroVolLike",
        "NeuroVecLike",
        "MaskLike",
    ):
        assert name in protocols.__all__
        assert hasattr(protocols, name)


def test_stdlib_typing_wins_with_package_dir_on_syspath():
    """With the package dir first on sys.path, ``import typing`` is stdlib.

    This reproduces the exact failure layout. It must succeed and resolve
    to the standard library, not to anything under the package tree.
    """
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import typing\n"
        "assert hasattr(typing, 'TYPE_CHECKING'), 'not stdlib typing'\n"
        "assert 'neuroim' not in (getattr(typing, '__file__', '') or ''), typing.__file__\n"
        "from typing import Any, Protocol  # would crash if shadowed\n"
        "print('ok')\n"
    ) % PKG_DIR
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"

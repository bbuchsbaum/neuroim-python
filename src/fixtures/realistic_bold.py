"""Re-export the draw-audit BOLD fixture for example-local tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_fixture_path = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "realistic_bold.py"
)
_spec = importlib.util.spec_from_file_location("_neuroim_realistic_bold", _fixture_path)
if _spec is None or _spec.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"Cannot load draw-audit fixture from {_fixture_path}")

_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

RealisticBOLD = _module.RealisticBOLD
make_realistic_bold = _module.make_realistic_bold
to_nibabel = _module.to_nibabel

__all__ = ["RealisticBOLD", "make_realistic_bold", "to_nibabel"]

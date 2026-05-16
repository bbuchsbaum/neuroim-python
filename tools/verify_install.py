#!/usr/bin/env python3
"""Reproducible clean-install proof (mote bd-01KRRJMS6T).

Proves the package installs and imports from a *clean* environment —
not from the repo's ``PYTHONPATH=src`` working tree. This is the check
that caught the undeclared-``pandas`` breakage in
``connected_components.py``: a clean ``pip install .`` succeeded but
``import neuroim`` crashed because pandas was a hidden hard dependency.

What it does, hermetically:
  1. create a throwaway virtualenv in a tempdir,
  2. ``pip install .`` from the project root (declared deps only),
  3. ``import neuroim`` and run a minimal typed-object smoke test,
  4. assert the curated public surface is intact (``read_image`` in
     ``__all__``),
  5. report PASS/FAIL and clean up.

Run via ``make verify-install`` (also chained from ``make
verify-evidence``). Exit 0 = clean install proven, non-zero = broken.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_PY = (3, 10)  # must match pyproject requires-python


def _find_interpreter() -> str | None:
    """A Python >= MIN_PY to build the clean venv with. The interpreter
    running ``make`` may be older (here it was 3.9, and the package
    correctly refuses to install on it) — so the proof must locate a
    conforming one rather than assume ``sys.executable`` qualifies."""
    if sys.version_info[:2] >= MIN_PY:
        return sys.executable
    for name in ("python3.13", "python3.12", "python3.11", "python3.10",
                 "python3", "python"):
        exe = shutil.which(name)
        if not exe:
            continue
        r = subprocess.run(
            [exe, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            try:
                major, minor = (int(x) for x in r.stdout.strip().split("."))
            except ValueError:
                continue
            if (major, minor) >= MIN_PY:
                return exe
    return None

SMOKE = (
    "import numpy as np, neuroim as ni; "
    "assert 'read_image' in ni.__all__, 'read_image missing from __all__'; "
    "sp = ni.NeuroSpace.from_affine(np.eye(4), (4, 4, 3)); "
    "v = ni.DenseNeuroVol(np.zeros((4, 4, 3)), sp); "
    "assert tuple(int(d) for d in v.shape) == (4, 4, 3), v.shape; "
    "print('clean-install import + smoke OK:', ni.__version__)"
)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    interp = _find_interpreter()
    if interp is None:
        print(f"FAIL: no Python >= {MIN_PY[0]}.{MIN_PY[1]} found to build "
              "the clean venv (the package requires it). Install a "
              "conforming interpreter or run via one.")
        return 1

    with tempfile.TemporaryDirectory(prefix="neuroim-cleaninstall-") as tmp:
        venv = Path(tmp) / "v"
        print(f"[1/3] creating clean venv at {venv} (via {interp})")
        r = _run([interp, "-m", "venv", str(venv)])
        if r.returncode != 0:
            print(r.stderr.strip())
            return 1

        py = venv / "bin" / "python"
        if not py.exists():  # Windows layout
            py = venv / "Scripts" / "python.exe"

        print("[2/3] pip install . (declared dependencies only)")
        r = _run([str(py), "-m", "pip", "install", "-q",
                  "--upgrade", "pip"])
        if r.returncode != 0:
            print(r.stderr.strip())
            return 1
        r = _run([str(py), "-m", "pip", "install", "-q", str(PROJECT_ROOT)])
        if r.returncode != 0:
            print("pip install FAILED:\n" + (r.stderr or r.stdout).strip())
            return 1

        print("[3/3] import neuroim + typed-object smoke")
        r = _run([str(py), "-c", SMOKE])
        if r.returncode != 0:
            print("CLEAN-INSTALL IMPORT FAILED:\n"
                  + (r.stderr or r.stdout).strip())
            return 1
        print(r.stdout.strip())

    print("PASS: clean-clone install proof reproducible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

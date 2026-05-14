"""Expose repository examples when running example tests with ``PYTHONPATH=src``."""

from __future__ import annotations

from pathlib import Path

_repo_examples = Path(__file__).resolve().parents[2] / "examples"
if _repo_examples.is_dir():
    __path__.append(str(_repo_examples))

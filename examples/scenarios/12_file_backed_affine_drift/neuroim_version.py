"""Rewrite: file-backed split-volume run through neuroim."""

from __future__ import annotations

from pathlib import Path

import neuroim as ni


def temporal_snr_from_split_run(
    paths: list[Path],
    mask: ni.LogicalNeuroVol,
) -> ni.DenseNeuroVol:
    """Compute masked temporal SNR from a file-backed vector."""
    vec = ni.FileBackedNeuroVec([str(path) for path in paths])
    return vec.temporal_snr(mask=mask)

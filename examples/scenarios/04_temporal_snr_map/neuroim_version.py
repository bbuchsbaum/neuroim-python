"""Rewrite: temporal SNR map using the neuroim public API.

After the S04 PAIN-1/PAIN-2 fixes landed, this is a one-liner: the
spatial contract is checked inside the public API, the reduction is
named, and the returned :class:`~neuroim.DenseNeuroVol` carries a
populated :class:`~neuroim.results.Receipt` in
``DenseNeuroVol.provenance``.
"""

from __future__ import annotations

import neuroim as ni


def temporal_snr_map(
    bold: ni.NeuroVec,
    mask: ni.LogicalNeuroVol,
) -> ni.DenseNeuroVol:
    """Return a masked 3-D temporal SNR volume with provenance."""
    return bold.temporal_snr(mask=mask)

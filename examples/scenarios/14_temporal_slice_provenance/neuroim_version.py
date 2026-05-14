"""Rewrite: drop pre-steady-state volumes, compute masked temporal SNR.

The aspiration is that ``bold[..., start:]`` returns a typed
``NeuroVec`` whose ``.provenance`` records ``TemporalSliceParams(start,
stop, step)``, and the downstream ``temporal_snr(mask=...)`` merges
that upstream Receipt — so a fresh-process reader of the written
``.nii.gz`` can recover ``method_name='temporal_slice+temporal_snr'``
or equivalent.

Today this aspiration is **not met**: ``DenseNeuroVec.__getitem__``
with a time-axis slice key falls through to NumPy and returns a bare
``ndarray`` (PAIN-13), and even re-wrapping into a ``DenseNeuroVec``
manually does not attach a ``TemporalSliceParams`` Receipt for the
downstream chain to merge (PAIN-14).

The neuroim_version function below implements the *aspirational*
shape so the acceptance tests can either pass once PAIN-13/PAIN-14
land, or remain ``xfail(strict=True)`` until they do. The body uses
only the current public surface — no monkey patching.
"""

from __future__ import annotations

import neuroim as ni


def temporal_snr_after_slice(
    bold: ni.NeuroVec,
    mask: ni.LogicalNeuroVol,
    *,
    start: int,
) -> ni.DenseNeuroVol:
    """Drop the first ``start`` timepoints, then compute masked tSNR.

    The aspiration: ``bold[..., start:]`` returns a typed ``NeuroVec``
    whose Receipt records the slice; ``temporal_snr`` merges that
    receipt into the returned ``DenseNeuroVol.provenance``.

    Today the slice returns a bare ``ndarray``; this function therefore
    re-wraps it into a ``DenseNeuroVec`` on a manually-derived space.
    The re-wrap loses any chance of an attached ``TemporalSliceParams``
    receipt — that's the PAIN-14 surface.
    """
    sliced = bold[..., start:]  # PAIN-13: returns ndarray, not NeuroVec
    if not isinstance(sliced, ni.NeuroVec):
        # Manual re-wrap so the rest of the pipeline has a typed input.
        # A fix for PAIN-13 should make this branch dead code.
        import numpy as np
        sliced_arr = np.asarray(sliced)
        spatial = bold.spatial_space
        from neuroim.axis import AxisSet4D, NamedAxis
        t_axis = NamedAxis("t", int(sliced_arr.shape[-1]))
        axes_4d = AxisSet4D(
            spatial.axes.i, spatial.axes.j, spatial.axes.k, t_axis
        )
        new_space = ni.NeuroSpace(
            dim=[int(d) for d in sliced_arr.shape],
            spacing=[float(s) for s in spatial.spacing] + [1.0],
            origin=[float(o) for o in spatial.origin] + [0.0],
            axes=axes_4d,
        )
        sliced = ni.DenseNeuroVec(sliced_arr, new_space)

    return sliced.temporal_snr(mask=mask)

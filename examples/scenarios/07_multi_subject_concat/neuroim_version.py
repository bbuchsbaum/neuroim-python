"""Rewrite: multi-subject concat through the neuroim public API.

The aspiration: ``NeuroVec.concat`` validates that the spatial
sub-affines of the inputs agree, raising a typed error on mismatch.
The merged result's :class:`Receipt` records the operation as
``"concat"`` and carries an input-space hash that reflects the
inputs' shared frame.

Today this aspiration is **not met** — see PAIN-8 in
:file:`REPORT.md` and tracker bd-01KRKTA660BNCJS20BB9F99VSK.  The
function below uses the public API as a careful user would and lets
the acceptance test falsify the missing contract.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import neuroim as ni


def concat_subjects_and_mean(
    subjects: Sequence[ni.NeuroVec],
    mask: ni.LogicalNeuroVol,
) -> np.ndarray:
    """Concatenate per-subject :class:`NeuroVec`\\s and return the masked mean.

    Parameters
    ----------
    subjects
        Iterable of 4-D :class:`~neuroim.NeuroVec`\\s.  Aspiration:
        ``concat`` raises when their spatial frames disagree (PAIN-8).
    mask
        3-D :class:`~neuroim.LogicalNeuroVol` in the subjects' shared
        spatial frame.
    """
    if len(subjects) < 2:
        raise ValueError("expected at least two subjects to concat")

    merged = subjects[0].concat(*subjects[1:])
    coords = np.argwhere(np.asarray(mask.data))
    roi = ni.ROICoords(coords, space=mask.space)
    return np.asarray(merged.series_roi(roi).values).mean(axis=1)

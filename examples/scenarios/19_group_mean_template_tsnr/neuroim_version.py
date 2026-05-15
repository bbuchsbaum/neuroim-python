"""Group-mean tSNR across subjects in template space — neuroim rewrite.

Per-subject the rewrite is the analysis we mean:

    resampled = ni.resample_vec(s.bold, template.space, interpolation=1)
    tsnr = resampled.temporal_snr(mask=template.mask)

Each subject's tSNR map carries a chained Receipt with
``method_name == "resample_vec(...)+temporal_snr"`` (Scenario 09 already
proves this).

The cross-subject reduce, however, has no first-class shape today.  The
rewrite has to:

  PAIN-1 (P1 ergonomic / API gap)
      There is no ``ni.group_mean(*maps)`` (or ``ni.mean_volumes(...)``)
      reducer.  ``ni.concat`` only stacks 4-D ``NeuroVec`` inputs along
      the time axis; it rejects 3-D ``NeuroVol`` inputs entirely, so it
      cannot be used to "stack and reduce" subject maps.  The rewrite
      hand-rolls ``np.stack`` + ``np.mean`` + ``DenseNeuroVol`` re-wrap.

  PAIN-2 (P2 contract)
      The group reduce needs a same-space gate across N inputs.  Today
      every caller writes a loop of ``verify.assert_same_space(maps[0],
      maps[i])``.  A first-class ``group_mean`` would own the contract.

  PAIN-3 (P2 provenance)
      Multi-input Receipt construction has no public helper.  The
      rewrite walks ``Receipt.merge`` across upstream Receipts manually
      and appends the new op's ``method_name``.  This works (S07 proves
      ``concat`` does the same internally for the time-axis case), but
      it is implementation detail leaking into user code.

When PAIN-1/2/3 land, the cross-subject step collapses to::

    group_map = ni.group_mean(*per_subject_tsnr)

and the resulting ``DenseNeuroVol``'s ``.provenance.method_name`` reads
``resample_vec(...)+temporal_snr+group_mean`` with both subjects'
upstream chains merged in.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

import neuroim as ni
from neuroim.results import (
    OpParams,
    Receipt,
    receipt_for,
)
from neuroim.verify import assert_same_space


# Plain classes with __slots__ rather than ``@dataclass`` because this
# module is loaded via ``importlib.util`` from a digit-prefixed directory
# (see scenarios/conftest.py).  Same Python-3.9 workaround as Scenario 18
# and the baseline lane in this scenario.


class TypedSubject:
    """A subject's typed BOLD + native brain mask."""

    __slots__ = ("name", "bold", "mask")

    def __init__(
        self,
        name: str,
        bold: "ni.DenseNeuroVec",
        mask: "ni.LogicalNeuroVol",
    ) -> None:
        self.name = name
        self.bold = bold
        self.mask = mask


class TypedTemplate:
    """A template's typed 3-D space and template-space brain mask."""

    __slots__ = ("space", "mask")

    def __init__(
        self, space: "ni.NeuroSpace", mask: "ni.LogicalNeuroVol"
    ) -> None:
        self.space = space
        self.mask = mask


def typed_from_nibabel(subjects, template) -> tuple[List[TypedSubject], TypedTemplate]:
    """Adapter so the test can drive both lanes from one synthesis call."""
    typed_subjects: List[TypedSubject] = []
    for s in subjects:
        bold_space = ni.NeuroSpace.from_affine(
            np.asarray(s.bold.affine), s.bold.shape
        )
        bold_vec = ni.DenseNeuroVec(
            np.asarray(s.bold.dataobj, dtype=np.float64), bold_space
        )
        mask_space = ni.NeuroSpace.from_affine(
            np.asarray(s.mask.affine), s.mask.shape[:3]
        )
        mask_vol = ni.LogicalNeuroVol(
            np.asarray(s.mask.dataobj).astype(bool), mask_space
        )
        typed_subjects.append(TypedSubject(name=s.name, bold=bold_vec, mask=mask_vol))

    template_space = ni.NeuroSpace.from_affine(
        np.asarray(template.affine), template.shape_3d
    )
    template_mask = ni.LogicalNeuroVol(
        np.asarray(template.mask.dataobj).astype(bool), template_space
    )
    return typed_subjects, TypedTemplate(space=template_space, mask=template_mask)


def _template_space_4d(
    template: TypedTemplate, nt: int
) -> "ni.NeuroSpace":
    """Build a 4-D template ``NeuroSpace`` for ``resample_vec`` from the
    template's 3-D space and the subject's number of timepoints.

    ``resample_vec`` requires a 4-D target ``NeuroSpace`` even though the
    spatial grid is the only thing that actually changes — a small
    incidental friction surfaced by this scenario but not its main
    thrust.  A first-class group reducer would not need this dance.
    """
    affine = np.asarray(template.space.trans, dtype=float)
    shape4 = (*tuple(int(d) for d in template.space.dim[:3]), int(nt))
    return ni.NeuroSpace.from_affine(affine, shape4)


def per_subject_tsnr(
    subject: TypedSubject, template: TypedTemplate
) -> ni.DenseNeuroVol:
    """Resample one subject into template grid and compute masked tSNR.

    Both steps already chain provenance natively (Scenario 09).  The
    returned map's ``.provenance.method_name`` is
    ``resample_vec(...)+temporal_snr``.
    """
    target_space = _template_space_4d(template, subject.bold.shape[3])
    resampled = ni.resample_vec(subject.bold, target_space, interpolation=1)
    return resampled.temporal_snr(mask=template.mask)


# ---------------------------------------------------------------------------
# PAIN-1: hand-rolled group reducer.  Replace with ``ni.group_mean`` once
# the API gap closes.  A first-class reducer would also ship a typed
# ``GroupReduceParams(OpParams)`` recording ``n_inputs``; we cannot define
# such a subclass inside this importlib-loaded module on Python 3.9, so
# the rewrite uses the base :class:`OpParams` and folds ``n_inputs`` into
# ``method_name`` instead.
# ---------------------------------------------------------------------------


def group_mean_volumes(maps: Sequence[ni.DenseNeuroVol]) -> ni.DenseNeuroVol:
    """Average N typed 3-D maps into one, threading multi-input provenance.

    PAIN-1: no first-class equivalent in ``neuroim`` today.
    PAIN-2: the same-space gate is a hand-written loop here; a first-class
    reducer would own it.
    PAIN-3: multi-input Receipt construction is a hand-written
    ``Receipt.merge`` walk; a first-class reducer would own it.
    """
    if len(maps) < 2:
        raise ValueError("group_mean_volumes requires at least two volumes")

    first = maps[0]
    # PAIN-2: contract loop.
    for v in maps[1:]:
        assert_same_space(first.space, v.space)

    stacked = np.stack(
        [np.asarray(v.data, dtype=np.float64) for v in maps], axis=0
    )
    mean_arr = stacked.mean(axis=0)
    out = ni.DenseNeuroVol(mean_arr, first.space)

    # PAIN-3: hand-walked multi-input Receipt merge.  Build a fresh
    # Receipt for this op, then fold every upstream Receipt into it via
    # ``Receipt.merge`` so the terminal ``method_name`` reads
    # ``<upstream_chain>+group_mean``.
    base = receipt_for(
        first.space,
        mask=None,
        n_voxels=int(np.prod(first.space.dim[:3])),
        params=OpParams(method_name=f"group_mean(n={len(maps)})"),
    )
    upstreams = [
        getattr(v, "provenance", None) for v in maps
    ]
    upstreams = [r for r in upstreams if isinstance(r, Receipt)]
    if upstreams:
        merged = upstreams[0]
        for r in upstreams[1:]:
            try:
                merged = merged.merge(r)
            except ValueError:
                # Subjects disagree on provenance space/mask; surface the
                # bare ``group_mean`` Receipt rather than fabricating a
                # chain that hides the conflict.
                merged = None
                break
        if merged is not None:
            try:
                base = merged.merge(
                    base, method_name=f"{merged.method_name}+{base.method_name}"
                )
            except ValueError:
                pass
    out.provenance = base
    return out


def neuroim_group_tsnr(
    subjects: Sequence[TypedSubject], template: TypedTemplate
) -> ni.DenseNeuroVol:
    """End-to-end neuroim form of the second-level workflow."""
    per_subject = [per_subject_tsnr(s, template) for s in subjects]
    return group_mean_volumes(per_subject)

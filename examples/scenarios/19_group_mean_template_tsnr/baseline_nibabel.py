"""Group-mean tSNR across subjects in template space — raw nibabel + numpy.

Canonical second-level QC workflow.  For each of N subjects:

    1. Resample the native-grid 4-D BOLD into a common template grid using
       ``nibabel.processing.resample_from_to`` with ``order=1`` (linear).
    2. Compute a masked temporal-SNR map in template space:
       ``mean_t / std_t`` masked to the template brain.

Then, across subjects, average the per-subject tSNR maps into one
group-level tSNR map.

The careful raw-`nibabel` user has to:

  * own the per-subject resample loop and the policy choice (``order=1``);
  * own the tSNR reduction (``np.mean`` / ``np.std`` along the time axis);
  * own the same-space contract for the group reduce — there is no
    library function that says "these N volumes share a frame";
  * write a sidecar manifest if they want a downstream reader to know
    which subjects fed the group map and how the data arrived in
    template space.

The scenario synthesizes deterministic two-subject data inline so the
acceptance test can drive both lanes from one fixture call.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


# Plain classes with __slots__ rather than ``@dataclass`` because this
# module is loaded via ``importlib.util`` from a digit-prefixed directory
# (see scenarios/conftest.py).  ``@dataclass`` field-resolution inspects
# ``sys.modules[cls.__module__]`` which is ``None`` for importlib-loaded
# modules on Python 3.9 — same workaround as Scenario 18.


class SubjectBundle:
    """One synthetic subject's native BOLD + native brain mask."""

    __slots__ = ("name", "bold", "mask")

    def __init__(
        self, name: str, bold: nib.Nifti1Image, mask: nib.Nifti1Image
    ) -> None:
        self.name = name
        self.bold = bold
        self.mask = mask


class TemplateBundle:
    """Common template grid and template-space brain mask."""

    __slots__ = ("affine", "shape_3d", "mask")

    def __init__(
        self,
        affine: np.ndarray,
        shape_3d: Tuple[int, int, int],
        mask: nib.Nifti1Image,
    ) -> None:
        self.affine = affine
        self.shape_3d = shape_3d
        self.mask = mask


def _ellipsoid_mask(shape: Tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    ii, jj, kk = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    cx, cy, cz = (nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2
    rx, ry, rz = nx * 0.42, ny * 0.42, nz * 0.42
    return (
        ((ii - cx) / rx) ** 2
        + ((jj - cy) / ry) ** 2
        + ((kk - cz) / rz) ** 2
    ) <= 1.0


def synthesize_subjects() -> Tuple[List[SubjectBundle], TemplateBundle]:
    """Build two deterministic subjects on different native grids + a template.

    Subject A is on a ``28 x 28 x 18``, ``2.5 x 2.5 x 3.0 mm`` grid.
    Subject B is on a ``26 x 26 x 16``, ``2.7 x 2.7 x 3.2 mm`` grid.
    The template is on a ``24 x 24 x 16``, ``3.0 x 3.0 x 3.5 mm`` grid.

    Each subject's BOLD is 30 timepoints of structured noise plus a
    deterministic boost in a small target ROI; the seed differs per
    subject so the per-subject tSNR maps are not identical and the group
    mean is non-trivial.
    """
    template_shape = (24, 24, 16)
    template_voxsize = (3.0, 3.0, 3.5)
    template_affine = np.diag([*template_voxsize, 1.0]).astype(float)
    template_mask_arr = _ellipsoid_mask(template_shape).astype(np.uint8)
    template_mask_img = nib.Nifti1Image(template_mask_arr, template_affine)
    template = TemplateBundle(
        affine=template_affine,
        shape_3d=template_shape,
        mask=template_mask_img,
    )

    subjects: List[SubjectBundle] = []
    for name, native_shape, native_voxsize, seed in (
        ("subA", (28, 28, 18), (2.5, 2.5, 3.0), 0xA1A1),
        ("subB", (26, 26, 16), (2.7, 2.7, 3.2), 0xB2B2),
    ):
        nt = 30
        rng = np.random.default_rng(seed)
        affine = np.diag([*native_voxsize, 1.0]).astype(float)
        nx, ny, nz = native_shape
        noise = rng.standard_normal((nx, ny, nz, nt))
        # Deterministic mean-baseline (so tSNR is finite across the brain)
        baseline = 100.0 + 5.0 * _ellipsoid_mask(native_shape).astype(float)[..., None]
        # Per-subject target ROI: small bump in mean signal — drives tSNR.
        cx, cy, cz = nx // 2, int(ny * 0.6), int(nz * 0.5)
        baseline[cx - 1 : cx + 2, cy - 1 : cy + 2, cz - 1 : cz + 2, :] += 8.0
        bold_data = baseline + noise
        bold_img = nib.Nifti1Image(bold_data.astype(np.float64), affine)
        mask_arr = _ellipsoid_mask(native_shape).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_arr, affine)
        subjects.append(SubjectBundle(name=name, bold=bold_img, mask=mask_img))

    return subjects, template


def resample_bold_to_template(
    bold_img: nib.Nifti1Image, template: TemplateBundle, *, order: int = 1
) -> nib.Nifti1Image:
    """Resample a native 4-D BOLD into the template grid frame-by-frame."""
    nt = bold_img.shape[3]
    out = np.empty((*template.shape_3d, nt), dtype=np.float64)
    target = (template.shape_3d, template.affine)
    for t in range(nt):
        frame = nib.Nifti1Image(
            np.asarray(bold_img.dataobj[..., t], dtype=np.float64),
            bold_img.affine,
        )
        resampled = resample_from_to(frame, target, order=order)
        out[..., t] = np.asarray(resampled.dataobj, dtype=np.float64)
    return nib.Nifti1Image(out, template.affine)


def temporal_snr_masked(
    bold_img: nib.Nifti1Image, mask_img: nib.Nifti1Image
) -> nib.Nifti1Image:
    """Per-voxel temporal SNR (mean / std along time), zeroed outside the mask."""
    if bold_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(
            f"mask shape {mask_img.shape[:3]} != bold shape {bold_img.shape[:3]}"
        )
    if not np.allclose(bold_img.affine, mask_img.affine):
        raise ValueError("mask affine does not match bold affine")
    data = np.asanyarray(bold_img.dataobj, dtype=np.float64)
    mask = np.asanyarray(mask_img.dataobj).astype(bool)
    mean_t = data.mean(axis=3)
    std_t = data.std(axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(std_t > 0, mean_t / std_t, 0.0)
    tsnr = np.where(mask, tsnr, 0.0)
    return nib.Nifti1Image(tsnr.astype(np.float64), bold_img.affine)


def group_mean_volumes(maps: Sequence[nib.Nifti1Image]) -> nib.Nifti1Image:
    """Average a list of 3-D maps that all share the same frame.

    Caller is responsible for confirming the same-space contract; nibabel
    has no first-class function that does this for a list of volumes.
    The careful user writes the loop below by hand.
    """
    if len(maps) < 2:
        raise ValueError("group_mean_volumes requires at least two volumes")
    first = maps[0]
    for v in maps[1:]:
        if v.shape[:3] != first.shape[:3]:
            raise ValueError(
                f"shape mismatch: {v.shape[:3]} != {first.shape[:3]}"
            )
        if not np.allclose(v.affine, first.affine):
            raise ValueError("affine mismatch between subject maps")
    stacked = np.stack(
        [np.asarray(v.dataobj, dtype=np.float64) for v in maps], axis=0
    )
    return nib.Nifti1Image(stacked.mean(axis=0), first.affine)


def baseline_group_tsnr(
    subjects: Sequence[SubjectBundle], template: TemplateBundle
) -> nib.Nifti1Image:
    """End-to-end raw-nibabel form of the second-level workflow.

    ``resample_from_to`` per subject + per-subject masked tSNR + manual
    same-space group reduce.  No provenance threading; the careful user
    would have to write a sidecar manifest naming the subjects, the
    interpolation policy, and the mask.
    """
    per_subject_tsnr = []
    for s in subjects:
        resampled = resample_bold_to_template(s.bold, template, order=1)
        tsnr = temporal_snr_masked(resampled, template.mask)
        per_subject_tsnr.append(tsnr)
    return group_mean_volumes(per_subject_tsnr)

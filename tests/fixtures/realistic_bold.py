"""Realistic BOLD-like synthetic fixture for the ME-3 draw audit.

The existing :file:`golden_tests/fixtures/tiny_bold.nii.gz` is 8x8x4x10 noise
— too small to stress receipt provenance, peak memory on file-backed reads,
or to make a thin-wrapper claim falsifiable.  This module ships a
deterministic generator (no binary check-in) that produces a 32x32x24x40
float64 BOLD volume with a structured signal, a paired brain-shaped mask,
a deliberately mis-oriented variant of the mask (LR-flipped affine), and a
behavioural regressor that correlates strongly inside a target ROI.

Consumed by:
- :mod:`tests.test_fixture_realistic_bold` for sanity tests.
- The forthcoming ``examples/draw_audit/`` baseline/rewrite pair (ME-3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import nibabel as nib

from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vec import DenseNeuroVec
from neuroim.neuro_vol import DenseNeuroVol, LogicalNeuroVol


@dataclass(frozen=True)
class RealisticBOLD:
    """Synthetic BOLD bundle for the draw audit."""

    bold: DenseNeuroVec
    mask: LogicalNeuroVol
    rotated_mask: LogicalNeuroVol
    regressor: np.ndarray
    target_roi_centers: np.ndarray


def make_realistic_bold(
    *,
    shape: Tuple[int, int, int, int] = (32, 32, 24, 40),
    voxsize: Tuple[float, float, float, float] = (3.0, 3.0, 3.5, 2.0),
    seed: int = 0xBEAD,
) -> RealisticBOLD:
    """Build a deterministic realistic BOLD fixture from a seed.

    Parameters
    ----------
    shape : tuple of int
        ``(nx, ny, nz, nt)``.  Defaults to ``(32, 32, 24, 40)`` so the
        float64 array is ~7.5 MB — large enough to make accidental full
        materialization observable, small enough for CI.
    voxsize : tuple of float
        ``(sx, sy, sz, tr)`` in mm and seconds.
    seed : int
        Reproducibility seed for the noise generator.
    """
    nx, ny, nz, nt = shape
    sx, sy, sz, _tr = voxsize
    rng = np.random.default_rng(seed)

    affine = np.diag([sx, sy, sz, 1.0]).astype(float)
    space = NeuroSpace.from_affine(affine, shape)
    mask_space = NeuroSpace.from_affine(affine, (nx, ny, nz))

    grid = np.mgrid[:nx, :ny, :nz].astype(float)
    cx, cy, cz = (nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2
    rx, ry, rz = nx * 0.35, ny * 0.40, nz * 0.40
    mask_data = (
        ((grid[0] - cx) / rx) ** 2
        + ((grid[1] - cy) / ry) ** 2
        + ((grid[2] - cz) / rz) ** 2
        <= 1.0
    )
    mask = LogicalNeuroVol(mask_data, mask_space)

    flipped_affine = affine.copy()
    flipped_affine[:, 0] = -flipped_affine[:, 0]
    rotated_space = NeuroSpace.from_affine(flipped_affine, (nx, ny, nz))
    rotated_mask = LogicalNeuroVol(mask_data, rotated_space)

    t = np.arange(nt, dtype=float)
    drift = 0.05 * np.sin(2.0 * np.pi * t / 30.0)[None, None, None, :]
    noise = rng.standard_normal(shape)

    cx_t, cy_t, cz_t = nx // 2, int(ny * 0.7), int(nz * 0.7)
    target = (
        slice(cx_t - 1, cx_t + 2),
        slice(cy_t - 1, cy_t + 2),
        slice(cz_t - 1, cz_t + 2),
    )
    regressor = np.zeros(nt, dtype=float)
    regressor[5:15] = 1.0
    regressor[25:35] = 1.0
    regressor = regressor - regressor.mean()
    norm = float(np.linalg.norm(regressor)) or 1.0
    regressor /= norm

    signal = np.zeros(shape, dtype=float)
    signal[target] = 2.0 * regressor[None, None, None, :]

    bold_data = drift + noise + signal
    bold = DenseNeuroVec(bold_data, space)

    xs, ys, zs = np.meshgrid(
        np.arange(cx_t - 1, cx_t + 2),
        np.arange(cy_t - 1, cy_t + 2),
        np.arange(cz_t - 1, cz_t + 2),
        indexing="ij",
    )
    target_coords = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=-1)

    return RealisticBOLD(
        bold=bold,
        mask=mask,
        rotated_mask=rotated_mask,
        regressor=regressor,
        target_roi_centers=target_coords,
    )


def to_nibabel(fixture: RealisticBOLD) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Project the typed fixture to nibabel images for the baseline lane."""
    bold_affine = np.asarray(fixture.bold.space.trans)[:4, :4]
    mask_affine = np.asarray(fixture.mask.space.trans)[:4, :4]
    bold_img = nib.Nifti1Image(
        np.asarray(fixture.bold.data, dtype=np.float64),
        bold_affine,
    )
    mask_img = nib.Nifti1Image(
        np.asarray(fixture.mask.data, dtype=np.uint8),
        mask_affine,
    )
    return bold_img, mask_img


def make_atlas(
    fixture: RealisticBOLD,
    *,
    bins: Tuple[int, int, int] = (3, 3, 2),
) -> DenseNeuroVol:
    """Build a deterministic integer-labelled atlas from the fixture's mask.

    The brain-shaped elliptical mask is partitioned into roughly equal
    cuboid bins along ``(i, j, k)``.  Cells that fall outside the mask are
    labelled 0 (background); inside-mask cells receive sequential 1-based
    labels.  The result is a ``DenseNeuroVol`` whose ``data`` is ``int32``
    and whose space matches ``fixture.mask.space``.

    Parameters
    ----------
    fixture : RealisticBOLD
        The fixture produced by :func:`make_realistic_bold`.
    bins : tuple of int, optional
        Number of bins along ``(i, j, k)``.  Default ``(3, 3, 2)`` yields up
        to 18 cells; only those that overlap the brain-shaped mask receive
        non-zero labels, so the realised parcel count is typically 8-12.
    """
    mask_arr = np.asarray(fixture.mask.data, dtype=bool)
    space = fixture.mask.space
    shape = mask_arr.shape
    bx, by, bz = bins

    edges_i = np.linspace(0, shape[0], bx + 1, dtype=int)
    edges_j = np.linspace(0, shape[1], by + 1, dtype=int)
    edges_k = np.linspace(0, shape[2], bz + 1, dtype=int)

    cell_id = np.zeros(shape, dtype=np.int32)
    for ci in range(bx):
        for cj in range(by):
            for ck in range(bz):
                bin_id = ci * (by * bz) + cj * bz + ck + 1
                cell_id[
                    edges_i[ci] : edges_i[ci + 1],
                    edges_j[cj] : edges_j[cj + 1],
                    edges_k[ck] : edges_k[ck + 1],
                ] = bin_id

    masked = np.where(mask_arr, cell_id, 0)
    # Remap to sequential 1..N labels (drop empty cells).
    unique = np.unique(masked)
    unique = unique[unique != 0]
    remap = {int(old): int(new) for new, old in enumerate(unique, start=1)}
    out = np.zeros(shape, dtype=np.int32)
    for old, new in remap.items():
        out[masked == old] = new
    return DenseNeuroVol(out, space)


def make_rotated_atlas(atlas: DenseNeuroVol) -> DenseNeuroVol:
    """Return ``atlas`` re-wrapped in an LR-flipped space.

    Used by the PAIN-2 attack: integer labels are byte-identical, but the
    affine differs from the BOLD's by a flipped x-axis.  A safe extraction
    API must reject this via ``assert_same_space``.
    """
    flipped = np.asarray(atlas.space.trans, dtype=float).copy()
    flipped[:, 0] = -flipped[:, 0]
    rotated_space = NeuroSpace.from_affine(flipped, atlas.shape)
    return DenseNeuroVol(np.asarray(atlas.data, dtype=np.int32).copy(), rotated_space)

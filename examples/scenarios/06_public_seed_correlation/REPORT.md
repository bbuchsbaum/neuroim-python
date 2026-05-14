# Scenario 06 Report: Public Seed-To-Voxel Correlation

## Workflow

This is the recognizable Nilearn-adjacent seed-connectivity workflow: extract a
seed time series from a 4-D BOLD image, correlate it with every in-mask voxel,
and write a 3-D correlation map.

## Baseline

The raw `nibabel`+`numpy` baseline must manually:

- compare BOLD and mask shape;
- compare BOLD and mask affine;
- invert the BOLD affine for the seed world coordinate;
- round and bounds-check the seed voxel;
- extract masked voxel time series;
- scatter the correlations back into a 3-D map.

## neuroim Rewrite

The neuroim rewrite uses:

- `NeuroVec.from_nibabel` and `NeuroVol.from_nibabel` for typed carriers;
- `bold.space.compatible_with(mask.space)` for the spatial contract;
- `bold.series_at_world(seed_xyz)` for seed extraction with OOB checking;
- `bold.series_roi(roi)` for validated time-by-voxel extraction;
- `NeuroVol.from_array(..., coords=...)` for scattering ROI values back to a
  spatial map;
- a `Receipt` on the derived map so `to_nibabel()` carries provenance through
  the NIfTI extension path.

## Verdict

This is a credibility-gate win rather than a bespoke toy win. The computation
is the same correlation math as the baseline, but neuroim makes the spatial
contract explicit and reusable, and the derived 3-D map can carry a receipt
across file handoff.

The runnable test checks numeric parity, output affine parity, receipt
presence, and the mismatched-mask affine failure mode.

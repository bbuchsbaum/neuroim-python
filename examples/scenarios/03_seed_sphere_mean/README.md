# Scenario 03 — Seed-Sphere Mean

> Compute the **mean BOLD time series** inside an 8 mm spherical ROI
> centred at a world-mm coordinate `(x, y, z)`.

This is the canonical "seed time-series" of functional-connectivity
analysis. It's the smallest task that:

1. exercises a **multi-voxel** ROI built from a *world-mm* radius,
2. exercises the `NeuroVec.series_roi_world(mni, radius)` API that
   landed during Scenario 01,
3. and lets the comparison check whether the rewrite's `Receipt`
   carries useful provenance about the ROI parameters.

> Slot history: originally drafted as scenario 02 against a different
> task slot; renumbered to 03 once another agent landed the
> `02_roi_mean_timeseries` scenario in the same slot. See the orphan
> stubs under `02_seed_sphere_mean/` for the breadcrumb.

## Inputs

- `bold` — 4-D BOLD image (`nibabel.Nifti1Image` in the baseline,
  `neuroim.NeuroVec` in the rewrite).
- `mni_xyz` — world-mm coordinate, length-3.
- `radius_mm` — sphere radius in mm.

## Output

- A 1-D time series of length `T`: the mean across all voxels in the
  sphere, per time point. The "typed" rewrite variant returns the full
  `ROIExtractionResult` so callers can read coords, per-voxel values,
  and the provenance `Receipt`.

## Behaviour at the edges

| Case | Required behaviour |
|---|---|
| Sphere fits inside the grid | Return the mean across in-sphere voxels. |
| Centre maps off the grid | Raise `ValueError` mentioning "outside the image grid". |
| `radius_mm == 0` | Collapse to the single nearest voxel. |
| `bold` is not 4-D | Raise `ValueError`. |

The baseline hand-codes a spacing-weighted distance map; the rewrite
gets the sphere from `series_roi_world`.

## Fixture

`tests/fixtures/realistic_bold.py::make_realistic_bold()`. Target
signal at voxel `(16, 22, 16)` ≡ world `(48.0, 66.0, 56.0)` mm.
An 8 mm sphere on the fixture's (3.0, 3.0, 3.5) mm voxels contains
**73 voxels** (matches between hand-mask and `series_roi_world`).

## Run

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    examples/scenarios/test_s03_seed_sphere_mean.py -q
```

See [`REPORT.md`](REPORT.md) for the verdict and surfaced pain points.

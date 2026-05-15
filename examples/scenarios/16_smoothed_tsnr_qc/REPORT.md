# Scenario 16 - Smoothed Temporal-SNR QC Map

## Task

Compute a masked temporal-SNR QC map from a 4-D BOLD run, smooth the map at a
requested FWHM in millimetres, and return simple summary values.

## Verdict

**Safety, readability, and inspectability win.** The neuroim
workflow is the analysis we meant:

```python
tsnr = bold.temporal_snr(mask=mask)
smoothed = ni.gaussian_blur(tsnr, mask=mask, fwhm_mm=6.0)
```

The raw nibabel/scipy baseline is numerically equivalent on matched inputs,
but it has to hand-code the tSNR reduction, FWHM-to-voxel sigma conversion,
mask shape checks, smoothing-mask application, and summary bookkeeping.

The safety win is clear: a same-shape smoothing mask with the wrong affine is
accepted by the baseline and rejected by neuroim inside `gaussian_blur`.

## What It Tests

| Check | Why it matters |
|---|---|
| Numeric parity | The neuroim path produces the same smoothed tSNR map and QC summary as the raw baseline on matched inputs. |
| Spatial contract | Neuroim rejects a same-shape, wrong-affine smoothing mask. |
| Units | Smoothing uses FWHM in millimetres and respects voxel spacing. |
| Provenance | The terminal smoothed map carries a chained `temporal_snr+gaussian_blur` Receipt, including after NIfTI round trip. |

## Scorecard

| Axis | Baseline nibabel/scipy | neuroim |
|---|---|---|
| Lines / read-time | Manual array reduction, sigma conversion, masking, and summary. | Two named operations plus summary. |
| Safety | Shape checks only unless the caller remembers affine checks. | `temporal_snr` and `gaussian_blur` own same-space checks. |
| Inspectability | No on-object provenance. | Terminal smoothing Receipt records the full `temporal_snr+gaussian_blur` chain. |

## Pain Points Surfaced

### PAIN-1 (P1, closed): gaussian_blur does not chain upstream provenance

Original impact: `bold.temporal_snr(mask=mask)` produces a Receipt and
`gaussian_blur(tsnr, mask=mask, fwhm_mm=6)` produces a Receipt, but the latter
does not record that the input map came from `temporal_snr`. A downstream
reader sees `method_name == "gaussian_blur"` rather than
`"temporal_snr+gaussian_blur"`.

Resolution: `gaussian_blur` now passes `upstream=vol` into `receipt_for`.
Scenario 16 asserts `method_name == "temporal_snr+gaussian_blur"` and verifies
the chained Receipt survives writing and reading the smoothed NIfTI.

## Follow-ups

| Priority | Suggested title |
|---|---|
| closed | Chain upstream provenance through `gaussian_blur` |

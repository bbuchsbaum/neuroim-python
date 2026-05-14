# Scenario 12 — File-Backed Affine Drift

## Task

Compute a masked temporal-SNR map from a BOLD run stored as separate 3-D
NIfTI files, one file per time point. Refuse the run when one time point has
the same shape but a different affine.

## Verdict

**Originally falsified a mission-bearing backend-safety claim; now fixed.**
The aligned path shows that file-backed neuroim can match careful nibabel
output and return typed provenance. The adversarial path now rejects a shifted
same-shape time point by comparing each loaded volume's space against the
first volume's space.

| Axis | Baseline nibabel | Neuroim today | Expected neuroim |
|---|---|---|---|
| Lines / read-time | Manual load, shape check, affine check, stack, tSNR | One file-backed vector plus `temporal_snr(mask=...)` | Same |
| Safety | Refuses shifted time-point affine | Refuses shifted time-point affine | done |
| Inspectability | Returns anonymous `Nifti1Image` | Returns typed `DenseNeuroVol` with `Receipt` | Same, after validated inputs |

## Pain Points Surfaced

### PAIN-12 (P0) — closed: `FileBackedNeuroVec` validates per-volume affine consistency

**Impact.** Split-volume BOLD is a realistic file-backed workflow. If one
time point has a shifted affine but the same voxel shape, downstream temporal
reductions can mix spatial frames while still returning a plausible typed map
and receipt. This violates the mission claim that storage backends expose the
same spatial safety as dense in-memory vectors.

**Fix.** `FileBackedNeuroVec._load_volume` now compares each volume's
`NeuroSpace` to the first volume's space after the existing shape check.
It raises a clear `ValueError` naming the offending volume and the
affine/space mismatch.

**Priority.** P0 — mission-bearing silent spatial error in a named storage
backend.

## Follow-ups

| Pain | Priority | Suggested issue title |
|---|---:|---|
| PAIN-12 | P0 | closed |

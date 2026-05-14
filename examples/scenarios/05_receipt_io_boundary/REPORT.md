# Scenario 05 — receipts across the IO boundary: Report

> Write a derived map to disk, then ask in a **clean** Python process:
> *what produced this .nii.gz, from what input, with what mask, what
> method?*  This is the realistic collaborator-handoff path that ME-9
> in-memory composition does not yet cover.

## Verdict

**Originally falsified the mission claim that ME-9 alone is
sufficient; now fixed and guarded.**  In memory, ME-9 makes Receipts
compose through `concat → series_roi → searchlight`.  At the time the
scenario landed, the write boundary silently dropped the Receipt —
`SearchlightResult.to_nibabel()` produced a `Nifti1Image` with zero
header extensions, and `neuroim.read_image(path)` returned a bare
`DenseNeuroVol` with no `.provenance`.

**Filed as PAIN-6 (P0, mission-bearing); fixed in the same iteration.**
The Receipt now travels as a NIfTI 'comment' header extension
prefixed with the marker :data:`neuroim.results.RECEIPT_NIFTI_PREFIX`
(JSON-serialized); :func:`neuroim.read_image` rehydrates it onto
``vol.provenance``.  The acceptance test now runs without xfail.

## Scoreboard

Counts are AST function-body statements for the *write* path, plus a
separate column for the *recover provenance from disk* path.

| Axis | Baseline (nib + hand-rolled sidecar) | Neuroim (today) | Neuroim (after PAIN-6) |
|---|---:|---:|---:|
| Write derived map | 8 | **3** | **3** |
| Write provenance | + 7 manual fields in JSON | **0** (Receipt rides with `to_nibabel`) | **0** |
| Read provenance back | 1 (`json.load`) | **N/A — returns None** | 1 (`img.provenance`) |
| Provenance ↔ data agreement guaranteed by writer? | **no** (user must keep manifest in sync by hand) | **N/A** (no provenance to disagree) | **yes** (the writer threads the Receipt; the reader checks it) |
| What survives a clean-process round-trip | manual sidecar fields (if the user remembered to write them) | nothing | full `Receipt` |

The shape of the win is *not* "fewer lines" — the baseline is already
short.  The shape of the win is **the manifest is computed, not
hand-curated**: the upstream API can not produce a derived map whose
provenance disagrees with its data.  Today the API ships neither the
manifest nor the cross-check.

## What the API caught vs what it did not

| Case | Baseline | Neuroim (today) | Neuroim (after PAIN-6) |
|---|---|---|---|
| In-memory Receipt populated | n/a | yes (ME-9) | yes |
| Receipt embedded on write | n/a — sidecar JSON | **no** | yes (header extension or sidecar) |
| Receipt re-hydrated on read | sidecar JSON | **no** | yes |
| Manifest stays in sync with data | no — sidecar is hand-written | **vacuous** (no manifest) | yes — derived by writer |

## Pain points surfaced

### PAIN-6 (P0, mission-bearing) — Receipt does not survive write/read

`SearchlightResult.to_nibabel()` produces a `Nifti1Image` with **zero
header extensions**.  Every Receipt field — `method_name`,
`n_voxels`, `radius`, `input_space_hash`, `mask_hash` — is dropped at
the moment the derived map is materialized to a nibabel object.  A
`nib.save` + `nib.load` round trip carries nothing forward; a
`neuroim.read_image(path)` returns a `DenseNeuroVol` without a
`.provenance` attribute.

**Why it matters.**  MISSION.md decision rule 4 (`Receipts by
default — silent space/orientation/mask mismatches are caught at the
contract layer`) is true in memory after ME-9 and PAIN-5.  At the IO
boundary it is empirically false: the collaborator who only has the
file path cannot tell what produced the map, with what input volume,
what mask, or what method.

**Suggested fix.**  Embed the Receipt as a NIfTI header extension
(JSON-serialized; NIfTI ecode 6 'comment' is the safe default, or a
dedicated ecode if we want a discoverable namespace).  Add a paired
sidecar (`<name>.neuroim.json`) as a fallback for non-NIfTI formats
and for inspectability outside the API.  `neuroim.read_image` is the
re-hydration point: it should return a typed result (or attach
`.provenance` to the `DenseNeuroVol`) when the extension is present.

Same surface check should apply to:

- `ROIExtractionResult` (it does not even have `to_nibabel` /
  `map_to_volume` today — separate gap to file);
- `write_vol(NeuroVol)` when the NeuroVol carries `.provenance`;
- Scenario 04's temporal-reduction maps (PAIN-2 there is the same
  shape of bug at a different surface — they should be fixed
  together).

Tracker: **bd-01KRKR7SX4GKW1QZ9KF6G73ZWR (P0)**.

### PAIN-7 (P2) — `ROIExtractionResult` has no `map_to_volume` / `to_nibabel`

While probing the IO boundary, the scenario found that
`ROIExtractionResult` exposes neither `map_to_volume` nor
`to_nibabel`.  Both ship on `SearchlightResult`.  MISSION.md mentions
`result.map_to_volume()` as an example of the "numeric projection" by
which native API contracts are gated, so the absence is itself a
mission-doc inconsistency.

The scenario worked around this by using `SearchlightResult` as the
write source.  But a user who wants to write an ROI mean-time-series
map (a 3-D image of per-voxel temporal means) cannot do so through
the public API without manually building a `DenseNeuroVol` and losing
the Receipt.  This is the same family of bug as PAIN-6.

Not filed as P0; folding into PAIN-6's fix is the natural path.

# Quickstart

Four canonical short patterns. Extract a time series at an MNI coordinate,
mean a BOLD across a brain mask, save a derived map with provenance,
recover that provenance in a fresh process. The first two patterns are
shown side-by-side against raw `nibabel`+`numpy` so the contract delta
is on-screen.

The verdicts in each block are not adjectives — they are the line counts and
contract-coverage results recorded in the runnable scenarios under
[`examples/scenarios/`](../examples/scenarios). Each pattern links to its
scenario REPORT for the evidence.

---

## 1. Extract a time series at an MNI coordinate

Open a BOLD image and pull the voxel time series at a world-mm
coordinate. One line names the operation; out-of-bounds raises by
default.

**neuroim**

```python
import neuroim as ni

bold = ni.read_image("sub-01_bold.nii.gz")
ts = bold.series_at_world((0.0, -52.0, 26.0))   # raises if outside the grid
```

**raw `nibabel`+`numpy`**

```python
import nibabel as nib
import numpy as np

img = nib.load("sub-01_bold.nii.gz")
ijk = np.linalg.inv(img.affine) @ np.array([0.0, -52.0, 26.0, 1.0])
i, j, k = np.round(ijk[:3]).astype(int)
nx, ny, nz = img.shape[:3]
if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
    raise ValueError("MNI coord maps outside the image grid")
ts = img.get_fdata()[i, j, k, :]
```

> **Verdict** (Scenario 01 — MNI Spotlight,
> [`examples/scenarios/01_mni_spotlight/REPORT.md`](../examples/scenarios/01_mni_spotlight/REPORT.md)):
> the affine-inversion idiom — `np.linalg.inv(affine) @ [x, y, z, 1]`
> plus manual rounding and bounds check — collapses to a single named
> domain operation, and the user-facing function body drops from
> **10 → 1 statement**. `series_at_world` raises on out-of-bounds by
> default (an `out_of_bounds="zero"` opt-in is available); the
> raw-nibabel form silently returns garbage when the bounds check is
> omitted, which is the failure mode S01's PAIN-3 was filed against
> and is now closed.

---

## 2. Mean time series across a brain mask

Take a 4-D BOLD and a 3-D mask, pull every in-mask voxel's time series,
and average across voxels. The neuroim form checks the mask's spatial
frame against the data's inside the public API.

**neuroim**

```python
import numpy as np
import neuroim as ni

bold = ni.read_image("sub-01_bold.nii.gz")
mask = ni.read_image("sub-01_mask.nii.gz", type="vol")
roi  = ni.ROICoords(np.argwhere(np.asarray(mask.data)), space=mask.space)
mean_ts = bold.series_roi(roi).values.mean(axis=1)   # contract gate is inside series_roi
```

**raw `nibabel`+`numpy`**

```python
import nibabel as nib
import numpy as np

bold_img = nib.load("sub-01_bold.nii.gz")
mask_img = nib.load("sub-01_mask.nii.gz")
if bold_img.shape[:3] != mask_img.shape or not np.allclose(bold_img.affine, mask_img.affine):
    raise ValueError("mask doesn't match BOLD spatial frame")   # user must remember to write this
mean_ts = bold_img.get_fdata()[mask_img.get_fdata().astype(bool)].mean(axis=0)
```

> **Verdict** (Scenario 02 — ROI mean time series,
> [`examples/scenarios/02_roi_mean_timeseries/REPORT.md`](../examples/scenarios/02_roi_mean_timeseries/REPORT.md)):
> the rewrite ties on raw line count and wins on safety. The mask/affine
> compatibility check moves from a line the user has to remember to write
> into the contract layer of `series_roi`. The headline failure mode S02
> surfaced — a wrong-affine mask silently scattering values into the
> wrong voxels — is blocked at the API surface (**PAIN-5, P0, fixed**)
> and the same `"spatial contract mismatch"` shape covers seed-sphere,
> temporal-SNR, and (soon) multi-subject concat. The full S02 verdict —
> with the receipt, error handling, and contract test — is **11 → 5**
> in the user-facing function body.

`bold.series_roi(roi)` returns an `ROIExtractionResult` carrying `.values`,
`.coords`, `.space`, and `.provenance`. Calling `.values.mean(axis=1)` is
the analysis step; everything else is contract.

---

## 3. Save a derived map — provenance rides on the file

Run a mean-searchlight across a mask, then write the result. There is no
side-by-side block here: raw `nibabel` cannot produce a self-describing
derived map without a hand-curated sidecar.

```python
import numpy as np
import nibabel as nib
import neuroim as ni

bold = ni.read_image("sub-01_bold.nii.gz")
mask = ni.read_image("sub-01_mask.nii.gz", type="vol")
sl   = ni.searchlight_apply(mask, radius=4.5, method=np.mean, data=bold)
nib.save(sl.to_nibabel(), "sub-01_sl_mean.nii.gz")   # Receipt rides in NIfTI ecode-6 ('comment')
```

The `SearchlightResult` carries a `Receipt` with `method_name`,
`input_space_hash`, `mask_hash`, `radius`, `n_voxels`, and
`source_affine_hash`. `sl.to_nibabel()` embeds the JSON-serialized
Receipt into the NIfTI header as an `ecode 6` `comment` extension
prefixed `neuroim/receipt/v1:`. No sidecar JSON, no curation step.

---

## 4. Recover provenance from disk

Open the file in a fresh process. The Receipt comes back with it.

```python
import neuroim as ni

img = ni.read_image("sub-01_sl_mean.nii.gz")
print(img.provenance.method_name)         # the searchlight method's name
print(img.provenance.input_space_hash)    # links back to the source BOLD's space
print(img.provenance.mask_hash)           # links back to the mask used
print(img.provenance.radius)              # 4.5
```

> **Verdict** (Scenario 05 — Receipts across the IO boundary,
> [`examples/scenarios/05_receipt_io_boundary/REPORT.md`](../examples/scenarios/05_receipt_io_boundary/REPORT.md)):
> at the time the scenario landed, the write boundary silently dropped the
> Receipt — `to_nibabel()` produced a NIfTI with **zero header extensions**
> and `read_image(path)` returned a bare `DenseNeuroVol`. This was the
> falsifying observation against MISSION decision rule 4 ("Receipts by
> default — silent space/orientation/mask mismatches are caught at the
> contract layer, not in debugging"). The fix embeds the Receipt as a
> `comment` extension (NIfTI ecode 6) on write and re-hydrates it on read,
> so the collaborator who only has the `.nii.gz` path can answer *what
> produced this, from what input, with what mask, what method?* — without
> a hand-curated sidecar. The on-disk format is a public contract,
> readable by **10 lines of `nibabel` alone**, no `neuroim` import
> required — see [`docs/spec/receipt-nifti-extension.md`](spec/receipt-nifti-extension.md).

---

## When neuroim refuses — three failure modes raw nibabel permits

The four patterns above are the happy path.  The shorter case for
neuroim is the cases where raw `nibabel` + `numpy` silently returns
wrong-but-plausible bytes and `neuroim` raises a typed error before
the bad value reaches your analysis.

Each subsection below maps to a closed P0 in the scenarios suite, so
the assertions are not aspirational — they are backed by the suite at
HEAD.

### Wrong-space mask (`series_roi`)

A `mask` whose `affine` differs from the BOLD's gets a free pass in
raw nibabel — `bold[mask]` does not know the two are in different
spatial frames and scatters whatever bytes happen to be at the same
voxel indices.

```python
# Raw nibabel — silently scatters when mask is in a different space.
mean = bold_data[mask_in_wrong_space].mean(axis=0)
```

```python
# neuroim — raises through the contract layer before extracting.
roi = ni.ROICoords(np.argwhere(mask.data), space=wrong_space)
bold.series_roi(roi)        # ValueError: spatial contract mismatch
```

Backed by `examples/scenarios/02_roi_mean_timeseries/REPORT.md` (PAIN-5,
closed).

### World coordinate outside the grid (`series_at_world`)

A world-mm coordinate that maps to a negative voxel index gets
wrapped by NumPy's signed indexing — you get a plausible-looking time
series from the *opposite corner* of the image.

```python
# Raw nibabel — np.linalg.inv(affine) @ [x, y, z, 1] -> negative index,
# then data[i, j, k, :] silently returns voxel from the far corner.
ts = bold_data[i, j, k, :]
```

```python
# neuroim — bounds-checks before indexing and raises.
bold.series_at_world((0.0, -52.0, 26.0))  # IndexError on OOB
```

Backed by `examples/scenarios/01_mni_spotlight/REPORT.md` (PAIN-2,
closed).

### Per-volume affine drift (`FileBackedNeuroVec`)

A multi-run experiment stored as one 3-D file per timepoint is the
canonical place for an affine mismatch to slip in (e.g. one run was
reconstructed in a slightly different orientation).  Raw nibabel
stacks the data and trusts the user; the result is a 4-D volume whose
later timepoints come from a *different spatial frame* than the
earlier ones.

```python
# Raw nibabel — stacks regardless of affine drift across volumes.
stacked = np.stack([nib.load(p).get_fdata() for p in run_paths], axis=-1)
```

```python
# neuroim — checks affine consistency on volume load and raises.
vec = ni.FileBackedNeuroVec(run_paths)
vec.temporal_snr()   # ValueError: Volume N has inconsistent affine/space
```

Backed by `examples/scenarios/12_file_backed_affine_drift/REPORT.md`
(PAIN-12, closed).

---

## Run the four patterns locally

Two tiny fixtures ship with the repository and exercise the same code
paths above:

```python
import neuroim as ni

bold = ni.read_image("golden_tests/fixtures/tiny_bold.nii.gz")
mask = ni.read_image("golden_tests/fixtures/tiny_mask.nii.gz", type="vol")
```

The full scenario tests (which assert each verdict above against runnable
baselines) are at:

```bash
PYTHONPATH=src:tests:. python -m pytest examples/scenarios -q
```

## Next

- [Get started](get-started.qmd) — the same flow as a Quarto-rendered tour.
- [Receipt NIfTI extension spec](spec/receipt-nifti-extension.md) — the
  on-disk format for provenance, written as a public contract.
- [`examples/scenarios/README.md`](../examples/scenarios/README.md) — the
  runnable evidence behind every verdict above.

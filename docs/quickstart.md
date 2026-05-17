# Quickstart

Four short patterns to copy from: extract a time series at an MNI
coordinate, average a BOLD series over a brain mask, save a derived map
with provenance, and recover that provenance in a fresh process.

If you're coming from `nibabel` + `numpy`, the first two patterns also
show the raw equivalent side by side, so you can see which steps neuroim
folds into the call. Runnable versions of everything here live in the
[Evidence](evidence/index.qmd) section.

---

## 1. Extract a time series at an MNI coordinate

Open a BOLD image and pull the voxel time series at a world-millimetre
coordinate. One line names the operation; out-of-grid coordinates raise
by default.

**neuroim**

```python
import neuroim as ni

bold = ni.read_image("sub-01_bold.nii.gz")
ts = bold.series_at_world((0.0, -52.0, 26.0))   # raises if outside the grid
```

**raw `nibabel` + `numpy`**

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

What to notice: `series_at_world` keeps the affine inversion, rounding,
bounds check, and indexing together as one domain operation. In raw
NumPy, forgetting the bounds check can turn a negative voxel index into
a valid-looking sample from the opposite side of the array.

---

## 2. Mean time series across a brain mask

Take a 4-D BOLD and a 3-D mask, pull every in-mask voxel's time series,
and average across voxels. The neuroim form checks the mask's spatial
frame against the data inside the public API.

**neuroim**

```python
import numpy as np
import neuroim as ni

bold = ni.read_image("sub-01_bold.nii.gz")
mask = ni.read_image("sub-01_mask.nii.gz", type="vol")
roi  = ni.ROICoords(np.argwhere(np.asarray(mask.data)), space=mask.space)
mean_ts = bold.series_roi(roi).values.mean(axis=1)
```

**raw `nibabel` + `numpy`**

```python
import nibabel as nib
import numpy as np

bold_img = nib.load("sub-01_bold.nii.gz")
mask_img = nib.load("sub-01_mask.nii.gz")
if bold_img.shape[:3] != mask_img.shape or not np.allclose(bold_img.affine, mask_img.affine):
    raise ValueError("mask doesn't match BOLD spatial frame")
mean_ts = bold_img.get_fdata()[mask_img.get_fdata().astype(bool)].mean(axis=0)
```

What to notice: `series_roi` checks the ROI's `NeuroSpace` before it
extracts values. Raw nibabel can do the same, but only if the caller
remembers to compare both shape and affine before indexing.

`bold.series_roi(roi)` returns an `ROIExtractionResult` carrying
`.values`, `.coords`, `.space`, and `.provenance`. Calling
`.values.mean(axis=1)` is the analysis step; everything else is the
contract around it.

---

## 3. Save a derived map with provenance

Run a mean searchlight across a mask, then write the result. There is no
side-by-side raw block here: raw `nibabel` can write the array, but it
will not create a self-describing derived map unless you add your own
metadata convention.

```python
import numpy as np
import nibabel as nib
import neuroim as ni

bold = ni.read_image("sub-01_bold.nii.gz")
mask = ni.read_image("sub-01_mask.nii.gz", type="vol")
sl   = ni.searchlight_apply(mask, radius=4.5, method=np.mean, data=bold)
nib.save(sl.to_nibabel(), "sub-01_sl_mean.nii.gz")
```

The `SearchlightResult` carries a `Receipt` with `method_name`,
`input_space_hash`, `mask_hash`, `radius`, `n_voxels`, and
`source_affine_hash`. `sl.to_nibabel()` embeds the serialized Receipt
in the NIfTI header as a `comment` extension prefixed
`neuroim/receipt/v1:`. The file itself now records the operation, input
space, and mask used to produce it.

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

What to notice: a collaborator who only has the `.nii.gz` can inspect
what produced it, from which spatial frame, with which mask, and with
which method. The on-disk Receipt is also readable with plain nibabel;
the schema is described in the [Receipt NIfTI extension spec](spec/receipt-nifti-extension.md).

---

## Mistakes it turns into errors

The patterns above are the everyday path. This last section is optional
background: a few common mistakes where neuroim stops with a clear error
instead of returning plausible-looking wrong numbers. You don't need it
to start working — it's here so the failure modes aren't a surprise.

### Wrong-space mask (`series_roi`)

A mask whose affine differs from the BOLD's can still be applied by raw
NumPy because `bold_data[mask]` only sees indices. It does not know that
the same index may refer to a different physical location.

```python
# Raw nibabel — silently uses voxel indices from the wrong spatial frame.
mean = bold_data[mask_in_wrong_space].mean(axis=0)
```

```python
# neuroim — raises through the contract layer before extracting.
roi = ni.ROICoords(np.argwhere(mask.data), space=wrong_space)
bold.series_roi(roi)        # ValueError: spatial contract mismatch
```

### World coordinate outside the grid (`series_at_world`)

A world-millimetre coordinate that maps to a negative voxel index can be
wrapped by NumPy's signed indexing. The result is a plausible-looking
time series from the opposite side of the image.

```python
# Raw nibabel — a negative i, j, or k can still index the array.
ts = bold_data[i, j, k, :]
```

```python
# neuroim — bounds-checks before indexing and raises.
bold.series_at_world((0.0, -52.0, 26.0))  # ValueError if outside the grid
```

### Per-volume affine drift (`FileBackedNeuroVec`)

A multi-run experiment stored as one 3-D file per timepoint is a common
place for an affine mismatch to slip in. Raw nibabel will stack arrays
unless you add the consistency check yourself.

```python
# Raw nibabel — stacks regardless of affine drift across volumes.
stacked = np.stack([nib.load(p).get_fdata() for p in run_paths], axis=-1)
```

```python
# neuroim — checks affine consistency on volume load and raises.
vec = ni.FileBackedNeuroVec(run_paths)
vec.temporal_snr()   # ValueError if a volume has an inconsistent affine
```

For runnable comparisons behind these examples, see the
[Evidence](evidence/index.qmd) page.

---

## Run the four patterns locally

Two tiny fixtures ship with the repository and exercise the same code
paths above:

```python
import neuroim as ni

bold = ni.read_image("golden_tests/fixtures/tiny_bold.nii.gz")
mask = ni.read_image("golden_tests/fixtures/tiny_mask.nii.gz", type="vol")
```

The evidence suite exercises the examples and boundary checks:

```bash
PYTHONPATH=src:tests:. python -m pytest examples/scenarios -q
```

## Next

- [Get started](get-started.qmd) — the same flow as a Quarto-rendered tour.
- [Receipt NIfTI extension spec](spec/receipt-nifti-extension.md) — the
  on-disk format for provenance, written as a public contract.
- [Evidence](evidence/index.qmd) — runnable examples behind the tutorial claims.

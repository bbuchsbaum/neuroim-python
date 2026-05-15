# Scenario 11 -- Atlas-based per-parcel mean BOLD time series

**Task.** Given a 4-D BOLD volume and an integer-labelled 3-D parcellation
atlas (0 = background, 1..N = parcels), return the `(N, T)` matrix of
per-parcel mean BOLD time series.  This is the canonical first step in
most fMRI workflows downstream of preprocessing: rest-state functional
connectivity, atlas-driven task-fMRI ROI analysis, parcel-level
predictive modelling, network analyses.

**Why this scenario exists.** S02 covered one mask -> one mean.  Real
work overwhelmingly uses an atlas with many labels (Schaefer 100/200/400,
AAL, Yeo, Harvard-Oxford).  No other scenario exercises
`ClusteredNeuroVol` / `ClusteredNeuroVec`, and none surfaces the
atlas-in-foreign-space failure mode -- the same bug class S02 PAIN-5
demonstrated for masks, played out at the atlas boundary.

## Verdict

| Axis | Baseline | neuroim after PAIN-1/3/4/A4 |
|---|---|---|
| User-facing body statements | 7 | 1 (call site) |
| Same-space mismatch caught | no | yes (inside `parcel_means`) |
| Output carries provenance Receipt | no | yes (`method_name="parcel_means"`) |
| Atlas source metadata | no | yes (`atlas_provenance`) |
| Classes discoverable from `ni.__all__` | n/a | yes |

**After PAIN-1/3: neuroim now wins the call-site shape for this workflow.**
The rewrite collapses the hand-rolled atlas loop to one statement:
`bold.parcel_means(atlas)`.  That method owns the same-space gate and returns
a `ClusteredNeuroVec` with a provenance `Receipt`.  The typed-atlas path now
adds source metadata on the result: canonical source, delivery backend, DOI,
template space, image hash, and label-table hash.

**The bug class is real and silent.**  The baseline returns an `(N, T)`
matrix of the right shape on an LR-flipped atlas; the numbers are
wrong-but-plausible.  No exception, no warning.  This is the exact
PAIN-5 shape, just at the atlas surface instead of the mask surface.
The mission rule 4 contract has to extend to this case to keep its
claim honest.

## PAINs

### PAIN-1 (P1 ergonomic) -- closed: first-class parcel extraction API

`DenseNeuroVec.parcel_means(atlas)` now turns an integer-labelled
`DenseNeuroVol` or `ClusteredNeuroVol` into a `ClusteredNeuroVec`.
`ClusteredNeuroVec.from_neurovec(vec, cvol)` delegates to the same method
for users who discover the clustered container first.

### PAIN-2 (P0 mission) -- closed with PAIN-1

The same-space check now lives inside `parcel_means`.  A user who calls the
first-class API with a foreign-affine atlas gets the same `ValueError` the
manual scenario helper previously supplied.

### PAIN-3 (P1 provenance) -- closed: ClusteredNeuroVec carries Receipt

`ClusteredNeuroVec` now accepts an optional `.provenance` Receipt populated
by `parcel_means`.  The receipt records the BOLD input space, atlas integer
labels as `mask_hash`, `n_voxels == n_clusters`, and
`method_name = "parcel_means"`.

### PAIN-4 (P2 surface) -- closed: classes added to `ni.__all__`

```python
>>> 'ClusteredNeuroVec' in neuroim.__all__
True
>>> 'ClusteredNeuroVol' in neuroim.__all__
True
```

`ClusteredNeuroVol` is exported next to `LogicalNeuroVol` and
`ClusteredNeuroVec` next to `MappedNeuroVec` in `src/neuroim/__init__.py`.
`len(neuroim.__all__) == 40` (was 38).  `tests/test_public_api.py` stays
green: budget, dir(), star-import, and implementation-detail-leakage
checks all pass.

### A4 atlas module update -- closed: typed atlas object proves value add

Scenario 11 now wraps the synthetic integer label image in a
`neuroim.atlas.VolumetricAtlas` via `typed_schaefer_fixture()`.  The numeric
projection still matches the raw nibabel baseline, but the neuroim result now
also carries `atlas_provenance` with the Schaefer source policy:
ThomasYeoLab/CBIG as canonical source, DOI `10.1093/cercor/bhx179`,
fixture/TemplateFlow/CBIG as delivery backend, plus content hashes for the
label image and label table.  This is the value over nibabel: not just fewer
lines and better space checks, but an inspectable answer to "which atlas
artifact produced these parcel time series?"

## Fixture

Atlas built from the same `realistic_bold` fixture used by S02-S10 via
the new `make_atlas` / `make_rotated_atlas` helpers in
`tests/fixtures/realistic_bold.py`.  The atlas partitions the brain-shaped
elliptical mask into 18 cuboid bins (3 x 3 x 2), labelled 1..18.

## Acceptance state

`PYTHONPATH=src:tests:. python -m pytest tests/test_s11_atlas_parcel_timeseries.py`:

- **7 passed, 0 xfailed** -- happy-path numeric parity, typed-atlas
  source provenance, built-in
  same-space gate, baseline's silent acceptance of the affine mismatch,
  first-class `parcel_means`, `ClusteredNeuroVec` provenance, and the
  public-namespace export check.  All four scenario PAINs are closed.

## Source

- Baseline: `examples/scenarios/11_atlas_parcel_timeseries/baseline_nibabel.py`
- Rewrite: `examples/scenarios/11_atlas_parcel_timeseries/neuroim_version.py`
- Test: `tests/test_s11_atlas_parcel_timeseries.py`
- Atlas fixture helpers: `tests/fixtures/realistic_bold.py::make_atlas`,
  `make_rotated_atlas`.

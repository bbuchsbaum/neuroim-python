# Scenario 20 - Parcellated Functional-Connectivity Matrix (Connectome)

## Task

Given a 4-D BOLD run and an integer-labelled 3-D atlas in the same spatial
frame, build the `N x N` parcel-to-parcel Pearson correlation matrix (the
parcellated connectome). One step past Scenario 11's parcel time series: the
mean time courses are the intermediate, the connectivity matrix is the
deliverable.

## Verdict

**Safety + inspectability win, ties on lines.** The neuroim workflow is the
analysis we meant:

```python
parcels = bold.parcel_means(atlas)                       # same-space gate + Receipt
ts = parcel_timeseries_matrix(parcels)                   # (n_time, n_parcels)
connectome = np.corrcoef(ts, rowvar=False)               # N x N
```

The raw nibabel/numpy baseline is numerically identical on matched inputs, but
it overlays atlas labels onto the BOLD **by voxel index** and never consults
the affine. An LR-flipped atlas (same dims, different affine) therefore sails
through with no error — and is so affine-blind that the wrong-frame connectome
is byte-identical to the matched one. The spatial-frame disagreement leaves no
trace.

neuroim's `parcel_means` refuses the frame mismatch up front via
`assert_same_space`, before any correlation is computed, and the connectome
inherits the extraction's `parcel_means` Receipt so the matrix stays traceable
to the atlas and input space that produced it.

## What It Tests

| Check | Why it matters |
|---|---|
| Numeric parity | The neuroim path produces the same `(N, N)` connectome as the raw baseline on matched inputs. |
| Valid correlation matrix | The connectome is symmetric, unit-diagonal, and bounded in `[-1, 1]`. |
| Spatial contract | neuroim rejects a same-shape, wrong-affine atlas; the baseline silently returns a plausible matrix. |
| Provenance | The connectome carries the `parcel_means` Receipt (`method_name`, `n_voxels == n_parcels`). |
| Public-surface accessor | `cluster_ids` + `cluster_timeseries` reproduce `ClusteredNeuroVec.ts`. |

## Scorecard

| Axis | Baseline nibabel/numpy | neuroim |
|---|---|---|
| Lines / read-time | Manual label loop, mean reduction, and `np.corrcoef`. | `parcel_means` + a public accessor loop + `np.corrcoef`. Net wash on lines; the contract/provenance come for free. |
| Safety | Shape check only; affine is ignored, so a wrong-frame atlas is invisible. | `parcel_means` owns the same-space gate; a wrong-frame atlas raises before correlation. |
| Inspectability | No on-object provenance; the matrix is a bare ndarray. | The connectome carries the `parcel_means` Receipt (atlas + input-space lineage). |

## Pain Points Surfaced

### PAIN-1 (P3, open): no first-class connectome / correlation-matrix reducer

The parcel-to-parcel correlation is the deliverable of a very common workflow,
but there is no typed reducer for it — the caller hand-rolls `np.corrcoef`. A
`ClusteredNeuroVec.connectome()` (or `connectivity_matrix()`) returning a typed
result that threads the upstream `parcel_means` Receipt would make the matrix a
first-class, provenance-carrying object rather than a bare ndarray. Low
priority: `np.corrcoef` is one honest line, and the extraction it consumes is
already typed and gated.

### PAIN-2 (P3, open): parcel time-by-cluster matrix accessor is semi-public

Building the `(n_time, n_parcels)` matrix from the curated surface requires a
`cluster_ids` + `cluster_timeseries` loop. The typed `ClusteredNeuroVec.ts`
attribute holds exactly this matrix but is not advertised on the public method
surface (`dir(ClusteredNeuroVec)`) or in `ni.__all__`. A documented
`.timeseries_matrix()` accessor would remove the loop and the reach for a
semi-public attribute.

## Follow-ups

| Priority | Suggested title |
|---|---|
| P3 | Typed `ClusteredNeuroVec.connectome()` reducer that threads `parcel_means` provenance |
| P3 | Public `ClusteredNeuroVec.timeseries_matrix()` accessor for the `(n_time, n_parcels)` payload |

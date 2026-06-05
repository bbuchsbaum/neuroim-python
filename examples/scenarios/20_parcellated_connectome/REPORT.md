# Scenario 20 - Parcellated Functional-Connectivity Matrix (Connectome)

## Task

Given a 4-D BOLD run and an integer-labelled 3-D atlas in the same spatial
frame, build the `N x N` parcel-to-parcel Pearson correlation matrix (the
parcellated connectome). One step past Scenario 11's parcel time series: the
mean time courses are the intermediate, the connectivity matrix is the
deliverable.

## Verdict

**Safety + inspectability win, now also a lines win.** The neuroim workflow is
the analysis we meant, and after the S20 pain fixes it is two typed calls:

```python
parcels = bold.parcel_means(atlas)          # same-space gate + Receipt
connectome = parcels.connectome()           # typed ConnectomeResult, chained Receipt
# -> connectome.labels, connectome.matrix (N x N), connectome.provenance
```

The raw nibabel/numpy baseline is numerically identical on matched inputs, but
it overlays atlas labels onto the BOLD **by voxel index** and never consults
the affine. An LR-flipped atlas (same dims, different affine) therefore sails
through with no error — and is so affine-blind that the wrong-frame connectome
is byte-identical to the matched one. The spatial-frame disagreement leaves no
trace.

neuroim's `parcel_means` refuses the frame mismatch up front via
`assert_same_space`, before any correlation is computed, and the
`ConnectomeResult` chains the extraction's Receipt
(`parcel_means+connectome`) so the matrix stays traceable to the atlas and
input space that produced it.

## What It Tests

| Check | Why it matters |
|---|---|
| Numeric parity | The neuroim path produces the same `(N, N)` connectome as the raw baseline on matched inputs. |
| Valid correlation matrix | The connectome is symmetric, unit-diagonal, and bounded in `[-1, 1]`. |
| Spatial contract | neuroim rejects a same-shape, wrong-affine atlas; the baseline silently returns a plausible matrix. |
| Provenance | The `ConnectomeResult` chains a `parcel_means+connectome` Receipt (`n_voxels == n_parcels`). |
| First-class API | `ClusteredNeuroVec.connectome()` and `timeseries_matrix()` replace the hand-rolled loop + bare `np.corrcoef`. |

## Scorecard

| Axis | Baseline nibabel/numpy | neuroim |
|---|---|---|
| Lines / read-time | Manual label loop, mean reduction, and `np.corrcoef`. | `parcel_means().connectome()` — two typed calls; the contract/provenance come for free. |
| Safety | Shape check only; affine is ignored, so a wrong-frame atlas is invisible. | `parcel_means` owns the same-space gate; a wrong-frame atlas raises before correlation. |
| Inspectability | No on-object provenance; the matrix is a bare ndarray. | `ConnectomeResult` chains a `parcel_means+connectome` Receipt (atlas + input-space lineage). |

## Pain Points Surfaced

### PAIN-1 (P3, closed): no first-class connectome / correlation-matrix reducer

Original impact: the parcel-to-parcel correlation is the deliverable of a very
common workflow, but there was no typed reducer — the caller hand-rolled
`np.corrcoef` over a bare ndarray with no provenance.

Resolution: `ClusteredNeuroVec.connectome(metric="correlation"|"covariance")`
returns a typed `ConnectomeResult` (`labels`, `matrix`, `metric`,
`provenance`, `n_nodes`, optional `.to_dataframe()`) whose Receipt chains the
upstream extraction (`parcel_means+connectome`). `ConnectomeResult` is exported
at the package root.

### PAIN-2 (P3, closed): parcel time-by-cluster matrix accessor is semi-public

Original impact: building the `(n_time, n_parcels)` matrix from the curated
surface required a `cluster_ids` + `cluster_timeseries` loop, or reaching for
the undocumented `ClusteredNeuroVec.ts` attribute.

Resolution: `ClusteredNeuroVec.timeseries_matrix()` returns a copy of the
`(n_time, n_clusters)` payload, column-ordered by sorted cluster id — a
documented, mutation-safe public accessor.

## Follow-ups

| Priority | Suggested title |
|---|---|
| closed | Typed `ClusteredNeuroVec.connectome()` reducer that threads `parcel_means` provenance |
| closed | Public `ClusteredNeuroVec.timeseries_matrix()` accessor for the `(n_time, n_parcels)` payload |

# Scenario 20 - Parcellated Functional-Connectivity Matrix (Connectome)

Task: given a 4-D BOLD run and an integer-labelled 3-D atlas (0 = background,
`1..N` = parcels) in the same spatial frame, build the `N x N` parcel-to-parcel
Pearson correlation matrix — the parcellated *connectome* that underpins most
resting-state and task functional-connectivity analyses (the canonical Nilearn
`NiftiLabelsMasker` -> `ConnectivityMeasure` pipeline).

This is the natural next step after Scenario 11 (atlas parcel time series): the
parcel mean time courses are an intermediate, and the deliverable is the
connectivity matrix computed from them.

The workflow crosses three concerns that raw `nibabel`+`numpy` code makes every
caller reimplement:

- a per-parcel temporal reduction from 4-D BOLD to an `(n_time, n_parcels)`
  matrix, keyed by atlas label;
- a spatial contract: the atlas must live in the **same** frame as the BOLD,
  or every label-to-voxel assignment is silently wrong and the connectome is
  plausible-but-meaningless;
- provenance: the matrix should be traceable to the extraction that produced
  it (which atlas, which input space, what method).

The scenario asks whether neuroim keeps the connectome workflow readable while
owning the same-space gate and provenance that the bare-array path leaves to
the user.

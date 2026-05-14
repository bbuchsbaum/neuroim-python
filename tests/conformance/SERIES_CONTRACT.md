# VoxelSeriesStore Conformance Contract

This document captures the **observed agreed contract** for the `series()` and
related operations across the five `NeuroVec` backends in this repository:
`DenseNeuroVec`, `SparseNeuroVec`, `BigNeuroVec`, `FileBackedNeuroVec`, and
`MappedNeuroVec`.

It is *post-hoc specification* of behaviour the existing implementations agree
on, written as a prerequisite to the WP-9 `VoxelSeriesStore` protocol
extraction. WP-8 originally tracked divergences as strict xfails; WP-9 resolved
those divergences and made them part of the green protocol contract.

## Agreed contract — every backend passes these

### Shape

`backend.shape == (X, Y, Z, T)` for every backend on the canonical 8x8x4x10
fixture.

### `series(x, y, z)` — single voxel by 3D coordinate

Returns a 1-D array of shape `(T,)`. Values match `fixture[x, y, z, :]`.

### `series(int_linear_index)` — single voxel by linear index

Returns a 1-D array of shape `(T,)`. Linear-index ordering is Fortran
(`order="F"`), matching `np.ravel_multi_index(..., order="F")`.

### `series(np.ndarray, shape=(N, 3))` — many voxels by coordinate matrix

Returns a 2-D array of shape **`(T, N)`** — rows are time, columns are
voxels. Values match `fixture[coords[:, 0], coords[:, 1], coords[:, 2], :].T`.

This resolves the open question from discussion post `post-01KRKDRE`: every
backend, including Dense (which internally builds `(N, T)` then `.T`s) and
Big/FileBacked (which build `(T, N)` directly), produces the same `(T, N)`
shape.

### `series(np.ndarray, shape=(N,))` — many voxels by linear indices

Returns a 2-D array of shape **`(T, N)`**. Linear indices are interpreted in
Fortran order over the spatial dimensions.

### `__getitem__((x, y, z, t))` — single value

Returns the scalar `fixture[x, y, z, t]`.

### `__getitem__((slice(None), slice(None), slice(None), t))` — single volume

Returns a 3-D ndarray of shape `(X, Y, Z)` containing `fixture[..., t]`.
**No backend wraps this in `DenseNeuroVol` today.** WP-9 may unify this to
return a `DenseNeuroVol` so the result carries a `NeuroSpace`.

### `series_3d(x, y, z)` matches `series(x, y, z)`

The deprecated R-flavoured alias produces the same output as the canonical
3-arg form.

## Divergences resolved by WP-9

### 1. `.dtype` attribute is not uniform

Resolved: every backend now exposes a `.dtype` attribute that returns a
`numpy.dtype`. Dense, sparse, big, and mapped vectors derive it from their
storage without requiring callers to materialize the full array manually.

### 2. Out-of-bounds policy for `series(Nx3)` is not uniform

Resolved: sparse now matches the other backends for the pre-existing contract:
out-of-bounds coordinates in an `(N, 3)` matrix leave the corresponding output
column filled with zeros.

## Cross-references

- WP-8 (this work): `bd-01KRKEAGDVXH2HS6AD2RETZA25`
- WP-9 (storage protocol refactor): `bd-01KRKEBYZ5VD29JHGXGMSFH0GZ`
- Discussion sticky v2: `post-01KRKE0YY4` in `neuroim-python-pythonic-value`
- Original risk flag: `post-01KRKDREY...` in the same topic

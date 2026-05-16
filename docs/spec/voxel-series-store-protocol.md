# VoxelSeriesStore protocol — v1

This document specifies `VoxelSeriesStore`, the structural protocol
for 4-D voxel-series storage, and the adapter that conforms the
shipped containers to it. It is a public contract: a third-party
backend that satisfies the signatures and semantics here is a
conforming store and can be passed wherever the protocol is accepted.

The protocol is defined in `neuroim.storage`. It is
`@runtime_checkable`, so `isinstance(obj, VoxelSeriesStore)` is a
valid (structural) membership test.

---

## The protocol

```python
@runtime_checkable
class VoxelSeriesStore(Protocol):
    @property
    def shape(self) -> tuple[int, int, int, int]: ...
    @property
    def dtype(self) -> np.dtype: ...
    def series(self, x: int | np.ndarray,
               y: int | None = None,
               z: int | None = None) -> np.ndarray: ...
    def volume(self, i: int) -> Any: ...
    def as_matrix(self) -> np.ndarray: ...


@runtime_checkable
class WritableVoxelSeriesStore(VoxelSeriesStore, Protocol):
    def __setitem__(self, key: Any, value: Any) -> None: ...
```

### Member semantics

| Member | Contract |
|--------|----------|
| `shape` | A 4-tuple `(nx, ny, nz, nt)`. The first three are the spatial grid; the fourth is the series length. |
| `dtype` | The `numpy.dtype` of the values `series`/`as_matrix` return. |
| `series(x, y, z)` | With three ints: the length-`nt` series at voxel `(x, y, z)`. With a single `ndarray` of voxel coordinates or flat indices: the stacked series for those voxels. Off-grid access raises per [`IDX-OOB`](contract-failure-vocabulary.md). |
| `volume(i)` | The 3-D volume at series index `i` (shape `(nx, ny, nz)`-compatible). |
| `as_matrix()` | A 2-D `(nt, n_voxels)` matrix — **time-major**, Fortran-flattened voxel order — suitable as the design-side input to reductions. |

A store is **read-only by default**. `WritableVoxelSeriesStore`
additionally supports `store[key] = value`.

---

## Conformance is via the adapter, not the container

This is the load-bearing subtlety. The shipped containers
(`DenseNeuroVec`, `SparseNeuroVec`, `BigNeuroVec`, `MappedNeuroVec`,
`FileBackedNeuroVec`) do **not** themselves satisfy
`isinstance(vec, VoxelSeriesStore)`. The conformer is
`NeuroVecStoreAdapter`, a frozen dataclass that wraps any vec:

```python
from neuroim.storage import VoxelSeriesStore, NeuroVecStoreAdapter

isinstance(bold, VoxelSeriesStore)                       # False
isinstance(NeuroVecStoreAdapter(bold), VoxelSeriesStore) # True
```

`DenseStore`, `SparseStore`, `BigStore`, `FileBackedStore`, and
`MappedStore` are all **aliases of `NeuroVecStoreAdapter`** — one
adapter spans every backend because the per-backend behaviour lives in
the wrapped vec's own `series`/`__getitem__`.

The consequence for the value proposition: "one protocol, every
backend" is true at the *storage* layer through this single adapter.
The *analysis* uniformity — `series_roi`, `temporal_snr`,
`searchlight_apply` behaving identically across dense/sparse/mapped/
file-backed inputs — is a separate, related guarantee carried by the
`NeuroVec` subclasses and their shared contract layer, not by this
protocol's five members. Do not claim the protocol itself provides the
analysis surface; it provides the storage surface the analysis layer
is built on.

---

## What a conforming third-party backend must do

1. Implement the five `VoxelSeriesStore` members with the semantics in
   the table above, including the **time-major** `as_matrix`
   orientation.
2. Raise the canonical off-grid refusals from
   [the contract-failure vocabulary](contract-failure-vocabulary.md)
   (`IDX-OOB` for voxel/index access).
3. If the backend stitches data from multiple sources with
   independent spatial frames (the `FileBackedNeuroVec` case), enforce
   per-unit affine consistency and raise `FB-DRIFT` rather than
   silently averaging two frames.
4. Either wrap itself in `NeuroVecStoreAdapter` or expose the protocol
   members directly so `isinstance(obj, VoxelSeriesStore)` is `True`.

A backend that implements every method but returns wrong-but-plausible
bytes on an off-grid access (instead of raising) is **non-conforming**
even though it is structurally a `VoxelSeriesStore`.

---

## Conformance test

`tests/test_voxel_series_store_protocol.py` (added with this spec)
asserts: `NeuroVecStoreAdapter` over each shipped backend satisfies
`isinstance(..., VoxelSeriesStore)`; `as_matrix()` is `(nt, n_voxels)`
time-major; off-grid `series` raises `IDX-OOB`. A change to the
member set or the `as_matrix` orientation is a spec-breaking change
and bumps this document to v2.

See the [VoxelSeriesStore concept page](../concepts/voxel-series-store.qmd)
for the narrative version.

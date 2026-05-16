# NeuroSpace invariants — v1

This document specifies the invariants `NeuroSpace` guarantees: what
is validated at construction, what equality and compatibility mean,
and how the spatial frame is hashed. It is a public contract — code
that produces or consumes a `NeuroSpace` (including third-party
backends) may rely on every statement here.

---

## Construction

`NeuroSpace(dim, spacing=None, origin=None, axes=None, trans=None)`.

Validated at construction, in order:

1. `dim` is coerced to an integer array. Empty → `ValueError`
   (`Dimensions cannot be empty`). Any non-positive entry →
   `ValueError` (`All dimensions must be positive`).
2. `spacing` defaults to all-ones; any non-positive entry →
   `ValueError` (`Spacing values must be positive`).
3. `origin` defaults to all-zeros.
4. `trans`:
   - If omitted, it is synthesized from `spacing` (diagonal) and
     `origin` (translation): 4×4 for ndim ≤ 3, `(ndim+1)×(ndim+1)`
     for ndim > 3.
   - If supplied, the shape is gated. ndim ≤ 3 requires 4×4 (else
     `trans must be 4x4 matrix for {ndim}D space, got {shape}`);
     ndim > 3 requires `(ndim+1)×(ndim+1)` (a bare 4×4 spatial
     sub-affine is accepted and extended; other shapes →
     `trans must be {n}x{n} matrix for {ndim}D space, got {shape}`).
   - When `trans` is supplied and `spacing`/`origin` are not, they are
     *derived from the affine* (`spacing = column norms`,
     `origin = translation column`). The affine is authoritative.
5. `trans` must be invertible. Singular and not rescuable by
   pseudo-inverse → `ValueError`
   (`Transformation matrix must be invertible`). The inverse is
   precomputed and stored as `inverse`.
6. `axes` defaults to nearest-anatomy (3-D with an affine) or
   positional `x,y,z,t,v…`. `axes.ndim` must equal `len(dim)`.

After construction the object is **frozen**: `dim`, `spacing`,
`origin`, `trans`, `inverse` are stored as read-only arrays, and any
attribute assignment raises `AttributeError`
(`NeuroSpace is immutable`).

---

## The affine

- `space.trans` is the raw transform. For ndim ≤ 3 it is 4×4; for
  ndim > 3 it is `(ndim+1)×(ndim+1)`.
- `space.affine` is a **property** that always returns a 4×4 *spatial*
  affine: for a higher-dimensional `trans` it extracts the `3×3`
  rotation/scale and the spatial translation column. This is the
  matrix `compatible_with` compares and the matrix that matches
  `nibabel`'s `img.affine` for the same file.
- `space.inverse` is the precomputed inverse of `trans`.

---

## Equality vs. compatibility — they are different

This distinction is load-bearing and surprises newcomers.

| | `a == b` (`__eq__`) | `a.compatible_with(b)` |
|---|---|---|
| Fields compared | `dim`, `origin`, `spacing`, **full `trans`** | **first 3 dims only**, `affine` (4×4) |
| Tolerance | exact (`np.array_equal`) | `atol=1e-6` |
| Time / extra axes | included (full `trans`) | ignored (4×4 spatial affine only) |
| On mismatch | returns `False` | **raises `ValueError`** |
| Axes (`AxisSet`) | not compared | not compared |

Consequences:

- A 4-D BOLD space and the matching 3-D mask space are **not `==`**
  (different `dim`, different `trans` shape) but **are
  `compatible_with`** (same first-3 dims, same 4×4 spatial affine).
  Every `mask=`/`roi` operation uses `compatible_with`, not `==`,
  which is why a 3-D mask is accepted against a 4-D BOLD.
- `compatible_with` is **true-or-raise**. It returns `True` or raises
  `ValueError` (`NeuroSpace mismatch in spatial dim: …` /
  `NeuroSpace mismatch in affine: …`). It never returns `False`. See
  [the contract-failure vocabulary](contract-failure-vocabulary.md).
- `==` is exact and includes the full transform, so two spaces that
  differ only in time-axis scaling are unequal but compatible.

---

## Hashing — two distinct hashes

| Hash | Defined in | Over | Used for |
|---|---|---|---|
| `hash(space)` (`__hash__`) | `NeuroSpace` | `dim`, `origin`, `spacing`, `trans` bytes/shape/dtype | Python set/dict membership; consistent with `__eq__` |
| `hash_neurospace(space)` | `neuroim.results` | `dim`, `spacing`, `origin` tuples + a hash of `trans` | the `input_space_hash` field of a [`Receipt`](receipt-nifti-extension.md) |

`hash_neurospace` is the one that appears on disk and in provenance.
It is a 16-hex-char SHA-256 truncation, stable across processes and
machines. `__hash__` is an in-process Python hash and must not be
relied on across processes. Do not conflate them.

---

## Coordinate conversion

- `space.grid_to_world(coords)` — voxel → world mm.
- `space.world_to_grid(coords)` — world mm → voxel (the inverse
  affine; rounding/bounds are the *caller's* or the consuming
  operation's responsibility — `series_at_world` is the operation that
  bounds-checks and raises `WORLD-OOB`).

There is no implicit conversion when comparing spaces.
`compatible_with` compares matrices, not transformed coordinates.

---

## Derivation operations return new spaces

`add_dim`, `drop_dim`, `from_affine`, reorientation — every operation
that "changes" a space returns a **new frozen `NeuroSpace`**. The
original keeps its identity and hash, so any `Receipt` that referenced
it stays valid. There is no in-place mutation path.

---

## Conformance

`tests/test_space_invariants.py` (added with this spec) pins: the
construction refusals, the frozen-attribute refusal, the
equality-vs-compatibility table, and the two-hash distinction. A
change to any row is a spec-breaking change and bumps this document to
v2.

See the [NeuroSpace concept page](../concepts/space.qmd) for the
narrative version.

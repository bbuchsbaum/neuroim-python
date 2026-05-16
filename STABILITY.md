# API stability contract

**Status: experimental (0.x). The public API is usable, documented, and
tested — but not stable. Do not pin production code to it yet.**

This contract is binding. The release schedule that applies it is in
[ROADMAP.md](ROADMAP.md); the durable charter is in [VISION.md](VISION.md) and
[MISSION.md](MISSION.md).

## What "experimental" means here

It does **not** mean unfinished or unreliable. The core typed model
(`NeuroSpace`, `NeuroVol`, `NeuroVec`, `NeuroHyperVec`, ROI types,
`Receipt`/provenance, the `VoxelSeriesStore` protocol) is implemented and
covered by a large passing test suite, and the documented Quick Start works.

It means: **the shape of the public API may still change between minor
releases.** You can build experiments and analyses on it today; you should
read the CHANGELOG before upgrading across a minor.

## The contract

### Versioning

`0.MINOR.PATCH` until 1.0:

- **Patch release** (`0.3.0 → 0.3.1`): bug fixes and known-gap closures only.
  No public name removed, no public signature broken. **Safe to pin.**
- **Minor release** (`0.3.x → 0.4.0`): the native public API *may* break.
  Anything that breaks was deprecated for at least one prior minor (see
  cadence below). **Read the CHANGELOG before upgrading.**
- **1.0.0 and later**: strict SemVer. Public API breaks only at the next major
  (`2.0.0`).

### Deprecation cadence

1. A public name or behavior slated to change emits a `DeprecationWarning`
   that names the replacement and the removal version.
2. It keeps working, unchanged, for **at least one full minor release**.
3. It is removed or changed no earlier than the *next* minor after the
   warning was introduced.

Exception (already ratified, charter decision D1): R/neuroim2 sentinels that
are not reachable as Python names (e.g. dictionary-style `globals()["as.matrix"]`
shims) may be dropped immediately, since no valid Python program can import
them.

### What is public

Public = names exported from `neuroim.__all__`, plus the documented
submodules (`neuroim.compat`, `neuroim.verify`) and the documented `atlas`
surface. The contract above applies to these.

**Not public** (may change at any time, no warning): anything underscore-
prefixed, anything not in `__all__`, internal modules, the
`golden_tests/`, `bench/`, and `examples/scenarios/` harnesses, and the exact
text of error messages.

### Two-lane parity (unchanged, stricter than the native contract)

`neuroim.compat` is **not** governed by the native API's freedom to change. It
preserves documented neuroim2 migration behavior under a **hard parity gate**
(byte-for-byte against neuroim2 fixtures, or numeric/semantic equivalence as
each fixture declares). Native result objects (`SearchlightResult`,
`ROIExtractionResult`, `Receipt`) are parity-gated as a **numeric projection**,
not a byte-for-byte schema. A native, safety-first API may be *stricter* than
its legacy counterpart as long as legacy and searchlight behavior is preserved.

### Provenance / Receipt schema

`Receipt` serialization (JSON and the NIfTI extension form) is **experimental**
until the schema is locked in the 0.6–0.9 stabilization window. Receipts
written by one 0.x minor are not guaranteed to round-trip byte-identically
through a later 0.x minor; the *semantic* fields (input/output space hash,
mask hash, method, library version) are intended to remain stable, but the
serialized layout may version. The schema is frozen at 1.0.

## What you can rely on today

- `neuroim.read_image`, `read_vol`, `read_vec`, `write_vol`, `write_vec` as the
  documented entry points.
- The typed spatial model and its construction via classmethods/factories.
- nibabel interop: `from_nibabel`, `to_nibabel`, `NeuroSpace.from_affine` — an
  explicit interchange guarantee (qform/sform preservation is NIfTI-only,
  best-effort for Analyze/MGH).
- Provenance receipts on ROI, searchlight, and derived-map results, surviving
  the NIfTI serialization boundary where the format can carry them (see
  `docs/spec/receipt-nifti-extension.md`).

## What you should not do yet

- Pin a production pipeline to an exact native API shape across minors.
- Depend on Receipt serialized-byte stability across minors.
- Depend on names not in `neuroim.__all__`, or on error-message text.

Report friction and breakage — pre-1.0 is exactly when that feedback changes
the design.

# Mission

**Build a Python-native neuroimaging analysis API that adds significant workflow value over raw nibabel by hiding low-level image mechanics behind typed spatial objects, while remaining fully interoperable with nibabel.**

## What We Build

- `neuroim.read_image`, `read_vol`, `read_vec`, `write_vol`, `write_vec` as the primary user-facing entry points; nibabel is the I/O substrate underneath, not the documented entry.
- `NeuroSpace`, `NeuroVol`, `NeuroVec`, and `NeuroHyperVec`: typed spatial containers with explicit affine, orientation, and axis contracts.
- ROI types (`ROICoords`, `ROIVol`, `ROIVecWindow`) and extraction APIs that return `ROIExtractionResult` objects with provenance.
- Searchlight infrastructure that returns `SearchlightResult` objects with values, centers, space, and a `Receipt`.
- Derived-map workflows, such as temporal SNR and searchlight/stat maps, that return typed spatial outputs with operation parameters and provenance instead of anonymous arrays.
- Dense, sparse, memory-mapped, file-backed, and mapped 4D voxel-series stores under one `VoxelSeriesStore` protocol.
- Explicit interop with nibabel: `from_nibabel(img)`, `to_nibabel(obj)`, `NeuroSpace.from_affine(affine, shape)` — a compatibility/interchange guarantee for callers who already hold a nibabel object. Visible where it matters; not the documented main entry.

## Decision Rules

1. **neuroim API is the surface; nibabel is substrate and baseline.** The public story is typed spatial objects and analysis workflows. `nib.load(...)` does not appear in our Quick Start. The default reader is `neuroim.read_image(...)`.
2. **Value-over-nibabel test.** A public feature is on mission only when it adds significant value over raw nibabel in a realistic workflow: fewer spatial-plumbing helpers, stronger protection against silent mistakes, or more inspectable derived outputs. External adoption is downstream evidence; the repo-level gate is a runnable scenario that demonstrates the value.
3. **Legacy parity, native contracts.** `neuroim.compat` preserves documented migration behavior from neuroim2 and is regression-gated against neuroim2 fixtures, using exact equality, numeric tolerance, or semantic equivalence as the fixture declares. The primary `neuroim` API is gated on Python-native contracts: numeric projection correctness, typed spatial objects, provenance receipts, backend-independent behavior, and useful validation errors.
4. **Structural provenance by default.** ROI, searchlight, and derived-map outputs carry provenance metadata; silent space/orientation/mask mismatches are caught at the contract layer, not in debugging. Receipt construction should flow through shared result/provenance paths or typed operation-parameter objects, not field-by-field diligence at every entry point.
5. **Curated public API.** `neuroim.__all__` stays small (≤ 40 names). Migration aliases live in `neuroim.compat`. Specialized surfaces live in subpackages (`neuroim.io`, `neuroim.filters`, `neuroim.plotting`).
6. **Protocols over inheritance.** Public typing surfaces are structural `Protocol`s extracted from observed contracts. ABCs are an internal implementation detail.
7. **Frozen value objects for spatial metadata.** `NeuroSpace` and axis classes are immutable; callers receive defensive copies.
8. **Red contract before refactor.** A failing acceptance test is filed before the implementation that turns it green.
9. **Scenarios decide feature priority.** New public API surface should be pulled by a runnable scenario, report, board post, or failing test that shows the raw-nibabel pain and the expected reduction in workaround code or risk.

## How We Work

- **mote** coordinates same-checkout work through claims, path reservations, notes, and a public discussion board. See [AGENTS.md](AGENTS.md).
- **beads** (`bd`/`br`) is the durable, git-backed project issue store. See [AGENTS.md](AGENTS.md).
- Strategy and design debate live on mote discussion topics; the current strategic thread is `neuroim-python-pythonic-value`.
- Implementation state lives on mote issue notes; durable project tasks live in beads.

## Current Milestone

0.3 evidence — prove that neuroim-python adds significant value over raw nibabel for derived-map workflows. The current milestone is tracked in mote as `bd-01KRKRZF7A9818NSM5QYQX9YJT`: **real workflow proof for derived maps, with structural provenance that survives serialization**.

The milestone is complete only when the scenario suite demonstrates the value-over-nibabel claim on realistic workflows, mission-bearing invariants are pinned by tests, provenance survives the write/read boundary where supported, and the work is committed, pushed, and reproducible from documented commands.

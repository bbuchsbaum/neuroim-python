# Mission

**Build a Python-native neuroimaging analysis API that hides low-level image mechanics behind typed spatial objects, while remaining fully interoperable with nibabel.**

## What We Build

- `neuroim.read_image`, `read_vol`, `read_vec`, `write_vol`, `write_vec` as the primary user-facing entry points; nibabel is the I/O substrate underneath, not the documented entry.
- `NeuroSpace`, `NeuroVol`, `NeuroVec`, and `NeuroHyperVec`: typed spatial containers with explicit affine, orientation, and axis contracts.
- ROI types (`ROICoords`, `ROIVol`, `ROIVecWindow`) and extraction APIs that return `ROIExtractionResult` objects with provenance.
- Searchlight infrastructure that returns `SearchlightResult` objects with values, centers, space, and a `Receipt`.
- Dense, sparse, memory-mapped, file-backed, and mapped 4D voxel-series stores under one `VoxelSeriesStore` protocol.
- Explicit interop with nibabel: `from_nibabel(img)`, `to_nibabel(obj)`, `NeuroSpace.from_affine(affine, shape)` — a compatibility/interchange guarantee for callers who already hold a nibabel object. Visible where it matters; not the documented main entry.

## Decision Rules

1. **neuroim API is the surface; nibabel is substrate.** The public story is typed spatial objects and analysis workflows. `nib.load(...)` does not appear in our Quick Start. The default reader is `neuroim.read_image(...)`.
2. **Strategic test (internal-only):** *would a competent nibabel user install neuroim-python because it makes their real analysis code shorter, safer, and more inspectable?* If yes, the work is on mission. If we are only rewrapping file I/O or preserving R-shaped APIs, we are drifting. This is the heuristic we apply to ourselves, not the way we describe the product to users.
3. **Legacy parity, native contracts.** `neuroim.compat` preserves documented migration behavior from neuroim2 and is regression-gated against neuroim2 fixtures, using exact equality, numeric tolerance, or semantic equivalence as the fixture declares. The primary `neuroim` API is gated on Python-native contracts: numeric projection correctness, typed spatial objects, provenance receipts, backend-independent behavior, and useful validation errors.
4. **Receipts by default.** ROI and searchlight outputs carry provenance metadata; silent space/orientation/mask mismatches are caught at the contract layer, not in debugging.
5. **Curated public API.** `neuroim.__all__` stays small (≤ 40 names). Migration aliases live in `neuroim.compat`. Specialized surfaces live in subpackages (`neuroim.io`, `neuroim.filters`, `neuroim.plotting`).
6. **Protocols over inheritance.** Public typing surfaces are structural `Protocol`s extracted from observed contracts. ABCs are an internal implementation detail.
7. **Frozen value objects for spatial metadata.** `NeuroSpace` and axis classes are immutable; callers receive defensive copies.
8. **Red contract before refactor.** A failing acceptance test is filed before the implementation that turns it green.

## How We Work

- **mote** coordinates same-checkout work through claims, path reservations, notes, and a public discussion board. See [AGENTS.md](AGENTS.md).
- **beads** (`bd`/`br`) is the durable, git-backed project issue store. See [AGENTS.md](AGENTS.md).
- Strategy and design debate live on mote discussion topics; the current strategic thread is `neuroim-python-pythonic-value`.
- Implementation state lives on mote issue notes; durable project tasks live in beads.

## Current Milestone

0.2 reshape — convert the public face from neuroim2 parity-shaped to Python-native analysis-shaped. Tracked under the epic *Pythonic Wedge* in mote (10 work-package issues, dependency-ordered). The decision matrix and acceptance criteria are pinned in the `neuroim-python-pythonic-value` topic (sticky `post-01KRKE0YY4`).

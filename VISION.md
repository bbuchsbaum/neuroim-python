# Vision

**neuroim-python makes neuroimaging arrays spatially intelligent, typed, validated, and analysis-ready.**

The product is a Python-native neuroimaging analysis API: typed spatial objects, validated coordinate/affine/mask relationships, ROI and searchlight workflows, sparse and lazy voxel-series extraction, and result objects that preserve provenance instead of handing back anonymous arrays.

Nibabel is the foundational I/O substrate and the baseline we must beat in real analysis workflows. neuroim-python is winning only when it adds significant value over raw nibabel: less spatial plumbing, stronger protection against silent spatial mistakes, and more inspectable derived outputs.

## Where We're Going

Working with neuroimaging data in Python becomes type-safe and analysis-shaped by default:

1. **Typed spatial contracts.** `NeuroSpace` is a value object that travels with every image. Coordinate transforms, mask alignment, and affine consistency are checkable at the type and contract layer, not by reading 4×4 matrices in a debugger.
2. **ROI and searchlight as first-class objects.** Voxel selection, time-by-voxel extraction, and searchlight analysis return typed result objects (`ROIExtractionResult`, `SearchlightResult`) with provenance receipts — not bare ndarrays that lose their spatial context the moment they leave the function.
3. **One protocol for every voxel-series store.** Dense, sparse, memory-mapped, file-backed, and on-the-fly mapped 4D containers expose the same `VoxelSeriesStore` contract, so analysis code never branches on storage backend.
4. **Validation receipts by default.** Every ROI extraction, searchlight, and derived-map workflow carries a `Receipt` with input-space hash, mask hash, operation parameters, seed where relevant, and library version. The silent space/orientation/mask mismatches that account for many neuroimaging bugs become visible, and provenance survives serialization when the file format can carry it.

The normal entry point is `neuroim.read_image(...)`. Nibabel interop (`from_nibabel`, `to_nibabel`) is an explicit guarantee for callers who already hold a nibabel object — it is not the hero of the Quick Start.

## What Success Looks Like

A realistic workflow that starts from raw nibabel/numpy plumbing and produces a derived neuroimaging result becomes shorter, safer, and more inspectable through neuroim-python:

- fewer handwritten affine, world/grid, and mask-alignment helpers;
- explicit failures where raw nibabel would silently return plausible wrong values;
- typed outputs that retain spatial context and provenance through downstream inspection and write/read boundaries.

The canonical falsifiable forms are the flagship workflow plus the scenario suite under `examples/scenarios/`. The 0.3 evidence gate is a public-source nibabel/nilearn-style workflow ported into that suite, including at least one 4-D-to-3-D derived-map step.

## Strategic Test

The question we apply to every public feature is: *does this create significant value over raw nibabel for a competent Python neuroimaging user?*

Adoption by outside users is downstream evidence, not the definition of success. The repo-testable criterion is whether a realistic workflow demonstrates meaningful surplus value over raw nibabel on three axes: less spatial plumbing, stronger safety, and better inspectability/provenance. If we are only rewrapping file I/O, preserving R-shaped APIs, or adding abstractions that do not improve a workflow, we are drifting.

## Non-Goals

- Reimplementing nibabel's file I/O, header parsing, or image proxies.
- Replacing nilearn, nipype, or any specific analysis pipeline.
- Preserving R/neuroim2 surface-level naming as the primary Python API.

R/neuroim2 parity remains a regression oracle for legacy and migration surfaces. It is not the product identity.

## Source of Truth

The live decision matrix and parity policy are pinned in the mote discussion topic `neuroim-python-pythonic-value` (sticky post `post-01KRKE0YY4` — consensus draft v2). The current 0.3 consensus is tracked in mote issue `bd-01KRKRZF7A9818NSM5QYQX9YJT`: real workflow proof for derived maps, with structural provenance that survives serialization. This file states the durable destination; those threads and motes state the current operational rules.

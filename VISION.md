# Vision

**neuroim-python makes neuroimaging arrays spatially intelligent, typed, validated, and analysis-ready.**

The product is a Python-native neuroimaging analysis API: typed spatial objects, validated coordinate/affine/mask relationships, ROI and searchlight workflows, sparse and lazy voxel-series extraction, and result objects that preserve provenance instead of handing back anonymous arrays.

Nibabel is the foundational I/O substrate — fully interoperable, not the product story.

## Where We're Going

Working with neuroimaging data in Python becomes type-safe and analysis-shaped by default:

1. **Typed spatial contracts.** `NeuroSpace` is a value object that travels with every image. Coordinate transforms, mask alignment, and affine consistency are checkable at the type and contract layer, not by reading 4×4 matrices in a debugger.
2. **ROI and searchlight as first-class objects.** Voxel selection, time-by-voxel extraction, and searchlight analysis return typed result objects (`ROIExtractionResult`, `SearchlightResult`) with provenance receipts — not bare ndarrays that lose their spatial context the moment they leave the function.
3. **One protocol for every voxel-series store.** Dense, sparse, memory-mapped, file-backed, and on-the-fly mapped 4D containers expose the same `VoxelSeriesStore` contract, so analysis code never branches on storage backend.
4. **Validation receipts by default.** Every ROI extraction and searchlight result carries a `Receipt` with input-space hash, mask hash, parameters, seed, and library version. The silent space/orientation/mask mismatches that account for most neuroimaging bugs become visible.

The normal entry point is `neuroim.read_image(...)`. Nibabel interop (`from_nibabel`, `to_nibabel`) is an explicit guarantee for callers who already hold a nibabel object — it is not the hero of the Quick Start.

## What Success Looks Like

A program that *extracts this ROI's time series, runs a searchlight, saves back to a NIfTI* — written through our API in ~15 typed, validated, reproducible lines — replaces the 50-line nibabel + numpy procedure that is the current default. The canonical falsifiable form is `tests/test_flagship_workflow.py`.

## Internal Strategic Test

A heuristic we apply to ourselves while building: *would a competent nibabel user install neuroim-python because it makes their real analysis code shorter, safer, and more inspectable?* If yes, the work is on mission. If we are only rewrapping file I/O or preserving R-shaped APIs, we are drifting.

This is an internal compass, not public positioning. The product story is the typed analysis API, not the relationship to nibabel.

## Non-Goals

- Reimplementing nibabel's file I/O, header parsing, or image proxies.
- Replacing nilearn, nipype, or any specific analysis pipeline.
- Preserving R/neuroim2 surface-level naming as the primary Python API.

R/neuroim2 parity remains a regression oracle for legacy and migration surfaces. It is not the product identity.

## Source of Truth

The live decision matrix and parity policy are pinned in the mote discussion topic `neuroim-python-pythonic-value` (sticky post `post-01KRKE0YY4` — consensus draft v2). This file states the durable destination; that thread states the current operational rules.

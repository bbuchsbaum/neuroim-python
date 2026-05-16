# Design note: future neuromaps / continuous-map integration

> Status: **design-only**. Mote `bd-01KRM2HKHX7DFPJPFZB8VARMN1` (Atlas
> F2). No implementation, no dependency on `neuromaps`. This note
> settles the modeling questions and the minimum red scenario required
> *before* any real adapter is written. Glasser / source-confidence
> questions are explicitly **out of scope** here (tracked separately).

## Why this is a design note, not code

`neuromaps` brings continuous brain maps (gradients, receptor density,
gene expression) and space-to-space transforms across volume, surface,
CIFTI, and densities. Pulling it in as a hard dependency would violate
the thin-core principle (cf. `pandas`, kept optional/lazy). The value
gate (MISSION decision rule 9) also requires a runnable scenario that
shows the pain before new public surface is added. So the deliverable
here is the decision record + the red scenario definition, nothing more.

## Q1 — Distinct `BrainMap`/`Annotation` type vs reuse `VolumetricAtlas`?

**Decision: a distinct continuous-map type, not an overload of the
parcellation/atlas type.** A parcellation is integer labels with a
label table; a neuromaps annotation is a continuous scalar field with
units and a source citation. They differ in dtype domain, the
reductions that are meaningful (parcel means vs. sampling/correlation),
and provenance fields. Forcing both through `VolumetricAtlas` would
reproduce the `LogicalNeuroVol`-as-`DenseNeuroVol` over-classification
smell. The continuous map is "a typed image (`NeuroVol`/surface
equivalent) + an `AnnotationProvenance`", not "an atlas".

## Q2 — Representation across volume / surface / CIFTI / mixed?

**Decision: one protocol, representation behind it — do not unify the
storage classes.** Mirror the `VoxelSeriesStore` pattern: a structural
`BrainMapLike` protocol (`.space_kind ∈ {volume, surface, cifti}`,
`.data`, `.provenance`, `.resample_to(target)`), with concrete
volume/surface/CIFTI carriers. Core ships only the volume carrier
(reuses `NeuroVol` + `NeuroSpace`); surface/CIFTI carriers live in an
optional package (Q4). Mixed maps are a *pair* of typed carriers with a
shared provenance receipt, never a magic union object.

## Q3 — Transform provenance vocabulary

**Decision: a typed `TransformParams(OpParams)` recording, at minimum:**

| Field | Why |
|---|---|
| `source_space`, `target_space` | the two frames (named, e.g. `MNI152`, `fsLR`) |
| `density` / `resolution` | surface density (e.g. `32k`) or volume mm |
| `method` | interpolation / areal / nearest |
| `backend` | which engine performed it (`neuromaps`, internal) |
| `source_hash`, `target_hash` | content hashes, as existing receipts do |

This composes with the existing `Receipt` chain (`receipt_for` /
`merge`) exactly like `ResampleParams` does today, so a transformed map
carries `…+neuromaps_transform(...)` in `method_name` and survives the
NIfTI extension round-trip. No new provenance machinery — just a new
`OpParams` subclass.

## Q4 — Core vs optional package boundary

**Decision:**

- **Core (`neuroim`)**: the `BrainMapLike` protocol, the volume carrier
  (already expressible with `NeuroVol`), and `TransformParams` in
  `neuroim.results`. Zero new third-party deps.
- **Optional (`neuroim_neuromaps`, future separate package)**: the
  actual `neuromaps` adapter, surface/CIFTI carriers, and any
  `neuromaps`/`nilearn`-surface dependency. Imported lazily; absence
  must raise the canonical contract-failure message, never a bare
  `ImportError` at `import neuroim`.

This keeps the thin-core guarantee and the lazy-optional-dep pattern
(`pandas`, `SearchlightResult.to_dataframe`) consistent.

## Minimum red scenario before any adapter

A scenario under `examples/scenarios/` that, with **no `neuromaps`
installed**:

1. constructs a continuous volume map + its `AnnotationProvenance`;
2. asserts that a same-space sampling/correlation against a parcellation
   works through the protocol with provenance recorded;
3. asserts that requesting a cross-space transform without the optional
   backend raises the **canonical contract-failure message** (not a raw
   `ImportError`), naming the optional package;
4. shows the before/after workaround-code reduction vs. doing the
   transform + provenance bookkeeping by hand.

Only when that red scenario exists and the protocol/`TransformParams`
land in core does writing a real `neuromaps` adapter become in-scope.

## Explicitly out of scope

Glasser parcellation specifics and atlas source-confidence scoring are
**not** part of continuous-map support and are tracked separately. This
note does not decide them.

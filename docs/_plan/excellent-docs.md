# Plan for Excellent Docs — neuroim-python

A docs site that is *about us*: typed spatial contracts, recoverable provenance, scenario-backed evidence. Not framed as "like X" or "unlike Y". Comparisons happen only where they are evidence (scenario verdicts), never as identity.

## 1. North star — what excellent means here

Excellent docs for this project pass four tests:

1. **Identity-first.** A reader who has never heard of nibabel, nilearn, or neuroim2 can read the home page, the concept index, and one scenario, and articulate what neuroim-python *is*: typed spatial objects + recoverable provenance + a value-over-baseline evidence gate. The hero never says "alternative to X."
2. **Promise-shaped.** Every page answers one of three questions: *what does it guarantee?*, *what does it refuse to do?*, *what evidence backs the guarantee?*
3. **Runnable.** Every doc page that makes a claim links to a scenario REPORT or a doctested snippet. The home page's 30-second example is the same code path as scenario S04.
4. **Two-surfaced.** A public contract layer (Receipt extension, `VoxelSeriesStore`, `NeuroSpace` invariants) is documented as a *specification* a stranger could implement against — independent of our codebase. The Python surface is documented as *usage* of that contract.

## 2. Information architecture

```
docs/
  index.qmd                 # hero, 30-second example, three promises
  start/                    # learn-by-doing path (5 pages, ~30 min)
    01-image.qmd            # read_image -> typed NeuroVol / NeuroVec
    02-space.qmd            # NeuroSpace as a value object you can interrogate
    03-roi.qmd              # ROI* + series_roi -> typed result with Receipt
    04-derived-maps.qmd     # searchlight / temporal_snr -> DenseNeuroVol with .provenance
    05-roundtrip.qmd        # write_vol / read_image — Receipt survives the file
  concepts/                 # the vocabulary (each page = 1 concept)
    space.qmd               # NeuroSpace contract
    receipt.qmd             # what's hashed, what isn't, why
    provenance.qmd          # rehydration semantics
    voxel-series-store.qmd  # one protocol behind dense/sparse/mapped/file-backed
    roi-result.qmd          # ROIExtractionResult / SearchlightResult
    contract-failures.qmd   # the failures we refuse to silently pass
    atlas.qmd               # typed parcellation as analysis input
  spec/                     # public on-disk + protocol contracts
    receipt-nifti-extension.md   # already exists — the model
    voxel-series-store-protocol.md
    space-invariants.md
    contract-failure-vocabulary.md
  cookbook/                 # task-oriented recipes
  evidence/                 # the scoreboard
  reference/                # quartodoc — generated, curated
```

`docs/porting/`, `docs/neuroim2_test_analysis.md`, `docs/neurohypervec_guide.md` move to `docs/_archive/` and are not in the navbar.

## 3. Home page

Three identity-shaped cards:

- **Contracts at the surface.** `series_roi`, `series_at_world`, `searchlight_apply`, `concat`, `temporal_snr` validate the spatial frame *before* returning data.
- **Provenance you can read with a hex editor.** Every typed result carries a content-addressable `Receipt`. `to_nibabel()` embeds it as a NIfTI `comment` extension; ten lines of `nibabel` alone can parse it.
- **One protocol behind every voxel-series backend.** Dense, sparse, mapped, file-backed: the same `VoxelSeriesStore` contract.

## 4. Start path — five pages, escalating contracts

| Page | New idea introduced |
|------|---------------------|
| 01-image | `read_image` returns typed objects |
| 02-space | `NeuroSpace` as a value object; `compatible_with` is the alignment check |
| 03-roi | ROI extraction returns a *result* with `.values`, `.coords`, `.space`, `.provenance` |
| 04-derived-maps | A `DenseNeuroVol` carrying `.provenance` is the unit of derived output |
| 05-roundtrip | Writing then reading is round-trip-safe for provenance, by spec |

Each page ends with "Refuses to:" — contract failures it would have caught, with scenario links.

## 5. Concepts — the vocabulary, in our terms

Each concept page is short (≤ 800 words), with the same spine:

- **What it is.** One paragraph.
- **Invariants.**
- **What it refuses.** Canonical error strings.
- **What it doesn't do.**
- **See also.** Cross-links into spec/ and reference/.

## 6. Spec — public contracts

Existing: `spec/receipt-nifti-extension.md`. Add:

- `voxel-series-store-protocol.md`
- `space-invariants.md`
- `contract-failure-vocabulary.md`

Each spec has a `Spec version` header; breaking changes log into `spec/changes.qmd`.

## 7. Cookbook — recipes named by user goal

Same five-section spine: Goal, Code, What's guaranteed, What survives the file, Trapdoors.

No nibabel comparisons inside cookbook pages — those live in `evidence/`.

## 8. Evidence — the scoreboard as a destination

`evidence/index.qmd` auto-renders a table from `examples/scenarios/*/REPORT.md`. The *only* place the docs say "vs. raw nibabel".

## 9. Reference — generated, curated

Group by concept, not by file. Every entry gets a manual `See also` block linking to the concept page, spec page (if applicable), and any cookbook recipe.

## 10. Editorial standards

- No "powerful," "easy," "intuitive," "modern," "comprehensive." (lint regex)
- No "see [other tool] for X."
- Every verdict-style claim links to a scenario or spec.
- Canonical phrasings: `compatible_with`, `Receipt`, `provenance`, `series_roi`, `series_at_world`.

## 11. Visual & code design

- `code-annotations: hover` on every contract `raise`.
- One repeated diagram: *array → typed object → operation → typed result with Receipt → file with extension → typed object again*.

## 12. Build, CI, freshness

- `quarto render` in CI; broken interlinks fail.
- Doctests inside `.qmd` pages run against shipped fixtures.
- Scenario scoreboard regenerated on every push.

## 13. Sequencing

1. **Land 1 — Identity.** New `index.qmd`, five `start/` pages, seven `concepts/` pages, archive `porting/`.
2. **Land 2 — Contracts as spec.** Three new `spec/` pages, `evidence/index.qmd` auto-rendered, docs lint pass.
3. **Land 3 — Cookbook & curated reference.** Seven recipes, quartodoc regroup, See-also links across reference pages, lifecycle diagram.

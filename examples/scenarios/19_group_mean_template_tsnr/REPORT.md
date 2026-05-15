# Scenario 19 — Group-mean tSNR across subjects in template space: Report

> Two synthetic subjects on different native grids.  Per subject:
> resample to a common template grid, compute a masked tSNR map.
> Across subjects: average the per-subject tSNR maps into a single
> group-level tSNR map.  This is the canonical second-level QC artifact.

## Verdict

**Per-subject win, group-reduce gap.** Each subject's typed tSNR map
already carries the `resample_vec(...)+temporal_snr` chain (Scenario 09
pinned this), and the per-subject path is shorter, safer, and more
inspectable than the raw-nibabel form.  But the **cross-subject reduce
has no first-class shape today**.  The careful raw-nibabel user owns a
manual `resample_from_to` loop, a per-frame tSNR reduction, and a
hand-rolled same-space-check + `np.stack`-and-mean group reducer with no
provenance.  The neuroim user owns a manual `np.stack`-and-mean group
reducer with provenance — that's an upgrade on safety and inspectability,
but the rewrite is **still hand-rolled**, and there is no
`ni.group_mean(*maps)` to call.

The terminal Receipt today reads (after the rewrite walks
`Receipt.merge` by hand):

```
resample_vec(order=1,source=…,target=…)+temporal_snr
+resample_vec(order=1,source=…,target=…)+temporal_snr
+group_mean(n=2)
```

Both subject chains are present; the group reduce is named.  But this
provenance only survives because the rewrite calls a *private*
`Receipt.merge` walk.  Custom reducers in user code cannot rely on a
public helper.

A second IO-layer surprise also emerges: `ni.write_vol(group_map, path)`
**does not** embed the receipt.  Only `group_map.to_nibabel()` +
`nib.save(...)` does.  Every prior scenario that asserted Receipt
round-trip used `to_nibabel`; this scenario is the first to ask whether
the curated `write_vol` path matches.  It does not — see PAIN-4.

## Scoreboard

| Axis | Baseline (raw nib + manifest) | Neuroim today | Neuroim after PAIN-1/2/3/4 |
|---|---:|---:|---:|
| Per-subject body | ~6 lines (resample loop + tSNR reduce) | **2** (`resample_vec` + `temporal_snr`) | 2 |
| Group-reduce body | ~8 lines (shape/affine loop + stack + mean) | ~10 lines (hand-rolled reducer + Receipt merge walk) | **1** (`ni.group_mean(*maps)`) |
| Same-space gate at group reduce | manual loop | manual loop in user code | owned by `group_mean` |
| Multi-input provenance recorded | no (would need sidecar) | yes, but via private `Receipt.merge` walk | yes, via public reducer |
| Receipt survives `write_vol` round-trip | n/a | **no** (must use `to_nibabel`) | yes |
| Output type | `nib.Nifti1Image` | `DenseNeuroVol` w/ chained `Receipt` | `DenseNeuroVol` w/ chained `Receipt` |

## What the API caught vs what it did not

| Case | Baseline | Neuroim today | Neuroim after PAINs land |
|---|---|---|---|
| Per-subject mask same-space | manual `shape ==` + `np.allclose(affine)` | caught by `temporal_snr(mask=...)` | caught by `temporal_snr(mask=...)` |
| Resample step in provenance | sidecar | `resample_vec(...)` Receipt (S09) | same |
| Subject-A vs Subject-B same-space at the group reduce | manual loop | manual loop in *user* code (PAIN-2) | inside `group_mean` |
| Both subject chains in terminal Receipt | sidecar with two subject IDs | yes via hand-walked `Receipt.merge` (PAIN-3) | yes via public reducer |
| Group reducer in `ni.__all__` | n/a | **no** (PAIN-1) | yes |
| `write_vol` embeds receipt | n/a | **no** (PAIN-4) | yes |

## Pain points surfaced

### PAIN-1 (P1 ergonomic / API gap) — no first-class group reducer for typed maps

`ni.concat` only accepts 4-D `NeuroVec` inputs and stacks along the time
axis.  There is no public function that takes N typed 3-D `NeuroVol`s
that share a frame and returns one typed `NeuroVol` whose Receipt records
the multi-input chain.  Every multi-subject scenario from S19 onward
will repeat the hand-rolled stack-mean-merge walk in user code.

**Suggested fix.** Add `ni.group_mean(*maps)` (or `ni.mean_volumes(maps)`).
It should:

1. Assert same-space across all inputs (PAIN-2 closure).
2. Stack data along a synthetic axis and reduce to a single
   `DenseNeuroVol` on the shared space.
3. Build a Receipt via `receipt_for(... params=GroupReduceParams(
   method_name="group_mean", n_inputs=len(maps)))` and merge every
   upstream Receipt into it through a public helper (PAIN-3 closure).

Once it lands, the cross-subject body collapses from ~10 lines to
`group_map = ni.group_mean(*per_subject_tsnr)`.

### PAIN-2 (P2 contract) — same-space gate across N inputs is user code

`assert_same_space` is pairwise.  A custom group reducer has to write
`for v in maps[1:]: assert_same_space(maps[0], v)` itself.  A first-class
`group_mean` (PAIN-1) would own this, mirroring the pattern that
`temporal_snr(mask=...)` already owns at the per-subject level.

**Suggested fix.** Either expose a public `verify.assert_same_space_all
(*objs)` that walks the loop, or fold the contract into `group_mean`
itself.

### PAIN-3 (P2 provenance) — no public multi-input Receipt helper

`Receipt.merge` is documented as the path for "intermediate ops that
consume two typed-result inputs", and `concat` uses it internally.  But
there is no public helper for the N-input case a custom reducer needs:
"merge these K upstream Receipts and append my op's `method_name`".  The
rewrite walks `Receipt.merge` by hand in a fold loop with pairwise
fallback when two inputs disagree.

**Suggested fix.** Expose `merge_receipts(*upstreams, params: OpParams)`
(or `reduce_receipts`) in `neuroim.results.__all__`.  It returns a
single Receipt whose `method_name` is
`f"{merged_chain}+{params.method_name}"` and whose mask/space follow
the existing pairwise `merge` semantics.  `concat`'s internal fold can
then call this helper instead of inlining it.

### PAIN-4 (P1 IO) — `write_vol` does not embed the receipt

`vol.to_nibabel()` builds a `Nifti1Image` with the Receipt encoded as a
`comment` extension (S05 PAIN-6 closure).  But `ni.write_vol(vol, path)`
constructs a fresh `nib.Nifti1Image(data, vol.trans)` with no extensions
and saves that.  The user has to write `nib.save(vol.to_nibabel(),
path)` instead — exactly the kind of "remember the magic dance"
friction the curated API is supposed to remove.

A collaborator who follows the public API (`ni.write_vol` → `ni.read_image`)
silently loses every chained Receipt at the IO boundary, even on maps
whose in-memory `.provenance` is fully populated.

**Suggested fix.** Have `write_vol` route through `vol.to_nibabel()`
when the input has a `.provenance` Receipt, so the receipt embeds by
default.  No back-compat risk — `read_image` already silently ignores
foreign comment extensions.

## How to reproduce

```bash
PYTHONPATH=src:tests:. .venv/bin/python -m pytest \
    tests/test_s19_group_mean_template_tsnr.py -q
```

Expected: **5 passed + 4 xfailed (strict)**.  Each strict xfail flips
to XPASS the moment the corresponding API gap closes (PAIN-1/2/3/4).

## Follow-ups

| Pain | Suggested issue title | Priority |
|---|---|---|
| PAIN-1 | Add `ni.group_mean(*maps)` first-class group reducer for typed 3-D maps | P1 |
| PAIN-2 | Expose `verify.assert_same_space_all(*objs)` (or fold into `group_mean`) | P2 |
| PAIN-3 | Expose `merge_receipts(*upstreams, params)` public helper in `neuroim.results.__all__` | P2 |
| PAIN-4 | `write_vol` must embed `vol.provenance` as a NIfTI comment extension by default | P1 |

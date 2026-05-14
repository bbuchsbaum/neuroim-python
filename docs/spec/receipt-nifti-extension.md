# Receipt NIfTI extension — v1

This document specifies how `neuroim-python` embeds a *provenance
Receipt* in a NIfTI-1 / NIfTI-2 file so that any reader — including
readers that do not import `neuroim` — can recover what produced the
volume.

The spec is intentionally short.  The receipt is one JSON blob in one
standard NIfTI header extension.  Anything that can read NIfTI
extensions can read a neuroim receipt.

---

## Encoding

A neuroim receipt is stored as a **NIfTI-1 header extension** with the
following layout:

| Field | Value |
|---|---|
| Extension code (ecode) | `6` (the standard `comment` ecode) |
| Content prefix | the literal ASCII string `neuroim/receipt/v1:` |
| Content body | a UTF-8 encoded JSON object (see schema below) |

A file may carry **at most one** neuroim receipt extension.  Writers
must replace any prior receipt extension when re-serializing; readers
must take the first matching extension if more than one is found.

The `neuroim/receipt/v1:` prefix is the version marker.  It is
mandatory.  Foreign `comment` extensions (free-text annotations from
other tools) are silently ignored on read.

A future v2 receipt — adding fields, changing semantics, or moving to
a structured-provenance form — will use a different prefix
(`neuroim/receipt/v2:`).  Readers that know only v1 must ignore
unknown prefixes without raising.

---

## JSON schema (v1)

The JSON object has exactly these eight fields, all present, in any
order:

| Field | Type | Semantics |
|---|---|---|
| `input_space_hash` | `string` | Hex digest (first 16 chars of SHA-256) of the input `NeuroSpace` (dim + spacing + origin + affine bytes). Identical inputs produce identical hashes. |
| `mask_hash` | `string` | Hex digest of the mask coordinates the operation consumed. Special value `"none"` means no mask was applied (wildcard). |
| `radius` | `number` or `null` | Searchlight / neighbourhood radius in mm. `null` when not applicable (e.g. ROI extraction). |
| `n_voxels` | `integer` | Number of voxels the operation produced or consumed (depending on op). |
| `method_name` | `string` | The operation's name (e.g. `"series_roi"`, `"searchlight"`, `"mean_over_time"`). Composed operations may use `"a+b"`. |
| `seed` | `integer` or `null` | RNG seed if the operation is randomized; `null` otherwise. |
| `neuroim_version` | `string` | The neuroim release that produced the file (e.g. `"0.2.0"`). |
| `source_affine_hash` | `string` | Hex digest of the source data's 4×4 spatial affine bytes. Special value `"none"` when no source affine was recorded. |

All hashes are the first 16 hex characters of a SHA-256 digest over
the input bytes plus shape/dtype metadata, matching the production of
`neuroim.results.hash_ndarray` and `neuroim.results.hash_neurospace`.
Hashes are *not* cryptographic signatures — they exist to make silent
space/mask mismatches visible, not to certify provenance.

---

## Reference reader (10 lines, nibabel only)

The following Python snippet reads a neuroim receipt without
importing `neuroim`.  It is the canonical compatibility check: any
implementation that produces v1 receipts must be readable by this
snippet, and any conforming reader must accept files produced by it.

```python
import json
import nibabel as nib

def read_neuroim_receipt(path):
    """Return the neuroim v1 receipt dict, or ``None`` if absent."""
    img = nib.load(str(path))
    for ext in getattr(img.header, "extensions", []) or []:
        if int(ext.get_code()) != 6:
            continue
        text = bytes(ext.get_content()).rstrip(b"\x00").decode("utf-8", "replace")
        if text.startswith("neuroim/receipt/v1:"):
            return json.loads(text[len("neuroim/receipt/v1:"):])
    return None
```

A collaborator who receives a `.nii.gz` produced by neuroim can run
this snippet and observe:

```python
>>> read_neuroim_receipt("derived_map.nii.gz")
{'input_space_hash': '31b52c58ed459f2d',
 'mask_hash': '8bbfc7b2b9dd91cf',
 'method_name': 'searchlight',
 'n_voxels': 5744,
 'neuroim_version': '0.2.0.dev0',
 'radius': 4.5,
 'seed': None,
 'source_affine_hash': '4dbf52aa8e98c6f8'}
```

Two receipts agree on `input_space_hash` and `mask_hash` *iff* they
were produced from the same input space and the same mask.  That
property is the load-bearing guarantee neuroim relies on to catch
silent space/mask mismatches.  Third-party tools can rely on it
without importing neuroim.

---

## Writing a receipt without neuroim

Symmetric reference writer for tools that want to interoperate:

```python
import json
import nibabel as nib
from nibabel.nifti1 import Nifti1Extension

def write_neuroim_receipt(img, receipt):
    """Replace any prior v1 receipt extension on ``img`` with ``receipt``."""
    payload = ("neuroim/receipt/v1:" + json.dumps(receipt, sort_keys=True)).encode("utf-8")
    img.header.extensions[:] = [
        ext for ext in img.header.extensions
        if not (int(ext.get_code()) == 6
                and bytes(ext.get_content()).rstrip(b"\x00").decode("utf-8", "replace").startswith("neuroim/receipt/v1:"))
    ]
    img.header.extensions.append(Nifti1Extension(6, payload))
    return img
```

The `receipt` argument is a plain dict matching the v1 schema.  A
file written this way is indistinguishable from one written by
`neuroim`'s `NeuroVol.to_nibabel()` and will be re-hydrated by
`neuroim.read_image()` onto `vol.provenance`.

---

## Compatibility test

`tests/test_receipt_spec_external_reader.py` exercises the reference
reader against a file written by `neuroim`.  It is the contract test
for this spec: any change to the embed/extract path that breaks the
external reader is a spec-breaking change and requires a v2 marker.

---

## Why this spec exists

A `Receipt` that only `import neuroim` can read is a private blob.
The mission claim — *"silent space/orientation/mask mismatches are
caught at the contract layer"* — has to survive the file boundary to
be true at handoff time.  Publishing the extension contract is what
turns the in-memory receipt into a *public* one.

See `examples/scenarios/05_receipt_io_boundary/REPORT.md` for the
falsifying scenario that motivated this work (PAIN-6).

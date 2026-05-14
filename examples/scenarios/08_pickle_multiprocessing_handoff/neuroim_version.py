"""Rewrite: pickle a typed derived map through a fresh Python process."""

from __future__ import annotations

import base64
import json
import pickle
import subprocess
import sys
from typing import Any

import neuroim as ni


def temporal_snr_volume(
    bold: ni.NeuroVec,
    mask: ni.LogicalNeuroVol,
) -> ni.DenseNeuroVol:
    """Return a typed temporal-SNR map with provenance."""
    return bold.temporal_snr(mask=mask)


def pickle_payload(obj: Any) -> bytes:
    """Serialize an object as a joblib/multiprocessing-style pickle payload."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def inspect_payload_in_fresh_process(payload: bytes) -> dict[str, Any]:
    """Unpickle a neuroim object in a clean Python process and summarize it."""
    code = r"""
import base64, json, pickle, sys
obj = pickle.loads(base64.b64decode(sys.stdin.read().encode("ascii")))
provenance = getattr(obj, "provenance", None)
summary = {
    "payload_type": type(obj).__name__,
    "shape": tuple(int(x) for x in obj.shape),
    "has_space": hasattr(obj, "space"),
    "has_provenance": provenance is not None,
    "method_name": None if provenance is None else provenance.method_name,
    "input_space_hash": None if provenance is None else provenance.input_space_hash,
    "mask_hash": None if provenance is None else provenance.mask_hash,
}
print(json.dumps(summary, sort_keys=True))
"""
    encoded = base64.b64encode(payload).decode("ascii")
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=encoded,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)

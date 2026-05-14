"""Contract tests for the public Receipt v1 NIfTI-extension schema."""

from __future__ import annotations

import json
import re
from pathlib import Path

import neuroim as ni
from neuroim.results import RECEIPT_NIFTI_PREFIX, Receipt, make_receipt


SPEC_PATH = Path(__file__).resolve().parents[1] / "docs/spec/receipt-nifti-extension.md"

RECEIPT_V1_FIELDS = (
    "input_space_hash",
    "mask_hash",
    "radius",
    "n_voxels",
    "method_name",
    "seed",
    "neuroim_version",
    "source_affine_hash",
)


def _spec_text() -> str:
    return SPEC_PATH.read_text(encoding="utf-8")


def _schema_fields_from_spec() -> tuple[str, ...]:
    fields: list[str] = []
    in_schema = False
    for line in _spec_text().splitlines():
        if line.startswith("## JSON schema (v1)"):
            in_schema = True
            continue
        if in_schema and line.startswith("## "):
            break
        match = re.match(r"\| `([^`]+)` \|", line)
        if match:
            fields.append(match.group(1))
    return tuple(fields)


def test_receipt_v1_marker_is_public_contract():
    assert RECEIPT_NIFTI_PREFIX == "neuroim/receipt/v1:"
    assert "Content prefix | the literal ASCII string `neuroim/receipt/v1:`" in _spec_text()
    assert "A future v2 receipt" in _spec_text()


def test_receipt_dataclass_fields_match_v1_spec_exactly():
    """Adding a Receipt field is a v2 schema change unless this test changes."""
    assert tuple(Receipt.__dataclass_fields__) == RECEIPT_V1_FIELDS
    assert _schema_fields_from_spec() == RECEIPT_V1_FIELDS


def test_receipt_json_payload_contains_exactly_v1_fields():
    receipt = make_receipt(
        input_space=ni.NeuroSpace((2, 2, 2)),
        mask_data=None,
        radius=None,
        n_voxels=8,
        method_name="schema-lock",
        seed=None,
    )
    recovered = Receipt.from_json(receipt.to_json())

    assert recovered == receipt
    payload = json.loads(receipt.to_json())
    assert set(payload) == set(RECEIPT_V1_FIELDS)
    assert set(recovered.__dict__) == set(RECEIPT_V1_FIELDS)

#!/usr/bin/env python3
"""Editorial lint for the authored docs surface.

Enforces the editorial standards in docs/_plan/excellent-docs.md §10:

1. Banned marketing adjectives — replace with the concrete fact.
2. Identity-by-comparison framing is allowed ONLY under docs/evidence/
   (where comparison is a measurement, not framing).
3. Canonical API tokens must be spelled/cased correctly.

Scope is the *authored* redesign surface only: index.qmd, start/,
concepts/, spec/, evidence/. Generated (reference/), archived
(_archive/, _plan/), and legacy pages are out of scope.

Exit 0 = clean, 1 = violations. Run: python tools/docs_lint.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

AUTHORED = [
    DOCS / "index.qmd",
    DOCS / "api-map.qmd",
    DOCS / "start",
    DOCS / "concepts",
    DOCS / "cookbook",
    DOCS / "spec",
    DOCS / "evidence",
]

# 1. Banned adjectives (word-boundary, case-insensitive).
BANNED_ADJECTIVES = [
    "powerful", "easy", "intuitive", "modern", "comprehensive",
    "seamless", "blazing", "effortless", "cutting-edge", "state-of-the-art",
]

# 2. Identity-by-comparison phrases. Allowed only under evidence/.
COMPARISON_PHRASES = [
    r"alternative to \w+",
    r"unlike (?:nibabel|nilearn|neuroim2|nipype)",
    r"better than (?:nibabel|nilearn|neuroim2)",
    r"(?:just )?like (?:nibabel|nilearn|neuroim2)\b",
    r"compared to (?:nibabel|nilearn|neuroim2)",
    r"drop-in replacement",
]

# 3. Canonical token misspellings -> the correct token.
CANON_MISSPELLINGS = {
    r"\bSeries_roi\b": "series_roi",
    r"\bseries_at_World\b": "series_at_world",
    r"\bseries_atworld\b": "series_at_world",
    r"\bCompatible_with\b": "compatible_with",
    r"\bcompatible with\(\)": "compatible_with()",
    r"\bprovenence\b": "provenance",
    r"\bprovance\b": "provenance",
    r"\bReciept\b": "Receipt",
}


def iter_files():
    for entry in AUTHORED:
        if entry.is_file():
            yield entry
        elif entry.is_dir():
            yield from sorted(entry.rglob("*.qmd"))
            yield from sorted(entry.rglob("*.md"))


def strip_code_blocks(text: str) -> str:
    """Blank out fenced code so prose-only rules don't fire on code."""
    return re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL)


def main() -> int:
    adj_re = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(a) for a in BANNED_ADJECTIVES) + r")(?![\w-])",
        re.IGNORECASE,
    )
    cmp_res = [re.compile(p, re.IGNORECASE) for p in COMPARISON_PHRASES]
    miss_res = [(re.compile(p), good) for p, good in CANON_MISSPELLINGS.items()]

    violations: list[str] = []

    for path in iter_files():
        rel = path.relative_to(DOCS)
        in_evidence = rel.parts and rel.parts[0] == "evidence"
        prose = strip_code_blocks(path.read_text(encoding="utf-8", errors="replace"))

        for i, line in enumerate(prose.splitlines(), 1):
            for m in adj_re.finditer(line):
                violations.append(f"{rel}:{i}: banned adjective '{m.group(1)}' — state the concrete fact instead")
            if not in_evidence:
                for cre in cmp_res:
                    m = cre.search(line)
                    if m:
                        violations.append(
                            f"{rel}:{i}: identity-by-comparison '{m.group(0)}' — "
                            f"comparison framing belongs only under docs/evidence/"
                        )
            for mre, good in miss_res:
                m = mre.search(line)
                if m:
                    violations.append(f"{rel}:{i}: '{m.group(0)}' — canonical token is '{good}'")

    if violations:
        print("Docs editorial lint: FAIL\n")
        for v in violations:
            print("  " + v)
        print(f"\n{len(violations)} violation(s).")
        return 1

    print(f"Docs editorial lint: OK ({sum(1 for _ in iter_files())} authored files clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

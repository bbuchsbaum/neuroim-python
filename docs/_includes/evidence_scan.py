"""Single source for the evidence scoreboard aggregation.

Both `docs/evidence/index.qmd` (full table) and `docs/index.qmd` (the
condensed teaser) import this so the homepage count can never drift from
the scoreboard — it is the same scan, parsed once. A parse failure here
fails BOTH renders visibly rather than silently emitting a wrong number.

Lives under `_includes/` so Quarto never renders it as a page.
"""
from __future__ import annotations

import re
from pathlib import Path


def find_repo(cwd: Path) -> Path:
    """Resolve the repo root from a Quarto render cwd (evidence/, docs/, or repo)."""
    if cwd.name == "evidence":
        return cwd.parents[1]
    if cwd.name == "docs":
        return cwd.parent
    return cwd


def scenario_rows(repo: Path) -> list[tuple[str, str, str, str]]:
    """Return (num, slug, title, verdict) for every numbered scenario REPORT.md.

    Identical parse for the full table and the teaser count — callers differ
    only in how they render the returned rows.
    """
    scen_root = repo / "examples" / "scenarios"
    rows: list[tuple[str, str, str, str]] = []
    for report in sorted(scen_root.glob("*/REPORT.md")):
        slug = report.parent.name
        if not slug[:2].isdigit():
            continue
        text = report.read_text(encoding="utf-8", errors="replace")

        h1 = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        title = h1.group(1).strip() if h1 else slug
        title = re.sub(r":\s*Report\s*$", "", title, flags=re.IGNORECASE)

        verdict = ""
        vm = re.search(
            r"^##\s*Verdict\s*\n+(.+?)(?:\n##\s|\Z)", text, re.DOTALL | re.MULTILINE
        )
        if vm:
            block = vm.group(1).strip()
            b = re.search(r"\*\*(.+?)\*\*", block, re.DOTALL)
            verdict = (b.group(1) if b else block.split("\n\n")[0]).strip()
            verdict = re.sub(r"\s+", " ", verdict)
            if len(verdict) > 160:
                verdict = verdict[:157].rstrip() + "…"
        verdict = verdict.replace("|", "\\|") or "—"

        rows.append((slug[:2], slug, title, verdict))
    return rows

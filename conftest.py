"""Repo-root pytest configuration.

Makes the real fixture and example source trees importable without
shipping them inside the installed wheel:

- ``tests/`` on ``sys.path`` so scenario/example tests resolve
  ``from fixtures.realistic_bold import ...`` to ``tests/fixtures/``.
- repo root on ``sys.path`` so ``from examples.draw_audit import ...``
  resolves to the real ``examples/`` tree (PEP 420 namespace package).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
for _p in (_ROOT, _ROOT / "tests"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

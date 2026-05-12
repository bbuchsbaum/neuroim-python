"""Compatibility shim for the older ``pyneuroim`` package name.

Use ``neuroim`` for new code.
"""

import importlib as _importlib

_neuroim = _importlib.import_module("neuroim")

__path__ = _neuroim.__path__

for _name, _value in _neuroim.__dict__.items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = _value

__all__ = getattr(_neuroim, "__all__", [k for k in globals() if not k.startswith("_")])

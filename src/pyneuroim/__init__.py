"""Compatibility alias for older golden tests.

The package is now named ``neuroimpy``.  Some parity fixtures still import the
earlier ``pyneuroim`` name, so this module re-exports the current public API.
"""

from neuroimpy import *  # noqa: F401,F403

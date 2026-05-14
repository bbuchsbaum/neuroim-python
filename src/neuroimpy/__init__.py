"""Deprecated compatibility shim for historical neuroim package roots."""

from warnings import warn as _warn

import neuroim as _neuroim

_warn(
    f"{__name__} is deprecated; import neuroim instead. "
    "This alias package will be removed after 0.3.",
    DeprecationWarning,
    stacklevel=2,
)

__path__ = _neuroim.__path__

from neuroim import *  # noqa: F401,F403
from neuroim import __version__  # noqa: F401,E402

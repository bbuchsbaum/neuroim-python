"""Single source of the implicit-array-conversion refusal.

A typed image deliberately does not implement a data-returning
``__array__``: an implicit ``np.asarray(img)`` / ``np.array(img)`` would
drop the :class:`~neuroim.neuro_space.NeuroSpace` silently, which is the
exact failure the typed-image contract exists to prevent. Without a
``__array__`` at all, NumPy falls back to wrapping the object as a 0-d
``object`` array — a silent, wrong-shaped result that is worse than an
error. Implementing ``__array__`` to *raise* turns that into an explicit,
readable refusal that names the deliberate way out.

The message wording is part of the documented "what it refuses" contract
(see ``docs/concepts/images.qmd``); keeping it in one place stops the
three container bases from drifting apart.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .exceptions import ImplicitArrayConversionError


def refuse_array_conversion(obj: Any, accessor: str) -> NoReturn:
    """Raise the canonical refusal for an implicit array cast.

    Parameters
    ----------
    obj
        The typed image instance NumPy tried to convert.
    accessor
        The explicit, type-appropriate way to obtain the ndarray
        (e.g. ``".as_array()"`` for a volume).
    """
    raise ImplicitArrayConversionError(
        f"{type(obj).__name__} does not implicitly convert to a NumPy "
        f"array: np.asarray()/np.array() would silently drop the spatial "
        f"frame (NeuroSpace). Use {accessor} to get the underlying ndarray "
        f"explicitly."
    )

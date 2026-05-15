"""Typed contract-failure exceptions.

These give the *documented* contract-failure vocabulary
(``docs/spec/contract-failure-vocabulary.md``) a class to catch instead
of a message substring to grep. Scope is exactly that spec — the broad
tail of incidental ``ValueError``/``TypeError`` raises is intentionally
left untyped.

Backward compatibility is structural: every concrete class multiply
inherits ``NeuroimError`` *and* the built-in the spec already lists for
that row (``ValueError``/``IndexError``/``AttributeError``/``TypeError``).
So every existing ``except ValueError`` / ``pytest.raises(ValueError,
match=...)`` keeps working unchanged, while new code can write
``except SpaceMismatchError`` or ``except NeuroimError``. The canonical
message strings are unchanged, so the spec stays at v1 and its
conformance test is unaffected.

Spec-ID → class map:

==================  ==========================  ===================
Spec ID             Class                       Built-in base
==================  ==========================  ===================
SPACE-DIM           SpaceMismatchError          ValueError
SPACE-AFFINE        SpaceMismatchError          ValueError
SPACE-ASSERT        SpaceMismatchError          ValueError
SPACE-HASH          SpaceMismatchError          ValueError
CONCAT-SPACE        SpaceMismatchError          ValueError
MASK-HASH           MaskMismatchError           ValueError
IDX-OOB             OutOfBoundsError            IndexError
WORLD-OOB           WorldOutOfBoundsError       ValueError
FB-DRIFT            BackendDriftError           ValueError
FROZEN-SPACE        ImmutableError              AttributeError
DIM-POS             InvalidSpaceError           ValueError
SPACING-POS         InvalidSpaceError           ValueError
TRANS-3D            InvalidSpaceError           ValueError
TRANS-ND            InvalidSpaceError           ValueError
TRANS-SINGULAR      InvalidSpaceError           ValueError
OOB-MODE            InvalidArgumentError        ValueError
(A1a refusal)       ImplicitArrayConversionError TypeError
==================  ==========================  ===================
"""

from __future__ import annotations


class NeuroimError(Exception):
    """Marker base for every typed contract failure neuroim raises.

    Catch this to mean "a neuroim contract was violated" regardless of
    which built-in the specific failure also presents as.
    """


class SpaceMismatchError(NeuroimError, ValueError):
    """Two spatial frames are not compatible (SPACE-* / CONCAT-SPACE)."""


class MaskMismatchError(NeuroimError, ValueError):
    """Two mask hashes disagree (MASK-HASH)."""


class OutOfBoundsError(NeuroimError, IndexError):
    """A voxel/index coordinate is outside the spatial grid (IDX-OOB)."""


class WorldOutOfBoundsError(NeuroimError, ValueError):
    """A world coordinate maps outside the image grid (WORLD-OOB)."""


class BackendDriftError(NeuroimError, ValueError):
    """A file-backed volume has an inconsistent affine/space (FB-DRIFT)."""


class ImmutableError(NeuroimError, AttributeError):
    """Mutation attempted on a frozen value object (FROZEN-SPACE)."""


class InvalidSpaceError(NeuroimError, ValueError):
    """A NeuroSpace was constructed with invalid geometry (DIM/SPACING/TRANS-*)."""


class InvalidArgumentError(NeuroimError, ValueError):
    """A contract-bearing argument was out of its documented domain (OOB-MODE)."""


class ImplicitArrayConversionError(NeuroimError, TypeError):
    """Implicit ``np.asarray``/``np.array`` on a typed image was refused.

    Raised by the container ``__array__`` guard (A1a). Subclasses
    ``TypeError`` so NumPy still treats the object as non-array and any
    ``except TypeError`` keeps working.
    """


__all__ = [
    "NeuroimError",
    "SpaceMismatchError",
    "MaskMismatchError",
    "OutOfBoundsError",
    "WorldOutOfBoundsError",
    "BackendDriftError",
    "ImmutableError",
    "InvalidSpaceError",
    "InvalidArgumentError",
    "ImplicitArrayConversionError",
]

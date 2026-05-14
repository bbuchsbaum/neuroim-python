from dataclasses import dataclass
from typing import List, Union

import numpy as np

def _readonly_direction(
    direction: Union[int, float, List[float], np.ndarray],
) -> np.ndarray:
    if isinstance(direction, (int, float)):
        arr = np.array([direction])
    else:
        arr = np.array(direction, copy=True)
    arr.setflags(write=False)
    return arr

@dataclass(frozen=True)
class NamedAxis:
    """Named axis with direction vector.

    Associates an axis name with its direction vector.

    Parameters
    ----------
    axis : str
        Name of the axis (e.g., "Left-to-Right")
    direction : array-like
        Direction vector (e.g., [1, 0, 0] for x-axis)

    """

    axis: str
    direction: Union[int, float, List[float], np.ndarray]

    def __post_init__(self):
        object.__setattr__(self, "direction", _readonly_direction(self.direction))

    def __repr__(self):
        return f"NamedAxis(axis='{self.axis}', direction={self.direction.tolist()})"

    def __eq__(self, other):
        if not isinstance(other, NamedAxis):
            return False
        return self.axis == other.axis and np.array_equal(
            self.direction, other.direction
        )

    def __hash__(self):
        return hash((self.axis, self.direction.dtype.str, self.direction.tobytes()))

# Anatomical axis constants (matching R's neuroim2)
LEFT_RIGHT = NamedAxis("Left-to-Right", [1, 0, 0])
RIGHT_LEFT = NamedAxis("Right-to-Left", [-1, 0, 0])
ANT_POST = NamedAxis("Anterior-to-Posterior", [0, -1, 0])
POST_ANT = NamedAxis("Posterior-to-Anterior", [0, 1, 0])
INF_SUP = NamedAxis("Inferior-to-Superior", [0, 0, 1])
SUP_INF = NamedAxis("Superior-to-Inferior", [0, 0, -1])

# Special axes
TIME = NamedAxis("Time", 1)
NullAxis = NamedAxis("None", 0)
TimeAxis = TIME  # Alias for compatibility

class AxisSet:
    """Base class for axis sets.

    Describes a coordinated set of named spatial axes.
    """

    def __init__(self, ndim: int):
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "ndim", ndim)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable")
        object.__setattr__(self, name, value)

    def _freeze(self):
        object.__setattr__(self, "_frozen", True)

    def __repr__(self):
        return f"{self.__class__.__name__}(ndim={self.ndim})"

    def __len__(self):
        return self.ndim

class AxisSetND(AxisSet):
    """Generic axis set for ndim > 5."""

    def __init__(self, axes: List[NamedAxis]):
        if len(axes) < 1:
            raise ValueError("AxisSetND requires at least one axis")
        super().__init__(len(axes))
        self.axes = tuple(axes)
        self._freeze()

    def __iter__(self):
        yield from self.axes

    def __repr__(self):
        return f"AxisSetND(ndim={self.ndim}, axes={[ax.axis for ax in self.axes]})"

class AxisSet1D(AxisSet):
    """1D axis set."""

    def __init__(self, i: NamedAxis):
        super().__init__(1)
        self.i = i
        self._freeze()

    def __repr__(self):
        return f"AxisSet1D(i={self.i})"

    def __iter__(self):
        yield self.i

class AxisSet2D(AxisSet):
    """2D axis set."""

    def __init__(self, i: NamedAxis, j: NamedAxis):
        super().__init__(2)
        self.i = i
        self.j = j
        self._freeze()

    def __repr__(self):
        return f"AxisSet2D(i={self.i}, j={self.j})"

    def __iter__(self):
        yield self.i
        yield self.j

class AxisSet3D(AxisSet):
    """3D axis set."""

    def __init__(self, i: NamedAxis, j: NamedAxis, k: NamedAxis):
        super().__init__(3)
        self.i = i
        self.j = j
        self.k = k
        self._freeze()

    def __repr__(self):
        return f"AxisSet3D(i={self.i}, j={self.j}, k={self.k})"

    def __iter__(self):
        yield self.i
        yield self.j
        yield self.k

    def drop_dim(self, dimnum: int) -> Union["AxisSet2D", "AxisSet1D"]:
        """Remove a specified dimension from the axis set.

        Parameters
        ----------
        dimnum : int
            Index of the dimension to drop (0-based)

        Returns
        -------
        AxisSet2D or AxisSet1D
            New axis set with dimension removed

        """
        axes = list(self)
        new_axes = [ax for i, ax in enumerate(axes) if i != dimnum]

        if len(new_axes) == 2:
            return AxisSet2D(*new_axes)
        elif len(new_axes) == 1:
            return AxisSet1D(new_axes[0])
        else:
            raise ValueError(f"Invalid dimnum {dimnum} for AxisSet3D")

class AxisSet4D(AxisSet):
    """4D axis set."""

    def __init__(self, i: NamedAxis, j: NamedAxis, k: NamedAxis, l: NamedAxis):
        super().__init__(4)
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self._freeze()

    def __repr__(self):
        return f"AxisSet4D(i={self.i}, j={self.j}, k={self.k}, l={self.l})"

    def __iter__(self):
        yield self.i
        yield self.j
        yield self.k
        yield self.l

class AxisSet5D(AxisSet):
    """5D axis set."""

    def __init__(
        self, i: NamedAxis, j: NamedAxis, k: NamedAxis, l: NamedAxis, m: NamedAxis
    ):
        super().__init__(5)
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.m = m
        self._freeze()

    def __repr__(self):
        return f"AxisSet5D(i={self.i}, j={self.j}, k={self.k}, l={self.l}, m={self.m})"

    def __iter__(self):
        yield self.i
        yield self.j
        yield self.k
        yield self.l
        yield self.m

def axis_set(
    ndim: int, axes: List[Union[str, NamedAxis]] = None
) -> Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND]:
    if axes is None:
        axes = ["x", "y", "z", "t", "v"][:ndim]
        if ndim > 5:
            axes.extend([f"d{i}" for i in range(6, ndim + 1)])

    if len(axes) != ndim:
        raise ValueError(f"Number of axes ({len(axes)}) must match ndim ({ndim})")

    named_axes = [ax if isinstance(ax, NamedAxis) else NamedAxis(ax, 1) for ax in axes]

    if ndim == 1:
        return AxisSet1D(named_axes[0])
    elif ndim == 2:
        return AxisSet2D(named_axes[0], named_axes[1])
    elif ndim == 3:
        return AxisSet3D(named_axes[0], named_axes[1], named_axes[2])
    elif ndim == 4:
        return AxisSet4D(named_axes[0], named_axes[1], named_axes[2], named_axes[3])
    elif ndim == 5:
        return AxisSet5D(
            named_axes[0], named_axes[1], named_axes[2], named_axes[3], named_axes[4]
        )
    else:
        return AxisSetND(named_axes)

def axis_names(
    x: Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND],
) -> List[str]:
    if isinstance(x, AxisSetND):
        return [ax.axis for ax in x.axes]
    return [getattr(x, attr).axis for attr in "ijklm"[: x.ndim]]

def axis_directions(
    x: Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND],
) -> List[int]:
    if isinstance(x, AxisSetND):
        return [ax.direction for ax in x.axes]
    return [getattr(x, attr).direction for attr in "ijklm"[: x.ndim]]

def flip_axis(
    x: Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND],
    which: Union[str, int],
) -> Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND]:
    if isinstance(which, str):
        which = axis_names(x).index(which)

    opposite = {
        "Left-to-Right": RIGHT_LEFT,
        "Right-to-Left": LEFT_RIGHT,
        "Anterior-to-Posterior": POST_ANT,
        "Posterior-to-Anterior": ANT_POST,
        "Inferior-to-Superior": SUP_INF,
        "Superior-to-Inferior": INF_SUP,
    }

    new_axes = []
    current_axes = (
        list(x)
        if isinstance(x, AxisSetND)
        else [getattr(x, attr) for attr in "ijklm"[: x.ndim]]
    )
    for i, axis in enumerate(current_axes):
        if i == which:
            new_axes.append(
                opposite.get(axis.axis, NamedAxis(axis.axis, -axis.direction))
            )
        else:
            new_axes.append(axis)

    return axis_set(x.ndim, new_axes)

def permute_axes(
    x: Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND],
    perm: List[int],
) -> Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND]:
    if len(perm) != x.ndim:
        raise ValueError(
            f"Length of permutation ({len(perm)}) must match ndim ({x.ndim})"
        )

    current_axes = (
        list(x)
        if isinstance(x, AxisSetND)
        else [getattr(x, attr) for attr in "ijklm"[: x.ndim]]
    )
    new_axes = [current_axes[i] for i in perm]

    return axis_set(x.ndim, new_axes)

def drop_axis(
    x: Union[AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND],
    which: Union[str, int],
) -> Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND]:
    if isinstance(which, str):
        which = axis_names(x).index(which)

    current_axes = (
        list(x)
        if isinstance(x, AxisSetND)
        else [getattr(x, attr) for attr in "ijklm"[: x.ndim]]
    )
    new_axes = [ax for i, ax in enumerate(current_axes) if i != which]

    return axis_set(x.ndim - 1, new_axes)

def add_axis(
    x: Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND],
    new_axis: str,
) -> Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND]:
    current_axes = (
        list(x)
        if isinstance(x, AxisSetND)
        else [getattr(x, attr) for attr in "ijklm"[: x.ndim]]
    )
    current_axis_names = [ax.axis for ax in current_axes]
    if new_axis in current_axis_names:
        raise ValueError(f"Axis {new_axis} already exists")

    return axis_set(x.ndim + 1, current_axes + [new_axis])

# Predefined 3D orientations (matching R's neuroim2)
OrientationList3D = {
    "AXIAL_LPI": AxisSet3D(LEFT_RIGHT, POST_ANT, INF_SUP),
    "AXIAL_LAI": AxisSet3D(LEFT_RIGHT, ANT_POST, INF_SUP),
    "AXIAL_RPI": AxisSet3D(RIGHT_LEFT, POST_ANT, INF_SUP),
    "AXIAL_RAI": AxisSet3D(RIGHT_LEFT, ANT_POST, INF_SUP),
    "AXIAL_LPS": AxisSet3D(LEFT_RIGHT, POST_ANT, SUP_INF),
    "AXIAL_LAS": AxisSet3D(LEFT_RIGHT, ANT_POST, SUP_INF),
    "AXIAL_RPS": AxisSet3D(RIGHT_LEFT, POST_ANT, SUP_INF),
    "AXIAL_RAS": AxisSet3D(RIGHT_LEFT, ANT_POST, SUP_INF),
    "CORONAL_LIP": AxisSet3D(LEFT_RIGHT, INF_SUP, POST_ANT),
    "CORONAL_LIA": AxisSet3D(LEFT_RIGHT, INF_SUP, ANT_POST),
    "CORONAL_RIP": AxisSet3D(RIGHT_LEFT, INF_SUP, POST_ANT),
    "CORONAL_RIA": AxisSet3D(RIGHT_LEFT, INF_SUP, ANT_POST),
    "CORONAL_LSP": AxisSet3D(LEFT_RIGHT, SUP_INF, POST_ANT),
    "CORONAL_LSA": AxisSet3D(LEFT_RIGHT, SUP_INF, ANT_POST),
    "CORONAL_RSP": AxisSet3D(RIGHT_LEFT, SUP_INF, POST_ANT),
    "CORONAL_RSA": AxisSet3D(RIGHT_LEFT, SUP_INF, ANT_POST),
    "SAGITTAL_PLI": AxisSet3D(POST_ANT, LEFT_RIGHT, INF_SUP),
    "SAGITTAL_PRI": AxisSet3D(POST_ANT, RIGHT_LEFT, INF_SUP),
    "SAGITTAL_ALI": AxisSet3D(ANT_POST, LEFT_RIGHT, INF_SUP),
    "SAGITTAL_ARI": AxisSet3D(ANT_POST, RIGHT_LEFT, INF_SUP),
    "SAGITTAL_PLS": AxisSet3D(POST_ANT, LEFT_RIGHT, SUP_INF),
    "SAGITTAL_PRS": AxisSet3D(POST_ANT, RIGHT_LEFT, SUP_INF),
    "SAGITTAL_ALS": AxisSet3D(ANT_POST, LEFT_RIGHT, SUP_INF),
    "SAGITTAL_ARS": AxisSet3D(ANT_POST, RIGHT_LEFT, SUP_INF),
}

# Predefined 2D orientations
OrientationList2D = {
    "AXIAL_LP": AxisSet2D(LEFT_RIGHT, POST_ANT),
    "AXIAL_LA": AxisSet2D(LEFT_RIGHT, ANT_POST),
    "AXIAL_RP": AxisSet2D(RIGHT_LEFT, POST_ANT),
    "AXIAL_RA": AxisSet2D(RIGHT_LEFT, ANT_POST),
    "CORONAL_LI": AxisSet2D(LEFT_RIGHT, INF_SUP),
    "CORONAL_LS": AxisSet2D(LEFT_RIGHT, SUP_INF),
    "CORONAL_RI": AxisSet2D(RIGHT_LEFT, INF_SUP),
    "CORONAL_RS": AxisSet2D(RIGHT_LEFT, SUP_INF),
    "SAGITTAL_PI": AxisSet2D(POST_ANT, INF_SUP),
    "SAGITTAL_PS": AxisSet2D(POST_ANT, SUP_INF),
    "SAGITTAL_AI": AxisSet2D(ANT_POST, INF_SUP),
    "SAGITTAL_AS": AxisSet2D(ANT_POST, SUP_INF),
}

def match_axis(axis_char: str) -> NamedAxis:
    """Match axis character to NamedAxis constant.

    Parameters
    ----------
    axis_char : str
        Single character axis code ('L', 'R', 'A', 'P', 'I', 'S')

    Returns
    -------
    NamedAxis
        Corresponding anatomical axis

    """
    normalized = (
        str(axis_char).replace("-", "").replace("_", "").replace(" ", "").upper()
    )
    full_to_short = {
        "LEFT": "L",
        "RIGHT": "R",
        "ANTERIOR": "A",
        "POSTERIOR": "P",
        "INFERIOR": "I",
        "SUPERIOR": "S",
        "LEFTTORIGHT": "L",
        "RIGHTTOLEFT": "R",
        "ANTERIORTOANTERIOR": "A",
        "POSTERIORTOANTERIOR": "P",
        "INFERIORTOSUPERIOR": "I",
        "SUPERIORTOINFERIOR": "S",
    }

    axis_key = full_to_short.get(normalized, normalized)

    mapping = {
        "L": LEFT_RIGHT,
        "R": RIGHT_LEFT,
        "A": ANT_POST,
        "P": POST_ANT,
        "I": INF_SUP,
        "S": SUP_INF,
    }

    if axis_key not in mapping:
        raise ValueError(f"Unknown axis character: {axis_char}")

    return mapping[axis_key]

def find_anatomy_3d(axis_codes: str) -> AxisSet3D:
    """Create AxisSet3D from 3-letter axis code.

    Parameters
    ----------
    axis_codes : str
        3-letter code like 'LPI', 'RAS', etc.

    Returns
    -------
    AxisSet3D
        Corresponding 3D axis set

    """
    if len(axis_codes) != 3:
        raise ValueError(f"axis_codes must be 3 characters, got {len(axis_codes)}")

    axes = [match_axis(c) for c in axis_codes]
    return AxisSet3D(*axes)

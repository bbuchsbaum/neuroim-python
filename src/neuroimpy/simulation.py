"""Simulation utilities for generating synthetic neuroimaging data.

Provides functions for creating synthetic fMRI data, preparing confound
matrices, and generating time weights for analysis.
"""

import numpy as np
from typing import Optional
from .neuro_space import NeuroSpace
from .neuro_vec import DenseNeuroVec


def _gamma_hrf(t, peak=6.0, undershoot=16.0, peak_amp=1.0, undershoot_amp=0.167):
    """Simple gamma-function HRF.

    Parameters
    ----------
    t : np.ndarray
        Time points in seconds.
    peak : float
        Time to peak (seconds).
    undershoot : float
        Time to undershoot (seconds).
    peak_amp : float
        Amplitude of the peak.
    undershoot_amp : float
        Amplitude of the undershoot relative to peak.

    Returns
    -------
    np.ndarray
        HRF values at each time point.
    """
    from scipy.stats import gamma as gamma_dist

    h = peak_amp * gamma_dist.pdf(t, peak) - undershoot_amp * gamma_dist.pdf(t, undershoot)
    # Normalize to unit peak
    if h.max() != 0:
        h = h / h.max()
    return h


def simulate_fmri(
    space: NeuroSpace,
    n_timepoints: int,
    n_regions: int = 5,
    noise_sd: float = 1.0,
    signal_sd: float = 2.0,
    tr: float = 2.0,
    seed: Optional[int] = None,
) -> DenseNeuroVec:
    """Generate synthetic fMRI data with block-design activation.

    Places random spherical activation regions in the volume, generates
    block-design signals convolved with a simple HRF (gamma function),
    and adds Gaussian noise.

    Parameters
    ----------
    space : NeuroSpace
        A 3D NeuroSpace defining the volume geometry.
    n_timepoints : int
        Number of time points to generate.
    n_regions : int
        Number of random spherical activation regions to place.
    noise_sd : float
        Standard deviation of additive Gaussian noise.
    signal_sd : float
        Standard deviation (amplitude) of the activation signal.
    tr : float
        Repetition time in seconds.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DenseNeuroVec
        A 4D dense neuroimaging vector with synthetic data.
    """
    rng = np.random.default_rng(seed)

    dims = tuple(int(d) for d in space.dim[:3])
    data = rng.normal(0, noise_sd, size=(*dims, n_timepoints))

    # Build HRF kernel
    hrf_times = np.arange(0, 30, tr)
    hrf = _gamma_hrf(hrf_times)

    # Block-design stimulus (alternating on/off blocks of ~10 TRs)
    block_len = 10
    stimulus = np.zeros(n_timepoints)
    on = True
    for start in range(0, n_timepoints, block_len):
        if on:
            stimulus[start : start + block_len] = 1.0
        on = not on

    # Convolve stimulus with HRF
    convolved = np.convolve(stimulus, hrf)[:n_timepoints]
    # Normalise to unit range then scale
    if convolved.max() - convolved.min() > 0:
        convolved = (convolved - convolved.min()) / (convolved.max() - convolved.min())
    convolved = convolved * signal_sd

    # Place spherical activation regions
    margin = 2
    max_radius = min(dims) // 4
    for _ in range(n_regions):
        # Random centre within the volume
        cx = rng.integers(margin, dims[0] - margin)
        cy = rng.integers(margin, dims[1] - margin)
        cz = rng.integers(margin, dims[2] - margin)
        radius = rng.integers(1, max(2, max_radius))

        # Build a spherical mask
        ix, iy, iz = np.ogrid[
            max(0, cx - radius) : min(dims[0], cx + radius + 1),
            max(0, cy - radius) : min(dims[1], cy + radius + 1),
            max(0, cz - radius) : min(dims[2], cz + radius + 1),
        ]
        dist2 = (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2
        sphere = dist2 <= radius**2

        # Add convolved signal within the sphere
        data[
            max(0, cx - radius) : min(dims[0], cx + radius + 1),
            max(0, cy - radius) : min(dims[1], cy + radius + 1),
            max(0, cz - radius) : min(dims[2], cz + radius + 1),
            :,
        ][sphere] += convolved[np.newaxis, :]

    # Build 4D space
    spacing_4d = np.append(space.spacing[:3], tr)
    origin_4d = np.append(space.origin[:3], 0.0)
    space_4d = NeuroSpace(
        dim=(*dims, n_timepoints),
        spacing=spacing_4d,
        origin=origin_4d,
    )

    return DenseNeuroVec(data, space_4d)


def prepare_confounds(
    motion_params: np.ndarray,
    include_derivatives: bool = True,
    include_squared: bool = False,
) -> np.ndarray:
    """Prepare a confound matrix from motion parameters.

    Takes an (n_timepoints x 6) array of rigid-body motion parameters and
    optionally augments it with temporal derivatives and squared terms.

    Parameters
    ----------
    motion_params : np.ndarray
        Motion parameters of shape (n_timepoints, 6).
    include_derivatives : bool
        If True, append first temporal derivatives (diff, zero-padded).
    include_squared : bool
        If True, append squared terms (applied after derivatives if both
        are True).

    Returns
    -------
    np.ndarray
        Expanded confound matrix.
    """
    motion_params = np.asarray(motion_params, dtype=float)
    if motion_params.ndim != 2 or motion_params.shape[1] != 6:
        raise ValueError("motion_params must be an (n_timepoints, 6) array")

    parts = [motion_params]

    if include_derivatives:
        deriv = np.diff(motion_params, axis=0, prepend=motion_params[:1, :])
        parts.append(deriv)

    combined = np.hstack(parts)

    if include_squared:
        combined = np.hstack([combined, combined ** 2])

    return combined


def make_time_weights(
    n_timepoints: int,
    method: str = "exponential",
    decay: float = 0.1,
) -> np.ndarray:
    """Generate normalised time-point weights.

    Parameters
    ----------
    n_timepoints : int
        Number of time points.
    method : str
        Weighting method: ``"exponential"``, ``"linear"``, or ``"uniform"``.
    decay : float
        Decay rate for ``"exponential"`` method.

    Returns
    -------
    np.ndarray
        1-D array of weights summing to 1.

    Raises
    ------
    ValueError
        If *method* is not one of the supported options.
    """
    t = np.arange(n_timepoints, dtype=float)

    if method == "exponential":
        w = np.exp(-decay * t)
    elif method == "linear":
        if n_timepoints == 1:
            w = np.ones(1)
        else:
            T = n_timepoints - 1
            w = t / T
    elif method == "uniform":
        w = np.ones(n_timepoints)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'exponential', 'linear', or 'uniform'.")

    total = w.sum()
    if total > 0:
        w = w / total

    return w

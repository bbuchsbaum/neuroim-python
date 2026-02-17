"""Connectivity-graph-based spatial filtering functions.

Provides graph-based smoothing, nuisance regression, and Laplacian
enhancement for neuroimaging data using sparse adjacency matrices.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Union
from .neuro_vol import NeuroVol, DenseNeuroVol, LogicalNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec
from .neuro_space import NeuroSpace


def cgb_make_graph(vol_or_space, mask=None):
    """Build a 6-connected adjacency graph from a volume or space.

    Each voxel is a node, and edges connect face-adjacent (6-neighbor)
    voxels.  When a mask is supplied, only masked voxels are included.

    Parameters
    ----------
    vol_or_space : NeuroVol, NeuroVec, or NeuroSpace
        Source of spatial dimensions.
    mask : LogicalNeuroVol or np.ndarray, optional
        Boolean mask selecting which voxels to include.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric binary adjacency matrix of shape ``(n, n)`` where
        ``n`` is the number of included voxels.
    """
    # Resolve shape
    if isinstance(vol_or_space, NeuroSpace):
        dims = tuple(int(d) for d in vol_or_space.dim[:3])
    elif hasattr(vol_or_space, "space"):
        dims = tuple(int(d) for d in vol_or_space.space.dim[:3])
    else:
        raise TypeError("vol_or_space must be a NeuroVol, NeuroVec, or NeuroSpace")

    nx, ny, nz = dims

    # Resolve mask
    if mask is not None:
        if isinstance(mask, LogicalNeuroVol):
            mask_arr = mask.data.astype(bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
    else:
        mask_arr = np.ones((nx, ny, nz), dtype=bool)

    # Map masked voxels to contiguous indices
    n_voxels = int(mask_arr.sum())
    if n_voxels == 0:
        return sparse.csr_matrix((0, 0), dtype=float)

    # Create index volume: -1 for unmasked, contiguous int for masked
    idx_vol = -np.ones((nx, ny, nz), dtype=np.intp)
    idx_vol[mask_arr] = np.arange(n_voxels)

    # 6-connectivity offsets (face-adjacent)
    offsets = [(1, 0, 0), (-1, 0, 0),
               (0, 1, 0), (0, -1, 0),
               (0, 0, 1), (0, 0, -1)]

    rows = []
    cols = []

    coords = np.argwhere(mask_arr)  # (n_voxels, 3)
    for dx, dy, dz in offsets:
        nbr = coords + np.array([dx, dy, dz])
        # Check bounds
        valid = (
            (nbr[:, 0] >= 0) & (nbr[:, 0] < nx) &
            (nbr[:, 1] >= 0) & (nbr[:, 1] < ny) &
            (nbr[:, 2] >= 0) & (nbr[:, 2] < nz)
        )
        nbr_valid = nbr[valid]
        # Check that neighbor is in mask
        nbr_idx = idx_vol[nbr_valid[:, 0], nbr_valid[:, 1], nbr_valid[:, 2]]
        in_mask = nbr_idx >= 0
        src_idx = idx_vol[coords[valid][in_mask, 0],
                          coords[valid][in_mask, 1],
                          coords[valid][in_mask, 2]]
        rows.append(src_idx)
        cols.append(nbr_idx[in_mask])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=float)

    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_voxels, n_voxels))
    return adj


def cgb_filter(data, graph, weights=None):
    """Graph-based filter: smooth data by averaging neighbor values.

    Parameters
    ----------
    data : np.ndarray
        1-D array of values, one per graph node.
    graph : scipy.sparse.csr_matrix
        Adjacency matrix from :func:`cgb_make_graph`.
    weights : np.ndarray, optional
        Edge-weight matrix (same shape as *graph*).  If ``None``,
        uniform weights derived from the adjacency are used.

    Returns
    -------
    np.ndarray
        Smoothed 1-D array, same length as *data*.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("data must be 1-D")

    if weights is not None:
        W = sparse.csr_matrix(weights)
    else:
        W = graph.astype(float)

    # Row-normalise so each row sums to 1
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for isolated nodes
    D_inv = sparse.diags(1.0 / row_sums)
    W_norm = D_inv @ W

    return np.asarray(W_norm @ data).ravel()


def cgb_smooth(vol, fwhm, mask=None):
    """Graph-based Gaussian smoothing.

    Builds a 6-connectivity graph, computes Gaussian weights based on
    *fwhm* and unit voxel distance, and iteratively averages neighbour
    values.

    Parameters
    ----------
    vol : NeuroVol
        Volume to smooth.
    fwhm : float
        Full width at half maximum (in voxels) controlling smoothing
        extent.
    mask : LogicalNeuroVol or np.ndarray, optional
        Mask selecting voxels to include.

    Returns
    -------
    DenseNeuroVol
        Smoothed volume.
    """
    if fwhm <= 0:
        raise ValueError("fwhm must be positive")

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    graph = cgb_make_graph(vol, mask=mask)
    n = graph.shape[0]
    if n == 0:
        return DenseNeuroVol(vol.data.copy(), vol.space)

    # Resolve mask array
    if mask is not None:
        if isinstance(mask, LogicalNeuroVol):
            mask_arr = mask.data.astype(bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
    else:
        mask_arr = np.ones(vol.shape[:3], dtype=bool)

    # Build Gaussian-weighted adjacency.
    # For 6-connected graph all edges have unit voxel distance.
    gauss_weight = np.exp(-1.0 / (2.0 * sigma ** 2))
    W = graph.astype(float) * gauss_weight

    # Add self-connections with weight 1
    W = W + sparse.eye(n, dtype=float)

    # Row normalise
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    W_norm = D_inv @ W

    # Extract masked data
    values = vol.data[mask_arr].astype(float)

    # Number of iterations scales with sigma
    n_iter = max(1, int(np.ceil(sigma)))
    for _ in range(n_iter):
        values = np.asarray(W_norm @ values).ravel()

    # Write back
    out = vol.data.copy().astype(float)
    out[mask_arr] = values
    return DenseNeuroVol(out, vol.space)


def cgb_smooth_loro(vec, fwhm, run_labels, mask=None):
    """Leave-one-run-out graph smoothing.

    For each run, smoothing is applied using data from all *other* runs.
    This prevents temporal leakage between runs in cross-validation
    designs.

    Parameters
    ----------
    vec : NeuroVec
        4-D data (x, y, z, time).
    fwhm : float
        Full width at half maximum for the Gaussian graph kernel.
    run_labels : array-like
        Integer run label for each time point (length must equal the
        number of volumes in *vec*).
    mask : LogicalNeuroVol or np.ndarray, optional
        Mask selecting voxels.

    Returns
    -------
    DenseNeuroVec
        Smoothed 4-D data.
    """
    run_labels = np.asarray(run_labels)
    n_time = vec.shape[3]
    if len(run_labels) != n_time:
        raise ValueError("run_labels length must match number of volumes")

    if fwhm <= 0:
        raise ValueError("fwhm must be positive")

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    graph = cgb_make_graph(vec, mask=mask)
    n = graph.shape[0]

    # Resolve mask
    if mask is not None:
        if isinstance(mask, LogicalNeuroVol):
            mask_arr = mask.data.astype(bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
    else:
        mask_arr = np.ones(vec.shape[:3], dtype=bool)

    # Build normalised weight matrix
    gauss_weight = np.exp(-1.0 / (2.0 * sigma ** 2))
    W = graph.astype(float) * gauss_weight
    W = W + sparse.eye(n, dtype=float)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    W_norm = D_inv @ W

    n_iter = max(1, int(np.ceil(sigma)))
    unique_runs = np.unique(run_labels)

    out_data = vec.data.copy().astype(float)

    for run in unique_runs:
        other_mask = run_labels != run
        run_mask = run_labels == run
        # Mean of other-run volumes
        other_mean = out_data[:, :, :, other_mask].mean(axis=3)
        other_vals = other_mean[mask_arr]

        # Smooth using the graph
        smoothed = other_vals.copy()
        for _ in range(n_iter):
            smoothed = np.asarray(W_norm @ smoothed).ravel()

        # Apply smoothed values to the current-run time points
        run_indices = np.where(run_mask)[0]
        for t in run_indices:
            out_data[:, :, :, t][mask_arr] = smoothed

    return DenseNeuroVec(out_data, vec.space)


def cgb_nuisance(vol, confounds, graph, mask=None):
    """Graph-based nuisance regression.

    For each voxel and its graph neighbours, regress out confound
    variables and return residuals.

    Parameters
    ----------
    vol : NeuroVol
        Volume whose values will be cleaned.
    confounds : np.ndarray
        Confound matrix, shape ``(n_voxels, p)`` where *p* is the number
        of confound regressors. The number of rows must match the number
        of voxels in the graph.
    graph : scipy.sparse.csr_matrix
        Adjacency matrix from :func:`cgb_make_graph`.
    mask : LogicalNeuroVol or np.ndarray, optional
        Mask selecting voxels.

    Returns
    -------
    DenseNeuroVol
        Volume of residuals after confound removal.
    """
    # Resolve mask
    if mask is not None:
        if isinstance(mask, LogicalNeuroVol):
            mask_arr = mask.data.astype(bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
    else:
        mask_arr = np.ones(vol.shape[:3], dtype=bool)

    values = vol.data[mask_arr].astype(float)
    confounds = np.asarray(confounds, dtype=float)
    if confounds.ndim == 1:
        confounds = confounds[:, np.newaxis]
    n = graph.shape[0]

    if confounds.shape[0] != n:
        raise ValueError("confounds rows must equal number of graph nodes")

    residuals = np.empty_like(values)
    graph_csr = sparse.csr_matrix(graph)

    for i in range(n):
        # Neighbourhood: self + connected voxels
        nbrs = graph_csr[i].indices
        hood = np.concatenate([[i], nbrs])

        y = values[hood]
        X = confounds[hood]

        # Add intercept
        X_aug = np.column_stack([np.ones(len(hood)), X])
        # OLS
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        predicted_center = X_aug[0] @ beta  # prediction for the center voxel
        residuals[i] = values[i] - predicted_center

    out = vol.data.copy().astype(float)
    out[mask_arr] = residuals
    return DenseNeuroVol(out, vol.space)


def laplace_enhance(vol, alpha=0.5, mask=None):
    """Laplacian enhancement using the graph Laplacian.

    ``enhanced = original + alpha * L @ original``

    where *L = D - A* is the combinatorial graph Laplacian.

    Parameters
    ----------
    vol : NeuroVol
        Volume to enhance.
    alpha : float
        Enhancement strength (default 0.5).
    mask : LogicalNeuroVol or np.ndarray, optional
        Mask selecting voxels.

    Returns
    -------
    DenseNeuroVol
        Enhanced volume.
    """
    graph = cgb_make_graph(vol, mask=mask)
    n = graph.shape[0]
    if n == 0:
        return DenseNeuroVol(vol.data.copy(), vol.space)

    # Resolve mask
    if mask is not None:
        if isinstance(mask, LogicalNeuroVol):
            mask_arr = mask.data.astype(bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
    else:
        mask_arr = np.ones(vol.shape[:3], dtype=bool)

    values = vol.data[mask_arr].astype(float)

    # Graph Laplacian L = D - A
    degree = np.asarray(graph.sum(axis=1)).ravel()
    D = sparse.diags(degree)
    L = D - graph

    enhanced = values + alpha * np.asarray(L @ values).ravel()

    out = vol.data.copy().astype(float)
    out[mask_arr] = enhanced
    return DenseNeuroVol(out, vol.space)

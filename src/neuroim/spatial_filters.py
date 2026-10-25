"""Spatial filtering functions for neuroimaging data.

Spatial filtering functions for typed neuroimaging volumes.
"""

import numpy as np
from scipy import ndimage
from typing import Union, Optional
from .neuro_vol import NeuroVol, DenseNeuroVol, LogicalNeuroVol
from .neuro_vec import NeuroVec, DenseNeuroVec
from .kernel import Kernel, gaussian_kernel


def _positive_finite_scalar(value, name: str) -> float:
    arr = np.asarray(value)
    if arr.ndim != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be one positive finite numeric scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be one positive finite numeric scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} must be one positive finite numeric scalar")
    return scalar


def _gaussian_support(window, truncate, shape) -> tuple[Optional[int], float]:
    truncate_value = _positive_finite_scalar(truncate, "truncate")
    if window is None:
        return None, truncate_value
    arr = np.asarray(window)
    if arr.ndim != 0 or isinstance(window, (bool, np.bool_)):
        raise ValueError("window must be one positive integer or None")
    try:
        value = float(window)
    except (TypeError, ValueError) as exc:
        raise ValueError("window must be one positive integer or None") from exc
    if not np.isfinite(value) or value < 1 or not value.is_integer():
        raise ValueError("window must be one positive integer or None")
    return min(int(value), max(int(d) for d in shape[:3])), truncate_value


def _gaussian_sigma(space, sigma, fwhm_mm):
    if fwhm_mm is None:
        return _positive_finite_scalar(sigma, "sigma")
    fwhm = _positive_finite_scalar(fwhm_mm, "fwhm_mm")
    spacing = np.asarray(space.spacing[:3], dtype=float)
    return (fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / spacing


def _insulated_gaussian(data, mask_bool, sigma, truncate, radius):
    """Smooth-in-mask Gaussian: in-mask neighbours only, renormalized.

    Computes ``blur(x * m) / blur(m)`` with zero padding, so out-of-mask values
    (including NaN/Inf) never reach in-mask outputs and edge voxels are not
    pulled toward the exterior (neuroim2 ``normalize = TRUE``; cf. AFNI
    3dBlurInMask). Only in-mask voxels of the result are meaningful.
    """
    weights = mask_bool.astype(float)
    if data.ndim == 4:
        weights = np.broadcast_to(weights[..., None], data.shape)
    masked = np.where(weights > 0, data, 0.0)
    kwargs = dict(sigma=sigma, truncate=truncate, radius=radius, mode="constant")
    num = ndimage.gaussian_filter(masked, **kwargs)
    den = ndimage.gaussian_filter(np.ascontiguousarray(weights), **kwargs)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def gaussian_blur(
    vol: NeuroVol,
    mask: Optional[LogicalNeuroVol] = None,
    sigma: float = 2,
    window: Optional[int] = None,
    *,
    fwhm_mm: Optional[float] = None,
    truncate: float = 4.0,
    normalize: bool = True,
) -> DenseNeuroVol:
    """Apply a Gaussian blur to a volumetric image.

    For fMRI workflows, prefer ``fwhm_mm=`` over the legacy ``sigma=``.  The
    R neuroim2 ``gaussian_blur`` interprets sigma in millimetres and applies
    per-axis voxel spacing inside its C++ kernel; the Python port historically
    accepted sigma in *voxel* units and did not apply spacing, which silently
    produced anisotropic-in-mm smoothing on anisotropic voxels.  Passing
    ``fwhm_mm`` restores the mm-space contract: the value is converted to
    per-axis voxel sigma via ``vol.space.spacing``, so smoothing is
    isotropic in millimetres regardless of voxel anisotropy.

    Parameters
    ----------
    vol : NeuroVol
        The image volume to be smoothed.
    mask : LogicalNeuroVol, optional
        Image mask defining the region where blurring is applied.  When
        supplied, ``verify.assert_same_space(vol, mask)`` is invoked first
        so a foreign-affine mask raises before any smoothing occurs.
    sigma : float, optional
        Voxel-space sigma for the legacy scalar form.  Used only when
        ``fwhm_mm`` is None.  Default is 2 voxels.
    window : int, optional
        Explicit kernel half-width in voxels. ``window=1`` is exactly a
        3x3x3 kernel. The default ``None`` derives support from the Gaussian
        width and ``truncate``, avoiding the under-smoothing caused by the old
        fixed default. Explicit values retain the legacy clipped-support path.
    fwhm_mm : float, optional (keyword only)
        Full width at half maximum, in millimetres.  When supplied, this
        takes precedence over ``sigma`` and is converted to per-axis voxel
        sigma using ``vol.space.spacing``.
    truncate : float, optional (keyword only)
        Standard deviations retained when ``window`` is ``None``. Default 4,
        matching :func:`scipy.ndimage.gaussian_filter`.
    normalize : bool, optional (keyword only)
        How the mask boundary is handled when ``mask`` is given. ``True``
        (default, as in neuroim2) insulates the blur to the mask: each
        in-mask voxel is computed from in-mask neighbours only and the kernel
        is renormalized by the in-mask weight, so out-of-mask values -- finite
        or NaN/Inf -- never leak in and edges are not biased toward the
        exterior. ``False`` keeps the legacy behaviour: the whole volume is
        blurred and in-mask voxels are copied from it. Voxels outside the mask
        keep their input values either way.

    Returns
    -------
    DenseNeuroVol
        The smoothed image.  Carries a populated
        :class:`~neuroim.results.Receipt` in ``.provenance`` recording
        ``method_name="gaussian_blur"`` or an upstream-chained method name
        such as ``"temporal_snr+gaussian_blur"``, the FWHM (when supplied),
        and the mask hash.

    A 4-D :class:`~neuroim.neuro_vec.NeuroVec` is smoothed spatially,
    volume-by-volume (no temporal blur), and a
    :class:`~neuroim.neuro_vec.DenseNeuroVec` is returned.
    """
    if isinstance(vol, NeuroVec):
        return _gaussian_blur_vec(
            vol,
            mask=mask,
            sigma=sigma,
            window=window,
            fwhm_mm=fwhm_mm,
            truncate=truncate,
            normalize=normalize,
        )
    radius_used, truncate_used = _gaussian_support(window, truncate, vol.shape)
    sigma_used: Union[float, np.ndarray] = _gaussian_sigma(vol.space, sigma, fwhm_mm)

    if mask is not None:
        from .verify import assert_same_space

        assert_same_space(vol, mask)

    data = vol.data
    if mask is not None and normalize:
        blurred_data = _insulated_gaussian(
            np.asarray(data, dtype=float),
            np.asarray(mask.data, dtype=bool),
            sigma_used,
            truncate_used,
            radius_used,
        )
    else:
        blurred_data = ndimage.gaussian_filter(
            data,
            sigma=sigma_used,
            truncate=truncate_used,
            radius=radius_used,
        )
    if mask is not None:
        output_data = data.copy()
        mask_indices = np.where(mask.data)
        output_data[mask_indices] = blurred_data[mask_indices]
    else:
        output_data = blurred_data

    from .results import SpatialFilterParams, receipt_for

    mask_payload = np.asarray(mask.data, dtype=bool) if mask is not None else None
    receipt = receipt_for(
        vol,
        mask=mask_payload,
        n_voxels=int(np.prod(vol.shape)) if mask is None else int(mask_payload.sum()),
        params=SpatialFilterParams(
            method_name="gaussian_blur",
            radius=float(fwhm_mm) if fwhm_mm is not None else float(sigma),
        ),
        upstream=vol,
    )
    return DenseNeuroVol(output_data, vol.space, provenance=receipt)


def _gaussian_blur_vec(
    vec: NeuroVec,
    *,
    mask: Optional[LogicalNeuroVol] = None,
    sigma: float = 2,
    window: Optional[int] = None,
    fwhm_mm: Optional[float] = None,
    truncate: float = 4.0,
    normalize: bool = True,
) -> DenseNeuroVec:
    """Spatially Gaussian-blur each volume of a 4-D NeuroVec.

    Mirrors :func:`gaussian_blur`'s sigma/FWHM contract on the three spatial
    axes and applies zero blur along time, so the canonical fMRI smoothing
    step is first-class for time-series input.
    """
    spatial = vec.spatial_space
    radius_used, truncate_used = _gaussian_support(window, truncate, vec.shape)
    spatial_sigma = _gaussian_sigma(spatial, sigma, fwhm_mm)

    if mask is not None:
        from .verify import assert_same_space

        assert_same_space(vec, mask)

    data = np.asarray(vec.to_dense().data, dtype=float)
    # SciPy 1.10 (the declared package minimum) supports ``radius`` but not
    # the later ``axes`` keyword. A zero sigma/radius on the trailing time
    # axis expresses the same spatial-only operation on that version.
    spatial_sigma_values = np.broadcast_to(np.asarray(spatial_sigma, dtype=float), (3,))
    sigma_4d = tuple(spatial_sigma_values) + (0.0,)
    radius_4d = None if radius_used is None else (radius_used,) * 3 + (0,)
    if mask is not None and normalize:
        blurred = _insulated_gaussian(
            data,
            np.asarray(mask.data, dtype=bool),
            sigma_4d,
            truncate_used,
            radius_4d,
        )
    else:
        blurred = ndimage.gaussian_filter(
            data,
            sigma=sigma_4d,
            truncate=truncate_used,
            radius=radius_4d,
        )

    if mask is not None:
        mask_bool = np.asarray(mask.data, dtype=bool)
        output = data.copy()
        output[mask_bool] = blurred[mask_bool]
    else:
        mask_bool = None
        output = blurred

    from .results import SpatialFilterParams, receipt_for

    n_voxels = (
        int(np.prod(vec.shape[:3])) if mask_bool is None else int(mask_bool.sum())
    )
    receipt = receipt_for(
        vec,
        mask=mask_bool,
        n_voxels=n_voxels,
        params=SpatialFilterParams(
            method_name="gaussian_blur",
            radius=float(fwhm_mm) if fwhm_mm is not None else float(sigma),
        ),
        upstream=vec,
    )
    result = DenseNeuroVec(output, vec.space)
    result.provenance = receipt
    return result


def guided_filter(
    vol: NeuroVol, radius: int = 4, epsilon: float = 0.49
) -> DenseNeuroVol:
    """Apply edge-preserving guided filter to a volumetric image.

    This function applies a guided filter to perform edge-preserving smoothing.
    The guided filter smooths the image while preserving edges, providing a
    balance between noise reduction and structural preservation.

    Parameters
    ----------
    vol : NeuroVol
        The image volume to be filtered
    radius : int
        Spatial radius of the filter (default is 4)
    epsilon : float
        Regularization parameter controlling the degree of smoothing
        and edge preservation (default is 0.49)

    Returns
    -------
    DenseNeuroVol
        The filtered image

    """
    if radius < 1:
        raise ValueError("Radius must be at least 1")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")

    data = vol.data
    mask_indices = np.where(data != 0)

    # Apply box blur to compute local means
    mean_I = box_blur(vol, mask_indices, radius)

    # Create temporary volumes for squared data
    vol_squared = DenseNeuroVol(data * data, vol.space)
    mean_II = box_blur(vol_squared, mask_indices, radius)

    var_I = mean_II - mean_I * mean_I
    mean_p = mean_I
    mean_Ip = mean_II

    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I

    # Create temporary volumes for a and b
    a_full = np.zeros_like(data)
    b_full = np.zeros_like(data)
    a_full[mask_indices] = a
    b_full[mask_indices] = b

    a_vol = DenseNeuroVol(a_full, vol.space)
    b_vol = DenseNeuroVol(b_full, vol.space)

    mean_a = box_blur(a_vol, mask_indices, radius)
    mean_b = box_blur(b_vol, mask_indices, radius)

    # Apply the guided filter formula
    output_data = data.copy()
    output_data[mask_indices] = mean_a * data[mask_indices] + mean_b

    return DenseNeuroVol(output_data, vol.space)


def _check_range_scale(range_scale) -> None:
    if range_scale is not None and (not np.isfinite(range_scale) or range_scale <= 0):
        raise ValueError("range_scale must be None or a positive finite value")


def _bilateral_3d(
    data: np.ndarray,
    mask_arr: np.ndarray,
    spatial_sigma: float,
    intensity_sigma: float,
    window: int,
    spacing,
    range_scale: Optional[float],
) -> np.ndarray:
    """Bilateral filter of one 3-D array, following neuroim2's kernel.

    Spatial distances are in millimetres (``spacing``); only in-mask, finite
    neighbours inside the volume contribute and the weights are renormalized.
    The intensity bandwidth is ``intensity_sigma * range_scale``, where
    ``range_scale`` defaults to the sample SD of the finite in-mask values.
    """
    data = np.asarray(data, dtype=float)
    if range_scale is None:
        finite = data[mask_arr]
        finite = finite[np.isfinite(finite)]
        intensity_sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    else:
        intensity_sd = float(range_scale)
    intensity_denom = 2.0 * intensity_sigma**2 * intensity_sd**2
    if not np.isfinite(intensity_denom) or intensity_denom < 1e-12:
        intensity_denom = 1e-12
    spatial_denom = 2.0 * spatial_sigma**2
    spacing = np.asarray(spacing, dtype=float)[:3]

    shape = data.shape
    padded = np.pad(
        np.where(mask_arr, data, np.nan),
        window,
        mode="constant",
        constant_values=np.nan,
    )
    num = np.zeros(shape)
    den = np.zeros(shape)
    rng = range(-window, window + 1)
    for dx in rng:
        for dy in rng:
            for dz in rng:
                dist2 = (
                    (dx * spacing[0]) ** 2
                    + (dy * spacing[1]) ** 2
                    + (dz * spacing[2]) ** 2
                )
                spatial_w = np.exp(-dist2 / spatial_denom)
                neigh = padded[
                    window + dx : window + dx + shape[0],
                    window + dy : window + dy + shape[1],
                    window + dz : window + dz + shape[2],
                ]
                valid = np.isfinite(neigh)
                safe = np.where(valid, neigh, 0.0)
                with np.errstate(invalid="ignore"):
                    w = spatial_w * np.exp(-((safe - data) ** 2) / intensity_denom)
                w = np.where(valid & np.isfinite(w), w, 0.0)
                num += w * safe
                den += w

    out = data.copy()
    update = mask_arr & np.isfinite(data) & (den > 0)
    out[update] = num[update] / den[update]
    return out


def bilateral_filter(
    vol: NeuroVol,
    mask: Optional[LogicalNeuroVol] = None,
    spatial_sigma: float = 2,
    intensity_sigma: float = 1,
    window: int = 1,
    range_scale: Optional[float] = None,
) -> DenseNeuroVol:
    """Apply bilateral filter to a volumetric image.

    This function smooths a volumetric image using a bilateral filter.
    The bilateral filter considers both spatial closeness and intensity
    similarity for smoothing.

    Parameters
    ----------
    vol : NeuroVol
        The image volume to be smoothed
    mask : LogicalNeuroVol, optional
        Image mask defining the region where filtering is applied. Only
        in-mask neighbours contribute; voxels outside the mask are returned
        unchanged. If not provided, the entire volume is considered.
    spatial_sigma : float
        Standard deviation of the spatial Gaussian kernel, in millimetres
        (distances use the voxel spacing). Default is 2.
    intensity_sigma : float
        Width of the intensity kernel in units of ``range_scale``
        (default is 1).
    window : int
        Number of voxels around the center voxel to include on each side.
        For example, window=1 for a 3x3x3 kernel (default is 1)
    range_scale : float, optional
        Intensity scale of the range kernel; the effective bandwidth is
        ``intensity_sigma * range_scale``. If None, it is estimated as the
        standard deviation of this input's finite in-mask values, so equal
        ``intensity_sigma`` values do not imply equal bandwidths across
        subjects, contrasts or observed/null maps. For batch or group work,
        choose one fixed ``range_scale`` (e.g. from a reference or pooled
        in-mask distribution) and reuse it with the same ``intensity_sigma``.

    Returns
    -------
    DenseNeuroVol
        The smoothed image

    """
    if window < 1:
        raise ValueError("Window size must be at least 1")
    if spatial_sigma <= 0 or intensity_sigma <= 0:
        raise ValueError("Sigma values must be positive")
    _check_range_scale(range_scale)

    data = np.asarray(vol.data)
    mask_arr = (
        np.ones(data.shape, dtype=bool)
        if mask is None
        else np.asarray(mask.data, dtype=bool)
    )
    output_data = _bilateral_3d(
        data,
        mask_arr,
        spatial_sigma,
        intensity_sigma,
        int(window),
        vol.space.spacing[:3],
        range_scale,
    )
    return DenseNeuroVol(output_data, vol.space)


def bilateral_filter_vec(
    vec: NeuroVec,
    mask: Optional[LogicalNeuroVol] = None,
    spatial_sigma: float = 2,
    intensity_sigma: float = 1,
    window: int = 1,
    range_scale: Optional[float] = None,
) -> DenseNeuroVec:
    """Apply bilateral filter to each volume of a NeuroVec.

    This function applies a bilateral filter to each volume of a NeuroVec object.

    Parameters
    ----------
    vec : NeuroVec
        The volumes to be filtered
    mask : LogicalNeuroVol, optional
        Binary mask specifying the region of interest.
        If not provided, the whole volume is considered.
    spatial_sigma : float
        Spatial sigma in millimetres (default is 2)
    intensity_sigma : float
        Intensity sigma in units of ``range_scale`` (default is 1)
    window : int
        Size of the window for the bilateral filter (default is 1)
    range_scale : float, optional
        Fixed intensity scale for every volume. If None, each volume
        estimates its own in-mask standard deviation (see
        :func:`bilateral_filter`), so bandwidths differ across volumes.

    Returns
    -------
    DenseNeuroVec
        The filtered volumes

    """
    if window < 1:
        raise ValueError("Window size must be at least 1")
    if spatial_sigma <= 0 or intensity_sigma <= 0:
        raise ValueError("Sigma values must be positive")
    _check_range_scale(range_scale)

    data = np.asarray(vec.to_dense().data if hasattr(vec, "to_dense") else vec.data)
    mask_arr = (
        np.ones(data.shape[:3], dtype=bool)
        if mask is None
        else np.asarray(mask.data, dtype=bool)
    )
    filtered = [
        _bilateral_3d(
            data[..., t],
            mask_arr,
            spatial_sigma,
            intensity_sigma,
            int(window),
            vec.space.spacing[:3],
            range_scale,
        )
        for t in range(data.shape[3])
    ]
    return DenseNeuroVec(np.stack(filtered, axis=-1), vec.space)


def bilateral_filter_4d(
    vec: NeuroVec,
    mask: Optional[LogicalNeuroVol] = None,
    spatial_sigma: float = 2,
    intensity_sigma: float = 1,
    temporal_sigma: float = 1,
    spatial_window: int = 1,
    temporal_window: int = 1,
    temporal_spacing: float = 1,
    range_scale: Optional[float] = None,
) -> DenseNeuroVec:
    """Apply a joint spatial-temporal bilateral filter to a NeuroVec.

    This mirrors neuroim2's ``bilateral_filter_4d`` contract: spatial
    neighbors are limited by ``spatial_window``, temporal neighbors by
    ``temporal_window``, and values outside the mask are left unchanged.

    With ``range_scale=None`` one intensity scale is estimated from all finite
    in-mask values across time, so equal ``intensity_sigma`` values do not
    imply equal bandwidths across runs or subjects. For batch or group work,
    reuse one fixed ``range_scale`` and the same ``intensity_sigma``; the
    effective bandwidth is ``intensity_sigma * range_scale``.
    """
    if spatial_window < 1:
        raise ValueError("spatial_window must be >= 1")
    if temporal_window < 0:
        raise ValueError("temporal_window must be >= 0")
    if spatial_sigma <= 0:
        raise ValueError("spatial_sigma must be positive")
    if intensity_sigma <= 0:
        raise ValueError("intensity_sigma must be positive")
    if temporal_sigma <= 0:
        raise ValueError("temporal_sigma must be positive")
    if temporal_spacing <= 0:
        raise ValueError("temporal_spacing must be positive")
    if range_scale is not None and (not np.isfinite(range_scale) or range_scale <= 0):
        raise ValueError("range_scale must be None or a positive finite value")

    data = vec.as_dense().data if hasattr(vec, "as_dense") else np.asarray(vec.data)
    data = np.asarray(data, dtype=float)
    if data.ndim != 4:
        raise ValueError("bilateral_filter_4d expects 4D NeuroVec data")

    nx, ny, nz, nt = data.shape
    if mask is None:
        mask_arr = np.ones((nx, ny, nz), dtype=bool)
    else:
        mask_arr = np.asarray(mask.data, dtype=bool)
        if mask_arr.shape != (nx, ny, nz):
            raise ValueError("mask spatial dimensions must match vec")

    output = data.copy()
    masked_vals = data[mask_arr, :]
    finite_vals = masked_vals[np.isfinite(masked_vals)]
    intensity_sd = (
        float(range_scale)
        if range_scale is not None
        else (float(np.std(finite_vals)) if finite_vals.size > 1 else 0.0)
    )
    intensity_var = (
        2.0 * intensity_sigma * intensity_sigma * intensity_sd * intensity_sd
    )
    if not np.isfinite(intensity_var) or intensity_var < 1e-12:
        intensity_var = 1e-12

    spacing = np.asarray(vec.spacing, dtype=float)
    if spacing.size < 3:
        spacing = np.pad(spacing, (0, 3 - spacing.size), constant_values=1.0)
    spatial_var = 2.0 * spatial_sigma * spatial_sigma
    temporal_var = 2.0 * temporal_sigma * temporal_sigma

    offsets = []
    weights = []
    for dt in range(-int(temporal_window), int(temporal_window) + 1):
        temporal_weight = np.exp(-((dt * temporal_spacing) ** 2) / temporal_var)
        for dx in range(-int(spatial_window), int(spatial_window) + 1):
            dx2 = (dx * spacing[0]) ** 2
            for dy in range(-int(spatial_window), int(spatial_window) + 1):
                dy2 = (dy * spacing[1]) ** 2
                for dz in range(-int(spatial_window), int(spatial_window) + 1):
                    dz2 = (dz * spacing[2]) ** 2
                    spatial_weight = np.exp(-(dx2 + dy2 + dz2) / spatial_var)
                    offsets.append((dx, dy, dz, dt))
                    weights.append(spatial_weight * temporal_weight)

    coords = np.argwhere(mask_arr)
    for x, y, z in coords:
        for t in range(nt):
            center = data[x, y, z, t]
            if not np.isfinite(center):
                continue

            val_sum = 0.0
            weight_sum = 0.0
            for (dx, dy, dz, dt), base_weight in zip(offsets, weights):
                xx = x + dx
                yy = y + dy
                zz = z + dz
                tt = t + dt
                if xx < 0 or xx >= nx or yy < 0 or yy >= ny or zz < 0 or zz >= nz:
                    continue
                if tt < 0 or tt >= nt or not mask_arr[xx, yy, zz]:
                    continue

                neighbor = data[xx, yy, zz, tt]
                if not np.isfinite(neighbor):
                    continue
                diff = center - neighbor
                weight = base_weight * np.exp(-(diff * diff) / intensity_var)
                val_sum += weight * neighbor
                weight_sum += weight

            output[x, y, z, t] = val_sum / weight_sum if weight_sum > 0 else center

    return DenseNeuroVec(output, vec.space)


def box_blur(vol: NeuroVol, mask_indices: tuple, radius: int) -> np.ndarray:
    """Helper function for guided filter: applies box blur to the data.

    Parameters
    ----------
    vol : NeuroVol
        Volume to blur
    mask_indices : tuple
        Indices where to apply the blur
    radius : int
        Radius of the box filter

    Returns
    -------
    np.ndarray
        Blurred values at mask indices

    Notes
    -----
    This is used internally by guided_filter.
    """
    data = vol.data
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size**3)

    # Apply convolution
    blurred = ndimage.convolve(data, kernel, mode="constant", cval=0.0)

    # Return values at mask indices
    return blurred[mask_indices]

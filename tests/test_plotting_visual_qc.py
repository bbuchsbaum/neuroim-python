"""Optional visual QC against neuroim2 plotting behavior."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroim.neuro_space import NeuroSpace
from neuroim.neuro_vol import DenseNeuroVol
from neuroim.plotting import _display_slice, plot_montage


_ORIENTATION_SCRIPT = r"""
if (!requireNamespace("neuroim2", quietly = TRUE)) {
  cat("SKIP: neuroim2 is not installed\n")
  quit(status = 42)
}
suppressPackageStartupMessages(library(neuroim2))

emit_case <- function(name, trans) {
  dims <- c(4L, 5L, 2L)
  sp <- NeuroSpace(dims, trans = trans)
  arr <- array(0, dim = dims)
  arr[1, 1, 1] <- 1
  arr[4, 5, 1] <- 2
  vol <- NeuroVol(arr, sp)
  sl <- slice(vol, 1L, along = 3L)
  out <- neuroim2:::orient_slice_for_raster(sl, neuroim2:::slice_to_matrix(sl))
  cat("case=", name, "\n", sep = "")
  cat("dim=", paste(dim(out$mat), collapse = ","), "\n", sep = "")
  cat("mat=", paste(as.numeric(out$mat), collapse = ","), "\n", sep = "")
}

emit_case("identity", diag(4))
flipx <- diag(4)
flipx[1, 1] <- -1
flipx[1, 4] <- 3
emit_case("flipx", flipx)
flipy <- diag(4)
flipy[2, 2] <- -1
flipy[2, 4] <- 4
emit_case("flipy", flipy)
"""


_RENDER_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("expected one output path", call. = FALSE)
}
if (!requireNamespace("neuroim2", quietly = TRUE)) {
  cat("SKIP: neuroim2 is not installed\n")
  quit(status = 42)
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  cat("SKIP: ggplot2 is not installed\n")
  quit(status = 43)
}
suppressPackageStartupMessages(library(neuroim2))

dims <- c(12L, 12L, 3L)
sp <- NeuroSpace(dims)
arr <- array(0, dim = dims)
for (k in seq_len(dims[3])) {
  arr[, , k] <- outer(seq_len(dims[1]), seq_len(dims[2]), function(i, j) i + j + 4 * k)
}
arr[3:5, 3:5, 1] <- 50
arr[8:10, 8:10, 2] <- 70
vol <- NeuroVol(arr, sp)

png_args <- list(filename = args[[1]], width = 540, height = 340, res = 96)
if (capabilities("cairo")) {
  png_args$type <- "cairo"
}
do.call(grDevices::png, png_args)
on.exit({
  if (grDevices::dev.cur() > 1L) grDevices::dev.off()
}, add = TRUE)
p <- plot_montage(
  vol,
  zlevels = c(1L, 2L),
  along = 3L,
  ncol = 2L,
  range = "data",
  title = "neuroim2 QC",
  style = "light"
)
print(p)
"""


def _run_r(script: str, *args: Path) -> subprocess.CompletedProcess[str]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not available")

    proc = subprocess.run(
        [rscript, "-", *(str(arg) for arg in args)],
        input=script,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    if proc.returncode == 42:
        pytest.skip("neuroim2 is not installed in the R library path")
    if proc.returncode == 43:
        pytest.skip("neuroim2 plotting dependencies are not installed")
    if proc.returncode != 0:
        pytest.fail(
            "R neuroim2 plotting QC failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _parse_orientation_cases(output: str) -> dict[str, np.ndarray]:
    cases: dict[str, np.ndarray] = {}
    current: str | None = None
    dims: tuple[int, int] | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("case="):
            current = line.split("=", 1)[1]
            dims = None
        elif line.startswith("dim="):
            dims = tuple(int(x) for x in line.split("=", 1)[1].split(","))
        elif line.startswith("mat=") and current is not None and dims is not None:
            values = np.fromstring(line.split("=", 1)[1], sep=",", dtype=float)
            cases[current] = values.reshape(dims, order="F")

    if set(cases) != {"identity", "flipx", "flipy"}:
        raise AssertionError(f"unexpected neuroim2 orientation output:\n{output}")
    return cases


def _python_marker_slice(case: str) -> np.ndarray:
    affine = np.eye(4)
    if case == "flipx":
        affine[0, 0] = -1.0
        affine[0, 3] = 3.0
    elif case == "flipy":
        affine[1, 1] = -1.0
        affine[1, 3] = 4.0

    data = np.zeros((4, 5, 2), dtype=float)
    data[0, 0, 0] = 1.0
    data[3, 4, 0] = 2.0
    vol = DenseNeuroVol(data, NeuroSpace((4, 5, 2), trans=affine))
    return np.asarray(_display_slice(vol, axis=2, index=0).data)


def _qc_output_dir(tmp_path: Path) -> Path:
    configured = os.environ.get("NEUROIM_PLOT_VISUAL_QC_DIR")
    out_dir = Path(configured) if configured else tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _assert_png_nonblank(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 1000
    image = plt.imread(path)
    assert image.ndim in {2, 3}
    assert float(np.nanstd(image)) > 1e-3


def test_slice_orientation_matches_neuroim2_raster_positions():
    proc = _run_r(_ORIENTATION_SCRIPT)
    r_cases = _parse_orientation_cases(proc.stdout)

    for case, r_matrix in r_cases.items():
        python_matrix = _python_marker_slice(case)
        # neuroim2 stores y rows top-to-bottom for ggplot.  Matplotlib renders
        # our matrix with origin="lower", so flipping rows compares the same
        # visible marker positions.
        np.testing.assert_array_equal(python_matrix, np.flipud(r_matrix))


def test_can_emit_neuroim2_visual_qc_artifacts(tmp_path):
    out_dir = _qc_output_dir(tmp_path)
    r_png = out_dir / "neuroim2_montage_qc.png"
    py_png = out_dir / "python_montage_qc.png"

    _run_r(_RENDER_SCRIPT, r_png)

    data = np.zeros((12, 12, 3), dtype=float)
    ii, jj = np.meshgrid(np.arange(1, 13), np.arange(1, 13), indexing="ij")
    for k in range(data.shape[2]):
        data[:, :, k] = ii + jj + 4 * (k + 1)
    data[2:5, 2:5, 0] = 50
    data[7:10, 7:10, 1] = 70
    vol = DenseNeuroVol(data, NeuroSpace(data.shape))

    fig, _ = plot_montage(
        vol,
        zlevels=[0, 1],
        ncols=2,
        range="data",
        title="python QC",
        colorbar=True,
    )
    try:
        fig.savefig(py_png, dpi=96)
    finally:
        plt.close(fig)

    _assert_png_nonblank(r_png)
    _assert_png_nonblank(py_png)

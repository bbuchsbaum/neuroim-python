from pathlib import Path

import numpy as np

import neuroim as ni


def test_docs_quickstart_flagship_workflow():
    root = Path(__file__).resolve().parents[1]

    bold = ni.read_image(str(root / "golden_tests/fixtures/tiny_bold.nii.gz"))
    mask = ni.read_image(
        str(root / "golden_tests/fixtures/tiny_mask.nii.gz"), type="vol"
    )

    assert bold.space.compatible_with(mask.space)

    roi = ni.spherical_roi(mask, centroid=(4, 4, 2), radius=2)
    time_by_voxel = bold.series_roi(roi)

    assert time_by_voxel.shape == (bold.shape[-1], len(roi.coords))

    mean_data = np.zeros(mask.shape, dtype=np.float32)
    mean_data[tuple(roi.coords.T)] = time_by_voxel.mean(axis=0)
    mean_map = ni.NeuroVol.from_array(mean_data, space=mask.space)

    out = mean_map.to_nibabel()
    assert out.shape == mask.shape
    assert np.allclose(out.affine, bold.space.affine)

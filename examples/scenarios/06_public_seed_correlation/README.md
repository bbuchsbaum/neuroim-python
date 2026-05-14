# Scenario 06: Public Seed-To-Voxel Correlation

This ports the core shape of Nilearn's public seed-to-voxel correlation
examples without adding Nilearn as a test dependency: start with a 4-D BOLD
image, choose one world-coordinate seed, correlate the seed time series with
all in-mask voxels, and return a 3-D correlation map.

The point is not that neuroim owns correlation. The point is that the ordinary
scientific-Python workflow has spatial contracts that live in handwritten
guard code. The neuroim rewrite moves those contracts into named image
objects and leaves the correlation math visible.

Source inspiration: Nilearn's seed-to-voxel correlation examples and tutorials
around seed-based connectivity maps.

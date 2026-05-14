"""Scenario 01: MNI spotlight.

Task: given a 4-D BOLD image and a world-mm coordinate, return the
time series at the voxel nearest that coordinate.

This is the simplest non-trivial fMRI op that exercises the
world-to-voxel transform — the single most error-prone step in
hand-written nibabel code.
"""

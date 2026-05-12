#!/usr/bin/env python
"""
Demo: Extracting orthogonal slices from a 3D neuroimaging volume.

This example demonstrates how to extract axial, sagittal, and coronal
slices from a 3D volume at a specific world-space coordinate.
"""

import numpy as np
from neuroim import NeuroSpace, DenseNeuroVol

# Create a sample 3D volume with a sphere in the center
print("Creating sample 3D volume...")
space = NeuroSpace(
    dim=(64, 64, 64),
    spacing=(2.0, 2.0, 2.0),  # 2mm voxels
    origin=(-63.0, -63.0, -63.0)  # Centered around origin
)

# Create sphere data
data = np.zeros((64, 64, 64))
center = np.array([32, 32, 32])
radius = 20

for i in range(64):
    for j in range(64):
        for k in range(64):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
            if dist <= radius:
                data[i, j, k] = 100 - dist * 2

vol = DenseNeuroVol(data, space)
print(f"Volume created: {vol}")

# Extract orthogonal slices at the center of the volume
print("\nExtracting orthogonal slices at volume center...")
center_world = space.centroid()
print(f"Center world coordinates: {center_world}")

# Method 1: Extract all slices at once
slices = vol.get_orthogonal_slices(center_world)
print(f"\nExtracted {len(slices)} slices:")
for slice_type, slice_obj in slices.items():
    print(f"  {slice_type}: shape={slice_obj.shape}, spacing={slice_obj.spacing}")

# Method 2: Extract individual slices
print("\nExtracting individual slices...")
from neuroim import extract_axial_slice, extract_sagittal_slice, extract_coronal_slice

axial = extract_axial_slice(vol, center_world)
sagittal = extract_sagittal_slice(vol, center_world) 
coronal = extract_coronal_slice(vol, center_world)

print(f"Axial slice: shape={axial.shape}, value range=[{axial.data.min():.1f}, {axial.data.max():.1f}]")
print(f"Sagittal slice: shape={sagittal.shape}, value range=[{sagittal.data.min():.1f}, {sagittal.data.max():.1f}]")
print(f"Coronal slice: shape={coronal.shape}, value range=[{coronal.data.min():.1f}, {coronal.data.max():.1f}]")

# Extract slices at different world coordinates
print("\nExtracting slices at different positions...")
positions = [
    (0.0, 0.0, 0.0),      # Origin
    (20.0, 0.0, 0.0),     # Right of center
    (0.0, 20.0, 0.0),     # Anterior to center
    (0.0, 0.0, 20.0),     # Superior to center
]

for pos in positions:
    world_point = np.array(pos)
    try:
        slices = vol.get_orthogonal_slices(world_point, ['axial'])
        axial = slices['axial']
        center_val = axial[32, 32] if 0 <= 32 < axial.shape[0] and 0 <= 32 < axial.shape[1] else 0
        print(f"  Position {pos}: center value = {center_val:.1f}")
    except ValueError as e:
        print(f"  Position {pos}: {e}")

# Get slice orientation information
print("\nSlice orientations:")
from neuroim import get_slice_orientation

for slice_type in ['axial', 'sagittal', 'coronal']:
    orientation = get_slice_orientation(vol, slice_type)
    print(f"  {slice_type}: {orientation}")

# Demonstrate with non-identity transformation
print("\nCreating volume with rotation...")
trans = np.array([
    [0.9, -0.1, 0.0, 10.0],
    [0.1,  0.9, 0.0, -5.0],
    [0.0,  0.0, 1.0, 20.0],
    [0.0,  0.0, 0.0,  1.0]
])

rotated_space = NeuroSpace(dim=(50, 50, 50), trans=trans)
rotated_vol = DenseNeuroVol(np.random.rand(50, 50, 50), rotated_space)

# Extract slices from rotated volume
world_point = np.array([25.0, 25.0, 45.0])
slices = rotated_vol.get_orthogonal_slices(world_point)
print(f"Extracted slices from rotated volume: {list(slices.keys())}")

print("\nDemo complete!")
# Synthetic LiDAR

This module contains the geometry-based synthetic LiDAR utilities used in the end-to-end pipeline.

It:

- loads a reconstructed point cloud from `.ply`
- assumes that point cloud shares a world frame with a provided `lidar_to_world` pose
- transforms points into the LiDAR frame
- filters by simple range and field-of-view rules
- optionally downsamples to a target point count
- writes the synthetic point cloud to `.ply` or `.npy`

## Limitations

- this is a geometry-based approximation, not true LiDAR ray simulation
- occlusion handling is not included
- intensity and return modeling are not included
- it is meant for quick pipeline checks, not realism

## End-to-End Use

The main end-to-end entry point is `evaluation/pipeline.py:run_simple_3dgs_lidar_evaluation(...)`, which connects:

- a reconstructed 3DGS `.ply`
- processed sample `lidar_to_world`
- synthetic point cloud writing
- basic evaluation against real processed LiDAR
# Evaluation

This module compares:

- a synthetic LiDAR-style point cloud generated from reconstructed 3DGS geometry
- a real nuScenes LiDAR point cloud from one processed sample
- both point clouds in a shared comparison frame, usually the LiDAR frame

## Metrics

Current metrics include:

- point count
- mean range
- max range
- nearest-neighbor distance summaries in both directions
- spatial extent summaries

The default output artifact is `evaluation_summary.json`, written by `evaluate_point_clouds(...)`.

## End-to-End Helper

`run_simple_3dgs_lidar_evaluation(...)` in `evaluation/pipeline.py` takes:

- a reconstructed 3DGS `.ply`
- one processed sample
- an output directory

It writes:

- `synthetic_lidar.ply`
- `synthetic_lidar.json`
- `alignment_debug.json`
- `evaluation_summary.json`
- `run_summary.json`

## Optional Centroid Baseline

The pipeline also supports an optional centroid-translation baseline for cases where the default shared-world-frame assumption is clearly misaligned.

This baseline is:

- translation only
- no rotation fitting
- no scale fitting
- no ICP
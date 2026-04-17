"""Lightweight point-cloud evaluation utilities for the active 3DGS path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lidar_sim.depth_to_pointcloud import load_point_cloud

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback is covered instead.
    cKDTree = None


def _range_summary(points: np.ndarray) -> dict[str, float | None]:
    if points.shape[0] == 0:
        return {"mean": None, "max": None}
    ranges = np.linalg.norm(points[:, :3], axis=1)
    return {
        "mean": float(np.mean(ranges)),
        "max": float(np.max(ranges)),
    }


def _extent_summary(points: np.ndarray) -> dict[str, list[float] | None]:
    if points.shape[0] == 0:
        return {"min_xyz": None, "max_xyz": None, "span_xyz": None}
    min_xyz = np.min(points[:, :3], axis=0)
    max_xyz = np.max(points[:, :3], axis=0)
    return {
        "min_xyz": min_xyz.astype(float).tolist(),
        "max_xyz": max_xyz.astype(float).tolist(),
        "span_xyz": (max_xyz - min_xyz).astype(float).tolist(),
    }


def _nearest_neighbor_distances(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    if source_points.shape[0] == 0 or target_points.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    source_xyz = source_points[:, :3].astype(np.float32, copy=False)
    target_xyz = target_points[:, :3].astype(np.float32, copy=False)

    if cKDTree is not None:
        tree = cKDTree(target_xyz)
        distances, _ = tree.query(source_xyz, k=1)
        return np.asarray(distances, dtype=np.float32)

    distances = []
    for point in source_xyz:
        deltas = target_xyz - point
        distances.append(float(np.min(np.linalg.norm(deltas, axis=1))))
    return np.asarray(distances, dtype=np.float32)


def _distance_summary(distances: np.ndarray) -> dict[str, float | None]:
    if distances.size == 0:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }


def evaluate_point_clouds(
    synthetic_source: str | Path | np.ndarray | dict[str, Any],
    real_source: str | Path | np.ndarray | dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare a synthetic point cloud against real nuScenes LiDAR with simple metrics."""
    synthetic_points = load_point_cloud(synthetic_source)
    real_points = load_point_cloud(real_source)

    synthetic_to_real = _nearest_neighbor_distances(synthetic_points, real_points)
    real_to_synthetic = _nearest_neighbor_distances(real_points, synthetic_points)

    summary = {
        "synthetic": {
            "point_count": int(synthetic_points.shape[0]),
            "range_summary": _range_summary(synthetic_points),
            "extent_summary": _extent_summary(synthetic_points),
        },
        "real": {
            "point_count": int(real_points.shape[0]),
            "range_summary": _range_summary(real_points),
            "extent_summary": _extent_summary(real_points),
        },
        "nearest_neighbor_distance": {
            "synthetic_to_real": _distance_summary(synthetic_to_real),
            "real_to_synthetic": _distance_summary(real_to_synthetic),
        },
        "assumptions": [
            "The synthetic cloud comes from reconstructed 3DGS geometry.",
            "Synthetic and real point clouds should be compared in the same frame, usually the LiDAR frame.",
            "No intensity, return-count, or physical visibility modeling.",
        ],
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    return summary

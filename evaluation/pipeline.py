"""End-to-end helpers for synthetic LiDAR generation and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.metrics import evaluate_point_clouds
from lidar_sim.depth_to_pointcloud import (
    SimpleLidarSimulationConfig,
    load_ply_points,
    simulate_point_cloud_lidar_with_report,
    write_point_cloud,
)
from src.utils.geometry import invert_transform, transform_points


def _load_processed_sample(sample_source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(sample_source, dict):
        return sample_source

    sample_path = Path(sample_source)
    if sample_path.suffix.lower() != ".pt":
        raise ValueError("The end-to-end 3DGS LiDAR helper expects a processed sample .pt file.")
    return torch.load(sample_path, map_location="cpu")


def _to_numpy_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def summarize_point_cloud_alignment(points: np.ndarray) -> dict[str, Any]:
    """Summarize a point cloud for alignment checks."""
    points = _to_numpy_array(points)
    xyz = points[:, :3] if points.ndim == 2 and points.shape[1] >= 3 else np.empty((0, 3), dtype=np.float32)
    if xyz.shape[0] == 0:
        return {
            "point_count": 0,
            "centroid": None,
            "min_xyz": None,
            "max_xyz": None,
            "span_xyz": None,
            "mean_radial_distance": None,
            "max_radial_distance": None,
        }

    centroid = np.mean(xyz, axis=0)
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    span_xyz = max_xyz - min_xyz
    radii = np.linalg.norm(xyz, axis=1)
    return {
        "point_count": int(xyz.shape[0]),
        "centroid": centroid.astype(float).tolist(),
        "min_xyz": min_xyz.astype(float).tolist(),
        "max_xyz": max_xyz.astype(float).tolist(),
        "span_xyz": span_xyz.astype(float).tolist(),
        "mean_radial_distance": float(np.mean(radii)),
        "max_radial_distance": float(np.max(radii)),
    }


def apply_centroid_alignment_baseline(
    reconstructed_points: np.ndarray,
    real_lidar_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a translation-only centroid baseline in the LiDAR frame."""
    reconstructed_points = _to_numpy_array(reconstructed_points)[:, :3]
    real_lidar_points = _to_numpy_array(real_lidar_points)[:, :3]

    reconstructed_summary = summarize_point_cloud_alignment(reconstructed_points)
    real_summary = summarize_point_cloud_alignment(real_lidar_points)

    if reconstructed_summary["centroid"] is None or real_summary["centroid"] is None:
        translation = np.zeros((3,), dtype=np.float32)
        translated_points = reconstructed_points.copy()
    else:
        translation = (
            np.asarray(real_summary["centroid"], dtype=np.float32)
            - np.asarray(reconstructed_summary["centroid"], dtype=np.float32)
        )
        translated_points = reconstructed_points + translation

    translated_summary = summarize_point_cloud_alignment(translated_points)
    report = {
        "used": True,
        "mode": "centroid_translation_baseline",
        "assumptions": [
            "This baseline shifts the reconstructed 3DGS cloud so its centroid matches the real LiDAR centroid.",
            "It does not estimate rotation, scale, or ICP alignment.",
            "It is only meant to produce a rough non-empty comparison.",
        ],
        "native_3dgs_centroid_before": reconstructed_summary["centroid"],
        "real_lidar_centroid_target": real_summary["centroid"],
        "translation_applied": translation.astype(float).tolist(),
        "native_3dgs_summary_before": reconstructed_summary,
        "translated_3dgs_summary_after": translated_summary,
    }
    return translated_points.astype(np.float32, copy=False), report


def _write_simulation_report(
    *,
    output_path: Path,
    source_ply_path: str | Path,
    config: SimpleLidarSimulationConfig,
    input_point_count: int,
    synthetic_points: np.ndarray,
    filter_report: dict[str, Any],
    alignment_mode: str,
    alignment_report: dict[str, Any] | None,
) -> dict[str, Any]:
    report = {
        "source_ply_path": str(Path(source_ply_path).resolve()),
        "output_path": str(output_path.resolve()),
        "config": asdict(config),
        "alignment_mode": alignment_mode,
        "alignment_report": alignment_report,
        **filter_report,
        "input_point_count": int(input_point_count),
        "output_point_count": int(synthetic_points.shape[0]),
        "assumptions": [
            "Geometry-based approximation; not a physical LiDAR simulator.",
            "The reconstructed point cloud is filtered in the LiDAR frame used for this run.",
            "No occlusion, intensity, or return modeling.",
        ],
    }

    report_path = output_path.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _build_run_summary_payload(
    *,
    ply_path: str | Path,
    real_sample_source: str | Path | dict[str, Any],
    output_dir: Path,
    simulation_config: SimpleLidarSimulationConfig,
    use_centroid_alignment_baseline: bool,
    synthetic_output_path: Path,
    alignment_debug_path: Path,
    evaluation_output_path: Path,
    simulation_report: dict[str, Any],
    evaluation_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_ply_path": str(Path(ply_path).resolve()),
        "real_sample_source": str(Path(real_sample_source).resolve())
        if not isinstance(real_sample_source, dict)
        else "<in-memory-sample>",
        "output_dir": str(output_dir.resolve()),
        "simulation_config": asdict(simulation_config),
        "use_centroid_alignment_baseline": use_centroid_alignment_baseline,
        "synthetic_point_cloud_path": str(synthetic_output_path.resolve()),
        "simulation_report_path": str(synthetic_output_path.with_suffix(".json").resolve()),
        "alignment_debug_path": str(alignment_debug_path.resolve()),
        "evaluation_summary_path": str(evaluation_output_path.resolve()),
        "alignment_mode": simulation_report["alignment_mode"],
        "synthetic_point_count": simulation_report["output_point_count"],
        "real_point_count": evaluation_summary["real"]["point_count"],
        "assumptions": [
            "This run uses a geometry-based approximation built from a reconstructed 3DGS point cloud.",
            "The default path assumes the reconstructed 3DGS point cloud and lidar_to_world share a world frame.",
            "The optional centroid baseline is translation-only and not a real registration method.",
            "No physical LiDAR ray simulation, occlusion reasoning, or intensity modeling.",
        ],
    }


def build_alignment_debug_report(
    reconstructed_points_world: np.ndarray,
    real_lidar_points: np.ndarray,
    lidar_to_world: np.ndarray,
    *,
    centroid_alignment_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare native and transformed point-cloud summaries for alignment checks."""
    lidar_to_world = _to_numpy_array(lidar_to_world)
    world_to_lidar = invert_transform(lidar_to_world)
    transformed_points_lidar = transform_points(_to_numpy_array(reconstructed_points_world)[:, :3], world_to_lidar)[:, :3]

    reconstructed_summary = summarize_point_cloud_alignment(reconstructed_points_world)
    real_lidar_summary = summarize_point_cloud_alignment(real_lidar_points)
    transformed_summary = summarize_point_cloud_alignment(transformed_points_lidar)

    centroid_offset = None
    span_ratio = None
    if transformed_summary["centroid"] is not None and real_lidar_summary["centroid"] is not None:
        centroid_offset = (
            np.asarray(transformed_summary["centroid"], dtype=np.float32)
            - np.asarray(real_lidar_summary["centroid"], dtype=np.float32)
        ).astype(float).tolist()
    if transformed_summary["span_xyz"] is not None and real_lidar_summary["span_xyz"] is not None:
        real_span = np.asarray(real_lidar_summary["span_xyz"], dtype=np.float32)
        transformed_span = np.asarray(transformed_summary["span_xyz"], dtype=np.float32)
        safe_real_span = np.where(np.abs(real_span) < 1e-6, np.nan, real_span)
        span_ratio = (transformed_span / safe_real_span).astype(float).tolist()

    return {
        "assumptions": [
            "This report does not perform alignment or scale fitting.",
            "The transformed 3DGS summary assumes the reconstructed point cloud shares a world frame with lidar_to_world.",
            "Large centroid offsets or span-ratio mismatches suggest frame or scale disagreement.",
        ],
        "centroid_alignment_baseline": centroid_alignment_report,
        "native_3dgs_point_cloud": reconstructed_summary,
        "real_lidar_points_in_lidar_frame": real_lidar_summary,
        "3dgs_points_transformed_by_current_world_to_lidar_assumption": transformed_summary,
        "comparison": {
            "transformed_3dgs_minus_real_lidar_centroid": centroid_offset,
            "transformed_3dgs_to_real_lidar_span_ratio": span_ratio,
        },
    }


def run_simple_3dgs_lidar_evaluation(
    ply_path: str | Path,
    real_sample_source: str | Path | dict[str, Any],
    output_dir: str | Path,
    *,
    simulation_config: SimpleLidarSimulationConfig | None = None,
    use_centroid_alignment_baseline: bool = False,
) -> dict[str, Any]:
    """Run synthetic LiDAR generation and evaluation for one processed sample."""
    simulation_config = simulation_config or SimpleLidarSimulationConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = _load_processed_sample(real_sample_source)
    if "lidar_to_world" not in sample:
        raise KeyError("Processed sample is missing required field 'lidar_to_world'.")
    if "lidar_points" not in sample:
        raise KeyError("Processed sample is missing required field 'lidar_points'.")

    lidar_to_world = _to_numpy_array(sample["lidar_to_world"])
    real_lidar_points = _to_numpy_array(sample["lidar_points"])
    reconstructed_points_world = load_ply_points(ply_path)
    synthetic_output_path = output_dir / "synthetic_lidar.ply"
    centroid_alignment_report = None
    if use_centroid_alignment_baseline:
        translated_points_lidar, centroid_alignment_report = apply_centroid_alignment_baseline(
            reconstructed_points_world,
            real_lidar_points,
        )
        synthetic_points, filter_report = simulate_point_cloud_lidar_with_report(
            translated_points_lidar,
            np.eye(4, dtype=np.float32),
            simulation_config,
        )
        write_point_cloud(synthetic_output_path, synthetic_points)
        simulation_report = _write_simulation_report(
            output_path=synthetic_output_path,
            source_ply_path=ply_path,
            config=simulation_config,
            input_point_count=reconstructed_points_world.shape[0],
            synthetic_points=synthetic_points,
            filter_report=filter_report,
            alignment_mode="centroid_translation_baseline",
            alignment_report=centroid_alignment_report,
        )
    else:
        synthetic_points, filter_report = simulate_point_cloud_lidar_with_report(
            reconstructed_points_world,
            lidar_to_world,
            simulation_config,
        )
        write_point_cloud(synthetic_output_path, synthetic_points)
        simulation_report = _write_simulation_report(
            output_path=synthetic_output_path,
            source_ply_path=ply_path,
            config=simulation_config,
            input_point_count=reconstructed_points_world.shape[0],
            synthetic_points=synthetic_points,
            filter_report=filter_report,
            alignment_mode="current_world_to_lidar_assumption",
            alignment_report=None,
        )

    alignment_debug = build_alignment_debug_report(
        reconstructed_points_world=reconstructed_points_world,
        real_lidar_points=real_lidar_points,
        lidar_to_world=lidar_to_world,
        centroid_alignment_report=centroid_alignment_report,
    )
    alignment_debug_path = output_dir / "alignment_debug.json"
    with alignment_debug_path.open("w", encoding="utf-8") as handle:
        json.dump(alignment_debug, handle, indent=2)

    evaluation_output_path = output_dir / "evaluation_summary.json"
    evaluation_summary = evaluate_point_clouds(
        synthetic_output_path,
        sample,
        output_path=evaluation_output_path,
    )

    run_summary = _build_run_summary_payload(
        ply_path=ply_path,
        real_sample_source=real_sample_source,
        output_dir=output_dir,
        simulation_config=simulation_config,
        use_centroid_alignment_baseline=use_centroid_alignment_baseline,
        synthetic_output_path=synthetic_output_path,
        alignment_debug_path=alignment_debug_path,
        evaluation_output_path=evaluation_output_path,
        simulation_report=simulation_report,
        evaluation_summary=evaluation_summary,
    )

    run_summary_path = output_dir / "run_summary.json"
    with run_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    run_summary["run_summary_path"] = str(run_summary_path.resolve())
    run_summary["simulation_report"] = simulation_report
    run_summary["alignment_debug"] = alignment_debug
    run_summary["evaluation_summary"] = evaluation_summary
    return run_summary

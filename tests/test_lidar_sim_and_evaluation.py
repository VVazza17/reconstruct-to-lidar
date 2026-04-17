import json
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import evaluate_point_clouds
from evaluation.pipeline import apply_centroid_alignment_baseline, run_simple_3dgs_lidar_evaluation
from lidar_sim.depth_to_pointcloud import (
    SimpleLidarSimulationConfig,
    load_point_cloud,
    make_loose_debug_simulation_config,
    simulate_lidar_from_ply,
    simulate_point_cloud_lidar,
    simulate_point_cloud_lidar_with_report,
    write_point_cloud,
)


def test_simulate_point_cloud_lidar_filters_by_range_fov_and_target_count(tmp_path: Path):
    points_world = np.array(
        [
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [-2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    lidar_to_world = np.eye(4, dtype=np.float32)
    config = SimpleLidarSimulationConfig(
        min_range=1.5,
        max_range=5.0,
        horizontal_fov_deg=90.0,
        target_count=2,
        random_seed=3,
    )

    simulated = simulate_point_cloud_lidar(points_world, lidar_to_world, config)

    assert simulated.shape == (2, 3)
    assert np.all(np.linalg.norm(simulated, axis=1) <= 5.0)
    assert np.all(simulated[:, 0] > 0.0)


def test_simulate_point_cloud_lidar_with_report_exposes_filter_stage_counts():
    points_world = np.array(
        [
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    simulated, report = simulate_point_cloud_lidar_with_report(
        points_world,
        np.eye(4, dtype=np.float32),
        SimpleLidarSimulationConfig(min_range=0.0, max_range=5.0, horizontal_fov_deg=90.0),
    )

    counts = report["filter_counts"]
    assert simulated.shape == (2, 3)
    assert counts["input_point_count"] == 4
    assert counts["after_finite_filter_count"] == 4
    assert counts["after_range_filter_count"] == 3
    assert counts["after_horizontal_fov_filter_count"] == 2
    assert counts["after_vertical_fov_filter_count"] == 2
    assert counts["after_downsampling_count"] == 2
    assert counts["final_output_point_count"] == 2
    assert report["zero_point_diagnosis"]["output_is_empty"] is False


def test_make_loose_debug_simulation_config_disables_most_filtering():
    config = make_loose_debug_simulation_config()

    assert config.min_range == 0.0
    assert config.max_range == 1_000.0
    assert config.horizontal_fov_deg == 360.0
    assert config.vertical_fov_deg is None
    assert config.target_count is None


def test_apply_centroid_alignment_baseline_matches_real_lidar_centroid():
    reconstructed = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    real_lidar = np.array(
        [
            [10.0, 5.0, -1.0],
            [12.0, 5.0, -1.0],
        ],
        dtype=np.float32,
    )

    translated, report = apply_centroid_alignment_baseline(reconstructed, real_lidar)

    assert np.allclose(np.mean(translated, axis=0), np.mean(real_lidar, axis=0))
    assert report["native_3dgs_centroid_before"] == [1.0, 0.0, 0.0]
    assert report["real_lidar_centroid_target"] == [11.0, 5.0, -1.0]
    assert report["translation_applied"] == [10.0, 5.0, -1.0]


def test_simulate_lidar_from_ply_writes_output_point_cloud_and_report(tmp_path: Path):
    source_points = np.array(
        [
            [2.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_ply = tmp_path / "scene.ply"
    output_ply = tmp_path / "synthetic_lidar.ply"
    write_point_cloud(source_ply, source_points)

    report = simulate_lidar_from_ply(
        source_ply,
        np.eye(4, dtype=np.float32),
        output_ply,
        SimpleLidarSimulationConfig(max_range=5.0),
    )

    assert output_ply.is_file()
    assert output_ply.with_suffix(".json").is_file()
    simulated_points = load_point_cloud(output_ply)
    assert simulated_points.shape == (2, 3)
    assert report["output_point_count"] == 2
    assert report["filter_counts"]["input_point_count"] == 3
    assert report["filter_counts"]["after_range_filter_count"] == 2
    assert report["zero_point_diagnosis"]["came_from_empty_source_geometry"] is False


def test_simulate_lidar_from_ply_reports_empty_output_diagnosis(tmp_path: Path):
    source_ply = tmp_path / "scene.ply"
    write_point_cloud(source_ply, np.array([[2.0, 0.0, 0.0]], dtype=np.float32))

    report = simulate_lidar_from_ply(
        source_ply,
        np.eye(4, dtype=np.float32),
        tmp_path / "synthetic_lidar.ply",
        SimpleLidarSimulationConfig(min_range=10.0, max_range=20.0),
    )

    assert report["output_point_count"] == 0
    assert report["zero_point_diagnosis"]["output_is_empty"] is True
    assert report["zero_point_diagnosis"]["likely_due_to_filtering"] is True
    assert report["zero_point_diagnosis"]["likely_cause"] == "range_filtering"


def test_load_ply_points_ascii_handles_zero_and_one_point_shapes(tmp_path: Path):
    one_point_ply = tmp_path / "one_point.ply"
    write_point_cloud(one_point_ply, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    one_point = load_point_cloud(one_point_ply)
    assert one_point.shape == (1, 3)
    assert np.allclose(one_point[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))

    zero_point_ply = tmp_path / "zero_point.ply"
    zero_point_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 0",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    zero_point = load_point_cloud(zero_point_ply)
    assert zero_point.shape == (0, 3)


def test_evaluate_point_clouds_computes_basic_metrics_and_writes_summary(tmp_path: Path):
    synthetic_points = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    real_points = np.array(
        [
            [1.1, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    summary_path = tmp_path / "evaluation_summary.json"

    summary = evaluate_point_clouds(synthetic_points, real_points, output_path=summary_path)

    assert summary["synthetic"]["point_count"] == 2
    assert summary["real"]["point_count"] == 3
    assert summary["synthetic"]["range_summary"]["mean"] == 1.5
    assert summary["real"]["range_summary"]["max"] == 5.0
    assert summary["nearest_neighbor_distance"]["synthetic_to_real"]["mean"] is not None
    assert summary["nearest_neighbor_distance"]["real_to_synthetic"]["max"] is not None
    assert summary_path.is_file()

    with summary_path.open("r", encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["synthetic"]["point_count"] == 2


def test_evaluate_point_clouds_accepts_processed_sample_pt_for_real_lidar(tmp_path: Path):
    sample_path = tmp_path / "sample.pt"
    torch.save(
        {
            "lidar_points": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [2.0, 0.0, 0.0, 0.6],
                ],
                dtype=torch.float32,
            )
        },
        sample_path,
    )

    summary = evaluate_point_clouds(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), sample_path)

    assert summary["real"]["point_count"] == 2
    assert summary["synthetic"]["point_count"] == 1


def test_run_simple_3dgs_lidar_evaluation_writes_all_expected_artifacts(tmp_path: Path):
    ply_path = tmp_path / "point_cloud.ply"
    write_point_cloud(
        ply_path,
        np.array(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    sample_path = tmp_path / "sample.pt"
    torch.save(
        {
            "lidar_points": torch.tensor(
                [
                    [2.1, 0.0, 0.0, 0.1],
                    [3.2, 0.0, 0.0, 0.2],
                ],
                dtype=torch.float32,
            ),
            "lidar_to_world": torch.eye(4, dtype=torch.float32),
        },
        sample_path,
    )

    output_dir = tmp_path / "run"
    result = run_simple_3dgs_lidar_evaluation(
        ply_path,
        sample_path,
        output_dir,
        simulation_config=SimpleLidarSimulationConfig(max_range=5.0),
    )

    assert (output_dir / "synthetic_lidar.ply").is_file()
    assert (output_dir / "synthetic_lidar.json").is_file()
    assert (output_dir / "alignment_debug.json").is_file()
    assert (output_dir / "evaluation_summary.json").is_file()
    assert (output_dir / "run_summary.json").is_file()
    assert result["simulation_report"]["output_point_count"] == 2
    assert result["alignment_debug"]["native_3dgs_point_cloud"]["point_count"] == 3
    assert result["alignment_debug"]["real_lidar_points_in_lidar_frame"]["point_count"] == 2
    assert (
        result["alignment_debug"]["3dgs_points_transformed_by_current_world_to_lidar_assumption"]["point_count"] == 3
    )
    assert result["evaluation_summary"]["synthetic"]["point_count"] == 2
    assert result["evaluation_summary"]["real"]["point_count"] == 2


def test_run_simple_3dgs_lidar_evaluation_records_centroid_alignment_baseline(tmp_path: Path):
    ply_path = tmp_path / "point_cloud.ply"
    write_point_cloud(
        ply_path,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    sample_path = tmp_path / "sample.pt"
    torch.save(
        {
            "lidar_points": torch.tensor(
                [
                    [10.0, 5.0, -1.0, 0.1],
                    [12.0, 5.0, -1.0, 0.2],
                ],
                dtype=torch.float32,
            ),
            "lidar_to_world": torch.eye(4, dtype=torch.float32),
        },
        sample_path,
    )

    result = run_simple_3dgs_lidar_evaluation(
        ply_path,
        sample_path,
        tmp_path / "aligned_run",
        simulation_config=SimpleLidarSimulationConfig(min_range=0.0, max_range=20.0),
        use_centroid_alignment_baseline=True,
    )

    assert result["use_centroid_alignment_baseline"] is True
    assert result["simulation_report"]["alignment_mode"] == "centroid_translation_baseline"
    assert result["simulation_report"]["alignment_report"]["translation_applied"] == [10.0, 5.0, -1.0]
    assert result["alignment_debug"]["centroid_alignment_baseline"]["used"] is True

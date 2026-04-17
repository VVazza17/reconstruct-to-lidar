import json
from pathlib import Path

import torch
from PIL import Image

from exports.gaussian_splatting.exporter import (
    GAUSSIAN_SPLATTING_METADATA_FILENAME,
    GAUSSIAN_SPLATTING_PLAN_FILENAME,
    validate_gaussian_splatting_scene_root,
    write_gaussian_splatting_dataset,
)


def create_fake_processed_dataset(tmp_path: Path) -> tuple[Path, dict]:
    processed_dir = tmp_path / "data" / "processed" / "fake_run"
    samples_dir = processed_dir / "samples"
    samples_dir.mkdir(parents=True)
    raw_dir = tmp_path / "data" / "raw" / "fake_nuscenes"
    raw_dir.mkdir(parents=True)

    image_path = raw_dir / "cam_front.png"
    Image.new("RGB", (24, 16), color=(12, 34, 56)).save(image_path)

    sample_payload = {
        "scene_id": "scene-0001",
        "sample_token": "sample-token-1",
        "timestamp": 123456789,
        "camera_name": "CAM_FRONT",
        "image": torch.zeros((3, 16, 24), dtype=torch.float32),
        "intrinsics": torch.eye(3, dtype=torch.float32),
        "cam_to_world": torch.eye(4, dtype=torch.float32),
        "lidar_points": torch.ones((5, 4), dtype=torch.float32),
        "lidar_to_world": torch.eye(4, dtype=torch.float32),
        "ego_pose": torch.eye(4, dtype=torch.float32),
        "metadata": {
            "camera_token": "camera-token-1",
            "lidar_token": "lidar-token-1",
            "camera_path": str(image_path),
            "lidar_path": str(raw_dir / "lidar.bin"),
            "time_diff_ms": 12.5,
        },
    }
    sample_path = samples_dir / "sample_000000.pt"
    torch.save(sample_payload, sample_path)

    manifest = {
        "version": 1,
        "dataset": {"name": "nuScenes", "version": "v1.0-mini", "dataroot": "C:\\datasets\\nuscenes"},
        "preprocessing": {"image_size": [16, 24]},
        "samples": [
            {
                "sample_id": "sample_000000",
                "path": "data/processed/fake_run/samples/sample_000000.pt",
                "scene_id": "scene-0001",
                "sample_token": "sample-token-1",
                "camera_name": "CAM_FRONT",
                "timestamp": 123456789,
                "time_diff_ms": 12.5,
            }
        ],
    }
    manifest_path = processed_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest_path, sample_payload


def test_gaussian_splatting_export_creates_images_dir_and_metadata_outputs(tmp_path: Path):
    manifest_path, _ = create_fake_processed_dataset(tmp_path)
    export_root = tmp_path / "exports" / "gaussian_splatting_run"

    payload = write_gaussian_splatting_dataset(manifest_path, export_root)

    assert (export_root / "images").is_dir()
    assert (export_root / GAUSSIAN_SPLATTING_METADATA_FILENAME).is_file()
    assert (export_root / GAUSSIAN_SPLATTING_PLAN_FILENAME).is_file()
    assert payload["stages_inputs_for_external_colmap"] is True
    assert payload["writes_sparse_reconstruction"] is False
    assert payload["colmap_handoff"]["run_colmap_externally"] is True


def test_gaussian_splatting_export_copies_raw_camera_image(tmp_path: Path):
    manifest_path, sample_payload = create_fake_processed_dataset(tmp_path)
    export_root = tmp_path / "exports" / "gaussian_splatting_run"
    source_path = Path(sample_payload["metadata"]["camera_path"])

    payload = write_gaussian_splatting_dataset(manifest_path, export_root)

    exported_entry = payload["exported_images"][0]
    staged_path = Path(exported_entry["staged_image_path"])
    assert staged_path.is_file()
    assert staged_path.read_bytes() == source_path.read_bytes()
    assert exported_entry["source_camera_path"] == str(source_path)


def test_gaussian_splatting_export_metadata_records_available_inputs_and_missing_colmap_outputs(tmp_path: Path):
    manifest_path, sample_payload = create_fake_processed_dataset(tmp_path)
    export_root = tmp_path / "exports" / "gaussian_splatting_run"

    payload = write_gaussian_splatting_dataset(manifest_path, export_root)

    assert payload["image_source_mode"] == "raw_camera_path"
    assert payload["intrinsics_availability"]["available"] is True
    assert payload["pose_availability"]["available"] is True
    assert payload["gaussian_splatting_colmap_assessment"]["sparse_reconstruction_available"] is False
    assert payload["gaussian_splatting_colmap_assessment"]["camera_image_id_mapping_available"] is False
    assert payload["writes_cameras_txt_or_bin"] is False
    assert payload["writes_images_txt_or_bin"] is False
    assert payload["writes_points3D_txt_or_bin"] is False
    assert payload["colmap_handoff"]["required_steps"] == [
        "feature_extraction",
        "feature_matching",
        "mapping",
    ]
    assert payload["colmap_handoff"]["expected_outputs"]["database_path"].endswith("database.db")
    assert payload["post_colmap_handoff"]["required_paths"]["sparse_model_dir"].endswith("sparse\\0") or payload[
        "post_colmap_handoff"
    ]["required_paths"]["sparse_model_dir"].endswith("sparse/0")
    assert "database.db exists" in payload["post_colmap_handoff"]["verification_checks"]
    assert payload["exported_images"][0]["source_camera_path"] == sample_payload["metadata"]["camera_path"]


def test_gaussian_splatting_export_plan_includes_frame_and_colmap_blocker_context(tmp_path: Path):
    manifest_path, _ = create_fake_processed_dataset(tmp_path)
    export_root = tmp_path / "exports" / "gaussian_splatting_run"

    write_gaussian_splatting_dataset(manifest_path, export_root)
    with (export_root / GAUSSIAN_SPLATTING_PLAN_FILENAME).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["frame_convention_report"]["pose_name"] == "cam_to_world"
    assert payload["gaussian_splatting_colmap_assessment"]["status"] == "blocked"
    assert payload["colmap_handoff"]["next_consumer"]["repository"].endswith("gaussian-splatting")
    assert payload["post_colmap_handoff"]["consumer"]["repository"].endswith("gaussian-splatting")
    assert any("positive camera-space z" in note for note in payload["current_frame_notes"])


def test_gaussian_splatting_export_does_not_mutate_processed_sample_file(tmp_path: Path):
    manifest_path, _ = create_fake_processed_dataset(tmp_path)
    export_root = tmp_path / "exports" / "gaussian_splatting_run"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    sample_path = (tmp_path / manifest["samples"][0]["path"]).resolve()
    before = torch.load(sample_path, map_location="cpu")

    write_gaussian_splatting_dataset(manifest_path, export_root)

    after = torch.load(sample_path, map_location="cpu")
    assert before["scene_id"] == after["scene_id"]
    assert before["metadata"]["camera_path"] == after["metadata"]["camera_path"]
    assert torch.equal(before["image"], after["image"])
    assert torch.equal(before["intrinsics"], after["intrinsics"])


def test_gaussian_splatting_scene_validator_reports_ready_for_external_colmap(tmp_path: Path):
    manifest_path, _ = create_fake_processed_dataset(tmp_path)
    export_root = tmp_path / "exports" / "gaussian_splatting_run"

    write_gaussian_splatting_dataset(manifest_path, export_root)
    report = validate_gaussian_splatting_scene_root(export_root)

    assert report["readiness_state"] == "ready_for_external_colmap"
    assert report["ready_for_external_colmap"] is True
    assert report["ready_for_official_gaussian_splatting"] is False
    assert "database.db" in report["missing_required_artifacts"]
    assert "sparse/0/" in report["missing_required_artifacts"]


def test_gaussian_splatting_scene_validator_reports_ready_for_official_gaussian_splatting(tmp_path: Path):
    manifest_path, _ = create_fake_processed_dataset(tmp_path)
    export_root = tmp_path / "exports" / "gaussian_splatting_run"

    write_gaussian_splatting_dataset(manifest_path, export_root)
    (export_root / "database.db").write_text("fake-colmap-db", encoding="utf-8")
    sparse_dir = export_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text("fake cameras", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("fake images", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("fake points", encoding="utf-8")

    report = validate_gaussian_splatting_scene_root(export_root)

    assert report["readiness_state"] == "ready_for_official_gaussian_splatting"
    assert report["ready_for_external_colmap"] is True
    assert report["ready_for_official_gaussian_splatting"] is True
    assert report["missing_required_artifacts"] == []


def test_gaussian_splatting_scene_validator_reports_missing_required_artifacts(tmp_path: Path):
    export_root = tmp_path / "exports" / "gaussian_splatting_run"
    export_root.mkdir(parents=True)

    report = validate_gaussian_splatting_scene_root(export_root)

    assert report["readiness_state"] == "missing_required_artifacts"
    assert report["ready_for_external_colmap"] is False
    assert report["ready_for_official_gaussian_splatting"] is False
    assert "images/" in report["missing_required_artifacts"]
    assert GAUSSIAN_SPLATTING_METADATA_FILENAME in report["missing_required_artifacts"]
    assert GAUSSIAN_SPLATTING_PLAN_FILENAME in report["missing_required_artifacts"]

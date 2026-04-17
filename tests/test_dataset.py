import json
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import ProcessedNuScenesDataset


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


def test_dataset_loads_manifest(tmp_path: Path):
    manifest_path, _ = create_fake_processed_dataset(tmp_path)

    dataset = ProcessedNuScenesDataset(manifest_path)

    assert dataset.manifest["version"] == 1
    assert len(dataset.samples) == 1


def test_dataset_len_matches_manifest_samples(tmp_path: Path):
    manifest_path, _ = create_fake_processed_dataset(tmp_path)

    dataset = ProcessedNuScenesDataset(manifest_path)

    assert len(dataset) == 1


def test_dataset_loads_one_sample(tmp_path: Path):
    manifest_path, expected_sample = create_fake_processed_dataset(tmp_path)

    dataset = ProcessedNuScenesDataset(manifest_path)
    sample = dataset[0]

    assert sample["scene_id"] == expected_sample["scene_id"]
    assert tuple(sample["image"].shape) == (3, 16, 24)
    assert tuple(sample["intrinsics"].shape) == (3, 3)
    assert tuple(sample["cam_to_world"].shape) == (4, 4)
    assert tuple(sample["lidar_points"].shape) == (5, 4)
    assert tuple(sample["lidar_to_world"].shape) == (4, 4)
    assert tuple(sample["ego_pose"].shape) == (4, 4)

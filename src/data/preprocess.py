"""Shared nuScenes preprocessing for the active 3DGS pipeline."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image

from src.data.nuscenes_index import NuScenesIndex
from src.data.sync import find_best_lidar_for_camera


LOGGER = logging.getLogger(__name__)


DEFAULT_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class PreprocessConfig:
    dataroot: str
    output_dir: str
    version: str = "v1.0-mini"
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    lidar_name: str = "LIDAR_TOP"
    image_height: int = 512
    image_width: int = 896
    max_time_diff_ms: float = 50.0
    scene_names: Optional[tuple[str, ...]] = None


class Preprocessor:
    def __init__(self, config: PreprocessConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or LOGGER

        self.output_dir = Path(config.output_dir)
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def normalize_image(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self.config.image_width, self.config.image_height))
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        image_array = np.transpose(image_array, (2, 0, 1))
        return torch.from_numpy(image_array.copy())

    def load_lidar_points(self, lidar_path: Path) -> torch.Tensor:
        point_cloud = LidarPointCloud.from_file(str(lidar_path))
        points = point_cloud.points.T.astype(np.float32)
        points = self.filter_lidar_points(points)
        return torch.from_numpy(points.copy())

    def filter_lidar_points(self, points: np.ndarray) -> np.ndarray:
        xyz = points[:, :3]
        finite_mask = np.isfinite(xyz).all(axis=1)

        # Remove points too close to the sensor origin. These are commonly noisy
        # and can destabilize later sanity checks such as camera projection.
        distance_mask = np.linalg.norm(xyz, axis=1) > 1.0
        return points[finite_mask & distance_mask]

    def save_sample(self, sample_id: str, sample_dict: dict) -> Path:
        sample_path = self.samples_dir / f"{sample_id}.pt"
        torch.save(sample_dict, sample_path)
        return sample_path

    def build_manifest(self, sample_entries: list[dict]) -> dict:
        return {
            "version": 1,
            "dataset": {
                "name": "nuScenes",
                "version": self.config.version,
                "dataroot": self.config.dataroot,
            },
            "preprocessing": {
                "cameras": list(self.config.cameras),
                "lidar_name": self.config.lidar_name,
                "image_size": [self.config.image_height, self.config.image_width],
                "max_time_diff_ms": self.config.max_time_diff_ms,
                "scene_names": list(self.config.scene_names) if self.config.scene_names else None,
                "git_commit": get_git_commit_hash(),
            },
            "samples": sample_entries,
        }

    def write_manifest(self, manifest: dict) -> Path:
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest_path

    def run(self) -> dict:
        index = NuScenesIndex(version=self.config.version, dataroot=self.config.dataroot, verbose=False)
        manifest_samples: list[dict] = []
        sample_counter = 0

        for sample_view in index.iter_scene_samples(scene_names=self.config.scene_names):
            sample = sample_view.sample

            for camera_name in self.config.cameras:
                if camera_name not in sample["data"]:
                    continue

                synced_pair = find_best_lidar_for_camera(
                    nusc=index.nusc,
                    sample=sample,
                    camera_name=camera_name,
                    lidar_name=self.config.lidar_name,
                    max_time_diff_ms=self.config.max_time_diff_ms,
                    logger=self.logger,
                    scene_name=sample_view.scene_name,
                )
                if synced_pair is None:
                    continue

                camera_path = index.get_sample_data_path(synced_pair.camera_token)
                lidar_path = index.get_sample_data_path(synced_pair.lidar_token)

                image = self.normalize_image(camera_path)
                lidar_points = self.load_lidar_points(lidar_path)
                intrinsics = index.get_intrinsics(synced_pair.camera_token)
                if intrinsics is None:
                    self.logger.warning(
                        "Skipping sample without camera intrinsics scene=%s sample=%s camera=%s",
                        sample_view.scene_name,
                        sample_view.sample_token,
                        camera_name,
                    )
                    continue

                cam_to_world = index.get_sensor_to_world(synced_pair.camera_token)
                lidar_to_world = index.get_sensor_to_world(synced_pair.lidar_token)
                ego_to_world = index.get_ego_to_world(synced_pair.camera_token)

                sample_id = f"sample_{sample_counter:06d}"
                sample_payload = {
                    "scene_id": sample_view.scene_name,
                    "sample_token": sample_view.sample_token,
                    "timestamp": synced_pair.camera_timestamp,
                    "camera_name": camera_name,
                    "image": image,
                    "intrinsics": torch.from_numpy(intrinsics),
                    "cam_to_world": torch.from_numpy(cam_to_world),
                    "lidar_points": lidar_points,
                    "lidar_to_world": torch.from_numpy(lidar_to_world),
                    "ego_pose": torch.from_numpy(ego_to_world),
                    "metadata": {
                        "camera_token": synced_pair.camera_token,
                        "lidar_token": synced_pair.lidar_token,
                        "camera_path": str(camera_path),
                        "lidar_path": str(lidar_path),
                        "time_diff_ms": synced_pair.time_diff_ms,
                    },
                }
                sample_path = self.save_sample(sample_id=sample_id, sample_dict=sample_payload)
                manifest_samples.append(
                    {
                        "sample_id": sample_id,
                        "path": str(sample_path),
                        "scene_id": sample_view.scene_name,
                        "sample_token": sample_view.sample_token,
                        "camera_name": camera_name,
                        "timestamp": synced_pair.camera_timestamp,
                        "time_diff_ms": synced_pair.time_diff_ms,
                    }
                )
                sample_counter += 1

        manifest = self.build_manifest(sample_entries=manifest_samples)
        self.write_manifest(manifest)
        return manifest


def get_git_commit_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def run_preprocessing(config: PreprocessConfig, logger: Optional[logging.Logger] = None) -> dict:
    preprocessor = Preprocessor(config=config, logger=logger)
    return preprocessor.run()

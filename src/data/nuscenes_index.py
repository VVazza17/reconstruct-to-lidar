from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from nuscenes.nuscenes import NuScenes

from src.utils.geometry import make_transform_from_pose, maybe_intrinsics


@dataclass(frozen=True)
class SampleRecordView:
    scene_name: str
    scene_token: str
    sample_token: str
    timestamp: int
    sample: dict


class NuScenesIndex:
    """Thin access layer over nuScenes tables for preprocessing."""

    def __init__(self, version: str, dataroot: str | Path, verbose: bool = False):
        if version != "v1.0-mini":
            raise ValueError("The active pipeline supports nuScenes v1.0-mini only.")

        self.version = version
        self.dataroot = Path(dataroot)
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)

    def iter_scene_samples(
        self,
        scene_names: Optional[Iterable[str]] = None,
    ) -> Iterator[SampleRecordView]:
        allowed_scene_names = set(scene_names) if scene_names is not None else None

        for scene in self.nusc.scene:
            scene_name = scene["name"]
            if allowed_scene_names is not None and scene_name not in allowed_scene_names:
                continue

            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = self.nusc.get("sample", sample_token)
                yield SampleRecordView(
                    scene_name=scene_name,
                    scene_token=scene["token"],
                    sample_token=sample["token"],
                    timestamp=sample["timestamp"],
                    sample=sample,
                )
                sample_token = sample["next"]

    def get_scene(self, token: str) -> dict:
        return self.nusc.get("scene", token)

    def get_sample(self, token: str) -> dict:
        return self.nusc.get("sample", token)

    def get_sample_data_record(self, token: str) -> dict:
        return self.nusc.get("sample_data", token)

    def get_calibrated_sensor(self, token: str) -> dict:
        return self.nusc.get("calibrated_sensor", token)

    def get_ego_pose(self, token: str) -> dict:
        return self.nusc.get("ego_pose", token)

    def get_sensor_name(self, sample_data_token: str) -> str:
        sample_data = self.get_sample_data_record(sample_data_token)
        calibrated_sensor = self.get_calibrated_sensor(sample_data["calibrated_sensor_token"])
        sensor = self.nusc.get("sensor", calibrated_sensor["sensor_token"])
        return sensor["channel"]

    def get_sample_data_path(self, token: str) -> Path:
        return Path(self.nusc.get_sample_data_path(token))

    def get_intrinsics(self, sample_data_token: str) -> Optional[object]:
        sample_data = self.get_sample_data_record(sample_data_token)
        calibrated_sensor = self.get_calibrated_sensor(sample_data["calibrated_sensor_token"])
        return maybe_intrinsics(calibrated_sensor.get("camera_intrinsic"))

    def get_sensor_to_ego(self, sample_data_token: str):
        sample_data = self.get_sample_data_record(sample_data_token)
        calibrated_sensor = self.get_calibrated_sensor(sample_data["calibrated_sensor_token"])
        return make_transform_from_pose(
            rotation_quaternion=calibrated_sensor["rotation"],
            translation=calibrated_sensor["translation"],
        )

    def get_ego_to_world(self, sample_data_token: str):
        sample_data = self.get_sample_data_record(sample_data_token)
        ego_pose = self.get_ego_pose(sample_data["ego_pose_token"])
        return make_transform_from_pose(
            rotation_quaternion=ego_pose["rotation"],
            translation=ego_pose["translation"],
        )

    def get_sensor_to_world(self, sample_data_token: str):
        # nuScenes stores sensor->ego calibration and ego->world pose separately.
        return self.get_ego_to_world(sample_data_token) @ self.get_sensor_to_ego(sample_data_token)

    def get_available_camera_names(self, sample: dict) -> list[str]:
        return sorted(
            sensor_name for sensor_name in sample["data"].keys() if sensor_name.startswith("CAM_")
        )

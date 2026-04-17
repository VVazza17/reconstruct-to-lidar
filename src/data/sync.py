from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncedPair:
    camera_token: str
    lidar_token: str
    camera_timestamp: int
    lidar_timestamp: int
    time_diff_ms: float


def _log_skipped_pair(
    logger: logging.Logger,
    scene_name: str | None,
    sample_token: str | None,
    camera_name: str,
    lidar_name: str,
    time_diff_ms: float,
    max_time_diff_ms: float,
) -> None:
    logger.warning(
        "Skipping camera/lidar pair scene=%s sample=%s camera=%s lidar=%s "
        "time_diff_ms=%.3f threshold_ms=%.3f",
        scene_name,
        sample_token,
        camera_name,
        lidar_name,
        time_diff_ms,
        max_time_diff_ms,
    )


def find_best_lidar_for_camera(
    nusc,
    sample: dict,
    camera_name: str,
    lidar_name: str,
    max_time_diff_ms: float = 50.0,
    logger: Optional[logging.Logger] = None,
    scene_name: str | None = None,
) -> Optional[SyncedPair]:
    """
    Pair a camera sample_data record with the sample's LiDAR record.

    The current pipeline uses the per-sample keyframe association from nuScenes and applies an
    explicit max timestamp threshold. Rejected pairs are logged for traceability.
    """
    active_logger = logger or LOGGER

    camera_token = sample["data"].get(camera_name)
    lidar_token = sample["data"].get(lidar_name)
    if camera_token is None:
        raise KeyError(f"Camera {camera_name} not found in sample {sample['token']}.")
    if lidar_token is None:
        raise KeyError(f"LiDAR {lidar_name} not found in sample {sample['token']}.")

    camera_sample_data = nusc.get("sample_data", camera_token)
    lidar_sample_data = nusc.get("sample_data", lidar_token)
    time_diff_ms = abs(camera_sample_data["timestamp"] - lidar_sample_data["timestamp"]) / 1000.0

    if time_diff_ms > max_time_diff_ms:
        _log_skipped_pair(
            logger=active_logger,
            scene_name=scene_name,
            sample_token=sample.get("token"),
            camera_name=camera_name,
            lidar_name=lidar_name,
            time_diff_ms=time_diff_ms,
            max_time_diff_ms=max_time_diff_ms,
        )
        return None

    return SyncedPair(
        camera_token=camera_token,
        lidar_token=lidar_token,
        camera_timestamp=camera_sample_data["timestamp"],
        lidar_timestamp=lidar_sample_data["timestamp"],
        time_diff_ms=time_diff_ms,
    )

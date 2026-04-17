"""Small helpers for exporter metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.data.dataset import ProcessedNuScenesDataset


@dataclass(frozen=True)
class ExportFieldStatus:
    name: str
    available: bool
    source: str
    note: str | None = None


@dataclass(frozen=True)
class ExportPlan:
    target_name: str
    external_repo: str
    source_manifest_path: Path
    expected_output_layout: tuple[str, ...]
    field_statuses: tuple[ExportFieldStatus, ...]
    missing_requirements: tuple[str, ...]
    notes: tuple[str, ...]


def load_representative_processed_sample(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset = ProcessedNuScenesDataset(manifest_path)
    if len(dataset) == 0:
        raise ValueError(f"Processed manifest '{manifest_path}' does not contain any samples.")

    manifest = dataset.manifest
    sample_entry = dataset.get_sample_entry(0)
    sample = dataset[0]
    return manifest, sample_entry, sample


def build_common_export_field_statuses(
    *,
    manifest: Mapping[str, Any],
    sample_entry: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> tuple[ExportFieldStatus, ...]:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata", {}), Mapping) else {}
    preprocessing = (
        manifest.get("preprocessing", {}) if isinstance(manifest.get("preprocessing", {}), Mapping) else {}
    )
    dataset_info = manifest.get("dataset", {}) if isinstance(manifest.get("dataset", {}), Mapping) else {}

    def metadata_field(name: str) -> bool:
        return name in metadata and metadata[name] not in (None, "")

    def sample_field(name: str) -> bool:
        return name in sample and sample[name] is not None

    def manifest_field(container: Mapping[str, Any], name: str) -> bool:
        return name in container and container[name] is not None

    return (
        ExportFieldStatus("scene_id", sample_field("scene_id"), "sample.scene_id"),
        ExportFieldStatus("sample_token", sample_field("sample_token"), "sample.sample_token"),
        ExportFieldStatus("sample_id", manifest_field(sample_entry, "sample_id"), "manifest.samples[].sample_id"),
        ExportFieldStatus("camera_name", sample_field("camera_name"), "sample.camera_name"),
        ExportFieldStatus("timestamp", sample_field("timestamp"), "sample.timestamp"),
        ExportFieldStatus("camera_intrinsics", sample_field("intrinsics"), "sample.intrinsics"),
        ExportFieldStatus("camera_pose_cam_to_world", sample_field("cam_to_world"), "sample.cam_to_world"),
        ExportFieldStatus("lidar_pose_lidar_to_world", sample_field("lidar_to_world"), "sample.lidar_to_world"),
        ExportFieldStatus("ego_pose", sample_field("ego_pose"), "sample.ego_pose"),
        ExportFieldStatus("raw_camera_path", metadata_field("camera_path"), "sample.metadata.camera_path"),
        ExportFieldStatus("raw_lidar_path", metadata_field("lidar_path"), "sample.metadata.lidar_path"),
        ExportFieldStatus("camera_token", metadata_field("camera_token"), "sample.metadata.camera_token"),
        ExportFieldStatus("lidar_token", metadata_field("lidar_token"), "sample.metadata.lidar_token"),
        ExportFieldStatus(
            "paired_time_diff_ms",
            metadata_field("time_diff_ms"),
            "sample.metadata.time_diff_ms",
        ),
        ExportFieldStatus(
            "processed_image_size",
            manifest_field(preprocessing, "image_size"),
            "manifest.preprocessing.image_size",
            note="This records the resized preprocessing output shape, not the raw source image size.",
        ),
        ExportFieldStatus(
            "dataset_version",
            manifest_field(dataset_info, "version"),
            "manifest.dataset.version",
        ),
        ExportFieldStatus(
            "dataset_dataroot",
            manifest_field(dataset_info, "dataroot"),
            "manifest.dataset.dataroot",
        ),
        ExportFieldStatus(
            "coordinate_frame_metadata",
            sample_field("cam_to_world") and sample_field("lidar_to_world") and sample_field("ego_pose"),
            "sample cam_to_world/lidar_to_world/ego_pose",
            note="World-frame transforms are explicit, but target-repository convention mapping is still exporter-specific.",
        ),
    )

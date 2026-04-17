from __future__ import annotations

from typing import Final


PROCESSED_SAMPLE_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "camera_token",
    "lidar_token",
    "camera_path",
    "lidar_path",
    "time_diff_ms",
)


PROCESSED_SAMPLE_FIELDS: Final[tuple[str, ...]] = (
    "scene_id",
    "sample_token",
    "timestamp",
    "camera_name",
    "image",
    "intrinsics",
    "cam_to_world",
    "lidar_points",
    "lidar_to_world",
    "ego_pose",
    "metadata",
)


PROCESSED_SAMPLE_NOTES: Final[tuple[str, ...]] = (
    "The shared processed sample contract targets nuScenes v1.0-mini first.",
    "Transforms are stored explicitly to avoid hidden coordinate-frame assumptions.",
    "Method-specific exporters should consume this contract instead of reading raw nuScenes tables directly.",
)


PROCESSED_SAMPLE_EXPORT_GAPS: Final[tuple[str, ...]] = (
    "The contract stores a resized image tensor and also keeps a raw camera_path reference, so exporters must choose which image representation they target.",
    "The contract does not yet explicitly state whether intrinsics should be interpreted against raw camera_path images or the resized tensor image shape from manifest.preprocessing.image_size.",
    "The contract does not yet include method-specific scene normalization, scale, or frame-convention mapping metadata for external repository formats.",
    "The contract does not yet include COLMAP sparse reconstruction artifacts, camera/image ids, or SfM points required by COLMAP-style real-scene pipelines.",
)

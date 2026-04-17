"""Frame-convention summaries used by export and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


CURRENT_FRAME_NOTES = (
    "sample.cam_to_world is produced by NuScenesIndex.get_sensor_to_world().",
    "NuScenesIndex.get_sensor_to_world() composes ego_to_world @ sensor_to_ego.",
    "Geometry utilities apply transforms as X_world = transform @ X_local using homogeneous column vectors.",
    "Projection utilities and existing visibility checks treat positive camera-space z as in front of the camera.",
    "The repository carries nuScenes quaternion poses through directly and does not relabel world or camera axes locally.",
)


MISSING_GAUSSIAN_SPLATTING_COLMAP_EVIDENCE = (
    "The repo does not provide COLMAP camera ids, image ids, or sparse point ids.",
    "The repo does not provide feature tracks or multi-view correspondences for SfM reconstruction.",
    "The repo does not validate a mapping from sample.intrinsics to a specific COLMAP camera model.",
    "The repo does not validate a mapping from sample.cam_to_world to COLMAP image extrinsics and quaternion conventions.",
    "The exporter stops at image staging and leaves COLMAP record generation to external tools.",
)


OFFICIAL_GAUSSIAN_SPLATTING_CONVENTION_NOTES = (
    "The official Gaussian Splatting real-scene path expects COLMAP-style sparse reconstruction inputs.",
    "That path depends on camera records, image records, and sparse 3D point artifacts, not just per-frame poses.",
    "Raw images and intrinsics alone are not enough to define a full export for that path.",
)


@dataclass(frozen=True)
class TransformValidation:
    is_rigid_4x4: bool
    has_homogeneous_last_row: bool
    note: str


@dataclass(frozen=True)
class FrameConventionReport:
    pose_name: str
    pose_definition: str
    transform_application: str
    transform_composition: str
    camera_forward_axis_assumption: str
    world_axis_labels_documented: bool
    camera_axis_labels_documented: bool
    handedness_documented: bool
    transform_validation: TransformValidation


@dataclass(frozen=True)
class GaussianSplattingCOLMAPAssessment:
    status: str
    official_reference: str
    official_conversion_notes: tuple[str, ...]
    raw_images_available: bool
    intrinsics_available: bool
    poses_available: bool
    sparse_reconstruction_available: bool
    camera_image_id_mapping_available: bool
    safe_export_established: bool
    blocking_reasons: tuple[str, ...]


def validate_cam_to_world_matrix(cam_to_world: Any) -> TransformValidation:
    shape = getattr(cam_to_world, "shape", None)
    if tuple(shape or ()) != (4, 4):
        return TransformValidation(
            is_rigid_4x4=False,
            has_homogeneous_last_row=False,
            note="cam_to_world must be a 4x4 homogeneous transform.",
        )

    last_row = cam_to_world[-1]
    has_homogeneous_last_row = tuple(float(value) for value in last_row) == (0.0, 0.0, 0.0, 1.0)
    return TransformValidation(
        is_rigid_4x4=True,
        has_homogeneous_last_row=has_homogeneous_last_row,
        note=(
            "The matrix shape matches a homogeneous transform. This validates storage shape only; "
            "it does not validate axis labels, handedness, or downstream exporter compatibility."
        ),
    )


def build_frame_convention_report(sample: Mapping[str, Any]) -> FrameConventionReport:
    cam_to_world = sample.get("cam_to_world")
    return FrameConventionReport(
        pose_name="cam_to_world",
        pose_definition="sensor_to_world homogeneous transform for the camera sample_data record",
        transform_application="X_world = cam_to_world @ X_camera",
        transform_composition="cam_to_world = ego_to_world @ sensor_to_ego",
        camera_forward_axis_assumption="positive camera-space z is treated as in front of the camera",
        world_axis_labels_documented=False,
        camera_axis_labels_documented=False,
        handedness_documented=False,
        transform_validation=validate_cam_to_world_matrix(cam_to_world),
    )


def frame_convention_report_dict(sample: Mapping[str, Any]) -> dict[str, Any]:
    report = build_frame_convention_report(sample)
    payload = asdict(report)
    payload["current_frame_notes"] = list(CURRENT_FRAME_NOTES)
    return payload


def build_gaussian_splatting_colmap_assessment(
    sample: Mapping[str, Any],
) -> GaussianSplattingCOLMAPAssessment:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata", {}), Mapping) else {}
    return GaussianSplattingCOLMAPAssessment(
        status="blocked",
        official_reference="graphdeco-inria/gaussian-splatting real-scene COLMAP input path",
        official_conversion_notes=OFFICIAL_GAUSSIAN_SPLATTING_CONVENTION_NOTES,
        raw_images_available=bool(metadata.get("camera_path")),
        intrinsics_available=sample.get("intrinsics") is not None,
        poses_available=sample.get("cam_to_world") is not None,
        sparse_reconstruction_available=False,
        camera_image_id_mapping_available=False,
        safe_export_established=False,
        blocking_reasons=MISSING_GAUSSIAN_SPLATTING_COLMAP_EVIDENCE,
    )


def gaussian_splatting_colmap_assessment_dict(sample: Mapping[str, Any]) -> dict[str, Any]:
    return asdict(build_gaussian_splatting_colmap_assessment(sample))

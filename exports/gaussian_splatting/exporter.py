"""Gaussian Splatting dataset staging and COLMAP handoff helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.data.dataset import ProcessedNuScenesDataset

from src.shared.export_plan import (
    ExportPlan,
    build_common_export_field_statuses,
    load_representative_processed_sample,
)
from src.shared.frame_conventions import (
    CURRENT_FRAME_NOTES,
    frame_convention_report_dict,
    gaussian_splatting_colmap_assessment_dict,
)
from src.shared.processed_sample import PROCESSED_SAMPLE_EXPORT_GAPS


GAUSSIAN_SPLATTING_OFFICIAL_REPO = "https://github.com/graphdeco-inria/gaussian-splatting"
GAUSSIAN_SPLATTING_PLAN_FILENAME = "export_plan.json"
GAUSSIAN_SPLATTING_METADATA_FILENAME = "export_metadata.json"
GAUSSIAN_SPLATTING_IMAGE_SOURCE_MODE = "raw_camera_path"

GAUSSIAN_SPLATTING_EXPECTED_OUTPUT_LAYOUT = (
    "<export_root>/images/<frame image files>",
    "<export_root>/database.db",
    "<export_root>/sparse/0/cameras.bin|txt",
    "<export_root>/sparse/0/images.bin|txt",
    "<export_root>/sparse/0/points3D.bin|txt",
)

GAUSSIAN_SPLATTING_NOTES = (
    "This adapter prepares data for the official Gaussian Splatting repository.",
    "Training and rendering still happen in the external official repository.",
    "For real captured scenes, the official repo expects COLMAP-style reconstruction inputs rather than a custom scene format.",
)

GAUSSIAN_SPLATTING_MISSING_REQUIREMENTS = (
    "The shared processed contract does not yet contain COLMAP sparse reconstruction artifacts required by the official real-scene pipeline.",
    "This exporter stages raw camera_path images for external COLMAP processing and does not export processed tensor images.",
    "This repo does not define how nuScenes intrinsics map to the COLMAP camera models accepted by the official repo.",
    "This repo does not define how camera/world frame conventions map into COLMAP image extrinsics and ids.",
    "This exporter does not create cameras.bin/txt, images.bin/txt, or points3D.bin/txt.",
    "Sparse reconstruction must be produced externally by COLMAP or another validated SfM pipeline.",
)

GAUSSIAN_SPLATTING_REQUIRED_SPARSE_ARTIFACT_GROUPS = (
    ("cameras.bin", "cameras.txt"),
    ("images.bin", "images.txt"),
    ("points3D.bin", "points3D.txt"),
)


def _build_colmap_handoff(export_root: Path | None = None) -> dict[str, Any]:
    root_text = "<export_root>" if export_root is None else str(export_root.resolve())
    return {
        "run_colmap_externally": True,
        "required_steps": [
            "feature_extraction",
            "feature_matching",
            "mapping",
        ],
        "expected_inputs": {
            "images_dir": f"{root_text}/images",
        },
        "expected_outputs": {
            "database_path": f"{root_text}/database.db",
            "sparse_model_dir": f"{root_text}/sparse/0",
            "sparse_artifacts": [
                "cameras.bin|txt",
                "images.bin|txt",
                "points3D.bin|txt",
            ],
        },
        "next_consumer": {
            "repository": GAUSSIAN_SPLATTING_OFFICIAL_REPO,
            "requirement": "Official Gaussian Splatting expects the COLMAP sparse reconstruction to exist before training.",
        },
        "automation_status": "not_run_by_this_repository",
    }


def _build_post_colmap_handoff(export_root: Path | None = None) -> dict[str, Any]:
    root_text = "<export_root>" if export_root is None else str(export_root.resolve())
    return {
        "ready_for_official_gaussian_splatting_after_colmap": True,
        "required_paths": {
            "scene_root": root_text,
            "images_dir": f"{root_text}/images",
            "database_path": f"{root_text}/database.db",
            "sparse_model_dir": f"{root_text}/sparse/0",
        },
        "required_sparse_artifacts": [
            "cameras.bin|txt",
            "images.bin|txt",
            "points3D.bin|txt",
        ],
        "verification_checks": [
            "database.db exists",
            "sparse/0 exists",
            "sparse/0 contains cameras, images, and points3D artifacts",
            "images directory still contains the staged source images",
            "COLMAP reconstruction succeeded before starting the official Gaussian Splatting workflow",
        ],
        "common_failure_cases": [
            "COLMAP did not produce a sparse/0 model",
            "database.db exists but mapping failed or produced no reconstruction",
            "images were moved or renamed after export, breaking the COLMAP scene layout",
            "Expected sparse artifacts are missing before attempting the official Gaussian Splatting workflow",
        ],
        "consumer": {
            "repository": GAUSSIAN_SPLATTING_OFFICIAL_REPO,
            "entry_condition": "Use the exported scene root only after external COLMAP has produced a valid sparse model.",
        },
        "automation_status": "documented_only_not_run_by_this_repository",
    }


def validate_gaussian_splatting_scene_root(export_root: str | Path) -> dict[str, Any]:
    """
    Validate a staged or post-COLMAP Gaussian Splatting scene root.

    Readiness levels:
    - `ready_for_external_colmap`: staged images plus export metadata are present
    - `ready_for_official_gaussian_splatting`: COLMAP sparse outputs are also present
    - `missing_required_artifacts`: the scene root is incomplete for the next step
    """
    export_root = Path(export_root)
    images_dir = export_root / "images"
    metadata_path = export_root / GAUSSIAN_SPLATTING_METADATA_FILENAME
    plan_path = export_root / GAUSSIAN_SPLATTING_PLAN_FILENAME
    database_path = export_root / "database.db"
    sparse_dir = export_root / "sparse" / "0"

    staged_images = sorted(path.name for path in images_dir.iterdir() if path.is_file()) if images_dir.is_dir() else []

    sparse_artifacts: dict[str, dict[str, Any]] = {}
    sparse_groups_complete = True
    for artifact_group in GAUSSIAN_SPLATTING_REQUIRED_SPARSE_ARTIFACT_GROUPS:
        present_path = None
        for candidate in artifact_group:
            candidate_path = sparse_dir / candidate
            if candidate_path.is_file():
                present_path = candidate_path
                break

        sparse_artifacts[artifact_group[0].split(".")[0]] = {
            "acceptable_names": list(artifact_group),
            "present": present_path is not None,
            "path": str(present_path) if present_path is not None else None,
        }
        sparse_groups_complete = sparse_groups_complete and present_path is not None

    required_for_colmap = {
        "images_dir_exists": images_dir.is_dir(),
        "images_present": len(staged_images) > 0,
        "export_metadata_exists": metadata_path.is_file(),
        "export_plan_exists": plan_path.is_file(),
    }
    ready_for_external_colmap = all(required_for_colmap.values())

    optional_colmap_outputs = {
        "database_exists": database_path.is_file(),
        "sparse_model_dir_exists": sparse_dir.is_dir(),
        "sparse_artifacts": sparse_artifacts,
    }
    ready_for_official_gaussian_splatting = (
        ready_for_external_colmap
        and optional_colmap_outputs["database_exists"]
        and optional_colmap_outputs["sparse_model_dir_exists"]
        and sparse_groups_complete
    )

    missing_required_artifacts: list[str] = []
    if not required_for_colmap["images_dir_exists"]:
        missing_required_artifacts.append("images/")
    if not required_for_colmap["images_present"]:
        missing_required_artifacts.append("staged image files in images/")
    if not required_for_colmap["export_metadata_exists"]:
        missing_required_artifacts.append(GAUSSIAN_SPLATTING_METADATA_FILENAME)
    if not required_for_colmap["export_plan_exists"]:
        missing_required_artifacts.append(GAUSSIAN_SPLATTING_PLAN_FILENAME)
    if ready_for_external_colmap and not optional_colmap_outputs["database_exists"]:
        missing_required_artifacts.append("database.db")
    if ready_for_external_colmap and not optional_colmap_outputs["sparse_model_dir_exists"]:
        missing_required_artifacts.append("sparse/0/")
    if ready_for_external_colmap and optional_colmap_outputs["sparse_model_dir_exists"] and not sparse_groups_complete:
        for key, artifact in sparse_artifacts.items():
            if not artifact["present"]:
                missing_required_artifacts.append(f"sparse/0/{key}.bin|txt")

    if ready_for_official_gaussian_splatting:
        readiness_state = "ready_for_official_gaussian_splatting"
    elif ready_for_external_colmap:
        readiness_state = "ready_for_external_colmap"
    else:
        readiness_state = "missing_required_artifacts"

    return {
        "scene_root": str(export_root.resolve()),
        "readiness_state": readiness_state,
        "ready_for_external_colmap": ready_for_external_colmap,
        "ready_for_official_gaussian_splatting": ready_for_official_gaussian_splatting,
        "required_for_external_colmap": required_for_colmap,
        "optional_colmap_outputs": optional_colmap_outputs,
        "staged_image_count": len(staged_images),
        "staged_images": staged_images,
        "missing_required_artifacts": missing_required_artifacts,
    }


def build_export_plan(manifest_path: str | Path) -> ExportPlan:
    """
    Summarize the current Gaussian Splatting export inputs and missing pieces.
    """
    manifest, sample_entry, sample = load_representative_processed_sample(manifest_path)
    return ExportPlan(
        target_name="gaussian_splatting",
        external_repo=GAUSSIAN_SPLATTING_OFFICIAL_REPO,
        source_manifest_path=Path(manifest_path),
        expected_output_layout=GAUSSIAN_SPLATTING_EXPECTED_OUTPUT_LAYOUT,
        field_statuses=build_common_export_field_statuses(
            manifest=manifest,
            sample_entry=sample_entry,
            sample=sample,
        ),
        missing_requirements=GAUSSIAN_SPLATTING_MISSING_REQUIREMENTS,
        notes=GAUSSIAN_SPLATTING_NOTES + PROCESSED_SAMPLE_EXPORT_GAPS,
    )


def _build_gaussian_splatting_plan_document(
    *,
    manifest_path: Path,
    manifest: dict,
    sample_entry: dict,
    sample: dict,
) -> dict:
    return {
        "adapter": "gaussian_splatting",
        "external_repo": GAUSSIAN_SPLATTING_OFFICIAL_REPO,
        "source_manifest_path": str(manifest_path.resolve()),
        "target_layout": list(GAUSSIAN_SPLATTING_EXPECTED_OUTPUT_LAYOUT),
        "status": "needs_external_colmap_outputs",
        "current_frame_notes": list(CURRENT_FRAME_NOTES),
        "frame_convention_report": frame_convention_report_dict(sample),
        "gaussian_splatting_colmap_assessment": gaussian_splatting_colmap_assessment_dict(sample),
        "colmap_handoff": _build_colmap_handoff(),
        "post_colmap_handoff": _build_post_colmap_handoff(),
        "available_inputs": {
            "sample_id": sample_entry.get("sample_id"),
            "scene_id": sample.get("scene_id"),
            "sample_token": sample.get("sample_token"),
            "camera_name": sample.get("camera_name"),
            "timestamp": sample.get("timestamp"),
            "raw_camera_path": sample.get("metadata", {}).get("camera_path"),
            "camera_intrinsics_available": sample.get("intrinsics") is not None,
            "camera_pose_available": sample.get("cam_to_world") is not None,
            "lidar_reference_available": sample.get("metadata", {}).get("lidar_path") is not None,
        },
        "missing_outputs": {
            "sparse_reconstruction_artifacts": [
                "sparse/0/cameras.bin|txt",
                "sparse/0/images.bin|txt",
                "sparse/0/points3D.bin|txt",
            ],
            "camera_image_id_mapping": True,
            "validated_intrinsics_to_colmap_mapping": True,
            "validated_pose_to_colmap_mapping": True,
        },
        "unsafe_assumptions_to_avoid": [
            "Do not invent COLMAP sparse points from per-frame poses alone.",
            "Do not invent camera ids or image ids without a stable COLMAP export scheme.",
            "Do not guess COLMAP quaternion/extrinsic conventions from cam_to_world without validation.",
            "Do not assume raw images plus intrinsics are sufficient for the official real-scene pipeline.",
        ],
        "dataset": manifest.get("dataset", {}),
        "preprocessing": manifest.get("preprocessing", {}),
        "notes": list(GAUSSIAN_SPLATTING_NOTES + PROCESSED_SAMPLE_EXPORT_GAPS),
        "unresolved_gaps": list(GAUSSIAN_SPLATTING_MISSING_REQUIREMENTS),
    }


def _require_raw_camera_path(sample: dict[str, Any]) -> Path:
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Processed sample metadata must be a dictionary for export.")

    camera_path = metadata.get("camera_path")
    if not camera_path:
        raise KeyError(
            "Processed sample metadata must include 'camera_path' for the Gaussian Splatting image staging step."
        )

    source_path = Path(camera_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Raw camera image does not exist: {source_path}")
    return source_path


def _build_staged_image_name(sample_entry: dict[str, Any], source_path: Path, index: int) -> str:
    sample_id = sample_entry.get("sample_id", f"sample_{index:06d}")
    suffix = source_path.suffix or ".jpg"
    return f"{sample_id}{suffix.lower()}"


def _build_export_metadata(
    *,
    manifest_path: Path,
    export_root: Path,
    manifest: dict[str, Any],
    exported_images: list[dict[str, Any]],
    sample: dict[str, Any],
) -> dict[str, Any]:
    return {
        "adapter": "gaussian_splatting",
        "external_repo": GAUSSIAN_SPLATTING_OFFICIAL_REPO,
        "source_manifest_path": str(manifest_path.resolve()),
        "export_root": str(export_root.resolve()),
        "image_source_mode": GAUSSIAN_SPLATTING_IMAGE_SOURCE_MODE,
        "stages_inputs_for_external_colmap": True,
        "writes_images": True,
        "writes_sparse_reconstruction": False,
        "writes_cameras_txt_or_bin": False,
        "writes_images_txt_or_bin": False,
        "writes_points3D_txt_or_bin": False,
        "colmap_handoff": _build_colmap_handoff(export_root),
        "post_colmap_handoff": _build_post_colmap_handoff(export_root),
        "target_layout": list(GAUSSIAN_SPLATTING_EXPECTED_OUTPUT_LAYOUT),
        "intrinsics_availability": {
            "available": sample.get("intrinsics") is not None,
            "source": "sample.intrinsics",
            "note": "Intrinsics are available, but no COLMAP camera model mapping is emitted in this step.",
        },
        "pose_availability": {
            "available": sample.get("cam_to_world") is not None,
            "source": "sample.cam_to_world",
            "note": "Poses are available, but no COLMAP extrinsic/quaternion mapping is emitted in this step.",
        },
        "current_frame_notes": list(CURRENT_FRAME_NOTES),
        "frame_convention_report": frame_convention_report_dict(sample),
        "gaussian_splatting_colmap_assessment": gaussian_splatting_colmap_assessment_dict(sample),
        "dataset": manifest.get("dataset", {}),
        "preprocessing": manifest.get("preprocessing", {}),
        "unresolved_gaps": list(GAUSSIAN_SPLATTING_MISSING_REQUIREMENTS),
        "exported_images": exported_images,
        "notes": [
            "This step only stages inputs for external COLMAP.",
            "COLMAP feature extraction, matching, and mapping run outside this repo.",
            "Official Gaussian Splatting still depends on sparse reconstruction artifacts produced externally.",
        ],
    }


def write_gaussian_splatting_plan(manifest_path: str | Path, export_root: str | Path) -> dict:
    """
    Write the export plan for the Gaussian Splatting handoff.
    """
    manifest, sample_entry, sample = load_representative_processed_sample(manifest_path)
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    payload = _build_gaussian_splatting_plan_document(
        manifest_path=Path(manifest_path),
        manifest=manifest,
        sample_entry=sample_entry,
        sample=sample,
    )
    output_path = export_root / GAUSSIAN_SPLATTING_PLAN_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def write_gaussian_splatting_dataset(manifest_path: str | Path, export_root: str | Path) -> dict[str, Any]:
    """
    Stage raw images and export metadata for an external COLMAP run.

    Current behavior:
    - create `<export_root>/images/`
    - copy raw camera source images referenced by processed samples into that folder
    - write `export_metadata.json`
    - write `export_plan.json`

    Intentionally omitted in this step:
    - any COLMAP sparse reconstruction artifacts
    - any camera/image id mapping
    - any intrinsics/extrinsics conversion into COLMAP records
    """
    manifest_path = Path(manifest_path)
    export_root = Path(export_root)
    images_dir = export_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = ProcessedNuScenesDataset(manifest_path)
    manifest = dataset.manifest
    exported_images: list[dict[str, Any]] = []
    first_sample: dict[str, Any] | None = None
    first_sample_entry: dict[str, Any] | None = None

    for index in range(len(dataset)):
        sample_entry = dataset.get_sample_entry(index)
        sample = dataset[index]
        if first_sample is None:
            first_sample = sample
            first_sample_entry = sample_entry

        source_path = _require_raw_camera_path(sample)
        staged_name = _build_staged_image_name(sample_entry, source_path, index)
        staged_path = images_dir / staged_name
        shutil.copy2(source_path, staged_path)

        metadata = sample.get("metadata", {})
        exported_images.append(
            {
                "sample_id": sample_entry.get("sample_id"),
                "scene_id": sample.get("scene_id"),
                "sample_token": sample.get("sample_token"),
                "camera_name": sample.get("camera_name"),
                "timestamp": sample.get("timestamp"),
                "source_camera_path": str(source_path),
                "staged_image_path": str(staged_path),
                "camera_token": metadata.get("camera_token"),
                "lidar_token": metadata.get("lidar_token"),
                "time_diff_ms": metadata.get("time_diff_ms"),
            }
        )

    if first_sample is None or first_sample_entry is None:
        raise ValueError(f"Processed manifest '{manifest_path}' does not contain any samples.")

    metadata_payload = _build_export_metadata(
        manifest_path=manifest_path,
        export_root=export_root,
        manifest=manifest,
        exported_images=exported_images,
        sample=first_sample,
    )
    metadata_path = export_root / GAUSSIAN_SPLATTING_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)

    plan_payload = _build_gaussian_splatting_plan_document(
        manifest_path=manifest_path,
        manifest=manifest,
        sample_entry=first_sample_entry,
        sample=first_sample,
    )
    plan_path = export_root / GAUSSIAN_SPLATTING_PLAN_FILENAME
    with plan_path.open("w", encoding="utf-8") as handle:
        json.dump(plan_payload, handle, indent=2)

    return metadata_payload

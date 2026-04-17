import torch

from src.shared.frame_conventions import (
    build_frame_convention_report,
    frame_convention_report_dict,
)


def make_sample(cam_to_world: torch.Tensor) -> dict:
    return {
        "cam_to_world": cam_to_world,
    }


def test_frame_convention_report_documents_current_pose_semantics():
    report = build_frame_convention_report(make_sample(torch.eye(4, dtype=torch.float32)))

    assert report.pose_name == "cam_to_world"
    assert report.pose_definition == "sensor_to_world homogeneous transform for the camera sample_data record"
    assert report.transform_application == "X_world = cam_to_world @ X_camera"
    assert report.transform_composition == "cam_to_world = ego_to_world @ sensor_to_ego"
    assert report.camera_forward_axis_assumption == "positive camera-space z is treated as in front of the camera"
    assert report.world_axis_labels_documented is False
    assert report.camera_axis_labels_documented is False
    assert report.handedness_documented is False


def test_frame_convention_report_validates_cam_to_world_storage_shape():
    report = build_frame_convention_report(make_sample(torch.eye(4, dtype=torch.float32)))

    assert report.transform_validation.is_rigid_4x4 is True
    assert report.transform_validation.has_homogeneous_last_row is True


def test_frame_convention_report_dict_includes_current_notes():
    payload = frame_convention_report_dict(make_sample(torch.eye(4, dtype=torch.float32)))

    assert "current_frame_notes" in payload
    assert any("positive camera-space z" in note for note in payload["current_frame_notes"])

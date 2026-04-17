from __future__ import annotations

import numpy as np
from typing import Optional


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from rotation and translation."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 homogeneous transform."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose transforms left-to-right: compose(A, B) == A @ B."""
    result = np.eye(4, dtype=np.float32)
    for transform in transforms:
        result = result @ np.asarray(transform, dtype=np.float32)
    return result


def quaternion_to_rotation_matrix(quaternion: np.ndarray | list[float]) -> np.ndarray:
    """Convert a nuScenes quaternion [w, x, y, z] into a rotation matrix."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float32)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")

    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def make_transform_from_pose(
    rotation_quaternion: np.ndarray | list[float],
    translation: np.ndarray | list[float],
) -> np.ndarray:
    """Build a transform from nuScenes pose/calibration records."""
    rotation = quaternion_to_rotation_matrix(rotation_quaternion)
    return make_transform(rotation=rotation, translation=np.asarray(translation, dtype=np.float32))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 rigid transform to Nx3 or NxM points.

    Only xyz coordinates are transformed. Extra channels are preserved.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, >=3]")

    xyz = np.asarray(points[:, :3], dtype=np.float32)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xyz_h = np.concatenate([xyz, ones], axis=1)
    transformed_xyz = (np.asarray(transform, dtype=np.float32) @ xyz_h.T).T[:, :3]

    if points.shape[1] == 3:
        return transformed_xyz

    extras = np.asarray(points[:, 3:], dtype=np.float32)
    return np.concatenate([transformed_xyz, extras], axis=1)


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project Nx3 camera-frame points into image coordinates."""
    if points_camera.ndim != 2 or points_camera.shape[1] < 3:
        raise ValueError("points_camera must have shape [N, >=3]")

    xyz = np.asarray(points_camera[:, :3], dtype=np.float32)
    depth = np.clip(xyz[:, 2:3], 1e-6, None)
    uvw = (np.asarray(intrinsics, dtype=np.float32) @ xyz.T).T
    return uvw[:, :2] / depth


def relative_transform(source_to_world: np.ndarray, target_to_world: np.ndarray) -> np.ndarray:
    """
    Compute the transform from the source frame into the target frame.

    If X_world = source_to_world @ X_source and
       X_world = target_to_world @ X_target,
    then X_target = relative_transform(source_to_world, target_to_world) @ X_source.
    """
    return invert_transform(target_to_world) @ np.asarray(source_to_world, dtype=np.float32)


def maybe_intrinsics(camera_intrinsic: Optional[list[list[float]]]) -> Optional[np.ndarray]:
    """Convert optional nuScenes camera intrinsics to ndarray."""
    if camera_intrinsic is None:
        return None
    return np.asarray(camera_intrinsic, dtype=np.float32)

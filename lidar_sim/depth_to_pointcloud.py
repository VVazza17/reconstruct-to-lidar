"""Point-cloud loading, filtering, and simple LiDAR approximation utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.geometry import invert_transform, transform_points


_PLY_DTYPE_MAP = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


@dataclass(frozen=True)
class SimpleLidarSimulationConfig:
    """Configuration for the lightweight point-cloud LiDAR approximation."""

    min_range: float = 1.0
    max_range: float = 80.0
    horizontal_fov_deg: float = 360.0
    vertical_fov_deg: float | None = None
    target_count: int | None = None
    random_seed: int = 0


def make_loose_debug_simulation_config(*, max_range: float = 1_000.0) -> SimpleLidarSimulationConfig:
    """Return a nearly unfiltered configuration for alignment checks."""
    return SimpleLidarSimulationConfig(
        min_range=0.0,
        max_range=max_range,
        horizontal_fov_deg=360.0,
        vertical_fov_deg=None,
        target_count=None,
        random_seed=0,
    )


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _read_ply_header(handle) -> tuple[str, int, list[tuple[str, type[np.generic]]]]:
    format_name: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, type[np.generic]]] = []
    in_vertex_element = False

    while True:
        raw_line = handle.readline()
        if not raw_line:
            raise ValueError("PLY file ended before end_header.")

        line = raw_line.decode("ascii").strip()
        if line == "end_header":
            break
        if line.startswith("format "):
            format_name = line.split()[1]
        elif line.startswith("element "):
            _, element_name, count_text = line.split()
            in_vertex_element = element_name == "vertex"
            if in_vertex_element:
                vertex_count = int(count_text)
        elif in_vertex_element and line.startswith("property "):
            _, property_type, property_name = line.split()
            if property_type == "list":
                raise ValueError("PLY list properties are not supported by this lightweight loader.")
            if property_type not in _PLY_DTYPE_MAP:
                raise ValueError(f"Unsupported PLY property type: {property_type}")
            vertex_properties.append((property_name, _PLY_DTYPE_MAP[property_type]))

    if format_name is None or vertex_count is None or not vertex_properties:
        raise ValueError("PLY header is missing format, vertex count, or vertex properties.")
    return format_name, vertex_count, vertex_properties


def load_ply_points(ply_path: str | Path) -> np.ndarray:
    """Load `Nx3` XYZ points from a PLY file."""
    ply_path = Path(ply_path)
    with ply_path.open("rb") as handle:
        format_name, vertex_count, vertex_properties = _read_ply_header(handle)
        property_names = [name for name, _ in vertex_properties]
        for required_name in ("x", "y", "z"):
            if required_name not in property_names:
                raise ValueError(f"PLY vertex data is missing required property '{required_name}'.")

        if format_name == "ascii":
            rows: list[list[float]] = []
            for _ in range(vertex_count):
                line = handle.readline().decode("ascii").strip()
                if not line:
                    raise ValueError("PLY vertex section ended early.")
                rows.append([float(part) for part in line.split()])
            if vertex_count == 0:
                array = np.empty((0, len(property_names)), dtype=np.float32)
            else:
                array = np.asarray(rows, dtype=np.float32).reshape(vertex_count, len(property_names))
            x_index, y_index, z_index = [property_names.index(name) for name in ("x", "y", "z")]
            return array[:, [x_index, y_index, z_index]]

        if format_name != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {format_name}")

        dtype = np.dtype([(name, np.dtype(dtype).newbyteorder("<")) for name, dtype in vertex_properties])
        structured = np.fromfile(handle, dtype=dtype, count=vertex_count)
        return np.stack(
            [
                structured["x"].astype(np.float32, copy=False),
                structured["y"].astype(np.float32, copy=False),
                structured["z"].astype(np.float32, copy=False),
            ],
            axis=1,
        )


def load_point_cloud(source: str | Path | np.ndarray | torch.Tensor | dict[str, Any], *, field: str = "lidar_points") -> np.ndarray:
    """Load a point cloud from `.ply`, `.npy`, processed `.pt`, or memory."""
    if isinstance(source, dict):
        if field not in source:
            raise KeyError(f"Point cloud source dictionary is missing field '{field}'.")
        return _to_numpy_array(source[field])[:, :3]

    if isinstance(source, (np.ndarray, torch.Tensor)):
        return _to_numpy_array(source)[:, :3]

    source_path = Path(source)
    suffix = source_path.suffix.lower()
    if suffix == ".ply":
        return load_ply_points(source_path)
    if suffix == ".npy":
        return np.load(source_path).astype(np.float32, copy=False)[:, :3]
    if suffix == ".pt":
        payload = torch.load(source_path, map_location="cpu")
        if field not in payload:
            raise KeyError(f"Processed sample '{source_path}' is missing field '{field}'.")
        return _to_numpy_array(payload[field])[:, :3]
    raise ValueError(f"Unsupported point cloud source format: {source_path.suffix}")


def write_point_cloud(path: str | Path, points: np.ndarray) -> Path:
    """Write a point cloud to `.ply` or `.npy`."""
    path = Path(path)
    points = _to_numpy_array(points)[:, :3]
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".npy":
        np.save(path, points.astype(np.float32, copy=False))
        return path

    if path.suffix.lower() != ".ply":
        raise ValueError("Point cloud output path must end with .ply or .npy.")

    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points:
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    return path


def simulate_point_cloud_lidar(
    reconstructed_points_world: np.ndarray,
    lidar_to_world: np.ndarray,
    config: SimpleLidarSimulationConfig,
) -> np.ndarray:
    """Filter reconstructed geometry into a synthetic LiDAR-style point cloud."""
    points_world = _to_numpy_array(reconstructed_points_world)[:, :3]
    world_to_lidar = invert_transform(_to_numpy_array(lidar_to_world))
    points_lidar = transform_points(points_world, world_to_lidar)[:, :3]

    finite_mask = np.isfinite(points_lidar).all(axis=1)
    ranges = np.linalg.norm(points_lidar, axis=1)
    range_mask = (ranges >= config.min_range) & (ranges <= config.max_range)
    mask = finite_mask & range_mask

    horizontal_fov = float(config.horizontal_fov_deg)
    if horizontal_fov < 360.0:
        horizontal_angles = np.degrees(np.arctan2(points_lidar[:, 1], points_lidar[:, 0]))
        mask &= np.abs(horizontal_angles) <= horizontal_fov / 2.0

    if config.vertical_fov_deg is not None:
        planar_range = np.linalg.norm(points_lidar[:, :2], axis=1)
        vertical_angles = np.degrees(np.arctan2(points_lidar[:, 2], np.clip(planar_range, 1e-6, None)))
        mask &= np.abs(vertical_angles) <= float(config.vertical_fov_deg) / 2.0

    filtered = points_lidar[mask]
    if config.target_count is not None and filtered.shape[0] > config.target_count:
        rng = np.random.default_rng(config.random_seed)
        selected = rng.choice(filtered.shape[0], size=config.target_count, replace=False)
        filtered = filtered[np.sort(selected)]

    return filtered.astype(np.float32, copy=False)


def simulate_point_cloud_lidar_with_report(
    reconstructed_points_world: np.ndarray,
    lidar_to_world: np.ndarray,
    config: SimpleLidarSimulationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the LiDAR approximation and return stage-by-stage filter counts."""
    points_world = _to_numpy_array(reconstructed_points_world)[:, :3]
    world_to_lidar = invert_transform(_to_numpy_array(lidar_to_world))
    points_lidar = transform_points(points_world, world_to_lidar)[:, :3]

    counts = {
        "input_point_count": int(points_world.shape[0]),
    }

    finite_mask = np.isfinite(points_lidar).all(axis=1)
    finite_points = points_lidar[finite_mask]
    counts["after_finite_filter_count"] = int(finite_points.shape[0])

    ranges = np.linalg.norm(finite_points, axis=1)
    range_mask = (ranges >= config.min_range) & (ranges <= config.max_range)
    range_points = finite_points[range_mask]
    counts["after_range_filter_count"] = int(range_points.shape[0])

    horizontal_fov = float(config.horizontal_fov_deg)
    horizontal_points = range_points
    if horizontal_fov < 360.0:
        horizontal_angles = np.degrees(np.arctan2(range_points[:, 1], range_points[:, 0]))
        horizontal_points = range_points[np.abs(horizontal_angles) <= horizontal_fov / 2.0]
    counts["after_horizontal_fov_filter_count"] = int(horizontal_points.shape[0])

    vertical_points = horizontal_points
    if config.vertical_fov_deg is not None:
        planar_range = np.linalg.norm(horizontal_points[:, :2], axis=1)
        vertical_angles = np.degrees(np.arctan2(horizontal_points[:, 2], np.clip(planar_range, 1e-6, None)))
        vertical_points = horizontal_points[np.abs(vertical_angles) <= float(config.vertical_fov_deg) / 2.0]
    counts["after_vertical_fov_filter_count"] = int(vertical_points.shape[0])

    downsampled_points = vertical_points
    if config.target_count is not None and vertical_points.shape[0] > config.target_count:
        rng = np.random.default_rng(config.random_seed)
        selected = rng.choice(vertical_points.shape[0], size=config.target_count, replace=False)
        downsampled_points = vertical_points[np.sort(selected)]
    counts["after_downsampling_count"] = int(downsampled_points.shape[0])
    counts["final_output_point_count"] = int(downsampled_points.shape[0])

    if counts["input_point_count"] == 0:
        zero_point_likely_cause = "empty_source_geometry"
    elif counts["after_finite_filter_count"] == 0:
        zero_point_likely_cause = "finite_filtering"
    elif counts["after_range_filter_count"] == 0:
        zero_point_likely_cause = "range_filtering"
    elif counts["after_horizontal_fov_filter_count"] == 0:
        zero_point_likely_cause = "horizontal_fov_filtering"
    elif counts["after_vertical_fov_filter_count"] == 0:
        zero_point_likely_cause = "vertical_fov_filtering"
    elif counts["after_downsampling_count"] == 0:
        zero_point_likely_cause = "downsampling"
    else:
        zero_point_likely_cause = "not_empty"

    report = {
        "filter_counts": counts,
        "zero_point_diagnosis": {
            "output_is_empty": counts["final_output_point_count"] == 0,
            "likely_cause": zero_point_likely_cause,
            "came_from_empty_source_geometry": counts["input_point_count"] == 0,
            "likely_due_to_filtering": counts["input_point_count"] > 0 and counts["final_output_point_count"] == 0,
        },
    }
    return downsampled_points.astype(np.float32, copy=False), report


def simulate_lidar_from_ply(
    ply_path: str | Path,
    lidar_to_world: np.ndarray,
    output_path: str | Path,
    config: SimpleLidarSimulationConfig,
) -> dict[str, Any]:
    """Load a reconstructed PLY, simulate LiDAR, and write the result."""
    reconstructed_points = load_ply_points(ply_path)
    synthetic_points, debug_report = simulate_point_cloud_lidar_with_report(
        reconstructed_points,
        lidar_to_world,
        config,
    )
    output_path = write_point_cloud(output_path, synthetic_points)

    report = {
        "source_ply_path": str(Path(ply_path).resolve()),
        "output_path": str(output_path.resolve()),
        "config": asdict(config),
        **debug_report,
        "input_point_count": int(reconstructed_points.shape[0]),
        "output_point_count": int(synthetic_points.shape[0]),
        "assumptions": [
            "Geometry-based approximation; not a physical LiDAR simulator.",
            "The reconstructed point cloud and lidar_to_world pose are assumed to share a world frame.",
            "No occlusion, intensity, or return modeling.",
        ],
    }

    report_path = output_path.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_project_root_to_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a processed nuScenes dataset manifest.")
    parser.add_argument(
        "--manifest-path",
        default="data/processed/nuscenes_mini/manifest.json",
        help="Path to the processed dataset manifest.json.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to inspect.",
    )
    return parser.parse_args()


def shape_or_missing(value: Any) -> str:
    if value is None:
        return "missing"
    shape = getattr(value, "shape", None)
    if shape is None:
        return "missing"
    return str(tuple(shape))


def main() -> None:
    add_project_root_to_path()
    from src.data.dataset import ProcessedNuScenesDataset

    args = parse_args()
    dataset = ProcessedNuScenesDataset(args.manifest_path)
    sample = dataset[args.index]

    print(f"manifest_path: {Path(args.manifest_path).resolve()}")
    print(f"dataset size: {len(dataset)}")
    print(f"sample index: {args.index}")
    print(f"sample keys: {sorted(sample.keys())}")
    print(f"image shape: {shape_or_missing(sample.get('image'))}")
    print(f"intrinsics shape: {shape_or_missing(sample.get('intrinsics'))}")
    print(f"cam_to_world shape: {shape_or_missing(sample.get('cam_to_world'))}")
    print(f"lidar_points shape: {shape_or_missing(sample.get('lidar_points'))}")
    print(f"lidar_to_world shape: {shape_or_missing(sample.get('lidar_to_world'))}")
    print(f"ego_pose shape: {shape_or_missing(sample.get('ego_pose'))}")


if __name__ == "__main__":
    main()

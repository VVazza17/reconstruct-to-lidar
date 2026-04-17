from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from _bootstrap import project_root

PROJECT_ROOT = project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a processed nuScenes sample.")
    parser.add_argument(
        "sample_path",
        nargs="?",
        help="Path to a .pt sample file. Omit this when using --manifest-path.",
    )
    parser.add_argument(
        "--manifest-path",
        help="Optional manifest.json path. If provided without sample_path, the first sample is inspected.",
    )
    return parser.parse_args()


def shape_or_missing(value: Any) -> str:
    if value is None:
        return "missing"
    shape = getattr(value, "shape", None)
    if shape is None:
        return "missing"
    return str(tuple(shape))


def resolve_sample_path(sample_path: str | None, manifest_path: str | None) -> Path:
    if sample_path is not None:
        return Path(sample_path)

    if manifest_path is None:
        raise ValueError("Provide a sample_path or use --manifest-path to inspect the first sample.")

    manifest_file = Path(manifest_path)
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in manifest: {manifest_file}")

    first_sample_path = Path(samples[0]["path"])
    if first_sample_path.is_absolute():
        return first_sample_path
    return (PROJECT_ROOT / first_sample_path).resolve()


def inspect_sample(sample_path: Path) -> None:
    sample = torch.load(sample_path, map_location="cpu")

    print(f"sample_path: {sample_path}")
    print(f"keys: {sorted(sample.keys())}")
    print(f"image shape: {shape_or_missing(sample.get('image'))}")
    print(f"intrinsics shape: {shape_or_missing(sample.get('intrinsics'))}")
    print(f"cam_to_world shape: {shape_or_missing(sample.get('cam_to_world'))}")
    print(f"lidar_points shape: {shape_or_missing(sample.get('lidar_points'))}")
    print(f"lidar_to_world shape: {shape_or_missing(sample.get('lidar_to_world'))}")
    print(f"ego_pose shape: {shape_or_missing(sample.get('ego_pose'))}")
    print("metadata:")
    print(json.dumps(sample.get("metadata", {}), indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    sample_path = resolve_sample_path(sample_path=args.sample_path, manifest_path=args.manifest_path)
    inspect_sample(sample_path=sample_path)


if __name__ == "__main__":
    main()

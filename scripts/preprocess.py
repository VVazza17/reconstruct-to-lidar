from __future__ import annotations

import argparse
import logging
from _bootstrap import add_project_root_to_path


def parse_args() -> argparse.Namespace:
    add_project_root_to_path()
    from src.data.preprocess import DEFAULT_CAMERAS

    parser = argparse.ArgumentParser(description="Preprocess synchronized nuScenes mini samples.")
    parser.add_argument("--dataroot", required=True, help="Path to the nuScenes dataset root.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/nuscenes_mini",
        help="Directory where processed samples and manifest.json will be written.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version. The active pipeline supports v1.0-mini only.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="Camera channels to preprocess.",
    )
    parser.add_argument("--lidar-name", default="LIDAR_TOP", help="LiDAR channel to pair with cameras.")
    parser.add_argument("--image-height", type=int, default=512, help="Resized image height.")
    parser.add_argument("--image-width", type=int, default=896, help="Resized image width.")
    parser.add_argument(
        "--max-time-diff-ms",
        type=float,
        default=50.0,
        help="Maximum allowed camera/LiDAR timestamp difference in milliseconds.",
    )
    parser.add_argument(
        "--scene-names",
        nargs="*",
        default=None,
        help="Optional subset of scene names, for example scene-0061.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> None:
    add_project_root_to_path()
    from src.data.preprocess import PreprocessConfig, run_preprocessing

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PreprocessConfig(
        dataroot=args.dataroot,
        output_dir=args.output_dir,
        version=args.version,
        cameras=tuple(args.cameras),
        lidar_name=args.lidar_name,
        image_height=args.image_height,
        image_width=args.image_width,
        max_time_diff_ms=args.max_time_diff_ms,
        scene_names=tuple(args.scene_names) if args.scene_names else None,
    )
    manifest = run_preprocessing(config=config)
    logging.getLogger(__name__).info(
        "Preprocessing finished: wrote %d samples to %s",
        len(manifest["samples"]),
        args.output_dir,
    )


if __name__ == "__main__":
    main()

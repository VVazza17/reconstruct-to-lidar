# Reconstruct to LiDAR: Synthetic LiDAR Generation from 3D Reconstruction

Preprocess nuScenes mini, stage scenes for external COLMAP and official 3D Gaussian Splatting, then generate and evaluate simple synthetic LiDAR from reconstructed geometry.

This repo keeps the data plumbing inside the project and leaves structure-from-motion and 3DGS training to external tools. The active path is:

- preprocess synchronized nuScenes camera and LiDAR samples
- stage an image-only scene root for COLMAP and official 3D Gaussian Splatting
- run COLMAP externally
- run official 3DGS externally
- bring the reconstructed `.ply` back here for simple synthetic LiDAR generation and evaluation

## Table of Contents

- [Repository Overview](#repository-overview)
- [Installation](#installation)
- [External Dependencies](#external-dependencies)
- [Dataset Setup](#dataset-setup)
- [Quickstart](#quickstart)
- [Running Tests](#running-tests)
- [Output Artifacts](#output-artifacts)
- [Limitations](#limitations)
- [License](#license)

## Repository Overview

This repository is scoped to the working 3DGS path on `nuScenes v1.0-mini`.

What is implemented here:

- nuScenes preprocessing and synchronized sample generation
- processed sample manifests and dataset loading
- image staging for external COLMAP and official Gaussian Splatting
- scene-root validation for pre-COLMAP and post-COLMAP states
- a simple geometry-based synthetic LiDAR generator from reconstructed point clouds
- basic synthetic-vs-real LiDAR evaluation

What is not implemented here:

- COLMAP execution
- 3D Gaussian Splatting training or rendering
- physically accurate LiDAR simulation

## Installation

Use the requirements files directly.

```bash
python -m venv .venv
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Runtime dependencies:

- `numpy`
- `nuscenes-devkit`
- `Pillow`
- `torch`
- `scipy`

Development dependencies:

- `pytest`
- `black`
- `ruff`

## External Dependencies

This repo depends on external tools for the reconstruction step.

- COLMAP
  - used to build `database.db` and `sparse/0/` from the staged images
  - not invoked by this repo
- Official 3D Gaussian Splatting
  - used to train or optimize a scene after COLMAP
  - not invoked by this repo

Manual handoff points are documented in:

- [`scripts/run_colmap.md`](scripts/run_colmap.md)
- [`exports/gaussian_splatting/README.md`](exports/gaussian_splatting/README.md)
- [`exports/gaussian_splatting/POST_COLMAP_HANDOFF.md`](exports/gaussian_splatting/POST_COLMAP_HANDOFF.md)

## Dataset Setup

This repo is set up for `nuScenes v1.0-mini`.

1. Download `v1.0-mini` from nuScenes.
2. Place the dataset in a local directory.
3. Pass that directory to the preprocessing script with `--dataroot`.

The preprocessing code expects the standard nuScenes layout under the chosen dataroot.

## Quickstart

This is the shortest path for a new user.

All commands and Python snippets below assume:

- your working directory is the repository root
- your Python 3.11 virtual environment is already activated

### 1. Preprocess nuScenes mini

```bash
python scripts/preprocess.py --dataroot /path/to/nuscenes --version v1.0-mini
```

By default this writes:

- `data/processed/nuscenes_mini/manifest.json`
- `data/processed/nuscenes_mini/samples/sample_*.pt`

### 2. Check the processed dataset

```bash
python scripts/check_dataset.py --manifest-path data/processed/nuscenes_mini/manifest.json --index 0
python scripts/inspect_sample.py --manifest-path data/processed/nuscenes_mini/manifest.json
```

### 3. Stage a scene root for COLMAP and 3DGS

Start Python from the repository root:

```bash
python
```

Then run:

```python
from exports.gaussian_splatting.exporter import (
    validate_gaussian_splatting_scene_root,
    write_gaussian_splatting_dataset,
)

manifest_path = "data/processed/nuscenes_mini/manifest.json"
export_root = "outputs/scene_test"

write_gaussian_splatting_dataset(manifest_path, export_root)
print(validate_gaussian_splatting_scene_root(export_root))
```

This creates:

- `outputs/scene_test/images/`
- `outputs/scene_test/export_metadata.json`
- `outputs/scene_test/export_plan.json`

### 4. Run COLMAP externally

Follow the command template in [`scripts/run_colmap.md`](scripts/run_colmap.md).

Expected outputs after COLMAP:

- `<export_root>/database.db`
- `<export_root>/sparse/0/`
- `cameras.bin` or `cameras.txt`
- `images.bin` or `images.txt`
- `points3D.bin` or `points3D.txt`

### 5. Validate the post-COLMAP scene root

Start Python from the repository root:

```bash
python
```

Then run:

```python
from exports.gaussian_splatting.exporter import validate_gaussian_splatting_scene_root

print(validate_gaussian_splatting_scene_root("outputs/scene_test"))
```

The validator distinguishes:

- `missing_required_artifacts`
- `ready_for_external_colmap`
- `ready_for_official_gaussian_splatting`

### 6. Run official 3D Gaussian Splatting externally

Use the validated export root with the official 3DGS repository. This repo does not run 3DGS for you.

At the end of that step you need a reconstructed point cloud `.ply`.

### 7. Generate synthetic LiDAR and evaluate it

Use the end-to-end Python helper with:

- the trained 3DGS `.ply`
- one processed sample `.pt`
- an output directory

Start Python from the repository root:

```bash
python
```

Then run:

```python
from evaluation.pipeline import run_simple_3dgs_lidar_evaluation

result = run_simple_3dgs_lidar_evaluation(
    ply_path="path/to/point_cloud.ply",
    real_sample_source="data/processed/nuscenes_mini/samples/sample_000000.pt",
    output_dir="outputs/lidar_eval_example",
)

print(result["run_summary_path"])
```

## Running Tests

Run the full active test suite:

```bash
pytest
```

Or run the explicit test command used for this repo:

```bash
python -m pytest tests/test_dataset.py tests/test_frame_conventions.py tests/test_gaussian_splatting_exporter.py tests/test_geometry.py tests/test_lidar_sim_and_evaluation.py tests/test_sync.py
```

Formatting and linting:

```bash
black .
ruff check .
```

## Output Artifacts

### Preprocessing

Default output root:

- `data/processed/nuscenes_mini/`

Important files:

- `manifest.json`
- `samples/sample_*.pt`

### Gaussian Splatting Export

Example export root:

- `outputs/scene_test/`

Important files:

- `images/`
- `export_metadata.json`
- `export_plan.json`
- `database.db` after external COLMAP
- `sparse/0/` after external COLMAP

### Synthetic LiDAR and Evaluation

Example evaluation root:

- `outputs/lidar_eval_example/`

Important files:

- `synthetic_lidar.ply`
- `synthetic_lidar.json`
- `alignment_debug.json`
- `evaluation_summary.json`
- `run_summary.json`

## Limitations

- Practical support is limited to `nuScenes v1.0-mini`
- COLMAP and official Gaussian Splatting are external manual steps
- the exporter stages images only; it does not generate COLMAP records
- the synthetic LiDAR path is geometry-based and not physically accurate
- no occlusion, intensity, multi-return, or sensor-specific beam model is implemented
- nearest-neighbor evaluation is basic and intended for comparison, not sensor validation

## License

This project is released under the MIT License.
# Gaussian Splatting Export

This module stages nuScenes images for an external COLMAP run and the official Gaussian Splatting workflow.

The flow is:

1. Shared preprocessing produces processed samples and a manifest.
2. This exporter stages raw source images into a scene root.
3. Validate the staged scene root.
4. Run COLMAP externally to create `database.db` and `sparse/0/`.
5. Validate the post-COLMAP scene root.
6. Hand the same scene root to the official Gaussian Splatting workflow.

---

## Export Output

The staging step writes:

- `images/` copied from raw `metadata.camera_path` source images
- `export_metadata.json` describing the staged export
- `export_plan.json` describing the COLMAP handoff state

These artifacts record:

- raw image availability from `camera_path`
- intrinsics availability from `sample.intrinsics`
- pose availability from `sample.cam_to_world`
- missing COLMAP sparse reconstruction artifacts
- missing camera and image id mappings
- missing conversion into COLMAP-compatible camera and image records

---

## Expected Layout After Export

```text
<export_root>/
+-- images/
|   +-- <staged frame images>
+-- export_metadata.json
+-- export_plan.json
```

---

## Expected Layout After COLMAP

```text
<export_root>/
+-- images/
+-- database.db
+-- sparse/
|   +-- 0/
|       +-- cameras.bin or cameras.txt
|       +-- images.bin or images.txt
|       +-- points3D.bin or points3D.txt
+-- export_metadata.json
+-- export_plan.json
```

---

## Required COLMAP Steps

Run these steps manually or with your own tooling:

1. Feature extraction against `<export_root>/images`
2. Feature matching against `database.db`
3. Sparse mapping to produce `<export_root>/sparse/0/`

An example command template is provided in `../../scripts/run_colmap.md`.

---

## Typical Flow

1. Export the scene with this repository.
2. Run `validate_gaussian_splatting_scene_root(...)` against the export root.
3. Confirm the validator reports `ready_for_external_colmap`.
4. Run external COLMAP.
5. Run the validator again.
6. Confirm it reports `ready_for_official_gaussian_splatting`.
7. Hand the same `<export_root>` scene root to the official Gaussian Splatting workflow.

---

## Handoff To Official Gaussian Splatting

Use the official Gaussian Splatting workflow only after COLMAP has produced a valid sparse reconstruction under `sparse/0/`.

The required inputs are:

- `<export_root>/images/`
- `<export_root>/database.db`
- `<export_root>/sparse/0/`

Before training, verify:

1. `database.db` exists.
2. `sparse/0/` exists.
3. `sparse/0/` contains `cameras.bin|txt`, `images.bin|txt`, and `points3D.bin|txt`.
4. `images/` still contains the staged images used by COLMAP.

A shorter post-COLMAP checklist is in [POST_COLMAP_HANDOFF.md](POST_COLMAP_HANDOFF.md).

---

## Validator States

The scene validator distinguishes:

- `missing_required_artifacts`: the export root is incomplete even for COLMAP
- `ready_for_external_colmap`: staged images and export metadata exist, but COLMAP outputs are still missing
- `ready_for_official_gaussian_splatting`: `database.db`, `sparse/0/`, and the required sparse artifacts are present

---

## Responsibility Boundary

This repo is responsible for:

- shared preprocessing
- image staging
- export metadata and handoff documentation

External tooling is responsible for:

- COLMAP feature extraction
- COLMAP matching
- COLMAP mapping
- official Gaussian Splatting training or rendering
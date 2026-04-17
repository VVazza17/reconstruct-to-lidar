# Post-COLMAP Handoff Checklist

This note lists the manual checks between a completed COLMAP run and the official Gaussian Splatting workflow.

For a quick repository-local readiness check before or after COLMAP, use `validate_gaussian_splatting_scene_root(...)` from [exporter.py](exporter.py).

---

## Expected Scene Layout

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

## What To Verify After COLMAP

1. `database.db` exists at `<export_root>/database.db`.
2. `sparse/0/` exists under `<export_root>/sparse/0/`.
3. `sparse/0/` contains:
   - `cameras.bin` or `cameras.txt`
   - `images.bin` or `images.txt`
   - `points3D.bin` or `points3D.txt`
4. `images/` still contains the staged images used for feature extraction and mapping.
5. The sparse model matches the staged image directory.

---

## What To Pass Into The Official Gaussian Splatting Workflow

Use the scene root, not an individual file:

- scene root: `<export_root>`
- images directory: `<export_root>/images`
- sparse model directory: `<export_root>/sparse/0`

Only start the official Gaussian Splatting workflow after those paths are present.

---

## Common Failure Cases

- `database.db` exists but `sparse/0/` does not
- `sparse/0/` exists but one or more sparse artifacts are missing
- images were moved, renamed, or replaced after export
- COLMAP completed feature extraction and matching but did not produce a usable sparse model
- Gaussian Splatting is started from `images/` alone without a completed COLMAP sparse reconstruction

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
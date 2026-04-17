# External COLMAP Command Template

This file is a manual command template for running COLMAP on a scene exported by this repository.

Replace these placeholders before use:

- `<EXPORT_ROOT>`: export directory created by this repository
- `<COLMAP>`: local COLMAP executable name or full path

---

## Expected Input

```text
<EXPORT_ROOT>/
|-- images/
|-- export_metadata.json
`-- export_plan.json
```

---

## Example Commands

```bash
<COLMAP> feature_extractor \
  --database_path <EXPORT_ROOT>/database.db \
  --image_path <EXPORT_ROOT>/images

<COLMAP> exhaustive_matcher \
  --database_path <EXPORT_ROOT>/database.db

<COLMAP> mapper \
  --database_path <EXPORT_ROOT>/database.db \
  --image_path <EXPORT_ROOT>/images \
  --output_path <EXPORT_ROOT>/sparse
```

---

## Expected Output

```text
<EXPORT_ROOT>/
|-- images/
|-- database.db
|-- sparse/
|   `-- 0/
|       |-- cameras.bin or cameras.txt
|       |-- images.bin or images.txt
|       `-- points3D.bin or points3D.txt
|-- export_metadata.json
`-- export_plan.json
```

---

## Notes

- This repository stages images but does not run COLMAP.
- This repository does not create `database.db` or `sparse/0/`.
- This repository does not choose COLMAP camera models or write COLMAP records.
- Run official Gaussian Splatting only after the COLMAP sparse model exists.
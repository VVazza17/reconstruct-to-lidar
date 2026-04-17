from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from torch.utils.data import Dataset, Subset

from src.shared.processed_sample import PROCESSED_SAMPLE_FIELDS


class ProcessedNuScenesDataset(Dataset):
    """
    Dataset wrapper for processed nuScenes samples.

    Assumptions:
    - The manifest contains a top-level "samples" list.
    - Each sample entry contains a "path" field pointing to a `.pt` file.
    - Relative paths in the manifest are resolved from the manifest location and
      its parents so custom output roots still work.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        self.samples = list(self.manifest.get("samples", []))
        if not isinstance(self.samples, list):
            raise ValueError("manifest.json must contain a top-level 'samples' list.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample_entry = self.samples[index]
        sample_path = self.resolve_sample_path(sample_entry["path"])
        sample = torch.load(sample_path, map_location="cpu")
        missing_fields = [field for field in PROCESSED_SAMPLE_FIELDS if field not in sample]
        if missing_fields:
            raise KeyError(
                f"Processed sample '{sample_path}' is missing required fields: {missing_fields}"
            )
        return sample

    def resolve_sample_path(self, sample_path: str | Path) -> Path:
        candidate = Path(sample_path)
        if candidate.is_absolute():
            return candidate

        search_roots = (self.manifest_path.parent, *self.manifest_path.parent.parents)
        for root in search_roots:
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return resolved

        raise FileNotFoundError(
            f"Could not resolve sample path '{sample_path}' from manifest '{self.manifest_path}'."
        )

    def get_sample_entry(self, index: int) -> dict:
        return self.samples[index]

    def filter_indices_by_scene(self, scene_ids: Iterable[str]) -> list[int]:
        allowed_scene_ids = set(scene_ids)
        return [
            index
            for index, sample_entry in enumerate(self.samples)
            if sample_entry.get("scene_id") in allowed_scene_ids
        ]


def split_dataset(
    dataset: ProcessedNuScenesDataset,
    *,
    train_scene_ids: Optional[Sequence[str]] = None,
    val_scene_ids: Optional[Sequence[str]] = None,
    train_indices: Optional[Sequence[int]] = None,
    val_indices: Optional[Sequence[int]] = None,
) -> tuple[Subset, Subset]:
    """
    Build train/validation subsets from either scenes or explicit indices.

    Exactly one split mode should be used:
    - `train_scene_ids` and `val_scene_ids`
    - or `train_indices` and `val_indices`
    """
    using_scene_split = train_scene_ids is not None or val_scene_ids is not None
    using_index_split = train_indices is not None or val_indices is not None

    if using_scene_split and using_index_split:
        raise ValueError("Choose either scene-based splitting or index-based splitting, not both.")
    if not using_scene_split and not using_index_split:
        raise ValueError("Provide scene ids or explicit indices for the split.")

    if using_scene_split:
        train_indices = dataset.filter_indices_by_scene(train_scene_ids or [])
        val_indices = dataset.filter_indices_by_scene(val_scene_ids or [])
    else:
        train_indices = list(train_indices or [])
        val_indices = list(val_indices or [])

    return Subset(dataset, list(train_indices)), Subset(dataset, list(val_indices))

"""Read the frozen image/text features used by the HMCL experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger


class FeatureStore:
    """Loader for the release HDF5 feature layout (real features only, no synthetic fallback)."""

    def __init__(self, root: str | Path, feature_dim: int = 512) -> None:
        self.root = Path(root)
        self.feature_dim = feature_dim

    def _path(self, dataset: str, split: str, modality: str) -> Path:
        return self.root / dataset / f"{dataset}_{split}_{modality}.h5"

    def _ensure_file_exists(self, dataset: str, split: str, modality: str) -> Path:
        path = self._path(dataset, split, modality)
        if not path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {path}. Extract real features first "
                "(scripts/extract_features.py); synthetic fallback has been removed."
            )
        return path

    def _read_h5(self, path: Path) -> tuple[torch.Tensor, list[int], float]:
        with h5py.File(path, "r") as handle:
            features = torch.from_numpy(handle["features"][:]).float()
            id_key = "labels" if "labels" in handle else "ids"
            ids = [int(value) for value in handle[id_key][:]]
            curvature = float(handle.attrs.get("curv", 0.1))
        return features, ids, curvature

    def load_classification(self, dataset: str, split: str) -> dict[str, object]:
        img_p = self._ensure_file_exists(dataset, split, "image")
        txt_p = self._ensure_file_exists(dataset, split, "text")
        image, labels, _ = self._read_h5(img_p)
        text, text_labels, _ = self._read_h5(txt_p)
        if labels != text_labels:
            raise ValueError(f"Image/text labels are not aligned for {dataset}:{split}")
        if image.shape[0] != text.shape[0]:
            raise ValueError(f"Image/text feature counts differ for {dataset}:{split}")
        return {
            "features": image,
            "text_features": text,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def load_retrieval(self, dataset: str, split: str) -> dict[str, object]:
        img_p = self._ensure_file_exists(dataset, split, "image")
        txt_p = self._ensure_file_exists(dataset, split, "text")
        image, image_ids, curvature = self._read_h5(img_p)
        text, text_ids, text_curvature = self._read_h5(txt_p)
        if text_curvature != 1.0 or curvature == 1.0:
            curvature = text_curvature

        ground_truth_path = (
            self.root / dataset / f"{dataset}_{split}_retrieval_gt.json"
        )
        image_to_text = defaultdict(set)
        text_to_image = defaultdict(set)
        if ground_truth_path.exists():
            data = json.loads(ground_truth_path.read_text())
            for key, values in data.get("image_to_text_gt", {}).items():
                image_to_text[int(key)].update(int(value) for value in values)
            for key, values in data.get("text_to_image_gt", {}).items():
                text_to_image[int(key)].update(int(value) for value in values)
        else:
            for i in range(min(len(image_ids), len(text_ids))):
                image_to_text[int(image_ids[i])].add(int(text_ids[i]))
                text_to_image[int(text_ids[i])].add(int(image_ids[i]))

        return {
            "features": image,
            "text_features": text,
            "image_ids": image_ids,
            "text_ids": text_ids,
            "image_to_text_gt": image_to_text,
            "text_to_image_gt": text_to_image,
            "curv": curvature,
        }

    def load(self, dataset: str, split: str, retrieval: bool) -> dict[str, object]:
        if retrieval:
            return self.load_retrieval(dataset, split)
        return self.load_classification(dataset, split)

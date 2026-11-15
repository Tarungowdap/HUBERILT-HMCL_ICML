"""Adapters for the official MERU and HyCoCLIP repositories.

The upstream projects are intentionally not vendored in this repository.  Feature
extraction adds one checked-out upstream repository to ``sys.path`` and uses its
model, tokenizer, dataset catalog, and checkpoint loader directly.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class BackboneSpec:
    package: str
    train_config: str


BACKBONES: dict[str, BackboneSpec] = {
    "meru_vit_l": BackboneSpec(
        package="meru",
        train_config="configs/train_meru_vit_l.py",
    ),
    "meru_vit_b": BackboneSpec(
        package="meru",
        train_config="configs/train_meru_vit_b.py",
    ),
    "hycoclip_vit_b": BackboneSpec(
        package="hycoclip",
        train_config="configs/train_hycoclip_vit_b.py",
    ),
}


@dataclass(frozen=True)
class UpstreamAPI:
    package: str
    lazy_config: Any
    lazy_factory: Any
    checkpoint_manager: Any
    dataset_catalog: Any
    class_names: dict[str, list[str]]
    tokenizer: Any
    coco_captions: Any
    flickr30k_captions: Any

    def load_model(
        self,
        checkpoint: str | Path,
        train_config: str | Path,
        device: torch.device,
    ) -> torch.nn.Module:
        config = self.lazy_config.load(str(train_config))
        model = self.lazy_factory.build_model(config, device).eval()
        self.checkpoint_manager(model=model).load(str(checkpoint))
        return model


def get_backbone_spec(backbone: str) -> BackboneSpec:
    try:
        return BACKBONES[backbone]
    except KeyError as error:
        supported = ", ".join(sorted(BACKBONES))
        raise ValueError(
            f"Unsupported backbone {backbone!r}; choose one of: {supported}"
        ) from error


def default_train_config(backbone: str, upstream_repo: str | Path) -> Path:
    spec = get_backbone_spec(backbone)
    return Path(upstream_repo).resolve() / spec.train_config


def load_upstream_api(backbone: str, upstream_repo: str | Path) -> UpstreamAPI:
    """Load the official package from a user-provided repository checkout."""

    spec = get_backbone_spec(backbone)
    repo = Path(upstream_repo).resolve()
    package_dir = repo / spec.package
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"Expected the official {spec.package} package at {package_dir}. "
            "Clone the upstream repository first; see README.md."
        )

    repo_string = str(repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)

    loaded = sys.modules.get(spec.package)
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if repo not in loaded_path.parents:
            raise RuntimeError(
                f"Python already imported {spec.package} from {loaded_path}, not {repo}. "
                "Run one backbone family per Python process."
            )

    config_module = importlib.import_module(f"{spec.package}.config")
    checkpoint_module = importlib.import_module(
        f"{spec.package}.utils.checkpointing"
    )
    catalog_module = importlib.import_module(f"{spec.package}.evaluation.catalog")
    names_module = importlib.import_module(f"{spec.package}.evaluation.class_names")
    tokenizer_module = importlib.import_module(f"{spec.package}.tokenizer")
    data_module = importlib.import_module(f"{spec.package}.data.evaluation")

    return UpstreamAPI(
        package=spec.package,
        lazy_config=config_module.LazyConfig,
        lazy_factory=config_module.LazyFactory,
        checkpoint_manager=checkpoint_module.CheckpointManager,
        dataset_catalog=catalog_module.DatasetCatalog,
        class_names=names_module.CLASS_NAMES,
        tokenizer=tokenizer_module.Tokenizer,
        coco_captions=data_module.CocoCaptions,
        flickr30k_captions=data_module.Flickr30kCaptions,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)

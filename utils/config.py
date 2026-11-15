"""HMCL optimization configurations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseCLConfig:
    curv: float = 0.1
    device: str = "auto"
    lr: float = 5e-4
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 1024
    entail_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.curv <= 0:
            raise ValueError("curv must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.entail_weight < 0:
            raise ValueError("entail_weight must be non-negative")


@dataclass(frozen=True)
class ClassificationConfig(BaseCLConfig):
    """Classification settings."""


@dataclass(frozen=True)
class RetrievalConfig(BaseCLConfig):
    """Retrieval settings."""

    lr: float = 5e-5
    weight_decay: float = 0.0
    epochs: int = 15
    batch_size: int = 1024
    entail_weight: float = 0.2

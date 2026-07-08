"""Typed config loader.

Loads a yaml file into typed dataclasses. The ``simulator`` and ``model``
blocks stay plain dicts because that is the contract of the VENDORED code
(``generate_dataset(config: dict)`` / ``build_spot_model(config: dict)``) —
retyping them here would invite drift against frozen modules. Everything the
NEW code consumes is typed.

No config value may be a hardcoded absolute path (CLAUDE.md rule 4); paths are
composed at runtime from :mod:`spotpipe.paths` plus relative names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["RunConfig", "InferenceConfig", "Config", "load_config"]


@dataclass(frozen=True)
class RunConfig:
    """Dataset-generation parameters for a run."""

    n_images: int = 50
    seed: int = 0
    split: str = "smoke"

    def __post_init__(self) -> None:
        if self.n_images < 1:
            raise ValueError(f"run.n_images must be >= 1, got {self.n_images}")


@dataclass(frozen=True)
class InferenceConfig:
    """Arguments forwarded to the vendored ``predict_spots``."""

    adc_max: float = 4095.0
    peak_threshold: float = 0.3
    nms_kernel: int = 3
    max_spots: int | None = None


@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: dict = field(default_factory=dict)      # vendored build_spot_model contract
    simulator: dict = field(default_factory=dict)  # vendored generate_dataset contract


def load_config(path: str | Path) -> Config:
    """Load and type-check a yaml config file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}: {path}")

    known = {"run", "inference", "model", "simulator"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config block(s) {sorted(unknown)} in {path}; expected {sorted(known)}")

    run = RunConfig(**(raw.get("run") or {}))
    inference = InferenceConfig(**(raw.get("inference") or {}))
    model = raw.get("model") or {}
    simulator = raw.get("simulator") or {}
    if not isinstance(model, dict) or not isinstance(simulator, dict):
        raise ValueError(f"'model' and 'simulator' blocks must be mappings in {path}")
    return Config(run=run, inference=inference, model=model, simulator=simulator)

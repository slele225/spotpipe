"""Training loop, curriculum, and target construction (build stage 3).

The curriculum ramps SCENE difficulty only (density / overlap / noise /
background), never detector constants; bias/variance is always measured on a
FIXED held-out evaluation set spanning the full final difficulty range. The
phase-1 loss is detection + localization + per-channel intensity NLL only --
no slope/ratio loss. See CLAUDE.md.
"""

from spotpipe.training.dataset import (  # noqa: F401
    Example,
    build_fixed_val_examples,
    collate,
    curriculum_scene_config,
    generate_examples,
)
from spotpipe.training.targets import build_targets  # noqa: F401
from spotpipe.training.train import (  # noqa: F401
    evaluate,
    intensity_match_metrics,
    overfit,
    predict_dataset,
    train,
)

__all__ = [
    "Example",
    "build_targets",
    "build_fixed_val_examples",
    "collate",
    "curriculum_scene_config",
    "generate_examples",
    "evaluate",
    "intensity_match_metrics",
    "overfit",
    "predict_dataset",
    "train",
]

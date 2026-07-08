"""Every vendored module imports, and the model runs a dummy forward pass."""

import importlib

import torch

VENDORED_MODULES = [
    "spotpipe.schema",
    "spotpipe.schema.schema",
    "spotpipe.simulator",
    "spotpipe.simulator.backgrounds",
    "spotpipe.simulator.noise",
    "spotpipe.simulator.psf",
    "spotpipe.simulator.forward_model",
    "spotpipe.simulator.generate_dataset",
    "spotpipe.simulator.benchmark_set",
    "spotpipe.simulator._features",
    "spotpipe.models",
    "spotpipe.models.backbone",
    "spotpipe.models.heads",
    "spotpipe.models.spot_model",
    "spotpipe.losses",
    "spotpipe.losses.detection",
    "spotpipe.losses.intensity",
    "spotpipe.losses.localization",
    "spotpipe.losses.ratio",
]

NEW_MODULES = ["spotpipe", "spotpipe.paths", "spotpipe.config", "spotpipe.cli",
               "spotpipe.data", "spotpipe.benchmark"]


def test_all_modules_import():
    for name in VENDORED_MODULES + NEW_MODULES:
        importlib.import_module(name)


def test_model_forward_dummy():
    from spotpipe.models import build_spot_model

    model = build_spot_model({"base_channels": 8, "blocks_per_branch": 1, "head_mid_channels": 16})
    x = torch.zeros(1, 2, 64, 64)
    out = model(x)
    assert set(out) == {"heatmap", "offset", "logI1", "logI2", "logvar1", "logvar2"}
    assert out["heatmap"].shape == (1, 1, 64, 64)
    assert out["offset"].shape == (1, 2, 64, 64)
    for k in ("logI1", "logI2", "logvar1", "logvar2"):
        assert out[k].shape == (1, 1, 64, 64)

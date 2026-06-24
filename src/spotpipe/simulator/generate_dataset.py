"""Dataset generation driver for the FV3000 forward model.

Produces N synthetic two-channel images from a config, each with its
ground-truth spot table (canonical schema) and per-image metadata, plus a
dataset-level manifest recording the git commit and config used. Reproducible
from a single integer seed.

Conventions (see CLAUDE.md):

* Detector constants are FIXED instrument constants: sampled ONCE per dataset
  (from the seed alone, independent of N) and reused for every image. The
  curriculum varies SCENE difficulty only, never detector constants.
* The validation/eval set is meant to be FIXED across the whole training
  curriculum and to span the full final difficulty range (incl. the dim x
  high-overlap corner). This driver generates images from whatever scene ranges
  the config gives it; curriculum ramping / fixed-val construction is wired in a
  later build stage on top of this primitive.

Outputs (under ``out_dir``):
  manifest.json                 -- git commit, config, detector, image index
  images/image_<id>.npz         -- {'image': uint16[2,H,W]}  (npz is git-ignored)
  spots/spots_<id>.csv          -- ground-truth spots, canonical schema
  meta/meta_<id>.json           -- per-image scene + detector metadata
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from spotpipe.schema import SCHEMA_COLUMNS, write_spots
from spotpipe.simulator import forward_model, noise

__all__ = ["load_simulator_config", "generate_dataset"]


def load_simulator_config(path: str | Path) -> dict:
    """Load a simulator YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _git_commit() -> str:
    """Current commit hash (``-dirty`` if the tree is modified), or ``UNKNOWN``.

    Mirrors ``scripts/new_experiment.py`` so a generated dataset pins the same
    notion of "shared code at a git commit" that an experiment does.
    """
    repo_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""
    return f"{commit}-dirty" if dirty else commit


def generate_dataset(
    config: dict,
    out_dir: str | Path,
    *,
    n_images: int,
    seed: int = 0,
    split: str = "train",
    id_prefix: str | None = None,
) -> dict:
    """Generate and persist ``n_images`` synthetic images.

    Parameters
    ----------
    config : parsed simulator config (``image``/``detector``/``scene`` blocks).
    out_dir : destination directory (created if needed).
    n_images : number of images to generate.
    seed : master seed; the whole dataset is reproducible from it.
    split : label recorded in the manifest (e.g. ``train`` / ``val``).
    id_prefix : image-id prefix; defaults to ``split``.

    Returns the manifest dict (also written to ``out_dir/manifest.json``).
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "spots").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    img_cfg = config.get("image", {})
    shape = (int(img_cfg.get("height", 256)), int(img_cfg.get("width", 256)))
    scene_cfg = config.get("scene", {})
    prefix = id_prefix if id_prefix is not None else split

    # Detector is the fixed instrument: derive it from the seed ALONE so it is
    # stable no matter how many images we generate.
    root = np.random.SeedSequence(int(seed))
    det_seed, img_seed = root.spawn(2)
    detector = noise.sample_detector_params(config.get("detector", {}), np.random.default_rng(det_seed))
    image_seeds = img_seed.spawn(int(n_images))

    index = []
    for i, child in enumerate(image_seeds):
        image_id = f"{prefix}_{i:05d}"
        rng = np.random.default_rng(child)
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sim = forward_model.simulate_image(
            image_id=image_id, shape=shape, scene=scene, detector=detector, rng=rng,
        )

        img_file = out_dir / "images" / f"image_{image_id}.npz"
        spots_file = out_dir / "spots" / f"spots_{image_id}.csv"
        meta_file = out_dir / "meta" / f"meta_{image_id}.json"

        np.savez_compressed(img_file, image=sim.image)
        write_spots(sim.spots, spots_file)
        with open(meta_file, "w", encoding="utf-8") as fh:
            json.dump(sim.meta, fh, indent=2)

        index.append({
            "image_id": image_id,
            "image_file": str(img_file.relative_to(out_dir)),
            "spots_file": str(spots_file.relative_to(out_dir)),
            "meta_file": str(meta_file.relative_to(out_dir)),
            "n_spots": sim.meta["n_spots"],
            "n_saturated": sim.meta["n_saturated"],
        })

    manifest = {
        "git_commit": _git_commit(),
        "seed": int(seed),
        "split": split,
        "n_images": int(n_images),
        "shape": [shape[0], shape[1]],
        "schema_columns": list(SCHEMA_COLUMNS),
        "config": config,
        "detector": forward_model._detector_to_meta(detector),
        "images": index,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest

"""Synthetic dataset plumbing for training: example generation, batching, the
scene-difficulty curriculum, and the FIXED evaluation set (build stage 3).

The forward model (``spotpipe.simulator``) is the data source. Each example is
one simulated two-channel image plus its dense training-target maps
(``training.targets``). Everything is reproducible from integer seeds.

Curriculum (CLAUDE.md): training ramps SCENE difficulty only -- density /
overlap / noise / background -- NEVER the detector constants (the detector is
sampled once and passed in unchanged). The evaluation set is generated ONCE at
the FULL final difficulty range, with the hard corners (dim x high-overlap,
beta = 0, and the +/- beta extremes) deliberately covered, and is held constant
across the whole curriculum so metrics are not confounded by curriculum
progress.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spotpipe.models.spot_model import normalize_counts
from spotpipe.schema import read_spots, records_to_dataframe, write_spots
from spotpipe.simulator import forward_model, noise
from spotpipe.training.targets import TARGET_KEYS, build_targets

__all__ = [
    "Example",
    "generate_examples",
    "collate",
    "curriculum_scene_config",
    "build_fixed_val_examples",
    "build_eval_examples",
    "write_eval_dir",
    "load_eval_examples",
]


@dataclass
class Example:
    """One simulated image: raw counts, dense targets, GT table, metadata."""

    image: torch.Tensor          # [2, H, W] float (raw counts, not yet scaled)
    targets: dict[str, torch.Tensor]
    spots: pd.DataFrame
    meta: dict


def generate_examples(
    scene_cfg: dict,
    detector: noise.DetectorParams,
    *,
    n_images: int,
    seed,
    shape: tuple[int, int],
    heatmap_sigma: float,
    id_prefix: str = "img",
) -> list[Example]:
    """Simulate ``n_images`` images from ``scene_cfg`` and the fixed ``detector``.

    ``seed`` may be an int or a ``numpy.random.SeedSequence``; per-image seeds are
    spawned from it so the batch is reproducible and independent of order.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed))
    children = root.spawn(int(n_images))
    examples: list[Example] = []
    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sim = forward_model.simulate_image(
            image_id=f"{id_prefix}_{i:05d}", shape=shape, scene=scene, detector=detector, rng=rng,
        )
        image = torch.from_numpy(sim.image.astype(np.float32))  # [2, H, W]
        targets = build_targets(sim.spots, shape, heatmap_sigma)
        examples.append(Example(image=image, targets=targets, spots=sim.spots, meta=sim.meta))
    return examples


def collate(examples: list[Example], adc_max: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stack examples into a normalised image batch and a batched target dict."""
    images = torch.stack([normalize_counts(e.image, adc_max) for e in examples])  # [B, 2, H, W]
    targets = {key: torch.stack([e.targets[key] for e in examples]) for key in TARGET_KEYS}
    return images, targets


# --------------------------------------------------------------------------- #
# Curriculum: ramp SCENE difficulty only.                                      #
# --------------------------------------------------------------------------- #
def _lerp(lo: float, hi: float, t: float) -> float:
    return float(lo + (hi - lo) * t)


def _log_lerp(lo: float, hi: float, t: float) -> float:
    """Interpolate geometrically (in log space); ``lo``, ``hi`` must be > 0."""
    import math

    return float(math.exp(_lerp(math.log(lo), math.log(hi), t)))


def curriculum_scene_config(base_scene_cfg: dict, t: float) -> dict:
    """Return a scene config eased toward easy at ``t=0`` and full range at ``t=1``.

    Only density / overlap / noise / background difficulty is ramped. The ratio
    law (beta incl. 0 and negatives), PSF widths, and registration shift are left
    at their FULL ranges at all times -- beta must never become a learnable prior
    (CLAUDE.md), and PSF/registration are not "curriculum difficulty". Detector
    constants are not part of the scene config and are never touched here.
    """
    t = float(min(max(t, 0.0), 1.0))
    cfg = copy.deepcopy(base_scene_cfg)

    # Density: start sparse (max pinned near the floor), widen to the full max.
    dens = cfg.setdefault("density", {})
    d_min = float(dens.get("min", 0.0006))
    d_max = float(dens.get("max", 0.012))
    dens["max"] = _log_lerp(d_min, d_max, t)
    # Over-sampling of the dense octave grows in.
    cfg["oversample_dense_fraction"] = _lerp(0.0, float(cfg.get("oversample_dense_fraction", 0.0)), t)

    # Overlap via clustering: clustered images phase in.
    clus = cfg.setdefault("clustering", {})
    clus["cluster_prob"] = _lerp(0.0, float(clus.get("cluster_prob", 0.0)), t)

    # Intensity: start bright (raise the dim floor) and with no dim over-sampling,
    # ramp toward the full dim tail.
    inten = cfg.setdefault("intensity", {})
    i_min = float(inten.get("log10_min", 1.3))
    i_max = float(inten.get("log10_max", 3.9))
    inten["log10_min"] = _lerp(i_max, i_min, t)            # bright-only -> full range
    inten["dim_bias"] = _lerp(1.0, float(inten.get("dim_bias", 1.0)), t)

    # Background: start near the floor, widen to full level / gradient / structure.
    bg = cfg.setdefault("background", {})
    for key in ("level", "gradient_frac", "structure_frac"):
        spec = bg.get(key)
        if isinstance(spec, dict) and "min" in spec and "max" in spec:
            spec["max"] = _lerp(float(spec["min"]), float(spec["max"]), t)

    # Ratio-law scatter (per-spot noise) eases from its floor up.
    rl = cfg.setdefault("ratio_law", {})
    scat = rl.get("scatter_std")
    if isinstance(scat, dict) and "min" in scat and "max" in scat:
        scat["max"] = _lerp(float(scat["min"]), float(scat["max"]), t)

    return cfg


# --------------------------------------------------------------------------- #
# Fixed evaluation set with deliberate hard-corner coverage.                   #
# --------------------------------------------------------------------------- #
def _pin_scene(
    base: dict,
    *,
    beta: float | None = None,
    density: float | None = None,
    oversample: float | None = None,
    log10_min: float | None = None,
    log10_max: float | None = None,
    dim_bias: float | None = None,
) -> dict:
    """Deep-copy the scene config and pin specific scalars (range min==max)."""
    cfg = copy.deepcopy(base)
    if beta is not None:
        cfg.setdefault("ratio_law", {})["beta"] = {"min": beta, "max": beta}
    if density is not None:
        cfg["density"] = {"min": density, "max": density}
        cfg["oversample_dense_fraction"] = 0.0
    if oversample is not None:
        cfg["oversample_dense_fraction"] = oversample
    inten = cfg.setdefault("intensity", {})
    if log10_min is not None:
        inten["log10_min"] = log10_min
    if log10_max is not None:
        inten["log10_max"] = log10_max
    if dim_bias is not None:
        inten["dim_bias"] = dim_bias
    return cfg


def _fixed_val_specs(base_scene_cfg: dict) -> list[dict]:
    """Scene configs spanning the full range, with the hard corners forced.

    Returned in priority order; the val builder takes as many as it needs and
    pads the remainder with full-range (unpinned) draws.
    """
    rl = base_scene_cfg.get("ratio_law", {})
    beta_spec = rl.get("beta", {"min": -0.6, "max": 0.6})
    beta_lo = float(beta_spec.get("min", -0.6))
    beta_hi = float(beta_spec.get("max", 0.6))

    dens = base_scene_cfg.get("density", {})
    d_max = float(dens.get("max", 0.012))
    inten = base_scene_cfg.get("intensity", {})
    i_min = float(inten.get("log10_min", 1.3))
    i_max = float(inten.get("log10_max", 3.9))

    return [
        _pin_scene(base_scene_cfg, beta=0.0),                  # beta = 0 (no slope)
        _pin_scene(base_scene_cfg, beta=beta_hi),              # strong + beta
        _pin_scene(base_scene_cfg, beta=beta_lo),              # strong - beta
        _pin_scene(                                            # dim x high-overlap corner
            base_scene_cfg, density=d_max, oversample=1.0,
            log10_min=i_min, log10_max=min(i_min + 0.3, i_max), dim_bias=2.0,
        ),
        _pin_scene(                                            # bright x sparse (easy corner)
            base_scene_cfg, density=float(dens.get("min", 0.0006)),
            log10_min=max(i_max - 0.3, i_min), log10_max=i_max,
        ),
    ]


def build_fixed_val_examples(
    base_scene_cfg: dict,
    detector: noise.DetectorParams,
    *,
    n_images: int,
    seed,
    shape: tuple[int, int],
    heatmap_sigma: float,
) -> list[Example]:
    """Build the held-out evaluation set ONCE, with hard-corner coverage.

    The first images use the forced-corner specs (beta=0, +/-beta extremes,
    dim x high-overlap, bright x sparse); any remaining images are full-range
    draws. All are generated from ``seed`` so the set is byte-stable across the
    whole curriculum.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed))
    children = root.spawn(int(n_images))
    specs = _fixed_val_specs(base_scene_cfg)

    examples: list[Example] = []
    for i, child in enumerate(children):
        scene_cfg = specs[i] if i < len(specs) else base_scene_cfg
        rng = np.random.default_rng(child)
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sim = forward_model.simulate_image(
            image_id=f"val_{i:05d}", shape=shape, scene=scene, detector=detector, rng=rng,
        )
        image = torch.from_numpy(sim.image.astype(np.float32))
        targets = build_targets(sim.spots, shape, heatmap_sigma)
        examples.append(Example(image=image, targets=targets, spots=sim.spots, meta=sim.meta))
    return examples


# --------------------------------------------------------------------------- #
# Hard-corner-rich fixed eval split (the FIXED val / test sets of phase 5b).    #
# --------------------------------------------------------------------------- #
def _eval_split_specs(base_scene_cfg: dict, *, n_hard_corner: int) -> list[tuple[str, dict]]:
    """Prioritised ``(kind, scene_cfg)`` list for a hard-corner-rich eval split.

    The dim x high-overlap corner is OVER-sampled (``n_hard_corner`` copies, each
    an independent draw because images get independent seeds) because the project's
    central low-bias / low-variance claim lives there, and the best-checkpoint
    selector needs enough matched pairs in that corner to be stable (CLAUDE.md;
    prompt 5b ``hard_corner_min_pairs``). The beta extremes (incl. beta = 0) and
    the easy bright x sparse corner are pinned once each; the remainder are
    full-range draws.
    """
    rl = base_scene_cfg.get("ratio_law", {})
    beta_spec = rl.get("beta", {"min": -0.6, "max": 0.6})
    beta_lo = float(beta_spec.get("min", -0.6))
    beta_hi = float(beta_spec.get("max", 0.6))

    dens = base_scene_cfg.get("density", {})
    d_min = float(dens.get("min", 0.0006))
    d_max = float(dens.get("max", 0.012))
    inten = base_scene_cfg.get("intensity", {})
    i_min = float(inten.get("log10_min", 1.3))
    i_max = float(inten.get("log10_max", 3.9))

    hard = _pin_scene(
        base_scene_cfg, density=d_max, oversample=1.0,
        log10_min=i_min, log10_max=min(i_min + 0.3, i_max), dim_bias=2.0,
    )
    specs: list[tuple[str, dict]] = [("hard_corner", hard) for _ in range(max(int(n_hard_corner), 0))]
    specs += [
        ("beta0", _pin_scene(base_scene_cfg, beta=0.0)),
        ("beta_hi", _pin_scene(base_scene_cfg, beta=beta_hi)),
        ("beta_lo", _pin_scene(base_scene_cfg, beta=beta_lo)),
        ("bright_sparse", _pin_scene(
            base_scene_cfg, density=d_min,
            log10_min=max(i_max - 0.3, i_min), log10_max=i_max,
        )),
    ]
    return specs


def build_eval_examples(
    base_scene_cfg: dict,
    detector: noise.DetectorParams,
    *,
    n_images: int,
    seed,
    shape: tuple[int, int],
    heatmap_sigma: float,
    n_hard_corner: int = 10,
    id_prefix: str = "val",
) -> list[Example]:
    """Build a FIXED eval split with deliberate, OVER-sampled hard-corner coverage.

    Like :func:`build_fixed_val_examples` but parameterised for the phase-5b roles:
    the first ``n_hard_corner`` images are independent dim x high-overlap draws, then
    the beta extremes / bright x sparse corner, then full-range draws. Every example's
    ``meta['eval_kind']`` records which spec produced it. The whole split is
    byte-stable from ``seed`` (used for BOTH the shared validation set and, with a
    different seed, the frozen test set -- they MUST differ so selection on val and
    reporting on test do not leak; CLAUDE.md / prompt 5b).
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed))
    children = root.spawn(int(n_images))
    specs = _eval_split_specs(base_scene_cfg, n_hard_corner=n_hard_corner)

    examples: list[Example] = []
    for i, child in enumerate(children):
        kind, scene_cfg = specs[i] if i < len(specs) else ("full_range", base_scene_cfg)
        rng = np.random.default_rng(child)
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sim = forward_model.simulate_image(
            image_id=f"{id_prefix}_{i:05d}", shape=shape, scene=scene, detector=detector, rng=rng,
        )
        image = torch.from_numpy(sim.image.astype(np.float32))
        targets = build_targets(sim.spots, shape, heatmap_sigma)
        meta = dict(sim.meta)
        meta["eval_kind"] = kind
        examples.append(Example(image=image, targets=targets, spots=sim.spots, meta=meta))
    return examples


def write_eval_dir(
    examples: list[Example],
    out_dir: str | Path,
    *,
    detector: noise.DetectorParams,
    split: str,
    seed: int,
    shape: tuple[int, int],
    heatmap_sigma: float,
    scene_config: dict | None = None,
    extra_manifest: dict | None = None,
) -> dict:
    """Persist eval ``Example``s in the ``generate_dataset`` on-disk layout.

    Writes ``manifest.json`` + ``images/`` + ``spots/`` + ``meta/`` so the SAME set
    is readable by BOTH the training-side loader (:func:`load_eval_examples`, which
    rebuilds dense targets) and the benchmark harness (``harness.load_eval_set``).
    Building the set ONCE and persisting it is what lets two training runs select on
    byte-identical validation data.
    """
    from spotpipe.simulator.generate_dataset import _git_commit

    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "spots").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    index = []
    for ex in examples:
        image_id = ex.meta["image_id"]
        img_file = out_dir / "images" / f"image_{image_id}.npz"
        spots_file = out_dir / "spots" / f"spots_{image_id}.csv"
        meta_file = out_dir / "meta" / f"meta_{image_id}.json"
        np.savez_compressed(img_file, image=ex.image.numpy().astype(np.float32))
        write_spots(ex.spots, spots_file)
        with open(meta_file, "w", encoding="utf-8") as fh:
            json.dump(ex.meta, fh, indent=2)
        index.append({
            "image_id": image_id,
            "image_file": str(img_file.relative_to(out_dir)),
            "spots_file": str(spots_file.relative_to(out_dir)),
            "meta_file": str(meta_file.relative_to(out_dir)),
            "n_spots": int(ex.meta.get("n_spots", len(ex.spots))),
            "eval_kind": ex.meta.get("eval_kind", "full_range"),
        })

    manifest = {
        "git_commit": _git_commit(),
        "seed": int(seed),
        "split": split,
        "n_images": len(examples),
        "shape": [int(shape[0]), int(shape[1])],
        "heatmap_sigma": float(heatmap_sigma),
        "detector": forward_model._detector_to_meta(detector),
        "scene_config": scene_config,
        "images": index,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_eval_examples(eval_dir: str | Path, *, heatmap_sigma: float) -> list[Example]:
    """Load a persisted eval split (``write_eval_dir`` layout) back into ``Example``s.

    Dense training targets are rebuilt from the ground-truth spot table with the
    given ``heatmap_sigma`` (targets are not stored on disk), so the loaded set is a
    drop-in for periodic validation. Images are returned as raw-count float32 tensors,
    exactly as ``generate_examples`` produces them.
    """
    eval_dir = Path(eval_dir)
    with open(eval_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    shape = tuple(int(s) for s in manifest["shape"])

    examples: list[Example] = []
    for entry in manifest["images"]:
        image = np.load(eval_dir / entry["image_file"])["image"].astype(np.float32)
        spots = records_to_dataframe(read_spots(eval_dir / entry["spots_file"]))
        with open(eval_dir / entry["meta_file"], "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        targets = build_targets(spots, shape, heatmap_sigma)
        examples.append(Example(
            image=torch.from_numpy(image), targets=targets, spots=spots, meta=meta,
        ))
    return examples

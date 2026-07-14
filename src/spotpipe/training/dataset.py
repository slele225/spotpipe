"""Simulator-backed training data: per-image generation, the scene curriculum,
a MULTI-WORKER dataloader, and the fixed evaluation set (measured-detector retrain).

Adapted from the old repo's ``spotpipe.training.dataset`` with three deliberate
departures (see ``training.intensity_window`` and CLAUDE.md / the retrain prompt):

1. **Per-image detector gains** (CHANGE 1) -- ``sample_image_detector`` draws a fresh
   detector per image instead of one fixed detector per dataset.
2. **Per-image intensity window** (CHANGE 2) -- the saturation-safe A1 range is solved
   from the sampled gains + PSF + ratio law, then the curriculum ramps the dim tail
   WITHIN that solved ceiling (bright-only at ``t=0`` -> full dim-biased tail at ``t=1``).
3. **Real multi-worker DataLoader** (CHANGE 5) -- :class:`SpotStreamDataset` +
   :func:`make_loader` generate images in worker processes so the GPU is never
   data-starved (the old inline single-process generation was the ~50-hour bug).

Curriculum (CLAUDE.md): training ramps SCENE difficulty only -- density / overlap /
background / scatter -- NEVER the detector constants, PSF widths, or the ratio-law
slope (which must never become a learnable prior). The intensity dim-tail ramp is
handled here (not in ``curriculum_scene_config``) because the ceiling is per-image.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from spotpipe.models.spot_model import normalize_counts
from spotpipe.schema import read_spots, records_to_dataframe, write_spots
from spotpipe.simulator import forward_model
from spotpipe.training.intensity_window import (
    DetectorConstants,
    sample_image_detector,
    solve_a1_ceiling,
)
from spotpipe.training.targets import TARGET_KEYS, build_targets

__all__ = [
    "Example",
    "IntensityWindowConfig",
    "curriculum_scene_config",
    "generate_examples",
    "collate",
    "SpotStreamDataset",
    "make_loader",
    "build_eval_examples",
    "write_eval_dir",
    "load_eval_examples",
    "summarize_solved_windows",
]


@dataclass
class Example:
    """One simulated image: raw counts, dense targets, GT table, metadata."""

    image: torch.Tensor          # [2, H, W] float (raw counts, not yet scaled)
    targets: dict[str, torch.Tensor]
    spots: pd.DataFrame
    meta: dict


@dataclass(frozen=True)
class IntensityWindowConfig:
    """Knobs for the per-image saturation-safe intensity window (CHANGE 2).

    ``full_decades`` is the dim-tail span (in log10 photons) below the solved
    ceiling at FULL difficulty; ``full_dim_bias`` the dim over-sampling exponent at
    full difficulty (curriculum ramps 1 -> this). ``floor_a1_photons`` keeps spots
    detectable (never solve into sub-photon intensities). ``target_frac`` /
    ``scatter_sigmas`` are the saturation-headroom knobs of ``solve_a1_ceiling``.
    """

    full_decades: float = 2.0
    full_dim_bias: float = 1.6
    floor_a1_photons: float = 10.0
    target_frac: float = 0.85
    scatter_sigmas: float = 3.5

    @classmethod
    def from_config(cls, tcfg: dict) -> "IntensityWindowConfig":
        w = (tcfg or {}).get("intensity_window", {}) or {}
        return cls(
            full_decades=float(w.get("full_decades", 2.0)),
            full_dim_bias=float(w.get("full_dim_bias", 1.6)),
            floor_a1_photons=float(w.get("floor_a1_photons", 10.0)),
            target_frac=float(w.get("target_frac", 0.85)),
            scatter_sigmas=float(w.get("scatter_sigmas", 3.5)),
        )


# --------------------------------------------------------------------------- #
# Curriculum: ramp SCENE difficulty only (density / overlap / background /      #
# scatter). Intensity is handled per image by the window solve; detector, PSF,  #
# and ratio-law slope are ALWAYS at full range.                                 #
# --------------------------------------------------------------------------- #
def _lerp(lo: float, hi: float, t: float) -> float:
    return float(lo + (hi - lo) * t)


def _log_lerp(lo: float, hi: float, t: float) -> float:
    return float(math.exp(_lerp(math.log(lo), math.log(hi), t)))


def curriculum_scene_config(base_scene_cfg: dict, t: float) -> dict:
    """Scene config eased toward easy at ``t=0`` and full range at ``t=1``.

    Ramps density / dense-oversampling / clustering (overlap) / background /
    ratio-law scatter only. The ratio law (slope incl. 0 and negatives), PSF
    widths, and registration shift stay at their FULL ranges at all times. The
    INTENSITY range is NOT ramped here -- it is solved per image (the ceiling is
    gain-dependent) and its dim tail is ramped in ``_resolve_intensity_window``.
    """
    t = float(min(max(t, 0.0), 1.0))
    cfg = copy.deepcopy(base_scene_cfg)

    dens = cfg.setdefault("density", {})
    d_min = float(dens.get("min", 0.0006))
    d_max = float(dens.get("max", 0.012))
    dens["max"] = _log_lerp(d_min, d_max, t)
    cfg["oversample_dense_fraction"] = _lerp(0.0, float(cfg.get("oversample_dense_fraction", 0.0)), t)

    clus = cfg.setdefault("clustering", {})
    clus["cluster_prob"] = _lerp(0.0, float(clus.get("cluster_prob", 0.0)), t)

    bg = cfg.setdefault("background", {})
    for key in ("level", "gradient_frac", "structure_frac"):
        spec = bg.get(key)
        if isinstance(spec, dict) and "min" in spec and "max" in spec:
            spec["max"] = _lerp(float(spec["min"]), float(spec["max"]), t)

    rl = cfg.setdefault("ratio_law", {})
    scat = rl.get("scatter_std")
    if isinstance(scat, dict) and "min" in scat and "max" in scat:
        scat["max"] = _lerp(float(scat["min"]), float(scat["max"]), t)

    return cfg


def _resolve_intensity_window(
    scene: forward_model.SceneParams,
    gain1: float,
    gain2: float,
    consts: DetectorConstants,
    wcfg: IntensityWindowConfig,
    t: float,
) -> dict:
    """Solve the per-image ceiling, then ramp the dim tail below it by curriculum ``t``.

    Returns the fields to write onto ``scene`` (``log10_min``/``log10_max``/
    ``dim_bias``) plus the raw ceiling info for reporting. The ratio-law params come
    from the ALREADY-SAMPLED scene (``scene.alpha`` = sim_intercept, ``scene.beta`` =
    sim_log_slope), so the ceiling respects this image's PSF and slope.
    """
    bg = max(
        float(scene.background1.get("level", 2.0)),
        float(scene.background2.get("level", 2.0)),
    )
    ceil = solve_a1_ceiling(
        gain1=gain1, gain2=gain2, sigma1=scene.sigma1, sigma2=scene.sigma2,
        sim_intercept=scene.alpha, sim_log_slope=scene.beta, scatter_std=scene.scatter_std,
        background=bg, knee1=consts.saturation_knee1, knee2=consts.saturation_knee2,
        target_frac=wcfg.target_frac, scatter_sigmas=wcfg.scatter_sigmas,
        floor_a1_photons=wcfg.floor_a1_photons,
    )
    log10_max = ceil["log10_max"]
    full_min = max(log10_max - wcfg.full_decades, math.log10(wcfg.floor_a1_photons))
    full_min = min(full_min, log10_max)                       # never invert
    log10_min = _lerp(log10_max, full_min, t)                 # bright-only -> full dim tail
    dim_bias = _lerp(1.0, wcfg.full_dim_bias, t)
    return {
        "log10_min": float(min(log10_min, log10_max)),
        "log10_max": float(log10_max),
        "dim_bias": float(dim_bias),
        "ceiling": ceil,
    }


# --------------------------------------------------------------------------- #
# Per-image simulation (gains + solved window + curriculum)                     #
# --------------------------------------------------------------------------- #
def _simulate_one(
    base_scene_cfg: dict,
    consts: DetectorConstants,
    wcfg: IntensityWindowConfig,
    rng: np.random.Generator,
    shape: tuple[int, int],
    heatmap_sigma: float,
    *,
    t: float,
    image_id: str,
    scene_override: dict | None = None,
) -> Example:
    """Simulate ONE image with per-image gains + solved intensity window at difficulty ``t``.

    Order matters: sample the detector gains, ramp the scene by ``t``, sample the
    scene (PSF + ratio law + density...), THEN solve the intensity ceiling from the
    sampled PSF/slope and write it back onto the scene before rendering.
    """
    det, gain1, gain2 = sample_image_detector(rng, consts)

    scene_cfg_t = curriculum_scene_config(base_scene_cfg, t)
    if scene_override:
        scene_cfg_t = _deep_merge(scene_cfg_t, scene_override)
    scene = forward_model.sample_scene_params(scene_cfg_t, rng, shape)

    win = _resolve_intensity_window(scene, gain1, gain2, consts, wcfg, t)
    scene.intensity_log10_max = win["log10_max"]
    scene.intensity_log10_min = win["log10_min"]
    scene.intensity_dim_bias = win["dim_bias"]

    sim = forward_model.simulate_image(
        image_id=image_id, shape=shape, scene=scene, detector=det, rng=rng,
    )
    image = torch.from_numpy(sim.image.astype(np.float32))
    targets = build_targets(sim.spots, shape, heatmap_sigma)
    meta = dict(sim.meta)
    meta["gains"] = {"gain1": gain1, "gain2": gain2}
    meta["intensity_window"] = {k: win[k] for k in ("log10_min", "log10_max", "dim_bias")}
    meta["curriculum_t"] = float(t)
    return Example(image=image, targets=targets, spots=sim.spots, meta=meta)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def generate_examples(
    base_scene_cfg: dict,
    consts: DetectorConstants,
    wcfg: IntensityWindowConfig,
    *,
    n_images: int,
    seed,
    shape: tuple[int, int],
    heatmap_sigma: float,
    t: float = 1.0,
    id_prefix: str = "img",
    scene_override: dict | None = None,
) -> list[Example]:
    """Simulate ``n_images`` independent images (per-image gains + solved window).

    ``seed`` may be an int or a ``numpy.random.SeedSequence``; per-image seeds are
    spawned from it so the batch is reproducible and independent of order.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed))
    children = root.spawn(int(n_images))
    out: list[Example] = []
    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        out.append(_simulate_one(
            base_scene_cfg, consts, wcfg, rng, shape, heatmap_sigma,
            t=t, image_id=f"{id_prefix}_{i:05d}", scene_override=scene_override,
        ))
    return out


def collate(examples: list[Example], adc_max: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stack examples into a normalised image batch and a batched target dict."""
    images = torch.stack([normalize_counts(e.image, adc_max) for e in examples])  # [B, 2, H, W]
    targets = {key: torch.stack([e.targets[key] for e in examples]) for key in TARGET_KEYS}
    return images, targets


# --------------------------------------------------------------------------- #
# Multi-worker streaming dataloader (CHANGE 5)                                  #
# --------------------------------------------------------------------------- #
class SpotStreamDataset(IterableDataset):
    """Yields one PRE-COLLATED training batch per step, generated in worker procs.

    Each training step ``s`` (``start_step..n_steps``) maps to exactly one batch,
    generated deterministically from ``SeedSequence([seed, s])`` and at difficulty
    ``t(s) = min(s / ramp_steps, 1)``. Steps are round-robin partitioned across
    workers (``get_worker_info``), so every step is produced by exactly one worker
    and different steps run in parallel -- the GPU consumes step ``s`` while workers
    prepare ``s+1, s+2, ...``. Batches may arrive slightly out of step order across
    workers; the LR / variance-warmup schedules are driven by the CONSUMING loop's
    own counter, not the batch's embedded step, so that reordering is harmless.

    Yields ``(images[B,2,H,W] float, targets dict, step:int, t:float)``. Use with
    ``DataLoader(batch_size=None)``.
    """

    def __init__(
        self,
        base_scene_cfg: dict,
        consts: DetectorConstants,
        wcfg: IntensityWindowConfig,
        *,
        shape: tuple[int, int],
        heatmap_sigma: float,
        batch_size: int,
        n_steps: int,
        adc_max: float,
        seed: int,
        ramp_steps: int,
        use_curriculum: bool,
        start_step: int = 1,
    ) -> None:
        super().__init__()
        self.base_scene_cfg = base_scene_cfg
        self.consts = consts
        self.wcfg = wcfg
        self.shape = shape
        self.heatmap_sigma = float(heatmap_sigma)
        self.batch_size = int(batch_size)
        self.n_steps = int(n_steps)
        self.adc_max = float(adc_max)
        self.seed = int(seed)
        self.ramp_steps = max(int(ramp_steps), 1)
        self.use_curriculum = bool(use_curriculum)
        self.start_step = int(start_step)

    def _t(self, step: int) -> float:
        return min(step / self.ramp_steps, 1.0) if self.use_curriculum else 1.0

    def _make_batch(self, step: int):
        t = self._t(step)
        examples = generate_examples(
            self.base_scene_cfg, self.consts, self.wcfg,
            n_images=self.batch_size, seed=np.random.SeedSequence([self.seed, step]),
            shape=self.shape, heatmap_sigma=self.heatmap_sigma, t=t,
            id_prefix=f"s{step:06d}",
        )
        images, targets = collate(examples, self.adc_max)
        return images, targets, step, t

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        nworkers = info.num_workers if info is not None else 1
        for step in range(self.start_step, self.n_steps + 1):
            if (step - self.start_step) % nworkers != wid:
                continue
            yield self._make_batch(step)


def make_loader(
    base_scene_cfg: dict,
    consts: DetectorConstants,
    wcfg: IntensityWindowConfig,
    *,
    shape: tuple[int, int],
    heatmap_sigma: float,
    batch_size: int,
    n_steps: int,
    adc_max: float,
    seed: int,
    ramp_steps: int,
    use_curriculum: bool,
    start_step: int = 1,
    num_workers: int,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Build the streaming DataLoader (CHANGE 5): workers generate ahead of the GPU.

    ``num_workers=0`` runs generation inline in the main process (CPU dev / tests).
    With workers, ``persistent_workers=True`` keeps them alive for the single
    continuous pass so they are never respawned mid-run.
    """
    ds = SpotStreamDataset(
        base_scene_cfg, consts, wcfg, shape=shape, heatmap_sigma=heatmap_sigma,
        batch_size=batch_size, n_steps=n_steps, adc_max=adc_max, seed=seed,
        ramp_steps=ramp_steps, use_curriculum=use_curriculum, start_step=start_step,
    )
    kwargs = dict(batch_size=None, num_workers=int(num_workers), pin_memory=bool(pin_memory))
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=int(prefetch_factor))
    return DataLoader(ds, **kwargs)


# --------------------------------------------------------------------------- #
# Fixed evaluation set with deliberate hard-corner coverage                     #
# --------------------------------------------------------------------------- #
def _eval_specs(base_scene_cfg: dict, *, n_hard_corner: int) -> list[tuple[str, dict]]:
    """Prioritised ``(kind, scene_override)`` list for a hard-corner-rich eval split.

    The dim x high-overlap corner (where the project's low-bias claim lives) is
    OVER-sampled: pinned high density + full dense-oversampling + high clustering,
    with the dim tail forced by a high ``dim_bias`` (the intensity ceiling is still
    solved per image from that image's gains). The ratio-law slope extremes (incl.
    0) and a bright x sparse easy corner are pinned once each; the rest are
    full-range draws.
    """
    rl = base_scene_cfg.get("ratio_law", {})
    beta_spec = rl.get("beta", {"min": -0.6, "max": 0.6})
    beta_lo = float(beta_spec.get("min", -0.6))
    beta_hi = float(beta_spec.get("max", 0.6))
    dens = base_scene_cfg.get("density", {})
    d_min = float(dens.get("min", 0.0006))
    d_max = float(dens.get("max", 0.012))

    hard = {
        "density": {"min": d_max, "max": d_max},
        "oversample_dense_fraction": 1.0,
        "clustering": {"cluster_prob": 0.9},
        "intensity": {"dim_bias": 2.5},          # force the dim tail of the solved window
    }
    specs: list[tuple[str, dict]] = [("hard_corner", hard) for _ in range(max(int(n_hard_corner), 0))]
    specs += [
        ("beta0", {"ratio_law": {"beta": {"min": 0.0, "max": 0.0}}}),
        ("beta_hi", {"ratio_law": {"beta": {"min": beta_hi, "max": beta_hi}}}),
        ("beta_lo", {"ratio_law": {"beta": {"min": beta_lo, "max": beta_lo}}}),
        ("bright_sparse", {
            "density": {"min": d_min, "max": d_min},
            "oversample_dense_fraction": 0.0,
            "clustering": {"cluster_prob": 0.0},
            "intensity": {"dim_bias": 1.0},      # bright end of the solved window
        }),
    ]
    return specs


def build_eval_examples(
    base_scene_cfg: dict,
    consts: DetectorConstants,
    wcfg: IntensityWindowConfig,
    *,
    n_images: int,
    seed,
    shape: tuple[int, int],
    heatmap_sigma: float,
    n_hard_corner: int = 10,
    id_prefix: str = "val",
) -> list[Example]:
    """Build a FIXED eval split at FULL difficulty (t=1) with hard-corner coverage.

    Byte-stable from ``seed`` (use different seeds for the selection-val and any
    reporting-test set so selection never leaks). Each image still gets its own
    sampled gains + solved intensity window; the overrides only bias the SCENE
    (density / overlap / dim-bias / slope) toward the forced corner.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed))
    children = root.spawn(int(n_images))
    specs = _eval_specs(base_scene_cfg, n_hard_corner=n_hard_corner)

    out: list[Example] = []
    for i, child in enumerate(children):
        kind, override = specs[i] if i < len(specs) else ("full_range", None)
        rng = np.random.default_rng(child)
        ex = _simulate_one(
            base_scene_cfg, consts, wcfg, rng, shape, heatmap_sigma,
            t=1.0, image_id=f"{id_prefix}_{i:05d}", scene_override=override,
        )
        ex.meta["eval_kind"] = kind
        out.append(ex)
    return out


def write_eval_dir(
    examples: list[Example], out_dir: str | Path, *, split: str, seed: int,
    shape: tuple[int, int], heatmap_sigma: float, extra_manifest: dict | None = None,
) -> dict:
    """Persist eval ``Example``s (npz images + schema spots + meta + manifest)."""
    from spotpipe.simulator.generate_dataset import _git_commit

    out_dir = Path(out_dir)
    for sub in ("images", "spots", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    index = []
    for ex in examples:
        image_id = ex.meta["image_id"]
        np.savez_compressed(out_dir / "images" / f"image_{image_id}.npz",
                            image=ex.image.numpy().astype(np.float32))
        write_spots(ex.spots, out_dir / "spots" / f"spots_{image_id}.csv")
        with open(out_dir / "meta" / f"meta_{image_id}.json", "w", encoding="utf-8") as fh:
            json.dump(ex.meta, fh, indent=2)
        index.append({
            "image_id": image_id,
            "image_file": f"images/image_{image_id}.npz",
            "spots_file": f"spots/spots_{image_id}.csv",
            "meta_file": f"meta/meta_{image_id}.json",
            "n_spots": int(ex.meta.get("n_spots", len(ex.spots))),
            "eval_kind": ex.meta.get("eval_kind", "full_range"),
        })
    manifest = {
        "git_commit": _git_commit(), "seed": int(seed), "split": split,
        "n_images": len(examples), "shape": [int(shape[0]), int(shape[1])],
        "heatmap_sigma": float(heatmap_sigma), "images": index,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_eval_examples(eval_dir: str | Path, *, heatmap_sigma: float) -> list[Example]:
    """Load a persisted eval split back into ``Example``s (targets rebuilt on load)."""
    eval_dir = Path(eval_dir)
    with open(eval_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    shape = tuple(int(s) for s in manifest["shape"])
    out: list[Example] = []
    for entry in manifest["images"]:
        image = np.load(eval_dir / entry["image_file"])["image"].astype(np.float32)
        spots = records_to_dataframe(read_spots(eval_dir / entry["spots_file"]))
        with open(eval_dir / entry["meta_file"], "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        targets = build_targets(spots, shape, heatmap_sigma)
        out.append(Example(image=torch.from_numpy(image), targets=targets, spots=spots, meta=meta))
    return out


# --------------------------------------------------------------------------- #
# Reporting: distribution of solved A1 windows across the training set (CHANGE 2 #4)
# --------------------------------------------------------------------------- #
def summarize_solved_windows(
    base_scene_cfg: dict,
    consts: DetectorConstants,
    wcfg: IntensityWindowConfig,
    *,
    shape: tuple[int, int],
    n_samples: int = 2000,
    seed: int = 0,
    t: float = 1.0,
) -> dict:
    """Sample ``n_samples`` per-image solved windows (no rendering) and summarise.

    Draws the gains + scene (PSF + ratio law) and solves the ceiling exactly as
    training does, at difficulty ``t`` (default full), then reports min/median/max
    of the lower/upper log10 bounds + the ceiling photons + degenerate/binding
    stats. This is the CHANGE-2 step-4 report shown at STOP #1.
    """
    root = np.random.SeedSequence([seed, 424242])
    children = root.spawn(int(n_samples))
    lo, hi, cap, binding, degen = [], [], [], [], 0
    for child in children:
        rng = np.random.default_rng(child)
        _det, g1, g2 = sample_image_detector(rng, consts)
        scene_cfg_t = curriculum_scene_config(base_scene_cfg, t)
        scene = forward_model.sample_scene_params(scene_cfg_t, rng, shape)
        win = _resolve_intensity_window(scene, g1, g2, consts, wcfg, t)
        lo.append(win["log10_min"])
        hi.append(win["log10_max"])
        cap.append(win["ceiling"]["a1_cap_photons"])
        binding.append(win["ceiling"]["binding_channel"])
        degen += int(win["ceiling"]["degenerate"])

    def _s(a):
        a = np.asarray(a, float)
        return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max()),
                "mean": float(a.mean())}

    binding = np.asarray(binding)
    return {
        "n_samples": int(n_samples), "curriculum_t": float(t),
        "log10_min": _s(lo), "log10_max": _s(hi),
        "a1_cap_photons": _s(cap),
        "window_decades": _s(np.asarray(hi) - np.asarray(lo)),
        "binding_ch1_fraction": float((binding == 1).mean()),
        "binding_ch2_fraction": float((binding == 2).mean()),
        "degenerate_fraction": float(degen / max(int(n_samples), 1)),
    }

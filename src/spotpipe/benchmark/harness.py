"""Benchmark harness: run methods over the FIXED synthetic eval set (stage 4).

Orchestration only -- it wires together the method-agnostic pieces:

  load eval set (schema GT + images + meta)
    -> for each method: adapter.predict -> canonical schema file
    -> match predictions to GT (location, gated)
    -> binned metrics (SNR x density) + slope recovery + calibration
    -> metrics tables (CSV + JSON) and the key figures
    -> everything under an output dir, with the git commit pinned.

This is a SYNTHETIC-only benchmark: we hold the ground truth, so bias/variance
are exactly measurable. The eval set is the one built by
``spotpipe.simulator.generate_dataset`` (manifest + ``images/`` + ``spots/`` +
``meta/``); the harness loads that layout. There is no real-data path here.

Key figures written to ``<out>/figures/``:
  * ``recovered_beta.png``       -- recovered-beta vs true-beta, all methods overlaid
                                    (the headline; identity line = unbiased recovery).
  * ``ratio_vs_snr.png``         -- log-ratio bias & spread vs SNR, all methods.
  * ``ratio_vs_density.png``     -- log-ratio bias & spread vs density, all methods.
  * ``calibration.png``          -- predicted-sigma vs realized-error (our model).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.benchmark.adapters import get_adapter
from spotpipe.benchmark.features import attach_features
from spotpipe.benchmark.matching import match_dataset
from spotpipe.benchmark.metrics import compute_metrics
from spotpipe.schema import SCHEMA_COLUMNS, write_spots
from spotpipe.simulator.generate_dataset import _git_commit

__all__ = [
    "EvalImage",
    "load_eval_set",
    "load_frozen_benchmark_set",
    "build_eval_set",
    "run_benchmark",
    "default_benchmark_config",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHODS = ["our_model", "classical_per_channel_aperture", "oracle_center_aperture_divide"]


# --------------------------------------------------------------------------- #
# Eval set                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class EvalImage:
    """One held-out image: raw counts, ground-truth schema table, metadata.

    ``photon`` is an OPTIONAL pre-attached two-channel photon-proportional image
    (``[2, H, W]``); the frozen-set loader attaches the real
    ``images_ch{1,2}_photon`` TIFFs here so adapters that extract intensity from
    photon images (e.g. ``cmeanalysis_plus_aperture``) use them directly. It is
    ``None`` for in-memory eval sets, where such adapters derive it from the raw
    counts + detector meta instead.
    """

    image_id: str
    image: np.ndarray          # [2, H, W]
    meta: dict
    gt: pd.DataFrame           # canonical schema, ground-truth rows
    photon: np.ndarray | None = None   # optional [2, H, W] photon-proportional image


def load_eval_set(eval_dir: str | Path) -> list[EvalImage]:
    """Load an eval set produced by ``generate_dataset`` (manifest-driven)."""
    eval_dir = Path(eval_dir)
    with open(eval_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    items: list[EvalImage] = []
    for entry in manifest["images"]:
        image = np.load(eval_dir / entry["image_file"])["image"]
        gt = pd.read_csv(eval_dir / entry["spots_file"])
        with open(eval_dir / entry["meta_file"], "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        items.append(EvalImage(image_id=entry["image_id"], image=image, meta=meta, gt=gt))
    return items


def load_frozen_benchmark_set(
    frozen_dir: str | Path,
    *,
    limit: int | None = None,
    with_photon: bool = True,
) -> list[EvalImage]:
    """Load the FROZEN benchmark/test set (``benchmark_set`` layout) as EvalImages.

    The frozen set differs from the ``generate_dataset`` layout that
    :func:`load_eval_set` reads: images are plain ``images/<id>.npy`` (uint16
    ``[2,H,W]`` raw counts), ground truth is a single ``ground_truth.csv`` (split
    here by ``image_id``), and per-channel photon TIFFs live in
    ``images_ch{1,2}_photon/``. When ``with_photon`` is set those photon images are
    attached to each :class:`EvalImage` so adapters extract intensity from the real
    photon-proportional images (the fair-photometry rule). Detection/localization
    still uses the raw ``image``; this loader never reads ``audit/``.

    ``limit`` keeps only the first N images (for smoke runs).
    """
    frozen_dir = Path(frozen_dir)
    with open(frozen_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    gt_all = pd.read_csv(frozen_dir / "ground_truth.csv")
    gt_all["image_id"] = gt_all["image_id"].astype(str)
    gt_by_image = {str(k): v.reset_index(drop=True) for k, v in gt_all.groupby("image_id")}

    entries = manifest["images"]
    if limit is not None:
        entries = entries[: int(limit)]

    items: list[EvalImage] = []
    for entry in entries:
        image_id = str(entry["image_id"])
        image = np.load(frozen_dir / entry["image_file"])
        with open(frozen_dir / entry["meta_file"], "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        gt = gt_by_image.get(image_id, pd.DataFrame(columns=SCHEMA_COLUMNS))

        photon = None
        if with_photon and entry.get("ch1_photon") and entry.get("ch2_photon"):
            import tifffile

            p1 = tifffile.imread(frozen_dir / entry["ch1_photon"])
            p2 = tifffile.imread(frozen_dir / entry["ch2_photon"])
            photon = np.stack([p1, p2], axis=0).astype(float)

        items.append(EvalImage(image_id=image_id, image=image, meta=meta, gt=gt, photon=photon))
    return items


def build_eval_set(
    sim_config: dict,
    *,
    n_images: int,
    seed: int = 0,
    shape: tuple[int, int] | None = None,
    id_prefix: str = "eval",
) -> list[EvalImage]:
    """Build an eval set in memory straight from the forward model (no disk).

    Convenience for quick runs / tests: mirrors ``generate_dataset`` (detector
    sampled once from the seed; scene per image) but keeps everything in memory.
    """
    from spotpipe.simulator import forward_model, noise

    img_cfg = sim_config.get("image", {})
    if shape is None:
        shape = (int(img_cfg.get("height", 256)), int(img_cfg.get("width", 256)))
    scene_cfg = sim_config.get("scene", {})

    root = np.random.SeedSequence(int(seed))
    det_seed, img_seed = root.spawn(2)
    detector = noise.sample_detector_params(sim_config.get("detector", {}), np.random.default_rng(det_seed))

    items: list[EvalImage] = []
    for i, child in enumerate(img_seed.spawn(int(n_images))):
        image_id = f"{id_prefix}_{i:05d}"
        rng = np.random.default_rng(child)
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sim = forward_model.simulate_image(
            image_id=image_id, shape=shape, scene=scene, detector=detector, rng=rng,
        )
        items.append(EvalImage(image_id=image_id, image=sim.image, meta=sim.meta, gt=sim.spots))
    return items


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
def default_benchmark_config() -> dict:
    """The built-in defaults (mirrors ``configs/benchmark.yaml``)."""
    return {
        "match_radius_px": 3.0,
        "match_method": "greedy",
        "snr_bins": [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, math.inf],
        "density_bins": [0.0, 1.0, 3.0, 6.0, math.inf],
        "density_radius_px": 4.0,
        "methods": list(DEFAULT_METHODS),
        "baseline": {
            "window_radius_px": 3.0,
            "bg_inner_px": 4.0,
            "bg_outer_px": 7.0,
            "detect_smooth_sigma": 1.2,
            "detect_threshold_rel": 4.0,
            "detect_footprint_px": 3,
            "min_separation_px": 2.0,
        },
        "our_model": {"peak_threshold": 0.3, "nms_kernel": 3, "max_spots": 2000},
        "cmeanalysis": {
            "detections_csv": None,        # normalized CME detections CSV (image_id,x,y,...)
            "detect_channel": 2,           # CME master channel (layout/provenance only)
            "p_detect_source": "constant", # constant | A | score | one_minus_pval | neg_log10_pval
            "window_radius_px": 3.0,       # aperture radius (mirrors the aperture baseline)
            "bg_inner_px": 4.0,
            "bg_outer_px": 7.0,
            "use_photon_images": True,
        },
        "uncertainty": {"n_sigma_bins": 8},
        "adc_max": 4095.0,
    }


def _merged_config(config: dict | None) -> dict:
    cfg = default_benchmark_config()
    if config:
        # benchmark configs may nest under a top-level 'benchmark:' key.
        block = config.get("benchmark", config)
        for k, v in block.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    cfg["snr_bins"] = _coerce_edges(cfg["snr_bins"])
    cfg["density_bins"] = _coerce_edges(cfg["density_bins"])
    return cfg


def _coerce_edges(edges) -> list[float]:
    """Turn YAML bin edges into floats; ``null`` / missing top becomes +inf."""
    out = []
    for e in edges:
        if e is None:
            out.append(math.inf)
        else:
            out.append(float(e))
    return out


# --------------------------------------------------------------------------- #
# Run                                                                          #
# --------------------------------------------------------------------------- #
def run_benchmark(
    eval_set: list[EvalImage],
    config: dict | None = None,
    *,
    out_dir: str | Path,
    methods: list[str] | None = None,
    model=None,
    device: str = "cpu",
    log_fn=print,
) -> dict:
    """Run the listed methods over ``eval_set`` and write tables + figures.

    ``model`` (a trained :class:`SpotModel`) is required iff ``our_model`` is in
    the method list. Methods whose adapter is still a stub (DECODE/Spotiflow/...)
    are skipped with a logged note rather than aborting the run.
    """
    cfg = _merged_config(config)
    methods = list(methods if methods is not None else cfg.get("methods", DEFAULT_METHODS))
    out_dir = Path(out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Use the eval set's own ADC ceiling so our-model input scaling matches.
    if eval_set:
        cfg["adc_max"] = float(eval_set[0].meta.get("detector", {}).get("adc_max", cfg["adc_max"]))

    meta_by_image = {item.image_id: item.meta for item in eval_set}
    gt_all = pd.concat([item.gt for item in eval_set], ignore_index=True) if eval_set else pd.DataFrame(columns=SCHEMA_COLUMNS)
    gt_feat = attach_features(gt_all, meta_by_image, density_radius_px=cfg["density_radius_px"])

    radius = float(cfg["match_radius_px"])
    match_method = cfg["match_method"]

    metrics_by_method: dict[str, dict] = {}
    skipped: list[str] = []
    for name in methods:
        kwargs = {"model": model} if name == "our_model" else {}
        adapter = get_adapter(name, **kwargs)
        try:
            pred = adapter.predict(eval_set, cfg)
        except NotImplementedError as exc:
            log_fn(f"[benchmark] skipping '{name}' (stub adapter): {exc}")
            skipped.append(name)
            continue

        write_spots(pred, out_dir / "predictions" / f"{name}.csv")
        pred_feat = attach_features(pred, meta_by_image, density_radius_px=cfg["density_radius_px"])
        match = match_dataset(gt_feat, pred_feat, max_distance=radius, method=match_method)
        m = compute_metrics(
            gt_feat, pred_feat, match, meta_by_image,
            snr_bins=cfg["snr_bins"], density_bins=cfg["density_bins"],
            n_sigma_bins=int(cfg["uncertainty"].get("n_sigma_bins", 8)),
        )
        metrics_by_method[name] = m

        with open(out_dir / "metrics" / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump(_json_safe(m), fh, indent=2)

        det = m["detection_overall"]
        log_fn(
            f"[benchmark] {name:>22s}: recall={_p(det['recall'])} precision={_p(det['precision'])} "
            f"f1={_p(det['f1'])} | "
            f"ratio bias={_p(m['binned']['overall']['intensity']['log_ratio']['bias'])} "
            f"std={_p(m['binned']['overall']['intensity']['log_ratio']['std'])}"
        )

    # Tables.
    table = _metrics_table(metrics_by_method)
    table.to_csv(out_dir / "metrics_table.csv", index=False)
    slopes = _slopes_table(metrics_by_method)
    slopes.to_csv(out_dir / "slopes.csv", index=False)

    # Figures.
    figures = _write_figures(metrics_by_method, out_dir / "figures")

    manifest = {
        "git_commit": _git_commit(),
        "config": _json_safe(cfg),
        "methods_run": list(metrics_by_method),
        "methods_skipped": skipped,
        "n_eval_images": len(eval_set),
        "n_gt_spots": int(len(gt_all)),
        "schema_columns": list(SCHEMA_COLUMNS),
        "figures": [str(Path(p).relative_to(out_dir)) for p in figures],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return {
        "out_dir": str(out_dir),
        "metrics": metrics_by_method,
        "table": table,
        "slopes": slopes,
        "figures": figures,
        "skipped": skipped,
        "manifest": manifest,
    }


def _p(v) -> str:
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #
def _intensity_cols(block: dict) -> dict:
    out = {}
    for q in ("logI1", "logI2", "log_ratio"):
        s = block[q]
        out[f"bias_{q}"] = s["bias"]
        out[f"std_{q}"] = s["std"]
        out[f"rmse_{q}"] = s["rmse"]
    out["n_pairs"] = block["log_ratio"]["n"]
    return out


def _det_cols(det: dict) -> dict:
    return {
        "n_gt": det["n_gt"], "n_pred": det["n_pred"], "n_matched": det["n_matched"],
        "recall": det["recall"], "precision": det["precision"], "f1": det["f1"],
    }


def _metrics_table(metrics_by_method: dict[str, dict]) -> pd.DataFrame:
    """Flatten every method's binned metrics into one tidy long-form table."""
    rows = []
    for method, m in metrics_by_method.items():
        b = m["binned"]
        # overall
        rows.append({"method": method, "axis": "overall", "bin": "all",
                     **_det_cols(b["overall"]["detection"]), **_intensity_cols(b["overall"]["intensity"])})
        # marginal SNR / density
        for axis, key in (("snr", "snr"), ("density", "density")):
            for r in b[key]:
                rows.append({
                    "method": method, "axis": axis, "bin": r["label"],
                    "bin_lo": r["lo"], "bin_hi": r["hi"],
                    **_det_cols(r["detection"]), **_intensity_cols(r["intensity"]),
                })
        # 2-D SNR x density
        for r in b["snr_x_density"]:
            rows.append({
                "method": method, "axis": "snr_x_density",
                "bin": f"snr={r['snr_bin']};dens={r['density_bin']}",
                **_det_cols(r["detection"]), **_intensity_cols(r["intensity"]),
            })
    return pd.DataFrame(rows)


def _slopes_table(metrics_by_method: dict[str, dict]) -> pd.DataFrame:
    """Per-image recovered-beta rows for BOTH variants (matched_only / end_to_end).

    ``end_to_end`` rows also carry ``precision`` / ``n_pred`` / ``n_tp``;
    ``matched_only`` rows leave those NaN.
    """
    rows = []
    for method, m in metrics_by_method.items():
        for variant in ("matched_only", "end_to_end"):
            for s in m["slope"].get(variant, []):
                rows.append({"method": method, "variant": variant, **s})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _write_figures(metrics_by_method: dict[str, dict], fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    written.append(_fig_recovered_beta(metrics_by_method, fig_dir / "recovered_beta.png"))
    written.append(_fig_ratio_vs_axis(metrics_by_method, "snr", "SNR (limiting channel)", fig_dir / "ratio_vs_snr.png"))
    written.append(_fig_ratio_vs_axis(metrics_by_method, "density", "local neighbours within radius", fig_dir / "ratio_vs_density.png"))
    calib = _fig_calibration(metrics_by_method, fig_dir / "calibration.png")
    if calib:
        written.append(calib)
    return [w for w in written if w]


def _fig_recovered_beta(metrics_by_method: dict[str, dict], path: Path) -> str:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    titles = {
        "matched_only": "matched-only (estimation quality)",
        "end_to_end": "end-to-end (all accepted preds)",
    }
    for ax, variant in zip(axes, ("matched_only", "end_to_end")):
        all_true, all_hat = [], []
        for method, m in metrics_by_method.items():
            rows = m["slope"].get(variant, [])
            tb = [s["true_beta"] for s in rows if _finite(s["true_beta"]) and _finite(s["beta_hat"])]
            bh = [s["beta_hat"] for s in rows if _finite(s["true_beta"]) and _finite(s["beta_hat"])]
            if tb:
                ax.scatter(tb, bh, s=28, alpha=0.7, label=method)
                all_true += tb
                all_hat += bh
        lo, hi = (min(all_true + all_hat), max(all_true + all_hat)) if all_true else (-0.6, 0.6)
        pad = 0.1 * (hi - lo + 1e-6)
        line = [lo - pad, hi + pad]
        ax.plot(line, line, "k--", lw=1, label="identity (unbiased)")
        ax.set_xlabel("true beta (ratio-law slope)")
        ax.set_title(titles[variant])
        ax.legend(fontsize=8)
    axes[0].set_ylabel("recovered beta (OLS: log_ratio ~ logI1)")
    fig.suptitle("Recovered-beta vs true-beta")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def _fig_ratio_vs_axis(metrics_by_method: dict[str, dict], axis_key: str, xlabel: str, path: Path) -> str:
    plt = _plt()
    fig, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, m in metrics_by_method.items():
        bins = m["binned"][axis_key]
        centers, bias, std = [], [], []
        for r in bins:
            lr = r["intensity"]["log_ratio"]
            if lr["n"] > 0 and _finite(lr["bias"]):
                centers.append(r["center"])
                bias.append(lr["bias"])
                std.append(lr["std"] if _finite(lr["std"]) else np.nan)
        if centers:
            ax_b.plot(centers, bias, "o-", alpha=0.8, label=method)
            ax_s.plot(centers, std, "o-", alpha=0.8, label=method)
    ax_b.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_b.set_xlabel(xlabel); ax_b.set_ylabel("log-ratio bias (pred - true)")
    ax_b.set_title("Ratio bias")
    ax_s.set_xlabel(xlabel); ax_s.set_ylabel("log-ratio spread (std)")
    ax_s.set_title("Ratio variance")
    for ax in (ax_b, ax_s):
        ax.legend(fontsize=8)
    fig.suptitle(f"Per-spot log-ratio recovery vs {axis_key}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def _fig_calibration(metrics_by_method: dict[str, dict], path: Path) -> str | None:
    # Calibration only exists for a method that emits uncertainty (our model).
    target = None
    for method, m in metrics_by_method.items():
        if m.get("calibration"):
            target = (method, m["calibration"])
            break
    if target is None:
        return None
    method, calib = target

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    max_v = 1e-6
    for ch, color in (("1", "tab:blue"), ("2", "tab:orange")):
        c = calib["channels"].get(ch)
        if not c or not c["curve"]:
            continue
        ps = [pt["pred_sigma"] for pt in c["curve"]]
        rr = [pt["realized_rms"] for pt in c["curve"]]
        ax.scatter(ps, rr, color=color, alpha=0.8,
                   label=f"ch{ch} (cov1σ={c['coverage_1sigma']:.2f})")
        max_v = max(max_v, max(ps + rr))
    ax.plot([0, max_v], [0, max_v], "k--", lw=1, label="ideal (calibrated)")
    ax.set_xlabel("predicted sigma (uncertainty)")
    ax.set_ylabel("realized RMS |residual|")
    ax.set_title(f"Uncertainty calibration -- {method}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


# --------------------------------------------------------------------------- #
# JSON sanitisation (valid JSON: no Infinity / NaN tokens)                     #
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _load_yaml(path: str | Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_model_from_checkpoint(path: str | Path, device: str = "cpu"):
    """Rebuild a :class:`SpotModel` from a training checkpoint (``checkpoint.pt``)."""
    import torch

    from spotpipe.models.spot_model import build_spot_model

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_spot_model(ckpt.get("config", {}).get("model", {}))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic spot-detection benchmark.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir", help="existing eval set (generate_dataset layout)")
    src.add_argument("--frozen-dir", help="frozen benchmark/test set (benchmark_set layout)")
    src.add_argument("--simulator-config", help="simulator YAML; build a fresh in-memory eval set")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--methods", default=None, help="comma-separated method list (overrides config)")
    parser.add_argument("--checkpoint", default=None, help="our-model checkpoint (.pt) for the our_model method")
    parser.add_argument("--n-images", type=int, default=8, help="images to build with --simulator-config")
    parser.add_argument("--limit", type=int, default=None, help="keep only the first N images (--frozen-dir)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    config = _load_yaml(args.config) if Path(args.config).exists() else {}
    methods = args.methods.split(",") if args.methods else None

    if args.eval_dir:
        eval_set = load_eval_set(args.eval_dir)
    elif args.frozen_dir:
        eval_set = load_frozen_benchmark_set(args.frozen_dir, limit=args.limit)
    else:
        sim_cfg = _load_yaml(args.simulator_config)
        eval_set = build_eval_set(sim_cfg, n_images=args.n_images, seed=args.seed)

    model = None
    requested = methods if methods is not None else _merged_config(config).get("methods", DEFAULT_METHODS)
    if "our_model" in requested:
        if args.checkpoint:
            model = load_model_from_checkpoint(args.checkpoint, device=args.device)
        else:
            print("[benchmark] no --checkpoint given; dropping 'our_model' from the run.")
            methods = [m for m in requested if m != "our_model"]

    result = run_benchmark(eval_set, config, out_dir=args.out, methods=methods, model=model, device=args.device)
    print(f"[benchmark] wrote {len(result['metrics'])} method(s) to {result['out_dir']}")
    print(f"[benchmark] figures: {', '.join(Path(f).name for f in result['figures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

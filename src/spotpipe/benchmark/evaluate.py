"""The ONE shared, blind evaluator (fresh build; disposable tier).

Turns schema-conformant prediction CSVs + benchmark ground truth into the
project's RESULTS: detection metrics, per-spot log-ratio bias/spread, and the
recovered curvature slope ``alpha``. Frozen behaviour lives in
``docs/evaluator_convention.md``; read it before touching this file.

⭐ THE FAIRNESS GUARANTEE. This module is **tool-agnostic**: it ingests folders
of CSVs and does not know or care which tool produced them. Our model and every
external baseline flow through the SAME code path -- there is no per-tool branch
and no ``if method == ...`` anywhere. If two methods were matched or fit with
even slightly different logic, any performance gap could be an artifact of the
evaluator rather than the methods. That is why the evaluator is built exactly
once and validated on ground truth (Gates A-D) before it is trusted.

The pipeline per condition:

1. Load GT (per-image schema CSVs, enumerated by the condition's ``meta.json``)
   and the method's single ``predictions.csv``.
2. Match predicted <-> GT spots **Hungarian**, within-image, gated at
   ``1.0 * max(sigma1, sigma2)`` px (sigma read from ``BENCH_MANIFEST.json``).
3. From the three disjoint outcome classes -- matched / unmatched-GT (FN) /
   unmatched-pred (FP) -- compute detection (recall, precision, F1), per-channel
   signed intensity bias + RMSE, and log-ratio bias/std/RMSE, pooling spots per
   condition (never per-image-then-average).
4. For a curvature set, additionally fit ``alpha`` = unweighted OLS slope of
   ``log(A2/A1)`` vs ``log(sqrt(A1))`` over the pooled matched spots, with the
   analytic OLS slope standard error.

Nothing vendored is touched.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.benchmark.matching import match_dataset
from spotpipe.schema import SCHEMA_COLUMNS

__all__ = [
    "BenchmarkInfo",
    "ConditionSpec",
    "AlphaFit",
    "fit_alpha",
    "detection_metrics",
    "evaluate_condition",
    "load_ground_truth",
    "load_predictions",
    "load_benchmark_info",
    "evaluate_method",
    "evaluate_all",
    "ground_truth_as_predictions",
]

_FAMILIES = ("snr_density", "curvature")

# Frozen defaults (docs/evaluator_convention.md). Configurable but pinned here so
# labelled results stay comparable across runs.
DEFAULT_MATCH_RADIUS_SIGMA: float = 1.0


# --------------------------------------------------------------------------- #
# Benchmark description (sigma + per-condition stratum metadata from manifest) #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConditionSpec:
    """One benchmark condition == one directory == one prediction CSV.

    ``meta`` carries the stratum metadata straight from ``BENCH_MANIFEST.json``
    (``target_snr`` / ``area_density_spots_per_px`` for snr_density cells;
    ``true_alpha`` / ``null_control`` / ``a1_spread_decades`` for curvature sets).
    """

    family: str          # "snr_density" | "curvature"
    label: str           # e.g. "snr=5_density=0.006" | "alpha=0.3"
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.family}/{self.label}"

    @property
    def is_curvature(self) -> bool:
        return self.family == "curvature"

    @property
    def true_alpha(self) -> float | None:
        v = self.meta.get("true_alpha")
        return None if v is None else float(v)


@dataclass(frozen=True)
class BenchmarkInfo:
    """Everything the evaluator needs about a benchmark, read from the manifest."""

    root: Path
    sigma1: float
    sigma2: float
    conditions: list[ConditionSpec]

    @property
    def sigma_ref(self) -> float:
        """The coarser channel sets the localization scale (see convention)."""
        return max(self.sigma1, self.sigma2)

    def match_distance_px(self, match_radius_sigma: float = DEFAULT_MATCH_RADIUS_SIGMA) -> float:
        return float(match_radius_sigma) * self.sigma_ref


def load_benchmark_info(bench_root: str | Path) -> BenchmarkInfo:
    """Read ``BENCH_MANIFEST.json`` into a :class:`BenchmarkInfo`.

    Sigma comes from ``benchmark_constants``; the condition list (with stratum
    metadata) from ``snr_density_cells`` + ``curvature_sets``. Never hardcodes
    sigma. Falls back to filesystem discovery for conditions not in the manifest.
    """
    bench_root = Path(bench_root)
    manifest_path = bench_root / "BENCH_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    consts = manifest.get("benchmark_constants", {})
    sigma1 = consts.get("sigma1_px")
    sigma2 = consts.get("sigma2_px")
    if sigma1 is None or sigma2 is None:
        raise ValueError(
            "BENCH_MANIFEST.json is missing benchmark_constants.sigma1_px/sigma2_px; "
            "the match gate is derived from these and must not be hardcoded.")

    conditions: list[ConditionSpec] = []
    for cell in manifest.get("snr_density_cells", []):
        conditions.append(ConditionSpec("snr_density", cell["label"], dict(cell)))
    for s in manifest.get("curvature_sets", []):
        conditions.append(ConditionSpec("curvature", s["label"], dict(s)))

    return BenchmarkInfo(
        root=bench_root,
        sigma1=float(sigma1),
        sigma2=float(sigma2),
        conditions=conditions,
    )


# --------------------------------------------------------------------------- #
# Loaders (GT is per-image; predictions are one CSV per condition)             #
# --------------------------------------------------------------------------- #
def load_ground_truth(bench_root: str | Path, cond: ConditionSpec) -> pd.DataFrame:
    """Load and concatenate a condition's per-image ground-truth schema CSVs.

    Enumerated by the condition's ``meta.json`` (falls back to globbing
    ``ground_truth/*.csv``). The returned frame carries the canonical schema
    columns including ``image_id``, so matching can group by image.
    """
    cond_dir = Path(bench_root) / cond.family / cond.label
    meta_path = cond_dir / "meta.json"
    gt_files: list[Path] = []
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        for rec in meta.get("images", []):
            gtf = rec.get("ground_truth_file")
            if gtf:
                gt_files.append(cond_dir / gtf)
    if not gt_files:
        gt_files = sorted((cond_dir / "ground_truth").glob("*.csv"))

    frames = [pd.read_csv(f) for f in gt_files if f.exists()]
    if not frames:
        return _empty_schema_frame()
    df = pd.concat(frames, ignore_index=True)
    df["image_id"] = df["image_id"].astype(str)
    return df


def load_predictions(method_root: str | Path, cond: ConditionSpec) -> pd.DataFrame | None:
    """Load one condition's ``predictions.csv`` for a method.

    Returns ``None`` when the file is absent or unreadable (a failed/missing cell
    -- reported, never crashed). The frame is validated to carry the schema
    columns.
    """
    path = Path(method_root) / cond.family / cond.label / "predictions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    missing = [c for c in ("image_id", "x", "y", "logI1", "logI2") if c not in df.columns]
    if missing:
        return None
    df["image_id"] = df["image_id"].astype(str)
    return df


def _empty_schema_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SCHEMA_COLUMNS})


# --------------------------------------------------------------------------- #
# Alpha fit -- the ONE place the factor of 2 is applied at fit time            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlphaFit:
    """Result of the unweighted OLS alpha fit over pooled matched spots.

    ``alpha`` is the slope of ``log(A2/A1)`` vs ``log(sqrt(A1))``; ``alpha_se`` is
    the analytic OLS regression standard error of that slope; ``n`` is the number
    of matched spots the fit used.
    """

    alpha: float
    alpha_se: float
    intercept: float
    n: int


def fit_alpha(logI1: np.ndarray, logI2: np.ndarray) -> AlphaFit:
    """Unweighted OLS slope of ``log(A2/A1)`` on ``log(sqrt(A1))``.

    ``x = 0.5 * logI1`` (because ``log(sqrt(A1)) = 0.5 * log(A1)``; the halving IS
    the frozen factor of 2 from ``docs/alpha_convention.md`` -- regressing on
    ``logI1`` instead would halve the recovered slope, Gate C). ``y = logI2 -
    logI1``. Equal weight per spot ON PURPOSE: size-correlated weighting could
    manufacture curvature (CLAUDE.md rule 3). Standard error is the analytic OLS
    slope SE ``sqrt( (sum resid^2 / (n-2)) / sum (x - xbar)^2 )``.

    NaN/non-finite rows are dropped. Returns ``alpha = nan`` with ``n`` = usable
    rows when fewer than 3 finite points remain (SE undefined).
    """
    logI1 = np.asarray(logI1, dtype=np.float64).ravel()
    logI2 = np.asarray(logI2, dtype=np.float64).ravel()
    x = 0.5 * logI1                      # log(sqrt(A1)) -- the frozen factor of 2
    y = logI2 - logI1                    # log(A2/A1)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 3:
        return AlphaFit(alpha=math.nan, alpha_se=math.nan, intercept=math.nan, n=n)

    xbar = x.mean()
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx <= 0:
        # No spread in log(sqrt(A1)) -> slope unidentifiable (should not happen on
        # a curvature set, which retains a wide A1 spread by construction).
        return AlphaFit(alpha=math.nan, alpha_se=math.nan, intercept=math.nan, n=n)

    slope = float(np.sum((x - xbar) * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * xbar)
    resid = y - (intercept + slope * x)
    s2 = float(np.sum(resid ** 2) / (n - 2))
    alpha_se = float(math.sqrt(s2 / sxx))
    return AlphaFit(alpha=slope, alpha_se=alpha_se, intercept=intercept, n=n)


# --------------------------------------------------------------------------- #
# Detection metrics                                                            #
# --------------------------------------------------------------------------- #
def detection_metrics(n_gt: int, n_pred: int, n_matched: int) -> tuple[float, float, float]:
    """``(recall, precision, f1)`` from the three counts.

    ``recall = TP / n_gt``, ``precision = TP / n_pred`` (FPs are the unmatched
    predictions of THIS condition -- the condition is the stratum, so precision is
    defined whenever ``n_pred > 0``; the OLD repo's ``--`` bug came from trying to
    inherit an FP's stratum from a matched GT). ``recall`` is NaN only when the
    condition has no GT; ``precision`` NaN only when it has no predictions.
    """
    recall = (n_matched / n_gt) if n_gt > 0 else math.nan
    precision = (n_matched / n_pred) if n_pred > 0 else math.nan
    if math.isfinite(recall) and math.isfinite(precision) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = math.nan
    return recall, precision, f1


# --------------------------------------------------------------------------- #
# Per-condition evaluation (pure; the oracle test feeds GT as predictions)     #
# --------------------------------------------------------------------------- #
def evaluate_condition(
    gt: pd.DataFrame,
    pred: pd.DataFrame | None,
    cond: ConditionSpec,
    *,
    match_distance_px: float,
    n_images: int | None = None,
) -> dict:
    """Evaluate ONE condition; return a tidy metrics row.

    ``pred=None`` marks a missing/failed cell: detection is scored against the GT
    (recall defined, precision NaN) and ``status='missing'``. Otherwise spots are
    matched Hungarian within-image at ``match_distance_px``, then pooled per
    condition for the metrics. For a curvature set the alpha fit is added over the
    matched PREDICTION intensities.
    """
    row: dict = {
        "method": None,               # filled by the caller
        "family": cond.family,
        "condition": cond.label,
        "target_snr": cond.meta.get("target_snr"),
        "density": cond.meta.get("area_density_spots_per_px"),
        "true_alpha": cond.true_alpha,
        "null_control": bool(cond.meta.get("null_control", False)),
        "n_images": n_images if n_images is not None else _count_images(gt),
        "status": "ok",
    }

    n_gt = int(len(gt))
    if pred is None:
        recall, precision, f1 = detection_metrics(n_gt, 0, 0)
        row.update(
            status="missing", n_gt=n_gt, n_pred=0, n_matched=0, n_fn=n_gt, n_fp=0,
            recall=recall, precision=precision, f1=f1,
            logI1_bias=math.nan, logI2_bias=math.nan,
            logI1_rmse=math.nan, logI2_rmse=math.nan,
            log_ratio_bias=math.nan, log_ratio_std=math.nan, log_ratio_rmse=math.nan,
            match_dist_px_mean=math.nan,
        )
        if cond.is_curvature:
            row.update(alpha_hat=math.nan, alpha_se=math.nan, alpha_n=0,
                       alpha_bias=math.nan)
        return row

    n_pred = int(len(pred))
    if n_gt == 0 and n_pred == 0:
        dm = match_dataset(_empty_schema_frame(), _empty_schema_frame(),
                           max_distance=match_distance_px, method="hungarian")
    else:
        dm = match_dataset(gt, pred, max_distance=match_distance_px, method="hungarian")

    n_matched = dm.n_matched
    n_fn = len(dm.unmatched_gt)
    n_fp = len(dm.unmatched_pred)
    recall, precision, f1 = detection_metrics(n_gt, n_pred, n_matched)

    # Gate D: any cell WITH predictions must have a defined precision.
    if n_pred > 0:
        assert math.isfinite(precision), (
            f"precision undefined for {cond.key} despite {n_pred} predictions "
            "(the old-repo FP-binning bug)")

    # Per-spot residuals over the matched pairs (pred - true), pooled per condition.
    if n_matched > 0:
        g = dm.gt_matched
        p = dm.pred_matched
        gl1 = g["logI1"].to_numpy(dtype=float)
        gl2 = g["logI2"].to_numpy(dtype=float)
        pl1 = p["logI1"].to_numpy(dtype=float)
        pl2 = p["logI2"].to_numpy(dtype=float)
        d1 = pl1 - gl1
        d2 = pl2 - gl2
        dr = (pl2 - pl1) - (gl2 - gl1)
        logI1_bias = _nanmean(d1)
        logI2_bias = _nanmean(d2)
        logI1_rmse = _nanrmse(d1)
        logI2_rmse = _nanrmse(d2)
        log_ratio_bias = _nanmean(dr)
        log_ratio_std = _nanstd(dr)
        log_ratio_rmse = _nanrmse(dr)
        match_dist_mean = float(np.mean(dm.distances)) if dm.distances.size else math.nan
    else:
        logI1_bias = logI2_bias = logI1_rmse = logI2_rmse = math.nan
        log_ratio_bias = log_ratio_std = log_ratio_rmse = match_dist_mean = math.nan

    row.update(
        n_gt=n_gt, n_pred=n_pred, n_matched=n_matched, n_fn=n_fn, n_fp=n_fp,
        recall=recall, precision=precision, f1=f1,
        logI1_bias=logI1_bias, logI2_bias=logI2_bias,
        logI1_rmse=logI1_rmse, logI2_rmse=logI2_rmse,
        log_ratio_bias=log_ratio_bias, log_ratio_std=log_ratio_std,
        log_ratio_rmse=log_ratio_rmse, match_dist_px_mean=match_dist_mean,
    )

    if cond.is_curvature:
        if n_matched > 0:
            fit = fit_alpha(dm.pred_matched["logI1"].to_numpy(dtype=float),
                            dm.pred_matched["logI2"].to_numpy(dtype=float))
        else:
            fit = AlphaFit(math.nan, math.nan, math.nan, 0)
        true_alpha = cond.true_alpha
        alpha_bias = (fit.alpha - true_alpha) if (
            true_alpha is not None and math.isfinite(fit.alpha)) else math.nan
        row.update(alpha_hat=fit.alpha, alpha_se=fit.alpha_se, alpha_n=fit.n,
                   alpha_bias=alpha_bias)

    return row


def _count_images(df: pd.DataFrame) -> int:
    return int(df["image_id"].nunique()) if len(df) and "image_id" in df.columns else 0


def _nanmean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else math.nan


def _nanstd(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.std(ddof=0)) if a.size else math.nan


def _nanrmse(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(math.sqrt(np.mean(a ** 2))) if a.size else math.nan


# --------------------------------------------------------------------------- #
# Method- and run-level drivers                                                #
# --------------------------------------------------------------------------- #
def evaluate_method(
    info: BenchmarkInfo,
    method_root: str | Path,
    method_name: str,
    *,
    match_radius_sigma: float = DEFAULT_MATCH_RADIUS_SIGMA,
    gt_cache: dict[str, pd.DataFrame] | None = None,
    log_fn=print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one method folder over every benchmark condition.

    Returns ``(metrics_by_condition, alpha_recovery)`` frames. Same code path for
    every method (the fairness guarantee). ``gt_cache`` avoids re-loading GT when
    several methods are scored in one run.
    """
    match_distance = info.match_distance_px(match_radius_sigma)
    gt_cache = gt_cache if gt_cache is not None else {}
    rows: list[dict] = []

    for cond in info.conditions:
        gt = gt_cache.get(cond.key)
        if gt is None:
            gt = load_ground_truth(info.root, cond)
            gt_cache[cond.key] = gt
        pred = load_predictions(method_root, cond)
        n_images = int(cond.meta.get("n_images")) if cond.meta.get("n_images") is not None else None
        row = evaluate_condition(gt, pred, cond,
                                 match_distance_px=match_distance, n_images=n_images)
        row["method"] = method_name
        rows.append(row)

    metrics = pd.DataFrame(rows)
    alpha = _alpha_recovery_frame(metrics, info)
    n_missing = int((metrics["status"] == "missing").sum())
    log_fn(f"[eval] {method_name}: {len(metrics)} conditions "
           f"({n_missing} missing), match_gate={match_distance:.3f}px "
           f"(= {match_radius_sigma} x max(sigma)={info.sigma_ref})")
    return metrics, alpha


def _alpha_recovery_frame(metrics: pd.DataFrame, info: BenchmarkInfo) -> pd.DataFrame:
    """Extract the per-curvature-set alpha rows into their own tidy frame."""
    cur = metrics[metrics["family"] == "curvature"].copy()
    if cur.empty:
        return pd.DataFrame(columns=[
            "method", "condition", "true_alpha", "alpha_hat", "alpha_se",
            "alpha_bias", "alpha_n", "null_control"])
    spread = {c.label: c.meta.get("a1_spread_decades") for c in info.conditions
              if c.is_curvature}
    cur["a1_spread_decades"] = cur["condition"].map(spread)
    cols = ["method", "condition", "true_alpha", "alpha_hat", "alpha_se",
            "alpha_bias", "alpha_n", "null_control", "a1_spread_decades",
            "recall", "precision", "f1", "log_ratio_rmse", "status"]
    cur = cur[[c for c in cols if c in cur.columns]]
    return cur.sort_values("true_alpha").reset_index(drop=True)


def discover_methods(results_root: str | Path) -> list[str]:
    """Method-folder names under a results root (each mirrors the benchmark tree)."""
    results_root = Path(results_root)
    if not results_root.exists():
        return []
    return sorted(
        d.name for d in results_root.iterdir()
        if d.is_dir() and any((d / fam).exists() for fam in _FAMILIES))


def evaluate_all(
    bench_root: str | Path,
    results_root: str | Path,
    out_dir: str | Path,
    *,
    methods: list[str] | None = None,
    match_radius_sigma: float = DEFAULT_MATCH_RADIUS_SIGMA,
    log_fn=print,
) -> dict:
    """Evaluate every method under ``results_root`` and write the result CSVs.

    Writes ``<out>/<method>/metrics_by_condition.csv`` + ``alpha_recovery.csv``,
    the combined cross-method tables, and ``summary_by_method.csv``. Returns a
    dict of the written paths and the in-memory frames.
    """
    info = load_benchmark_info(bench_root)
    results_root = Path(results_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = discover_methods(results_root)
    if not methods:
        raise FileNotFoundError(
            f"no method folders under {results_root} (expected <root>/<method>/snr_density/...)")

    gt_cache: dict[str, pd.DataFrame] = {}
    all_metrics: list[pd.DataFrame] = []
    all_alpha: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for method in methods:
        metrics, alpha = evaluate_method(
            info, results_root / method, method,
            match_radius_sigma=match_radius_sigma, gt_cache=gt_cache, log_fn=log_fn)

        mdir = out_dir / method
        mdir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(mdir / "metrics_by_condition.csv", index=False)
        alpha.to_csv(mdir / "alpha_recovery.csv", index=False)

        all_metrics.append(metrics)
        all_alpha.append(alpha)
        summary_rows.append(_method_summary(method, metrics, alpha))

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_alpha = pd.concat(all_alpha, ignore_index=True) if all_alpha else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    combined_metrics.to_csv(out_dir / "combined_metrics_by_condition.csv", index=False)
    combined_alpha.to_csv(out_dir / "combined_alpha_recovery.csv", index=False)
    summary.to_csv(out_dir / "summary_by_method.csv", index=False)

    log_fn(f"[eval] wrote {len(methods)} method table(s) + combined + summary -> {out_dir}")
    return {
        "out_dir": out_dir,
        "methods": methods,
        "metrics": combined_metrics,
        "alpha": combined_alpha,
        "summary": summary,
        "match_distance_px": info.match_distance_px(match_radius_sigma),
    }


def _method_summary(method: str, metrics: pd.DataFrame, alpha: pd.DataFrame) -> dict:
    """Per-method headline aggregates for the cross-method table."""
    snr = metrics[metrics["family"] == "snr_density"]
    cur = metrics[metrics["family"] == "curvature"]
    null_row = alpha[alpha.get("null_control", False) == True] if not alpha.empty else alpha  # noqa: E712
    alpha_bias = alpha["alpha_bias"].to_numpy(dtype=float) if not alpha.empty else np.array([])
    alpha_bias = alpha_bias[np.isfinite(alpha_bias)]
    return {
        "method": method,
        "n_conditions": int(len(metrics)),
        "n_missing": int((metrics["status"] == "missing").sum()),
        "mean_f1": _col_nanmean(metrics, "f1"),
        "mean_recall": _col_nanmean(metrics, "recall"),
        "mean_precision": _col_nanmean(metrics, "precision"),
        "snr_mean_f1": _col_nanmean(snr, "f1"),
        "snr_mean_log_ratio_rmse": _col_nanmean(snr, "log_ratio_rmse"),
        "snr_mean_log_ratio_bias": _col_nanmean(snr, "log_ratio_bias"),
        "curvature_mean_f1": _col_nanmean(cur, "f1"),
        "alpha_mae": float(np.mean(np.abs(alpha_bias))) if alpha_bias.size else math.nan,
        "alpha_null_control_hat": (
            float(null_row["alpha_hat"].iloc[0]) if len(null_row) else math.nan),
        "alpha_null_control_se": (
            float(null_row["alpha_se"].iloc[0]) if len(null_row) else math.nan),
    }


def _col_nanmean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return math.nan
    a = df[col].to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else math.nan


# --------------------------------------------------------------------------- #
# Oracle helper (Gate A): score the GROUND TRUTH as if it were predictions     #
# --------------------------------------------------------------------------- #
def ground_truth_as_predictions(
    bench_root: str | Path,
    *,
    match_radius_sigma: float = DEFAULT_MATCH_RADIUS_SIGMA,
    method_name: str = "oracle_gt",
    log_fn=print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feed each condition's GT to the evaluator AS its own predictions.

    This is Gate A / the calibration of the instrument: a perfect method scores
    recall=precision=f1=1, zero intensity/ratio bias, and recovers each set's
    injected ``true_alpha`` (to within its OLS standard error -- GT carries real
    biological log-ratio scatter, so recovery is exact only in expectation). Uses
    the exact same ``evaluate_condition`` code path as any real method.
    """
    info = load_benchmark_info(bench_root)
    match_distance = info.match_distance_px(match_radius_sigma)
    rows: list[dict] = []
    for cond in info.conditions:
        gt = load_ground_truth(info.root, cond)
        # GT as predictions: same frame on both sides (blind -- no special path).
        row = evaluate_condition(gt, gt.copy(), cond, match_distance_px=match_distance,
                                 n_images=_count_images(gt))
        row["method"] = method_name
        rows.append(row)
    metrics = pd.DataFrame(rows)
    alpha = _alpha_recovery_frame(metrics, info)
    log_fn(f"[eval][oracle] scored GT-as-predictions over {len(metrics)} conditions")
    return metrics, alpha

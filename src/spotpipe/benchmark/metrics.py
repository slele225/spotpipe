"""Benchmark metrics: binned bias/variance, slope recovery, calibration (stage 4).

All metrics are computed on the canonical schema only and binned by the two
difficulty axes (SNR and local density; see :mod:`spotpipe.benchmark.features`),
both 2-D and marginally. The headline is **signed bias** -- the project's claim
is *unbiasedness* of per-spot intensities and the ``I2/I1`` ratio, especially in
the dim x high-overlap corner -- so residual means are reported, never collapsed
to RMSE alone.

What is computed (per method), binned by SNR, by density, and by SNR x density:

* **Detection** -- recall (GT spots matched / GT spots, binned by *true* SNR &
  density), precision (matched predictions / predictions, binned by the
  *prediction's* SNR & density), and their F1. Recall and precision bin on their
  own population's features (a false negative has only a GT bin; a false positive
  has only a prediction bin), which is the only consistent choice; this is
  documented and intended.
* **Per-channel intensity** -- signed bias ``mean(logI_pred - logI_true)`` and
  spread ``std`` (+ MAE/RMSE) on matched pairs, per channel.
* **Ratio** -- signed bias and spread of ``log_ratio_pred - log_ratio_true`` on
  matched pairs. This is the headline per-spot quantity.

Plus two non-binned products:

* **Slope recovery** -- per image, OLS slope of predicted ``log_ratio`` on
  predicted ``logI1``, which estimates the ratio-law slope ``beta`` DIRECTLY: the
  forward model has ``log A2 = (1+beta) log A1 + alpha + noise``, so ``log_ratio =
  log A2 - log A1`` regressed on ``log A1`` has slope ``beta`` (we deliberately do
  NOT regress ``logI2`` on ``logI1`` -- that slope would be ``1 + beta`` and is
  not what we compare to ``beta``). Compared against the image's true ``beta``; an
  unbiased, low-error method sits on the identity line, a noisy method shows
  regression-dilution attenuation toward 0. Fitting downstream of inference (never
  trained on) is exactly the CLAUDE.md design. Two variants are reported:

  - ``matched_only`` -- fit over matched GT/prediction pairs only; a clean
    diagnostic of intensity/ratio *estimation* quality (no detection confound).
  - ``end_to_end`` -- fit over ALL accepted predicted spots (including false
    positives); the pipeline-level metric. Per-image detection precision is
    reported alongside, since false positives are what can drag this slope.
* **Uncertainty calibration** (our model only) -- predicted ``uncertainty{1,2}``
  vs realized ``|residual|``: a per-sigma-bin curve (predicted sigma vs realized
  RMS error) and 1-sigma / 2-sigma coverage. Baselines emit no uncertainty and
  are skipped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from spotpipe.benchmark.matching import DatasetMatch

__all__ = [
    "compute_metrics",
    "fit_slope",
    "bin_edges_labels",
]


# --------------------------------------------------------------------------- #
# Binning helpers                                                              #
# --------------------------------------------------------------------------- #
def _assign_bins(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Map values to half-open bins ``[edges[i], edges[i+1])``; -1 if none/NaN."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, -1, dtype=int)
    finite = np.isfinite(values)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = finite & (values >= lo) & (values < hi)
        out[m] = i
    return out


def bin_edges_labels(edges: list[float]) -> list[dict]:
    """One descriptor per bin: index, lo, hi, a plotting center, and a label."""
    out = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if math.isinf(hi):
            center = lo * 1.5 if lo > 0 else lo + 1.0
            label = f">={_fmt(lo)}"
        else:
            center = 0.5 * (lo + hi)
            label = f"[{_fmt(lo)},{_fmt(hi)})"
        out.append({"index": i, "lo": lo, "hi": hi, "center": center, "label": label})
    return out


def _fmt(v: float) -> str:
    if math.isinf(v):
        return "inf"
    return f"{v:g}"


def _residual_stats(resid: np.ndarray) -> dict:
    """Signed bias (mean), spread (std), MAE and RMSE of a residual array."""
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    n = int(resid.size)
    if n == 0:
        return {"n": 0, "bias": math.nan, "std": math.nan, "mae": math.nan, "rmse": math.nan}
    return {
        "n": n,
        "bias": float(resid.mean()),
        "std": float(resid.std(ddof=1)) if n > 1 else 0.0,
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
    }


# --------------------------------------------------------------------------- #
# Slope recovery                                                              #
# --------------------------------------------------------------------------- #
def fit_slope(log_ratio: np.ndarray, logI1: np.ndarray) -> dict:
    """OLS slope/intercept of ``log_ratio`` on ``logI1`` (estimates ``beta``).

    Returns ``{slope, intercept, n}`` with ``slope = nan`` if fewer than two
    finite points or ``logI1`` has no spread (an unfittable image).
    """
    x = np.asarray(logI1, dtype=float)
    y = np.asarray(log_ratio, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 2 or np.ptp(x) < 1e-9:
        return {"slope": math.nan, "intercept": math.nan, "n": n}
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "n": n}


# --------------------------------------------------------------------------- #
# Core                                                                         #
# --------------------------------------------------------------------------- #
def compute_metrics(
    gt_feat: pd.DataFrame,
    pred_feat: pd.DataFrame,
    match: DatasetMatch,
    meta_by_image: dict[str, dict],
    *,
    snr_bins: list[float],
    density_bins: list[float],
    n_sigma_bins: int = 8,
) -> dict:
    """Compute all binned + slope + calibration metrics for one method.

    Parameters
    ----------
    gt_feat / pred_feat : the FULL ground-truth / prediction schema frames, each
        already augmented with ``snr`` / ``n_neighbors`` columns
        (:func:`features.attach_features`).
    match : the :class:`DatasetMatch` produced from those same two frames.
    meta_by_image : per-image metadata (carries the true ``beta`` for slope).
    snr_bins / density_bins : bin edges (last edge may be ``inf``).
    n_sigma_bins : number of quantile bins for the uncertainty calibration curve.
    """
    pairs = _build_pairs(match)
    gt_feat = _mark(gt_feat, match.gt_matched, "matched")
    pred_feat = _mark(pred_feat, match.pred_matched, "is_tp")

    result = {
        "detection_overall": _detection_overall(gt_feat, pred_feat),
        "binned": {
            "snr": _binned_axis(pairs, gt_feat, pred_feat, "snr", snr_bins),
            "density": _binned_axis(pairs, gt_feat, pred_feat, "n_neighbors", density_bins),
            "snr_x_density": _binned_2d(
                pairs, gt_feat, pred_feat, "snr", snr_bins, "n_neighbors", density_bins
            ),
            "overall": _binned_overall(pairs, gt_feat, pred_feat),
        },
        "slope": {
            "matched_only": _slope_recovery_matched(match, meta_by_image),
            "end_to_end": _slope_recovery_end_to_end(pred_feat, meta_by_image),
        },
        "calibration": _calibration(pairs, n_sigma_bins=n_sigma_bins),
    }
    return result


def _build_pairs(match: DatasetMatch) -> pd.DataFrame:
    """Aligned matched-pair table with true/pred values, residuals, features."""
    gt = match.gt_matched.reset_index(drop=True)
    pred = match.pred_matched.reset_index(drop=True)
    if len(gt) == 0:
        cols = [
            "image_id", "snr", "snr1", "snr2", "n_neighbors",
            "logI1_true", "logI1_pred", "resid_logI1",
            "logI2_true", "logI2_pred", "resid_logI2",
            "log_ratio_true", "log_ratio_pred", "resid_log_ratio",
            "uncertainty1", "uncertainty2", "distance",
        ]
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})

    pairs = pd.DataFrame({
        "image_id": gt["image_id"].to_numpy(),
        # difficulty axes come from the GROUND TRUTH spot (true difficulty)
        "snr": gt["snr"].to_numpy(float),
        "snr1": gt["snr1"].to_numpy(float),
        "snr2": gt["snr2"].to_numpy(float),
        "n_neighbors": gt["n_neighbors"].to_numpy(float),
        "logI1_true": gt["logI1"].to_numpy(float),
        "logI1_pred": pred["logI1"].to_numpy(float),
        "logI2_true": gt["logI2"].to_numpy(float),
        "logI2_pred": pred["logI2"].to_numpy(float),
        "log_ratio_true": gt["log_ratio"].to_numpy(float),
        "log_ratio_pred": pred["log_ratio"].to_numpy(float),
        "uncertainty1": pred["uncertainty1"].to_numpy(float),
        "uncertainty2": pred["uncertainty2"].to_numpy(float),
        "distance": match.distances,
    })
    pairs["resid_logI1"] = pairs["logI1_pred"] - pairs["logI1_true"]
    pairs["resid_logI2"] = pairs["logI2_pred"] - pairs["logI2_true"]
    pairs["resid_log_ratio"] = pairs["log_ratio_pred"] - pairs["log_ratio_true"]
    return pairs


def _mark(feat: pd.DataFrame, matched: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add a boolean ``col`` flagging rows present in ``matched`` (by id+spot_id)."""
    feat = feat.copy()
    if len(matched):
        keys = set(zip(matched["image_id"].astype(str), matched["spot_id"].astype(int)))
    else:
        keys = set()
    feat[col] = [
        (str(i), int(s)) in keys
        for i, s in zip(feat["image_id"], feat["spot_id"])
    ]
    return feat


# --- detection ------------------------------------------------------------- #
# Recall and precision bin on their OWN population's features: a false negative
# has only a GT bin (true SNR/density), a false positive has only a prediction
# bin (predicted SNR/density). So recall uses the GT-binned TP count and
# precision uses the prediction-binned TP count; conflating them would let
# precision exceed 1. Overall (unbinned) the two TP counts coincide.
def _detection_counts(n_gt, n_pred, n_tp_recall, n_tp_precision) -> dict:
    recall = n_tp_recall / n_gt if n_gt else math.nan
    precision = n_tp_precision / n_pred if n_pred else math.nan
    if n_gt and n_pred and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = math.nan
    return {
        "n_gt": int(n_gt), "n_pred": int(n_pred),
        "n_matched": int(n_tp_recall), "n_tp_pred": int(n_tp_precision),
        "recall": recall, "precision": precision, "f1": f1,
    }


def _detection_overall(gt_feat, pred_feat) -> dict:
    n_gt = len(gt_feat)
    n_pred = len(pred_feat)
    n_tp_gt = int(gt_feat["matched"].sum()) if n_gt else 0
    n_tp_pred = int(pred_feat["is_tp"].sum()) if n_pred else 0
    return _detection_counts(n_gt, n_pred, n_tp_gt, n_tp_pred)


def _detection_in_mask(gt_mask, pred_mask, gt_feat, pred_feat) -> dict:
    n_gt = int(gt_mask.sum())
    n_pred = int(pred_mask.sum())
    n_tp_gt = int((gt_mask & gt_feat["matched"].to_numpy()).sum())
    n_tp_pred = int((pred_mask & pred_feat["is_tp"].to_numpy()).sum())
    return _detection_counts(n_gt, n_pred, n_tp_gt, n_tp_pred)


# --- intensity / ratio residuals ------------------------------------------- #
def _residual_block(pairs_sub: pd.DataFrame) -> dict:
    return {
        "logI1": _residual_stats(pairs_sub["resid_logI1"].to_numpy()),
        "logI2": _residual_stats(pairs_sub["resid_logI2"].to_numpy()),
        "log_ratio": _residual_stats(pairs_sub["resid_log_ratio"].to_numpy()),
    }


def _bin_row(descriptor, detection, residuals) -> dict:
    row = dict(descriptor)
    row["detection"] = detection
    row["intensity"] = residuals
    return row


def _binned_axis(pairs, gt_feat, pred_feat, col, edges) -> list[dict]:
    descriptors = bin_edges_labels(edges)
    gt_bin = _assign_bins(gt_feat[col].to_numpy(), edges)
    pred_bin = _assign_bins(pred_feat[col].to_numpy(), edges)
    pair_bin = _assign_bins(pairs[col].to_numpy(), edges) if len(pairs) else np.empty(0, int)

    rows = []
    for d in descriptors:
        i = d["index"]
        det = _detection_in_mask(gt_bin == i, pred_bin == i, gt_feat, pred_feat)
        sub = pairs[pair_bin == i] if len(pairs) else pairs
        rows.append(_bin_row(d, det, _residual_block(sub)))
    return rows


def _binned_2d(pairs, gt_feat, pred_feat, col_a, edges_a, col_b, edges_b) -> list[dict]:
    da, db = bin_edges_labels(edges_a), bin_edges_labels(edges_b)
    gt_a = _assign_bins(gt_feat[col_a].to_numpy(), edges_a)
    gt_b = _assign_bins(gt_feat[col_b].to_numpy(), edges_b)
    pred_a = _assign_bins(pred_feat[col_a].to_numpy(), edges_a)
    pred_b = _assign_bins(pred_feat[col_b].to_numpy(), edges_b)
    if len(pairs):
        pair_a = _assign_bins(pairs[col_a].to_numpy(), edges_a)
        pair_b = _assign_bins(pairs[col_b].to_numpy(), edges_b)

    rows = []
    for a in da:
        for b in db:
            gt_mask = (gt_a == a["index"]) & (gt_b == b["index"])
            pred_mask = (pred_a == a["index"]) & (pred_b == b["index"])
            det = _detection_in_mask(gt_mask, pred_mask, gt_feat, pred_feat)
            if len(pairs):
                sub = pairs[(pair_a == a["index"]) & (pair_b == b["index"])]
            else:
                sub = pairs
            rows.append({
                "snr_bin": a["label"], "density_bin": b["label"],
                "snr_index": a["index"], "density_index": b["index"],
                "snr_center": a["center"], "density_center": b["center"],
                "detection": det, "intensity": _residual_block(sub),
            })
    return rows


def _binned_overall(pairs, gt_feat, pred_feat) -> dict:
    det = _detection_overall(gt_feat, pred_feat)
    return {"detection": det, "intensity": _residual_block(pairs)}


# --- slope recovery -------------------------------------------------------- #
def _true_beta(meta_by_image, image_id) -> float:
    tb = meta_by_image.get(str(image_id), {}).get("scene", {}).get("beta", math.nan)
    return float(tb) if tb is not None else math.nan


def _slope_recovery_matched(match: DatasetMatch, meta_by_image) -> list[dict]:
    """`matched_only` variant: recovered beta from MATCHED pairs only.

    Diagnostic of intensity/ratio estimation quality with no detection confound
    (every spot used is a true spot). Fits predicted ``log_ratio`` on predicted
    ``logI1`` (slope == beta) per image.
    """
    gt = match.gt_matched.reset_index(drop=True)
    pred = match.pred_matched.reset_index(drop=True)
    rows = []
    if len(gt) == 0:
        return rows
    pred = pred.copy()
    pred["__image_id"] = gt["image_id"].to_numpy()  # group by GT image id
    for image_id, sub in pred.groupby("__image_id"):
        fit = fit_slope(sub["log_ratio"].to_numpy(), sub["logI1"].to_numpy())
        rows.append({
            "image_id": str(image_id),
            "true_beta": _true_beta(meta_by_image, image_id),
            "beta_hat": fit["slope"],
            "intercept_hat": fit["intercept"],
            "n_spots": fit["n"],
        })
    return rows


def _slope_recovery_end_to_end(pred_feat: pd.DataFrame, meta_by_image) -> list[dict]:
    """`end_to_end` variant: recovered beta from ALL accepted predictions.

    The pipeline-level metric -- it includes false positives, so per-image
    detection ``precision`` (matched preds / preds) is logged next to the slope
    (requirement #4). Fits predicted ``log_ratio`` on predicted ``logI1`` per
    image. ``pred_feat`` must already carry the ``is_tp`` flag.
    """
    rows = []
    if len(pred_feat) == 0:
        return rows
    has_tp = "is_tp" in pred_feat.columns
    for image_id, sub in pred_feat.groupby("image_id"):
        fit = fit_slope(sub["log_ratio"].to_numpy(), sub["logI1"].to_numpy())
        n_pred = int(len(sub))
        n_tp = int(sub["is_tp"].sum()) if has_tp else 0
        rows.append({
            "image_id": str(image_id),
            "true_beta": _true_beta(meta_by_image, image_id),
            "beta_hat": fit["slope"],
            "intercept_hat": fit["intercept"],
            "n_spots": fit["n"],
            "n_pred": n_pred,
            "n_tp": n_tp,
            "precision": (n_tp / n_pred) if n_pred else math.nan,
        })
    return rows


# --- uncertainty calibration ----------------------------------------------- #
def _calibration(pairs: pd.DataFrame, *, n_sigma_bins: int) -> dict | None:
    """Predicted-sigma vs realized-error calibration; None if no uncertainties."""
    if len(pairs) == 0:
        return None
    have_unc = np.isfinite(pairs["uncertainty1"].to_numpy()).any() or \
        np.isfinite(pairs["uncertainty2"].to_numpy()).any()
    if not have_unc:
        return None

    out = {"channels": {}}
    for ch, sig_col, res_col in (("1", "uncertainty1", "resid_logI1"),
                                 ("2", "uncertainty2", "resid_logI2")):
        sigma = pairs[sig_col].to_numpy(float)
        resid = pairs[res_col].to_numpy(float)
        ok = np.isfinite(sigma) & np.isfinite(resid) & (sigma > 0)
        sigma, resid = sigma[ok], resid[ok]
        if sigma.size == 0:
            out["channels"][ch] = None
            continue
        abs_r = np.abs(resid)
        cov1 = float(np.mean(abs_r <= sigma))
        cov2 = float(np.mean(abs_r <= 2.0 * sigma))
        curve = _calibration_curve(sigma, resid, n_sigma_bins)
        out["channels"][ch] = {
            "n": int(sigma.size),
            "coverage_1sigma": cov1,   # well-calibrated -> ~0.683
            "coverage_2sigma": cov2,   # well-calibrated -> ~0.954
            "curve": curve,
        }
    return out


def _calibration_curve(sigma: np.ndarray, resid: np.ndarray, n_bins: int) -> list[dict]:
    """Quantile-bin by predicted sigma; per bin: mean predicted vs realized RMS."""
    n = sigma.size
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(sigma)
    sigma, resid = sigma[order], resid[order]
    splits = np.array_split(np.arange(n), n_bins)
    curve = []
    for sl in splits:
        if sl.size == 0:
            continue
        s = sigma[sl]
        r = resid[sl]
        curve.append({
            "n": int(sl.size),
            "pred_sigma": float(s.mean()),
            "realized_rms": float(np.sqrt(np.mean(r ** 2))),
        })
    return curve

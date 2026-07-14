"""Ground-truth <-> prediction matching for the shared evaluator.

Ported UNCHANGED from the old repo (``spotpipe.benchmark.matching``). Pairs
predicted spots to ground-truth spots **by location**, one-to-one, within a
maximum-distance gate (a radius in pixels). A matched pair contributes to
true-positive detection and to the per-spot intensity/ratio residuals; an
unmatched GT spot is a false negative; an unmatched prediction is a false
positive.

The module is deliberately **method-agnostic**: it reads only the canonical
:mod:`spotpipe.schema` columns (``image_id``, ``x``, ``y``), so our model and any
external baseline are matched identically -- the fairness guarantee of the ONE
shared blind evaluator (CLAUDE.md). Two assignment strategies:

* ``greedy`` (default): sort all within-gate candidate pairs by distance and
  assign nearest-first, skipping any spot already taken. Deterministic, no
  third-party dependency.
* ``hungarian``: globally optimal minimum-total-distance assignment via
  :func:`scipy.optimize.linear_sum_assignment`, then drop any matched pair that
  exceeds the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "MatchResult",
    "DatasetMatch",
    "match_xy",
    "match_spots",
    "match_dataset",
]


@dataclass
class MatchResult:
    """One image's matching, as positional indices into the input frames.

    ``matches`` is an ``(M, 2)`` int array of ``(gt_row, pred_row)`` index pairs
    (positions in the per-image GT / prediction frames passed in). ``distances``
    is the matched-pair separation in pixels, aligned with ``matches``.
    ``unmatched_gt`` / ``unmatched_pred`` are the leftover row indices (false
    negatives / false positives).
    """

    matches: np.ndarray              # (M, 2) int: [gt_idx, pred_idx]
    distances: np.ndarray            # (M,) float
    unmatched_gt: np.ndarray         # (F_n,) int
    unmatched_pred: np.ndarray       # (F_p,) int

    @property
    def n_matched(self) -> int:
        return int(self.matches.shape[0])


@dataclass
class DatasetMatch:
    """Matching aggregated over a whole dataset (many images).

    The aligned frames make the downstream metrics trivial: ``gt_matched`` and
    ``pred_matched`` have the SAME number of rows in the SAME order, so a matched
    pair is row ``i`` of each. All four frames carry their original schema columns
    (including ``image_id``), so metrics can re-group by image.
    """

    gt_matched: pd.DataFrame         # GT rows that matched, in pair order
    pred_matched: pd.DataFrame       # prediction rows, aligned row-for-row
    unmatched_gt: pd.DataFrame       # false negatives
    unmatched_pred: pd.DataFrame     # false positives
    distances: np.ndarray            # (M,) px, aligned with the matched frames

    @property
    def n_gt(self) -> int:
        return len(self.gt_matched) + len(self.unmatched_gt)

    @property
    def n_pred(self) -> int:
        return len(self.pred_matched) + len(self.unmatched_pred)

    @property
    def n_matched(self) -> int:
        return len(self.gt_matched)


def match_xy(
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    *,
    max_distance: float,
    method: str = "greedy",
) -> MatchResult:
    """One-to-one match two sets of points by Euclidean distance within a gate.

    Parameters
    ----------
    gt_xy : ``(N, 2)`` array of ground-truth ``(x, y)`` coordinates.
    pred_xy : ``(P, 2)`` array of predicted ``(x, y)`` coordinates.
    max_distance : gate radius in pixels; no pair farther than this is matched.
    method : ``"greedy"`` (nearest-first) or ``"hungarian"`` (globally optimal).
    """
    gt_xy = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    pred_xy = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
    n, p = len(gt_xy), len(pred_xy)

    if n == 0 or p == 0:
        return MatchResult(
            matches=np.empty((0, 2), dtype=int),
            distances=np.empty(0, dtype=float),
            unmatched_gt=np.arange(n, dtype=int),
            unmatched_pred=np.arange(p, dtype=int),
        )

    # Full pairwise distance matrix (N, P). Eval images are small; this is fine.
    dmat = np.sqrt(
        ((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2)
    )

    if method == "hungarian":
        gt_idx, pred_idx, dists = _assign_hungarian(dmat, max_distance)
    elif method == "greedy":
        gt_idx, pred_idx, dists = _assign_greedy(dmat, max_distance)
    else:
        raise ValueError(f"unknown match method {method!r} (use 'greedy' or 'hungarian')")

    matched_gt = set(gt_idx.tolist())
    matched_pred = set(pred_idx.tolist())
    return MatchResult(
        matches=np.stack([gt_idx, pred_idx], axis=1) if len(gt_idx) else np.empty((0, 2), dtype=int),
        distances=dists,
        unmatched_gt=np.array([i for i in range(n) if i not in matched_gt], dtype=int),
        unmatched_pred=np.array([j for j in range(p) if j not in matched_pred], dtype=int),
    )


def _assign_greedy(dmat: np.ndarray, max_distance: float):
    """Nearest-first greedy one-to-one assignment within the gate."""
    n, p = dmat.shape
    # Candidate pairs within the gate, sorted by ascending distance.
    gi, pj = np.where(dmat <= max_distance)
    if gi.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
    order = np.argsort(dmat[gi, pj], kind="stable")
    gi, pj = gi[order], pj[order]

    gt_taken = np.zeros(n, dtype=bool)
    pred_taken = np.zeros(p, dtype=bool)
    gt_idx, pred_idx, dists = [], [], []
    for g, j in zip(gi, pj):
        if gt_taken[g] or pred_taken[j]:
            continue
        gt_taken[g] = pred_taken[j] = True
        gt_idx.append(int(g))
        pred_idx.append(int(j))
        dists.append(float(dmat[g, j]))
    return np.array(gt_idx, dtype=int), np.array(pred_idx, dtype=int), np.array(dists, dtype=float)


def _assign_hungarian(dmat: np.ndarray, max_distance: float):
    """Globally optimal assignment, then drop out-of-gate pairs."""
    from scipy.optimize import linear_sum_assignment

    # Pad costs above the gate so the optimiser never prefers an illegal pair;
    # we filter them out after solving.
    big = float(dmat.max()) * 10.0 + max_distance * 10.0 + 1.0
    cost = np.where(dmat <= max_distance, dmat, big)
    rows, cols = linear_sum_assignment(cost)
    keep = dmat[rows, cols] <= max_distance
    rows, cols = rows[keep], cols[keep]
    return rows.astype(int), cols.astype(int), dmat[rows, cols].astype(float)


def match_spots(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    max_distance: float,
    method: str = "greedy",
) -> MatchResult:
    """Match one image's GT and prediction frames by their ``x`` / ``y`` columns.

    Returns positional indices into ``gt`` / ``pred`` (their ``.iloc`` positions),
    so the caller maps them back to whatever rows it passed in.
    """
    gt_xy = gt[["x", "y"]].to_numpy(dtype=float) if len(gt) else np.empty((0, 2))
    pred_xy = pred[["x", "y"]].to_numpy(dtype=float) if len(pred) else np.empty((0, 2))
    return match_xy(gt_xy, pred_xy, max_distance=max_distance, method=method)


def match_dataset(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    max_distance: float,
    method: str = "greedy",
) -> DatasetMatch:
    """Match a whole dataset, image by image, into aligned matched frames.

    Spots are paired only WITHIN the same ``image_id``. The returned
    ``gt_matched`` / ``pred_matched`` frames are aligned row-for-row (pair ``i``
    is row ``i`` of each), ready for residual metrics. Images present in only one
    of the two tables contribute purely false negatives or false positives.
    """
    if "image_id" not in gt.columns or "image_id" not in pred.columns:
        raise ValueError("both frames must carry an 'image_id' column")

    image_ids = list(dict.fromkeys([*gt["image_id"].tolist(), *pred["image_id"].tolist()]))

    gt_m_parts, pred_m_parts, fn_parts, fp_parts = [], [], [], []
    dist_parts = []
    for image_id in image_ids:
        g = gt[gt["image_id"] == image_id]
        p = pred[pred["image_id"] == image_id]
        res = match_spots(g, p, max_distance=max_distance, method=method)

        if res.n_matched:
            gt_m_parts.append(g.iloc[res.matches[:, 0]])
            pred_m_parts.append(p.iloc[res.matches[:, 1]])
            dist_parts.append(res.distances)
        if res.unmatched_gt.size:
            fn_parts.append(g.iloc[res.unmatched_gt])
        if res.unmatched_pred.size:
            fp_parts.append(p.iloc[res.unmatched_pred])

    def _concat(parts: list[pd.DataFrame], template: pd.DataFrame) -> pd.DataFrame:
        if parts:
            return pd.concat(parts, ignore_index=True)
        return template.iloc[0:0].copy()

    return DatasetMatch(
        gt_matched=_concat(gt_m_parts, gt),
        pred_matched=_concat(pred_m_parts, pred),
        unmatched_gt=_concat(fn_parts, gt),
        unmatched_pred=_concat(fp_parts, pred),
        distances=np.concatenate(dist_parts) if dist_parts else np.empty(0, dtype=float),
    )

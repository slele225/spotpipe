"""Model-inference adapter -- run vendored HRNet checkpoints over the on-disk
benchmark and emit schema-conforming prediction CSVs (fresh, disposable tier).

This is the TEMPLATE every baseline runner copies, so it is built clean and
crash-resumable. It modifies NOTHING vendored: the model, its heads, the
simulator, losses and schema all stay frozen. The detection/read convention is
exactly the vendored :func:`spotpipe.models.predict_spots` path (threshold ->
local-max NMS -> optional top-k cap -> sub-pixel centre via the offset head ->
read ``logI1``/``logI2``/``logvar`` at the peak). We batch the FORWARD pass for
GPU efficiency and decode each image with :func:`_decode_image`, which is a
faithful transcription of the vendored per-image decode; ``test_infer`` asserts
``_decode_image`` reproduces ``predict_spots`` bit-for-bit so the two can never
drift.

Provenance honesty (REQUIRED), DERIVED not assumed. Every ``RUN_MANIFEST.json``
records the checkpoint, its recorded training SHA and a provenance status that is
computed from that SHA by :func:`is_legacy_checkpoint`:

* LEGACY/REFERENCE -- trained on the OLD dirty-tree simulator (``93fc0aa8...``),
  e.g. ``hrnet_large`` / ``hrnet_small``. Outputs go to ``our_model_<name>_legacy``.
  A pipeline-derisking reference run, NOT the reproducible headline.
* CLEAN RETRAIN/HEADLINE -- trained on the clean vendored simulator, e.g.
  ``hrnet_large_measured`` (git ``26b0d48``). Outputs go to ``our_model_<name>``
  (no ``_legacy`` suffix -- calling it legacy would be a lie).

Output layout (mirrors the benchmark; one CSV per condition per method)::

    <results_root>/<method_name>/
      snr_density/snr={S}_density={D}/predictions.csv
      curvature/alpha={A}/predictions.csv
      RUN_MANIFEST.json

Overnight-survival properties (this is the template):

* Resumable: a condition whose ``predictions.csv`` already exists AND validates
  against the frozen schema is SKIPPED, so crash-and-restart never redoes work.
* Incremental: each condition's CSV and the running manifest are written as the
  condition completes -- nothing is held in memory until the end.
* All cores, one worker pool: every image of every to-run condition streams
  through a SINGLE ``DataLoader`` (``num_workers`` = core count, ``pin_memory``)
  built once per checkpoint, so TIFF decode is parallel and prefetched ahead of
  GPU compute AND the worker processes are spawned once -- not respawned per
  condition (Windows spawn made that per-condition respawn the dominant cost).
  Batches are condition-homogeneous so each still writes to exactly one CSV; the
  forward pass is batched, never a single-image Python loop.
* GPU check (fail loud): the device is printed at startup; a requested GPU that
  is absent aborts, and a silent CPU fallback prints a clear warning (the
  50-hour bug was compute starvation).
* Location-agnostic: every path resolves through :mod:`spotpipe.paths`; no
  absolute paths, so it runs identically on Windows local and a Linux GPU box.
* Timing: per-condition and total wall time plus a dataload-vs-compute split are
  printed and stored, so a starved GPU is visible.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from spotpipe.models import build_spot_model, predict_spots  # noqa: F401 (predict_spots used in tests)
from spotpipe.models.spot_model import SpotModel, normalize_counts
from spotpipe.schema import SCHEMA_COLUMNS, SpotRecord, records_to_dataframe, write_spots

__all__ = [
    "InferenceParams",
    "CheckpointBundle",
    "load_checkpoint",
    "discover_checkpoints",
    "discover_conditions",
    "run_inference",
    "method_name",
    "is_legacy_checkpoint",
]

_FAMILIES = ("snr_density", "curvature")


# --------------------------------------------------------------------------- #
# Inference parameters (sourced from the checkpoint, with vendored defaults)   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InferenceParams:
    """Decode knobs forwarded to the vendored convention.

    Defaults match :func:`spotpipe.models.predict_spots`. When a checkpoint
    records a ``benchmark.our_model`` block (both carried checkpoints do), those
    values win, so we detect with the SAME thresholds the checkpoint was
    benchmarked at rather than re-guessing them here.
    """

    adc_max: float = 4095.0
    peak_threshold: float = 0.3
    nms_kernel: int = 3
    max_spots: int | None = 2000
    logvar_min: float = -10.0
    logvar_max: float = 6.0

    @classmethod
    def from_checkpoint_config(cls, config: dict) -> "InferenceParams":
        bench = (config or {}).get("benchmark", {}) or {}
        om = bench.get("our_model", {}) or {}
        det = (config or {}).get("detector", {}) or {}
        adc = det.get("adc_max")
        return cls(
            adc_max=float(adc) if adc is not None else cls.adc_max,
            peak_threshold=float(om.get("peak_threshold", cls.peak_threshold)),
            nms_kernel=int(om.get("nms_kernel", cls.nms_kernel)),
            max_spots=(int(om["max_spots"]) if om.get("max_spots") is not None else cls.max_spots),
            logvar_min=float(om.get("logvar_min", cls.logvar_min)),
            logvar_max=float(om.get("logvar_max", cls.logvar_max)),
        )


@dataclass
class CheckpointBundle:
    """A loaded checkpoint plus the provenance we must report honestly."""

    name: str
    model: SpotModel
    params: InferenceParams
    training_git_sha: str
    checkpoint_rel: str          # path relative to repo root, for the manifest


def load_checkpoint(name: str, *, checkpoints_root: Path, repo_root: Path) -> CheckpointBundle:
    """Load a carried checkpoint into a ready-to-eval :class:`CheckpointBundle`.

    Reads ``<checkpoints_root>/<name>/best_checkpoint.pt`` (``{model_state,
    config}``), rebuilds the frozen model from its recorded ``model`` block, and
    pulls the training SHA from the sibling ``manifest.json``.
    """
    ckpt_dir = checkpoints_root / name
    ckpt_path = ckpt_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(blob, dict) and "model_state" in blob and "config" in blob):
        raise ValueError(f"unexpected checkpoint format in {ckpt_path}: keys={list(blob)[:8]}")
    config = blob["config"]
    model = build_spot_model(config.get("model", {}))
    missing, unexpected = model.load_state_dict(blob["model_state"], strict=False)
    if missing or unexpected:
        raise ValueError(
            f"state_dict mismatch loading {name}: missing={list(missing)[:6]} "
            f"unexpected={list(unexpected)[:6]}")
    model.eval()

    params = InferenceParams.from_checkpoint_config(config)

    training_git_sha = "unknown"
    manifest_path = ckpt_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            training_git_sha = json.load(fh).get("git_commit", "unknown")

    return CheckpointBundle(
        name=name,
        model=model,
        params=params,
        training_git_sha=training_git_sha,
        checkpoint_rel=str(ckpt_path.relative_to(repo_root)),
    )


def discover_checkpoints(checkpoints_root: Path) -> list[str]:
    """Names of carried checkpoints (subdirs holding ``best_checkpoint.pt``)."""
    if not checkpoints_root.exists():
        return []
    return sorted(d.name for d in checkpoints_root.iterdir()
                  if (d / "best_checkpoint.pt").exists())


# Training SHAs of the OLD dirty-tree simulator. A checkpoint whose recorded
# training commit is one of these -- or whose provenance is unknown -- is LEGACY
# and its outputs are labelled/flagged as such. Anything else is a clean-tree
# retrain (e.g. ``hrnet_large_measured``, git 26b0d48) and is NOT legacy: it is a
# reproducible headline model. Provenance is DERIVED here, never assumed.
_LEGACY_TRAINING_SHA_PREFIXES: tuple[str, ...] = ("93fc0aa8",)

_LEGACY_NOTE = (
    "Trained on the OLD dirty-tree simulator (not the clean vendored one). "
    "This is a pipeline-derisking reference run, NOT the reproducible headline "
    "-- the headline comes from a clean retrain.")
_CLEAN_NOTE = (
    "Trained on the clean vendored simulator (retrain). Reproducible headline "
    "model -- NOT legacy.")


def is_legacy_checkpoint(training_git_sha: str) -> bool:
    """True iff this checkpoint was trained on the OLD dirty-tree simulator."""
    sha = (training_git_sha or "unknown").strip().lower()
    if sha in ("", "unknown"):
        return True                      # unprovenanced -> assume legacy (fail safe)
    return sha.startswith(_LEGACY_TRAINING_SHA_PREFIXES)


def method_name(checkpoint_name: str, training_git_sha: str = "unknown") -> str:
    """Labelled method folder for a checkpoint.

    Legacy (dirty-tree-trained) checkpoints keep the ``_legacy`` suffix, per the
    provenance rule. Clean retrains get a plain ``our_model_<name>`` folder --
    calling them legacy would be a lie.
    """
    suffix = "_legacy" if is_legacy_checkpoint(training_git_sha) else ""
    return f"our_model_{checkpoint_name}{suffix}"


# --------------------------------------------------------------------------- #
# Benchmark condition discovery + per-condition image list                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """One benchmark condition == one directory == one output CSV."""

    family: str          # "snr_density" | "curvature"
    label: str           # e.g. "snr=5_density=0.006" | "alpha=0.3"
    directory: Path

    @property
    def key(self) -> str:
        return f"{self.family}/{self.label}"


def discover_conditions(bench_root: Path) -> list[Condition]:
    """All benchmark conditions under ``bench_root`` (both families), sorted."""
    conds: list[Condition] = []
    for family in _FAMILIES:
        fam = bench_root / family
        if not fam.exists():
            continue
        for d in sorted(fam.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                conds.append(Condition(family=family, label=d.name, directory=d))
    return conds


def _condition_images(cond: Condition, *, limit: int | None = None) -> list[tuple[str, Path]]:
    """(image_id, tif_path) for a condition, from its ``meta.json`` image list.

    Falls back to globbing ``images/*.tif`` if the meta list is absent. Applies
    an optional ``limit`` (used by the smoke subset).
    """
    meta_path = cond.directory / "meta.json"
    pairs: list[tuple[str, Path]] = []
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        for rec in meta.get("images", []):
            img_file = rec.get("image_file")
            if img_file:
                pairs.append((str(rec["image_id"]), cond.directory / img_file))
    if not pairs:
        for tif in sorted((cond.directory / "images").glob("*.tif")):
            image_id = tif.stem[len("image_"):] if tif.stem.startswith("image_") else tif.stem
            pairs.append((image_id, tif))
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


# --------------------------------------------------------------------------- #
# Dataset: parallel TIFF decode across DataLoader workers                       #
# --------------------------------------------------------------------------- #
class _BenchmarkImageDataset(Dataset):
    """Yields ``(cond_index, image_id, raw_counts[2,H,W] float32)`` across ALL
    to-run conditions.

    Built ONCE over every ``(condition, image)`` pair that still needs running,
    so a SINGLE DataLoader / worker pool serves the whole run instead of one
    respawn per condition. Windows spawns fresh worker processes, and with ~43
    conditions per checkpoint (and this adapter re-run many times overnight) a
    per-condition DataLoader paid that spawn tax dozens of times per run. The
    condition index rides along as per-item metadata so each decoded record
    routes back to the right per-condition CSV. Skipped conditions contribute NO
    pairs here, so their images are never loaded (skip-if-exists preserved).

    Kept dead simple and picklable (module-level) so DataLoader workers can
    decode TIFFs in parallel on every platform. Raw 12-bit counts are returned;
    the fixed ADC normalisation happens once on the compute device, exactly as
    the vendored path does.
    """

    def __init__(self, items: list[tuple[int, str, Path]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        cond_index, image_id, path = self._items[idx]
        arr = np.asarray(tifffile.imread(path))  # [2,H,W] uint16
        if arr.ndim != 3 or arr.shape[0] != 2:
            raise ValueError(f"expected [2,H,W] image, got {arr.shape} at {path}")
        return cond_index, image_id, torch.from_numpy(arr.astype(np.float32))


def _collate(batch):
    cond_indices = [b[0] for b in batch]
    ids = [b[1] for b in batch]
    imgs = torch.stack([b[2] for b in batch], dim=0)  # [B,2,H,W]
    return cond_indices, ids, imgs


# --------------------------------------------------------------------------- #
# Decode: faithful transcription of the vendored per-image convention          #
# --------------------------------------------------------------------------- #
def _decode_image(
    preds: dict[str, torch.Tensor],
    idx: int,
    image_id: str,
    params: InferenceParams,
) -> list[SpotRecord]:
    """Decode one image's batched head outputs into canonical schema records.

    This mirrors :func:`spotpipe.models.predict_spots` EXACTLY -- sigmoid ->
    local-max NMS at ``nms_kernel`` -> ``peak_threshold`` gate -> optional
    top-k ``max_spots`` cap -> sub-pixel centre from the offset head -> nearest
    read of ``logI1``/``logI2``/``logvar`` at the peak -> derived, clamp-bounded
    uncertainty. ``test_infer`` asserts this reproduces the vendored output, so
    batching the forward pass introduces no convention drift.
    """
    import math

    heat = torch.sigmoid(preds["heatmap"])[idx, 0]  # [H, W]
    pad = params.nms_kernel // 2
    pooled = F.max_pool2d(heat[None, None], params.nms_kernel, stride=1, padding=pad)[0, 0]
    keep = (heat >= pooled) & (heat > params.peak_threshold)
    ys, xs = torch.where(keep)
    scores = heat[ys, xs]

    if params.max_spots is not None and scores.numel() > params.max_spots:
        top = torch.topk(scores, int(params.max_spots))
        sel = top.indices
        ys, xs, scores = ys[sel], xs[sel], scores[sel]

    offset = preds["offset"][idx]
    logI1 = preds["logI1"][idx, 0]
    logI2 = preds["logI2"][idx, 0]
    logvar1 = preds["logvar1"][idx, 0]
    logvar2 = preds["logvar2"][idx, 0]

    def _uncertainty(raw_logvar: float) -> float:
        clamped = min(max(raw_logvar, params.logvar_min), params.logvar_max)
        return math.exp(0.5 * clamped)

    records: list[SpotRecord] = []
    for k in range(scores.numel()):
        r = int(ys[k])
        c = int(xs[k])
        dx = float(offset[0, r, c])
        dy = float(offset[1, r, c])
        records.append(
            SpotRecord.from_logs(
                image_id=image_id,
                spot_id=k,
                x=float(c) + dx,
                y=float(r) + dy,
                p_detect=float(scores[k]),
                logI1=float(logI1[r, c]),
                logI2=float(logI2[r, c]),
                sigma1_hat=math.nan,
                sigma2_hat=math.nan,
                uncertainty1=_uncertainty(float(logvar1[r, c])),
                uncertainty2=_uncertainty(float(logvar2[r, c])),
                flags="",
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Per-condition run                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_predictions_csv(path: Path) -> bool:
    """True if ``path`` is a readable CSV with EXACTLY the frozen schema columns."""
    if not path.exists():
        return False
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    return tuple(header.columns) == SCHEMA_COLUMNS


def _write_condition_csv(
    records: list[SpotRecord],
    out_csv: Path,
    *,
    n_images: int,
    dataload_s: float,
    compute_s: float,
) -> dict:
    """Write one condition's ``predictions.csv`` and return its stats dict.

    The condition becomes durable on disk the instant this returns, so nothing is
    held in memory to the end of the run (incremental-write contract). Empty
    ``records`` (a condition with no spots, or no images) still writes a
    schema-valid header-only CSV, exactly as before.
    """
    df = records_to_dataframe(records)
    assert list(df.columns) == list(SCHEMA_COLUMNS)
    write_spots(df, out_csv)
    return {
        "done": True,
        "skipped": False,
        "n_images": int(n_images),
        "n_spots": int(len(df)),
        # Per-condition wall time is ambiguous once loads overlap compute across
        # the shared loader, so report the honest work time: this condition's own
        # dataload wait + compute. The one-time worker spawn lands on whichever
        # condition owns the first batch (it happens once per run, not per cond).
        "seconds": round(dataload_s + compute_s, 4),
        "dataload_s": round(dataload_s, 4),
        "compute_s": round(compute_s, 4),
        "predictions_csv": str(out_csv.name),
    }


def _run_conditions(
    bundle: CheckpointBundle,
    run_list: list[tuple[Condition, Path, list[tuple[str, Path]]]],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    on_condition_done,
) -> None:
    """Run inference over MANY conditions through ONE DataLoader.

    A single dataset/loader (one worker pool for the whole run -- no per-condition
    respawn) streams every image of every to-run condition. Batches are built
    condition-homogeneous, so a batch never straddles two output CSVs and
    per-condition timing stays clean. Each condition's CSV is flushed the moment
    its last image is decoded, and ``on_condition_done(cond, stats)`` is invoked
    -- in condition order -- so the manifest stays durable and results are never
    held in memory to the end.

    ``run_list`` is ``[(Condition, out_csv, pairs), ...]`` for conditions that
    still need running; skipped and empty-image conditions are handled upstream,
    so every entry here has at least one image.
    """
    # Flatten to (cond_index, image_id, path); the index carries the condition
    # through the workers. Build condition-homogeneous batches in condition order.
    items: list[tuple[int, str, Path]] = []
    batches: list[list[int]] = []
    expected: list[int] = []
    for cond_index, (_cond, _out_csv, pairs) in enumerate(run_list):
        start = len(items)
        for image_id, path in pairs:
            items.append((cond_index, image_id, path))
        idxs = list(range(start, len(items)))
        for i in range(0, len(idxs), batch_size):
            batches.append(idxs[i:i + batch_size])
        expected.append(len(pairs))

    # ONE loader for the entire run: the worker pool is spawned once, not per
    # condition. batch_sampler yields our pre-grouped homogeneous batches, in
    # order, so batches (and thus per-condition completion) arrive deterministically.
    dataset = _BenchmarkImageDataset(items)
    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        persistent_workers=(num_workers > 0 and len(batches) > 0),
        collate_fn=_collate,
    )

    model = bundle.model
    params = bundle.params
    is_cuda = device.type == "cuda"

    records_by_cond: dict[int, list[SpotRecord]] = {i: [] for i in range(len(run_list))}
    seen_by_cond = [0] * len(run_list)
    dataload_by_cond = [0.0] * len(run_list)
    compute_by_cond = [0.0] * len(run_list)

    t_batch_start = time.perf_counter()
    with torch.no_grad():
        for cond_indices, image_ids, imgs in loader:
            ci = cond_indices[0]  # homogeneous batch -- all items share the condition
            dataload_by_cond[ci] += time.perf_counter() - t_batch_start

            t_c = time.perf_counter()
            imgs = normalize_counts(imgs, params.adc_max).to(device, non_blocking=pin_memory)
            preds = model(imgs)
            if is_cuda:
                torch.cuda.synchronize()
            for i, image_id in enumerate(image_ids):
                records_by_cond[ci].extend(_decode_image(preds, i, image_id, params))
            seen_by_cond[ci] += len(image_ids)
            compute_by_cond[ci] += time.perf_counter() - t_c

            # Flush this condition the instant its last image is decoded. Batches
            # arrive in sampler order, so a condition's final batch is the last we
            # see for it -- this triggers exactly once, in condition order.
            if seen_by_cond[ci] == expected[ci]:
                cond, out_csv, _pairs = run_list[ci]
                stats = _write_condition_csv(
                    records_by_cond[ci], out_csv, n_images=seen_by_cond[ci],
                    dataload_s=dataload_by_cond[ci], compute_s=compute_by_cond[ci])
                records_by_cond[ci] = []   # release memory; nothing held to the end
                on_condition_done(cond, stats)

            t_batch_start = time.perf_counter()


# --------------------------------------------------------------------------- #
# Device resolution (fail loud)                                                #
# --------------------------------------------------------------------------- #
def _resolve_device(requested: str, *, log_fn) -> torch.device:
    """Resolve ``auto|cpu|cuda`` to a device; abort or warn per the GPU rule."""
    cuda_ok = torch.cuda.is_available()
    if requested == "cuda":
        assert cuda_ok, "device=cuda requested but torch.cuda.is_available() is False"
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    # auto
    if cuda_ok:
        return torch.device("cuda")
    log_fn("[infer][WARNING] no CUDA device found -- falling back to CPU. "
           "A full benchmark on CPU will be SLOW; pass --device cuda on a GPU box "
           "to fail loud if the GPU is missing (guards the compute-starvation bug).")
    return torch.device("cpu")


def _device_label(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        try:
            return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
        except Exception:
            return f"cuda:{idx}"
    return "cpu"


# --------------------------------------------------------------------------- #
# Top-level driver                                                            #
# --------------------------------------------------------------------------- #
def run_inference(
    checkpoint: str,
    *,
    bench_root: Path,
    results_root: Path,
    repo_root: Path,
    checkpoints_root: Path,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int | None = None,
    smoke: bool = False,
    smoke_conditions: int = 2,
    smoke_images: int = 3,
    log_fn=print,
) -> dict:
    """Run one checkpoint over the benchmark; write per-condition CSVs + manifest.

    ``checkpoint`` is a single checkpoint name or ``"all"`` (both carried
    checkpoints, each a separate labelled method). Returns the last method's
    manifest dict.
    """
    if checkpoint == "all":
        names = discover_checkpoints(checkpoints_root)
        if not names:
            raise FileNotFoundError(f"no checkpoints under {checkpoints_root}")
    else:
        names = [checkpoint]

    dev = _resolve_device(device, log_fn=log_fn)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)
    pin_memory = dev.type == "cuda"

    log_fn(f"[infer] device={_device_label(dev)}  cuda_available={torch.cuda.is_available()}")
    log_fn(f"[infer] batch_size={batch_size}  num_workers={num_workers}  "
           f"pin_memory={pin_memory}  smoke={smoke}")

    conditions = discover_conditions(bench_root)
    if smoke:
        subset: list[Condition] = []
        for family in _FAMILIES:
            fam = [c for c in conditions if c.family == family][:smoke_conditions]
            subset.extend(fam)
        conditions = subset
    log_fn(f"[infer] benchmark={bench_root}  conditions={len(conditions)}"
           + ("  (smoke subset)" if smoke else ""))

    image_limit = smoke_images if smoke else None
    last_manifest: dict = {}

    for name in names:
        bundle = load_checkpoint(name, checkpoints_root=checkpoints_root, repo_root=repo_root)
        method = method_name(name, bundle.training_git_sha)
        legacy = is_legacy_checkpoint(bundle.training_git_sha)
        method_dir = results_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = method_dir / "RUN_MANIFEST.json"

        n_params = sum(p.numel() for p in bundle.model.parameters())
        log_fn(f"\n[infer] === method={method}  checkpoint={name} "
               f"({n_params / 1e6:.2f}M params) ===")
        log_fn(f"[infer] training_git_sha={bundle.training_git_sha}  "
               + ("(LEGACY/REFERENCE -- trained on OLD dirty-tree simulator; "
                  "NOT the reproducible headline)" if legacy else
                  "(CLEAN RETRAIN -- reproducible headline model; NOT legacy)"))
        log_fn(f"[infer] inference params: peak_threshold={bundle.params.peak_threshold} "
               f"nms_kernel={bundle.params.nms_kernel} max_spots={bundle.params.max_spots} "
               f"adc_max={bundle.params.adc_max}")

        manifest = _init_manifest(bundle, method, dev, bench_root, results_root,
                                  repo_root, smoke, batch_size, num_workers)

        t_method = time.perf_counter()
        counts = {"done": 0, "skipped": 0}

        def _record_done(cond: Condition, stats: dict) -> None:
            """Persist one completed condition to the manifest (incremental)."""
            manifest["conditions"][cond.key] = stats
            counts["done"] += 1
            starve = ""
            if stats["compute_s"] > 0 and stats["dataload_s"] > stats["compute_s"]:
                starve = "  [dataload-bound]"
            log_fn(f"  [run ] {cond.key:<40} {stats['n_images']:>3} imgs "
                   f"{stats['n_spots']:>6} spots  {stats['seconds']:.2f}s "
                   f"(load {stats['dataload_s']:.2f}s / compute {stats['compute_s']:.2f}s){starve}")
            _write_json(manifest_path, manifest)   # durable after every condition

        # First pass: settle skip-if-exists and empty-image conditions (whose
        # images are never loaded), and collect the rest for a single shared
        # loader so worker processes are spawned ONCE, not per condition.
        run_list: list[tuple[Condition, Path, list[tuple[str, Path]]]] = []
        for cond in conditions:
            out_dir = method_dir / cond.family / cond.label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "predictions.csv"

            if _is_valid_predictions_csv(out_csv):
                stats = {"done": True, "skipped": True,
                         "predictions_csv": out_csv.name,
                         "note": "skip-if-exists: schema-valid CSV already present"}
                manifest["conditions"][cond.key] = stats
                counts["skipped"] += 1
                log_fn(f"  [skip] {cond.key:<40} (exists, schema-valid) ~0.00s")
                _write_json(manifest_path, manifest)
                continue

            pairs = _condition_images(cond, limit=image_limit)
            if not pairs:
                # No images: write the schema-valid empty CSV now; nothing to load.
                stats = _write_condition_csv([], out_csv, n_images=0,
                                             dataload_s=0.0, compute_s=0.0)
                _record_done(cond, stats)
                continue
            run_list.append((cond, out_csv, pairs))

        if run_list:
            _run_conditions(
                bundle, run_list, device=dev, batch_size=batch_size,
                num_workers=num_workers, pin_memory=pin_memory,
                on_condition_done=_record_done)

        n_done, n_skipped = counts["done"], counts["skipped"]
        method_seconds = time.perf_counter() - t_method
        _finalize_manifest(manifest, method_seconds)
        _write_json(manifest_path, manifest)
        log_fn(f"[infer] method {method} done: {n_done} run, {n_skipped} skipped, "
               f"{manifest['totals']['n_images']} imgs, {manifest['totals']['n_spots']} spots, "
               f"{method_seconds:.2f}s  (load {manifest['totals']['dataload_s']:.2f}s / "
               f"compute {manifest['totals']['compute_s']:.2f}s)")
        log_fn(f"[infer] manifest -> {manifest_path}")
        last_manifest = manifest

    return last_manifest


def _init_manifest(bundle, method, dev, bench_root, results_root, repo_root,
                   smoke, batch_size, num_workers) -> dict:
    return {
        "method": method,
        "checkpoint": bundle.name,
        "checkpoint_path": bundle.checkpoint_rel,
        "provenance": {
            "status": ("LEGACY/REFERENCE" if is_legacy_checkpoint(bundle.training_git_sha)
                       else "CLEAN RETRAIN/HEADLINE"),
            "training_git_sha": bundle.training_git_sha,
            "note": (_LEGACY_NOTE if is_legacy_checkpoint(bundle.training_git_sha)
                     else _CLEAN_NOTE),
        },
        "device": _device_label(dev),
        "cuda_available": bool(torch.cuda.is_available()),
        "inference_params": {
            "adc_max": bundle.params.adc_max,
            "peak_threshold": bundle.params.peak_threshold,
            "nms_kernel": bundle.params.nms_kernel,
            "max_spots": bundle.params.max_spots,
            "logvar_min": bundle.params.logvar_min,
            "logvar_max": bundle.params.logvar_max,
        },
        "dataloader": {"batch_size": batch_size, "num_workers": num_workers},
        "benchmark_root": _rel_or_str(bench_root, repo_root),
        "results_root": _rel_or_str(results_root, repo_root),
        "smoke": bool(smoke),
        "schema_columns": list(SCHEMA_COLUMNS),
        "conditions": {},
        "totals": {},
    }


def _finalize_manifest(manifest: dict, method_seconds: float) -> None:
    conds = manifest["conditions"].values()
    ran = [c for c in conds if not c.get("skipped")]
    manifest["totals"] = {
        "n_conditions": len(manifest["conditions"]),
        "n_conditions_run": len(ran),
        "n_conditions_skipped": sum(1 for c in conds if c.get("skipped")),
        "n_images": int(sum(c.get("n_images", 0) for c in ran)),
        "n_spots": int(sum(c.get("n_spots", 0) for c in ran)),
        "seconds": round(method_seconds, 4),
        "dataload_s": round(sum(c.get("dataload_s", 0.0) for c in ran), 4),
        "compute_s": round(sum(c.get("compute_s", 0.0) for c in ran), 4),
    }


def _rel_or_str(path: Path, repo_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root))
    except Exception:
        return str(path)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)

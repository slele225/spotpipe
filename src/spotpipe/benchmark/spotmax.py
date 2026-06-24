"""SpotMAX + aperture adapter helpers (external-detector method).

This is the in-repo side of the ``spotmax_ai_plus_aperture`` method. It mirrors
:mod:`spotpipe.benchmark.spotiflow` and :mod:`spotpipe.benchmark.cmeanalysis`
exactly: an EXTERNAL detector (SpotMAX / Cell-ACDC) localizes spots and reaches
us ONLY as a normalized detections CSV; everything here consumes that CSV and
nothing else. **This module never imports the external ``spotmax`` package** --
importing ``spotpipe`` must not require SpotMAX installed (the external
dependency is invoked purely as a CLI subprocess, ``spotmax -p config.ini``, run
in its own environment; only the SpotMAX *output tables* come back to us).

Role split (the fair first version, identical in spirit to the other adapters):

* SpotMAX is the **detector / localizer ONLY**. It runs (headless, via its INI
  workflow) on a RAW detector image -- here the pixelwise ``raw_max`` of the two
  raw channels, exported as SpotMAX-compatible TIFFs by
  ``scripts/export_spotmax_input.py``. It reports spot centres in its own output
  tables (``0_detected_spots`` / ``1_valid_spots`` / ``2_spotfit``).
* SpotMAX's NATIVE per-spot intensities / amplitudes are **NOT** used as the
  canonical ``I1`` / ``I2`` (they would be a different, separately-named method
  whose units must be verified first). Only position (and an optional confidence
  for ``p_detect``) crosses over.
* Canonical ``I1`` / ``I2`` are extracted HERE, by the same **aperture + annulus
  background** estimator the in-repo aperture baseline uses
  (:func:`spotpipe.benchmark.baselines.aperture_photometry`), read from the
  **photon-proportional** images. Raw counts are never divided; the simulator's
  true-background (the non-fair simulator background) files are never read.

Two-stage CSV contract
----------------------
1. **Neutral detections CSV** (``scripts/convert_spotmax_output.py`` parses the
   SpotMAX output tables into this; :func:`parse_spotmax_output` builds it).
   Columns::

       image_id, x, y, p_detect, native_source_file, native_row, native_columns_json

   * ``x`` = sub-pixel column, ``y`` = sub-pixel row, spotpipe's 0-indexed
     convention (origin top-left). SpotMAX's native coordinate columns are
     mapped here via :func:`resolve_xy_columns`; the chosen columns are recorded
     so the convention is traceable rather than assumed.
   * ``p_detect`` is a SpotMAX confidence if one is available, else ``NaN``.
   * ``native_columns_json`` preserves the full native row for traceability.

2. **Normalized detections CSV** -- the adapter's actual input. The neutral CSV
   IS a valid normalized CSV (it has ``image_id, x, y`` plus optional
   ``p_detect``); the adapter ignores the provenance columns.

The coordinate-column resolution and the non-positive-intensity policy are the
two things most likely to need correcting after a first real SpotMAX run, so
both are explicit, configurable, and unit-tested here rather than buried.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.benchmark import baselines
from spotpipe.schema import SpotRecord, records_to_dataframe

__all__ = [
    "SPOTMAX_METHOD",
    "SPOTMAX_METHOD_AI",
    "SPOTMAX_METHOD_THRESHOLD",
    "SPOTMAX_METHODS",
    "SPOTS_ENDNAME",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "NEUTRAL_COLUMNS",
    "X_COLUMN_CANDIDATES",
    "Y_COLUMN_CANDIDATES",
    "P_DETECT_COLUMN_CANDIDATES",
    "TABLE_PRIORITY",
    "resolve_xy_columns",
    "resolve_p_detect_column",
    "find_spotmax_tables",
    "parse_spotmax_output",
    "load_normalized_spotmax_detections",
    "merge_neutral_detections",
    "spotmax_plus_aperture",
    "neutral_to_canonical",
    "load_photon_images",
    "build_detect_image",
    "position_name",
    "export_positions",
    "rewrite_ini_folder_path",
    "SpotmaxPlusApertureAdapter",
]

# Two HONEST method names share one adapter -- they differ ONLY in how SpotMAX was
# configured to detect (the SpotMAX INI), never in the in-repo photometry:
#   * spotmax_ai_plus_aperture        -- SpotMAX "spotMAX AI" detector.
#   * spotmax_threshold_plus_aperture -- SpotMAX Thresholding + peak_local_max
#                                        (the classical, non-AI detector path).
# Both then use the SAME aperture+annulus photometry on the photon images. The
# method name is carried into `flags` so the detector used is never ambiguous.
SPOTMAX_METHOD_AI = "spotmax_ai_plus_aperture"
SPOTMAX_METHOD_THRESHOLD = "spotmax_threshold_plus_aperture"
SPOTMAX_METHODS: tuple[str, ...] = (SPOTMAX_METHOD_AI, SPOTMAX_METHOD_THRESHOLD)
SPOTMAX_METHOD = SPOTMAX_METHOD_AI  # back-compat default

# Endname (suffix before the extension) of the exported detection-channel TIFF.
# The SpotMAX INI's spots-channel endname must match this. Single source of truth
# shared by the exporter and the batch runner.
SPOTS_ENDNAME = "spots"

# Adapter input contract (normalized detections CSV).
REQUIRED_COLUMNS: tuple[str, ...] = ("image_id", "x", "y")
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "p_detect", "native_source_file", "native_row", "native_columns_json",
)
# Neutral detections CSV (the traceable intermediate produced from the
# SpotMAX output tables).
NEUTRAL_COLUMNS: tuple[str, ...] = (
    "image_id", "x", "y", "p_detect",
    "native_source_file", "native_row", "native_columns_json",
)

_EPS = 1e-6

# --------------------------------------------------------------------------- #
# Native-column resolution (SpotMAX output column names are NOT assumed)       #
# --------------------------------------------------------------------------- #
# SpotMAX is scikit-image flavoured, so its spot tables use (z, y, x) ordering
# with x = column and y = row -- which is exactly spotpipe's convention. The exact
# column NAMES vary by version / table; a real run (v1.3.1) emits BOTH a GLOBAL
# whole-frame coordinate (``x`` / ``y``) and an object-LOCAL one (``x_local`` /
# ``y_local``, relative to a segmented object's bounding box). We feed SpotMAX the
# whole 2-D detection frame and extract intensity from the whole photon image, so
# the GLOBAL coordinate is the one that aligns -- it is preferred here; local
# variants are only a last-resort fallback (and equal the global one when there is
# no segmentation). We do NOT hard-code one name: we search these ordered
# candidates and RECORD which we used so the convention is verified, not assumed.
X_COLUMN_CANDIDATES: tuple[str, ...] = (
    "x", "x_global", "x_global_pxl", "col", "column", "X", "centroid-1", "centroid_x",
    "x_local", "x_local_pxl", "x_local_px", "x_pxl",
)
Y_COLUMN_CANDIDATES: tuple[str, ...] = (
    "y", "y_global", "y_global_pxl", "row", "Y", "centroid-0", "centroid_y",
    "y_local", "y_local_pxl", "y_local_px", "y_pxl",
)
# Confidence is NOT a stable SpotMAX field; default to NaN unless one of these is
# present (higher = more confident). p-value-like fields are intentionally left
# out (smaller = more confident -> would need a transform, so not used raw).
P_DETECT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "spotmax_p_detect", "p_detect", "confidence", "score", "spot_prob",
    "prediction_prob", "probability",
)
# Which output table to prefer when several are present. We use SpotMAX as a
# localizer, so valid (filtered) spots first, then all detected spots; spotfit is
# only a fallback for positions (its native intensities are never used). Matched
# as case-insensitive substrings (a real run names them e.g.
# ``1_1_valid_spots_<basename>.csv`` / ``1_0_detected_spots_<basename>.csv``).
TABLE_PRIORITY: tuple[str, ...] = ("valid_spots", "detected_spots", "spotfit")

# SpotMAX's output directory name, matched CASE-INSENSITIVELY: a real run writes
# ``spotMAX_output`` (lower-s), and case-sensitive filesystems would otherwise
# miss it.
_OUTPUT_DIR_NAME = "spotmax_output"
# Per-object AGGREGATE tables (one row per segmented object, not per spot) are
# never the per-spot table we want; excluded by this substring.
_AGGREGATE_MARKER = "aggregated"
_TABLE_SUFFIXES = (".csv", ".h5", ".hdf", ".hdf5", ".txt")


def resolve_xy_columns(
    columns,
    *,
    x_col: str | None = None,
    y_col: str | None = None,
) -> tuple[str, str]:
    """Pick the (x=column, y=row) source columns from a SpotMAX table's columns.

    Explicit ``x_col`` / ``y_col`` (e.g. from the CLI after inspecting a real
    run) win. Otherwise the first present candidate from
    :data:`X_COLUMN_CANDIDATES` / :data:`Y_COLUMN_CANDIDATES` is used. Raises
    ``ValueError`` -- listing the available columns -- if none match, so a column
    naming we have not seen is surfaced loudly instead of silently guessed.
    """
    cols = list(columns)
    cset = set(cols)

    def _pick(explicit, candidates, axis):
        if explicit is not None:
            if explicit not in cset:
                raise ValueError(
                    f"requested {axis} column {explicit!r} not in SpotMAX table "
                    f"columns {cols}"
                )
            return explicit
        for cand in candidates:
            if cand in cset:
                return cand
        raise ValueError(
            f"could not find a {axis} coordinate column in SpotMAX table columns "
            f"{cols}; tried {list(candidates)}. Pass an explicit --x-col/--y-col "
            "after inspecting the real output (x = column, y = row)."
        )

    return _pick(x_col, X_COLUMN_CANDIDATES, "x"), _pick(y_col, Y_COLUMN_CANDIDATES, "y")


def resolve_p_detect_column(columns, *, p_col: str | None = None) -> str | None:
    """Return the confidence column to use for ``p_detect``, or ``None``.

    Explicit ``p_col`` wins (and must exist). Otherwise the first present
    candidate from :data:`P_DETECT_COLUMN_CANDIDATES`; if none is present we
    return ``None`` and ``p_detect`` is left ``NaN`` (the honest default --
    SpotMAX does not natively expose a higher-is-better detection probability).
    """
    cset = set(columns)
    if p_col is not None:
        if p_col not in cset:
            raise ValueError(
                f"requested p_detect column {p_col!r} not in SpotMAX table "
                f"columns {list(columns)}"
            )
        return p_col
    for cand in P_DETECT_COLUMN_CANDIDATES:
        if cand in cset:
            return cand
    return None


# --------------------------------------------------------------------------- #
# Output-table discovery + parsing -> neutral detections                       #
# --------------------------------------------------------------------------- #
def _read_table(path: Path) -> pd.DataFrame:
    """Read one SpotMAX output table (.csv or .h5) into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix in (".h5", ".hdf", ".hdf5"):
        return pd.read_hdf(path)
    raise ValueError(f"unsupported SpotMAX table format {path.suffix!r} ({path})")


def find_spotmax_tables(
    run_dir: str | Path,
    *,
    table_priority: tuple[str, ...] = TABLE_PRIORITY,
) -> dict[str, Path]:
    """Map each ``Position_*`` folder to its single best SpotMAX output table.

    Searches ``run_dir`` recursively for the SpotMAX output directory
    (``spotMAX_output``, matched case-insensitively) and, in each, picks the
    highest-priority table (``table_priority``) actually present, EXCLUDING
    per-object aggregate tables. Returns ``{position_name: table_path}``. SpotMAX
    prefixes tables with run/table numbers (e.g. ``1_1_valid_spots_<basename>.csv``);
    we match priority keys as case-insensitive substrings so the exact prefix /
    suffix / basename does not need to be known in advance.
    """
    run_dir = Path(run_dir)
    out: dict[str, Path] = {}
    sm_dirs = sorted(p for p in run_dir.rglob("*")
                     if p.is_dir() and p.name.lower() == _OUTPUT_DIR_NAME)
    for sm_dir in sm_dirs:
        # Resolve the owning Position by walking up to a 'Position'-named ancestor
        # (the output dir is a sibling of Images, or nested -- handled either way).
        position = _position_name_for(sm_dir)
        tables = [
            p for p in sm_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _TABLE_SUFFIXES
            and _AGGREGATE_MARKER not in p.name.lower()
        ]
        chosen: Path | None = None
        for key in table_priority:
            matches = sorted(p for p in tables if key.lower() in p.name.lower())
            if matches:
                chosen = matches[0]
                break
        if chosen is not None:
            out[position] = chosen
    return out


def _position_name_for(path: Path) -> str:
    """Nearest ancestor directory whose name starts with 'Position'; else parent."""
    for parent in path.parents:
        if parent.name.lower().startswith("position"):
            return parent.name
    return path.parent.name


def parse_spotmax_output(
    run_dir: str | Path,
    id_map: dict[str, str],
    *,
    table_priority: tuple[str, ...] = TABLE_PRIORITY,
    x_col: str | None = None,
    y_col: str | None = None,
    p_col: str | None = None,
) -> pd.DataFrame:
    """Parse SpotMAX output tables under ``run_dir`` into a neutral detections frame.

    ``id_map`` maps ``Position_xxxxxx`` -> benchmark ``image_id`` (written by the
    exporter). For each position's chosen table, every row becomes one neutral
    detection: position/confidence mapped into ``x, y, p_detect`` and the full
    native row preserved as ``native_columns_json``. Native intensities are NOT
    mapped. Returns a DataFrame with exactly :data:`NEUTRAL_COLUMNS`.
    """
    tables = find_spotmax_tables(run_dir, table_priority=table_priority)
    rows: list[dict] = []
    for position, table_path in sorted(tables.items()):
        image_id = id_map.get(position)
        if image_id is None:
            # A Position with no id-map entry: skip rather than guess its image.
            continue
        df = _read_table(table_path)
        if len(df) == 0:
            continue
        xcol, ycol = resolve_xy_columns(df.columns, x_col=x_col, y_col=y_col)
        pcol = resolve_p_detect_column(df.columns, p_col=p_col)
        src_name = table_path.name
        for native_row, (_, r) in enumerate(df.iterrows()):
            rows.append({
                "image_id": str(image_id),
                "x": float(r[xcol]),
                "y": float(r[ycol]),
                "p_detect": float(r[pcol]) if pcol is not None and pd.notna(r[pcol]) else math.nan,
                "native_source_file": src_name,
                "native_row": int(native_row),
                "native_columns_json": json.dumps(_jsonable(r.to_dict()), sort_keys=True),
            })
    return pd.DataFrame(rows, columns=list(NEUTRAL_COLUMNS))


def _jsonable(d: dict) -> dict:
    """Coerce a native row dict to JSON-serialisable scalars (numpy -> python)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[str(k)] = int(v)
        elif isinstance(v, (np.floating,)):
            fv = float(v)
            out[str(k)] = None if not math.isfinite(fv) else fv
        elif isinstance(v, (np.bool_,)):
            out[str(k)] = bool(v)
        elif isinstance(v, float):
            out[str(k)] = None if not math.isfinite(v) else v
        else:
            out[str(k)] = v if isinstance(v, (str, int, bool, type(None))) else str(v)
    return out


# --------------------------------------------------------------------------- #
# Normalized-detections loader (the adapter's input)                           #
# --------------------------------------------------------------------------- #
def load_normalized_spotmax_detections(path: str | Path) -> pd.DataFrame:
    """Load + validate a normalized SpotMAX detections CSV.

    The neutral CSV produced by :func:`parse_spotmax_output` is a valid input
    (it has the required ``image_id, x, y`` plus optional ``p_detect``). Raises
    ``ValueError`` if a required column is missing. ``image_id`` -> str and
    ``x`` / ``y`` -> float for type-stable grouping / photometry; ``p_detect`` is
    coerced to float (non-numeric / absent -> NaN, never fabricated).
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"normalized SpotMAX detections CSV {str(path)!r} is missing required "
            f"columns {missing}; required = {list(REQUIRED_COLUMNS)}. Produce it with "
            "scripts/convert_spotmax_output.py."
        )
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    if "p_detect" in df.columns:
        df["p_detect"] = pd.to_numeric(df["p_detect"], errors="coerce").astype(float)
    return df


# --------------------------------------------------------------------------- #
# Detections + photon images -> canonical schema                               #
# --------------------------------------------------------------------------- #
def _flags_for(method_name: str, detect_image: str) -> str:
    return f"{method_name};detect_image={detect_image};photometry=aperture_annulus"


def spotmax_plus_aperture(
    photon: np.ndarray,
    det_df: pd.DataFrame,
    *,
    image_id: str,
    cfg: dict | None = None,
    method_name: str = SPOTMAX_METHOD_AI,
) -> tuple[pd.DataFrame, dict]:
    """SpotMAX centres + aperture photometry on the photon images -> canonical schema.

    ``photon`` is the two-channel ``[2, H, W]`` photon-proportional image (already
    offset-subtracted + gain-corrected). ``det_df`` are the normalized detection
    rows for THIS image. ``I1`` / ``I2`` are aperture + annulus reads at the
    SpotMAX centres in each channel (``gain=1.0`` -- the photon image is already
    gain-corrected), reusing
    :func:`spotpipe.benchmark.baselines.aperture_photometry` so the estimator is
    byte-for-byte identical to the aperture baseline. ``sigma*_hat`` and
    ``uncertainty*`` are left NaN.

    Non-positive extracted intensities are handled EXPLICITLY per the
    ``nonpositive`` policy (``cfg['nonpositive']``):

    * ``clamp`` (default) -- keep the spot, clamp ``I`` at a tiny floor (the
      estimator already floors at ``_EPS``), and append ``nonpos_clamped`` to its
      flags so the clamp is visible downstream.
    * ``reject`` -- drop the spot (recorded in the returned stats) rather than
      emit a clamped intensity.

    Returns ``(canonical_df, stats)`` where ``stats`` reports
    ``n_in / n_out / n_nonpositive`` so the caller can print transparent counts
    (never a silent drop).
    """
    cfg = cfg or {}
    photon = np.asarray(photon, dtype=float)
    if photon.ndim != 3 or photon.shape[0] < 2:
        raise ValueError(f"photon image must be [2, H, W]; got shape {photon.shape}")

    xs = det_df["x"].to_numpy(dtype=float)
    ys = det_df["y"].to_numpy(dtype=float)
    n_in = int(xs.size)
    if n_in == 0:
        return records_to_dataframe([]), {"n_in": 0, "n_out": 0, "n_nonpositive": 0}

    r_ap, r_in, r_out = baselines._aperture_radii(cfg)
    I1 = baselines.aperture_photometry(photon[0], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=1.0)
    I2 = baselines.aperture_photometry(photon[1], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=1.0)

    # aperture_photometry floors the signal at _EPS, so a clamped read lands at
    # exactly _EPS (gain=1.0). That is the non-positive marker.
    floor = _EPS
    nonpos = (I1 <= floor) | (I2 <= floor)
    n_nonpositive = int(np.count_nonzero(nonpos))

    if "p_detect" in det_df.columns:
        p_detect = pd.to_numeric(det_df["p_detect"], errors="coerce").to_numpy(dtype=float)
    else:
        p_detect = np.full(n_in, math.nan, dtype=float)

    policy = str(cfg.get("nonpositive", "clamp")).lower()
    detect_image = str(cfg.get("detect_image", "raw_max"))
    base_flags = _flags_for(method_name, detect_image)

    records = []
    for k in range(n_in):
        if nonpos[k] and policy == "reject":
            continue
        flags = base_flags + (";nonpos_clamped" if nonpos[k] else "")
        records.append(
            SpotRecord.from_logs(
                image_id=image_id,
                spot_id=len(records),
                x=float(xs[k]),
                y=float(ys[k]),
                p_detect=float(p_detect[k]),
                logI1=float(math.log(max(I1[k], floor))),
                logI2=float(math.log(max(I2[k], floor))),
                sigma1_hat=math.nan,
                sigma2_hat=math.nan,
                uncertainty1=math.nan,
                uncertainty2=math.nan,
                flags=flags,
            )
        )
    out = records_to_dataframe(records)
    stats = {"n_in": n_in, "n_out": int(len(out)), "n_nonpositive": n_nonpositive}
    return out, stats


# --------------------------------------------------------------------------- #
# Merge + canonical conversion (shared by the convert script + batch runner)    #
# --------------------------------------------------------------------------- #
def merge_neutral_detections(paths) -> pd.DataFrame:
    """Concatenate per-batch neutral detection CSVs into one neutral frame.

    Each path must hold a neutral CSV (:data:`NEUTRAL_COLUMNS`). Batches partition
    the frozen set by image, so a plain concat is the correct merge (no image
    appears in two batches). Validates columns, coerces ``image_id`` -> str, and
    returns exactly :data:`NEUTRAL_COLUMNS`. An empty input yields an empty,
    correctly-typed frame.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in NEUTRAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"neutral detections CSV {str(p)!r} is missing columns {missing}; "
                f"expected {list(NEUTRAL_COLUMNS)}"
            )
        frames.append(df[list(NEUTRAL_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=list(NEUTRAL_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out["image_id"] = out["image_id"].astype(str)
    return out


def load_photon_images(bench_dir: str | Path, image_ids) -> dict[str, np.ndarray]:
    """Load the ``[2, H, W]`` photon image for each requested image_id.

    Reads the frozen set's manifest and the ``images_ch{1,2}_photon`` TIFFs (the
    fair-photometry inputs). Image ids absent from the manifest are skipped. Never
    reads raw counts or the simulator true-background files.
    """
    import tifffile

    bench_dir = Path(bench_dir)
    with open(bench_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    by_id = {str(e["image_id"]): e for e in manifest["images"]}
    out: dict[str, np.ndarray] = {}
    for image_id in sorted(set(map(str, image_ids))):
        entry = by_id.get(image_id)
        if entry is None:
            continue
        p1 = tifffile.imread(bench_dir / entry["ch1_photon"])
        p2 = tifffile.imread(bench_dir / entry["ch2_photon"])
        out[image_id] = np.stack([p1, p2], axis=0).astype(float)
    return out


def neutral_to_canonical(
    neutral_df: pd.DataFrame,
    bench_dir: str | Path,
    *,
    cfg: dict | None = None,
    method_name: str = SPOTMAX_METHOD_AI,
) -> tuple[pd.DataFrame, dict]:
    """Neutral detections + photon images -> canonical 16-column predictions.

    Groups ``neutral_df`` by ``image_id``, extracts ``I1`` / ``I2`` by aperture +
    annulus photometry on each image's photon image (the SINGLE photometry path,
    shared with the harness adapter), and emits the canonical schema with honest
    ``method_name`` flags. Returns ``(canonical_df, stats)`` where ``stats``
    reports ``n_in / n_out / n_nonpositive / n_missing_image`` (transparent
    counts; non-positive intensities are clamped+flagged or rejected per
    ``cfg['nonpositive']``).
    """
    cfg = cfg or {}
    neutral_df = neutral_df.copy()
    neutral_df["image_id"] = neutral_df["image_id"].astype(str)
    photon_by_image = load_photon_images(bench_dir, neutral_df["image_id"].unique())

    total = {"n_in": 0, "n_out": 0, "n_nonpositive": 0, "n_missing_image": 0}
    frames = []
    for image_id, sub in neutral_df.groupby("image_id"):
        photon = photon_by_image.get(str(image_id))
        if photon is None:
            total["n_missing_image"] += int(len(sub))
            continue
        df, stats = spotmax_plus_aperture(
            photon, sub, image_id=str(image_id), cfg=cfg, method_name=method_name,
        )
        for k in ("n_in", "n_out", "n_nonpositive"):
            total[k] += stats[k]
        frames.append(df)
    pred = pd.concat(frames, ignore_index=True) if frames else records_to_dataframe([])
    return pred, total


# --------------------------------------------------------------------------- #
# Detection-input export + INI rewrite (shared by exporter + batch runner)      #
# --------------------------------------------------------------------------- #
def build_detect_image(ch1: np.ndarray, ch2: np.ndarray, detect_image: str) -> np.ndarray:
    """Collapse the two raw channels into the single 2-D image SpotMAX detects on.

    ``raw_max`` (default + recommended first protocol) is the pixelwise max so a
    spot bright in EITHER channel can be found. Output is float32.
    """
    a = np.asarray(ch1, dtype=np.float32)
    b = np.asarray(ch2, dtype=np.float32)
    if detect_image == "raw_max":
        return np.maximum(a, b)
    if detect_image == "raw_sum":
        return a + b
    if detect_image == "master_ch1":
        return a
    if detect_image == "master_ch2":
        return b
    raise ValueError(
        f"unknown detect_image {detect_image!r}; supported: "
        "raw_max | raw_sum | master_ch1 | master_ch2"
    )


def position_name(index: int) -> str:
    """1-based SpotMAX Position folder name (``Position_000001`` ...)."""
    return f"Position_{index + 1:06d}"


def export_positions(
    bench_dir: str | Path,
    entries: list[dict],
    input_root: str | Path,
    detect_image: str,
    *,
    start_index: int = 0,
) -> list[dict]:
    """Write a Cell-ACDC / SpotMAX ``Position_*/Images`` tree for ``entries``.

    For each frozen-set manifest ``entry`` it reads the two RAW channels, builds
    the detection image (``detect_image``), and writes
    ``<input_root>/Position_<n>/Images/Position_<n>_{SPOTS_ENDNAME}.tif``. Returns
    id-map rows (``position`` <-> benchmark ``image_id`` + provenance) for the
    caller to persist as ``id_map.csv``. Reads only the raw channels (never the
    photon or simulator true-background files).
    """
    import tifffile

    bench_dir = Path(bench_dir)
    input_root = Path(input_root)
    input_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, entry in enumerate(entries):
        image_id = str(entry["image_id"])
        position = position_name(start_index + i)
        ch1 = tifffile.imread(bench_dir / entry["ch1_raw"])
        ch2 = tifffile.imread(bench_dir / entry["ch2_raw"])
        detect = build_detect_image(ch1, ch2, detect_image)

        images_dir = input_root / position / "Images"
        images_dir.mkdir(parents=True, exist_ok=True)
        tif_path = images_dir / f"{position}_{SPOTS_ENDNAME}.tif"
        tifffile.imwrite(tif_path, detect.astype(np.float32))

        rows.append({
            "position": position,
            "image_id": image_id,
            "detect_image": detect_image,
            "spots_tif": str(tif_path.relative_to(input_root)),
            "src_ch1_raw": entry["ch1_raw"],
            "src_ch2_raw": entry["ch2_raw"],
            "height": int(detect.shape[0]),
            "width": int(detect.shape[1]),
        })
    return rows


def rewrite_ini_folder_path(template_text: str, new_folder, *, key: str = "Folder path") -> str:
    """Return ``template_text`` with the ``<key> = ...`` line repointed at ``new_folder``.

    Targeted, line-based rewrite (not configparser) so the rest of the GUI-saved
    SpotMAX INI is preserved byte-for-byte (section names with spaces, ordering,
    comments). Case-insensitive on the key. Raises ``ValueError`` if no such line
    exists, so a renamed key is surfaced rather than silently ignored.
    """
    import re

    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=)(.*)$", re.IGNORECASE)
    out_lines, found = [], False
    for line in template_text.splitlines():
        m = pat.match(line)
        if m:
            out_lines.append(f"{m.group(1)} {new_folder}")
            found = True
        else:
            out_lines.append(line)
    if not found:
        raise ValueError(
            f"INI template has no '{key} =' line to rewrite; pass --ini-folder-key "
            "with the actual key name from your GUI-saved INI."
        )
    return "\n".join(out_lines) + "\n"


# --------------------------------------------------------------------------- #
# Adapter                                                                      #
# --------------------------------------------------------------------------- #
class SpotmaxPlusApertureAdapter:
    """SpotMAX centres + aperture photometry on the PHOTON images -> canonical schema.

    Construct via the registry: ``get_adapter('spotmax_ai_plus_aperture')``.

    ``predict`` reads the normalized detections CSV at
    ``config['spotmax']['detections_csv']`` (the neutral CSV produced by
    ``scripts/convert_spotmax_output.py`` is a valid input), then for each eval
    item extracts ``I1`` / ``I2`` by aperture + annulus photometry at the SpotMAX
    centres on the photon image (``item.photon`` when attached -- the frozen
    set's ``images_ch{1,2}_photon`` TIFFs -- else derived from raw counts +
    detector meta via the shared CMEAnalysis photon helper). ``gain=1.0`` because
    the photon image is already gain-corrected. ``sigma*_hat`` /
    ``uncertainty*`` are left NaN. Non-positive intensities are clamped+flagged
    (or rejected) per ``config['spotmax']['nonpositive']``, with transparent
    counts logged.
    """

    name = SPOTMAX_METHOD_AI

    def __init__(self, *, method_name: str = SPOTMAX_METHOD_AI, log_fn=print, **kwargs):
        # Instance-level name so the harness/registry report the honest method name
        # (AI vs threshold), and so the emitted `flags` match it.
        self.name = method_name
        self.method_name = method_name
        self._log = log_fn

    def predict(self, eval_set, config: dict) -> pd.DataFrame:
        # Imported lazily to avoid an import cycle (adapters <-> spotmax).
        from spotpipe.benchmark.adapters import Adapter, CmeAnalysisPlusApertureAdapter

        scfg = dict(config.get("spotmax", {}))
        det_path = scfg.get("detections_csv")
        if not det_path:
            raise ValueError(
                f"{self.name} requires config['spotmax']['detections_csv'] (the "
                "normalized/neutral SpotMAX detections CSV). Produce it by running "
                "SpotMAX in its own env then converting: "
                "scripts/export_spotmax_input.py -> spotmax -p config.ini -> "
                "scripts/convert_spotmax_output.py."
            )
        det_path = Path(det_path)
        if not det_path.exists():
            raise FileNotFoundError(
                f"{self.name}: detections CSV {str(det_path)!r} not found. Run the "
                "detector first (in a separate SpotMAX env): "
                "scripts/export_spotmax_input.py -> 'spotmax -p config.ini' -> "
                "scripts/convert_spotmax_output.py."
            )

        det = load_normalized_spotmax_detections(det_path)
        by_image = {str(k): v for k, v in det.groupby("image_id")}

        total = {"n_in": 0, "n_out": 0, "n_nonpositive": 0}
        frames = []
        for item in eval_set:
            sub = by_image.get(str(item.image_id))
            if sub is None or len(sub) == 0:
                continue  # never fabricate spots for an image with no detections
            photon = CmeAnalysisPlusApertureAdapter._photon_for(item)
            df, stats = spotmax_plus_aperture(
                photon, sub, image_id=item.image_id, cfg=scfg, method_name=self.name,
            )
            for k in total:
                total[k] += stats[k]
            frames.append(df)

        self._log(
            f"[spotmax] aperture photometry: {total['n_in']} detections in, "
            f"{total['n_out']} emitted, {total['n_nonpositive']} non-positive "
            f"({str(scfg.get('nonpositive', 'clamp')).lower()})"
        )
        if not frames:
            return records_to_dataframe([])
        out = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else records_to_dataframe([])
        return Adapter._concat([out])

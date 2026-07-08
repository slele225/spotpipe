"""Canonical spot-output schema.

This module defines the *single* canonical schema that every spot-detection
method in this project must emit -- our own model and any external baseline
adapter alike. All downstream analysis (log-ratio / slope fitting) and all
benchmarking read this schema and nothing else, so it is load-bearing: every
method must emit exactly these columns, in this order, with these meanings.

Units & conventions (see CLAUDE.md for the full rationale):

* Intensities ``I1``/``I2`` and their logs ``logI1``/``logI2`` are per-spot
  TOTAL INTEGRATED intensities in photon-proportional units, regressed directly
  by the network at each spot center -- never read back from image pixels.
* Per-channel detector offset is assumed already subtracted before any log or
  ratio is taken, in both simulation bookkeeping and inference.
* ``log_ratio = logI2 - logI1`` and ``ratio = I2 / I1`` (channel 2 over channel
  1) are stored explicitly for convenience; helpers below keep them consistent.
* ``uncertainty1``/``uncertainty2`` are the network's predicted per-spot
  heteroscedastic uncertainties on ``logI1``/``logI2`` (e.g. predicted standard
  deviation in log-intensity), used to weight the downstream slope fit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "SCHEMA_COLUMNS",
    "SpotRecord",
    "records_to_dataframe",
    "dataframe_to_records",
    "write_spots",
    "read_spots",
]


@dataclass
class SpotRecord:
    """One detected spot in the canonical output schema.

    The field order here defines the canonical column order on disk
    (see :data:`SCHEMA_COLUMNS`). Do not reorder without updating every method
    adapter -- identical emission across methods is the whole point.
    """

    image_id: str          # identifier of the source image / field of view
    spot_id: int           # unique spot index within the image
    x: float               # sub-pixel column coordinate (pixels)
    y: float               # sub-pixel row coordinate (pixels)
    p_detect: float        # detection confidence / probability in [0, 1]
    logI1: float           # log total integrated intensity, channel 1 (photon-prop.)
    logI2: float           # log total integrated intensity, channel 2 (photon-prop.)
    I1: float              # total integrated intensity, channel 1 (photon-prop.)
    I2: float              # total integrated intensity, channel 2 (photon-prop.)
    log_ratio: float       # logI2 - logI1
    ratio: float           # I2 / I1
    sigma1_hat: float      # estimated PSF width, channel 1 (pixels)
    sigma2_hat: float      # estimated PSF width, channel 2 (pixels)
    uncertainty1: float    # predicted heteroscedastic uncertainty on logI1
    uncertainty2: float    # predicted heteroscedastic uncertainty on logI2
    flags: str = ""        # free-form, comma-joined status flags (e.g. "saturated")

    @classmethod
    def from_logs(
        cls,
        *,
        image_id: str,
        spot_id: int,
        x: float,
        y: float,
        p_detect: float,
        logI1: float,
        logI2: float,
        sigma1_hat: float,
        sigma2_hat: float,
        uncertainty1: float,
        uncertainty2: float,
        flags: str = "",
    ) -> "SpotRecord":
        """Construct a record from the log-intensities the network predicts.

        ``I1``/``I2``/``log_ratio``/``ratio`` are derived so the redundant
        columns stay mutually consistent. This is the recommended constructor
        for both our model and baseline adapters.
        """
        return cls(
            image_id=image_id,
            spot_id=spot_id,
            x=x,
            y=y,
            p_detect=p_detect,
            logI1=logI1,
            logI2=logI2,
            I1=math.exp(logI1),
            I2=math.exp(logI2),
            log_ratio=logI2 - logI1,
            ratio=math.exp(logI2 - logI1),
            sigma1_hat=sigma1_hat,
            sigma2_hat=sigma2_hat,
            uncertainty1=uncertainty1,
            uncertainty2=uncertainty2,
            flags=flags,
        )


# Canonical column order, derived from the dataclass field order so the two can
# never drift apart.
SCHEMA_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(SpotRecord))


def records_to_dataframe(records: Iterable[SpotRecord]) -> pd.DataFrame:
    """Convert records to a DataFrame with the canonical columns in order."""
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows, columns=list(SCHEMA_COLUMNS))
    return df


def dataframe_to_records(df: pd.DataFrame) -> list[SpotRecord]:
    """Convert a canonical-schema DataFrame back into records.

    Raises ``ValueError`` if any required column is missing.
    """
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required schema columns: {missing}")
    records: list[SpotRecord] = []
    for row in df[list(SCHEMA_COLUMNS)].itertuples(index=False, name=None):
        kwargs = dict(zip(SCHEMA_COLUMNS, row))
        # Normalise types that pandas may have widened/narrowed.
        kwargs["image_id"] = str(kwargs["image_id"])
        kwargs["spot_id"] = int(kwargs["spot_id"])
        flags = kwargs.get("flags")
        kwargs["flags"] = "" if (flags is None or (isinstance(flags, float) and math.isnan(flags))) else str(flags)
        records.append(SpotRecord(**kwargs))
    return records


def write_spots(records: Sequence[SpotRecord] | pd.DataFrame, path: str | Path) -> Path:
    """Write spot records (or an already-built DataFrame) to canonical CSV.

    Always writes exactly :data:`SCHEMA_COLUMNS`, in order, with a header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = records if isinstance(records, pd.DataFrame) else records_to_dataframe(records)
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot write canonical CSV; missing columns: {missing}")
    df.to_csv(path, columns=list(SCHEMA_COLUMNS), index=False)
    return path


def read_spots(path: str | Path) -> list[SpotRecord]:
    """Read a canonical CSV back into a list of :class:`SpotRecord`."""
    df = pd.read_csv(path)
    return dataframe_to_records(df)

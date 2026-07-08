"""FROZEN interface package. The vendored ``schema.py`` is the source of truth.

This re-export exists only so the old import path ``from spotpipe.schema
import ...`` keeps working with schema.py vendored inside a package directory.
Never add fields here (CLAUDE.md rule 2).
"""

from spotpipe.schema.schema import (  # noqa: F401
    SCHEMA_COLUMNS,
    SpotRecord,
    dataframe_to_records,
    read_spots,
    records_to_dataframe,
    write_spots,
)

__all__ = [
    "SCHEMA_COLUMNS",
    "SpotRecord",
    "records_to_dataframe",
    "dataframe_to_records",
    "write_spots",
    "read_spots",
]

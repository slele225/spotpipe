"""`spotpipe smoke` runs end-to-end and its outputs are schema-valid."""

import json

import pandas as pd
import pytest

from spotpipe.cli import run_smoke
from spotpipe.paths import get_paths
from spotpipe.schema import SCHEMA_COLUMNS, read_spots


@pytest.fixture(scope="module")
def smoke_out(tmp_path_factory):
    cfg = get_paths().configs / "smoke.yaml"
    return run_smoke(cfg, tmp_path_factory.mktemp("smoke"))


def test_smoke_runs_and_writes_manifest(smoke_out):
    manifest = json.loads((smoke_out / "dataset" / "manifest.json").read_text())
    assert manifest["n_images"] == 50
    assert manifest["schema_columns"] == list(SCHEMA_COLUMNS)
    assert manifest["seed"] == 0


def test_ground_truth_is_schema_valid(smoke_out):
    manifest = json.loads((smoke_out / "dataset" / "manifest.json").read_text())
    for entry in manifest["images"][:5]:
        df = pd.read_csv(smoke_out / "dataset" / entry["spots_file"])
        assert list(df.columns) == list(SCHEMA_COLUMNS)
        read_spots(smoke_out / "dataset" / entry["spots_file"])  # parses into records


def test_predictions_are_schema_valid(smoke_out):
    preds = sorted((smoke_out / "predictions").glob("pred_*.csv"))
    assert len(preds) == 50
    for p in preds[:5]:
        df = pd.read_csv(p)
        assert list(df.columns) == list(SCHEMA_COLUMNS)
        read_spots(p)

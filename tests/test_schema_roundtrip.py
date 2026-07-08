"""The frozen schema round-trips records -> CSV -> records losslessly."""

import math

from spotpipe.schema import (
    SCHEMA_COLUMNS,
    SpotRecord,
    dataframe_to_records,
    read_spots,
    records_to_dataframe,
    write_spots,
)

EXPECTED_COLUMNS = (
    "image_id", "spot_id", "x", "y", "p_detect",
    "logI1", "logI2", "I1", "I2", "log_ratio", "ratio",
    "sigma1_hat", "sigma2_hat", "uncertainty1", "uncertainty2", "flags",
)


def _records():
    return [
        SpotRecord.from_logs(
            image_id="img_00000", spot_id=0, x=10.25, y=3.5, p_detect=0.9,
            logI1=4.0, logI2=4.5, sigma1_hat=1.2, sigma2_hat=1.4,
            uncertainty1=0.1, uncertainty2=0.2, flags="saturated",
        ),
        SpotRecord.from_logs(
            image_id="img_00000", spot_id=1, x=50.0, y=60.0, p_detect=0.5,
            logI1=2.0, logI2=1.5, sigma1_hat=math.nan, sigma2_hat=math.nan,
            uncertainty1=0.3, uncertainty2=0.4,
        ),
    ]


def test_schema_columns_frozen():
    # The interface is FROZEN (CLAUDE.md rule 2): this test failing means a
    # schema field was added/renamed/reordered, which is forbidden.
    assert SCHEMA_COLUMNS == EXPECTED_COLUMNS


def test_from_logs_consistency():
    r = _records()[0]
    assert r.I1 == math.exp(r.logI1)
    assert r.log_ratio == r.logI2 - r.logI1
    assert math.isclose(r.ratio, r.I2 / r.I1)


def _assert_records_equal(back, records):
    assert len(back) == len(records)
    for a, b in zip(back, records):
        assert a.image_id == b.image_id and a.spot_id == b.spot_id
        assert a.flags == b.flags
        for col in ("x", "y", "p_detect", "logI1", "logI2", "I1", "I2",
                    "log_ratio", "ratio", "uncertainty1", "uncertainty2"):
            assert math.isclose(getattr(a, col), getattr(b, col), rel_tol=1e-12)
        for col in ("sigma1_hat", "sigma2_hat"):
            va, vb = getattr(a, col), getattr(b, col)
            assert (math.isnan(va) and math.isnan(vb)) or math.isclose(va, vb, rel_tol=1e-12)


def test_dataframe_roundtrip():
    records = _records()
    df = records_to_dataframe(records)
    assert list(df.columns) == list(SCHEMA_COLUMNS)
    _assert_records_equal(dataframe_to_records(df), records)


def test_csv_roundtrip(tmp_path):
    records = _records()
    path = write_spots(records, tmp_path / "spots.csv")
    _assert_records_equal(read_spots(path), records)

"""Preprocessing: leakage, chronological split and history features."""
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (FEATURES, START, TARGET, assert_no_leakage, engineer_features,
                                 fill_missing)


def test_no_leakage_columns_in_features(pipeline, cfg):
    leakage = [c.lower() for c in cfg["columns"]["leakage"]]
    for f in pipeline["meta"]["features"]:
        assert not any(f.lower().startswith(l) for l in leakage)
        assert f != TARGET
    assert_no_leakage(pipeline["meta"]["features"], cfg)
    with pytest.raises(AssertionError):
        assert_no_leakage(FEATURES + ["Charging Time (hh:mm:ss)"], cfg)


def test_chronological_split_has_no_overlap(pipeline):
    tr, va, te = pipeline["train"], pipeline["val"], pipeline["test"]
    assert tr[START].max() <= va[START].min()
    assert va[START].max() <= te[START].min()
    n = len(tr) + len(va) + len(te)
    assert abs(len(tr) / n - 0.70) < 0.01 and abs(len(te) / n - 0.15) < 0.01


def test_period_and_target_cleaning(pipeline):
    full = pd.concat([pipeline["train"], pipeline["val"], pipeline["test"]])
    assert full[START].min() >= pd.Timestamp("2012-01-01")
    assert full[START].max() < pd.Timestamp("2020-01-01")
    assert (full[TARGET] > 0).all()
    assert full[TARGET].max() <= pipeline["info"]["outlier_threshold_kwh"]


def test_first_session_uses_global_train_mean(pipeline):
    gm = pipeline["meta"]["global_train_mean_kwh"]
    full = pd.concat([pipeline["train"], pipeline["val"], pipeline["test"]])
    first = full[full["user_prev_sessions"] == 0]
    assert len(first) > 0
    assert np.allclose(first["user_prev_mean_kwh"], gm)


def test_history_features_use_only_earlier_sessions(cfg):
    c = cfg["columns"]
    df = pd.DataFrame({
        START: pd.to_datetime(["2015-01-03", "2015-01-01", "2015-01-02", "2015-01-04"]),
        TARGET: [30.0, 10.0, 20.0, 5.0],
        c["user"]: ["u1", "u1", "u1", np.nan],
        c["driver_zip"]: ["94301", None, "94301", "94301"],
        c["station"]: ["S", "S", "S", "S"],
        c["port_type"]: "Level 2", c["plug_type"]: "J1772",
        c["port_number"]: 1, c["lat"]: 37.4, c["lon"]: -122.1,
    })
    out, gm = engineer_features(fill_missing(df, cfg), cfg)
    # sorted by time: u1 sessions are 10 (Jan 1), 20 (Jan 2), 30 (Jan 3)
    assert out["user_prev_sessions"].tolist() == [0, 1, 2, 0]
    assert out["user_prev_mean_kwh"].iloc[0] == pytest.approx(gm)
    assert out["user_prev_mean_kwh"].iloc[1] == pytest.approx(10.0)
    assert out["user_prev_mean_kwh"].iloc[2] == pytest.approx(15.0)
    assert out["user_prev_mean_kwh"].iloc[3] == pytest.approx(gm)   # unknown user
    assert out["station_prev_mean_kwh"].iloc[3] == pytest.approx(20.0)
    assert out["driver_zip_known"].tolist() == [0, 1, 1, 1]


def test_rare_categories_mapped_to_other(pipeline):
    levels = pipeline["meta"]["categories"]
    for col, lv in levels.items():
        assert lv[-1] == "other"
        for split in ("train", "val", "test"):
            assert set(pipeline[split][col].unique()) <= set(lv)

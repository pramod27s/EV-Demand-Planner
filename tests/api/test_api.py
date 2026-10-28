"""API tests. Artifacts are produced by running the real pipeline (preprocess -> baselines
-> tiny tuning -> summarize -> export) on synthetic data in a temporary directory."""
from __future__ import annotations

import copy
import os
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from src.config import processed_dir
from src.data.preprocess import run_pipeline, save_outputs

BODY = {"station": None, "port_type": None, "plug_type": None, "port_number": 1,
        "start_datetime": "2019-06-14T09:30:00", "user_prev_sessions": 12, "user_prev_mean_kwh": 8.4}


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory, raw, cfg):
    from src.experiments import export_artifacts, run_baselines, run_tuning, summarize

    root = tmp_path_factory.mktemp("project")
    c = copy.deepcopy(cfg)
    c["paths"] = {"processed_dir": str(root / "processed"), "results_dir": str(root / "results")}
    c["baselines"]["random_forest"]["n_estimators"] = 10
    c["quick"] = {"tuning_sample_size": 500, "pop_size": 4, "max_iter": 2, "seeds": [1, 2]}
    saved = os.environ.pop("RESULTS_DIR", None)
    try:
        save_outputs(run_pipeline(raw, c), processed_dir(c))
        run_baselines.run(c)
        run_tuning.run(c, quick=True)
        summarize.run(c)
        export_artifacts.run(c)
    finally:
        if saved is not None:
            os.environ["RESULTS_DIR"] = saved
    return root / "results"


@pytest.fixture(scope="session")
def client(artifacts_dir):
    with TestClient(create_app(artifacts_dir)) as c:
        yield c


@pytest.fixture(scope="session")
def body(client):
    opts = client.get("/api/options").json()
    return {**BODY, "station": opts["stations"][0], "port_type": opts["port_types"][0],
            "plug_type": opts["plug_types"][0]}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["artifacts_ok"] is True
    assert set(j["models_loaded"]) == {"default", "sma", "pso", "ga"}


def test_options_non_empty(client):
    j = client.get("/api/options").json()
    assert j["stations"] and j["port_types"] and j["plug_types"]
    assert "other" not in j["port_types"]


def test_predict_returns_four_positive_predictions(client, body):
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["predictions"]) == 4
    assert all(p["kwh"] > 0 for p in j["predictions"])
    assert {p["model"] for p in j["predictions"]} == set(j["test_rmse"])
    assert j["features_used"]["hour"] == 9 and j["features_used"]["dayofweek"] == 4


def test_predict_without_user_mean_uses_global_mean(client, body):
    b = {**body, "user_prev_sessions": 0}
    b.pop("user_prev_mean_kwh")
    j = client.post("/api/predict", json=b).json()
    assert j["features_used"]["user_prev_mean_kwh"] > 0


def test_predict_unknown_station_is_422(client, body):
    r = client.post("/api/predict", json={**body, "station": "NOWHERE #9"})
    assert r.status_code == 422
    assert body["station"] in r.json()["detail"]["allowed"]


def test_predict_validation_errors(client, body):
    assert client.post("/api/predict", json={**body, "user_prev_sessions": -1}).status_code == 422
    assert client.post("/api/predict", json={**body, "user_prev_mean_kwh": 150}).status_code == 422


def test_stations(client, body):
    stations = client.get("/api/stations").json()
    assert stations and {"name", "lat", "lon", "session_count", "mean_predicted_kwh"} <= set(stations[0])
    one = client.get(f"/api/stations/{quote(body['station'], safe='')}")
    assert one.status_code == 200
    assert len(one.json()["hourly_mean_kwh"]) == 24
    assert client.get("/api/stations/NOPE").status_code == 404


def test_results_endpoints(client):
    assert {"steps", "splits", "n_sessions"} <= set(client.get("/api/dataset/summary").json())
    rows = client.get("/api/results/comparison").json()["rows"]
    assert {"XGBoost (default)", "XGBoost + SMA"} <= {r["model"] for r in rows}
    conv = client.get("/api/results/convergence").json()
    assert {"sma", "pso", "ga", "random"} <= set(conv)
    assert {"mean", "std", "seeds"} <= set(conv["sma"])
    assert set(client.get("/api/results/convergence", params={"optimizer": "sma"}).json()) == {"sma"}
    hp = client.get("/api/results/hyperparams").json()
    assert "n_estimators" in hp["sma"]["params"]
    ts = client.get("/api/results/test-sample", params={"model": "default"}).json()
    assert len(ts["actual"]) == len(ts["predicted"]) > 0
    fi = client.get("/api/results/feature-importance").json()
    assert fi and {"feature", "gain"} <= set(fi[0])


def test_optimization_history(client):
    runs = client.get("/api/optimization/runs").json()
    assert runs["sma"] == [1, 2]
    h = client.get("/api/optimization/history", params={"optimizer": "sma", "seed": 1}).json()
    assert len(h["history"]) == 2 and "learning_rate" in h["history"][0]["best_params"]
    assert client.get("/api/optimization/history", params={"optimizer": "sma", "seed": 99}).status_code == 404


def test_area_analysis_with_sample(client):
    csv = client.get("/api/area/sample").text
    assert csv.startswith("Station Name,Latitude,Longitude,Start Date")
    r = client.post("/api/area/analyze", params={"k": 2}, content=csv, headers={"Content-Type": "text/csv"})
    assert r.status_code == 200, r.text
    j = r.json()
    s = j["summary"]
    assert s["rows_used"] > 0 and s["has_actual"] and "rmse" in s["accuracy"]
    assert not s["far_from_training"]
    assert j["locations"] and len(j["hourly"]) == 24
    assert len(j["placement"]["sites"]) == 2
    assert sum(site["share"] for site in j["placement"]["sites"]) == pytest.approx(1)


def test_area_analysis_minimal_columns_far_away(client):
    csv = ("timestamp,lat,lng\n2024-03-04 08:15,12.9756,77.6066\n2024-03-04 18:40,12.9784,77.6408\n"
           "2024-03-05 12:05,12.9716,77.5946\nnot a date,12.97,77.59\n")
    r = client.post("/api/area/analyze", params={"k": 5}, content=csv, headers={"Content-Type": "text/csv"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["summary"]["rows_dropped"] == 1 and j["summary"]["far_from_training"]
    assert j["placement"]["k"] == 3                       # capped at the number of locations
    assert all(loc["mean_kwh"] > 0 for loc in j["locations"])


def test_area_bad_files(client):
    r = client.post("/api/area/analyze", content="a,b\n1,2\n", headers={"Content-Type": "text/csv"})
    assert r.status_code == 422 and "Missing required column" in r.json()["detail"]
    assert client.post("/api/area/analyze", content="", headers={"Content-Type": "text/csv"}).status_code == 422


def test_area_predictions_csv_and_template(client):
    csv = client.get("/api/area/sample").text
    r = client.post("/api/area/predictions", content=csv, headers={"Content-Type": "text/csv"})
    assert r.status_code == 200
    header = r.text.splitlines()[0].split(",")
    assert header[-1] == "predicted_kwh" and len(r.text.splitlines()) == len(csv.splitlines())
    assert client.get("/api/area/template").text.startswith("Station Name,Latitude")


def test_area_estimate_any_location(client):
    body = {"name": "MG ROAD #1", "lat": 12.9756, "lon": 77.6066, "location_mean_kwh": 8.1,
            "start_datetime": "2024-03-04T08:15:00", "user_prev_sessions": 0}
    r = client.post("/api/area/estimate", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["kwh"] > 0 and j["low_kwh"] <= j["kwh"] <= j["high_kwh"]
    assert client.post("/api/area/estimate", json={**body, "lat": 123}).status_code == 422


def test_area_demand_file(client):
    # measured kWh per locality and month (like Hyderabad's utility data): no start times
    csv = ("Location,Latitude,Longitude,Month,Energy (kWh)\n"
           "Madhapur,17.4483,78.3915,2024-01,15728\nMadhapur,17.4483,78.3915,2024-02,16010\n"
           "Film Nagar,17.4126,78.4071,2024-01,810\nGachibowli,17.4401,78.3489,2024-02,9200\n")
    r = client.post("/api/area/analyze", params={"k": 2}, content=csv, headers={"Content-Type": "text/csv"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["summary"]["mode"] == "demand" and j["summary"]["months"] == 2
    assert j["summary"]["total_kwh"] == pytest.approx(15728 + 16010 + 810 + 9200)
    assert j["locations"][0]["name"] == "Madhapur" and len(j["locations"][0]["monthly_kwh"]) == 2
    assert [m["month"] for m in j["monthly"]] == ["2024-01", "2024-02"]
    assert len(j["placement"]["sites"]) == 2
    # demand files have no sessions, so there is nothing to download as estimates
    assert client.post("/api/area/predictions", content=csv, headers={"Content-Type": "text/csv"}).status_code == 422


def test_area_samples(client):
    listed = {s["id"]: s for s in client.get("/api/area/samples").json()}
    assert "palo_alto" in listed and listed["palo_alto"]["kind"] == "sessions"
    assert client.get("/api/area/samples/palo_alto").text.startswith("Station Name,Latitude")
    assert client.get("/api/area/samples/atlantis").status_code == 404


def test_503_when_artifacts_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    with TestClient(create_app()) as c:
        h = c.get("/api/health").json()
        assert h["artifacts_ok"] is False
        for path in ("/api/options", "/api/stations", "/api/results/comparison"):
            r = c.get(path)
            assert r.status_code == 503
            assert "python -m src.experiments.export_artifacts" in r.json()["detail"]

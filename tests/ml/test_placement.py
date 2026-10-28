"""Charger placement (weighted p-median solved by SMA): src/placement."""
import numpy as np
import pytest

from src.placement import PlacementProblem, haversine_km, solve_placement


def test_haversine_known_distance():
    # 1 degree of latitude is ~111.2 km
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, rel=1e-3)
    assert haversine_km(37.44, -122.16, 37.44, -122.16) == pytest.approx(0)


def test_objective_is_zero_when_sites_sit_on_the_only_locations():
    prob = PlacementProblem([37.40, 37.45], [-122.20, -122.10], [1, 1], k=2)
    # encode the two locations exactly as sites
    la0, la1, lo0, lo1 = prob.box
    u = [(37.40 - la0) / (la1 - la0), (-122.20 - lo0) / (lo1 - lo0),
         (37.45 - la0) / (la1 - la0), (-122.10 - lo0) / (lo1 - lo0)]
    assert prob(np.array(u)) == pytest.approx(0, abs=1e-6)


def test_sma_finds_two_demand_clusters():
    rng = np.random.default_rng(0)
    lat = np.r_[rng.normal(37.44, 0.002, 40), rng.normal(37.42, 0.002, 40)]
    lon = np.r_[rng.normal(-122.16, 0.002, 40), rng.normal(-122.13, 0.002, 40)]
    res = solve_placement(lat, lon, np.ones(80), k=2, pop_size=30, max_iter=100, seed=1)
    centres = sorted((round(s["lat"], 2), round(s["lon"], 2)) for s in res["sites"])
    assert centres == [(37.42, -122.13), (37.44, -122.16)]
    assert sum(s["served_locations"] for s in res["sites"]) == 80
    assert sum(s["share"] for s in res["sites"]) == pytest.approx(1)
    assert all(b <= a for a, b in zip(res["convergence_km"], res["convergence_km"][1:]))


def test_heavier_demand_pulls_the_single_site():
    lat, lon = [37.40, 37.50], [-122.10, -122.10]
    res = solve_placement(lat, lon, [10, 1], k=1, pop_size=20, max_iter=60, seed=1)
    assert abs(res["sites"][0]["lat"] - 37.40) < abs(res["sites"][0]["lat"] - 37.50)

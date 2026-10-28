"""Charger placement as a weighted p-median problem, solved with the Slime Mould Algorithm.

Given demand locations ``(lat_i, lon_i)`` with weights ``w_i`` (estimated kWh drawn there),
choose ``k`` sites that minimise the demand-weighted mean distance from each location to
its nearest site:

    minimise  F(S) = Σ_i w_i · min_j d(loc_i, site_j) / Σ_i w_i

``d`` is the haversine (great-circle) distance in km. A candidate solution is a vector
``u ∈ [0,1]^(2k)`` decoded to ``k`` (lat, lon) pairs inside the locations' bounding box,
so the same from-scratch SMA used for hyperparameter tuning can search it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.optimizers.sma import SMA

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (inputs in degrees, broadcastable)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class PlacementProblem:
    lat: np.ndarray
    lon: np.ndarray
    weight: np.ndarray
    k: int

    def __post_init__(self):
        self.lat, self.lon = np.asarray(self.lat, float), np.asarray(self.lon, float)
        self.weight = np.asarray(self.weight, float)
        pad_lat = max(1e-3, 0.05 * (self.lat.max() - self.lat.min()))
        pad_lon = max(1e-3, 0.05 * (self.lon.max() - self.lon.min()))
        self.box = (self.lat.min() - pad_lat, self.lat.max() + pad_lat,
                    self.lon.min() - pad_lon, self.lon.max() + pad_lon)

    @property
    def dim(self) -> int:
        return 2 * self.k

    def decode(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u = [lat_1, lon_1, ..., lat_k, lon_k] in [0,1] -> site coordinates in the bounding box."""
        u = np.asarray(u, float).reshape(self.k, 2)
        la0, la1, lo0, lo1 = self.box
        return la0 + u[:, 0] * (la1 - la0), lo0 + u[:, 1] * (lo1 - lo0)

    def distances(self, site_lat, site_lon) -> np.ndarray:
        """N x k matrix of distances (km) from every location to every site."""
        return haversine_km(self.lat[:, None], self.lon[:, None], site_lat[None, :], site_lon[None, :])

    def __call__(self, u: np.ndarray) -> float:
        d = self.distances(*self.decode(u)).min(axis=1)
        return float(np.sum(self.weight * d) / np.sum(self.weight))


def solve_placement(lat, lon, weight, k: int, pop_size: int, max_iter: int, seed: int, z: float = 0.03) -> dict:
    """Run SMA on the placement problem; return the sites, what each serves and the convergence."""
    prob = PlacementProblem(lat, lon, weight, k)
    res = SMA(pop_size=pop_size, max_iter=max_iter, seed=seed, z=z).optimize(prob, prob.dim)
    site_lat, site_lon = prob.decode(res.best_position)
    d = prob.distances(site_lat, site_lon)
    nearest = d.argmin(axis=1)
    total = prob.weight.sum()
    sites = []
    for j in range(k):
        m = nearest == j
        w = prob.weight[m]
        sites.append({
            "rank": 0,
            "lat": float(site_lat[j]), "lon": float(site_lon[j]),
            "served_kwh": float(w.sum()), "share": float(w.sum() / total) if total else 0.0,
            "served_locations": int(m.sum()),
            "mean_distance_km": float(np.sum(w * d[m, j]) / w.sum()) if w.sum() else 0.0,
        })
    sites.sort(key=lambda s: s["served_kwh"], reverse=True)
    for i, s in enumerate(sites, start=1):
        s["rank"] = i
    return {
        "sites": sites,
        "objective_km": res.best_fitness,
        "convergence_km": res.convergence,
        "evaluations": res.evaluations,
        "assignment": nearest.tolist(),
    }

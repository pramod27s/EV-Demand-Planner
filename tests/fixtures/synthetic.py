"""Synthetic stand-in for the Palo Alto CSV — used ONLY by the automated tests.

It has the same column names as the real file (including the leakage columns), so the
preprocessing, model and API code can be tested without the real dataset. It is never
used to produce reported results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATIONS = [
    ("PALO ALTO CA / HAMILTON #1", 37.4446, -122.1600, "Level 2", "J1772"),
    ("PALO ALTO CA / HAMILTON #2", 37.4447, -122.1602, "Level 2", "J1772"),
    ("PALO ALTO CA / BRYANT #1", 37.4465, -122.1625, "Level 2", "J1772"),
    ("PALO ALTO CA / HIGH #1", 37.4440, -122.1630, "Level 2", "J1772"),
    ("PALO ALTO CA / WEBSTER #1", 37.4483, -122.1588, "Level 1", "NEMA 5-20R"),
    ("PALO ALTO CA / TED THOMPSON #3", 37.4280, -122.1430, "Level 2", "J1772"),
]


def make_raw(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Generate ``n`` fake charging sessions between mid-2011 and 2020."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2011-08-01") + pd.to_timedelta(
        np.sort(rng.uniform(0, 9.3 * 365, n)), unit="D")
    start = start.floor("min")
    st_idx = rng.integers(0, len(STATIONS), n)
    users = rng.integers(0, 150, n).astype(float)
    users[rng.random(n) < 0.05] = np.nan
    user_bias = rng.normal(0, 2, 150)
    hour = start.hour.to_numpy()
    level1 = np.array([STATIONS[i][3] == "Level 1" for i in st_idx])
    energy = (6 + 3 * np.sin(2 * np.pi * hour / 24) - 3 * level1
              + np.where(np.isnan(users), 0, user_bias[np.nan_to_num(users).astype(int)])
              + rng.gamma(2, 1.5, n))
    energy = np.clip(energy, 0.05, None)
    energy[rng.random(n) < 0.01] = 0          # invalid rows the pipeline must drop
    duration_h = energy / 6.6 + rng.uniform(0, 2, n)
    end = start + pd.to_timedelta(duration_h, unit="h")

    def hms(hours: np.ndarray) -> list[str]:
        secs = (hours * 3600).astype(int)
        return [f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" for s in secs]

    zips = rng.choice(["94301", "94303", "94040", None], n, p=[0.4, 0.3, 0.2, 0.1])
    return pd.DataFrame({
        "Station Name": [STATIONS[i][0] for i in st_idx],
        "MAC Address": "0024:B100:0002:XXXX",
        "Org Name": "City of Palo Alto",
        "Start Date": start.strftime("%m/%d/%Y %H:%M"),
        "Start Time Zone": "PDT",
        "End Date": end.strftime("%m/%d/%Y %H:%M"),
        "End Time Zone": "PDT",
        "Transaction Date (Pacific Time)": end.strftime("%m/%d/%Y %H:%M"),
        "Total Duration (hh:mm:ss)": hms(duration_h),
        "Charging Time (hh:mm:ss)": hms(energy / 6.6),
        "Energy (kWh)": np.round(energy, 3),
        "GHG Savings (kg)": np.round(energy * 0.42, 3),
        "Gasoline Savings (gallons)": np.round(energy * 0.125, 3),
        "Port Type": [STATIONS[i][3] for i in st_idx],
        "Port Number": rng.integers(1, 3, n),
        "Plug Type": [STATIONS[i][4] for i in st_idx],
        "EVSE ID": st_idx + 1000,
        "Address 1": "250 Hamilton Ave",
        "City": "Palo Alto",
        "State/Province": "California",
        "Postal Code": 94301,
        "Country": "United States",
        "Latitude": [STATIONS[i][1] for i in st_idx],
        "Longitude": [STATIONS[i][2] for i in st_idx],
        "Currency": "USD",
        "Fee": np.round(energy * 0.23, 2),
        "Ended By": rng.choice(["Plug Out at Vehicle", "Customer"], n),
        "Plug In Event Id": np.arange(n),
        "Driver Postal Code": zips,
        "User ID": users,
        "County": "Santa Clara County",
        "System S/N": 123456,
        "Model Number": "CT4020-HD",
    })

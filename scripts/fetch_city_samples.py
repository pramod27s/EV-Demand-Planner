"""Download public EV charging-session datasets from other cities into ``data/sample/``.

Each city gets a folder (e.g. ``boulder_co/``) with files named after the city:
  <city>_sessions.csv   Workspace-ready sessions (Station Name, Latitude, Longitude, Start Date,
                 Port Type, User ID, Energy (kWh)) — upload it on the Workspace page
  <city>_stations.csv   every station with its geocoded coordinates
  SOURCE.md             where the data comes from
Hyderabad publishes monthly kWh per supply area instead of sessions, so it gets
<city>_demand.csv (Location, Latitude, Longitude, Month, Energy (kWh), ...)., its licence and how it was processed

The source files have addresses or site names but no coordinates, so stations are geocoded
with OpenStreetMap Nominatim (one request per second, cached in data/sample/geocode_cache.json).
This is a one-off helper; the ML pipeline and the web app never download anything.

    python scripts/fetch_city_samples.py              # all cities
    python scripts/fetch_city_samples.py cary_nc      # one city
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample"
CACHE = OUT / "geocode_cache.json"
UA = {"User-Agent": "sma-ev-demand-student-project/1.0 (BMSCE Bio-Inspired Algorithms coursework)"}
COLUMNS = ["Station Name", "Latitude", "Longitude", "Start Date", "Port Type", "User ID", "Energy (kWh)"]


# ---------------------------------------------------------------------------------
# Geocoding (OpenStreetMap Nominatim)
# ---------------------------------------------------------------------------------
_cache: dict[str, list[float] | None] = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def geocode(query: str) -> tuple[float, float] | None:
    if query in _cache:
        return tuple(_cache[query]) if _cache[query] else None
    time.sleep(1.1)  # Nominatim usage policy: at most 1 request per second
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": query, "format": "json", "limit": 1}, headers=UA, timeout=30)
    r.raise_for_status()
    hits = r.json()
    _cache[query] = [float(hits[0]["lat"]), float(hits[0]["lon"])] if hits else None
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(_cache, indent=1))
    return tuple(_cache[query]) if _cache[query] else None


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def attach_coordinates(df: pd.DataFrame, key: str, queries, centre: tuple[float, float],
                       max_km: float = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Geocode each distinct ``key`` value; drop sessions whose station could not be located.

    A hit further than ``max_km`` from the city centre is rejected (a same-named place elsewhere).
    """
    stations = df.groupby(key).size().rename("sessions").reset_index()
    coords = []
    for _, row in stations.merge(df.drop_duplicates(key), on=key).iterrows():
        hit = None
        for q in queries(row):
            cand = geocode(q)
            if cand and km(cand, centre) <= max_km:
                hit = cand
                break
        coords.append(hit)
    stations["Latitude"] = [c[0] if c else None for c in coords]
    stations["Longitude"] = [c[1] if c else None for c in coords]
    missing = stations[stations["Latitude"].isna()]
    if len(missing):
        print(f"    could not geocode {len(missing)} station(s): {missing[key].tolist()}")
    out = df.merge(stations[[key, "Latitude", "Longitude"]], on=key, how="left")
    return out[out["Latitude"].notna()], stations


# ---------------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------------
def boulder() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    url = ("https://services.arcgis.com/ePKBjXrBZ2vEEgWd/arcgis/rest/services/"
           "Electric_Vehicle_Charging_Station_Data/FeatureServer/0/query")
    rows, offset = [], 0
    while True:  # the feature service pages its results (max 1,000-2,000 per request)
        r = requests.get(url, params={"where": "1=1", "outFields": "*", "f": "json", "resultOffset": offset,
                                      "resultRecordCount": 2000, "orderByFields": "ObjectId2"}, headers=UA, timeout=60)
        r.raise_for_status()
        feats = r.json().get("features", [])
        if not feats:
            break
        rows += [f["attributes"] for f in feats]
        offset += len(feats)
        print(f"    {offset:,} records downloaded", flush=True)  # progress
    raw = pd.DataFrame(rows)
    df = pd.DataFrame({
        "Station Name": raw["Station_Name"].str.strip(),
        "Address": raw["Address"].str.strip(), "Zip": raw["Zip_Postal_Code"].astype(str),
        "Start Date": pd.to_datetime(raw["Start_Date___Time"], format="%m/%d/%Y %H:%M", errors="coerce"),
        "Port Type": raw["Port_Type"], "User ID": None,
        "Energy (kWh)": pd.to_numeric(raw["Energy__kWh_"], errors="coerce"),
    })
    df, stations = attach_coordinates(df, "Address", lambda r: [
        f"{r['Address']}, Boulder, Colorado {r['Zip']}, USA", f"{r['Address']}, Boulder, Colorado, USA"],
        centre=(40.015, -105.270))
    meta = {
        "title": "Boulder, Colorado (USA)", "publisher": "City of Boulder",
        "source": "https://open-data.bouldercolorado.gov/ — 'Electric Vehicle Charging Station Data' (ArcGIS feature service)",
        "licence": "City of Boulder open data terms (public open data portal)",
        "notes": "Same ChargePoint export format as the Palo Alto training data. Times are local (Mountain Time).",
        "raw_rows": len(raw),
    }
    return df, stations, meta


def cary() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    url = "https://data.townofcary.org/api/explore/v2.1/catalog/datasets/electric-vehicle-charging-stations/exports/csv"
    r = requests.get(url, params={"delimiter": ","}, headers=UA, timeout=120)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    start = pd.to_datetime(raw["start_date"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = pd.DataFrame({
        "Station Name": raw["station_name"].str.strip(), "Address": raw["address_1"].str.strip(),
        "Zip": raw["zip_postal_code"].astype(str), "Start Date": start, "Port Type": "Level 2", "User ID": None,
        "Energy (kWh)": pd.to_numeric(raw["energy_kwh"], errors="coerce"),
    })
    df, stations = attach_coordinates(df, "Address", lambda r: [
        f"{r['Address']}, Cary, North Carolina {r['Zip']}, USA", f"{r['Address']}, Cary, North Carolina, USA"],
        centre=(35.792, -78.781))
    meta = {
        "title": "Cary, North Carolina (USA)", "publisher": "Town of Cary",
        "source": "https://data.townofcary.org/explore/dataset/electric-vehicle-charging-stations/",
        "licence": "CC0 1.0 Universal (public domain)",
        "notes": "Start times converted from UTC to local Eastern Time. Port type not published; all town chargers are Level 2.",
        "raw_rows": len(raw),
    }
    return df, stations, meta


LEEDS_FILES = [  # Data Mill North, "Electric vehicle chargepoints operated by Leeds City Council"
    "https://datamillnorth.org/download/2lx68/5e6417b4-27a9-41fb-9505-509934e4dc0d/2019%20EVCP%20use%20Q1%20and%20Q2.csv",
    "https://datamillnorth.org/download/2lx68/bffbd0db-90b5-429a-9259-72cb90726da4/2019%20EVCP%20use%20Q3%20and%20Q4.csv",
    "https://datamillnorth.org/download/2lx68/9541db2e-c3e4-4f75-8f5a-8100054df0f6/2020%20EVCP%20use%20Q1%20and%20Q2.csv",
    "https://datamillnorth.org/download/2lx68/cef9ca1b-2b6c-4dd3-a5dd-09ea3e8ac4db/2020%20EVCP%20use%20Q3.csv",
    "https://datamillnorth.org/download/2lx68/b8b07a16-f62a-4b54-8ae9-54cfe7b8681c/2020%20EVCP%20use%20Q4.csv",
    "https://datamillnorth.org/download/2lx68/fc2f9674-6952-43c8-be63-e18b4a26047e/2021%20EVCP%20use%20Q1.csv",
]


def leeds() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    parts = []
    for url in LEEDS_FILES:
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        part = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", low_memory=False)
        part.columns = [str(c).strip() for c in part.columns]
        parts.append(part.loc[:, ~part.columns.str.startswith("Unnamed")])
    raw = pd.concat(parts, ignore_index=True)
    start = pd.to_datetime(raw["Start Date"].astype(str) + " " + raw["Start Time"].astype(str),
                           format="%d/%m/%Y %H:%M", errors="coerce")
    model = raw.get("Model", pd.Series("", index=raw.index)).astype(str)
    port = model.str.contains(r"50kW|Rapid|DC", case=False, regex=True).map({True: "DC Fast", False: "Level 2"})
    df = pd.DataFrame({
        "Station Name": raw["Site"].astype(str).str.strip() + " #" + raw["CP ID"].astype(str),
        "Site": raw["Site"].astype(str).str.strip(), "Start Date": start, "Port Type": port,
        "User ID": raw["Unique User ID"].astype(str), "Energy (kWh)": pd.to_numeric(raw["Total kWh"], errors="coerce"),
    })
    df = df[~df["Site"].str.contains("TEST SITE", case=False)]          # operator's test charger
    df, stations = attach_coordinates(df, "Site", lambda r: [
        f"{r['Site']}, Leeds, UK", f"{r['Site'].replace(' Car Park', '')}, Leeds, UK",
        f"{r['Site'].split(',')[0]}, Leeds, UK", f"{r['Site'].replace(' and ', ' & ')}, Leeds, UK",
        f"{r['Site'].replace(' Park and Ride', '')}, Leeds, UK"],
        centre=(53.800, -1.549))
    meta = {
        "title": "Leeds (UK)", "publisher": "Leeds City Council (via Data Mill North)",
        "source": "https://datamillnorth.org/dataset/electric-vehicle-chargepoints-operated-by-leeds-city-council",
        "licence": "Open Government Licence v3.0",
        "notes": "Quarterly files 2019 Q1 – 2021 Q1 combined. Station = site + charge point ID; rapid (50 kW) units marked 'DC Fast'.",
        "raw_rows": len(raw),
    }
    return df, stations, meta


OPENCITY_API = "https://data.opencity.in/api/3/action/package_search"
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], start=1)}


def hyderabad() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Monthly kWh drawn by EV-charging connections per supply area (TSSPDCL, via OpenCity).

    There are no individual sessions, so the output is a *demand* file: one row per
    locality and month. Localities ("sections") are geocoded; their supply areas are summed.
    """
    r = requests.get(OPENCITY_API, params={"q": "Hyderabad EV Charging Stations Monthly Consumption", "rows": 20},
                     headers=UA, timeout=60)
    r.raise_for_status()
    parts = []
    for pkg in r.json()["result"]["results"]:
        if not pkg["title"].startswith("Hyderabad EV Charging Stations Monthly Consumption"):
            continue
        for res in pkg["resources"]:
            name = (res.get("name") or "").lower()
            words = name.replace("-", " ").split()
            month = next((MONTHS[w] for w in words if w in MONTHS), None)
            year = next((int(w) for w in words if w.isdigit() and len(w) == 4), None)
            if not (month and year) or (res.get("format") or "").upper() != "CSV":
                continue
            f = requests.get(res["url"], headers=UA, timeout=120)
            f.raise_for_status()
            part = pd.read_csv(io.BytesIO(f.content))
            part.columns = [c.strip().lower() for c in part.columns]
            part["month"] = f"{year}-{month:02d}"
            parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    raw["units"] = pd.to_numeric(raw["units"], errors="coerce")
    raw = raw[raw["units"] > 0]
    for c in ("section", "subdivision", "division", "area"):
        raw[c] = raw[c].astype(str).str.strip().str.title()

    df = raw.groupby(["section", "subdivision", "month"], as_index=False).agg(
        units=("units", "sum"), areas=("area", "nunique"), load=("load", "sum"))
    df, stations = attach_coordinates(df, "section", lambda r: [
        f"{r['section']}, Hyderabad, Telangana, India", f"{r['subdivision']}, Hyderabad, Telangana, India"],
        centre=(17.385, 78.487), max_km=45)
    df = df.rename(columns={"section": "Location", "units": "Energy (kWh)", "month": "Month",
                            "areas": "Supply areas", "load": "Connected load (kW)"})
    meta = {
        "title": "Hyderabad, Telangana (India)", "publisher": "TSSPDCL (Southern Power Distribution Co. of Telangana), via OpenCity",
        "source": "https://data.opencity.in/ — 'Hyderabad EV Charging Stations Monthly Consumption Data' (2021-2024)",
        "licence": "Public domain (as listed on OpenCity)",
        "notes": ("Monthly electricity units (kWh) billed to EV-charging connections per supply area. There are no "
                  "individual sessions, so this is a demand file: the Workspace shows the demand map, monthly trend and "
                  "SMA-suggested charger sites, but not per-session estimates. Supply areas are summed per locality "
                  "(section) and placed at the locality's geocoded centre."),
        "raw_rows": len(raw),
    }
    return df, stations, meta


def save_demand(key: str, df: pd.DataFrame, stations: pd.DataFrame, meta: dict) -> dict:
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    name = CITY_FILE_PREFIX[key]
    cols = ["Location", "Latitude", "Longitude", "Month", "Energy (kWh)", "Supply areas", "Connected load (kW)"]
    df = df.sort_values(["Month", "Location"])
    df[cols].to_csv(d / f"{name}_demand.csv", index=False)
    stations.to_csv(d / f"{name}_locations.csv", index=False)
    info = {**meta, "sessions": len(df), "stations": int(df["Location"].nunique()),
            "start": df["Month"].min(), "end": df["Month"].max()}
    lines = [
        f"# {info['title']}", "",
        "| | |", "|---|---|",
        f"| Publisher | {info['publisher']} |", f"| Source | {info['source']} |",
        f"| Licence | {info['licence']} |", f"| Retrieved | {date.today().isoformat()} |",
        f"| Rows | {info['sessions']:,} locality-months (from {info['raw_rows']:,} supply-area rows) |",
        f"| Localities | {info['stations']} |", f"| Period | {info['start']} to {info['end']} |", "",
        info["notes"], "",
        f"Locality coordinates were geocoded with OpenStreetMap Nominatim (see `{name}_locations.csv`); "
        "localities that could not be located were dropped.",
        f"Regenerate with `python scripts/fetch_city_samples.py {key}`.", "",
    ]
    (d / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")
    return info


BENGALURU_BBOX = (12.80, 77.40, 13.20, 77.85)   # south, west, north, east


def bengaluru() -> dict:
    """Reference data only: Bengaluru publishes no per-location charging demand.

    * BESCOM city-wide yearly EV-charging consumption (OpenCity, public domain)
    * public charging-station locations mapped in OpenStreetMap (ODbL)
    """
    d = OUT / "bengaluru_in"
    d.mkdir(parents=True, exist_ok=True)

    # ---- BESCOM yearly totals ----
    r = requests.get(OPENCITY_API, params={"q": "BESCOM EV Consumption Data", "rows": 5}, headers=UA, timeout=60)
    r.raise_for_status()
    pkg = next(x for x in r.json()["result"]["results"] if x["title"].startswith("BESCOM EV Consumption"))
    res = next(x for x in pkg["resources"] if (x.get("format") or "").upper() == "CSV")
    raw = requests.get(res["url"], headers=UA, timeout=60)
    raw.raise_for_status()
    (d / "bengaluru_bescom_raw.csv").write_bytes(raw.content)
    t = pd.read_csv(io.BytesIO(raw.content)).set_index("Headers").T
    t.index.name = "Financial year"
    names = {"LT EV Charging Stations": "LT stations", "LT Units in KWH": "LT energy (kWh)",
             "HT EV Charging Stations": "HT stations", "HT Units Sold in KWH": "HT energy (kWh)"}
    yearly = t[[c for c in names if c in t.columns]].rename(columns=names)
    yearly = yearly.apply(pd.to_numeric, errors="coerce").sort_index()
    yearly.to_csv(d / "bengaluru_bescom_yearly.csv")

    # ---- OpenStreetMap charging stations ----
    s_, w, n, e = BENGALURU_BBOX
    query = (f'[out:json][timeout:60];(node["amenity"="charging_station"]({s_},{w},{n},{e});'
             f'way["amenity"="charging_station"]({s_},{w},{n},{e}););out center tags;')
    o = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, headers=UA, timeout=120)
    o.raise_for_status()
    rows = []
    for el in o.json()["elements"]:
        tags = el.get("tags", {})
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        rows.append({"Name": tags.get("name", ""), "Operator": tags.get("operator", ""), "Latitude": lat, "Longitude": lon,
                     "Capacity": tags.get("capacity", ""), "Address": ", ".join(
                         v for k, v in tags.items() if k in ("addr:street", "addr:suburb", "addr:city") and v),
                     "OSM id": f"{el['type']}/{el['id']}"})
    stations = pd.DataFrame(rows).sort_values(["Operator", "Name"])
    stations.to_csv(d / "bengaluru_stations_osm.csv", index=False)

    lines = [
        "# Bengaluru, Karnataka (India) — reference data", "",
        "Bengaluru does **not** publish per-station or per-session EV-charging data, so this folder is reference",
        "material for the report. It cannot be analysed in the Workspace, which needs energy per location.", "",
        "| File | What it is | Source | Licence |", "|---|---|---|---|",
        "| `bengaluru_bescom_yearly.csv` | City-wide yearly EV-charging stations and energy (LT and HT connections) | "
        "BESCOM via OpenCity — 'BESCOM EV Consumption Data' | Public domain (as listed on OpenCity) |",
        "| `bengaluru_bescom_raw.csv` | The same file exactly as published | as above | as above |",
        f"| `bengaluru_stations_osm.csv` | {len(stations)} public charging stations mapped in OpenStreetMap "
        "(name, operator, coordinates) | OpenStreetMap via the Overpass API | ODbL 1.0 — © OpenStreetMap contributors |", "",
        f"Retrieved {date.today().isoformat()}. Regenerate with `python scripts/fetch_city_samples.py bengaluru_in`.", "",
        "## Yearly totals (BESCOM)", "",
        "| Year | LT stations | LT energy (kWh) | HT stations | HT energy (kWh) |", "|---|---|---|---|---|",
    ]
    for year, row in yearly.iterrows():
        cells = [f"{row.get(c):,.0f}" if pd.notna(row.get(c)) else "—"
                 for c in ("LT stations", "LT energy (kWh)", "HT stations", "HT energy (kWh)")]
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    lines += ["", "## Caveats",
              "* OpenStreetMap is volunteer-mapped and far from complete: Bengaluru has about 4,400 public charging",
              "  stations, but only the ones listed here are mapped. There is no usage data in OpenStreetMap.",
              "* Operators (Tata Power, Bolt.Earth, ...) keep session-level data private.", ""]
    (d / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")
    return {"sessions": len(stations), "stations": len(stations), "start": str(yearly.index.min()), "end": str(yearly.index.max())}


CITIES = {"boulder_co": boulder, "cary_nc": cary, "leeds_uk": leeds, "hyderabad_in": hyderabad}
CITY_FILE_PREFIX = {"boulder_co": "boulder", "cary_nc": "cary", "leeds_uk": "leeds", "hyderabad_in": "hyderabad"}
DEMAND_CITIES = {"hyderabad_in"}   # monthly demand files, not sessions
REFERENCE_CITIES = {"bengaluru_in": bengaluru}   # reference material only (no demand per location)


# ---------------------------------------------------------------------------------
def save(key: str, df: pd.DataFrame, stations: pd.DataFrame, meta: dict) -> dict:
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    name = CITY_FILE_PREFIX[key]
    df = df[df["Start Date"].notna() & (df["Energy (kWh)"] > 0)].sort_values("Start Date")
    df.assign(**{"Start Date": df["Start Date"].dt.strftime("%Y-%m-%d %H:%M")})[COLUMNS].to_csv(d / f"{name}_sessions.csv", index=False)
    stations.to_csv(d / f"{name}_stations.csv", index=False)
    info = {**meta, "sessions": len(df), "stations": int(df["Station Name"].nunique()),
            "start": str(df["Start Date"].min())[:10], "end": str(df["Start Date"].max())[:10]}
    (d / "SOURCE.md").write_text(
        f"# {info['title']}\n\n"
        f"| | |\n|---|---|\n| Publisher | {info['publisher']} |\n| Source | {info['source']} |\n"
        f"| Licence | {info['licence']} |\n| Retrieved | {date.today().isoformat()} |\n"
        f"| Sessions | {info['sessions']:,} (of {info['raw_rows']:,} raw rows) |\n| Stations | {info['stations']} |\n"
        f"| Period | {info['start']} → {info['end']} |\n\n"
        f"{info['notes']}\n\nStation coordinates were geocoded from addresses/site names with OpenStreetMap Nominatim "
        f"(see `{name}_stations.csv`); rows without a valid time, a positive energy value or a locatable station were dropped.\n"
        f"Regenerate with `python scripts/fetch_city_samples.py {key}`.\n", encoding="utf-8")
    return info


def main(argv: list[str]) -> None:
    keys = argv or [*CITIES, *REFERENCE_CITIES]
    summary = []
    for key in keys:
        print(f"[{key}] downloading ...")
        if key in REFERENCE_CITIES:
            info = REFERENCE_CITIES[key]()
        else:
            info = (save_demand if key in DEMAND_CITIES else save)(key, *CITIES[key]())
        print(f"    {info['sessions']:,} sessions, {info['stations']} stations, {info['start']} to {info['end']}")
        summary.append((key, info))
    print("done:", ", ".join(k for k, _ in summary))


if __name__ == "__main__":
    main(sys.argv[1:])

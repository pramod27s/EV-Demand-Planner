# Bengaluru, Karnataka (India) — reference data

Bengaluru does **not** publish per-station or per-session EV-charging data, so this folder is reference
material for the report. It cannot be analysed in the Workspace, which needs energy per location.

| File | What it is | Source | Licence |
|---|---|---|---|
| `bengaluru_bescom_yearly.csv` | City-wide yearly EV-charging stations and energy (LT and HT connections) | BESCOM via OpenCity — 'BESCOM EV Consumption Data' | Public domain (as listed on OpenCity) |
| `bengaluru_bescom_raw.csv` | The same file exactly as published | as above | as above |
| `bengaluru_stations_osm.csv` | 55 public charging stations mapped in OpenStreetMap (name, operator, coordinates) | OpenStreetMap via the Overpass API | ODbL 1.0 — © OpenStreetMap contributors |

Retrieved 2026-10-28. Regenerate with `python scripts/fetch_city_samples.py bengaluru_in`.

## Yearly totals (BESCOM)

| Year | LT stations | LT energy (kWh) | HT stations | HT energy (kWh) |
|---|---|---|---|---|
| 2020-21 | 38 | 28,820 | — | — |
| 2021-22 | 79 | 326,166 | — | — |
| 2022-23 | 330 | 2,295,349 | 14 | 9,861,717 |

## Caveats
* OpenStreetMap is volunteer-mapped and far from complete: Bengaluru has about 4,400 public charging
  stations, but only the ones listed here are mapped. There is no usage data in OpenStreetMap.
* Operators (Tata Power, Bolt.Earth, ...) keep session-level data private.

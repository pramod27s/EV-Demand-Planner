# Leeds (UK)

| | |
|---|---|
| Publisher | Leeds City Council (via Data Mill North) |
| Source | https://datamillnorth.org/dataset/electric-vehicle-chargepoints-operated-by-leeds-city-council |
| Licence | Open Government Licence v3.0 |
| Retrieved | 2026-10-28 |
| Sessions | 10,111 (of 13,264 raw rows) |
| Stations | 22 |
| Period | 2019-01-02 → 2021-03-31 |

Quarterly files 2019 Q1 – 2021 Q1 combined. Station = site + charge point ID; rapid (50 kW) units marked 'DC Fast'.

Station coordinates were geocoded from addresses/site names with OpenStreetMap Nominatim (see `leeds_stations.csv`); rows without a valid time, a positive energy value or a locatable station were dropped.
Regenerate with `python scripts/fetch_city_samples.py leeds_uk`.

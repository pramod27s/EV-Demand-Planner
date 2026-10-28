# Boulder, Colorado (USA)

| | |
|---|---|
| Publisher | City of Boulder |
| Source | https://open-data.bouldercolorado.gov/ — 'Electric Vehicle Charging Station Data' (ArcGIS feature service) |
| Licence | City of Boulder open data terms (public open data portal) |
| Retrieved | 2026-10-28 |
| Sessions | 125,381 (of 148,136 raw rows) |
| Stations | 50 |
| Period | 2018-01-01 → 2023-11-30 |

Same ChargePoint export format as the Palo Alto training data. Times are local (Mountain Time).

Station coordinates were geocoded from addresses/site names with OpenStreetMap Nominatim (see `boulder_stations.csv`); rows without a valid time, a positive energy value or a locatable station were dropped.
Regenerate with `python scripts/fetch_city_samples.py boulder_co`.

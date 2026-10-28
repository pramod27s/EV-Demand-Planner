# Hyderabad, Telangana (India)

| | |
|---|---|
| Publisher | TSSPDCL (Southern Power Distribution Co. of Telangana), via OpenCity |
| Source | https://data.opencity.in/ — 'Hyderabad EV Charging Stations Monthly Consumption Data' (2021-2024) |
| Licence | Public domain (as listed on OpenCity) |
| Retrieved | 2026-10-28 |
| Rows | 1,954 locality-months (from 3,549 supply-area rows) |
| Localities | 109 |
| Period | 2021-06 to 2024-08 |

Monthly electricity units (kWh) billed to EV-charging connections per supply area. There are no individual sessions, so this is a demand file: the Workspace shows the demand map, monthly trend and SMA-suggested charger sites, but not per-session estimates. Supply areas are summed per locality (section) and placed at the locality's geocoded centre.

Locality coordinates were geocoded with OpenStreetMap Nominatim (see `hyderabad_locations.csv`); localities that could not be located were dropped.
Regenerate with `python scripts/fetch_city_samples.py hyderabad_in`.

# Sample datasets from other cities

Real, public EV charging data from four cities outside the training area. Upload any of these files on
the Workspace page (*My data (upload)*). The Workspace's built-in *Sample city* is Palo Alto.

| File | City | Sessions | Stations | Period | Licence | Driver IDs |
|---|---|---|---|---|---|---|
| `boulder_co/boulder_sessions.csv` | Boulder, Colorado, USA | 125,381 | 50 | 2018-01 → 2023-11 | City of Boulder open data | no |
| `cary_nc/cary_sessions.csv` | Cary, North Carolina, USA | 18,010 | 22 | 2012-04 → 2023-01 | CC0 1.0 (public domain) | no |
| `leeds_uk/leeds_sessions.csv` | Leeds, United Kingdom | 10,111 | 22 | 2019-01 → 2021-03 | Open Government Licence v3.0 | yes |
| `hyderabad_in/hyderabad_demand.csv` | **Hyderabad, India** | 1,954 locality-months* | 109 localities | 2021-06 → 2024-08 | Public domain (OpenCity) | — |

\* No Indian city publishes individual charging sessions. Hyderabad's utility (TSSPDCL) publishes the **monthly
kWh billed to EV-charging connections per supply area**; these are summed per locality and geocoded. It is a
*demand file*: the Workspace shows the measured demand map, the monthly trend (2021 → 2024) and SMA-suggested
charger sites, but no per-session estimates. 41 supply areas in towns outside Hyderabad (e.g. Siddipet,
Suryapet, more than 45 km away) were left out.

**Reference only — Bengaluru, India** (`bengaluru_in/`): Bengaluru publishes no per-location charging data, so this
folder is for the report, not for upload. It holds BESCOM's city-wide yearly totals (38 → 330 low-tension stations
and 28,820 → 2,295,349 kWh from 2020-21 to 2022-23, about 80× growth) and the 55 charging stations mapped in
OpenStreetMap (ODbL, © OpenStreetMap contributors). See `bengaluru_in/SOURCE.md`.

Each folder contains:

* `<city>_sessions.csv` — `Station Name, Latitude, Longitude, Start Date, Port Type, User ID, Energy (kWh)`
* `<city>_stations.csv` — each address/site with its geocoded coordinates
  (Hyderabad: `hyderabad_demand.csv` = `Location, Latitude, Longitude, Month, Energy (kWh), Supply areas,
  Connected load (kW)` and `hyderabad_locations.csv`)
* `SOURCE.md` — publisher, source URL, licence, retrieval date and processing notes

## How they were made
`python scripts/fetch_city_samples.py` downloads the official files, keeps sessions with a valid start time
and a positive energy value (the same rule as the training pipeline), and geocodes each station address or
site name with OpenStreetMap Nominatim. Hits more than 30 km from the city centre are rejected. Geocoding
results are cached in `geocode_cache.json`.

## What to expect in the Workspace (honest results)
The model was trained only on Palo Alto, so these files test how well it **transfers**:

| City | Estimate RMSE | R² | Note |
|---|---|---|---|
| Boulder | 8.8 kWh | 0.02 | no driver history in the data |
| Cary | 10.2 kWh | 0.03 | no driver history; long-tailed energy (max 209 kWh) |
| Leeds | 9.7 kWh | −0.05 | different country, chargers and driving patterns |
| *Palo Alto test set* | *5.3 kWh* | *0.42* | *for comparison* |

The exact kWh per session does **not** transfer, which is the limitation stated in the project spec (§15). What
remains useful is the *relative* picture: which locations draw the most energy, when demand peaks, and where
new chargers would sit closest to demand. The Workspace shows a "far from Palo Alto" warning for these files.
Retraining on a city's own data (the same pipeline, pointed at that city's CSV) is the way to get accurate
per-session estimates there.

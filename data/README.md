# data/

Static reference files the pipeline reads from disk instead of mounting
Google Drive (which only works interactively in Colab, not on an unattended
GitHub Actions runner).

## Needed here

- `India_State_Boundary.json` — India state boundaries, WGS84 GeoJSON
- `India_District_Boundary.json` — India district boundaries, WGS84 GeoJSON

These were originally added for this repo's earlier rainfall-map product
(`Rainfall_7day_Interactive.ipynb`), sourced from `Interactive_Map/` in
Google Drive and pre-processed there (mapshaper-exported, coordinates
rounded to 5 decimal places, trimmed to just the `STATE` / `District`+`STATE`
attribute fields the map actually uses). They're generic India boundaries,
not rainfall-specific, so `temp_map_bot.py` reuses them as-is for the
temperature map -- nothing here needs redoing per run, or per product.

## Until they're here

`temp_map_bot.py`'s `get_boundaries()` falls back to an empty
FeatureCollection for whichever file is missing rather than failing the
whole build — the map still generates and uploads on schedule, just without
the state/district boundary toggle layers filled in until these are added.

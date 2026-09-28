# data/

Static reference files the pipeline reads from disk instead of mounting
Google Drive (which only works interactively in Colab, not on an unattended
GitHub Actions runner).

## Needed here

- `India_State_Boundary.json` — India state boundaries, WGS84 GeoJSON
- `India_District_Boundary.json` — India district boundaries, WGS84 GeoJSON

These are the same two files the original notebook (`Rainfall_7day_Interactive.ipynb`)
reads from `Interactive_Map/` in Google Drive, already pre-processed there
(mapshaper-exported, coordinates rounded to 5 decimal places, trimmed to
just the `STATE` / `District`+`STATE` attribute fields the map actually
uses). Export/copy them here once from that same Drive folder — nothing in
this pipeline reprocesses them, so this is a one-time step, not something
that needs redoing per run.

## Until they're here

`rainfall_map_bot.py`'s `get_boundaries()` falls back to an empty
FeatureCollection for whichever file is missing rather than failing the
whole build — the map still generates and uploads on schedule, just without
the state/district boundary toggle layers filled in until these are added.

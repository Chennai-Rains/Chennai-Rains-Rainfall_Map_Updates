# Temperature Map — automated interactive temperature map

Twice a day (4 AM / 4 PM IST), pulls the latest 7-day temperature outlook
from ECMWF HRES and ICON and publishes an interactive HTML map — Daily
Max, Daily Min, a 6-hourly diurnal cycle, and UTCI heat stress (ECMWF
only) — to `plots.chennairains.com`.

`temp_map_bot.py` is extracted from the interactive Colab notebook
(`Temp_7Day_interactive.ipynb`) that this project was originally developed
in — that notebook stays the place to develop and test new logic live.
This script is only the part that needs to run unattended on a schedule.
Like the [radar nowcast bot](https://github.com/Chennai-Rains/IMD_Radar_Updates),
it uploads via FTP; unlike it, this pipeline keeps **no state between
runs** — every cycle downloads everything fresh and builds one complete,
self-contained map.

## Why no WeatherNext 3

This repo previously automated a different product — a 7-day rainfall
blend map across 8 model sources, including WeatherNext 3. That product
was pulled back out of GitHub automation (see git history for the full
debugging trail) because WeatherNext 3's source bucket only grants list
access to individually Google-Form-registered accounts, not something a
CI identity can join without a heavier OAuth-refresh-token setup. The
rainfall map now runs manually via Colab instead.

This temperature map only ever needed ECMWF HRES and ICON (WeatherNext's
temperature accuracy was found to be noticeably worse than the numerical
models in testing), so it was never blocked by that issue and can run
fully unattended on this repo's schedule from day one.

## One thing changed from the notebook to run headless

State/district boundaries are read from `data/` in this repo (already
committed there from the earlier rainfall-map work — same two files,
reused as-is since they're generic India boundaries) instead of a Google
Drive mount.

## How it runs

`.github/workflows/temperature-map.yml` fires at 4 AM and 4 PM IST (cron)
or on demand (the **Run workflow** button under the repo's Actions tab).
Each run:

1. Installs `libeccodes-dev` + the Python dependencies
2. Runs `python temp_map_bot.py` — fetches HRES + ICON (best-effort; a
   source that's down or not yet published for this run just means that
   model is missing from the toggle this cycle, not a failed build),
   builds `output/Temp_interactive.html`
3. Uploads it to your cPanel hosting over FTP

## One-time setup

Add these under **Settings → Secrets and variables → Actions** on this
repo (same names/values as the radar nowcast bot repo, since they upload
to the same site — GitHub secrets don't carry over between repos, so
they need to be added here separately even though the values are the same):

| Secret | Value |
|---|---|
| `FTP_HOST` | Your cPanel FTP hostname (e.g. `ftp.chennairains.com`) |
| `FTP_USERNAME` | The FTP account username |
| `FTP_PASSWORD` | The FTP account password |
| `FTP_REMOTE_DIR` | The remote path to upload into — same folder the radar bot uploads to (`plots.chennairains.com`'s document root), with a trailing slash |

Trigger a manual run from the **Actions** tab (**Update interactive
temperature map** → **Run workflow**) to confirm it works end to end
before waiting for the first scheduled tick.

## Running locally

```bash
pip install -r requirements.txt
# libeccodes-dev must also be installed as a system package
python temp_map_bot.py
```

Produces `output/Temp_interactive.html` — open it directly in a browser to
check it before it ever reaches the live site.

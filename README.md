# Rainfall Map Updates — automated toggleable rainfall blend map

Twice a day (4 AM / 4 PM IST), pulls the latest 7-day rainfall forecast
from 8 atomic model sources — ECMWF HRES, EPS, AIFS (Deterministic), AIFS
Ensembles, GEM, ICON, GFS, and WeatherNext 3 — and publishes a fully
client-side toggleable HTML map (all blending/coloring happens in the
browser via JS, no backend) to `plots.chennairains.com`.

`rainfall_map_bot.py` is extracted from the interactive Colab notebook
(`Rainfall_7day_Interactive.ipynb`) that this project was originally
developed in — that notebook stays the place to develop and test new
model-blend logic live. This script is only the part that needs to run
unattended on a schedule. Unlike the [IMD radar nowcast bot](https://github.com/Chennai-Rains/IMD_Radar_Updates),
this pipeline keeps **no state between runs** — every cycle downloads
everything fresh and builds one complete, self-contained map, so there's
nothing to persist or track.

## Two things changed from the notebook to run headless

The notebook was written for interactive use in Colab, which two of its
steps depend on:

1. **WeatherNext 3** data lives in a Google Cloud Storage bucket the
   notebook reads via `google.colab.auth.authenticate_user()` — an
   interactive sign-in that doesn't exist outside Colab. This pipeline
   uses an anonymous GCS client instead, on the assumption the bucket is
   actually publicly readable. **This hasn't been verified against a real
   run yet** — watch the first scheduled/manual run's logs for this
   specifically; if it turns out real credentials are needed, the fix is a
   GCP service-account key added as a secret, not a code change.
2. **State/district boundaries** are read from `data/` in this repo
   instead of a Google Drive mount — see `data/README.md`. The map still
   builds and publishes without them (just missing the boundary toggle
   layers) until those two files are added.

## How it runs

`.github/workflows/rainfall-map.yml` fires at 4 AM and 4 PM IST (cron) or
on demand (the **Run workflow** button under the repo's Actions tab). Each
run:

1. Installs `libeccodes-dev` + the Python dependencies
2. Runs `python rainfall_map_bot.py` — fetches all 8 sources (best-effort;
   a source that's down or not yet published for this run just means that
   model is missing from the toggle this cycle, not a failed build),
   builds `output/toggle_rainfall_map.html`
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

Then add the two boundary GeoJSON files to `data/` (see `data/README.md`).

Once secrets are in place, trigger a manual run from the **Actions** tab
(**Update toggleable rainfall map** → **Run workflow**) to confirm it
works end to end before waiting for the first scheduled tick.

## Running locally

```bash
pip install -r requirements.txt
# libeccodes-dev must also be installed as a system package
python rainfall_map_bot.py
```

Produces `output/toggle_rainfall_map.html` — open it directly in a browser
to check it before it ever reaches the live site.

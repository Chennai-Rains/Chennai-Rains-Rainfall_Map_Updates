"""
Automated build of the toggleable 7-day rainfall blend map for chennairains.com.

Ported from Rainfall_7day_Interactive.ipynb (the Colab dev notebook -- keep
using that one for interactive development/debugging of the model-blend
logic itself). This script is only the part meant to run unattended, twice
a day, via GitHub Actions: fetch each of the 8 atomic model outputs fresh,
build the fully client-side toggleable HTML map, and let the workflow
FTP-upload it to plots.chennairains.com.

No cross-run state is needed here, unlike the radar nowcast bot -- every
run downloads everything from scratch and produces one complete,
self-contained map, so there's nothing to persist between cycles.

Two things had to change from the notebook to run headless instead of
interactively in Colab:

  1. WeatherNext 3 data lives in a GCS bucket the notebook reads via
     google.colab.auth.authenticate_user() -- an interactive sign-in that
     doesn't exist outside Colab. Replaced with an anonymous GCS client
     (storage.Client.create_anonymous_client()), on the theory that the
     bucket is actually publicly readable and Colab's auth call is just
     boilerplate carried over from other Google example notebooks, not a
     real requirement. THIS COULD NOT BE VERIFIED before deploying -- the
     sandbox this port was written in has no network route to
     storage.googleapis.com at all. Watch the first real run's logs for
     this specifically. If the bucket does turn out to need real
     credentials, the fix is a GCP service-account key stored as a GitHub
     secret, not a code change here.

  2. The state/district boundary GeoJSON files are read from Google Drive
     via drive.mount() -- Colab-only, and unnecessary to mount Drive for
     anyway since these two files never change run to run. They're
     committed into this repo's data/ folder instead and read straight
     from disk (see get_boundaries() below).

Everything else -- the ECMWF/GEM/ICON/GFS downloads, the per-model 24h
delta extraction, quantization, and the HTML/JS template -- is unchanged
from the notebook, including its existing tolerance for a source being
unavailable on a given run (get_atomic_24h returns None per-model rather
than failing the whole build; the client-side map just shows "no data for
this combination" for that gap).
"""

import os
import io
import bz2
import json
import time
import base64
import datetime
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from ecmwf.opendata import Client

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Step 1 -- Config: extent, canonical shared grid, color scale ===
# (unchanged from the notebook)

lon_min, lon_max, lat_min, lat_max = 60, 105, 6, 45
bounds_full = [[lat_min, lon_min], [lat_max, lon_max]]
optimization_area = [47, 58, 4, 107]  # [North, West, South, East], buffered beyond display extent

GRID_STEP = 0.1  # matches HRES/AIFS-Det/regridded-ICON native resolution
canonical_lon = np.arange(lon_min, lon_max + 1e-6, GRID_STEP)
canonical_lat = np.arange(lat_max, lat_min - 1e-6, -GRID_STEP)
GRID_W, GRID_H = len(canonical_lon), len(canonical_lat)

colors = ['#B3D2F5', '#9FC7F5', '#6EADF5', '#4399FA', '#117FFA', '#037801', '#05BA02', '#06F002',
          '#FFFF00', '#FFEB3B', '#FDD835', '#FBC02D', '#F9A825', '#F57C00', '#EF6C00', '#E65100',
          '#D84315', '#BF360C', '#F06292', '#E91E63', '#AD1457', '#880E4F', '#D3D3D3']
levels_24h = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 125, 200, 300]

CARTO_VOYAGER_URL = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2eki_1_8725e9310ea39758b1e8f374'
CARTO_ATTR = '&copy; <a href="https://carto.com/attributions">CARTO</a> &copy; OpenStreetMap contributors'

MODEL_KEYS = ['hres', 'gem', 'gfs', 'icon', 'eps', 'aifs_det', 'aifs_ens', 'weathernext']

# Module-level globals that get_atomic_24h() and friends read as free
# variables (same pattern as the notebook's own cell-to-cell globals, and
# the same pattern already used in the radar bot's automation glue) --
# populated by fetch_ecmwf(), fetch_aifs_ens(), fetch_weathernext() below.
client = Client(source="ecmwf")
tp_mean = None
tp_hres_mm = None
tp_aifs_det_mm = None
tp_aifs_mean = None
wn_precip_hourly_mean = None
wn_max_lead_hours = None
available_steps = []
aifs_available_steps = []
run_time = None
date_str = None
hour_int = None
required_steps = []
steps_end = []
steps_start = []
valid_days = []


# === Step 2 -- ECMWF run detection + EPS + HRES + AIFS Deterministic ===
# (unchanged from the notebook)

def get_forecast_data():
    for offset_hours in [0, 6, 12, 18, 24]:
        try:
            base_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=offset_hours)
            test_date = base_time.strftime("%Y%m%d")
            test_hour = (base_time.hour // 6) * 6
            print(f"Checking availability for Run: {test_date} {test_hour:02d} UTC...")
            first_00z = ((24 - test_hour) % 24) + 24 if test_hour != 0 else 24
            target_steps = [first_00z + (i * 24) for i in range(7)]
            req_steps = sorted(list(set([s - 24 for s in target_steps] + target_steps)))
            if 0 in req_steps:
                req_steps.remove(0)
            client.retrieve(date=test_date, time=test_hour, type="pf", param="tp",
                             step=req_steps[-1], target="probe.grib", area=optimization_area)
            print(f"Found stable Run: {test_date} {test_hour:02d} UTC")
            return test_date, test_hour, req_steps, target_steps, [s - 24 for s in target_steps]
        except Exception as e:
            if "429" in str(e):
                print("Rate limited. Waiting 5 seconds...")
                time.sleep(5)
            print(f"Run {test_hour:02d}Z not ready at this offset.")
            continue
    raise Exception("Could not find any available forecast runs.")


def fetch_ecmwf_hres_eps_aifsdet():
    global tp_mean, tp_hres_mm, tp_aifs_det_mm, available_steps, valid_days
    global run_time, date_str, hour_int, required_steps, steps_end, steps_start

    date_str_, hour_int_, required_steps_, steps_end_, steps_start_ = get_forecast_data()
    date_str, hour_int, required_steps, steps_end, steps_start = (
        date_str_, hour_int_, required_steps_, steps_end_, steps_start_)
    run_time = datetime.datetime.strptime(f"{date_str} {hour_int}", "%Y%m%d %H")

    print("Downloading EPS ensemble...")
    client.retrieve(date=date_str, time=hour_int, type="pf", param="tp", step=required_steps,
                     target='ecmwf_tp_ens_dynamic.grib', area=optimization_area, grid=[0.25, 0.25])
    ds = xr.open_dataset('ecmwf_tp_ens_dynamic.grib', engine='cfgrib')
    tp_mean = (ds['tp'] * 1000 * 1.4).mean(dim='number')

    print("Downloading HRES deterministic...")
    client.retrieve(date=date_str, time=hour_int, model="ifs", type="fc", param="tp", step=required_steps,
                     target='ecmwf_tp_hres_dynamic.grib', area=optimization_area, grid=[0.1, 0.1])
    ds_hres = xr.open_dataset('ecmwf_tp_hres_dynamic.grib', engine='cfgrib')
    tp_hres_mm = ds_hres['tp'] * 1000

    print("Downloading AIFS Deterministic...")
    client_aifs_det = Client(source="ecmwf", model="aifs-single")
    client_aifs_det.retrieve(date=date_str, time=hour_int, stream="oper", type="fc", param="tp", step=required_steps,
                              target='ecmwf_tp_aifs_det_dynamic.grib', area=optimization_area, grid=[0.25, 0.25])
    ds_aifs_det = xr.open_dataset('ecmwf_tp_aifs_det_dynamic.grib', engine='cfgrib')
    tp_aifs_det_mm = ds_aifs_det['tp']

    available_steps = [int(s / np.timedelta64(1, 'h')) for s in tp_mean.step.values]
    valid_days = [i for i in range(len(steps_end)) if steps_end[i] in available_steps]
    print(f"Loaded EPS+HRES+AIFS-Det for run {run_time}. Valid days: {[d + 1 for d in valid_days]}")


# === Step 3 -- AIFS Ensembles ===
# (unchanged from the notebook)

def fetch_aifs_ens():
    global tp_aifs_mean, aifs_available_steps
    client_aifs = Client(source="ecmwf", model="aifs-ens")
    aifs_required_steps = sorted(set(steps_start) | set(steps_end))
    client_aifs.retrieve(date=date_str, time=hour_int, stream="enfo", type="pf", param="tp",
                          step=aifs_required_steps, target='ecmwf_tp_aifs_ens_dynamic.grib',
                          area=optimization_area, grid=[0.25, 0.25])
    ds_aifs = xr.open_dataset('ecmwf_tp_aifs_ens_dynamic.grib', engine='cfgrib')
    tp_aifs_mean = ds_aifs['tp'].mean(dim='number')
    aifs_available_steps = [int(s / np.timedelta64(1, 'h')) for s in tp_aifs_mean.step.values]
    print(f"AIFS-ENS loaded for steps: {aifs_available_steps}")


# === Step 4 -- GEM (deterministic) ===
# (unchanged from the notebook)

def fetch_gem():
    base_url_gem = f"https://dd.weather.gc.ca/{date_str}/WXO-DD/model_gdps/15km/{hour_int:02d}/"
    gem_forecast_steps_to_use = [step for step in steps_end if step >= 24]
    print("Downloading GEM GDPS 24h precip files...")
    for step in gem_forecast_steps_to_use:
        s_str = f"{step:03d}"
        url_gem = f"{base_url_gem}{s_str}/{date_str}T{hour_int:02d}Z_MSC_GDPS_Precip-Accum24h_Sfc_LatLon0.15_PT{s_str}H.grib2"
        target_file_gem = f"gem_gdps_24h_{s_str}.grib2"
        if not os.path.exists(target_file_gem):
            try:
                r = requests.get(url_gem)
                r.raise_for_status()
                with open(target_file_gem, 'wb') as f:
                    f.write(r.content)
                print(f"Downloaded {target_file_gem}")
            except Exception as e:
                print(f"Failed to download GEM {s_str}h: {e}")
    print("GEM download phase complete.")


# === Step 5 -- ICON Global (deterministic, regridded) ===
# (unchanged from the notebook)

import bz2  # noqa: E402  (kept next to its first use, matching the notebook's cell layout)
import eccodes  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

_icon_target_lon = np.arange(optimization_area[1], optimization_area[3] + 0.1, 0.1)
_icon_target_lat = np.arange(optimization_area[2], optimization_area[0] + 0.1, 0.1)
_icon_grid_lon, _icon_grid_lat = np.meshgrid(_icon_target_lon, _icon_target_lat)
_icon_cumulative_cache, _icon_coords_cache = {}, {}
icon_base_url = None
icon_run_str = None


def _download_icon_file(url_path, local_name):
    if os.path.exists(local_name):
        return local_name
    url = f"{icon_base_url}/{url_path}.bz2"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"ICON file not available: {url} ({r.status_code}).")
            return None
        with open(local_name, 'wb') as f:
            f.write(bz2.decompress(r.content))
        return local_name
    except Exception as e:
        print(f"ICON download failed for {url}: {e}")
        return None


def _read_icon_values(path):
    with open(path, 'rb') as f:
        gid = eccodes.codes_grib_new_from_file(f)
        if gid is None:
            return None
        vals = eccodes.codes_get_array(gid, 'values')
        eccodes.codes_release(gid)
    return vals


def get_icon_grid_coords():
    if 'coords' in _icon_coords_cache:
        return _icon_coords_cache['coords']
    clat_path = _download_icon_file(f"clat/icon_global_icosahedral_time-invariant_{icon_run_str}_CLAT.grib2",
                                     f"icon_clat_{icon_run_str}.grib2")
    clon_path = _download_icon_file(f"clon/icon_global_icosahedral_time-invariant_{icon_run_str}_CLON.grib2",
                                     f"icon_clon_{icon_run_str}.grib2")
    if clat_path is None or clon_path is None:
        raise RuntimeError("Could not download ICON CLAT/CLON grid-coordinate files.")
    icon_lats = _read_icon_values(clat_path)
    icon_lons = _read_icon_values(clon_path)
    icon_lons = np.where(icon_lons > 180, icon_lons - 360, icon_lons)
    buf = 1.0
    mask = (icon_lats >= optimization_area[2] - buf) & (icon_lats <= optimization_area[0] + buf) & \
           (icon_lons >= optimization_area[1] - buf) & (icon_lons <= optimization_area[3] + buf)
    _icon_coords_cache['coords'] = (icon_lats, icon_lons, mask)
    print(f"ICON grid coordinates loaded: {mask.sum()} points within crop.")
    return _icon_coords_cache['coords']


def get_icon_cumulative_mm(step):
    if step in _icon_cumulative_cache:
        return _icon_cumulative_cache[step]
    icon_lats, icon_lons, mask = get_icon_grid_coords()
    fname = f"icon_global_icosahedral_single-level_{icon_run_str}_{step:03d}_TOT_PREC.grib2"
    path = _download_icon_file(f"tot_prec/{fname}", fname)
    if path is None:
        return None
    values = _read_icon_values(path)
    if values is None:
        return None
    regridded = griddata((icon_lons[mask], icon_lats[mask]), values[mask],
                          (_icon_grid_lon, _icon_grid_lat), method='linear')
    nan_mask = np.isnan(regridded)
    if nan_mask.any():
        regridded[nan_mask] = griddata((icon_lons[mask], icon_lats[mask]), values[mask],
                                        (_icon_grid_lon[nan_mask], _icon_grid_lat[nan_mask]), method='nearest')
    da = xr.DataArray(regridded, dims=['latitude', 'longitude'],
                       coords={'latitude': _icon_target_lat, 'longitude': _icon_target_lon})
    _icon_cumulative_cache[step] = da
    return da


def get_icon_24h_delta_mm(end_step, start_step):
    icon_end = get_icon_cumulative_mm(end_step)
    if icon_end is None:
        return None
    if start_step == 0:
        return icon_end
    icon_start = get_icon_cumulative_mm(start_step)
    if icon_start is None:
        return icon_end
    return icon_end - icon_start


def init_icon_urls():
    global icon_base_url, icon_run_str
    icon_base_url = f"https://opendata.dwd.de/weather/nwp/icon/grib/{hour_int:02d}"
    icon_run_str = f"{date_str}{hour_int:02d}"


# === Step 6 -- GFS (deterministic) ===
# (unchanged from the notebook)

def _gfs_base_url():
    return (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
            f"file=gfs.t{hour_int:02d}z.pgrb2.0p25.f{{step:03d}}&lev_surface=on&var_APCP=on"
            f"&subregion=&leftlon={lon_min}&rightlon={lon_max}&toplat={lat_max}&bottomlat={lat_min}"
            f"&dir=%2Fgfs.{date_str}%2F{hour_int:02d}%2Fatmos")


def get_gfs_step_file(s):
    f = f"gfs_3h_apcp_{s:03d}.grib2"
    if not os.path.exists(f):
        url = _gfs_base_url().format(step=s)
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(f, 'wb') as file:
                file.write(r.content)
    return f


def fetch_gfs():
    print("Downloading GFS 3-hourly files...")
    for end_step in steps_end:
        for s in [end_step - i for i in range(0, 24, 3)]:
            get_gfs_step_file(s)
    print("GFS download phase complete.")


def get_gfs_24h_mm(end_step):
    accumulation_steps = [end_step - i for i in range(0, 24, 3)]
    daily_accumulation, lons_gfs, lats_gfs = None, None, None
    for s in accumulation_steps:
        sf = f"gfs_3h_apcp_{s:03d}.grib2"
        if os.path.exists(sf):
            ds_s = xr.open_dataset(sf, engine='cfgrib')
            v = next((var for var in ['tp', 'apcp', 'unknown'] if var in ds_s.data_vars), None)
            if v:
                if daily_accumulation is None:
                    daily_accumulation = ds_s[v].values * 1.0
                    lons_gfs, lats_gfs = ds_s.longitude.values, ds_s.latitude.values
                else:
                    daily_accumulation += ds_s[v].values * 1.0
            ds_s.close()
    if daily_accumulation is None:
        return None
    return xr.DataArray(daily_accumulation, dims=['latitude', 'longitude'],
                         coords={'latitude': lats_gfs, 'longitude': lons_gfs})


# === Step 7 -- WeatherNext 3 ===
# MODIFIED from the notebook: google.colab.auth.authenticate_user() replaced
# with an anonymous GCS client (see module docstring, point 1).

def fetch_weathernext():
    global wn_precip_hourly_mean, wn_max_lead_hours

    import obstore
    import zarr

    STATS_BUCKET = "weathernext3_statistics_spatial"
    STATS_PREFIX = "weathernext_3_0_0_statistics/zarr/2026_to_present/"

    # Listing done via obstore (skip_signature=True), not
    # google-cloud-storage: the first real run hit
    # "Anonymous credentials cannot be refreshed" from
    # storage.Client.create_anonymous_client().list_blobs() -- a known
    # google-auth gotcha where AnonymousCredentials explicitly refuses any
    # refresh() call, which some code paths trigger unconditionally before
    # a request. obstore is a separate (Rust-based) client that never goes
    # through google-auth's Python credentials machinery, so it doesn't hit
    # this at all -- confirmed working for the zarr read itself (below) on
    # that same first run, it was only the listing step that failed.
    list_store = obstore.store.GCSStore(bucket=STATS_BUCKET, skip_signature=True)
    listing = obstore.list_with_delimiter(list_store, prefix=STATS_PREFIX)
    # obstore's common_prefixes come back WITHOUT a trailing delimiter
    # (unlike google-cloud-storage's list_blobs(delimiter=...).prefixes,
    # which include it) -- confirmed against an in-memory store since this
    # couldn't be checked against the real bucket from the dev sandbox.
    # Normalized here so "+ predictions.zarr" below still lands as a
    # sibling path inside the run directory, not concatenated onto its name.
    run_dirs = sorted(p if p.endswith("/") else p + "/" for p in listing["common_prefixes"])
    six_hourly_runs = [d for d in run_dirs if any(f"_{h}hr_" in d for h in ("00", "06", "12", "18"))]
    latest_wn_run = (six_hourly_runs[-1] if six_hourly_runs else run_dirs[-1]) + "predictions.zarr"
    print(f"Using WeatherNext 3 run: {latest_wn_run}")

    wn_store = obstore.store.GCSStore(bucket=STATS_BUCKET, prefix=latest_wn_run, skip_signature=True)
    zstore_wn = zarr.storage.ObjectStore(wn_store)
    ds_wn_stats = xr.open_zarr(zstore_wn, chunks={})
    precip_vars = [v for v in ds_wn_stats.data_vars if "precip" in v.lower() and v.endswith("_mean")]
    WN_PRECIP_VAR = precip_vars[0] if precip_vars else "total_precipitation_1hr_mean"
    wn_precip_hourly_mean_local = ds_wn_stats[WN_PRECIP_VAR].sel(
        lat_0p1=slice(lat_min, lat_max), lon_0p1=slice(lon_min, lon_max)
    ).rename({"lat_0p1": "latitude", "lon_0p1": "longitude"})
    wn_units = ds_wn_stats[WN_PRECIP_VAR].attrs.get("units", "")
    WN_TO_MM = 1000.0 if wn_units.lower() == "m" else 1.0
    wn_precip_hourly_mean = (wn_precip_hourly_mean_local * WN_TO_MM).astype("float32").load()
    wn_max_lead_hours = float(wn_precip_hourly_mean.lead_time.max().values / np.timedelta64(1, 'h'))
    print(f"WeatherNext 3 loaded, max lead time: {wn_max_lead_hours}h")


# === Step 8 -- Per-model 24h delta, interpolated onto the canonical grid ===
# (unchanged from the notebook)

def _to_canonical(arr):
    return arr.interp(latitude=canonical_lat, longitude=canonical_lon)


def get_atomic_24h(model_key, end_step, start_step):
    try:
        if model_key == 'hres':
            end = tp_hres_mm.sel(step=np.timedelta64(end_step, 'h'))
            val = end if start_step == 0 else end - tp_hres_mm.sel(step=np.timedelta64(start_step, 'h'))
            return _to_canonical(val)

        if model_key == 'eps':
            end = tp_mean.sel(step=np.timedelta64(end_step, 'h'))
            val = end if start_step == 0 else end - tp_mean.sel(step=np.timedelta64(start_step, 'h'))
            return _to_canonical(val)

        if model_key == 'aifs_det':
            if end_step not in available_steps or (start_step != 0 and start_step not in available_steps):
                return None
            end = tp_aifs_det_mm.sel(step=np.timedelta64(end_step, 'h'))
            val = end if start_step == 0 else end - tp_aifs_det_mm.sel(step=np.timedelta64(start_step, 'h'))
            return _to_canonical(val)

        if model_key == 'aifs_ens':
            if end_step not in aifs_available_steps or (start_step != 0 and start_step not in aifs_available_steps):
                return None
            end = tp_aifs_mean.sel(step=np.timedelta64(end_step, 'h'))
            val = end if start_step == 0 else end - tp_aifs_mean.sel(step=np.timedelta64(start_step, 'h'))
            return _to_canonical(val)

        if model_key == 'gem':
            gem_file = f"gem_gdps_24h_{end_step:03d}.grib2"
            if not os.path.exists(gem_file):
                return None
            ds_gem = xr.open_dataset(gem_file, engine='cfgrib')
            v_gem = 'tp' if 'tp' in ds_gem.data_vars else 'unknown'
            val = ds_gem[v_gem]
            result = _to_canonical(val)
            ds_gem.close()
            return result

        if model_key == 'gfs':
            arr = get_gfs_24h_mm(end_step)
            return _to_canonical(arr) if arr is not None else None

        if model_key == 'icon':
            arr = get_icon_24h_delta_mm(end_step, start_step)
            return _to_canonical(arr) if arr is not None else None

        if model_key == 'weathernext':
            if wn_precip_hourly_mean is None or end_step > wn_max_lead_hours:
                return None
            window = wn_precip_hourly_mean.sel(
                lead_time=slice(np.timedelta64(start_step + 1, 'h'), np.timedelta64(end_step, 'h'))
            ).sum(dim="lead_time")
            return _to_canonical(window)

        return None
    except Exception as e:
        print(f"get_atomic_24h({model_key}, end={end_step}, start={start_step}) failed: {e}")
        return None


# === Step 9 -- Quantize + generate all atomic grids ===
# (unchanged from the notebook)

def quantize_and_encode(arr):
    vals = np.nan_to_num(arr.values, nan=0.0)
    vals = np.clip(vals, 0, 254)
    q = vals.round().astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode('ascii')


def build_all_grids():
    grid_data = {}
    day_labels = {}
    for idx in valid_days:
        end_step = steps_end[idx]
        start_step = steps_start[idx]
        day_num = idx + 1
        target_date = run_time + datetime.timedelta(hours=start_step)
        day_labels[day_num] = f"Day {day_num} ({target_date.strftime('%d %b')})"

        for model_key in MODEL_KEYS:
            arr = get_atomic_24h(model_key, end_step, start_step)
            if arr is None:
                print(f"{model_key} Day {day_num}: not available, skipping.")
                continue
            grid_data.setdefault(model_key, {})[day_num] = quantize_and_encode(arr)

    total_cells_expected = len(MODEL_KEYS) * len(valid_days)
    total_cells_actual = sum(len(v) for v in grid_data.values())
    print(f"\nGenerated {total_cells_actual} of {total_cells_expected} possible (model, day) grids.")
    for mk in MODEL_KEYS:
        print(f"  {mk}: {len(grid_data.get(mk, {}))} / {len(valid_days)} days")
    return grid_data, day_labels


# === Step 9b -- State/district boundaries and state capitals ===
# MODIFIED from the notebook: read from this repo's data/ folder instead of
# a Google Drive mount (see module docstring, point 2).

state_capitals = [
    {"state": "Andhra Pradesh", "capital": "Amaravati", "lat": 16.51, "lon": 80.52},
    {"state": "Arunachal Pradesh", "capital": "Itanagar", "lat": 27.08, "lon": 93.61},
    {"state": "Assam", "capital": "Dispur", "lat": 26.14, "lon": 91.79},
    {"state": "Bihar", "capital": "Patna", "lat": 25.59, "lon": 85.14},
    {"state": "Chhattisgarh", "capital": "Raipur", "lat": 21.25, "lon": 81.63},
    {"state": "Goa", "capital": "Panaji", "lat": 15.49, "lon": 73.83},
    {"state": "Gujarat", "capital": "Gandhinagar", "lat": 23.22, "lon": 72.64},
    {"state": "Haryana", "capital": "Chandigarh", "lat": 30.73, "lon": 76.78},
    {"state": "Himachal Pradesh", "capital": "Shimla", "lat": 31.10, "lon": 77.17},
    {"state": "Jharkhand", "capital": "Ranchi", "lat": 23.34, "lon": 85.31},
    {"state": "Karnataka", "capital": "Bengaluru", "lat": 12.97, "lon": 77.59},
    {"state": "Kerala", "capital": "Thiruvananthapuram", "lat": 8.52, "lon": 76.94},
    {"state": "Madhya Pradesh", "capital": "Bhopal", "lat": 23.26, "lon": 77.41},
    {"state": "Maharashtra", "capital": "Mumbai", "lat": 19.08, "lon": 72.88},
    {"state": "Manipur", "capital": "Imphal", "lat": 24.82, "lon": 93.94},
    {"state": "Meghalaya", "capital": "Shillong", "lat": 25.58, "lon": 91.89},
    {"state": "Mizoram", "capital": "Aizawl", "lat": 23.73, "lon": 92.72},
    {"state": "Nagaland", "capital": "Kohima", "lat": 25.68, "lon": 94.11},
    {"state": "Odisha", "capital": "Bhubaneswar", "lat": 20.30, "lon": 85.82},
    {"state": "Punjab", "capital": "Chandigarh", "lat": 30.73, "lon": 76.78},
    {"state": "Rajasthan", "capital": "Jaipur", "lat": 26.91, "lon": 75.79},
    {"state": "Sikkim", "capital": "Gangtok", "lat": 27.33, "lon": 88.61},
    {"state": "Tamil Nadu", "capital": "Chennai", "lat": 13.08, "lon": 80.27},
    {"state": "Telangana", "capital": "Hyderabad", "lat": 17.39, "lon": 78.49},
    {"state": "Tripura", "capital": "Agartala", "lat": 23.83, "lon": 91.29},
    {"state": "Uttar Pradesh", "capital": "Lucknow", "lat": 26.85, "lon": 80.95},
    {"state": "Uttarakhand", "capital": "Dehradun", "lat": 30.32, "lon": 78.03},
    {"state": "West Bengal", "capital": "Kolkata", "lat": 22.57, "lon": 88.36},
    {"state": "Andaman and Nicobar Islands", "capital": "Sri Vijaya Puram", "lat": 11.62, "lon": 92.73},
    {"state": "Chandigarh", "capital": "Chandigarh", "lat": 30.73, "lon": 76.78},
    {"state": "Dadra & Nagar Haveli and Daman & Diu", "capital": "Daman", "lat": 20.43, "lon": 72.84},
    {"state": "Lakshadweep", "capital": "Kavaratti", "lat": 10.57, "lon": 72.64},
    {"state": "Delhi (NCT)", "capital": "New Delhi", "lat": 28.61, "lon": 77.21},
    {"state": "Puducherry", "capital": "Puducherry", "lat": 11.94, "lon": 79.81},
    {"state": "Jammu and Kashmir", "capital": "Srinagar", "lat": 34.08, "lon": 74.80},
    {"state": "Ladakh", "capital": "Leh", "lat": 34.15, "lon": 77.58},
]


_EMPTY_FEATURE_COLLECTION = {"type": "FeatureCollection", "features": []}


def get_boundaries():
    """State/district boundaries were pre-processed once in the notebook
    (rounded to 5dp, trimmed to just the attribute fields actually used) and
    committed to this repo already in that clean form -- see data/README.md
    -- so this just loads them, no reprocessing needed each run.

    Falls back to an empty FeatureCollection (map still builds, just without
    the boundary/district toggle layers) if the files aren't there yet --
    lets this run twice a day on schedule from day one instead of the whole
    build failing until data/*.json is committed."""
    state_boundary_path = DATA_DIR / "India_State_Boundary.json"
    district_boundary_path = DATA_DIR / "India_District_Boundary.json"

    if state_boundary_path.exists():
        with open(state_boundary_path) as f:
            state_boundary_geojson = json.load(f)
        print(f"State boundaries: {len(state_boundary_geojson['features'])} features")
    else:
        print(f"[boundaries] {state_boundary_path} not found -- state boundary layer will be empty "
              f"until it's committed to data/. See data/README.md.")
        state_boundary_geojson = _EMPTY_FEATURE_COLLECTION

    if district_boundary_path.exists():
        with open(district_boundary_path) as f:
            district_boundary_geojson = json.load(f)
        print(f"District boundaries: {len(district_boundary_geojson['features'])} features")
    else:
        print(f"[boundaries] {district_boundary_path} not found -- district boundary layer will be "
              f"empty until it's committed to data/. See data/README.md.")
        district_boundary_geojson = _EMPTY_FEATURE_COLLECTION

    return state_boundary_geojson, district_boundary_geojson


# === Step 10 -- Build the fully client-side toggleable map ===
# (unchanged from the notebook)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Toggleable Rainfall Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin:0; padding:0; height:100%; font-family: sans-serif; }
  #map { position:absolute; top:0; bottom:0; left:0; right:340px; }
  #sidebar {
    position:absolute; top:0; right:0; width:340px; height:100%; overflow-y:auto;
    background:#fff; box-shadow:-2px 0 6px rgba(0,0,0,0.15); padding:14px; box-sizing:border-box;
  }
  .panel { border:1px solid #ddd; border-radius:6px; padding:10px; margin-bottom:12px; }
  .panel-title { font-weight:bold; font-size:13px; margin-bottom:6px; color:#333; }
  .panel label { display:block; font-size:13px; margin:4px 0; cursor:pointer; }
  #daySelect { width:100%; padding:4px; margin-top:4px; }
  #statusMsg { font-size:12px; color:#a33; min-height:16px; margin-top:6px; }
  #legend { margin-top:14px; }
  .leaflet-image-layer {
    image-rendering: pixelated;
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
  }
  .legend-row { display:flex; align-items:center; margin:2px 0; }
  .legend-swatch { width:14px; height:14px; margin-right:6px; border:1px solid #999; display:inline-block; }
  .legend-label { font-size:11px; }
  .capital-label {
    background: rgba(255,255,255,0.85);
    border: none;
    box-shadow: none;
    font-size: 10px;
    padding: 1px 4px;
    font-weight: 600;
  }
  #valueReadout {
    position: absolute;
    top: 10px;
    left: 60px;
    z-index: 1000;
    background: rgba(255,255,255,0.97);
    padding: 9px 18px;
    border-radius: 6px;
    border: 2px solid #333;
    box-shadow: 0 2px 10px rgba(0,0,0,0.45);
    font-size: 19px;
    font-weight: 700;
    color: #111;
    display: none;
    pointer-events: none;
  }

  @media (max-width: 768px) {
    html, body { height: auto; }
    #map {
      position: relative; top: auto; bottom: auto; left: 0; right: 0;
      width: 100%; height: 65vh;
    }
    #sidebar {
      position: relative; top: auto; right: auto;
      width: 100%; height: auto; max-height: none;
      box-shadow: none; border-top: 2px solid #ddd;
    }
  }
</style>
</head>
<body>
<div id="map"><div id="valueReadout"></div></div>
<div id="sidebar">
  <h3 style="margin-top:0;">Rainfall Blend Builder</h3>
  <p style="font-size:12px; color:#666;">Pick models within ONE category below (any subset within
  a category blends together, equal-weight). Picking a model clears other categories -- ensembles
  and deterministic models aren't blended together except via the curated "Same Family" pairs. <b>Rainfall accumulation is for 24 hours beginning at 5:30 AM for the dates in the drop down</b> </p>

  <div class="panel" data-panel-group="deterministic">
    <div class="panel-title">Deterministic</div>
    <label><input type="checkbox" class="model-cb" data-panel="deterministic" value="hres"> ECMWF HRES</label>
    <label><input type="checkbox" class="model-cb" data-panel="deterministic" value="gem"> GEM</label>
    <label><input type="checkbox" class="model-cb" data-panel="deterministic" value="gfs"> GFS</label>
    <label><input type="checkbox" class="model-cb" data-panel="deterministic" value="icon"> ICON</label>
  </div>

  <div class="panel" data-panel-group="ensembles">
    <div class="panel-title">Ensembles</div>
    <label><input type="checkbox" class="model-cb" data-panel="ensembles" value="eps"> EPS</label>
  </div>

  <div class="panel" data-panel-group="ml">
    <div class="panel-title">ML Models</div>
    <label><input type="checkbox" class="model-cb" data-panel="ml" value="aifs_det"> AIFS (Deterministic)</label>
    <label><input type="checkbox" class="model-cb" data-panel="ml" value="aifs_ens"> AIFS Ensembles</label>
    <label><input type="checkbox" class="model-cb" data-panel="ml" value="weathernext"> WeatherNext 3</label>
  </div>

  <div class="panel" data-panel-group="same_family">
    <div class="panel-title">Same Family (curated pairs)</div>
    <label><input type="radio" name="sf" class="sf-radio" value="hres,aifs_det"> HRES + AIFS (Deterministic)</label>
    <label><input type="radio" name="sf" class="sf-radio" value="hres,eps"> HRES + EPS</label>
    <label><input type="radio" name="sf" class="sf-radio" value="eps,aifs_ens"> EPS + AIFS Ensembles</label>
    <label><input type="radio" name="sf" class="sf-radio" value="aifs_det,aifs_ens"> AIFS (Deterministic) + AIFS Ensembles</label>
  </div>

  <div class="panel">
    <div class="panel-title">Reference Layers</div>
    <label><input type="checkbox" id="stateBoundaryToggle"> State Boundaries</label>
    <label><input type="checkbox" id="districtBoundaryToggle"> District Boundaries</label>
    <label><input type="checkbox" id="capitalsToggle"> State Capitals</label>
  </div>

  <div class="panel">
    <div class="panel-title">Day</div>
    <select id="daySelect">__DAY_OPTIONS_HTML__</select>
  </div>

  <div class="panel">
    <div class="panel-title">Rainfall Transparency</div>
    <input type="range" id="opacitySlider" min="0" max="100" value="90" style="width:100%;">
    <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;">
      <span>See-through</span><span>Opaque</span>
    </div>
  </div>

  <div id="statusMsg"></div>

  <div id="legend">
    <div class="panel-title">Rainfall (mm)</div>
    __LEGEND_ROWS_HTML__
  </div>
</div>

<script>
const COLOR_LUT = __COLOR_LUT_JSON__;
const GRID_W = __GRID_W__;
const GRID_H = __GRID_H__;
const BOUNDS = __BOUNDS_JSON__;
const MAP_CENTER = __MAP_CENTER_JSON__;
const CARTO_URL = __CARTO_URL_JSON__;
const CARTO_ATTR = __CARTO_ATTR_JSON__;
const GRID_DATA_B64 = __GRID_DATA_JSON__;
const STATE_BOUNDARY_GEOJSON = __STATE_BOUNDARY_JSON__;
const DISTRICT_BOUNDARY_GEOJSON = __DISTRICT_BOUNDARY_JSON__;
const STATE_CAPITALS = __STATE_CAPITALS_JSON__;
const GRID_DATA = {};
for (const modelKey in GRID_DATA_B64) {
  GRID_DATA[modelKey] = {};
  for (const day in GRID_DATA_B64[modelKey]) {
    const binStr = atob(GRID_DATA_B64[modelKey][day]);
    const arr = new Uint8Array(binStr.length);
    for (let i = 0; i < binStr.length; i++) arr[i] = binStr.charCodeAt(i);
    GRID_DATA[modelKey][day] = arr;
  }
}

function averageModels(modelKeys, day) {
  const arrays = modelKeys.map(mk => GRID_DATA[mk] && GRID_DATA[mk][day]).filter(Boolean);
  if (arrays.length === 0) return null;
  const n = arrays.length;
  const len = arrays[0].length;
  const result = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += arrays[j][i];
    result[i] = sum / n;
  }
  return result;
}

const map = L.map('map').setView(MAP_CENTER, 5);
L.tileLayer(CARTO_URL, {attribution: CARTO_ATTR}).addTo(map);
map.fitBounds(BOUNDS, { animate: false });

const GRID_STEP_LAT = (BOUNDS[1][0] - BOUNDS[0][0]) / (GRID_H - 1);
const GRID_STEP_LON = (BOUNDS[1][1] - BOUNDS[0][1]) / (GRID_W - 1);
const gridRenderer = L.canvas({ padding: 0.1 });

const gridRectangles = new Array(GRID_W * GRID_H);
for (let row = 0; row < GRID_H; row++) {
  const cellLat = BOUNDS[1][0] - row * GRID_STEP_LAT;
  for (let col = 0; col < GRID_W; col++) {
    const cellLon = BOUNDS[0][1] + col * GRID_STEP_LON;
    const cellBounds = [
      [cellLat - GRID_STEP_LAT / 2, cellLon - GRID_STEP_LON / 2],
      [cellLat + GRID_STEP_LAT / 2, cellLon + GRID_STEP_LON / 2]
    ];
    gridRectangles[row * GRID_W + col] = L.rectangle(cellBounds, {
      renderer: gridRenderer,
      stroke: false,
      fillOpacity: 0,
      fillColor: '#000000',
      interactive: false
    });
  }
}
const gridLayerGroup = L.layerGroup(gridRectangles).addTo(map);

let currentOpacity = 0.9;
let currentDataArr = null;

function latLngToGridIndex(lat, lng) {
  const [[latMin, lonMin], [latMax, lonMax]] = BOUNDS;
  if (lat < latMin || lat > latMax || lng < lonMin || lng > lonMax) return null;
  const latFrac = (latMax - lat) / (latMax - latMin);
  const lonFrac = (lng - lonMin) / (lonMax - lonMin);
  let row = Math.floor(latFrac * GRID_H);
  let col = Math.floor(lonFrac * GRID_W);
  row = Math.min(GRID_H - 1, Math.max(0, row));
  col = Math.min(GRID_W - 1, Math.max(0, col));
  return row * GRID_W + col;
}

function updatePointerReadout(e) {
  const readout = document.getElementById('valueReadout');
  if (!currentDataArr) { readout.style.display = 'none'; return; }
  const idx = latLngToGridIndex(e.latlng.lat, e.latlng.lng);
  if (idx === null) { readout.style.display = 'none'; return; }
  const mm = currentDataArr[idx];
  readout.textContent = mm.toFixed(1) + ' mm';
  readout.style.display = 'block';
}

map.on('mousemove', updatePointerReadout);
map.on('click', updatePointerReadout);
map.on('mouseout', () => { document.getElementById('valueReadout').style.display = 'none'; });

let stateBoundaryLayer = null;
let districtBoundaryLayer = null;
let capitalsLayer = null;

function getStateBoundaryLayer() {
  if (!stateBoundaryLayer) {
    stateBoundaryLayer = L.geoJSON(STATE_BOUNDARY_GEOJSON, {
      style: { color: '#333333', weight: 1.5, opacity: 0.8, fill: false }
    });
  }
  return stateBoundaryLayer;
}
function getDistrictBoundaryLayer() {
  if (!districtBoundaryLayer) {
    districtBoundaryLayer = L.geoJSON(DISTRICT_BOUNDARY_GEOJSON, {
      style: { color: '#888888', weight: 0.5, opacity: 0.5, fill: false }
    });
  }
  return districtBoundaryLayer;
}
function getCapitalsLayer() {
  if (!capitalsLayer) {
    capitalsLayer = L.layerGroup();
    STATE_CAPITALS.forEach(c => {
      const marker = L.circleMarker([c.lat, c.lon], {
        radius: 4, color: '#000000', weight: 1, fillColor: '#ff3333', fillOpacity: 1
      });
      marker.bindTooltip(c.capital, { permanent: true, direction: 'right', offset: [5, 0], className: 'capital-label' });
      marker.addTo(capitalsLayer);
    });
  }
  return capitalsLayer;
}

function bringReferenceLayersToFront() {
  if (stateBoundaryLayer && map.hasLayer(stateBoundaryLayer)) stateBoundaryLayer.bringToFront();
  if (districtBoundaryLayer && map.hasLayer(districtBoundaryLayer)) districtBoundaryLayer.bringToFront();
  if (capitalsLayer && map.hasLayer(capitalsLayer)) capitalsLayer.bringToFront();
}

document.getElementById('stateBoundaryToggle').addEventListener('change', (e) => {
  const layer = getStateBoundaryLayer();
  if (e.target.checked) { layer.addTo(map); layer.bringToFront(); } else { map.removeLayer(layer); }
});
document.getElementById('districtBoundaryToggle').addEventListener('change', (e) => {
  const layer = getDistrictBoundaryLayer();
  if (e.target.checked) { layer.addTo(map); layer.bringToFront(); } else { map.removeLayer(layer); }
});
document.getElementById('capitalsToggle').addEventListener('change', (e) => {
  const layer = getCapitalsLayer();
  if (e.target.checked) { layer.addTo(map); layer.bringToFront(); } else { map.removeLayer(layer); }
});

function getActiveModels() {
  const checked = Array.from(document.querySelectorAll('.model-cb:checked')).map(cb => cb.value);
  if (checked.length > 0) return checked;
  const sf = document.querySelector('.sf-radio:checked');
  if (sf) return sf.value.split(',');
  return [];
}

function updateOverlay() {
  const models = getActiveModels();
  const day = document.getElementById('daySelect').value;
  const statusEl = document.getElementById('statusMsg');

  if (models.length === 0) {
    gridRectangles.forEach(rect => rect.setStyle({ fillOpacity: 0 }));
    currentDataArr = null;
    statusEl.textContent = 'Select at least one model to view rainfall.';
    return;
  }
  const avg = averageModels(models, day);
  if (avg === null) {
    gridRectangles.forEach(rect => rect.setStyle({ fillOpacity: 0 }));
    currentDataArr = null;
    statusEl.textContent = 'No data available for this combination on this day.';
    return;
  }
  statusEl.textContent = '';
  currentDataArr = avg;

  for (let i = 0; i < avg.length; i++) {
    const vClamped = Math.min(254, Math.max(0, Math.round(avg[i])));
    const [r, g, b] = COLOR_LUT[vClamped];
    const isDry = vClamped === 0 && COLOR_LUT[0][3] === 0;
    gridRectangles[i].setStyle({
      fillColor: `rgb(${r},${g},${b})`,
      fillOpacity: isDry ? 0 : currentOpacity
    });
  }

  bringReferenceLayersToFront();
}

document.querySelectorAll('.model-cb').forEach(cb => {
  cb.addEventListener('change', () => {
    if (cb.checked) {
      const myPanel = cb.dataset.panel;
      document.querySelectorAll('.model-cb').forEach(other => {
        if (other.dataset.panel !== myPanel) other.checked = false;
      });
      document.querySelectorAll('.sf-radio').forEach(r => r.checked = false);
    }
    updateOverlay();
  });
});
document.querySelectorAll('.sf-radio').forEach(r => {
  r.addEventListener('change', () => {
    document.querySelectorAll('.model-cb').forEach(cb => cb.checked = false);
    updateOverlay();
  });
});
document.getElementById('daySelect').addEventListener('change', updateOverlay);
document.getElementById('opacitySlider').addEventListener('input', (e) => {
  currentOpacity = parseInt(e.target.value, 10) / 100;
  if (currentDataArr) {
    for (let i = 0; i < currentDataArr.length; i++) {
      const vClamped = Math.min(254, Math.max(0, Math.round(currentDataArr[i])));
      if (vClamped !== 0) gridRectangles[i].setStyle({ fillOpacity: currentOpacity });
    }
  }
});
</script>
<a href="index.html" style="position:fixed;left:50%;transform:translateX(-50%);bottom:calc(10px + env(safe-area-inset-bottom,0px));z-index:100000;background:#10252e;color:#fff;font:600 13px/1 system-ui,sans-serif;padding:9px 14px;border-radius:999px;text-decoration:none;box-shadow:0 2px 8px rgba(0,0,0,.3);opacity:.92">&larr; All forecasts</a>
</body>
</html>
"""


def build_map_html(grid_data, day_labels, state_boundary_geojson, district_boundary_geojson, out_path):
    """(unchanged from the notebook's Step 10/11 fix cell, minus the FIX
    comment which was about a bug already resolved before this port and the
    files.download() call, which makes no sense outside Colab -- the
    GitHub Actions workflow's FTP step is this script's equivalent.)"""
    bin_colors = colors[:10] + colors[11:22]
    assert len(bin_colors) == len(levels_24h) - 1

    def _rainfall_color_for(v, levels, bcolors):
        if v < levels[0]:
            return None
        for i in range(len(levels) - 1):
            if levels[i] <= v < levels[i + 1]:
                return bcolors[i]
        return bcolors[-1]

    WET_ALPHA = 217

    color_lut = []
    for v in range(255):
        hexcolor = _rainfall_color_for(v, levels_24h, bin_colors)
        if hexcolor is None:
            color_lut.append([0, 0, 0, 0])
        else:
            r = int(hexcolor[1:3], 16)
            g = int(hexcolor[3:5], 16)
            b = int(hexcolor[5:7], 16)
            color_lut.append([r, g, b, WET_ALPHA])

    legend_rows_html = ""
    for i in range(len(levels_24h) - 1):
        legend_rows_html += (f'<div class="legend-row"><span class="legend-swatch" '
                             f'style="background:{bin_colors[i]};"></span>'
                             f'<span class="legend-label">{levels_24h[i]}-{levels_24h[i+1]} mm</span></div>')

    day_options_html = "".join(f'<option value="{d}">{day_labels[d]}</option>' for d in sorted(day_labels.keys()))

    final_html = HTML_TEMPLATE
    final_html = final_html.replace("__DAY_OPTIONS_HTML__", day_options_html)
    final_html = final_html.replace("__LEGEND_ROWS_HTML__", legend_rows_html)
    final_html = final_html.replace("__COLOR_LUT_JSON__", json.dumps(color_lut))
    final_html = final_html.replace("__GRID_W__", str(GRID_W))
    final_html = final_html.replace("__GRID_H__", str(GRID_H))
    final_html = final_html.replace("__BOUNDS_JSON__", json.dumps(bounds_full))
    final_html = final_html.replace("__MAP_CENTER_JSON__", json.dumps([(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]))
    final_html = final_html.replace("__CARTO_URL_JSON__", json.dumps(CARTO_VOYAGER_URL))
    final_html = final_html.replace("__CARTO_ATTR_JSON__", json.dumps(CARTO_ATTR))
    final_html = final_html.replace("__GRID_DATA_JSON__", json.dumps(grid_data))
    final_html = final_html.replace("__STATE_BOUNDARY_JSON__", json.dumps(state_boundary_geojson, separators=(",", ":")))
    final_html = final_html.replace("__DISTRICT_BOUNDARY_JSON__", json.dumps(district_boundary_geojson, separators=(",", ":")))
    final_html = final_html.replace("__STATE_CAPITALS_JSON__", json.dumps(state_capitals))

    with open(out_path, 'w') as f:
        f.write(final_html)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {out_path} ({size_mb:.1f} MB)")


# === Orchestration ===

def run_pipeline() -> None:
    """One full cycle: fetch all 8 atomic models (best-effort -- a source
    that's down or not-yet-published just means that model is missing from
    the client-side toggle this cycle, matching the notebook's existing
    per-model tolerance in get_atomic_24h), build the toggleable map, and
    write it to output/ for the workflow's FTP step to upload. No state is
    kept between runs -- there's nothing here that depends on the previous
    cycle, unlike the radar nowcast bot."""
    fetch_ecmwf_hres_eps_aifsdet()  # also required before ICON URLs are known (date_str/hour_int)
    init_icon_urls()

    # Each source is independent of the others; a failure in one (a feed
    # that's down, not yet published for this run, or -- for WeatherNext --
    # the anonymous-access assumption turning out to be wrong) shouldn't
    # take down the whole build. get_atomic_24h() already tolerates a
    # missing/never-fetched source per-model (returns None), so the worst
    # case from a caught exception here is that model just doesn't appear
    # as a toggle option this cycle.
    for label, fn in [
        ("AIFS Ensembles", fetch_aifs_ens),
        ("GEM", fetch_gem),
        ("GFS", fetch_gfs),
        ("WeatherNext 3", fetch_weathernext),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"[fetch] {label} failed, this model will be missing from the map this cycle: {e}")

    grid_data, day_labels = build_all_grids()
    state_boundary_geojson, district_boundary_geojson = get_boundaries()

    out_path = OUTPUT_DIR / "toggle_rainfall_map.html"
    build_map_html(grid_data, day_labels, state_boundary_geojson, district_boundary_geojson, str(out_path))


if __name__ == "__main__":
    run_pipeline()

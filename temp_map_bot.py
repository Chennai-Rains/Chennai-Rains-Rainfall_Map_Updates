"""
Automated build of the interactive 7-day temperature map (Daily Max/Min,
6-hourly diurnal cycle, UTCI heat stress) for chennairains.com.

Ported from Temp_7Day_interactive.ipynb (the Colab dev notebook -- keep
using that one for interactive development/debugging of the model logic
itself). This script is only the part meant to run unattended, twice a
day, via GitHub Actions.

This repo previously automated the 7-day rainfall blend map
(rainfall_map_bot.py, now removed) -- that product got pulled back out of
GitHub automation because one of its 8 sources, WeatherNext 3, turned out
to require an individually Google-Form-registered account that a CI
identity can't join without a heavier OAuth setup (see this repo's git
history for the debugging trail). This temperature map only uses ECMWF
HRES and ICON -- no WeatherNext at all -- so it doesn't hit that wall and
can run fully unattended on the same twice-daily schedule from day one.

No cross-run state is needed here, same as the rainfall map was -- every
run downloads everything from scratch and produces one complete,
self-contained map.

One thing changed from the notebook to run headless: the state/district
boundary GeoJSON files are read from this repo's data/ folder (already
committed there from the rainfall map work -- same two files, still
valid) instead of a Google Drive mount, via get_boundaries() below.
"""

import base64
import bz2
import datetime
import json
import os
import time
from pathlib import Path

import eccodes
import numpy as np
import requests
import xarray as xr
from ecmwf.opendata import Client
from scipy.interpolate import griddata

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
ASSETS_DIR = Path("assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Step 1 -- Config: extent, canonical grid, color scale ===
# (unchanged from the notebook)

lon_min, lon_max, lat_min, lat_max = 60, 105, 6, 45
bounds_full = [[lat_min, lon_min], [lat_max, lon_max]]
optimization_area = [47, 58, 4, 107]  # [North, West, South, East], buffered beyond display extent

GRID_STEP = 0.25  # coarser than the rainfall map's 0.1 -- temperature is spatially much smoother
canonical_lon = np.arange(lon_min, lon_max + 1e-6, GRID_STEP)
canonical_lat = np.arange(lat_max, lat_min - 1e-6, -GRID_STEP)
GRID_W, GRID_H = len(canonical_lon), len(canonical_lat)

# Temperature quantization: uint8 with an offset, since temps can in theory dip below 0.
# OFFSET=-10 covers -10C to 244C at 1C resolution -- far more than this region ever needs.
TEMP_OFFSET = -10


def quantize_temp(arr_celsius):
    vals = np.nan_to_num(arr_celsius, nan=-10.0)  # missing -> coldest bin (reads as -10C)
    vals = np.clip(vals - TEMP_OFFSET, 0, 254)
    return vals.round().astype(np.uint8)


CARTO_VOYAGER_URL = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2eki_1_8725e9310ea39758b1e8f374'
CARTO_ATTR = '&copy; <a href="https://carto.com/attributions">CARTO</a> &copy; OpenStreetMap contributors'

MODEL_KEYS = ['hres', 'icon']  # WeatherNext deliberately excluded -- see module docstring

client = Client(source="ecmwf")

# Module-level globals populated by detect_run()/fetch_ecmwf()/init_icon_urls() below --
# same free-variable pattern as the radar bot and the old rainfall_map_bot.py.
run_time = None
date_str = None
hour_int = None
required_steps = []
steps_end = []
steps_start = []
six_hour_steps_full = []
extra_night_step = None

ds_ec_instant = None

icon_base_url = None
icon_run_str = None
_icon_target_lon = None
_icon_target_lat = None
_icon_grid_lon = None
_icon_grid_lat = None
_icon_coords_cache = {}

utci_all_steps = None
steps_utci = []


def _to_canonical(arr):
    return arr.interp(latitude=canonical_lat, longitude=canonical_lon)


# === Step 2 -- Run detection (same probe-based approach as the rainfall map) ===

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


def detect_run():
    global run_time, date_str, hour_int, required_steps, steps_end, steps_start
    global six_hour_steps_full, extra_night_step
    date_str, hour_int, required_steps, steps_end, steps_start = get_forecast_data()
    run_time = datetime.datetime.strptime(f"{date_str} {hour_int}", "%Y%m%d %H")
    # 6-hourly steps spanning the full Day 1-7 range, for the diurnal-cycle product
    six_hour_steps_full = list(range(steps_start[0], steps_end[6] + 1, 6))
    # One extra step beyond Day 7's boundary, needed only to close out Day 7's overnight
    # (5:30 PM -> 5:30 AM) window for the Daily Min calc -- not exposed in the 28-block slider.
    extra_night_step = steps_end[6] + 6
    print(f"6-hourly steps (Day 1-7): {six_hour_steps_full}")
    print(f"Extra step for Day 7's overnight window: {extra_night_step}")
    print(f"Daily boundary steps: starts={steps_start}, ends={steps_end}")


# === Step 3 -- ECMWF HRES (instantaneous 2m temperature, 6-hourly across Day 1-7) ===

def fetch_ecmwf_instant():
    global ds_ec_instant
    print("Downloading ECMWF 2t (instantaneous, 6-hourly)...")
    client.retrieve(date=date_str, time=hour_int, type="fc", param="2t",
                     step=six_hour_steps_full, target='ecmwf_t2_instant.grib',
                     area=optimization_area, grid=[0.25, 0.25])
    ds_ec_instant = xr.open_dataset('ecmwf_t2_instant.grib', engine='cfgrib',
                                     backend_kwargs={'indexpath': ''})


def get_ecmwf_instant(step):
    try:
        val = ds_ec_instant['t2m'].sel(step=np.timedelta64(step, 'h')) - 273.15
        return _to_canonical(val)
    except Exception as e:
        print(f"ECMWF instant T2 step {step} failed: {e}")
        return None


# === Step 4 -- ICON (instantaneous T_2M, 6-hourly across Day 1-7) ===

def init_icon_urls():
    global icon_base_url, icon_run_str, _icon_target_lon, _icon_target_lat
    global _icon_grid_lon, _icon_grid_lat, _icon_coords_cache
    icon_base_url = f"https://opendata.dwd.de/weather/nwp/icon/grib/{hour_int:02d}"
    icon_run_str = f"{date_str}{hour_int:02d}"
    _icon_target_lon = np.arange(optimization_area[1], optimization_area[3] + 0.1, 0.1)
    _icon_target_lat = np.arange(optimization_area[2], optimization_area[0] + 0.1, 0.1)
    _icon_grid_lon, _icon_grid_lat = np.meshgrid(_icon_target_lon, _icon_target_lat)
    _icon_coords_cache = {}


def _download_icon_file(url_path, local_name):
    if os.path.exists(local_name):
        return local_name
    url = f"{icon_base_url}/{url_path}.bz2"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None
        with open(local_name, 'wb') as f:
            f.write(bz2.decompress(r.content))
        return local_name
    except Exception:
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
    return _icon_coords_cache['coords']


def _get_icon_field(param_dir, param_file, step):
    icon_lats, icon_lons, mask = get_icon_grid_coords()
    step_str = f"{step:03d}"
    fname = f"icon_global_icosahedral_single-level_{icon_run_str}_{step_str}_{param_file}.grib2"
    path = _download_icon_file(f"{param_dir}/{fname}", fname)
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
    if float(np.nanmean(regridded)) > 100:
        da = da - 273.15
    return da


def get_icon_instant(step):
    val = _get_icon_field('t_2m', 'T_2M', step)
    return _to_canonical(val) if val is not None else None


# === Step 5 -- UTCI (heat stress index, ECMWF only) ===

def fetch_utci():
    global utci_all_steps, steps_utci
    print("Downloading ECMWF UTCI components (2t/2d/10u/10v/ssrd, 3-hourly, Day 1-7)...")
    # HRES publishes 3-hourly steps only through 144h, then 6-hourly beyond that.
    steps_utci = list(range(0, 144, 3)) + list(range(144, steps_end[6] + 1, 6))
    client.retrieve(type='fc', date=date_str, time=hour_int, step=steps_utci,
                     param=['2t', '2d', '10u', '10v', 'ssrd'],
                     target='utci_data_full.grib2', area=optimization_area, grid=[0.25, 0.25])

    common_args = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
    ds_2m = xr.open_dataset('utci_data_full.grib2', **common_args,
                             filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
    ds_10m = xr.open_dataset('utci_data_full.grib2', **common_args,
                              filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 10})
    ds_surf = xr.open_dataset('utci_data_full.grib2', **common_args,
                               filter_by_keys={'typeOfLevel': 'surface'})

    T_utci = ds_2m['t2m'] - 273.15
    Td_utci = ds_2m['d2m'] - 273.15
    va_utci = np.sqrt(ds_10m['u10'] ** 2 + ds_10m['v10'] ** 2) * 0.74

    ssrd = ds_surf['ssrd']
    ssrd_diff = ssrd.diff('step') / 10800.0
    first_step_flux = (ssrd.isel(step=[0]) / 10800.0)
    rad_flux = xr.concat([first_step_flux, ssrd_diff], dim='step')
    rad_flux = rad_flux.assign_coords(step=ssrd.step)

    e_utci = 6.112 * np.exp(17.67 * Td_utci / (Td_utci + 243.5))
    utci_all_steps = (T_utci + (0.0055 * T_utci ** 2) + (0.13 * T_utci) - (0.15 * va_utci) +
                      (0.01 * e_utci) + (0.002 * rad_flux) - 1.2)
    print("UTCI computed at all 3-hourly steps.")


def get_utci_daily_max(day_idx):
    """Daily max UTCI for day_idx (0-6), using this pipeline's own day boundaries rather than
    calendar-day resampling, so it lines up exactly with Day 1-7 everywhere else here."""
    start_s, end_s = steps_start[day_idx], steps_end[day_idx]
    day_steps = [s for s in steps_utci if start_s < s <= end_s]
    if not day_steps:
        return None
    try:
        day_slice = utci_all_steps.sel(step=[np.timedelta64(s, 'h') for s in day_steps])
        daily_max = day_slice.max(dim='step')
        return _to_canonical(daily_max)
    except Exception as e:
        print(f"UTCI daily max for day {day_idx + 1} failed: {e}")
        return None


# === Step 6 -- Unified instantaneous dispatch (ECMWF + ICON only) ===

def get_instant(model_key, step):
    try:
        if model_key == 'hres':
            return get_ecmwf_instant(step)
        if model_key == 'icon':
            return get_icon_instant(step)
    except Exception as e:
        print(f"get_instant({model_key}, {step}) failed: {e}")
    return None


# === Step 7 -- Build instantaneous grids, then derive Daily Max/Min + UTCI ===

def build_all_grids():
    """Daily Max = highest reading in the 12h daytime window (5:30 AM-5:30 PM).
    Daily Min = lowest reading in the 12h nighttime window (5:30 PM-5:30 AM next day),
    closed out for Day 7 by the separately-fetched extra_night_step."""
    instant_data = {}
    instant_block_labels = {}
    block_to_day = {}
    IST_OFFSET = datetime.timedelta(hours=5, minutes=30)

    for idx, step in enumerate(six_hour_steps_full):
        block_num = idx + 1
        block_ist = run_time + datetime.timedelta(hours=step) + IST_OFFSET
        day_num_for_block = next((i + 1 for i in range(7) if steps_start[i] < step <= steps_end[i]), None)
        instant_block_labels[block_num] = f"Day {day_num_for_block}: {block_ist.strftime('%I:%M %p, %d %b')}"
        block_to_day[block_num] = day_num_for_block

        for model_key in MODEL_KEYS:
            arr = get_instant(model_key, step)
            if arr is not None:
                instant_data.setdefault(model_key, {})[block_num] = quantize_temp(arr.values)

    print(f"Instantaneous grids generated: { {mk: len(v) for mk, v in instant_data.items()} }")

    extra_night_data = {}
    for model_key in MODEL_KEYS:
        arr = get_instant(model_key, extra_night_step)
        if arr is not None:
            extra_night_data[model_key] = quantize_temp(arr.values)
    print(f"Extra overnight-closing step fetched for: {list(extra_night_data.keys())}")

    daily_max_data = {}
    daily_min_data = {}
    day_labels = {}

    for day_num in range(1, 8):
        day_blocks = sorted([b for b, d in block_to_day.items() if d == day_num])
        if len(day_blocks) < 4:
            print(f"Day {day_num}: expected 4 blocks, found {len(day_blocks)} -- skipping.")
            continue
        b_530am, b_1130am, b_530pm, b_1130pm = day_blocks

        next_day_blocks = sorted([b for b, d in block_to_day.items() if d == day_num + 1])
        next_530am = next_day_blocks[0] if next_day_blocks else None  # None only for Day 7

        start_ist = run_time + datetime.timedelta(hours=steps_start[day_num - 1]) + IST_OFFSET
        day_labels[day_num] = f"Day {day_num} ({start_ist.strftime('%d %b')})"

        for model_key in MODEL_KEYS:
            m = instant_data.get(model_key, {})

            daytime_arrays = [m[b] for b in (b_530am, b_1130am, b_530pm) if b in m]
            if daytime_arrays:
                tmax = np.maximum.reduce(daytime_arrays)
                daily_max_data.setdefault(model_key, {})[day_num] = base64.b64encode(tmax.tobytes()).decode('ascii')

            nighttime_arrays = [m[b] for b in (b_530pm, b_1130pm) if b in m]
            if next_530am is not None and next_530am in m:
                nighttime_arrays.append(m[next_530am])
            elif next_530am is None and model_key in extra_night_data:
                nighttime_arrays.append(extra_night_data[model_key])
            if nighttime_arrays:
                tmin = np.minimum.reduce(nighttime_arrays)
                daily_min_data.setdefault(model_key, {})[day_num] = base64.b64encode(tmin.tobytes()).decode('ascii')

    print(f"Daily max (5:30AM-5:30PM peak) grids: { {mk: len(v) for mk, v in daily_max_data.items()} }")
    print(f"Daily min (5:30PM-5:30AM trough) grids: { {mk: len(v) for mk, v in daily_min_data.items()} }")

    for model_key in instant_data:
        for block_num in instant_data[model_key]:
            instant_data[model_key][block_num] = base64.b64encode(
                instant_data[model_key][block_num].tobytes()).decode('ascii')

    # UTCI: ECMWF only, nested the same way as the other fields so the JS lookup stays uniform.
    utci_data = {}
    for day_idx in range(7):
        day_num = day_idx + 1
        arr = get_utci_daily_max(day_idx)
        if arr is not None:
            utci_data.setdefault('hres', {})[day_num] = base64.b64encode(
                quantize_temp(arr.values).tobytes()).decode('ascii')
    print(f"UTCI daily-max grids: { {mk: len(v) for mk, v in utci_data.items()} }")

    return {
        "daily_max_data": daily_max_data,
        "daily_min_data": daily_min_data,
        "instant_data": instant_data,
        "utci_data": utci_data,
        "day_labels": day_labels,
        "instant_block_labels": instant_block_labels,
    }


# === Step 8 -- State/district boundaries + state capitals ===

_EMPTY_FEATURE_COLLECTION = {"type": "FeatureCollection", "features": []}


def get_boundaries():
    """Same two pre-processed GeoJSON files already committed to data/ for the rainfall map
    (rounded to 5dp, trimmed to just the attribute fields actually used) -- reused as-is here,
    they're generic India state/district boundaries, not rainfall-specific.

    Falls back to an empty FeatureCollection (map still builds, just without the boundary
    toggle layers) if the files aren't there for some reason."""
    state_boundary_path = DATA_DIR / "India_State_Boundary.json"
    district_boundary_path = DATA_DIR / "India_District_Boundary.json"

    if state_boundary_path.exists():
        with open(state_boundary_path) as f:
            state_boundary_geojson = json.load(f)
        print(f"State boundaries: {len(state_boundary_geojson['features'])} features")
    else:
        print(f"[boundaries] {state_boundary_path} not found -- state boundary layer will be empty.")
        state_boundary_geojson = _EMPTY_FEATURE_COLLECTION

    if district_boundary_path.exists():
        with open(district_boundary_path) as f:
            district_boundary_geojson = json.load(f)
        print(f"District boundaries: {len(district_boundary_geojson['features'])} features")
    else:
        print(f"[boundaries] {district_boundary_path} not found -- district boundary layer will be empty.")
        district_boundary_geojson = _EMPTY_FEATURE_COLLECTION

    return state_boundary_geojson, district_boundary_geojson


STATE_CAPITALS = [
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


# === Step 9 -- Branding: logo + attribution watermark ===
# Same approach as the radar nowcast bot (see Chennai-Rains/IMD_Radar_Updates) -- base64-embedded
# so the map stays one self-contained file, watermark tiled faintly so a screenshot still carries
# attribution and can't just be cropped from a corner.

LOGO_PATH = ASSETS_DIR / "chennairains_logo.jpg"


def _logo_data_uri():
    if not LOGO_PATH.exists():
        print(f"Logo asset not found at {LOGO_PATH} -- skipping branding.")
        return None
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_logo_tag(height_px=26):
    logo_uri = _logo_data_uri()
    if not logo_uri:
        return ""
    return (
        f'<a href="https://www.chennairains.com" target="_blank" rel="noopener" '
        f'style="text-decoration:none; flex-shrink:0;">'
        f'<img src="{logo_uri}" alt="ChennaiRains" '
        f'style="height:{height_px}px; width:{height_px}px; display:block; '
        f'border-radius:6px;"></a>'
    )


def build_watermark_data_uri(text="Temperature map by www.chennairains.com"):
    watermark_svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='460' height='260'>
        <text x='230' y='135' transform='rotate(-28 230 135)'
              font-family='Arial, sans-serif' font-size='13'
              fill='rgba(0,0,0,0.14)' text-anchor='middle'
              font-weight='600'>{text}</text>
    </svg>
    """
    return "data:image/svg+xml;base64," + base64.b64encode(watermark_svg.encode("utf-8")).decode("ascii")


# === Step 10 -- Build the client-side map: single-model select + field-type select ===
# (HTML/JS unchanged from the notebook, aside from the __LOGO_TAG__/__WATERMARK_URI__
# placeholders added for branding)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Temperature Map (T2 Max/Min/Diurnal)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin:0; padding:0; height:100%; font-family: sans-serif; }
  #map { position:absolute; top:0; bottom:0; left:0; right:340px; }
  #sidebar {
    position:absolute; top:0; right:0; width:340px; height:100%; overflow-y:auto;
    background:#fff; box-shadow:-2px 0 6px rgba(0,0,0,0.15); padding:14px; box-sizing:border-box;
  }
  #sidebarTitle { display:flex; align-items:center; gap:8px; margin-top:0; }
  #watermark {
    position:absolute; inset:0; z-index:650; pointer-events:none;
    background-image: url('__WATERMARK_URI__'); background-repeat: repeat;
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
  #timeLabel { text-align:center; font-size:13px; font-weight:600; margin-top:8px; color:#222; }
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
<div id="map"><div id="valueReadout"></div><div id="watermark"></div></div>
<div id="sidebar">
  <h3 id="sidebarTitle">__LOGO_TAG__<span>Temperature (T2 Max / Min / Diurnal)</span></h3>

  <div class="panel">
    <div class="panel-title">Model</div>
    <label><input type="radio" name="modelSelect" value="hres" checked> ECMWF HRES</label>
    <label><input type="radio" name="modelSelect" value="icon"> ICON</label>
  </div>

  <div class="panel">
    <div class="panel-title">Field</div>
    <label><input type="radio" name="fieldSelect" value="daily_max" checked> Daily Max</label>
    <label><input type="radio" name="fieldSelect" value="daily_min"> Daily Min</label>
    <label><input type="radio" name="fieldSelect" value="diurnal"> 6-Hourly Diurnal Cycle</label>
    <label><input type="radio" name="fieldSelect" value="utci"> UTCI (Heat Stress)</label>
    <p style="font-size:11px; color:#888; margin:4px 0 0 0;">UTCI is only available for ECMWF HRES.</p>
  </div>

  <div class="panel" id="dayPanel">
    <div class="panel-title">Day</div>
    <select id="daySelect">__DAY_OPTIONS_HTML__</select>
  </div>

  <div class="panel" id="timePanel" style="display:none;">
    <div class="panel-title">Time</div>
    <input type="range" id="timeSlider" min="1" max="__NUM_BLOCKS__" value="1" step="1" style="width:100%;">
    <div id="timeLabel">__FIRST_BLOCK_LABEL__</div>
  </div>

  <div class="panel">
    <div class="panel-title">Reference Layers</div>
    <label><input type="checkbox" id="stateBoundaryToggle"> State Boundaries</label>
    <label><input type="checkbox" id="districtBoundaryToggle"> District Boundaries</label>
    <label><input type="checkbox" id="capitalsToggle"> State Capitals</label>
  </div>

  <div class="panel">
    <div class="panel-title">Transparency</div>
    <input type="range" id="opacitySlider" min="0" max="100" value="90" style="width:100%;">
    <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;">
      <span>See-through</span><span>Opaque</span>
    </div>
  </div>

  <div id="statusMsg"></div>

  <div id="tempLegend">
    <div class="panel-title">Temperature (&deg;C)</div>
    __LEGEND_ROWS_HTML__
  </div>
  <div id="utciLegend" style="display:none;">
    <div class="panel-title">UTCI Heat Stress Category</div>
    __UTCI_LEGEND_ROWS_HTML__
  </div>
  <p style="font-size:10px; color:#999; margin-top:14px; line-height:1.4;">
    Experimental product. Based on ECMWF HRES / ICON model output. Not an official forecast.
  </p>
</div>

<script>
const TEMP_COLOR_LUT = __TEMP_COLOR_LUT_JSON__;
const TEMP_OFFSET = __TEMP_OFFSET__;
const GRID_W = __GRID_W__;
const GRID_H = __GRID_H__;
const BOUNDS = __BOUNDS_JSON__;
const GRID_STEP_LAT = (BOUNDS[1][0] - BOUNDS[0][0]) / (GRID_H - 1);
const GRID_STEP_LON = (BOUNDS[1][1] - BOUNDS[0][1]) / (GRID_W - 1);
const MAP_CENTER = __MAP_CENTER_JSON__;
const CARTO_URL = __CARTO_URL_JSON__;
const CARTO_ATTR = __CARTO_ATTR_JSON__;
const DAILY_MAX_DATA_B64 = __DAILY_MAX_JSON__;
const DAILY_MIN_DATA_B64 = __DAILY_MIN_JSON__;
const INSTANT_DATA_B64 = __INSTANT_JSON__;
const STATE_BOUNDARY_GEOJSON = __STATE_BOUNDARY_JSON__;
const DISTRICT_BOUNDARY_GEOJSON = __DISTRICT_BOUNDARY_JSON__;
const STATE_CAPITALS = __STATE_CAPITALS_JSON__;
const BLOCK_LABELS = __BLOCK_LABELS_JSON__;
const UTCI_COLOR_LUT = __UTCI_COLOR_LUT_JSON__;
const UTCI_DATA_B64 = __UTCI_JSON__;
const UTCI_CATEGORY_BOUNDS = __UTCI_CATEGORY_BOUNDS_JSON__;

function decodeAll(dataB64) {
  const out = {};
  for (const model in dataB64) {
    out[model] = {};
    for (const key in dataB64[model]) {
      const binStr = atob(dataB64[model][key]);
      const arr = new Uint8Array(binStr.length);
      for (let i = 0; i < binStr.length; i++) arr[i] = binStr.charCodeAt(i);
      out[model][key] = arr;
    }
  }
  return out;
}
const DAILY_MAX_DATA = decodeAll(DAILY_MAX_DATA_B64);
const DAILY_MIN_DATA = decodeAll(DAILY_MIN_DATA_B64);
const INSTANT_DATA = decodeAll(INSTANT_DATA_B64);
const UTCI_DATA = decodeAll(UTCI_DATA_B64);

const map = L.map('map').setView(MAP_CENTER, 5);
L.tileLayer(CARTO_URL, {attribution: CARTO_ATTR}).addTo(map);
map.fitBounds(BOUNDS);

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
let currentField = 'daily_max';

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

function utciCategoryLabel(tempC) {
  for (const [lo, hi, label] of UTCI_CATEGORY_BOUNDS) {
    if (tempC >= lo && tempC < hi) return label;
  }
  return UTCI_CATEGORY_BOUNDS[UTCI_CATEGORY_BOUNDS.length - 1][2];
}

function updatePointerReadout(e) {
  const readout = document.getElementById('valueReadout');
  if (!currentDataArr) { readout.style.display = 'none'; return; }
  const idx = latLngToGridIndex(e.latlng.lat, e.latlng.lng);
  if (idx === null) { readout.style.display = 'none'; return; }
  const tempC = currentDataArr[idx] + TEMP_OFFSET;
  let text = tempC.toFixed(1) + '°C';
  if (currentField === 'utci') {
    text += ' (' + utciCategoryLabel(tempC) + ')';
  }
  readout.textContent = text;
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

function getSelectedModel() {
  const el = document.querySelector('input[name="modelSelect"]:checked');
  return el ? el.value : 'hres';
}
function getSelectedField() {
  const el = document.querySelector('input[name="fieldSelect"]:checked');
  return el ? el.value : 'daily_max';
}

function updateOverlay() {
  const model = getSelectedModel();
  const field = getSelectedField();
  const statusEl = document.getElementById('statusMsg');
  let dataArr = null;
  let lut = TEMP_COLOR_LUT;

  if (field === 'diurnal') {
    const block = document.getElementById('timeSlider').value;
    dataArr = INSTANT_DATA[model] && INSTANT_DATA[model][block];
  } else if (field === 'utci') {
    const day = document.getElementById('daySelect').value;
    dataArr = UTCI_DATA[model] && UTCI_DATA[model][day];
    lut = UTCI_COLOR_LUT;
  } else {
    const day = document.getElementById('daySelect').value;
    const source = field === 'daily_max' ? DAILY_MAX_DATA : DAILY_MIN_DATA;
    dataArr = source[model] && source[model][day];
  }

  if (!dataArr) {
    gridRectangles.forEach(rect => rect.setStyle({ fillOpacity: 0 }));
    currentDataArr = null;
    statusEl.textContent = 'No data available for this model/field/time combination.';
    return;
  }
  statusEl.textContent = '';
  currentDataArr = dataArr;
  currentField = field;

  for (let i = 0; i < dataArr.length; i++) {
    const [r, g, b] = lut[dataArr[i]];
    gridRectangles[i].setStyle({
      fillColor: `rgb(${r},${g},${b})`,
      fillOpacity: currentOpacity
    });
  }

  bringReferenceLayersToFront();
}

document.querySelectorAll('input[name="modelSelect"]').forEach(r => r.addEventListener('change', updateOverlay));
document.querySelectorAll('input[name="fieldSelect"]').forEach(r => {
  r.addEventListener('change', () => {
    const field = getSelectedField();
    const isDiurnal = field === 'diurnal';
    document.getElementById('dayPanel').style.display = isDiurnal ? 'none' : 'block';
    document.getElementById('timePanel').style.display = isDiurnal ? 'block' : 'none';
    document.getElementById('tempLegend').style.display = field === 'utci' ? 'none' : 'block';
    document.getElementById('utciLegend').style.display = field === 'utci' ? 'block' : 'none';
    updateOverlay();
  });
});
document.getElementById('daySelect').addEventListener('change', updateOverlay);
document.getElementById('timeSlider').addEventListener('input', (e) => {
  document.getElementById('timeLabel').textContent = BLOCK_LABELS[e.target.value];
  updateOverlay();
});
document.getElementById('opacitySlider').addEventListener('input', (e) => {
  currentOpacity = parseInt(e.target.value, 10) / 100;
  if (currentDataArr) {
    gridRectangles.forEach(rect => rect.setStyle({ fillOpacity: currentOpacity }));
  }
});

updateOverlay();
</script>
<a href="index.html" style="position:fixed;left:50%;transform:translateX(-50%);bottom:calc(10px + env(safe-area-inset-bottom,0px));z-index:100000;background:#10252e;color:#fff;font:600 13px/1 system-ui,sans-serif;padding:9px 14px;border-radius:999px;text-decoration:none;box-shadow:0 2px 8px rgba(0,0,0,.3);opacity:.92">&larr; All forecasts</a>
</body>
</html>
"""


def build_map_html(grids, day_labels, state_boundary_geojson, district_boundary_geojson, out_path):
    import matplotlib
    import matplotlib.colors as mcolors

    temp_cmap = matplotlib.colormaps.get_cmap('RdYlBu_r')
    temp_norm = mcolors.Normalize(vmin=10, vmax=45)
    alpha = 217

    temp_color_lut = []
    for byte in range(255):
        temp_c = byte + TEMP_OFFSET
        r, g, b, _ = temp_cmap(temp_norm(temp_c))
        temp_color_lut.append([round(r * 255), round(g * 255), round(b * 255), alpha])

    legend_levels_c = [10, 15, 20, 25, 30, 35, 40, 45]
    legend_rows_html = ""
    for i in range(len(legend_levels_c) - 1):
        mid = (legend_levels_c[i] + legend_levels_c[i + 1]) / 2
        r, g, b, _ = temp_cmap(temp_norm(mid))
        hexcolor = mcolors.to_hex((r, g, b))
        legend_rows_html += (f'<div class="legend-row"><span class="legend-swatch" '
                              f'style="background:{hexcolor};"></span>'
                              f'<span class="legend-label">{legend_levels_c[i]}-{legend_levels_c[i + 1]} C</span></div>')

    # UTCI: official thermal-stress categories (Brode et al.) -- an international standard,
    # not something to re-derive from our own data.
    UTCI_CATEGORIES = [
        (-100, -40, '#4B0082', 'Extreme cold stress'),
        (-40, -27, '#0000CD', 'Very strong cold stress'),
        (-27, -13, '#4169E1', 'Strong cold stress'),
        (-13, 0, '#87CEEB', 'Moderate cold stress'),
        (0, 9, '#B0E0E6', 'Slight cold stress'),
        (9, 26, '#90EE90', 'No thermal stress'),
        (26, 32, '#FFFF00', 'Moderate heat stress'),
        (32, 38, '#FFA500', 'Strong heat stress'),
        (38, 46, '#FF4500', 'Very strong heat stress'),
        (46, 200, '#8B0000', 'Extreme heat stress'),
    ]

    def _categorize_utci(temp_c):
        for lo, hi, color, label in UTCI_CATEGORIES:
            if lo <= temp_c < hi:
                return color
        return UTCI_CATEGORIES[-1][2]

    utci_color_lut = []
    for byte in range(255):
        temp_c = byte + TEMP_OFFSET
        hexcolor = _categorize_utci(temp_c)
        r = int(hexcolor[1:3], 16)
        g = int(hexcolor[3:5], 16)
        b = int(hexcolor[5:7], 16)
        utci_color_lut.append([r, g, b, alpha])

    utci_legend_rows_html = ""
    for lo, hi, color, label in UTCI_CATEGORIES:
        utci_legend_rows_html += (f'<div class="legend-row"><span class="legend-swatch" '
                                   f'style="background:{color};"></span>'
                                   f'<span class="legend-label">{label}</span></div>')

    day_options_html = "".join(f'<option value="{d}">{day_labels[d]}</option>' for d in sorted(day_labels.keys()))

    final_html = HTML_TEMPLATE
    final_html = final_html.replace("__LOGO_TAG__", build_logo_tag())
    final_html = final_html.replace("__WATERMARK_URI__", build_watermark_data_uri())
    final_html = final_html.replace("__DAY_OPTIONS_HTML__", day_options_html)
    final_html = final_html.replace("__NUM_BLOCKS__", str(len(six_hour_steps_full)))
    final_html = final_html.replace("__FIRST_BLOCK_LABEL__", grids["instant_block_labels"].get(1, ""))
    final_html = final_html.replace("__LEGEND_ROWS_HTML__", legend_rows_html)
    final_html = final_html.replace("__TEMP_COLOR_LUT_JSON__", json.dumps(temp_color_lut))
    final_html = final_html.replace("__TEMP_OFFSET__", str(TEMP_OFFSET))
    final_html = final_html.replace("__GRID_W__", str(GRID_W))
    final_html = final_html.replace("__GRID_H__", str(GRID_H))
    final_html = final_html.replace("__BOUNDS_JSON__", json.dumps(bounds_full))
    final_html = final_html.replace("__MAP_CENTER_JSON__", json.dumps([(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]))
    final_html = final_html.replace("__CARTO_URL_JSON__", json.dumps(CARTO_VOYAGER_URL))
    final_html = final_html.replace("__CARTO_ATTR_JSON__", json.dumps(CARTO_ATTR))
    final_html = final_html.replace("__DAILY_MAX_JSON__", json.dumps(grids["daily_max_data"]))
    final_html = final_html.replace("__DAILY_MIN_JSON__", json.dumps(grids["daily_min_data"]))
    final_html = final_html.replace("__INSTANT_JSON__", json.dumps(grids["instant_data"]))
    final_html = final_html.replace("__STATE_BOUNDARY_JSON__", json.dumps(state_boundary_geojson, separators=(",", ":")))
    final_html = final_html.replace("__DISTRICT_BOUNDARY_JSON__", json.dumps(district_boundary_geojson, separators=(",", ":")))
    final_html = final_html.replace("__STATE_CAPITALS_JSON__", json.dumps(STATE_CAPITALS))
    final_html = final_html.replace("__BLOCK_LABELS_JSON__", json.dumps(grids["instant_block_labels"]))
    final_html = final_html.replace("__UTCI_COLOR_LUT_JSON__", json.dumps(utci_color_lut))
    final_html = final_html.replace("__UTCI_JSON__", json.dumps(grids["utci_data"]))
    final_html = final_html.replace("__UTCI_LEGEND_ROWS_HTML__", utci_legend_rows_html)
    final_html = final_html.replace("__UTCI_CATEGORY_BOUNDS_JSON__",
                                     json.dumps([[lo, hi, label] for lo, hi, color, label in UTCI_CATEGORIES]))

    with open(out_path, 'w') as f:
        f.write(final_html)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {out_path} ({size_mb:.1f} MB)")


# === Orchestration ===

def run_pipeline() -> None:
    """One full cycle: detect the latest run, fetch HRES + ICON instantaneous fields, derive
    Daily Max/Min + UTCI, build the map, write it to output/ for the workflow's FTP step."""
    detect_run()
    fetch_ecmwf_instant()
    init_icon_urls()

    try:
        fetch_utci()
    except Exception as e:
        print(f"[fetch] UTCI failed, that field will be missing from the map this cycle: {e}")

    grids = build_all_grids()
    state_boundary_geojson, district_boundary_geojson = get_boundaries()

    out_path = OUTPUT_DIR / "temperature_map.html"
    build_map_html(grids, grids["day_labels"], state_boundary_geojson, district_boundary_geojson, str(out_path))


if __name__ == "__main__":
    run_pipeline()

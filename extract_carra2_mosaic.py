"""
extract_carra2_mosaic.py

Standalone script: extracts one or more CARRA2 variables that share the same
native grid (e.g. t2m + wspd10) over a lat/lon crop around the MOSAiC drift
track, for an arbitrary date range, into a single combined netCDF.

Meant to be run as a subprocess to avoid potential notebook crashes that happened with
xarray in jupyter notebook. Using xarray materialised the entire field in memory,
regardless of the crop, which caused repetaed memory crashes.

eccodes sidesteps this process, the peak memory is bounded by one full grid message (~65 MB)

Variables passed together MUST share a native grid (same Ny/Nx) - this is
checked and will raise a clear error rather than silently combining
mismatched grids. sif is on a different grid than t2m/wspd10 in this
dataset, so extract it with a separate call/output file.
"""
import sys, json
import numpy as np
import pandas as pd
import xarray as xr
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release


def log(tag, **extra):
    """Lightweight memory/progress logging. Kept dependency-light (psutil
    imported locally) so this file can be copied somewhere without psutil
    and still run, just without the memory numbers."""
    extras = " ".join(f"{k}={v}" for k, v in extra.items())
    try:
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"[log] {tag}: RSS={rss:.1f} MB {extras}", flush=True)
    except ImportError:
        print(f"[log] {tag}: {extras}", flush=True)


def extract_cropped_multivar(file_dir, shortname_variable_map, time_start, time_end,
                              min_lat, max_lat, min_lon, max_lon):
    """
    Single pass through the GRIB file, pulling out every shortName in
    shortname_variable_map (e.g. {"2t": "t2m", "10si": "wspd10"}) that falls
    inside [time_start, time_end] (inclusive of the whole end day), cropped
    to the given bbox.

    One pass instead of one-per-variable halves the number of full-file
    metadata scans versus calling this once per variable separately.

    Returns an xr.Dataset with one data variable per entry in
    shortname_variable_map, sharing lat/lon/time coords.
    """
    start_ts = pd.Timestamp(time_start)
    end_ts = pd.Timestamp(time_end) + pd.Timedelta(days=1)

    times = {name: [] for name in shortname_variable_map.values()}
    fields = {name: [] for name in shortname_variable_map.values()}
    grid_shape = None          # (ny, nx) - checked against every message
    lat_crop = lon_crop = None
    y0 = y1 = x0 = x1 = None
    n_seen = {sn: 0 for sn in shortname_variable_map}
    n_kept = {sn: 0 for sn in shortname_variable_map}

    with open(file_dir, 'rb') as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                short = codes_get(gid, 'shortName')
                if short not in shortname_variable_map:
                    continue  # cheap - no array decode happens for skipped messages
                n_seen[short] += 1

                data_date = codes_get(gid, 'dataDate')   # e.g. 20200110
                data_time = codes_get(gid, 'dataTime')    # e.g. 300 -> 03:00
                valid_dt = pd.Timestamp(str(data_date)) + pd.Timedelta(hours=data_time // 100)
                if not (start_ts <= valid_dt < end_ts):
                    continue  # skip - still no array decode for out-of-window messages

                ny = codes_get(gid, 'Ny')
                nx = codes_get(gid, 'Nx')

                if grid_shape is None:
                    grid_shape = (ny, nx)
                elif (ny, nx) != grid_shape:
                    # Flag loudly rather than silently combining mismatched grids -
                    # this would otherwise produce a corrupted-looking but
                    # silently-wrong combined file.
                    raise ValueError(
                        f"Grid mismatch: '{short}' is {ny}x{nx}, expected {grid_shape} "
                        f"(from an earlier variable in this call). These variables don't "
                        f"share a native grid - extract them with separate calls/files."
                    )

                if lat_crop is None:
                    # Grid + crop bbox are fixed across all messages - compute once.
                    lats = np.array(codes_get_array(gid, 'latitudes')).reshape(ny, nx)
                    lons = (np.array(codes_get_array(gid, 'longitudes')).reshape(ny, nx) + 180) % 360 - 180
                    mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
                    yy, xx = np.where(mask)
                    if yy.size == 0:
                        raise ValueError("No grid cells matched the crop bounds - check min/max lat/lon.")
                    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
                    lat_crop = lats[y0:y1, x0:x1]
                    lon_crop = lons[y0:y1, x0:x1]
                    del lats, lons  # full-grid arrays - drop immediately once cropped

                values = np.array(codes_get_array(gid, 'values')).reshape(ny, nx)
                save_name = shortname_variable_map[short]
                fields[save_name].append(values[y0:y1, x0:x1].copy())
                times[save_name].append(valid_dt)
                n_kept[short] += 1
                del values
            finally:
                codes_release(gid)  # always release, even on skip/error - avoids leaking eccodes handles

    for sn in shortname_variable_map:
        print(f"  {sn}: scanned {n_seen[sn]}, kept {n_kept[sn]} in window", flush=True)
        if n_kept[sn] == 0:
            raise ValueError(f"No '{sn}' messages found in {time_start}..{time_end}")

    data_vars = {}
    ref_times = None
    for save_name, tlist in times.items():
        order = np.argsort(tlist)
        sorted_times = np.array(tlist)[order]
        if ref_times is None:
            ref_times = sorted_times
        elif not np.array_equal(sorted_times, ref_times):
            # Sharing a grid doesn't guarantee identical timestamps across
            # variables (e.g. one has a gap). Flagging this rather than
            # silently misaligning two variables under one time coordinate.
            print(f"  WARNING: '{save_name}' timestamps differ from the first "
                  f"variable's. Check for missing/extra messages before trusting "
                  f"this combined file's alignment.", flush=True)
        stacked = np.stack(fields[save_name])[order]
        data_vars[save_name] = (("time", "y", "x"), stacked)

    return xr.Dataset(
        data_vars,
        coords={
            "time": ref_times,
            "latitude": (("y", "x"), lat_crop),
            "longitude": (("y", "x"), lon_crop),
        },
    )


def main():
    # Params passed as a single JSON arg to keep the subprocess call simple.
    params = json.loads(sys.argv[1])
    log("start")

    ds = extract_cropped_multivar(
        params["file_dir"],
        params["shortname_variable_map"],
        params["time_start"], params["time_end"],
        params["min_lat"], params["max_lat"],
        params["min_lon"], params["max_lon"],
    )
    log("after extraction")

    out_path = params["out_path"]
    ds.to_netcdf(out_path)
    log("after to_netcdf")
    print("Saved", out_path, flush=True)


if __name__ == "__main__":
    main()
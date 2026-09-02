import xarray as xr
import numpy as np
import pandas as pd

def convert_units(dataset, varname):
    """
    Konversi unit data iklim ke format standar.

    Perbaikan dari versi asli:
      - Deteksi unit sekarang membaca `dataset[varname].attrs['units']`,
        bukan `hasattr(dataset[varname], 'units')`. DataArray xarray tidak
        punya attribute Python `.units` secara native -- info unit selalu
        ada di dalam `.attrs` dict. `hasattr(...)` yang lama karena itu
        (hampir) selalu False, sehingga konversi unit tidak pernah benar-
        benar berjalan lewat jalur ini.
      - Dataset di-`.copy()` di awal supaya fungsi ini tidak memutasi
        objek `dataset` milik caller secara diam-diam (side effect).
    """
    dataset = dataset.copy()
    units = dataset[varname].attrs.get('units', '')

    # Konversi presipitasi dari kg/m2/s ke mm/hari (1 kg/m2/s = 86400 mm/day)
    if 'pr' in varname:
        if 'kg m-2 s-1' in units:
            dataset[varname] = dataset[varname] * 86400
            dataset[varname].attrs['units'] = 'mm/day'

    # Konversi suhu dari Kelvin ke Celsius
    elif varname in ['tas', 'tasmax', 'tasmin']:
        if 'K' in units:
            dataset[varname] = dataset[varname] - 273.15
            dataset[varname].attrs['units'] = '°C'


    return dataset

def grouped_dataset(dataset, mode, agg=None):
    if 'time' not in dataset.dims:
        raise ValueError("Input dataset must have a 'time' dimension.")
    if not hasattr(dataset['time'], 'dt'):
        raise ValueError("The 'time' dimension must have datetime-like values.")
    
    if agg == 'mean':
        agg_func = lambda grouped: grouped.mean(dim='time', skipna=True) if hasattr(grouped, 'mean') else grouped.mean(skipna=True)
    elif agg == 'sum':
        agg_func = lambda grouped: grouped.sum(dim='time', skipna=True) if hasattr(grouped, 'sum') else grouped.sum(skipna=True)
    elif agg == 'max':
        agg_func = lambda grouped: grouped.max(dim='time', skipna=True) if hasattr(grouped, 'max') else grouped.max(skipna=True)
    elif agg == 'min':
        agg_func = lambda grouped: grouped.min(dim='time', skipna=True) if hasattr(grouped, 'min') else grouped.min(skipna=True)
    elif agg == 'median':
        agg_func = lambda grouped: grouped.median(dim='time', skipna=True) if hasattr(grouped, 'median') else grouped.median(skipna=True)
    elif agg is not None:
        raise ValueError(f"Invalid aggregation method '{agg}'. Choose from 'mean', 'sum', 'max', 'min', 'median', or None.")
    else:
        agg_func = None

    if mode == 'ANN':
        grouped = dataset.groupby('time.year')
        return agg_func(grouped).rename({"year":"time"}) if agg_func else grouped
    
    elif mode == 'ME':
        grouped = dataset.resample(time='ME')  # Monthly resampling
        return agg_func(grouped) if agg_func else grouped
    
    elif mode in ['DJF', 'MAM', 'JJA', 'SON']:
        season_months = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
        months = season_months[mode]
        ds_season = dataset.sel(time=dataset['time.month'].isin(months))
        
        if mode == 'DJF':
            # Handle December-January-February overlap
            ds_dec = dataset.sel(time=dataset['time.month'] == 12)
            ds_jan_feb = dataset.sel(time=dataset['time.month'].isin([1, 2]))
            ds_dec_shifted = ds_dec.shift(time=1)
            ds_season = xr.concat([ds_dec_shifted, ds_jan_feb], dim='time')
            ds_season = ds_season.where(ds_season['time.year'] > ds_season['time.year'].min(), drop=True)
        
        # Ensure the 'time' dimension is a single chunk
        ds_season = ds_season.chunk(dict(time=-1))
        
        grouped = ds_season.groupby('time.year')
        return agg_func(grouped).rename({"year":"time"}) if agg_func else grouped
    

def group_dataset(dataset, mode):
    """
    Group dataset based on the specified time mode while preserving chunking for Dask-backed datasets.
    
    Parameters:
        dataset (xarray.DataArray or xarray.Dataset): Input dataset with a 'time' dimension.
        mode (str): Time grouping mode. Options: 'year', 'monthly', 'DJF', 'MAM', 'JJA', 'SON'.
    
    Returns:
        Grouped dataset or resampled dataset.
    """
    if mode in ['ANN', 'ME']:
        dataset = dataset  # Ensure single chunk along time for groupby/resample
    
    if mode == 'ANN':
        return dataset.groupby('time.year')
    
    elif mode == 'ME':
        return dataset.resample(time='ME')  # Month-end resampling
    
    elif mode == 'DJF':
        # Select DJF months and create season_year coordinate
        ds_djf = dataset.sel(time=dataset.time.dt.month.isin([12, 1, 2]))
        ds_djf['season_year'] = xr.where(ds_djf.time.dt.month == 12, 
                                         ds_djf.time.dt.year + 1, 
                                         ds_djf.time.dt.year)
        
        # Filter out incomplete seasons (less than 3 months)
        counts          = ds_djf.groupby('season_year').count(dim='time')
        valid_years     = counts.where(counts >= 3, drop=True).season_year
        ds_djf_filtered = ds_djf.sel(time=ds_djf.season_year.isin(valid_years))
        
        return ds_djf_filtered.groupby('season_year')
    
    elif mode in ['MAM', 'JJA', 'SON']:
        # Map mode to corresponding months
        month_map = {
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11]
        }
        months = month_map[mode]
        
        # Select months and group by year
        ds_season = dataset.sel(time=dataset.time.dt.month.isin(months))
        return ds_season.groupby(ds_season.time.dt.year)
    
    else:
        raise ValueError("Invalid mode. Choose from 'ANN', 'ME', 'DJF', 'MAM', 'JJA', or 'SON'.")

def get_coordinate(dataset, possible_names, excluded_names=None):
    """
    Find a coordinate/variable in the dataset using a list of possible names,
    while excluding unwanted ones.

    Parameters:
    - dataset: xarray.Dataset or netCDF4 Dataset
    - possible_names: list of plausible names (e.g., ['lon', 'longitude'])
    - excluded_names: list of names to ignore (e.g., ['lat', 'time'])

    Returns:
    - str: Found coordinate name, or None if not found
    """
    if excluded_names is None:
        excluded_names = []

    # Get available dimensions/variables/coordinates
    available = list(dataset.variables.keys())  # Works for both xarray and netCDF4

    for name in possible_names:
        if name in available and name not in excluded_names:
            return name

    # Fallback: case-insensitive match
    lower_avail = [v.lower() for v in available]
    for name in possible_names:
        if name.lower() in lower_avail:
            idx = lower_avail.index(name.lower())
            actual_name = available[idx]
            if actual_name not in excluded_names:
                return actual_name

    return None

def get_lon(dataset):
    """Check and return the longitude coordinate name."""
    possible_names = ['lon', 'longitude', 'LON', 'XLONG']
    excluded_names = ['lat', 'LAT', 'XLAT', 'time', 'date', 'year', 'month']
    return get_coordinate(dataset, possible_names, excluded_names)

def get_lat(dataset):
    """Check and return the latitude coordinate name."""
    possible_names = ['lat', 'latitude', 'LAT', 'XLAT']
    excluded_names = ['lon', 'LON', 'XLONG', 'time', 'date', 'year', 'month']
    return get_coordinate(dataset, possible_names, excluded_names)

def get_time(dataset):
    """Check and return the time coordinate name."""
    possible_names = ['time', 'date', 'datetime', 'dates', 'times']
    excluded_names = ['year', 'month', 'day', 'lat', 'lon']
    return get_coordinate(dataset, possible_names, excluded_names)

def get_var(dataset, var_type):
    """
    Generic function to get common climate variable names.
    """
    var_options = {
        'temp': ['temp', 'temperature', 'T2', 'T2C', 't2m','tas', 'tasmax', 'tasmin', 'air_temperature'],
        'precip': ['precip', 'precipitation', 'rain', 'pr', 'RAIN'],
        'u_wind': ['u', 'u_wind', 'U10', 'u10'],
        'v_wind': ['v', 'v_wind', 'V10', 'v10'],
        'pressure': ['pressure', 'pres', 'PSFC', 'slp'],
        'humidity': ['humidity', 'hur', 'Q2'],
    }
    possible_names = var_options.get(var_type.lower(), [])
    return get_coordinate(dataset, possible_names)

def convert_time_to_datetime(dataset, varname=None):
    if varname is None:
        varname = get_time(dataset)
    
    if varname not in dataset.coords:
        print(f"No '{varname}' coordinate found.")
        return dataset
    
    time_var = dataset[varname]
    
    # Cek kalender
    calendar = time_var.attrs.get('calendar', 'standard').lower()
    if calendar in ['noleap', '360_day']:
        print(f"Skipping conversion for non-standard calendar: {calendar}")
        return dataset  # Lewati konversi
    
    # Konversi ke datetime64 hanya untuk kalender standar
    if not isinstance(time_var.values[0], pd.Timestamp):
        print("Converting to datetime64...")
        dataset = xr.decode_cf(dataset, decode_times=True)
    
    return dataset

def convert_time_to_object(dataset, time_format='%Y-%m-%d', varname=None):
    """
    Convert the time coordinate from datetime64[ns] to object (string) format.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the time coordinate.
        time_format (str): The format string for the time conversion (default: '%Y-%m-%d').
        varname (str, optional): The name of the time coordinate. If None, it will be detected automatically.

    Returns:
        xarray.Dataset: The dataset with the time coordinate converted to object (string) format.
    """
    if varname is None:
        varname = get_time(dataset)
    
    if varname in dataset.coords:
        if dataset[varname].dtype == 'datetime64[ns]':
            print(f"Converting time coordinate '{varname}' to object (string) format...")
            # Convert datetime64 to string using the specified format
            time_as_string = pd.to_datetime(dataset[varname].values).strftime(time_format)
            # Assign the new time coordinate to the dataset
            dataset = dataset.assign_coords({varname: time_as_string})
        else:
            print(f"Time coordinate '{varname}' is already in {dataset[varname].dtype} format. No conversion needed.")
    else:
        print(f"No '{varname}' coordinate found in the dataset.")
    
    return dataset


# --- Hapus 29 Februari ---
def remove_29feb(ds):
    return ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))

# --- Opsional: Hilangkan 30 Februari ---
def remove_invalid_feb_dates(ds):
    return ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day.isin([29, 30]))))


def time_to_360day(data):
    timename = get_time(data)
    data     = xr.decode_cf(data)
    data     = data.resample(**{timename: 'D'}).nearest()
    data     = data.convert_calendar(calendar='360_day', dim=timename, align_on='year', missing='fill')
    return data
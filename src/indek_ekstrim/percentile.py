"""
percentile.py
=============
Lapisan khusus untuk index suhu berbasis persentil ETCCDI (TX90p, TN10p,
TG90p, WSDI, CSDI, dst.) -- arsitekturnya beda dari `engine.py` karena
butuh 2 tahap:

  1. Baseline threshold: hitung persentil ke-q PER HARI-KALENDER
     (dayofyear 1..366, dengan window +-N hari di sekitarnya) dari
     periode referensi ("base period", mis. 1961-1990).
     
  2. Exceedance: bandingkan SELURUH periode data terhadap threshold
     hari-kalender yang sesuai, lalu agregasi per tahun/musim (persentase
     hari, jumlah hari absolut, atau total hari dalam spell/runtun).

Catatan performa
----------------
`compute_doy_threshold()` sengaja memuat `base_data` penuh ke memori
(`.load()`). Ini valid karena periode referensi standar ETCCDI (mis.
30 tahun) jauh lebih kecil dari seluruh dataset, dan threshold hanya
dihitung SEKALI lalu dipakai ulang untuk semua tahun. Kalau base period-mu
sangat besar (grid resolusi tinggi x banyak tahun) dan tidak muat memori,
ini perlu direvisi jadi versi dask-native (mis. `dask.array.percentile`
per chunk spasial).
"""

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .PreProp import grouped_dataset
from .math_utils import count_days_in_runs


def compute_doy_threshold(
    base_data: xr.DataArray,
    q: float,
    window: int = 5,
) -> xr.DataArray:
    """
    Hitung threshold persentil ke-q per hari-kalender (dayofyear 1..366)
    dari `base_data`, memakai window +-`window` hari di sekitar tiap
    hari-kalender (digabung dari SEMUA tahun di base_data), sesuai
    metode standar ETCCDI.

    Parameters
    ----------
    base_data : xr.DataArray
        Data 1 variabel (sudah convert_units) untuk periode referensi
        saja, mis. `var_data.sel(time=slice('1961','1990'))`. Dims
        minimal (time, ...spatial/station).
    q : float
        Persentil, mis. 0.90 atau 0.10.
    window : int
        Lebar window di kedua sisi hari-kalender (default 5 -> total
        11 hari per window, digabung lintas tahun).

    Returns
    -------
    xr.DataArray dims (dayofyear, ...spatial/station), dayofyear = 1..366.
    """
    base_data = base_data.load()  # lihat catatan performa di docstring modul
    doy = base_data["time"].dt.dayofyear.values
    data = base_data.values
    spatial_shape = data.shape[1:]

    thresh = np.full((366,) + spatial_shape, np.nan)
    for d in range(1, 367):
        window_days = {((d - 1 + off) % 366) + 1 for off in range(-window, window + 1)}
        mask = np.isin(doy, list(window_days))
        if not mask.any():
            continue
        with np.errstate(all="ignore"):
            thresh[d - 1] = np.nanquantile(data[mask], q, axis=0)

    other_dims = tuple(dim for dim in base_data.dims if dim != "time")
    coords = {k: v for k, v in base_data.coords.items() if k != "time" and k in other_dims + ("dayofyear",)}
    coords = {dim: base_data.coords[dim] for dim in other_dims if dim in base_data.coords}
    coords["dayofyear"] = np.arange(1, 367)

    return xr.DataArray(
        thresh,
        dims=("dayofyear",) + other_dims,
        coords=coords,
        name=f"{base_data.name}_p{int(round(q * 100))}",
        attrs={"units": base_data.attrs.get("units", ""), "quantile": q, "window_days": window},
    )


def compute_exceedance(
    data: xr.DataArray,
    threshold_doy: xr.DataArray,
    op: str,
) -> xr.DataArray:
    """
    Bandingkan tiap hari di `data` (periode penuh) terhadap
    `threshold_doy` (hasil `compute_doy_threshold`) sesuai hari-kalendernya.

    Returns
    -------
    xr.DataArray float (1.0 = exceed, 0.0 = tidak, NaN = data asli NaN),
    dims sama seperti `data` (time, ...spatial/station).
    """
    if op not in ("above", "below"):
        raise ValueError(f"op harus 'above' atau 'below', dapat: {op!r}")

    doy = data["time"].dt.dayofyear
    thresh_aligned = threshold_doy.sel(dayofyear=doy)

    if op == "above":
        exceed_bool = data > thresh_aligned
    else:
        exceed_bool = data < thresh_aligned

    # NaN di data asli -> tetap NaN (bukan dianggap False), supaya agregasi
    # skipna di bawah tidak keliru menganggap hari kosong sebagai "tidak exceed".
    exceed = xr.where(data.isnull(), np.nan, exceed_bool.astype(float))
    return exceed


def _spell_days(mask, min_run):
    return count_days_in_runs(mask, min_run=min_run)


def aggregate_index(
    exceed: xr.DataArray,
    mode: str,
    slice_mode: str,
    min_run: int = 6,
    name: Optional[str] = None,
) -> xr.DataArray:
    """
    Agregasi `exceed` (hasil `compute_exceedance`, dims time + spatial)
    per tahun/musim (`slice_mode`) sesuai `mode`:
      - 'pct'   : persentase hari exceed relatif thd total hari valid per periode.
      - 'abs'   : jumlah hari exceed (absolut).
      - 'spell' : total hari dalam runtun >= min_run hari berturut-turut (WSDI/CSDI).
    """
    grouped = grouped_dataset(exceed, mode=slice_mode)  # agg=None -> raw groupby/resample

    if mode == "abs":
        result = grouped.sum(dim="time", skipna=True)
    elif mode == "pct":
        count_true = grouped.sum(dim="time", skipna=True)
        count_total = grouped.count(dim="time")
        result = (count_true / count_total) * 100
    elif mode == "spell":
        result = xr.apply_ufunc(
            _spell_days,
            grouped,
            input_core_dims=[["time"]],
            kwargs={"min_run": min_run},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
            keep_attrs=True,
        )
    else:
        raise ValueError(f"Unknown aggregate mode: {mode!r}")

    if name:
        result = result.rename(name)
    return result
"""
stations.py
===========
Adapter untuk data curah hujan per-stasiun (pandas.DataFrame) supaya bisa
dipakai oleh pipeline `rain_indices()` yang sama tanpa mengubah satu pun
baris di `ettcdi.py`, `engine.py`, atau `pipeline.py`.

Kenapa ini bisa "gratis" (tanpa nulis ulang core logic)?
----------------------------------------------------------
- Semua fungsi index di `ettcdi.py` menerima array 1D generik -- tidak
  peduli itu potongan waktu dari satu grid-cell (lat, lon) atau satu
  stasiun. Tidak ada asumsi spasial di dalamnya.
- `xr.apply_ufunc(..., input_core_dims=[['time']], vectorize=True)` di
  `engine.py` otomatis broadcast ke SEMUA dimensi lain selain 'time' --
  baik itu (lat, lon) pada data grid, maupun (station,) pada data stasiun.

Jadi satu-satunya yang perlu dilakukan adalah membungkus DataFrame jadi
`xr.Dataset` berdimensi (time, station), lalu pipeline yang sama berjalan
apa adanya.

Format DataFrame yang didukung
-------------------------------
1. WIDE  : index = datetime, kolom = nama/ID stasiun.
           df.loc['2020-01-01', 'StasiunA'] -> nilai curah hujan.
2. LONG  : kolom eksplisit [time_col, station_col, value_col].
           Cocok untuk data hasil query database / CSV ekspor SIMBI, dsb.
"""

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


def from_wide_dataframe(
    df: pd.DataFrame,
    varname: str = "pr",
    units: Optional[str] = "mm/day",
    station_dim: str = "wmo_id",
) -> xr.Dataset:
    """
    Konversi DataFrame wide-format (index=waktu, kolom=stasiun) ke
    xr.Dataset berdimensi (time, station), siap dipakai `rain_indices()`.

    Parameters
    ----------
    df : pd.DataFrame
        Index harus datetime-like (atau bisa dikonversi lewat pd.to_datetime).
        Setiap kolom = satu stasiun.
    varname : str
        Nama variabel curah hujan pada Dataset hasil (default 'pr').
    units : str, optional
        Attrs unit yang ditempel ke variabel (mis. 'mm/day' kalau data
        sudah dalam mm/hari, atau 'kg m-2 s-1' kalau masih mentah dari
        model dan perlu dikonversi PreProp.convert_units).
    station_dim : str
        Nama dimensi/koordinat stasiun pada hasil (default 'station').

    Returns
    -------
    xr.Dataset dengan dims (time, station), siap dioper ke rain_indices().

    Contoh
    ------
    >>> df = pd.read_csv('curah_hujan_stasiun.csv', index_col=0, parse_dates=True)
    >>> ds = from_wide_dataframe(df, varname='pr', units='mm/day')
    >>> result = rain_indices(ds, varname='pr', slice_mode='year')
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "time"
    df.columns.name = station_dim

    da = xr.DataArray(
        df.values,
        dims=["time", station_dim],
        coords={"time": df.index, station_dim: df.columns},
        name=varname,
    )
    if units is not None:
        da.attrs["units"] = units

    return da.to_dataset()


def from_long_dataframe(
    df: pd.DataFrame,
    time_col: str = "time",
    station_col: str = "station_id",
    value_col: str = "value",
    varname: str = "pr",
    units: Optional[str] = "mm/day",
    station_dim: str = "station",
) -> xr.Dataset:
    """
    Konversi DataFrame long-format (satu baris = satu observasi
    time x station) ke xr.Dataset berdimensi (time, station).

    Parameters
    ----------
    df : pd.DataFrame
        Harus punya kolom `time_col`, `station_col`, `value_col`.
    time_col, station_col, value_col : str
        Nama kolom di df untuk waktu, ID stasiun, dan nilai curah hujan.
    varname, units, station_dim : lihat `from_wide_dataframe`.

    Returns
    -------
    xr.Dataset dengan dims (time, station).

    Catatan
    -------
    Data di-pivot ke wide format terlebih dahulu (via `pivot_table`).
    Kalau ada duplikat (time, station) di data long, nilai akan
    di-rata-rata -- cek datamu dulu kalau ini tidak diinginkan.
    """
    wide = df.pivot_table(index=time_col, columns=station_col, values=value_col)
    return from_wide_dataframe(wide, varname=varname, units=units, station_dim=station_dim)


def result_to_dataframe(result: xr.Dataset, station_dim: str = "station") -> pd.DataFrame:
    """
    Konversi hasil `rain_indices()` (xr.Dataset dims (time, station)) balik
    ke pandas DataFrame long-format, memudahkan ekspor ke CSV/Excel.

    Returns
    -------
    pd.DataFrame dengan kolom: time, station, <nama index 1>, <nama index 2>, ...
    """
    df = result.to_dataframe().reset_index()
    return df
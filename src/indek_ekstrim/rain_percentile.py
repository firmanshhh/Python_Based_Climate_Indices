"""
rain_percentile.py
===================
Index curah hujan berbasis persentil ala ETCCDI standar (R95p, R99p,
R95pTOT, R99pTOT).

Perbedaan kunci dari implementasi lama (RqP/RqPtot di ettcdi.py, sebelum
refactor ini): threshold DIHITUNG SEKALI dari hari basah (curah hujan >
`wet_day_threshold`, default 1mm) di periode referensi (`base_period`),
lalu dipakai SAMA untuk semua tahun -- BUKAN dihitung ulang dari data
window/tahun yang sedang dihitung sendiri.

Kenapa ini penting (bukan cuma soal gaya kode)
-----------------------------------------------
Definisi ETCCDI R95p/R99p secara eksplisit memakai threshold TETAP dari
base period supaya index ini bisa menangkap TREN jangka panjang (makin
sering/jarang terjadi curah hujan ekstrim relatif ke iklim masa lalu).
Kalau threshold dihitung ulang dari data tahun itu sendiri, maka index
akan selalu menangkap "~5% hari terbasah tahun ini" apa pun kondisi
iklimnya -- tren perubahan iklim justru teredam/tersembunyi, bukan
tertangkap.

Beda dari index suhu (tg90p/tx90p dkk di percentile.py): threshold
curah hujan ETCCDI adalah SATU nilai tetap per titik spasial/stasiun
(bukan per hari-kalender/dayofyear) -- curah hujan ekstrim tidak
didefinisikan bergantung musim seperti suhu.
"""

from typing import Optional

import numpy as np
import xarray as xr

from .math_utils import divide


def compute_wet_day_threshold(
    base_data: xr.DataArray,
    q: float,
    wet_day_threshold: float = 1.0,
) -> xr.DataArray:
    """
    Hitung threshold persentil ke-q dari hari BASAH (curah hujan >
    `wet_day_threshold`) di `base_data`, SATU nilai per titik
    spasial/stasiun -- ETCCDI R95p/R99p memang didefinisikan begitu
    (beda dari index suhu yang per hari-kalender).

    Parameters
    ----------
    base_data : xr.DataArray
        Data curah hujan (sudah convert_units) untuk periode referensi
        saja, mis. `rain_data.sel(time=slice('1961','1990'))`.
    q : float
        Persentil, mis. 0.95 atau 0.99.
    wet_day_threshold : float
        Batas hari "basah" (mm), default 1.0 sesuai definisi ETCCDI.

    Returns
    -------
    xr.DataArray dims sama seperti base_data TANPA 'time' (mis. (lat, lon)
    atau (station,)).
    """
    base_data = base_data.load()  # base period biasanya kecil (~30 thn), aman di-load
    data      = base_data.values
    wet       = np.where(data > wet_day_threshold, data, np.nan)

    with np.errstate(all="ignore"):
        threshold = np.nanquantile(wet, q, axis=0)

    other_dims = tuple(dim for dim in base_data.dims if dim != "time")
    coords = {dim: base_data.coords[dim] for dim in other_dims if dim in base_data.coords}

    return xr.DataArray(
        threshold,
        dims=other_dims,
        coords=coords,
        name=f"{base_data.name}_wetday_p{int(round(q * 100))}",
        attrs={"quantile": q, "wet_day_threshold": wet_day_threshold},
    )


def _rqp_sum(data, threshold, wet_day_threshold=1.0):
    """R95P/R99P: total curah hujan hari basah di atas threshold TETAP."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isnan(data)) or np.isnan(threshold):
        return np.nan
    return np.nansum(data[data > threshold])


def _rqp_pct(data, threshold, wet_day_threshold=1.0):
    """R95PTOT/R99PTOT: kontribusi (%) curah hujan di atas threshold TETAP terhadap total hari basah."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isnan(data)) or np.isnan(threshold):
        return np.nan
    wet = data[data > wet_day_threshold]
    total = np.nansum(wet)
    rqp = np.nansum(data[data > threshold])
    return divide(rqp, total) * 100


def aggregate_rain_percentile(
    sliced_data,
    threshold: xr.DataArray,
    mode: str,
    wet_day_threshold: float = 1.0,
    name: Optional[str] = None,
) -> xr.DataArray:
    """
    Terapkan threshold TETAP (`threshold`, tanpa dim 'time') ke tiap
    window waktu (`sliced_data`, groupby/resample object dengan core
    dim 'time') -- `apply_ufunc` otomatis broadcast `threshold` ke
    setiap window karena tidak punya core dim 'time'.
    """
    if mode not in ("sum", "pct"):
        raise ValueError(f"mode harus 'sum' atau 'pct', dapat: {mode!r}")
    func = _rqp_sum if mode == "sum" else _rqp_pct

    result = xr.apply_ufunc(
        func,
        sliced_data,
        threshold,
        input_core_dims=[["time"], []],
        kwargs={"wet_day_threshold": wet_day_threshold},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=True,
    )
    if name:
        result = result.rename(name)
    return result
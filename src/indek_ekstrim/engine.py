"""
engine.py
=========
Lapisan "cara menghitung" — mengubah satu IndexSpec (dari config.py) menjadi
xr.DataArray hasil, TANPA memicu compute (`.values` / `dask.compute()`) di
tengah jalan. Compute hanya boleh terjadi sekali, di `pipeline.py`.

Perbaikan dibanding versi asli (applyFunc / applyFunc2DMoth):
  1. Tidak ada eval(func) -> pakai ettcdi.FUNC_MAP.
  2. Tidak ada `.values` di sini -> hasil tetap lazy (dask-backed).
  3. Tidak ada dask.compute() bersarang di dalam task delayed lain
     (dulu applyFunc2DMoth memanggil compute() sendiri padahal sudah
     dibungkus delayed() di level atas -> risiko thread starvation).
  4. Nama waktu (time/lat/lon) diambil dari koordinat asli, bukan
     ditebak lewat exclusion list yang rapuh.
"""

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .config import IndexSpec
from .ettcdi import FUNC_MAP


def _apply_simple(sliced_data: xr.DataArray, spec: IndexSpec, name: str) -> xr.DataArray:
    """Index 1-tahap: langsung apply_ufunc(func, data, **params)."""
    func = FUNC_MAP[spec.func]
    result = xr.apply_ufunc(
        func,
        sliced_data,
        input_core_dims=[["time"]],
        kwargs=spec.params,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=True,
    )
    return result.rename(name)


def _apply_quantile(
    dataset: xr.DataArray,
    sliced_data: xr.DataArray,
    spec: IndexSpec,
    name: str,
    slice_mode: str,
) -> xr.DataArray:
    """
    RqP/RqPtot menghitung quantile threshold sendiri secara internal dari
    data yang diterima -- TIDAK perlu qvalue precomputed. Diperlakukan
    sama seperti index simple.
    """
    func = FUNC_MAP[spec.func]
    q = spec.params["q"]

    result = xr.apply_ufunc(
        func,
        sliced_data,
        input_core_dims=[["time"]],
        kwargs={"q": q},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=True,
    )
    return result.rename(name)


def _apply_dual(
    sliced_data1: xr.DataArray,
    sliced_data2: xr.DataArray,
    spec: IndexSpec,
    name: str,
) -> xr.DataArray:
    """
    Index 2-variabel (mis. DTR, ETR): butuh window waktu YANG SAMA dari
    dua variabel berbeda (tasmax & tasmin) secara bersamaan, bukan
    dipanggil terpisah lalu dikurangkan setelah aggregasi.

    `input_core_dims=[["time"], ["time"]]` memastikan tiap pasangan
    window (satu dari var1, satu dari var2) yang sudah selaras di
    koordinat 'time' dioper bersamaan ke `func`.
    """
    func = FUNC_MAP[spec.func]
    result = xr.apply_ufunc(
        func,
        sliced_data1,
        sliced_data2,
        input_core_dims=[["time"], ["time"]],
        kwargs=spec.params,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=True,
    )
    return result.rename(name)


def compute_index(
    name: str,
    spec: IndexSpec,
    dataset: xr.DataArray,
    sliced_data: xr.DataArray,
    slice_mode: str,
    sliced_data2: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Entry point tunggal dari pipeline.py untuk menghitung satu index (lazy).

    `sliced_data2` cuma dipakai untuk `kind="dual"` -- pipeline.py yang
    bertanggung jawab menyiapkan & mengoper data variabel kedua
    (`spec.var2`) yang sudah di-grouped dengan `slice_mode` yang sama.
    """
    if spec.kind == "simple":
        return _apply_simple(sliced_data, spec, name)
    elif spec.kind == "quantile":
        return _apply_quantile(dataset, sliced_data, spec, name, slice_mode)
    elif spec.kind == "dual":
        if sliced_data2 is None:
            raise ValueError(
                f"Index '{name}' kind='dual' butuh sliced_data2 (variabel '{spec.var2}')."
            )
        return _apply_dual(sliced_data, sliced_data2, spec, name)
    raise ValueError(f"Unknown IndexSpec.kind: {spec.kind!r}")
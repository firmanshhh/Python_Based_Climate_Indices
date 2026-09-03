"""
pipeline.py
===========
Entry point publik: `rain_indices(dataset, ...)`.

Perbaikan struktural dibanding versi asli:
  1. `sliced_data` di-`.persist()` SEKALI setelah preprocessing, dipakai
     bersama oleh semua index -> preprocessing tidak dihitung ulang 21x.
  2. Semua index dibangun sebagai graph lazy (lewat engine.py), lalu
     di-compute HANYA SEKALI di akhir lewat `dask.compute(...)`.
     Tidak ada lagi `.values` atau `compute()` yang tersebar/bersarang
     di tengah jalan (sumber risiko thread starvation & recompute).
  3. Duplikasi key index (prcptot/PRCPTOT, cdd/CDD, cwd/CWD) dihapus --
     tinggal satu definisi per index di config.py.
  4. Progress log per-index tetap ada (opsional), tapi tidak memaksa
     compute individual -- hanya menandai graph sudah dibangun.
"""

from typing import Dict, Optional

import xarray as xr
from dask import compute

from .PreProp import convert_units, grouped_dataset
from .config import RAIN_INDICES, TEMP_INDICES, TEMP_PERCENTILE_INDICES, RAIN_PERCENTILE_INDICES
from .engine import compute_index
from .percentile import compute_doy_threshold, compute_exceedance, aggregate_index
from .rain_percentile import compute_wet_day_threshold, aggregate_rain_percentile


VALID_SLICE_MODES = ("ANN", "ME", "DJF", "MAM", "JJA", "SON")


def rain_indices(
    dataset: xr.Dataset,
    varname: str = "pr",
    slice_period: tuple = None,
    slice_mode: str = "ANN",
    chunk_size: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Hitung seluruh index curah hujan di RAIN_INDICES (config.py) dari
    `dataset`, dan kembalikan sebagai satu xr.Dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset input berisi variabel `varname` (default 'pr').
    varname : str
        Nama variabel curah hujan di dataset.
    slice_mode : str
        'year', 'monthly', 'DJF', 'MAM', 'JJA', atau 'SON'.
    chunk_size : dict, optional
        Chunking eksplisit, mis. {'time': 365}. Kalau None, dipilih
        otomatis berdasarkan slice_mode (lihat catatan di bawah).
    verbose : bool
        Cetak progress pembangunan graph per index.

    Returns
    -------
    xr.Dataset berisi semua index sebagai data_vars.

    Catatan performa
    ----------------
    - Chunking 'auto' dipilih untuk slice_mode='year' agar dask tetap bisa
      memparalelkan across time (BUKAN satu chunk raksasa/time=-1, yang
      justru mematikan paralelisme).
    - `rain_data.persist()` (SEBELUM groupby) memastikan preprocessing
      (unit conversion, chunking) hanya dihitung sekali dan dipakai
      bersama oleh seluruh index. Tidak bisa persist SETELAH groupby
      karena `grouped_dataset()` mengembalikan objek GroupBy, bukan
      array biasa (lihat komentar di kode).
    """
    if slice_mode not in VALID_SLICE_MODES:
        raise ValueError(f"slice_mode must be one of {VALID_SLICE_MODES}")
    if varname not in dataset.variables:
        raise KeyError(f"Variable '{varname}' not found in the dataset.")

    # --- Preprocessing (dilakukan sekali) ---
    # convert_units() sudah melakukan .copy() sendiri secara internal,
    # jadi tidak perlu di-copy lagi di sini.
    dataset = convert_units(dataset, varname)
    rain_data = dataset[varname]

    # --- Tentukan slice_periode data
    if slice_period is None:
        rain_data = rain_data
    else:
        slice_start, slice_end = slice_period
        rain_data = rain_data.sel(time=slice(slice_start, slice_end))

    if chunk_size is not None:
        rain_data = rain_data.chunk(chunk_size)
    elif slice_mode == "year":
        rain_data = rain_data.chunk({"time": "auto"})
    # mode lain: biarkan chunking asli / auto dari upstream (mode DJF dkk
    # sudah menangani chunking-nya sendiri di dalam grouped_dataset()).

    # PENTING: `grouped_dataset(..., agg=None)` mengembalikan objek GroupBy
    # (DatasetGroupBy/DataArrayGroupBy), yang TIDAK punya method `.persist()`.
    # Maka yang di-persist adalah `rain_data` (array biasa) SEBELUM di-groupby
    # -- ini yang memastikan preprocessing dihitung sekali dan dipakai
    # bersama oleh semua index, tanpa error AttributeError.
    rain_data   = rain_data.persist()
    sliced_data = grouped_dataset(rain_data, mode=slice_mode)

    # --- Bangun graph lazy untuk semua index ---
    lazy_results = {}
    for name, spec in RAIN_INDICES.items():
        if verbose:
            print(f"[rain_indices] building graph: {name}")
        lazy_results[name] = compute_index(
            name=name,
            spec=spec,
            dataset=rain_data,
            sliced_data=sliced_data,
            slice_mode=slice_mode,
        )

    # --- Compute SEKALI untuk semua index sekaligus ---
    if verbose:
        print(f"[rain_indices] computing {len(lazy_results)} indices in parallel...")
    (computed,) = compute(lazy_results)
    return xr.Dataset(computed)


def temp_indices(
    dataset: xr.Dataset,
    slice_period: tuple = None,
    slice_mode: str = "ANN",
    chunk_size: Optional[Dict[str, int]] = None,
    indices: Optional[list] = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Hitung index suhu di TEMP_INDICES (config.py) dari `dataset`.

    Berbeda dari `rain_indices()`: tiap index suhu bisa berasal dari
    variabel sumber yang berbeda (`tasmax`, `tasmin`, atau `tas`), lewat
    `IndexSpec.var`. Fungsi ini otomatis:
      1. Mendeteksi variabel mana saja yang benar-benar dibutuhkan
         (berdasarkan `indices` yang diminta, atau semua TEMP_INDICES).
      2. Kalau suatu variabel tidak ada di `dataset`, index yang bergantung
         padanya di-skip (dengan warning), bukan meng-crash seluruh pipeline
         -- supaya bisa jalan walau cuma punya tasmax saja, dsb.
      3. convert_units() + persist() per-variabel HANYA SEKALI, dipakai
         bersama oleh semua index yang butuh variabel itu (sama seperti
         optimisasi di `rain_indices()`).

    Parameters
    ----------
    dataset : xr.Dataset
        Harus berisi minimal satu dari 'tasmax', 'tasmin', 'tas'
        (tergantung index apa yang diminta).
    slice_period : tuple(str, str), optional
        Batasi rentang waktu yang dihitung index-nya, mis.
        ('2015-01-01', '2100-12-31') untuk dataset yang mencakup
        historis+skenario tapi index cuma mau dihitung utk periode
        skenario. Kalau None, pakai seluruh rentang waktu di `dataset`.
    slice_mode : str
        'ANN', 'ME', 'DJF', 'MAM', 'JJA', atau 'SON'.
    chunk_size : dict, optional
        Chunking eksplisit, mis. {'time': 365}.
    indices : list[str], optional
        Subset nama index yang mau dihitung (mis. ['TXx', 'FD']).
        Kalau None, hitung SEMUA yang ada di TEMP_INDICES.
    verbose : bool
        Cetak progress.

    Returns
    -------
    xr.Dataset berisi index suhu yang berhasil dihitung sebagai data_vars.

    Catatan
    -------
    Index berbasis persentil (TN10p, TX90p, WSDI, CSDI, dst.) BELUM
    didukung fungsi ini -- lihat catatan di config.py bagian TEMP_INDICES.
    Index dua-variabel (DTR, ETR) SUDAH didukung lewat `kind="dual"`.
    """
    wanted = indices if indices is not None else list(TEMP_INDICES.keys())
    unknown = set(wanted) - set(TEMP_INDICES)
    if unknown:
        raise KeyError(f"Index tidak dikenal di TEMP_INDICES: {sorted(unknown)}")

    # --- Tentukan variabel mana saja yang benar-benar dipakai & tersedia ---
    prepared_vars: Dict[str, xr.DataArray] = {}   # varname -> sliced_data (grouped, lazy)
    raw_vars: Dict[str, xr.DataArray] = {}         # varname -> rain_data (persisted, ungrouped)
    skipped = []

    def _prepare(varname: str) -> None:
        """Convert-units + chunk + persist + group SATU KALI per varname,
        dipakai ulang oleh index apa pun (simple maupun dual) yang butuh
        variabel yang sama -- termasuk 'tasmin' yang dipakai bareng oleh
        FD/TR dan DTR/ETR."""
        if varname in prepared_vars:
            return
        if verbose:
            print(f"[temp_indices] preparing variable: {varname}")
        ds_conv = convert_units(dataset, varname)
        var_data = ds_conv[varname]

        # --- Tentukan slice_period data (sebelum chunk/persist, sama
        # seperti urutan di rain_indices()) ---
        if slice_period is not None:
            slice_start, slice_end = slice_period
            var_data = var_data.sel(time=slice(slice_start, slice_end))

        if chunk_size is not None:
            var_data = var_data.chunk(chunk_size)
        elif slice_mode == "year":
            var_data = var_data.chunk({"time": "auto"})
        var_data = var_data.persist()
        raw_vars[varname] = var_data
        prepared_vars[varname] = grouped_dataset(var_data, mode=slice_mode)

    lazy_results = {}
    for name in wanted:
        spec = TEMP_INDICES[name]
        varname = spec.var
        if varname is None:
            raise ValueError(f"TEMP_INDICES['{name}'] tidak punya 'var' -- config salah.")

        # Untuk kind="dual", butuh var DAN var2 sama-sama tersedia.
        needed_vars = [varname] + ([spec.var2] if spec.kind == "dual" else [])
        missing = [v for v in needed_vars if v not in dataset.variables]
        if missing:
            skipped.append((name, missing))
            continue

        for v in needed_vars:
            _prepare(v)

        if verbose:
            print(f"[temp_indices] building graph: {name} (var={needed_vars})")

        if spec.kind == "dual":
            lazy_results[name] = compute_index(
                name=name,
                spec=spec,
                dataset=raw_vars[varname],
                sliced_data=prepared_vars[varname],
                slice_mode=slice_mode,
                sliced_data2=prepared_vars[spec.var2],
            )
        else:
            lazy_results[name] = compute_index(
                name=name,
                spec=spec,
                dataset=raw_vars[varname],
                sliced_data=prepared_vars[varname],
                slice_mode=slice_mode,
            )

    if skipped:
        for name, missing_vars in skipped:
            print(f"[temp_indices] SKIP '{name}': variabel {missing_vars} tidak ada di dataset.")

    if not lazy_results:
        raise KeyError(
            "Tidak ada index yang bisa dihitung -- dataset tidak punya "
            "variabel 'tasmax'/'tasmin'/'tas' yang dibutuhkan."
        )

    if verbose:
        print(f"[temp_indices] computing {len(lazy_results)} indices in parallel...")
    (computed,) = compute(lazy_results)

    return xr.Dataset(computed)


def temp_percentile_indices(
    dataset: xr.Dataset,
    base_period: tuple,
    slice_period: tuple = None,
    slice_mode: str = "ANN",
    chunk_size: Optional[Dict[str, int]] = None,
    indices: Optional[list] = None,
    window: int = 5,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Hitung index suhu berbasis persentil di TEMP_PERCENTILE_INDICES (config.py):
    tg90/tg10/tn90/tn10/tx90/tx10 (+ versi *abs), dan wsdi/csdi.

    Berbeda dari `rain_indices()`/`temp_indices()`: fungsi ini butuh
    `base_period` (periode referensi) untuk menghitung threshold
    persentil per hari-kalender SEBELUM membandingkan ke seluruh data.
    Lihat `percentile.py` untuk detail metodenya.

    Parameters
    ----------
    dataset : xr.Dataset
        Harus berisi variabel yang dibutuhkan index yang diminta
        ('tasmax', 'tasmin', dan/atau 'tas'), mencakup periode referensi
        MAUPUN periode yang ingin dihitung indexnya (biasanya satu
        dataset yang sama, base_period cuma subset waktunya).
    base_period : tuple(str, str)
        Rentang waktu periode referensi, mis. ('1961-01-01', '1990-12-31').
        Dioper langsung ke `.sel(time=slice(*base_period))`. HARUS berada
        di dalam rentang waktu `dataset` (bukan `slice_period`) --
        threshold selalu dihitung dari data PENUH sebelum `slice_period`
        diterapkan.
    slice_period : tuple(str, str), optional
        Batasi rentang waktu yang dihitung index-nya (SETELAH threshold
        dihitung dari `base_period`), mis. ('2015-01-01', '2100-12-31')
        untuk dataset historis+skenario yang index-nya cuma mau dihitung
        utk periode skenario. Kalau None, pakai seluruh rentang waktu di
        `dataset`.
    slice_mode : str
        'ANN', 'ME', 'DJF', 'MAM', 'JJA', atau 'SON'.
    chunk_size : dict, optional
        Chunking eksplisit untuk data periode PENUH (bukan base period --
        base period selalu di-load ke memori, lihat percentile.py).
    indices : list[str], optional
        Subset nama index (mis. ['tx90', 'wsdi']). None -> semua di
        TEMP_PERCENTILE_INDICES.
    window : int
        Lebar window (+-hari) untuk threshold per hari-kalender (default 5).
    verbose : bool
        Cetak progress.

    Returns
    -------
    xr.Dataset berisi index yang berhasil dihitung sebagai data_vars.
    Dim waktu hasil: 'year' untuk slice_mode ANN/musiman, 'time' untuk 'ME'
    (konsisten dengan `rain_indices()`/`temp_indices()`).

    Contoh
    ------
    >>> result = percentile_indices(
    ...     ds, base_period=("1961-01-01", "1990-12-31"),
    ...     slice_mode="ANN", indices=["tx90", "tn10", "wsdi"],
    ... )
    """
    wanted = indices if indices is not None else list(TEMP_PERCENTILE_INDICES.keys())
    unknown = set(wanted) - set(TEMP_PERCENTILE_INDICES)
    if unknown:
        raise KeyError(f"Index tidak dikenal di TEMP_PERCENTILE_INDICES: {sorted(unknown)}")

    base_start, base_end = base_period

    prepared_full: Dict[str, xr.DataArray] = {}                 # var -> full persisted data (utk threshold)
    prepared_target: Dict[str, xr.DataArray] = {}               # var -> data yg sudah dipersempit slice_period (utk exceedance/agregasi)
    thresholds: Dict[tuple, xr.DataArray] = {}                  # (var, q) -> threshold_doy
    exceedances: Dict[tuple, xr.DataArray] = {}                 # (var, q, op) -> exceed array
    skipped = []

    lazy_results = {}
    for name in wanted:
        spec = TEMP_PERCENTILE_INDICES[name]

        if spec.var not in dataset.variables:
            skipped.append((name, spec.var))
            continue

        if spec.var not in prepared_full:
            if verbose:
                print(f"[temp_percentile_indices] preparing variable: {spec.var}")
            ds_conv = convert_units(dataset, spec.var)
            var_data = ds_conv[spec.var]
            if chunk_size is not None:
                var_data = var_data.chunk(chunk_size)
            var_data = var_data.persist()
            prepared_full[spec.var] = var_data

            # --- Data target (dipersempit slice_period) HANYA dipakai utk
            # exceedance & agregasi -- threshold tetap dari prepared_full
            # (data penuh) di atas. ---
            if slice_period is not None:
                slice_start, slice_end = slice_period
                prepared_target[spec.var] = var_data.sel(time=slice(slice_start, slice_end))
            else:
                prepared_target[spec.var] = var_data

        key_thresh = (spec.var, spec.q)
        if key_thresh not in thresholds:
            if verbose:
                print(f"[temp_percentile_indices] computing baseline threshold: var={spec.var} q={spec.q}")
            base_data = prepared_full[spec.var].sel(time=slice(base_start, base_end))
            thresholds[key_thresh] = compute_doy_threshold(base_data, spec.q, window=window)

        key_exceed = (spec.var, spec.q, spec.op)
        if key_exceed not in exceedances:
            exceedances[key_exceed] = compute_exceedance(
                prepared_target[spec.var], thresholds[key_thresh], spec.op
            )

        if verbose:
            print(f"[temp_percentile_indices] building graph: {name}")
        lazy_results[name] = aggregate_index(
            exceedances[key_exceed],
            mode=spec.mode,
            slice_mode=slice_mode,
            min_run=spec.min_run,
            name=name,
        )

    if skipped:
        for name, varname in skipped:
            print(f"[temp_percentile_indices] SKIP '{name}': variabel '{varname}' tidak ada di dataset.")

    if not lazy_results:
        raise KeyError(
            "Tidak ada index yang bisa dihitung -- dataset tidak punya "
            "variabel 'tasmax'/'tasmin'/'tas' yang dibutuhkan."
        )

    if verbose:
        print(f"[temp_percentile_indices] computing {len(lazy_results)} indices in parallel...")
    (computed,) = compute(lazy_results)

    return xr.Dataset(computed)


def rain_percentile_indices(
    dataset: xr.Dataset,
    base_period: tuple,
    slice_period: tuple = None,
    varname: str = "pr",
    slice_mode: str = "ANN",
    chunk_size: Optional[Dict[str, int]] = None,
    indices: Optional[list] = None,
    wet_day_threshold: float = 1.0,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Hitung index curah hujan berbasis persentil ETCCDI standar
    (R95P, R99P, R95PTOT, R99PTOT) -- threshold TETAP dari `base_period`,
    dipakai sama untuk semua tahun. Lihat rain_percentile.py untuk
    penjelasan kenapa ini berbeda dari implementasi lama.
 
    Parameters
    ----------
    dataset : xr.Dataset
        Harus berisi variabel `varname` (default 'pr'), mencakup
        `base_period` MAUPUN periode yang ingin dihitung indexnya.
    base_period : tuple(str, str)
        Rentang waktu periode referensi, mis. ('1961-01-01', '1990-12-31').
    varname : str
        Nama variabel curah hujan (default 'pr').
    slice_mode : str
        'ANN', 'ME', 'DJF', 'MAM', 'JJA', atau 'SON'.
    chunk_size : dict, optional
        Chunking eksplisit untuk data periode PENUH.
    indices : list[str], optional
        Subset nama index (mis. ['R95P']). None -> semua di RAIN_PERCENTILE_INDICES.
    wet_day_threshold : float
        Batas hari "basah" (mm), default 1.0 sesuai definisi ETCCDI.
    verbose : bool
        Cetak progress.
 
    Returns
    -------
    xr.Dataset berisi index yang berhasil dihitung sebagai data_vars.
 
    Contoh
    ------
    >>> result = rain_percentile_indices(
    ...     ds, base_period=("1961-01-01", "1990-12-31"), slice_mode="ANN",
    ... )
    """
    wanted  = indices if indices is not None else list(RAIN_PERCENTILE_INDICES.keys())
    unknown = set(wanted) - set(RAIN_PERCENTILE_INDICES)
    if unknown:
        raise KeyError(f"Index tidak dikenal di RAIN_PERCENTILE_INDICES: {sorted(unknown)}")
    if varname not in dataset.variables:
        raise KeyError(f"Variable '{varname}' not found in the dataset.")
 
    base_start,  base_end  = base_period
 
    # --- Preprocessing (sekali) ---
    dataset   = convert_units(dataset, varname)
    rain_data = dataset[varname]
    if chunk_size is not None:
        rain_data = rain_data.chunk(chunk_size)
    rain_data = rain_data.persist()
 
    # --- Threshold TETAP dari base_period, sekali per q yang dibutuhkan ---
    needed_q   = {RAIN_PERCENTILE_INDICES[name].q for name in wanted}
    base_data  = rain_data.sel(time=slice(base_start, base_end))

    thresholds = {}
    for q in needed_q:
        if verbose:
            print(f"[rain_percentile_indices] computing baseline threshold: q={q}")
        thresholds[q] = compute_wet_day_threshold(base_data, q, wet_day_threshold=wet_day_threshold)

    # --- Tentukan slice_periode data
    if slice_period is None:
        rain_data = rain_data
    else:
        slice_start, slice_end = slice_period
        rain_data = rain_data.sel(time=slice(slice_start, slice_end))
        
    sliced_data = grouped_dataset(rain_data, mode=slice_mode)
 
    # --- Bangun graph lazy untuk semua index ---
    lazy_results = {}
    for name in wanted:
        spec = RAIN_PERCENTILE_INDICES[name]
        if verbose:
            print(f"[rain_percentile_indices] building graph: {name}")
        lazy_results[name] = aggregate_rain_percentile(
            sliced_data,
            thresholds[spec.q],
            mode=spec.mode,
            wet_day_threshold=wet_day_threshold,
            name=name,
        )
 
    if verbose:
        print(f"[rain_percentile_indices] computing {len(lazy_results)} indices in parallel...")
    (computed,) = compute(lazy_results)
 
    return xr.Dataset(computed)
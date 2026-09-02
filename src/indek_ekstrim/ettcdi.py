"""
ettcdi.py
=========
Kumpulan fungsi index iklim (ETCCDI-style). Setiap fungsi menerima satu
window waktu (1D array, biasanya 1 tahun atau 1 bulan) dan mengembalikan
satu nilai skalar (hasil `xr.apply_ufunc(..., vectorize=True)`).

Perubahan dari versi asli:
  - `divide` dipindah ke `math_utils.py`, dihapus duplikasinya.
  - `cdd`/`cwd`: loop Python -> vectorized run-length encoding (`longest_run`).
  - `RxNDay`: loop Python untuk rolling-sum -> `np.convolve`.
  - `sdii`: BUG DIPERBAIKI. Versi lama menghitung "streak terpanjang hari
    di atas threshold" (logika `cwd`), padahal SDII = total curah hujan
    hari basah dibagi JUMLAH hari basah (bukan runtun terpanjang).
"""

import numpy as np

from .math_utils import divide, longest_run


# ---------------------------------------------------------------------------
# Precipitation indices
# ---------------------------------------------------------------------------

def cdd(data, threshold=1):
    """Consecutive Dry Days: runtun terpanjang hari dengan curah < threshold."""
    data = np.asarray(data, dtype=float)
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    # NaN diperlakukan transparan (tidak mereset & tidak menghitung streak),
    # sama seperti semantik `continue` pada versi loop asli.
    filled = np.where(np.isnan(data), threshold, data)
    mask = filled < threshold
    return longest_run(mask)


def cwd(data, threshold=1):
    """Consecutive Wet Days: runtun terpanjang hari dengan curah >= threshold."""
    data = np.asarray(data, dtype=float)
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    filled = np.where(np.isnan(data), threshold, data)
    mask = filled >= threshold
    return longest_run(mask)


def sdii(data, threshold=1):
    """
    Simple Daily Intensity Index = total curah hujan hari basah / jumlah hari basah.
    (Diperbaiki dari versi lama yang salah menghitung streak, bukan count.)
    """
    data = np.asarray(data, dtype=float)
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    wet = data[(data >= threshold) & ~np.isnan(data)]
    if wet.size == 0:
        return 0.0
    return divide(np.nansum(wet), wet.size)


def prcptot(data):
    """Total curah hujan tahunan/bulanan dari hari dengan curah > 0."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isnan(data)) or np.all(data == 0):
        return np.nan
    return np.nansum(data[data > 0])


def RxNDay(data, windows):
    """Curah hujan maksimum kumulatif dalam jendela N hari berurutan."""
    data = np.asarray(data, dtype=float)
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    if len(data) < windows:
        return np.nan
    if windows == 1:
        return np.nanmax(data)
    kernel = np.ones(windows)
    # np.convolve tidak nan-aware; ganti NaN jadi 0 hanya untuk perhitungan
    # rolling-sum (konsisten dengan `prcptot` yang juga mengabaikan NaN).
    safe = np.nan_to_num(data, nan=0.0)
    rolling_sum = np.convolve(safe, kernel, mode='valid')
    return np.max(rolling_sum)


def HHnMM(data, threshold):
    """Jumlah hari dengan curah hujan >= threshold (mm)."""
    data = np.asarray(data, dtype=float)
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    return np.nansum(data >= threshold)


def FHnMM(data, threshold):
    """Persentase hari 'sangat basah' (>=threshold) relatif terhadap hari basah (>=1mm)."""
    data = np.asarray(data, dtype=float)
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    HH = np.nansum(data >= 1)
    HHNMM = np.nansum(data >= threshold)
    return divide(HHNMM, HH) * 100


def quant(data, q):
    """Wrapper tipis di atas np.quantile, dipakai untuk R95P/R99P dsb."""
    return np.quantile(data, q)


def RqP(data, q=0.95):
    """Total curah hujan hari basah ekstrim di atas kuantil q."""
    data = np.asarray(data, dtype=float)
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    hh = data[data > 1]
    if hh.size == 0:
        return np.nan
    threshold_q = np.quantile(hh, q=q)
    return np.nansum(data[data > threshold_q])


def RqPtot(data, q=0.95):
    """Kontribusi (%) curah hujan ekstrim di atas kuantil q terhadap total."""
    data = np.asarray(data, dtype=float)
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    hh = data[data > 1]
    if hh.size == 0:
        return np.nan
    threshold_q = np.quantile(hh, q=q)
    rqp = np.nansum(data[data > threshold_q])
    total = np.nansum(hh)
    return divide(rqp, total) * 100


# ---------------------------------------------------------------------------
# Temperature indices (tidak diubah, sudah vectorized)
# ---------------------------------------------------------------------------
def dtr(data_tasmax, data_tasmin):
    """
    Diurnal Temperature Range = rata-rata (Tmax - Tmin) harian dalam window.
    Menerima DUA array 1D terpisah (bukan satu array/dict seperti versi
    lama) -- dipasangkan otomatis oleh `engine._apply_dual` lewat
    `xr.apply_ufunc(..., input_core_dims=[["time"], ["time"]])`, jadi
    setiap panggilan sudah menerima window waktu yang SAMA dari tasmax
    dan tasmin.
    """
    data_tasmax = np.asarray(data_tasmax, dtype=float)
    data_tasmin = np.asarray(data_tasmin, dtype=float)
    if np.all(np.isnan(data_tasmax)) or np.all(np.isnan(data_tasmin)):
        return np.nan
    return np.nanmean(data_tasmax - data_tasmin)
 
 
def Tm(data):
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    return np.nanmean(data)
 
 
def Tx(data):
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    return np.nanmax(data)
 
 
def Tn(data):
    if np.all(data == 0) or np.all(np.isnan(data)):
        return np.nan
    return np.nanmin(data)
 
 
def etr(data_tasmax, data_tasmin):
    """
    Extreme Temperature Range = max(Tmax dalam window) - min(Tmin dalam window).
    Sama seperti `dtr`, menerima dua array 1D terpisah (tasmax, tasmin).
    """
    data_tasmax = np.asarray(data_tasmax, dtype=float)
    data_tasmin = np.asarray(data_tasmin, dtype=float)
    if np.all(np.isnan(data_tasmax)) or np.all(np.isnan(data_tasmin)):
        return np.nan
    return np.nanmax(data_tasmax) - np.nanmin(data_tasmin)

def fd(data, threshold=0):
    """Frost Days: jumlah hari dengan Tmin < threshold (°C, default 0)."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isnan(data)):
        return np.nan
    return np.nansum(data < threshold)


def icing_days(data, threshold=0):
    """Icing Days: jumlah hari dengan Tmax < threshold (°C, default 0)."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isnan(data)):
        return np.nan
    return np.nansum(data < threshold)


def su(data, threshold=25):
    """Summer Days: jumlah hari dengan Tmax > threshold (°C, default 25)."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isnan(data)):
        return np.nan
    return np.nansum(data > threshold)


def tr(data, threshold=20):
    """Tropical Nights: jumlah hari dengan Tmin > threshold (°C, default 20)."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isnan(data)):
        return np.nan
    return np.nansum(data > threshold)


# Registry fungsi: dipakai oleh engine.py sebagai pengganti eval(func).
FUNC_MAP = {
    'cdd': cdd,
    'cwd': cwd,
    'sdii': sdii,
    'prcptot': prcptot,
    'RxNDay': RxNDay,
    'HHnMM': HHnMM,
    'FHnMM': FHnMM,
    'quant': quant,
    'RqP': RqP,
    'RqPtot': RqPtot,
    'dtr': dtr,
    'etr': etr,
    'Tm': Tm,
    'Tx': Tx,
    'Tn': Tn,
    'fd': fd,
    'id': icing_days,
    'su': su,
    'tr': tr,
}
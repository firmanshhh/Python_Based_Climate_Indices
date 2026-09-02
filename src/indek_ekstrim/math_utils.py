"""
math_utils.py
=============
Helper matematis generik yang dipakai oleh index functions di `ettcdi.py`.
Dipisah dari `ettcdi.py` supaya:
  - Tidak ada duplikasi definisi `divide` (dulu didefinisikan 2x).
  - Mudah ditest terpisah (unit test kecil, tidak butuh dataset iklim).
"""

import numpy as np


def divide(numerator, denominator):
    """
    Pembagian aman untuk SKALAR (bukan array) dengan aturan iklim-index umum:
      - 0 / 0            -> NaN  (tidak terdefinisi)
      - 0 / non-zero     -> 0.0
      - NaN di salah satu -> NaN
      - selain itu       -> numerator / denominator

    Catatan: versi sebelumnya memakai `np.any(...)` yang ditujukan untuk
    array, padahal semua pemanggil di codebase ini mengoper skalar hasil
    `.sum()` / `np.quantile()`. Disederhanakan agar tidak menyesatkan.
    """
    if np.isnan(numerator) or np.isnan(denominator):
        return np.nan
    if denominator == 0:
        return np.nan if numerator == 0 else 0.0
    return numerator / denominator


def longest_run(mask: np.ndarray) -> int:
    """
    Menghitung panjang run (rentetan True berturut-turut) terpanjang dalam
    array boolean 1D, tanpa loop Python — dipakai oleh `cdd` dan `cwd`.

    Contoh: [F, T, T, T, F, T] -> 3
    """
    if not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return int((ends - starts).max())


def count_days_in_runs(mask: np.ndarray, min_run: int = 6) -> float:
    """
    Menjumlahkan TOTAL hari yang berada dalam runtun True berturut-turut
    sepanjang >= min_run -- dipakai oleh WSDI/CSDI. Beda dari `longest_run`:
    ini bukan mencari 1 runtun terpanjang, tapi menjumlahkan SEMUA hari
    dari SEMUA runtun yang memenuhi syarat panjang minimum.

    Contoh: [T,T,T,T,T,T,T, F, T,T,T,T,T,T,T,T] dengan min_run=6
      -> runtun pertama panjang 7 (>=6, ikut dihitung)
      -> runtun kedua panjang 8 (>=6, ikut dihitung)
      -> total = 7 + 8 = 15
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    lengths = ends - starts
    return float(lengths[lengths >= min_run].sum())
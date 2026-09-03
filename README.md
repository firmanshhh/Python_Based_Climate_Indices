# indek_ekstrim

Package untuk menghitung index iklim ekstrem (curah hujan & suhu, gaya
ETCCDI) dari data `xarray` — baik data **grid** (dims `time, lat, lon`)
maupun data **titik/stasiun** (dims `time, station`), memakai kode
perhitungan yang sama persis untuk keduanya.

```python
from indek_ekstrim import (
    rain_indices,
    temp_indices,
    temp_percentile_indices,
    rain_percentile_indices,
)
```

> Rumus matematis tiap index ada di dokumen terpisah:
> **[`INDEX_FORMULAS.md`](./INDEX_FORMULAS.md)**.
> Panduan menambah index baru ada di:
> **[`ADDING_NEW_INDEX.md`](./ADDING_NEW_INDEX.md)**.

---

## Daftar Isi

1. [Konsep dasar & arsitektur](#1-konsep-dasar--arsitektur)
2. [Format data input](#2-format-data-input)
3. [`rain_indices()` — index curah hujan](#3-rain_indices--index-curah-hujan)
4. [`temp_indices()` — index suhu sederhana & dua-variabel](#4-temp_indices--index-suhu-sederhana--dua-variabel)
5. [`temp_percentile_indices()` — index suhu berbasis persentil](#5-temp_percentile_indices--index-suhu-berbasis-persentil)
6. [`rain_percentile_indices()` — index curah hujan berbasis persentil](#6-rain_percentile_indices--index-curah-hujan-berbasis-persentil)
7. [`slice_period` — membatasi rentang waktu output (semua pipeline)](#7-slice_period--membatasi-rentang-waktu-output-semua-pipeline)
8. [Data stasiun](#8-data-stasiun)
9. [Katalog lengkap index yang tersedia](#9-katalog-lengkap-index-yang-tersedia)
10. [Struktur file & tanggung jawab tiap modul](#10-struktur-file--tanggung-jawab-tiap-modul)
11. [Menambah index baru](#11-menambah-index-baru)

---

## 1. Konsep dasar & arsitektur

Tiga hal yang dipegang teguh di seluruh package ini:

- **Lazy sampai akhir.** Semua perhitungan dibangun sebagai graph dask
  (`xr.apply_ufunc(..., dask="parallelized")`) dan baru benar-benar
  di-*compute* SEKALI di akhir tiap fungsi pipeline (`dask.compute(...)`).
  Tidak ada `.values`/`.compute()` tersebar di tengah proses.
- **Preprocessing dipakai bareng.** Variabel yang sama (mis. `tasmax`)
  hanya di-`convert_units()` + `.persist()` SEKALI, lalu dipakai ulang
  oleh semua index yang butuh variabel itu — bukan diulang per index.
- **Grid & stasiun pakai kode yang sama.** Fungsi index (`ettcdi.py`)
  menerima array 1D generik tanpa asumsi spasial. `apply_ufunc` otomatis
  broadcast ke dimensi apa pun selain `time` — entah itu `(lat, lon)`
  atau `(station,)`.
- **Threshold SELALU dari periode referensi tetap, bukan dari data yang
  sedang dihitung sendiri** — ini prinsip inti ETCCDI untuk semua index
  berbasis persentil (`temp_percentile_indices`, `rain_percentile_indices`).
  Lihat [§9](#9-katalog-lengkap-index-yang-tersedia) dan
  `INDEX_FORMULAS.md` untuk detail.

### Empat pipeline utama

| Fungsi | Untuk index... | Butuh `base_period`? |
|---|---|---|
| `rain_indices()` | curah hujan (`pr`) | Tidak |
| `temp_indices()` | suhu 1-variabel (Tx, Tn, FD, SU, ...) & 2-variabel (DTR, ETR) | Tidak |
| `temp_percentile_indices()` | suhu berbasis persentil (tx90, wsdi, dst.) | **Ya** |
| `rain_percentile_indices()` | curah hujan berbasis persentil (R95P, R99P, dst.) | **Ya** |

Semua 4 pipeline juga menerima `slice_period` opsional — lihat [§7](#7-slice_period--membatasi-rentang-waktu-output-semua-pipeline).

### Nama dimensi waktu di hasil

- `slice_mode="ANN"` atau musiman (`DJF/MAM/JJA/SON`) → dimensi hasil bernama **`year`**.
- `slice_mode="ME"` (bulanan) → dimensi hasil tetap bernama **`time`**.

Ini konsisten di keempat pipeline.

---

## 2. Format data input

### Data grid

```python
import xarray as xr

ds = xr.Dataset(
    {
        "pr":     (["time", "lat", "lon"], pr_values,     {"units": "kg m-2 s-1"}),
        "tasmax": (["time", "lat", "lon"], tasmax_values,  {"units": "K"}),
        "tasmin": (["time", "lat", "lon"], tasmin_values,  {"units": "K"}),
        "tas":    (["time", "lat", "lon"], tas_values,     {"units": "K"}),
    },
    coords={"time": time_index, "lat": lat_values, "lon": lon_values},
)
```

`attrs['units']` **wajib diisi** kalau ingin konversi unit otomatis jalan
(`convert_units()` di `PreProp.py` mengecek string ini, bukan menebak).
Unit yang didukung otomatis:
- Presipitasi: `kg m-2 s-1` → `mm/day` (dikali 86400).
- Suhu (`tas`/`tasmax`/`tasmin`): `K` → `°C` (dikurangi 273.15).

Kalau datamu sudah dalam `mm/day` atau `°C`, cukup pastikan `attrs['units']`
TIDAK mengandung `'kg m-2 s-1'`/`'K'` supaya tidak dikonversi dua kali.

**Waktu (`time`) harus monotonic** (terurut naik, tanpa duplikat). Kalau
datamu hasil `xr.concat([hist, scen], dim='time')` (menggabung historis +
skenario), pastikan sudah di-`sortby('time')` dan bebas duplikat sebelum
dipakai:

```python
ds = xr.concat([hist, scen], dim="time").sortby("time").drop_duplicates(dim="time")
```

### Data stasiun

Lihat [§8](#8-data-stasiun) — pakai `from_wide_dataframe()` untuk
mengubah `pandas.DataFrame` jadi `xr.Dataset` dims `(time, station)`.

---

## 3. `rain_indices()` — index curah hujan

```python
rain_indices(
    dataset: xr.Dataset,
    varname: str = "pr",
    slice_period: tuple[str, str] | None = None,
    slice_mode: str = "ANN",       # "ANN" | "ME" | "DJF" | "MAM" | "JJA" | "SON"
    chunk_size: dict | None = None,
    verbose: bool = True,
) -> xr.Dataset
```

Menghitung **semua** index di `RAIN_INDICES` (`config.py`) sekaligus —
lihat rumus lengkap tiap index di `INDEX_FORMULAS.md` §1.

```python
result = rain_indices(ds, varname="pr", slice_mode="ANN")
result["PRCPTOT"]   # dims: (year, lat, lon)
```

---

## 4. `temp_indices()` — index suhu sederhana & dua-variabel

```python
temp_indices(
    dataset: xr.Dataset,
    slice_period: tuple[str, str] | None = None,
    slice_mode: str = "ANN",
    chunk_size: dict | None = None,
    indices: list[str] | None = None,   # None -> semua di TEMP_INDICES
    verbose: bool = True,
) -> xr.Dataset
```

Menghitung index di `TEMP_INDICES` (`config.py`): ekstrem harian
(`TXx/TXn/TNx/TNn/TMm`), hari-hitung threshold tetap (`FD/ID/SU/TR`), dan
index dua-variabel (`DTR/ETR`, butuh `tasmax` & `tasmin` sekaligus).

```python
# Semua index suhu (butuh tasmax, tasmin, tas — sesuai kebutuhan tiap index)
result = temp_indices(ds, slice_mode="ANN")

# Subset saja
result = temp_indices(ds, slice_mode="ANN", indices=["TXx", "FD", "DTR"])
```

**Variabel hilang tidak bikin crash.** Kalau `ds` cuma punya `tasmax`,
index yang butuh `tasmin`/`tas` (termasuk `DTR`/`ETR` yang butuh
`tasmin` juga) otomatis di-*skip* dengan pesan:

```
[temp_indices] SKIP 'FD': variabel ['tasmin'] tidak ada di dataset.
```

---

## 5. `temp_percentile_indices()` — index suhu berbasis persentil

```python
temp_percentile_indices(
    dataset: xr.Dataset,
    base_period: tuple[str, str],        # WAJIB, mis. ("1961-01-01", "1990-12-31")
    slice_period: tuple[str, str] | None = None,
    slice_mode: str = "ANN",
    chunk_size: dict | None = None,
    indices: list[str] | None = None,    # None -> semua di TEMP_PERCENTILE_INDICES
    window: int = 5,                     # lebar window +-hari untuk threshold per hari-kalender
    verbose: bool = True,
) -> xr.Dataset
```

Menghitung index di `TEMP_PERCENTILE_INDICES` (`config.py`):
persentase/jumlah hari ekstrem relatif terhadap persentil ke-90/ke-10
(`tg90/tg10/tn90/tn10/tx90/tx10` + versi `*abs`), dan durasi spell
(`wsdi`, `csdi`). Rumus lengkap ada di `INDEX_FORMULAS.md` §3.

```python
result = temp_percentile_indices(
    ds,
    base_period=("1961-01-01", "1990-12-31"),
    slice_mode="ANN",
    indices=["tx90", "tn10", "wsdi"],
)
```

**Cara kerja (2 tahap):**
1. **Threshold**: dari data di `base_period`, hitung persentil ke-q
   PER HARI-KALENDER (dayofyear 1–366, digabung dari semua tahun di
   `base_period`, pakai window ±`window` hari).
2. **Exceedance**: bandingkan SELURUH periode data (bukan cuma
   `base_period`) terhadap threshold hari-kalender yang sesuai, lalu
   agregasi per tahun/musim.

**Catatan performa**: data di `base_period` di-*load* penuh ke memori
(bukan lazy/dask) karena baseline biasanya cuma ~30 tahun — jauh lebih
kecil dari keseluruhan dataset, dan cuma dihitung sekali.

---

## 6. `rain_percentile_indices()` — index curah hujan berbasis persentil

```python
rain_percentile_indices(
    dataset: xr.Dataset,
    base_period: tuple[str, str],        # WAJIB, mis. ("1961-01-01", "1990-12-31")
    slice_period: tuple[str, str] | None = None,
    varname: str = "pr",
    slice_mode: str = "ANN",
    chunk_size: dict | None = None,
    indices: list[str] | None = None,    # None -> semua di RAIN_PERCENTILE_INDICES
    wet_day_threshold: float = 1.0,      # batas hari "basah" (mm), default sesuai ETCCDI
    verbose: bool = True,
) -> xr.Dataset
```

Menghitung `R95P`, `R99P`, `R95PTOT`, `R99PTOT` sesuai **definisi ETCCDI
standar**: threshold persentil ke-95/99 dihitung SEKALI dari SELURUH
hari basah di `base_period` — **digabung semua bulan, SATU angka tetap
per titik spasial** — lalu dipakai **sama untuk semua tahun/bulan/musim**.

```python
result = rain_percentile_indices(
    ds, base_period=("1961-01-01", "1990-12-31"),
    slice_mode="ANN", indices=["R95P", "R95PTOT"],
)
```

### ⚠️ Threshold TIDAK berbeda per bulan/musim — ini yang benar secara ETCCDI

Kalau kamu pakai `slice_mode="ME"` (bulanan) atau `slice_mode="JJA"`
(musiman), **threshold-nya tetap satu angka yang sama** — dipakai persis
sama untuk Januari maupun Juli. Yang berubah cuma **agregasinya**
(dijumlahkan per bulan/musim, bukan per tahun penuh).

Ini **bukan bug** — definisi ETCCDI R95p/R99p memang tidak membedakan
threshold per bulan/musim (beda dengan index suhu `tx90` yang memang
per hari-kalender). Kalau iklimmu punya musim kemarau yang jarang hujan
ekstrem, `R95P` bulan-bulan kemarau akan sering keluar **0** — itu
temuan yang benar (curah hujan ekstrem relatif terhadap iklim TAHUNAN
penuh memang jarang terjadi di musim kering), bukan kesalahan
perhitungan. Lihat diskusi lengkap & alasan metodologisnya di
`INDEX_FORMULAS.md` §4.

---

## 7. `slice_period` — membatasi rentang waktu output (semua pipeline)

Semua 4 pipeline menerima `slice_period: tuple(str, str) | None` untuk
membatasi rentang waktu yang **dihitung index-nya** — berguna untuk
dataset gabungan historis + skenario (mis. hasil `xr.concat([hist, scen])`)
di mana kamu cuma mau index untuk periode skenario saja, tapi
threshold/preprocessing tetap mempertimbangkan seluruh dataset.

**Untuk `rain_indices()`/`temp_indices()`** (tanpa `base_period`):
`slice_period` langsung membatasi data SEBELUM index dihitung.

```python
result = temp_indices(hist_scen, slice_period=("2015-01-01", "2100-12-31"), slice_mode="ANN")
```

**Untuk `temp_percentile_indices()`/`rain_percentile_indices()`** (dengan
`base_period`): urutannya penting — **threshold dihitung dulu dari
`base_period` di data PENUH**, baru **SETELAH itu** `slice_period`
diterapkan untuk membatasi periode yang dihitung index-nya. `base_period`
dan `slice_period` boleh (dan lazimnya) berada di rentang yang berbeda:

```python
result = rain_percentile_indices(
    hist_scen,
    base_period=("1981-01-01", "2014-12-31"),   # threshold dari sini (historis)
    slice_period=("2015-01-01", "2100-12-31"),  # tapi index cuma dihitung utk periode ini (skenario)
    slice_mode="ANN",
)
```

Kalau `slice_period=None` (default), seluruh rentang waktu di `dataset`
dipakai — backward-compatible dengan pemanggilan tanpa parameter ini.

---

## 8. Data stasiun

```python
from indek_ekstrim import from_wide_dataframe, from_long_dataframe, result_to_dataframe
```

### `from_wide_dataframe(df, varname="pr", units="mm/day", station_dim="station")`

Untuk `df` dengan index = waktu, kolom = nama/ID stasiun:

```python
df = pd.read_csv("curah_hujan_stasiun.csv", index_col=0, parse_dates=True)
ds = from_wide_dataframe(df, varname="pr", units="mm/day")
result = rain_indices(ds, varname="pr", slice_mode="ANN")
```

### `from_long_dataframe(df, time_col="time", station_col="station_id", value_col="value", ...)`

Untuk `df` long-format (satu baris = satu observasi time × station):

```python
ds = from_long_dataframe(df, time_col="tanggal", station_col="id_stasiun", value_col="curah_hujan")
```

### `result_to_dataframe(result, station_dim="station")`

Konversi hasil (`xr.Dataset` dims `(year/time, station)`) balik ke
`pandas.DataFrame` long-format, siap diekspor CSV:

```python
df = result_to_dataframe(rain_result, station_dim="station")
df.to_csv("hasil_index.csv", index=False)
```

---

## 9. Katalog lengkap index yang tersedia

Rumus matematis lengkap tiap index ada di **[`INDEX_FORMULAS.md`](./INDEX_FORMULAS.md)**.
Ringkasannya:

### `RAIN_INDICES` (curah hujan, `rain_indices()`)

| Nama | Deskripsi |
|---|---|
| `RX1DAY..RX10DAY` | Curah hujan maksimum kumulatif dalam jendela N hari |
| `HH` / `HH20MM` / `HH50MM` / `HH100MM` / `HH150MM` | Jumlah hari curah hujan ≥ threshold (mm) |
| `FH20MM` / `FH50MM` / `FH100MM` / `FH150MM` | Persentase hari sangat basah relatif thd hari basah |
| `PRCPTOT` | Total curah hujan tahunan/musiman |
| `CDD` | Consecutive Dry Days (runtun terpanjang hari kering) |
| `CWD` | Consecutive Wet Days (runtun terpanjang hari basah) |
| `SDII` | Simple Daily Intensity Index |

### `TEMP_INDICES` (suhu sederhana & dua-variabel, `temp_indices()`)

| Nama | Variabel | Deskripsi |
|---|---|---|
| `TXx` / `TXn` | `tasmax` | Max / min Tmax dalam periode |
| `TNx` / `TNn` | `tasmin` | Max / min Tmin dalam periode |
| `TMm` | `tas` | Rata-rata Tas |
| `FD` | `tasmin` | Frost Days: Tmin < 0°C |
| `ID` | `tasmax` | Icing Days: Tmax < 0°C |
| `SU` | `tasmax` | Summer Days: Tmax > 25°C |
| `TR` | `tasmin` | Tropical Nights: Tmin > 20°C |
| `DTR` | `tasmax` + `tasmin` | Rata-rata (Tmax − Tmin) |
| `ETR` | `tasmax` + `tasmin` | max(Tmax) − min(Tmin) dalam periode |

### `TEMP_PERCENTILE_INDICES` (suhu berbasis persentil, `temp_percentile_indices()`, butuh `base_period`)

| Nama | Variabel | Deskripsi |
|---|---|---|
| `tg90` / `tg10` | `tas` | % hari Tas di atas p90 / di bawah p10 (per hari-kalender) |
| `tn90` / `tn10` | `tasmin` | % hari Tmin di atas p90 / di bawah p10 |
| `tx90` / `tx10` | `tasmax` | % hari Tmax di atas p90 / di bawah p10 |
| `*abs` (`tg90abs`, dst.) | sama seperti di atas | Jumlah hari absolut (bukan %) |
| `wsdi` | `tasmax` | Warm Spell Duration Index: total hari dalam runtun ≥6 hari Tmax > p90 |
| `csdi` | `tasmin` | Cold Spell Duration Index: total hari dalam runtun ≥6 hari Tmin < p10 |

### `RAIN_PERCENTILE_INDICES` (curah hujan berbasis persentil, `rain_percentile_indices()`, butuh `base_period`)

| Nama | Deskripsi |
|---|---|
| `R95P` / `R99P` | Total curah hujan hari basah di atas threshold TETAP (persentil ke-95/99 dari base_period, digabung semua bulan) |
| `R95PTOT` / `R99PTOT` | Kontribusi (%) curah hujan ekstrim tsb terhadap total curah hujan hari basah pada periode yang sama |

---

## 10. Struktur file & tanggung jawab tiap modul

```
indek_ekstrim/
├── __init__.py         # ekspor API publik
├── config.py           # SATU-SATUNYA tempat mendaftarkan index (IndexSpec, PercentileIndexSpec, RainPercentileIndexSpec, *_INDICES)
├── ettcdi.py           # fungsi index individual (1D array -> skalar), + FUNC_MAP
├── math_utils.py       # helper matematis generik (divide, longest_run, count_days_in_runs)
├── engine.py           # apply_ufunc wiring untuk IndexSpec kind="simple"/"dual"
├── percentile.py       # threshold hari-kalender + exceedance + agregasi suhu (TEMP_PERCENTILE_INDICES)
├── rain_percentile.py  # threshold hari-basah TETAP + agregasi curah hujan (RAIN_PERCENTILE_INDICES)
├── PreProp.py          # convert_units, grouped_dataset, deteksi nama koordinat, dll.
├── pipeline.py         # 4 entry point publik: rain_indices / temp_indices / temp_percentile_indices / rain_percentile_indices
└── stations.py         # adapter DataFrame <-> xr.Dataset untuk data stasiun
```

Alur data secara umum: **config.py** (deklarasi index) → **pipeline.py**
(orkestrasi: siapkan variabel, persist, groupby, terapkan `slice_period`)
→ **engine.py**/**percentile.py**/**rain_percentile.py** (bangun graph
lazy per index) → `dask.compute()` sekali di `pipeline.py` → `xr.Dataset`
hasil.

---

## 11. Menambah index baru

Lihat dokumen terpisah: **[`ADDING_NEW_INDEX.md`](./doc/ADDING_NEW_INDEX.md)**
— mencakup 3 pola berbeda (1-variabel, 2-variabel, berbasis persentil)
lengkap dengan contoh kode & checklist testing.

# rain_indices_pkg

Package untuk menghitung index iklim (curah hujan & suhu, gaya ETCCDI) dari
data `xarray` — baik data **grid** (dims `time, lat, lon`) maupun data
**titik/stasiun** (dims `time, station`), memakai kode perhitungan yang
sama persis untuk keduanya.

```python
from rain_indices_pkg import rain_indices, temp_indices, percentile_indices
```

---

## Daftar Isi

1. [Konsep dasar & arsitektur](#1-konsep-dasar--arsitektur)
2. [Format data input](#2-format-data-input)
3. [`rain_indices()` — index curah hujan](#3-rain_indices--index-curah-hujan)
4. [`temp_indices()` — index suhu sederhana & dua-variabel](#4-temp_indices--index-suhu-sederhana--dua-variabel)
5. [`percentile_indices()` — index suhu berbasis persentil](#5-percentile_indices--index-suhu-berbasis-persentil)
6. [Data stasiun: `from_wide_dataframe` / `from_long_dataframe` / `result_to_dataframe`](#6-data-stasiun)
7. [Katalog lengkap index yang tersedia](#7-katalog-lengkap-index-yang-tersedia)
8. [Struktur file & tanggung jawab tiap modul](#8-struktur-file--tanggung-jawab-tiap-modul)
9. [Menambah index baru](#9-menambah-index-baru)

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

### Tiga pipeline utama

| Fungsi | Untuk index... | Butuh `base_period`? |
|---|---|---|
| `rain_indices()` | curah hujan (`pr`) | Tidak |
| `temp_indices()` | suhu 1-variabel (Tx, Tn, FD, SU, ...) & 2-variabel (DTR, ETR) | Tidak |
| `percentile_indices()` | suhu berbasis persentil (tx90, wsdi, dst.) | **Ya** |

### Nama dimensi waktu di hasil

- `slice_mode="ANN"` atau musiman (`DJF/MAM/JJA/SON`) → dimensi hasil bernama **`year`**.
- `slice_mode="ME"` (bulanan) → dimensi hasil tetap bernama **`time`**.

Ini konsisten di ketiga pipeline.

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

### Data stasiun

Lihat [bagian 6](#6-data-stasiun) — pakai `from_wide_dataframe()` untuk
mengubah `pandas.DataFrame` jadi `xr.Dataset` dims `(time, station)`.

---

## 3. `rain_indices()` — index curah hujan

```python
rain_indices(
    dataset: xr.Dataset,
    varname: str = "pr",
    slice_mode: str = "ANN",       # "ANN" | "ME" | "DJF" | "MAM" | "JJA" | "SON"
    chunk_size: dict | None = None,
    verbose: bool = True,
) -> xr.Dataset
```

Menghitung **semua** index di `RAIN_INDICES` (`config.py`) sekaligus.

```python
result = rain_indices(ds, varname="pr", slice_mode="ANN")
result["PRCPTOT"]   # dims: (year, lat, lon)
```

---

## 4. `temp_indices()` — index suhu sederhana & dua-variabel

```python
temp_indices(
    dataset: xr.Dataset,
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

## 5. `percentile_indices()` — index suhu berbasis persentil

```python
percentile_indices(
    dataset: xr.Dataset,
    base_period: tuple[str, str],        # WAJIB, mis. ("1961-01-01", "1990-12-31")
    slice_mode: str = "ANN",
    chunk_size: dict | None = None,
    indices: list[str] | None = None,    # None -> semua di PERCENTILE_INDICES
    window: int = 5,                     # lebar window +-hari untuk threshold per hari-kalender
    verbose: bool = True,
) -> xr.Dataset
```

Menghitung index di `PERCENTILE_INDICES` (`config.py`): persentase/jumlah
hari ekstrem relatif terhadap persentil ke-90/ke-10 (`tg90/tg10/tn90/tn10/
tx90/tx10` + versi `*abs`), dan durasi spell (`wsdi`, `csdi`).

```python
result = percentile_indices(
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
kecil dari keseluruhan dataset, dan cuma dihitung sekali. Kalau grid-mu
sangat besar dan base period-nya tidak muat memori, ini perlu direvisi
jadi versi dask-native.

**Threshold & exceedance di-cache per `(variabel, q)`** — minta `tx90`
dan `tx90abs` bareng tidak menghitung threshold `tasmax` p90 dua kali.

---

## 6. Data stasiun

```python
from rain_indices_pkg import from_wide_dataframe, from_long_dataframe, result_to_dataframe
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

## 7. Katalog lengkap index yang tersedia

### `RAIN_INDICES` (curah hujan)

| Nama | Deskripsi |
|---|---|
| `RX1DAY..RX10DAY` | Curah hujan maksimum kumulatif dalam jendela N hari |
| `HH` / `HH20MM` / `HH50MM` / `HH100MM` / `HH150MM` | Jumlah hari curah hujan ≥ threshold (mm) |
| `FH20MM` / `FH50MM` / `FH100MM` / `FH150MM` | Persentase hari sangat basah relatif thd hari basah |
| `PRCPTOT` | Total curah hujan tahunan/musiman |
| `CDD` | Consecutive Dry Days (runtun terpanjang hari kering) |
| `CWD` | Consecutive Wet Days (runtun terpanjang hari basah) |
| `SDII` | Simple Daily Intensity Index |
| `R95P` / `R99P` | Total curah hujan hari basah ekstrim di atas persentil ke-95/99 |
| `R95PTOT` / `R99PTOT` | Kontribusi (%) curah hujan ekstrim terhadap total |

### `TEMP_INDICES` (suhu sederhana & dua-variabel)

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

### `PERCENTILE_INDICES` (suhu berbasis persentil, butuh `base_period`)

| Nama | Variabel | Deskripsi |
|---|---|---|
| `tg90` / `tg10` | `tas` | % hari Tas di atas p90 / di bawah p10 |
| `tn90` / `tn10` | `tasmin` | % hari Tmin di atas p90 / di bawah p10 |
| `tx90` / `tx10` | `tasmax` | % hari Tmax di atas p90 / di bawah p10 |
| `*abs` (`tg90abs`, dst.) | sama seperti di atas | Jumlah hari absolut (bukan %) |
| `wsdi` | `tasmax` | Warm Spell Duration Index: total hari dalam runtun ≥6 hari Tmax > p90 |
| `csdi` | `tasmin` | Cold Spell Duration Index: total hari dalam runtun ≥6 hari Tmin < p10 |

---

## 8. Struktur file & tanggung jawab tiap modul

```
rain_indices_pkg/
├── __init__.py     # ekspor API publik
├── config.py       # SATU-SATUNYA tempat mendaftarkan index (IndexSpec, PercentileIndexSpec, *_INDICES)
├── ettcdi.py        # fungsi index individual (1D array -> skalar), + FUNC_MAP
├── math_utils.py    # helper matematis generik (divide, longest_run, count_days_in_runs)
├── engine.py        # apply_ufunc wiring untuk IndexSpec kind="simple"/"quantile"/"dual"
├── percentile.py     # threshold hari-kalender + exceedance + agregasi (untuk PercentileIndexSpec)
├── PreProp.py         # convert_units, grouped_dataset, deteksi nama koordinat, dll.
├── pipeline.py         # 3 entry point publik: rain_indices / temp_indices / percentile_indices
└── stations.py          # adapter DataFrame <-> xr.Dataset untuk data stasiun
```

Alur data secara umum: **config.py** (deklarasi index) → **pipeline.py**
(orkestrasi: siapkan variabel, persist, groupby) → **engine.py**/
**percentile.py** (bangun graph lazy per index) → `dask.compute()` sekali
di `pipeline.py` → `xr.Dataset` hasil.

---

## 9. Menambah index baru

Lihat dokumen terpisah: **[`ADDING_NEW_INDEX.md`](./ADDING_NEW_INDEX.md)**
— mencakup 3 pola berbeda (1-variabel, 2-variabel, berbasis persentil)
lengkap dengan contoh kode & checklist testing.

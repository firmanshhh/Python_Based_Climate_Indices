# Menambah Index Baru — Panduan Alur

Package ini punya **3 pola arsitektur** tergantung jenis index yang mau
ditambah. Langkah pertama & terpenting: **kenali dulu index barumu masuk
pola yang mana**, sebelum mulai ngoding.

```
┌─────────────────────────────────────────────────────────────┐
│  Apakah index butuh 2 variabel SEKALIGUS dalam 1 window?     │
│  (mis. Tmax - Tmin di HARI YANG SAMA)                        │
└───────────────┬─────────────────────────────┬───────────────┘
                YA                             TIDAK
                │                                │
                ▼                                ▼
        ┌───────────────┐         ┌─────────────────────────────┐
        │  POLA B: dual  │         │ Apakah butuh threshold dari  │
        │  (DTR, ETR)    │         │ periode referensi (base      │
        └───────────────┘         │ period) yang dibandingkan ke │
                                   │ hari-kalender tertentu?       │
                                   │ (mis. persentil ke-90 tiap    │
                                   │ tanggal 1 Jan, 2 Jan, dst.)   │
                                   └───────┬───────────────┬───────┘
                                          YA              TIDAK
                                          │                │
                                          ▼                ▼
                                ┌──────────────────┐ ┌──────────────┐
                                │ POLA C: percentile │ │ POLA A:      │
                                │ (tx90, wsdi, dst.)  │ │ simple       │
                                └──────────────────┘ │ (Tx, Tn, FD)  │
                                                       └──────────────┘
```

---

## POLA A — Index 1-variabel, 1-tahap ("simple")

**Ciri-ciri:** Index cuma butuh SATU array 1D per window waktu (satu
tahun/musim dari SATU variabel), dan menghasilkan satu angka. Contoh:
`Tx` (max), `FD` (count hari di bawah threshold), `SDII`.

Ini pola PALING SERING dipakai — 90% index yang sudah ada masuk pola ini.

### Langkah-langkah

**1. Tulis fungsinya di `ettcdi.py`**

```python
def my_new_index(data, threshold=10):
    """Deskripsi singkat apa yang dihitung index ini."""
    data = np.asarray(data, dtype=float)
    if np.all(np.isnan(data)):
        return np.nan
    # ... logika perhitungan, HARUS vectorized (hindari loop Python
    #     kalau bisa; kalau butuh runtun/streak, pakai
    #     math_utils.longest_run / count_days_in_runs)
    return hasil_skalar
```

**Aturan penting:**
- Terima SATU array 1D (`data`), kembalikan SATU skalar (float/NaN).
- Selalu tangani kasus semua-NaN (`return np.nan`), jangan biarkan
  `apply_ufunc` meledak di tengah dataset besar.
- Parameter tambahan (`threshold`, dst.) lewat keyword argument dengan
  default value.

**2. Daftarkan ke `FUNC_MAP`** (masih di `ettcdi.py`, paling bawah):

```python
FUNC_MAP = {
    ...
    'my_new_index': my_new_index,
}
```

> Key di `FUNC_MAP` HARUS sama persis (case-sensitive) dengan string
> `func=` yang dipakai di `config.py` — ini penyebab error paling umum
> (`KeyError`) kalau typo.

**3. Tambahkan `IndexSpec` di `config.py`**

Kalau berbasis curah hujan → masuk `RAIN_INDICES`. Kalau suhu → masuk
`TEMP_INDICES` dengan `var=` menunjuk variabel sumbernya:

```python
TEMP_INDICES: Dict[str, IndexSpec] = {
    ...
    "MYIDX": IndexSpec("my_new_index", {"threshold": 10}, var="tasmax"),
}
```

- `func` = nama key di `FUNC_MAP` (langkah 2).
- `params` (posisi kedua) = dict kwargs yang dioper ke fungsimu.
- `var` = variabel sumber (`'tasmax'`/`'tasmin'`/`'tas'`) — WAJIB untuk
  `TEMP_INDICES`, TIDAK dipakai untuk `RAIN_INDICES` (semua dari `pr`).

**4. Selesai — TIDAK perlu ubah `engine.py` atau `pipeline.py`.**

**5. Test:**

```python
result = temp_indices(ds, slice_mode="ANN", indices=["MYIDX"])
print(result["MYIDX"])
```

Cek juga manual di satu titik/tahun untuk memastikan angkanya benar:

```python
import numpy as np
sub = ds["tasmax"].sel(time=slice("2019-01-01","2019-12-31")).isel(lat=0, lon=0)
print(my_new_index(sub.values, threshold=10))                # manual
print(result["MYIDX"].sel(year=2019).isel(lat=0, lon=0).values)  # dari pipeline, harus sama
```

---

## POLA B — Index 2-variabel dalam window yang sama ("dual")

**Ciri-ciri:** Butuh dua variabel (mis. `tasmax` DAN `tasmin`) dari HARI
YANG SAMA secara bersamaan — bukan dihitung terpisah lalu digabung
setelah agregasi. Contoh yang sudah ada: `DTR`, `ETR`.

### Langkah-langkah

**1. Tulis fungsinya di `ettcdi.py`** — terima DUA array 1D:

```python
def my_dual_index(data_var1, data_var2):
    """Deskripsi. data_var1 & data_var2 sudah selaras di 'time'."""
    data_var1 = np.asarray(data_var1, dtype=float)
    data_var2 = np.asarray(data_var2, dtype=float)
    if np.all(np.isnan(data_var1)) or np.all(np.isnan(data_var2)):
        return np.nan
    return hasil_skalar  # mis. np.nanmean(data_var1 - data_var2)
```

**2. Daftarkan ke `FUNC_MAP`** — sama seperti Pola A.

**3. Tambahkan `IndexSpec` dengan `kind="dual"` + `var2`:**

```python
"MYDUAL": IndexSpec("my_dual_index", kind="dual", var="tasmax", var2="tasmin"),
```

**4. Selesai — `engine._apply_dual` dan `pipeline.temp_indices()` sudah
generik**, otomatis menyiapkan kedua variabel dan mengoper keduanya ke
fungsimu lewat `input_core_dims=[["time"], ["time"]]`.

**5. Test** — sama seperti Pola A, plus pastikan index di-*skip* dengan
benar kalau salah satu dari dua variabel tidak ada di dataset:

```python
ds_incomplete = ds[["tasmax"]]  # sengaja hilangkan tasmin
result = temp_indices(ds_incomplete, indices=["MYDUAL"], verbose=True)
# harus muncul: [temp_indices] SKIP 'MYDUAL': variabel ['tasmin'] tidak ada di dataset.
```

---

## POLA C — Index berbasis persentil / threshold hari-kalender ("percentile")

**Ciri-ciri:** Butuh **periode referensi** (`base_period`) untuk
menghitung threshold persentil PER HARI-KALENDER (dayofyear), lalu
dibandingkan ke seluruh data. Contoh yang sudah ada: `tx90`, `tn10`,
`wsdi`, `csdi`.

**Ini BUKAN kasus untuk `IndexSpec` biasa** — jangan dipaksakan ke
`RAIN_INDICES`/`TEMP_INDICES`. Arsitekturnya beda: threshold dihitung
SEKALI (bukan per-window), lalu dipakai berulang.

### Kapan index barumu masuk sini?

Kalau definisinya mengandung kata-kata seperti: *"persentil ke-N dari
periode referensi 1961-1990"*, *"relatif terhadap normal/klimatologi"*,
atau *"durasi spell di atas/bawah threshold"* — ini Pola C.

**Ada 2 varian Pola C, jangan tertukar:**

| | Index suhu (`percentile.py`) | Index curah hujan (`rain_percentile.py`) |
|---|---|---|
| Registry | `TEMP_PERCENTILE_INDICES` | `RAIN_PERCENTILE_INDICES` |
| Threshold | Per **hari-kalender** (dayofyear, window ±N hari) | **Satu nilai tetap** per titik spasial/stasiun |
| Contoh | `tx90`, `wsdi` | `R95P`, `R99P` |
| Kenapa beda? | Suhu ekstrim bergantung musim (30°C beda makna di Januari vs Juli) | Curah hujan ekstrim ETCCDI didefinisikan tidak bergantung musim |

Kalau index barumu tentang **suhu**, ikuti pola `percentile.py` di
bawah. Kalau tentang **curah hujan** (atau variabel lain yang tidak
musiman), ikuti pola `rain_percentile.py` — lihat bagian
[Pola C-2](#pola-c-2--index-curah-hujanvariabel-non-musiman-berbasis-persentil)
di akhir dokumen ini.

### Langkah-langkah

**1. Cek dulu apakah `mode` yang ada di `percentile.py` sudah cukup:**

| `mode` | Untuk apa |
|---|---|
| `"pct"` | % hari exceed relatif thd total hari per periode |
| `"abs"` | Jumlah hari exceed (angka absolut) |
| `"spell"` | Total hari dalam runtun ≥ `min_run` hari berturut-turut |

Kalau index barumu cuma kombinasi variabel + arah (`above`/`below`) +
salah satu dari 3 mode di atas → **tidak perlu tulis kode baru sama
sekali**, cukup tambah entri di `config.py`:

```python
TEMP_PERCENTILE_INDICES: Dict[str, PercentileIndexSpec] = {
    ...
    "my_pctidx": PercentileIndexSpec("tasmax", 0.95, "above", "pct"),
    # var="tasmax", q=0.95, op="above" (bandingkan '>'), mode="pct" (%)
}
```

Langsung bisa dipakai:
```python
result = temp_percentile_indices(ds, base_period=("1961-01-01","1990-12-31"),
                              indices=["my_pctidx"])
```

**2. Kalau butuh agregasi yang BENAR-BENAR baru** (bukan pct/abs/spell),
mis. index yang butuh statistik lain dari runtun (rata-rata panjang
spell, bukan total hari) — tambah fungsi baru di `percentile.py`:

```python
def _my_new_aggregation(mask, **kwargs):
    # mask = array 0/1/NaN per hari dalam 1 window (tahun/musim)
    ...
    return hasil_skalar
```

lalu tambahkan cabang di `aggregate_index()`:

```python
elif mode == "my_new_mode":
    result = xr.apply_ufunc(
        _my_new_aggregation, grouped,
        input_core_dims=[["time"]],
        kwargs={...},
        vectorize=True, dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True}, keep_attrs=True,
    )
```

Lalu tambah `Literal["pct","abs","spell","my_new_mode"]` di
`PercentileIndexSpec.mode` (`config.py`).

**3. Test — WAJIB cek statistik masuk akal, bukan cuma "tidak error":**

Untuk index `mode="pct"`, buat data sintetis dengan iklim STASIONER
(tidak trending) dan cek angkanya mendekati `(1-q)*100` atau `q*100`
tergantung `op`:

```python
result = temp_percentile_indices(ds_stationer, base_period=(...), indices=["my_pctidx"])
print(float(result["my_pctidx"].mean()))  # harusnya ~ (1-0.95)*100 = 5%, kalau op="above"
```

Kalau meleset jauh dari situ, kemungkinan ada bug di arah perbandingan
(`above` vs `below`) atau di alignment `dayofyear`.

---

## POLA C-2 — Index curah hujan/variabel non-musiman berbasis persentil

**Ciri-ciri:** Sama seperti Pola C, tapi threshold-nya **satu nilai
tetap** per titik spasial/stasiun (bukan per hari-kalender) — karena
tidak ada alasan musiman untuk variasi threshold sepanjang tahun.
Contoh yang sudah ada: `R95P`, `R99P`, `R95PTOT`, `R99PTOT`.

### Langkah-langkah

**1. Cek dulu apakah `mode` yang ada di `rain_percentile.py` sudah cukup:**

| `mode` | Untuk apa |
|---|---|
| `"sum"` | Total nilai di atas threshold (mis. R95P) |
| `"pct"` | Kontribusi (%) nilai di atas threshold terhadap total (mis. R95PTOT) |

Kalau cukup, **tidak perlu kode baru** — cukup tambah entri:

```python
RAIN_PERCENTILE_INDICES: Dict[str, RainPercentileIndexSpec] = {
    ...
    "R90P": RainPercentileIndexSpec(0.90, "sum"),
}
```

Langsung dipakai:
```python
result = rain_percentile_indices(ds, base_period=("1961-01-01","1990-12-31"),
                                   indices=["R90P"])
```

**2. Kalau butuh agregasi baru** — tambah fungsi di `rain_percentile.py`
(pola sama seperti `_rqp_sum`/`_rqp_pct`), lalu tambah cabang di
`aggregate_rain_percentile()`.

**3. Test — WAJIB verifikasi threshold benar-benar TETAP** (bukan
berubah per tahun) dan cocok dengan hitungan manual:

```python
from rain_indices_pkg.rain_percentile import compute_wet_day_threshold

base_data = ds["pr"].sel(time=slice("1961-01-01","1990-12-31"))
th = compute_wet_day_threshold(base_data, 0.90)
print(th.values)  # harus 1 angka per pixel, TIDAK per tahun

# manual check 1 tahun tertentu, pakai threshold di atas (BUKAN dihitung ulang dari tahun itu)
sub = ds["pr"].sel(time=slice("2015-01-01","2015-12-31")).isel(lat=0, lon=0).values
manual = float((sub[sub > float(th.isel(lat=0, lon=0))]).sum())
pipeline_val = float(result["R90P"].sel(year=2015).isel(lat=0, lon=0))
assert abs(manual - pipeline_val) < 1e-6
```

---

## Checklist ringkas (semua pola)

- [ ] Fungsi di `ettcdi.py` (atau `percentile.py` untuk mode baru) sudah
      handle kasus semua-NaN.
- [ ] Nama fungsi terdaftar persis sama di `FUNC_MAP` (Pola A/B) —
      cek typo, case-sensitive.
- [ ] Entri baru di `config.py` (`RAIN_INDICES`/`TEMP_INDICES`/
      `TEMP_PERCENTILE_INDICES`) dengan `var`/`var2` yang benar.
- [ ] **TIDAK** ada perubahan di `pipeline.py`/`engine.py` untuk Pola A
      dan B — kalau sampai perlu ubah, kemungkinan besar index-nya
      salah pola.
- [ ] Sudah dites: jalan tanpa error, DAN angkanya dicek manual/statistik
      masuk akal (bukan cuma "tidak crash").
- [ ] Sudah dites kasus variabel sumber hilang dari dataset → harus
      di-*skip* dengan pesan jelas, bukan crash seluruh pipeline.

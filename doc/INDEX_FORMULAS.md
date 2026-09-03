# Definisi & Rumus Index

Dokumen ini menjelaskan **rumus matematis** tiap index yang dihitung
package ini — pelengkap `README.md` (yang fokus ke *cara pakai*) dan
`ADDING_NEW_INDEX.md` (yang fokus ke *cara menambah*).

Notasi umum:
- $x_i$ = nilai variabel pada hari ke-$i$ dalam satu window (tahun/bulan/musim).
- $N$ = jumlah hari dalam window tersebut.
- Semua index dihitung **per window** (per tahun untuk `slice_mode="ANN"`,
  per bulan untuk `"ME"`, per musim untuk `"DJF"/"MAM"/"JJA"/"SON"`) dan
  **per titik spasial/stasiun** secara independen.

---

## Daftar Isi

1. [Index curah hujan — `RAIN_INDICES`](#1-index-curah-hujan--rain_indices)
2. [Index suhu sederhana & dua-variabel — `TEMP_INDICES`](#2-index-suhu-sederhana--dua-variabel--temp_indices)
3. [Index suhu berbasis persentil — `TEMP_PERCENTILE_INDICES`](#3-index-suhu-berbasis-persentil--temp_percentile_indices)
4. [Index curah hujan berbasis persentil — `RAIN_PERCENTILE_INDICES`](#4-index-curah-hujan-berbasis-persentil--rain_percentile_indices)
5. [Fungsi pendukung (`math_utils.py`)](#5-fungsi-pendukung-math_utilspy)

---

## 1. Index curah hujan — `RAIN_INDICES`

Variabel: $x_i$ = curah hujan hari ke-$i$ (mm/hari). "Hari basah" = $x_i \geq 1$mm.

| Index | Rumus | Keterangan |
|---|---|---|
| `RX1DAY` | $\max_i(x_i)$ | Curah hujan 1 hari maksimum dalam window |
| `RX3DAY`, `RX5DAY`, `RX7DAY`, `RX10DAY` | $\displaystyle\max_{k} \sum_{i=k}^{k+n-1} x_i$ | Curah hujan kumulatif maksimum untuk jendela bergulir $n$ hari ($n$ = 3/5/7/10), memakai rolling-sum |
| `HH` | $\sum_i \mathbb{1}[x_i \geq 1]$ | Jumlah hari basah (≥1mm) |
| `HH20MM` / `HH50MM` / `HH100MM` / `HH150MM` | $\sum_i \mathbb{1}[x_i \geq T]$, $T \in \{20,50,100,150\}$ | Jumlah hari curah hujan ≥ $T$ mm |
| `FH20MM` / `FH50MM` / `FH100MM` / `FH150MM` | $\dfrac{\sum_i \mathbb{1}[x_i \geq T]}{\sum_i \mathbb{1}[x_i \geq 1]} \times 100$ | Persentase hari "sangat basah" (≥$T$mm) relatif terhadap hari basah (≥1mm) |
| `PRCPTOT` | $\sum_i x_i \cdot \mathbb{1}[x_i > 0]$ | Total curah hujan pada hari-hari basah |
| `CDD` | $\max(\text{panjang runtun } x_i < 1\text{mm berturut-turut})$ | Consecutive Dry Days — runtun terpanjang hari kering |
| `CWD` | $\max(\text{panjang runtun } x_i \geq 1\text{mm berturut-turut})$ | Consecutive Wet Days — runtun terpanjang hari basah |
| `SDII` | $\dfrac{\sum_i x_i \cdot \mathbb{1}[x_i \geq 1]}{\sum_i \mathbb{1}[x_i \geq 1]}$ | Simple Daily Intensity Index — rata-rata curah hujan PER HARI BASAH |

**Catatan implementasi**: `RxNDay` memakai `np.convolve` (rolling-sum
tervektorisasi, bukan loop Python). `CDD`/`CWD` memakai
`math_utils.longest_run()` (run-length encoding tervektorisasi).

---

## 2. Index suhu sederhana & dua-variabel — `TEMP_INDICES`

Variabel: $x_i$ = suhu hari ke-$i$ (°C, dari `tasmax`/`tasmin`/`tas`
tergantung index).

| Index | Rumus | Variabel sumber |
|---|---|---|
| `TXx` | $\max_i(x_i)$ | `tasmax` |
| `TXn` | $\min_i(x_i)$ | `tasmax` |
| `TNx` | $\max_i(x_i)$ | `tasmin` |
| `TNn` | $\min_i(x_i)$ | `tasmin` |
| `TMm` | $\dfrac{1}{N}\sum_i x_i$ | `tas` |
| `FD` (Frost Days) | $\sum_i \mathbb{1}[x_i < 0]$ | `tasmin` |
| `ID` (Icing Days) | $\sum_i \mathbb{1}[x_i < 0]$ | `tasmax` |
| `SU` (Summer Days) | $\sum_i \mathbb{1}[x_i > 25]$ | `tasmax` |
| `TR` (Tropical Nights) | $\sum_i \mathbb{1}[x_i > 20]$ | `tasmin` |
| `DTR` | $\dfrac{1}{N}\sum_i (x^{tasmax}_i - x^{tasmin}_i)$ | `tasmax` **dan** `tasmin` (hari yang sama) |
| `ETR` | $\max_i(x^{tasmax}_i) - \min_i(x^{tasmin}_i)$ | `tasmax` **dan** `tasmin` |

**Catatan**: `DTR` = rata-rata *Diurnal Temperature Range* (Tmax−Tmin
harian, dirata-ratakan sepanjang window). `ETR` = *Extreme Temperature
Range* (selisih ekstrem window, BUKAN dirata-ratakan harian) — dua
definisi ini sering tertukar, package ini mengikuti konvensi ETCCDI:
DTR = rata-rata harian, ETR = selisih dari 2 nilai ekstrem window.

---

## 3. Index suhu berbasis persentil — `TEMP_PERCENTILE_INDICES`

### 3.1 Tahap threshold (dilakukan sekali, dari `base_period`)

Untuk tiap hari-kalender $d$ (dayofyear, $d = 1, \dots, 366$), dan
window $w$ (default $w=5$):

$$
T_q(d) = \text{Quantile}_q\Big(\{x_t : \text{dayofyear}(t) \in [d-w, d+w] \pmod{366},\ t \in \text{base\_period}\}\Big)
$$

Threshold digabung dari **semua tahun** di `base_period` untuk tiap
$d$, sehingga tiap hari-kalender punya threshold sendiri yang mulus
berubah sepanjang tahun (bukan blok diskrit per bulan).

### 3.2 Tahap exceedance & agregasi

Untuk tiap hari $t$ di **seluruh periode data** (bukan cuma `base_period`):

$$
E(t) = \begin{cases}
1 & \text{jika } x_t > T_{q}(\text{dayofyear}(t)) \text{ (untuk op="above")} \\
1 & \text{jika } x_t < T_{q}(\text{dayofyear}(t)) \text{ (untuk op="below")} \\
0 & \text{selainnya}
\end{cases}
$$

Lalu diagregasi per window ($N$ = jumlah hari di window):

| `mode` | Rumus | Index yang pakai |
|---|---|---|
| `"pct"` | $\dfrac{\sum_t E(t)}{N} \times 100$ | `tg90`, `tg10`, `tn90`, `tn10`, `tx90`, `tx10` |
| `"abs"` | $\sum_t E(t)$ | `tg90abs`, `tg10abs`, `tn90abs`, `tn10abs`, `tx90abs`, `tx10abs` |
| `"spell"` | $\displaystyle\sum_{\text{runtun } r,\ \text{len}(r) \geq 6} \text{len}(r)$, dari runtun $E(t)=1$ berturut-turut | `wsdi` (op="above", var=`tasmax`, q=0.90), `csdi` (op="below", var=`tasmin`, q=0.10) |

| Index | var | q | op | mode |
|---|---|---|---|---|
| `tg90`/`tg90abs` | `tas` | 0.90 | above | pct/abs |
| `tg10`/`tg10abs` | `tas` | 0.10 | below | pct/abs |
| `tn90`/`tn90abs` | `tasmin` | 0.90 | above | pct/abs |
| `tn10`/`tn10abs` | `tasmin` | 0.10 | below | pct/abs |
| `tx90`/`tx90abs` | `tasmax` | 0.90 | above | pct/abs |
| `tx10`/`tx10abs` | `tasmax` | 0.10 | below | pct/abs |
| `wsdi` | `tasmax` | 0.90 | above | spell (min_run=6) |
| `csdi` | `tasmin` | 0.10 | below | spell (min_run=6) |

**Contoh baca**: `tx90` = 15 berarti 15% hari di window itu punya Tmax
lebih tinggi dari threshold ke-90 hari-kalender yang bersangkutan
(dibanding klimatologi `base_period`). `wsdi` = 12 berarti total 12
hari (dari satu atau lebih *warm spell*) di mana Tmax berada di atas
threshold p90 selama ≥6 hari berturut-turut.

---

## 4. Index curah hujan berbasis persentil — `RAIN_PERCENTILE_INDICES`

### 4.1 Tahap threshold (dilakukan sekali, dari `base_period`)

Beda dari suhu — **SATU nilai threshold tetap** per titik spasial,
digabung dari **semua hari basah di semua bulan** `base_period`
(TIDAK per hari-kalender, TIDAK per bulan):

$$
T_q = \text{Quantile}_q\Big(\{x_t : x_t > 1\text{mm},\ t \in \text{base\_period}\}\Big)
$$

### 4.2 Tahap agregasi (per window: tahun/bulan/musim)

$$
\text{R}q\text{P} = \sum_{t \in \text{window}} x_t \cdot \mathbb{1}[x_t > T_q]
$$

$$
\text{R}q\text{PTOT} = \dfrac{\text{R}q\text{P}}{\sum_{t \in \text{window}} x_t \cdot \mathbb{1}[x_t > 1\text{mm}]} \times 100
$$

| Index | $q$ | Rumus |
|---|---|---|
| `R95P` | 0.95 | $\text{R95P} = \sum x_t \cdot \mathbb{1}[x_t > T_{0.95}]$ |
| `R99P` | 0.99 | $\text{R99P} = \sum x_t \cdot \mathbb{1}[x_t > T_{0.99}]$ |
| `R95PTOT` | 0.95 | $\text{R95P} / (\text{total curah hujan hari basah}) \times 100$ |
| `R99PTOT` | 0.99 | $\text{R99P} / (\text{total curah hujan hari basah}) \times 100$ |

**PENTING**: threshold $T_q$ ini **satu angka tetap**, dipakai sama
untuk semua bulan sepanjang tahun — sesuai definisi ETCCDI standar.
Kalau `slice_mode="ME"`/`"JJA"`/dst., threshold-nya **tidak berubah**,
cuma penjumlahan $\sum_{t \in \text{window}}$ yang dipersempit ke
bulan/musim tersebut. Lihat diskusi lengkap soal ini di
[README.md § 6](./README.md#6-rain_percentile_indices--index-curah-hujan-berbasis-persentil).

---

## 5. Fungsi pendukung (`math_utils.py`)

| Fungsi | Rumus / Perilaku |
|---|---|
| `divide(a, b)` | $a/b$, dengan aturan: $0/0 \to \text{NaN}$, $0/(\text{bukan }0) \to 0$, NaN di salah satu $\to$ NaN |
| `longest_run(mask)` | $\max(\text{panjang semua runtun True berturut-turut di } mask)$ |
| `count_days_in_runs(mask, min_run)` | $\displaystyle\sum_{\text{runtun } r,\ \text{len}(r) \geq \text{min\_run}} \text{len}(r)$ |

---

## Referensi

Definisi index di dokumen ini mengikuti konvensi **ETCCDI** (Expert Team
on Climate Change Detection and Indices):
- Zhang, X., et al. (2005). *Avoiding inhomogeneity in percentile-based
  indices of temperature extremes*. Journal of Climate.
- Karl, T.R., Nicholls, N., Ghazi, A. (1999). *CLIVAR/GCOS/WMO workshop
  on indices and indicators for climate extremes*. Climatic Change.
- ClimPACT2 documentation: https://github.com/ARCCSS-extremes/climpact2

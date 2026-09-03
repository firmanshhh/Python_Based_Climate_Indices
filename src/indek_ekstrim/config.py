"""
config.py
=========
Satu-satunya tempat mendefinisikan daftar index curah hujan yang dihitung.

Menambah index baru = tambah satu entri di RAIN_INDICES, tidak perlu ubah
kode di `engine.py` atau `pipeline.py`.

Struktur tiap entri:
    "NAMA_OUTPUT": IndexSpec(func="nama_fungsi_di_ettcdi", params={...})

- `func`  : key yang ada di `ettcdi.FUNC_MAP` (lihat ettcdi.py).
- `params`: kwargs yang dioper ke fungsi tsb (mis. threshold, windows, q).
- `kind`  : "simple" untuk fungsi 1-input (data saja), "quantile" untuk
            fungsi 2-tahap (butuh quantile dulu: RqP/RqPtot).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class IndexSpec:
    func: str
    params: Dict[str, Any] = field(default_factory=dict)
    kind: Literal["simple", "quantile", "dual"] = "simple"
    # Nama variabel sumber di dataset (mis. 'tasmax', 'tasmin', 'tas').
    # None -> pakai `varname` default yang dioper ke pipeline (dipakai oleh
    # RAIN_INDICES, yang semuanya berasal dari satu variabel 'pr').
    # Dipakai oleh TEMP_INDICES karena tiap indeks suhu bisa berasal dari
    # variabel yang berbeda-beda.
    var: Optional[str] = None
    # Variabel KEDUA, khusus untuk kind="dual" (mis. DTR, ETR yang butuh
    # tasmax & tasmin dalam window yang sama).
    var2: Optional[str] = None


RAIN_INDICES: Dict[str, IndexSpec] = {
    "RX1DAY":  IndexSpec("RxNDay", {"windows": 1}),
    "RX3DAY":  IndexSpec("RxNDay", {"windows": 3}),
    "RX5DAY":  IndexSpec("RxNDay", {"windows": 5}),
    "RX7DAY":  IndexSpec("RxNDay", {"windows": 7}),
    "RX10DAY": IndexSpec("RxNDay", {"windows": 10}),

    "HH":      IndexSpec("HHnMM", {"threshold": 1}),
    "HH20MM":  IndexSpec("HHnMM", {"threshold": 20}),
    "HH50MM":  IndexSpec("HHnMM", {"threshold": 50}),
    "HH100MM": IndexSpec("HHnMM", {"threshold": 100}),
    "HH150MM": IndexSpec("HHnMM", {"threshold": 150}),

    "FH20MM":  IndexSpec("FHnMM", {"threshold": 20}),
    "FH50MM":  IndexSpec("FHnMM", {"threshold": 50}),
    "FH100MM": IndexSpec("FHnMM", {"threshold": 100}),
    "FH150MM": IndexSpec("FHnMM", {"threshold": 150}),

    "PRCPTOT": IndexSpec("prcptot"),
    "CDD":     IndexSpec("cdd"),
    "CWD":     IndexSpec("cwd"),
    "SDII":    IndexSpec("sdii"),
}

# ---------------------------------------------------------------------------
# Temperature indices (ETCCDI)
# ---------------------------------------------------------------------------
# Catatan:
# - Indeks di bawah ini semuanya "single-variable, single-pass" -- setiap
#   fungsi hanya butuh satu window waktu dari SATU variabel input, jadi
#   cocok dipakai lewat `engine._apply_simple` (apply_ufunc biasa) tanpa
#   perubahan struktural.
# - Indeks berbasis persentil (TN10p, TX90p, WSDI, CSDI, dst.) SENGAJA
#   belum dimasukkan: perlu baseline periode referensi (mis. 1961-1990)
#   + smoothing window 5-hari yang dihitung terpisah SEBELUM apply_ufunc
#   per-tahun. Ini beda arsitektur, jangan dipaksakan ke pola IndexSpec
#   sederhana di atas -- lihat catatan di pipeline.py.
# - DTR dan ETR (butuh tasmax & tasmin SEKALIGUS dalam satu window) juga
#   belum dimasukkan karena engine.py saat ini hanya mengoper satu
#   DataArray (`sliced_data`) ke apply_ufunc. Perlu varian
#   `_apply_simple` baru yang menerima >1 variabel -- lihat TODO di
#   pipeline.py/engine.py kalau mau ditambahkan.

TEMP_INDICES: Dict[str, IndexSpec] = {
    # Ekstrem suhu maksimum/minimum harian
    "TXx": IndexSpec("Tx", var="tasmax"),   # Max Tmax
    "TXn": IndexSpec("Tn", var="tasmax"),   # Min Tmax
    "TXm": IndexSpec("Tm", var="tasmax"),   # Mean Tmax
    "TNx": IndexSpec("Tx", var="tasmin"),   # Max Tmin
    "TNn": IndexSpec("Tn", var="tasmin"),   # Min Tmin
    "TNm": IndexSpec("Tm", var="tasmin"),   # Mean Tmin
    "TMx": IndexSpec("Tx", var="tas"),      # Max Tas
    "TMn": IndexSpec("Tn", var="tas"),      # Min Tas
    "TMm": IndexSpec("Tm", var="tas"),      # Mean Tas

    # Hari-hitung berbasis threshold tetap
    "FD": IndexSpec("fd", var="tasmin"),    # Frost Days: Tmin < 0°C
    "ID": IndexSpec("id", var="tasmax"),    # Icing Days: Tmax < 0°C
    "SU": IndexSpec("su", var="tasmax"),    # Summer Days: Tmax > 25°C
    "TR": IndexSpec("tr", var="tasmin"),    # Tropical Nights: Tmin > 20°C

    # Dua-variabel (tasmax & tasmin dalam window yang sama)
    "DTR": IndexSpec("dtr", kind="dual", var="tasmax", var2="tasmin"),  # rata-rata (Tmax-Tmin)
    "ETR": IndexSpec("etr", kind="dual", var="tasmax", var2="tasmin"),  # max(Tmax) - min(Tmin)
}

@dataclass(frozen=True)
class TempPercentileIndexSpec:
    var: str                              # 'tasmax' / 'tasmin' / 'tas'
    q: float                              # 0.90 atau 0.10
    op: Literal["above", "below"]         # bandingkan '>' atau '<' threshold
    mode: Literal["pct", "abs", "spell"] = "pct"
    # 'pct'   -> persentase hari exceed relatif thd total hari per periode
    # 'abs'   -> jumlah hari exceed (angka absolut, bukan persen)
    # 'spell' -> WSDI/CSDI: total hari dalam runtun >= min_run hari berturut2
    min_run: int = 6                      # dipakai hanya kalau mode='spell'
    
TEMP_PERCENTILE_INDICES: Dict[str, TempPercentileIndexSpec] = {
    # --- Persentase hari (relatif terhadap total hari per tahun/musim) ---
    "tg90": TempPercentileIndexSpec("tas",    0.90, "above", "pct"),
    "tg10": TempPercentileIndexSpec("tas",    0.10, "below", "pct"),
    "tn90": TempPercentileIndexSpec("tasmin", 0.90, "above", "pct"),
    "tn10": TempPercentileIndexSpec("tasmin", 0.10, "below", "pct"),
    "tx90": TempPercentileIndexSpec("tasmax", 0.90, "above", "pct"),
    "tx10": TempPercentileIndexSpec("tasmax", 0.10, "below", "pct"),
 
    # --- Jumlah hari absolut (bukan persen) ---
    "tg90abs": TempPercentileIndexSpec("tas",    0.90, "above", "abs"),
    "tg10abs": TempPercentileIndexSpec("tas",    0.10, "below", "abs"),
    "tn90abs": TempPercentileIndexSpec("tasmin", 0.90, "above", "abs"),
    "tn10abs": TempPercentileIndexSpec("tasmin", 0.10, "below", "abs"),
    "tx90abs": TempPercentileIndexSpec("tasmax", 0.90, "above", "abs"),
    "tx10abs": TempPercentileIndexSpec("tasmax", 0.10, "below", "abs"),
 
    # --- Spell duration ---
    "wsdi": TempPercentileIndexSpec("tasmax", 0.90, "above", "spell", min_run=6),  # Warm Spell
    "csdi": TempPercentileIndexSpec("tasmin", 0.10, "below", "spell", min_run=6),  # Cold Spell (bonus, pasangan standar WSDI)
}

@dataclass(frozen=True)
class RainPercentileIndexSpec:
    q: float                          # 0.95 atau 0.99
    mode: Literal["sum", "pct"]       # 'sum' -> R95P/R99P (total mm), 'pct' -> R95PTOT/R99PTOT (%)
 
RAIN_PERCENTILE_INDICES: Dict[str, RainPercentileIndexSpec] = {
    "R95P":    RainPercentileIndexSpec(0.95, "sum"),
    "R99P":    RainPercentileIndexSpec(0.99, "sum"),
    "R95PTOT": RainPercentileIndexSpec(0.95, "pct"),
    "R99PTOT": RainPercentileIndexSpec(0.99, "pct"),
}
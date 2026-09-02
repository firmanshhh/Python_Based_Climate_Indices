import sys
from pathlib import Path
import xarray as xr
import glob
import os


# Tambahkan folder src/ (satu tingkat di atas notebooks/) ke sys.path
SRC_PATH = "."
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
print("src path terdaftar:", SRC_PATH)
from indek_ekstrim import rain_indices
from indek_ekstrim.config import RAIN_INDICES

model          = 'MPI-ESM1-2-HR'
latmin, latmax = -15, 10
lonmin, lonmax = 90, 145

working_dir    = "/home/firmansyah-02/PROJECT/05.BCSD_Indonesia"
datadir        = Path(f"{working_dir}/data")
globaldt       = datadir / "Indonesia" / "raw" / "obs" / "pr"
outpath        = datadir / "Indonesia" / "indeks" / 'obs' /'pr'
os.makedirs(outpath, exist_ok=True)
all_files      = glob.glob(os.path.join(globaldt, "*.nc"))
ds = xr.open_mfdataset(all_files, chunks={"time": "auto"})
ds = ds.convert_calendar("standard", use_cftime=True, align_on="date")
ds = ds.rename({'latitude':'lat','longitude':'lon', 'precip':'pr'})
for slice_mode in ['ANN', 'DJF', 'MAM', 'JJA', 'SON']: 
    result = rain_indices(
        ds,
        varname="pr",
        slice_mode=slice_mode,   # bisa 'year', 'monthly', 'DJF', 'MAM', 'JJA', 'SON'
        verbose=True,
    )
    out_name  = f"Indeks_CH_ChirpsV3-RNL_{slice_mode}.nc"
    full_path = os.path.join(outpath, out_name)
    result.to_netcdf(full_path, mode='w', format='NETCDF4', engine='netcdf4')


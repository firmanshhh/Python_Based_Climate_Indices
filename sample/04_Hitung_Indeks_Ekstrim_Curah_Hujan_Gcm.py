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
datadir   = Path(f"{working_dir}/data")
globaldt  = datadir / "Indonesia" / "corrected" / 'pr'
outpath   = datadir / "Indonesia" / "indeks" / 'pr'
os.makedirs(outpath, exist_ok=True)

for period in ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']:
    for slice_mode in ['ANN', 'DJF', 'MAM', 'JJA', 'SON']:
        ffiles = os.path.join(f'{globaldt}/pr_day_{model}_{period}_corrected.nc')
        dataset = xr.open_dataset(ffiles, chunks={"time": "auto"})
        dataset = dataset.rename_vars({'scen':'pr'})   
        result = rain_indices(
            dataset,
            varname="pr",
            slice_mode=slice_mode,   # bisa 'year', 'monthly', 'DJF', 'MAM', 'JJA', 'SON'
            verbose=True,
        )
        out_name = os.path.basename(ffiles).replace(".nc","Indek_ch.nc")
        full_path = os.path.join(outpath, out_name)
        result.to_netcdf(full_path, mode='w', format='NETCDF4', engine='netcdf4')


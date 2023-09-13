# reference: https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
import xarray as xr
import numpy as np
from scipy.interpolate import griddata

def interp_nan():
    ds = xr.open_dataset('./saved_aod_2023.nc')
    aod = ds['aod'].values # (time, y, x)
    # interpolate nan values
    ntime, ny, nx = aod.shape

    x = np.arange(nx)
    y = np.arange(ny)

    xx, yy = np.meshgrid(x, y)

    for i in range(ntime):
        print("processing time step: ", i)
        aod_i = aod[i,:,:]
        # mask invalid values
        array = np.ma.masked_invalid(aod_i)
        xx1 = xx[~array.mask]
        yy1 = yy[~array.mask]
        newarr = array[~array.mask]

        GD1 = griddata((xx1, yy1), newarr.ravel(),
                       (xx, yy),
                       method='nearest')
        aod[i,:,:] = GD1
    return aod


if __name__ == "__main__":
    aod = interp_nan()
    ds = xr.open_dataset('./saved_aod_2023.nc')
    ds['aod'] = (('time', 'y', 'x'), aod)
    ds.to_netcdf('./saved_aod_2023_interp.nc')

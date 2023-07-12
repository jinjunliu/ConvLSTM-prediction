import xarray as xr
from ConvLSTM_pytorch.convlstm import ConvLSTMCell
import numpy as np


def read_data(data_path, var_name):
    """funtion to read data from netcdf file

    Args:
        data_path (str): path to netcdf file
        var_name (str): name of variable to read
    Returns:
        np.ndarray: data
    """
    data = xr.open_dataset(data_path)
    data = data[var_name].values
    return data



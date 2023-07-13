# ConvLSTM-prediction

This is a simple example of how to use ConvLSTM to predict the next month of sea surface temperature (SST) based on a sequence of historical SSTs.

## Dataset

Downlowd the dataset from [here](https://downloads.psl.noaa.gov/Datasets/COBE/sst.mon.mean.nc) and put it to the `data` directory.

## Environment

- Python 3.10
- Pytorch 1.13.1
- Numpy 1.23.5
- Matplotlib 3.6.2
- xarray 2023.6.0
- netcdf4 1.6.2
- h5netcdf 1.2.0

See more details in `environment_droplet.yml`. You can create a new environment by running `conda env create -f environment_droplet.yml`.

## Usage

Run `python main.py` to train the model and predict the next month of SST.

# ConvLSTM-prediction

This is a simple example of how to use ConvLSTM to predict the next month of sea surface temperature (SST) based on a sequence of historical SSTs.

This repo is modified from https://github.com/czifan/ConvLSTM.pytorch. The [submodule](https://github.com/ndrplz/ConvLSTM_pytorch/tree/9f662ba24b0a38e82cf0f5208c7f4859deb85ffe) is currently not used.

## Dataset

Downlowd the dataset from https://downloads.psl.noaa.gov/Datasets/COBE/sst.mon.mean.nc and put it to the `datas` directory.

## Environment

Some necessary packages:

- Python 3.10
- Pytorch 1.13.1
- Numpy 1.23.5
- Matplotlib 3.6.2
- xarray 2023.6.0
- netcdf4 1.6.2
- h5netcdf 1.2.0

See more details in `environment_droplet.yml`. You can create a new environment by running `conda env create -f environment_droplet.yml`.

## Usage

Run `python main.py --config 3x3_16_3x3_32_3x3_64_nc` to train the model and predict the next month of SST.

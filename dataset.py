import xarray as xr

def load_data():
    data_path = "./data"
    file_name = "sst.mon.mean.nc"
    var_name = "sst"
    data = xr.open_dataset(data_path + "/" + file_name)
    return data[var_name]


def split_data(data):
    # seperate data into train and test
    train_data = data.sel(time=slice("1980-01-01",
                                        "2010-12-31")).values
    test_data = data.sel(time=slice("2011-01-01",
                                        "2014-12-31")).values
    return train_data, test_data


if __name__ == "__main__":
    data = load_data()
    print(data.shape)

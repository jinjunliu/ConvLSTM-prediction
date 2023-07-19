import xarray as xr

def load_data():
    data_path = "../datas"
    file_name = "sst.mon.mean.nc"
    var_name = "sst"
    data = xr.open_dataset(data_path + "/" + file_name)
    data = data[var_name]
    # reverse the latitude dimension to make it increasing
    data = data.reindex(lat=list(reversed(data.lat)))
    # select the region of interest, this will generate a dataset with shape (1590, 64, 64)
    data = data.sel(lon=slice(296, 360), lat=slice(0, 64))
    return data


def split_data(data):
    # seperate data into train and test
    train_data = data.sel(time=slice("1980-01-01",
                                        "2010-12-31")).values
    valid_data = data.sel(time=slice("2006-01-01",
                                        "2010-12-31")).values
    test_data = data.sel(time=slice("2011-01-01",
                                        "2015-12-31")).values
    return train_data, valid_data, test_data


if __name__ == "__main__":
    data = load_data()
    print(data.shape)
    print(data.time.values[0])
    print(data.time.values[-1])

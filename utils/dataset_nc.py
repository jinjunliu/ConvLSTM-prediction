import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import torch


def load_nc_data():
    """load nc data and select the region of interest
    
    Returns:
        data: a xr.dataarray of shape (1590, 64, 64)
    """
    data_path = "./datas"
    file_name = "sst.mon.mean.nc"
    var_name = "sst"
    with xr.open_dataset(data_path + "/" + file_name) as data:
        data = data[var_name]
    # reverse the latitude dimension to make it increasing
    data = data.reindex(lat=list(reversed(data.lat)))
    # select the region of interest, this will generate a dataset with shape (1590, 64, 64)
    data = data.sel(lon=slice(296, 360), lat=slice(0, 64))
    return data


def split_data(data, split='train'):
    """split data into training, validation and testing sets

    Args:
        data: a xr.dataarray of shape (1590, 64, 64)
        split: a string, either 'train', 'valid' or 'test'

    Returns:
        One of the following:
        train_data: a numpy array of shape (50*12, 64, 64) = (600, 64, 64) # 50 years
        valid_data: a numpy array of shape (10*12, 64, 64) = (120, 64, 64) # 10 years
        test_data: a numpy array of shape (5*12, 64, 64) = (60, 64, 64) # 5 years
    """
    train_data = data.sel(time=slice("1950-01-01",
                                        "1999-12-31")).values
    valid_data = data.sel(time=slice("2000-01-01",
                                        "2010-12-31")).values
    test_data = data.sel(time=slice("2011-01-01",
                                        "2015-12-31")).values
    if split == 'train':
        return train_data
    elif split == 'valid':
        return valid_data
    elif split == 'test':
        return test_data
    else:
        raise ValueError('Invalid split name: {}'.format(split))


class NcDataset(Dataset):
    def __init__(self, config, split='train'):
        super().__init__()
        self.datas = load_nc_data()
        self.input_size = config.input_size
        self.num_frames_input = config.num_frames_input
        self.num_frames_output = config.num_frames_output
        self.num_frames = config.num_frames_input + config.num_frames_output
        self.datas = split_data(self.datas, split)
        print('Loaded {} samples ({})'.format(self.__len__(), split))

    def __getitem__(self, index):
        data = self.datas[index:index+self.num_frames]
        inputs = data[:self.num_frames_input]
        targets = data[self.num_frames_input:]
        inputs = inputs[..., np.newaxis]
        targets = targets[..., np.newaxis]
        # replace nan values with 0
        inputs = np.nan_to_num(inputs)
        targets = np.nan_to_num(targets)
        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float().contiguous()
        targets = torch.from_numpy(targets).permute(0, 3, 1, 2).float().contiguous()
        return inputs, targets
    

    def __len__(self):
        return self.datas.shape[0] - self.num_frames + 1


if __name__ == "__main__":
    data = load_nc_data()
    print(data.shape)
    print(data.time.values[0])
    print(data.time.values[-1])

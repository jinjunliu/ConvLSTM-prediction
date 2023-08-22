# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from utils.dataset import MovingMNISTDataset
from utils.dataset_nc import NcDataset
from networks.ConvLSTM import ConvLSTM
import torch
from torch.utils.data import DataLoader
from utils.utils import save_checkpoint
from utils.utils import build_logging
from utils.functions import train
from utils.functions import valid
from utils.functions import test
#from networks.CrossEntropyLoss import CrossEntropyLoss
from networks.BinaryDiceLoss import BinaryDiceLoss
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='3x3_16_3x3_32_3x3_64_nc')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    name = args.config
    if name == '3x3_16_3x3_32_3x3_64': from configs.config_3x3_16_3x3_32_3x3_64 import config
    elif name == '3x3_32_3x3_64_3x3_128': from configs.config_3x3_32_3x3_64_3x3_128 import config
    elif name == '3x3_16_3x3_32_3x3_64_nc': from configs.config_3x3_16_3x3_32_3x3_64_nc import config
    else: raise ValueError('Invalid config name: {}'.format(name))
    logger = build_logging(config)
    model = ConvLSTM(config).to(config.device)
    #criterion = CrossEntropyLoss().to(config.device)
    criterion = torch.nn.MSELoss().to(config.device) # important: use MSE loss because the output is a float number
    # criterion = BinaryDiceLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_dataset = NcDataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                            num_workers=config.num_workers, shuffle=True, pin_memory=True)
    valid_dataset = NcDataset(config, split='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                            num_workers=config.num_workers, shuffle=False, pin_memory=True)
    test_dataset = NcDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size,
                            num_workers=config.num_workers, shuffle=False, pin_memory=True)
    # print train loader size
    for i, (inputs, targets) in enumerate(train_loader):
        print(inputs.shape, targets.shape)
        break
    train_records, valid_records, test_records = [], [], []
    for epoch in range(config.epochs):
        epoch_records = train(config, logger, epoch, model, train_loader, criterion, optimizer)
        train_records.append(np.mean(epoch_records['loss']))
        epoch_records = valid(config, logger, epoch, model, valid_loader, criterion)
        valid_records.append(np.mean(epoch_records['loss']))
        epoch_records = test(config, logger, epoch, model, test_loader, criterion)
        test_records.append(np.mean(epoch_records['loss']))
        plt.plot(range(epoch + 1), train_records, label='train')
        plt.plot(range(epoch + 1), valid_records, label='valid')
        plt.plot(range(epoch + 1), test_records, label='test')
        plt.legend()
        plt.savefig(os.path.join(config.output_dir, '{}.png'.format(name)))
        plt.close()
        # save model, first check if the model is the best
        is_best = False
        if not valid_records or valid_records[-1] < min(valid_records):
            is_best = True
        save_checkpoint(model, is_best, config.model_dir)


if __name__ == '__main__':
    main()

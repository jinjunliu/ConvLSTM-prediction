# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def train(config, logger, epoch, model, train_loader, criterion, optimizer):
    model.train()
    epoch_records = {'loss': []}
    num_batchs = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print(inputs.max())
        inputs = inputs.float().to(config.device)
        targets = targets.float().to(config.device)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_records['loss'].append(losses.item())
        if batch_idx and batch_idx % config.display == 0:
            logger.info('EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
    return epoch_records

def valid(config, logger, epoch, model, valid_loader, criterion):
    model.eval()
    epoch_records = {'loss': []}
    num_batchs = len(valid_loader)
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        with torch.no_grad():
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            epoch_records['loss'].append(losses.item())
            if batch_idx and batch_idx % config.display == 0:
                logger.info('[V] EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                    epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
    return epoch_records

def test(config, logger, epoch, model, test_loader, criterion):
    model.eval()
    epoch_records = {'loss': []}
    num_batchs = len(test_loader)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            inputs = inputs.float().to(config.device)
            targets = targets.float().to(config.device)
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            epoch_records['loss'].append(losses.item())
            if batch_idx and batch_idx % config.display == 0:
                logger.info('[T] EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batchs,
                                                                                    epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
            # if batch_idx and batch_idx % config.draw == 0:
            # if True: # always draw
            if epoch > 10:
                batch_size = inputs.shape[0]
                for i in range(batch_size):
                    _, axarr = plt.subplots(3, targets.shape[1],
                                            figsize=(targets.shape[1] * 5, 15))
                    for t in range(targets.shape[1]):
                        # flip y axis
                        vmin, vmax = 0, 0.2
                        axarr[0][t].imshow(np.flip(inputs[i, t, 0].detach().cpu().numpy(), axis=0), cmap='jet', vmin=vmin, vmax=vmax)
                        axarr[1][t].imshow(np.flip(targets[i, t, 0].detach().cpu().numpy(), axis=0), cmap='jet', vmin=vmin, vmax=vmax)
                        axarr[2][t].imshow(np.flip(outputs[i, t, 0].detach().cpu().numpy(), axis=0), cmap='jet', vmin=vmin, vmax=vmax)
                    plt.savefig(os.path.join(config.cache_dir, '{:03d}_{:05d}_{:05d}.png'.format(epoch, batch_idx, i)))
                    plt.close()
    return epoch_records

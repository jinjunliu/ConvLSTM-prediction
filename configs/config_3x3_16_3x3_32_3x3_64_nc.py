# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
root_dir = os.path.join(os.getcwd(), '.')
print(root_dir)

class Config:
    gpus = [0, ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        num_workers = 8 * len(gpus)
        # train_batch_size = 64
        # valid_batch_size = 2 * train_batch_size
        # test_batch_size = 2 * train_batch_size
        train_batch_size = 10
        valid_batch_size = 2*train_batch_size
        test_batch_size = 2*train_batch_size
    else:
        num_workers = 0
        train_batch_size = 2
        valid_batch_size = 2 * train_batch_size
        test_batch_size = 2 * train_batch_size
    # data_file = 'datas/sst.mon.mean.nc'
    # var_name = 'sst'
    data_file = 'datas/saved_aod_2023_interp_cubic.nc'
    var_name = 'aod'

    num_frames_input = 6
    num_frames_output = 6
    input_size = (64, 64)
    display = 1
    draw = 1
    # train_dataset = (0, 1000)
    # valid_dataset = (1000, 1200)
    # test_dataset = (1200, 1590)
    epochs = 20

    # (type, activation, in_ch, out_ch, kernel_size, padding, stride)
    # encoder = [('conv', 'leaky', 1, 16, 3, 1, 2),
    #          ('convlstm', '', 16, 16, 3, 1, 1),
    #          ('conv', 'leaky', 16, 32, 3, 1, 2),
    #          ('convlstm', '', 32, 32, 3, 1, 1),
    #          ('conv', 'leaky', 32, 64, 3, 1, 2),
    #          ('convlstm', '', 64, 64, 3, 1, 1)]
    # decoder = [('deconv', 'leaky', 64, 32, 4, 1, 2),
    #            ('convlstm', '', 64, 32, 3, 1, 1),
    #            ('deconv', 'leaky', 32, 16, 4, 1, 2),
    #            ('convlstm', '', 32, 16, 3, 1, 1),
    #            ('deconv', 'leaky', 16, 16, 4, 1, 2),
    #            ('convlstm', '', 17, 16, 3, 1, 1),
    #            ('conv', 'sigmoid', 16, 1, 1, 0, 1)]

    encoder = [('conv', 'leaky', 1, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 128, 3, 1, 2),
             ('convlstm', '', 128, 128, 3, 1, 1)]
    decoder = [('deconv', 'leaky', 128, 64, 4, 1, 2),
               ('convlstm', '', 128, 64, 3, 1, 1),
               ('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 32, 4, 1, 2),
               ('convlstm', '', 33, 32, 3, 1, 1),
               ('conv', 'sigmoid', 32, 1, 1, 0, 1)]

    data_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cache_dir = os.path.join(output_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

config = Config()

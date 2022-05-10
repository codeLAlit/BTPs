import datetime
import os
import random
import time
import gc
import sys
import numpy as np

import scipy.spatial.distance as spd

from skimage import io
from skimage import util
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import colors
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F

import list_dataset

from torchsummary import summary

class classifierPlain(nn.Module):

    def __init__(self, in_channels):
        super(classifierPlain, self).__init__()
        self.dimr1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=2, bias=False)
        self.dimr2 = nn.Conv2d(in_channels//2, in_channels//4, kernel_size=2, stride=2, bias=False)        
        self.dimr3 = nn.Conv2d(in_channels//4, 1, kernel_size=1, stride=2, bias=False)

        self.up1 = nn.ConvTranspose2d(1, 4, kernel_size=1, stride=2, bias=False)
        self.up2 = nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2, bias=False)
        self.up3 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=2, bias=False)
        
        self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias =False)
        self.soft = nn.Softmax()
    
    def forward(self, x):
        x = self.dimr1(x)
        x = self.dimr2(x)
        x = self.dimr3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        out = self.conv(x)
        out = self.soft(out)
        return out

model = classifierPlain(324)
summary(model, (1, 324, 224, 224), batch_dim=None)
# cudnn.benchmark = True
# args = {
#     'epoch_num': 1200,            # Number of epochs.
#     'lr': 1e-3,                   # Learning rate.
#     'weight_decay': 5e-6,         # L2 penalty.
#     'momentum': 0.9,              # Momentum.
#     'batch_size': 1,              # Batch size.
#     'num_workers': 8,             # Number of workers on data loader.
#     'print_freq': 1,              # Printing frequency for mini-batch loss.
#     'w_size': 224,                # Width size for image resizing.
#     'h_size': 224,                # Height size for image resizing.
#     'test_freq': 600,            # Test each test_freq epochs.
#     'save_freq': 600,            # Save model each save_freq epochs.
#     'input_channels': 4,          # Number of input channels in samples/DNN.
#     'num_classes': 5,             # Number of original output classes in dataset.
# }

# conv_name = 'fcndensenet121'

# args['hidden_classes'] = '1'
# print('hidden: ' + '1')

# dataset_name = 'Potsdam'

# hidden = [1]

# num_known_classes = args['num_classes'] - len(hidden)
# num_unknown_classes = len(hidden)


# weights = [1.0 for i in range(num_known_classes)]
# if 4 not in hidden:
#     weights[-1] = 2.0

# weights = torch.FloatTensor(weights)


# # Setting experiment name.
# exp_name = 'classifier_'+conv_name + '_' + dataset_name + '_base_dsm_' + args['hidden_classes']

# # Setting device [0|1|2].
# args['device'] = 'cuda'

# val_set = list_dataset.ListDataset(dataset_name, 'Val', (args['h_size'], args['w_size']), 'statistical', hidden, overlap=False, use_dsm=True)
# val_loader = DataLoader(val_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)

# cmap_list = [
#     (1.0, 1.0, 1.0, 1.0), # street. 
#     (0.0, 0.0, 1.0, 1.0), # Building
#     (0.0, 1.0, 1.0, 1.0), # Grass.
#     (0.0, 1.0, 0.0, 1.0), # Tree.
#     (1.0, 1.0, 0.0, 1.0), # Car.
#     (1.0, 0.0, 0.0, 1.0),  # surfaces
#     # (0.0, 0.0, 0.0, 1.0), # Boundaries (if present).
#     ]
# lab_cmap = colors.ListedColormap(cmap_list)
# cmap_list_1 = [
# (1.0, 1.0, 1.0, 1.0), # street.
# (0.0, 1.0, 1.0, 1.0), # Grass.
# (0.0, 1.0, 0.0, 1.0), # Tree.
# (1.0, 1.0, 0.0, 1.0), # Car.
# (1.0, 0.0, 0.0, 1.0),  # surfaces       
# (0.0, 0.0, 1.0, 1.0), # Building  
# # (0.0, 0.0, 0.0, 1.0), # Boundaries (if present).
# ]
# lab_cmap_1 = colors.ListedColormap(cmap_list_1)	   

# np.random.seed(1)
# img, msk, msk_true, spl = val_set[0]
# print(img.shape, msk.shape, msk_true.shape)

# a, b = 1, 1
# plt.figure(figsize=(16, 6))
# plt.subplot(141)
# plt.title("Image")
# plt.imshow(np.transpose((img[a, b, 0:3,:,:] + 0.5), (1, 2, 0)))
# plt.subplot(142)
# plt.title("Mask")
# plt.imshow(msk[a, b, :, :], cmap=lab_cmap_1)
# plt.clim(0, 5)
# plt.subplot(143)
# plt.title("True Mask")
# plt.imshow(msk_true[a, b, :, :], cmap=lab_cmap)
# plt.clim(0, 5)
# plt.subplot(144)
# plt.title("Classifier mask")
# plt.imshow((msk[a, b, :, :]==5)*1.0, cmap='gray')
# plt.clim(0, 1)
# plt.savefig('test.png')

# print(np.unique((msk[a, b, :, :]==5)*1.0))
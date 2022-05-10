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
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d, LovaszLoss, FocalLoss2d
from torchinfo import summary
# from openmax import *

cudnn.benchmark = True
args = {
    'epoch_num': 200,            # Number of epochs.
    'lr': 1e-3,                   # Learning rate.
    'weight_decay': 5e-6,         # L2 penalty.
    'momentum': 0.9,              # Momentum.
    'batch_size': 1,              # Batch size.
    'num_workers': 8,             # Number of workers on data loader.
    'print_freq': 1,              # Printing frequency for mini-batch loss.
    'w_size': 224,                # Width size for image resizing.
    'h_size': 224,                # Height size for image resizing.
    'test_freq': 600,            # Test each test_freq epochs.
    'save_freq': 600,            # Save model each save_freq epochs.
    'input_channels': 4,          # Number of input channels in samples/DNN.
    'num_classes': 6,             # Number of original output classes in dataset.
}

conv_name = sys.argv[1]

args['hidden_classes'] = sys.argv[2]
print('hidden: ' + sys.argv[2])

dataset_name = sys.argv[3]

hidden = []

if '_' in args['hidden_classes']:
    hidden = [int(h) for h in args['hidden_classes'].split('_')]
else:
    hidden = [int(args['hidden_classes'])]

num_known_classes = args['num_classes'] - len(hidden)
num_unknown_classes = len(hidden)


weights = [1.0 for i in range(num_known_classes)]
if 4 not in hidden:
    weights[-1] = 2.0

weights = torch.FloatTensor(weights)


# Setting experiment name.
exp_name = 'classifier_'+conv_name + '_' + dataset_name + '_base_dsm_' + args['hidden_classes']

# Setting device [0|1|2].
args['device'] = 'cuda'


class classifierPlain(nn.Module):

    def __init__(self, in_channels, img_shape):
        super(classifierPlain, self).__init__()
        self.shape = img_shape
        self.dimr = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.dimr(x)
        out = self.soft(out)
        return out


def train(train_loader, net, classifier, c_optimizer, criterion, epoch, num_known_classes, num_unknown_classes, hidden, args, save_images, save_model):
    net.eval()
        
    # Creating output directory.
    check_mkdir(os.path.join(outp_path, exp_name, 'epoch_' + str(epoch)))
    loss_hist = []
    totbatches = len(train_loader)
    # Iterating over batches.

    def train_batch(val):
    	# Obtaining images, labels and paths for batch.
        inps_batch, labs_batch, true_batch, img_name = data
        
        inps_batch = inps_batch.squeeze()
        labs_batch = labs_batch.squeeze()
        true_batch = true_batch.squeeze()
        minibatch_loss = 0
        
        # Iterating over patches inside batch.
        for j in range(inps_batch.size(0)):
            ## training the classifier
            print('    MiniBatch %d/%d' % (j + 1, inps_batch.size(0)))
            sys.stdout.flush()
            
            tic = time.time()
            
            for k in range(inps_batch.size(1)):
                
                inps = inps_batch[j, k].unsqueeze(0)
                labs = labs_batch[j, k].unsqueeze(0)
                true = true_batch[j, k].unsqueeze(0)
                #print(inps_batch[j, k].shape, inps.shape)
                # Casting tensors to cuda.
                inps, labs, true = inps.to(args['device']), labs.to(args['device']), true.to(args['device'])
                
                # Casting to cuda variables.
                inps = Variable(inps).to(args['device'])
                labs = Variable(labs).to(args['device'])
                true = Variable(true).to(args['device'])
                # print(inps.shape)
                # Forwarding.
                with torch.no_grad():
                    if conv_name == 'fcnwideresnet50':
                        outs, classif1, fv2 = net(inps, feat=True)
                    elif conv_name == 'fcndensenet121':
                        outs, classif1, fv2 = net(inps, feat=True)	

                feat_flat = torch.cat([outs, classif1, fv2], 1)        
                feat_flat = feat_flat.to(args['device'])
                classifier_truth = (labs[:, :, :]==5) # UUC number 5

                if not val:
                    classifier.train()
                    c_optimizer.zero_grad()
                    out = classifier(feat_flat)
                    loss = criterion(out[0, :, :, :], classifier_truth.float())
                    # print(loss)
                    loss.backward()
                    loss_data = loss.item()
                    c_optimizer.step()
                else:
                    classifier.eval()
                    with torch.no_grad():
                        out = classifier(feat_flat)
                    loss = criterion(out[0, :, :, :], classifier_truth.float())
                    loss_data = loss.item()

                minibatch_loss += loss_data	
            toc = time.time()
            print('        Elapsed Time: %.2f' % (toc - tic))

        loss_hist.append(minibatch_loss/(inps_batch.size(0)*inps_batch.size(1)))
        print('Batch loss = %.2f' %(loss_hist[-1]))        	


    for i, data in enumerate(train_loader):
        
        if i==len(train_loader)-1:
            print('Classifier validation Batch %d/%d' % (i + 1, len(train_loader)))
            sys.stdout.flush()
            train_batch(val=True)
            if args['best_record']['val_loss'] > loss_hist[-1]:
                args['best_record']['val_loss'] = loss_hist[-1]
                torch.save(classifier.state_dict(), os.path.join(outp_path, exp_name, f'model_classifier_{epoch}_best.pth'))
        else:
            print('Classifier train Batch %d/%d' % (i + 1, len(train_loader)))
            sys.stdout.flush()
            train_batch(val=False)


if __name__=='__main__':

    net_path = 'ckpt/fcndensenet121_Potsdam_base_dsm_1/model_190_best.pth'
    # net_path = 'model_600.pth'
    net = FCNDenseNet121(args['input_channels'], num_classes=args['num_classes'], pretrained=False, skip=True, hidden_classes=hidden).to(args['device'])

    net.load_state_dict(torch.load(net_path, map_location=args['device']))
    net.eval()
    criterion = CrossEntropyLoss2d(weight=weights, size_average=False, ignore_index=args['num_classes']).to(args['device'])
    epoch = 400
    val_set = list_dataset.ListDataset(dataset_name, 'Val', (args['h_size'], args['w_size']), 'statistical', hidden, overlap=False, use_dsm=True)
    val_loader = DataLoader(val_set, batch_size=None, num_workers=args['num_workers'], shuffle=False)
    outp_path = './'
    exp_name = 'classifier_'+conv_name + '_' + dataset_name + '_base_dsm_' + args['hidden_classes']
    check_mkdir(outp_path)
    check_mkdir(os.path.join(outp_path, exp_name))
    args['best_record'] = {'epoch': 0, 'lr': 1e-4, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'iou': 0}

    cmap_list = [
    (1.0, 1.0, 1.0, 1.0), # street. 
    (0.0, 0.0, 1.0, 1.0), # Building
    (0.0, 1.0, 1.0, 1.0), # Grass.
    (0.0, 1.0, 0.0, 1.0), # Tree.
    (1.0, 1.0, 0.0, 1.0), # Car.
    (1.0, 0.0, 0.0, 1.0),  # surfaces
    # (0.0, 0.0, 0.0, 1.0), # Boundaries (if present).
    ]
    lab_cmap = colors.ListedColormap(cmap_list)
    cmap_list_1 = [
    (1.0, 1.0, 1.0, 1.0), # street.
    (0.0, 1.0, 1.0, 1.0), # Grass.
    (0.0, 1.0, 0.0, 1.0), # Tree.
    (1.0, 1.0, 0.0, 1.0), # Car.
    (1.0, 0.0, 0.0, 1.0),  # surfaces       
    (0.0, 0.0, 1.0, 1.0), # Building  
    # (0.0, 0.0, 0.0, 1.0), # Boundaries (if present).
    ]
    lab_cmap_1 = colors.ListedColormap(cmap_list_1)	      

    lr_c = 1e-3  
    wd_c = 5e-6
    moment_c =0.9
    epoch_c = 400
    in_channels = 325
    classifier = classifierPlain(in_channels, (args['h_size'], args['w_size'])).to(args['device'])
    # summary(classifier, input_size = (4, 324, 224, 224))
    criterion = nn.BCELoss()
    c_optimizer = torch.optim.Adam([param for name, param in classifier.named_parameters() if name[-4:] == 'bias'], lr=lr_c, weight_decay=wd_c, betas=(moment_c, 0.99))
    c_scheduler = torch.optim.lr_scheduler.StepLR(c_optimizer, epoch_c//3, 0.2)
    curr_epoch = 1
    # exit()
    for ec in range(curr_epoch, epoch_c+1):
        print('Epoch %d/%d' %(ec, epoch_c))
        train(val_loader, net, classifier, c_optimizer, criterion, ec, num_known_classes, num_unknown_classes, hidden, args, save_images=False, save_model=False)
        c_scheduler.step()

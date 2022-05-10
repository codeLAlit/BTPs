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
from skimage import color
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

# from openmax import *

cudnn.benchmark = True
args = {
    'epoch_num': 1200,            # Number of epochs.
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
    'num_classes': 5,             # Number of original output classes in dataset.
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
exp_name = conv_name + '_' + dataset_name + '_base_dsm_' + args['hidden_classes']

# Setting device [0|1|2].
args['device'] = 'cuda'



def testmodel(test_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, args, save_images, save_model):
    
    # Setting network for evaluation mode.
    net.eval()
    
    with torch.no_grad():
        
        # Creating output directory.
        check_mkdir(os.path.join(outp_path, exp_name, 'epoch_' + str(epoch)))
        loss = []
        totbatches = len(test_loader)
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            
            print('Test Batch %d/%d' % (i + 1, len(test_loader)))
            sys.stdout.flush()
            
            # Obtaining images, labels and paths for batch.
            inps_batch, labs_batch, true_batch, img_name = data
            
            inps_batch = inps_batch.squeeze()
            labs_batch = labs_batch.squeeze()
            true_batch = true_batch.squeeze()
            
            minibatch_loss = 0
            # Iterating over patches inside batch.
            for j in range(inps_batch.size(0)):
                
                print('    Test MiniBatch %d/%d' % (j + 1, inps_batch.size(0)))
                sys.stdout.flush()
                
                tic = time.time()
                
                for k in range(inps_batch.size(1)):
                    
                    inps = inps_batch[j, k].unsqueeze(0)
                    labs = labs_batch[j, k].unsqueeze(0)
                    true = true_batch[j, k].unsqueeze(0)
                    
                    # Casting tensors to cuda.
                    inps, labs, true = inps.to(args['device']), labs.to(args['device']), true.to(args['device'])
                    
                    # Casting to cuda variables.
                    inps = Variable(inps).to(args['device'])
                    labs = Variable(labs).to(args['device'])
                    true = Variable(true).to(args['device'])
                    
                    # Forwarding.
                    if conv_name == 'fcnwideresnet50':
                        outs, classif1, fv2 = net(inps, feat=True)
                    elif conv_name == 'fcndensenet121':
                        outs, classif1, fv2 = net(inps, feat=True)
                    
                    # Computing probabilities.
                    soft_outs = F.softmax(outs, dim=1)
                    
                    # Obtaining prior predictions.
                    prds = soft_outs.data.max(1)[1]
                    inps_np = inps.detach().squeeze(0).cpu().numpy()
                    labs_np = labs.detach().squeeze(0).cpu().numpy()
                    true_np = true.detach().squeeze(0).cpu().numpy()

                    if(save_images):
                        pred_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_prd_' + str(j) + '_' + str(k) + '.png'))
                            
                        plt.figure(figsize=(16, 6))
                        plt.subplot(141)
                        plt.title("Image")
                        plt.imshow(np.transpose((inps_np[0:3,:,:] + 0.5), (1, 2, 0)))
                        plt.subplot(142)
                        plt.title("Mask")
                        plt.imshow(labs_np, cmap=lab_cmap_1)
                        plt.clim(0, 5)
                        plt.subplot(143)
                        plt.title("True Mask")
                        plt.imshow(true_np, cmap=lab_cmap)
                        plt.clim(0, 5)
                        plt.subplot(144)
                        plt.title("Prediction")
                        plt.imshow(prds.cpu().squeeze().numpy(), cmap=lab_cmap_1)
                        plt.clim(0, 5)
                        plt.savefig(pred_path)
                        plt.clf()
                        plt.close()
                        
                    minibatch_loss += criterion(outs, prds).item()

                toc = time.time()
                print('        Elapsed Time: %.2f' % (toc - tic))

            loss.append(minibatch_loss/(inps_batch.size(0)*inps_batch.size(1)))
            print('Batch loss = %.2f' %(loss[-1]))

        sys.stdout.flush()
        batch_loss = np.asarray(loss).mean()
        print('Test loss: %.2f' %(batch_loss))
        if batch_loss < args['best_record']['val_loss']:
            args['best_record']['val_loss'] = batch_loss          
            args['best_record']['epoch'] = epoch
        
            if save_model:
                
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'model_' + str(epoch) + '_best.pth'))
                torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + str(epoch) + '_best.pth'))

if __name__=='__main__':
    net_add = 'ckpt/fcndensenet121_Potsdam_base_dsm_1/model_470_best.pth'
    net = FCNDenseNet121(args['input_channels'], num_classes=args['num_classes'], pretrained=False, skip=True, hidden_classes=hidden).to(args['device'])


    val_set = list_dataset.ListDataset(dataset_name, 'Val', (args['h_size'], args['w_size']), 'statistical', hidden, overlap=False, use_dsm=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)

    # img, msk, msk_true, spl = val_set[0]
    # print((msk==5).shape)
    
    net.load_state_dict(torch.load(net_add, map_location=args['device']))
    criterion = CrossEntropyLoss2d(weight=weights, size_average=False, ignore_index=args['num_classes']).to(args['device'])
    epoch = 470

    outp_path = './outtest'
    exp_name = conv_name + '_' + dataset_name + '_base_dsm_' + args['hidden_classes']
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

    testmodel(val_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, args, save_images=False, save_model=False)

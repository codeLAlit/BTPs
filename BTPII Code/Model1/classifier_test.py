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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.FloatTensor

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

class Plotter():
    def __init__(self):
        self.soft = nn.Softmax(dim=0)
        from matplotlib import colors
        '''
        Vaihingen/Potsdam classes:
            0 = Street
            1 = Building
            2 = Grass
            3 = Tree
            4 = Car
            5 = Surfaces
            6 = Unknown
        '''
        # no hidden class cmap
        cmap_list = [
            # (1.0, 1.0, 1.0, 1.0), # street. 
            # (0.0, 0.0, 1.0, 1.0), # Building
            # (0.0, 1.0, 1.0, 1.0), # Grass.
            # (0.0, 1.0, 0.0, 1.0), # Tree.
            # (1.0, 1.0, 0.0, 1.0), # Car.
            # (1.0, 0.0, 0.0, 1.0),  # surfaces
            # (1.0, 0.0, 1.0, 1.0), # Unknown.
            (1.0, 1.0, 1.0, 1.0), # street.
            (0.0, 1.0, 1.0, 1.0), # Grass.
    	    (0.0, 1.0, 0.0, 1.0), # Tree.
    	    (1.0, 1.0, 0.0, 1.0), # Car.
    	    (1.0, 0.0, 0.0, 1.0),  # surfaces       
    	    (0.0, 0.0, 1.0, 1.0), # Building 
        ]
        self.lab_cmap = colors.ListedColormap(cmap_list)

    def plot(self, img, gt, pred, save, name=None):
        
        img = img.detach().permute((1, 2, 0)).cpu().numpy()
        gt = gt.detach().cpu().numpy()
        pred_class = self.soft(pred)
        pred = pred.detach().permute((1, 2, 0)).cpu().numpy()
        pred_class = pred_class.detach().permute((1, 2, 0)).cpu().numpy()
        # htmp = (pred[:, :, -1]>=0.5)
        # print(np.unique(htmp))
        pred_class = np.argmax(pred_class, axis=2)
        immax = np.max(img.reshape(-1))
        immin = np.min(img.reshape(-1))
        #print(immax, immin)
        plt.figure(figsize=(20,8))
        plt.subplot(131)
        plt.title("Image")
        #plt.imshow((img-immin)/(immax-immin))
        plt.imshow((img[:, :, :3]+1)/2)
        plt.subplot(132)
        plt.title("Ground Truth")
        plt.imshow(gt, cmap=self.lab_cmap)
        plt.clim(0, 5)
        plt.subplot(133)
        plt.title("Prediction")
        plt.imshow(pred_class, cmap=self.lab_cmap)
        plt.clim(0, 5)
        if save:
            plt.savefig(name)
        else:
            plt.show()
        plt.clf()
        plt.close()

def test(test_loader, net, classifier, c_optimizer, criterion, epoch, num_known_classes, num_unknown_classes, hidden, args, save_images, save_model):
    net.eval()
    plotter = Plotter()
    # Creating output directory.
    # check_mkdir(os.path.join(outp_path, exp_name, 'epoch_' + str(epoch)))
    loss_hist = []
    totbatches = len(test_loader)
    # Iterating over batches.

    def test_batch(val):
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
            kprint = np.random.randint(inps_batch.size(1))
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
                
                #print(np.unique(labs.detach().cpu().numpy()))
                # print(inps.shape)
                # Forwarding.
                with torch.no_grad():
                    if conv_name == 'fcnwideresnet50':
                        outs, classif1, fv2 = net(inps, feat=True)
                    elif conv_name == 'fcndensenet121':
                        outs, classif1, fv2 = net(inps, feat=True)	
                #print(outs.shape)
                feat_flat = torch.cat([outs, classif1, fv2], 1)        
                feat_flat = feat_flat.to(args['device'])
                classifier_truth = (labs[:, :, :]==5) # UUC number 5

                with torch.no_grad():
                    out = classifier(feat_flat)
                
                
                combined_pred = torch.concat([outs, out], axis=1)
                #print(inps.squeeze(0).shape, combined_pred.squeeze(0).shape, labs.squeeze(0).shape)
                
                if k==kprint:
                    plotter.plot(inps.squeeze(0), labs.squeeze(0), combined_pred.squeeze(0), True, '../preds_classifier/pred_{}_{}.png'.format(i, j))
                loss = criterion(combined_pred, labs)
                loss_data = loss.item()
                # print(loss_data)
                minibatch_loss += loss_data	
            toc = time.time()
            print('        Elapsed Time: %.2f' % (toc - tic))

        loss_hist.append(minibatch_loss/(inps_batch.size(0)*inps_batch.size(1)))
        print('Batch loss = %.2f' %(loss_hist[-1]))        	
        pass_loss = np.mean(loss_hist)
        return pass_loss

    pass_loss = 0
    for i, data in enumerate(test_loader):
        print('Classifier testing Batch %d/%d' % (i + 1, len(test_loader)))
        sys.stdout.flush()
        pass_loss += test_batch(val=True)
    return pass_loss
        #make preds properly


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.__name__ = 'multi_class_ce'
        self.nll_loss = nn.NLLLoss(weight, ignore_index=ignore_index, reduction='mean')
    
    def forward(self, inputs, targets):
    	#print(inputs.shape)
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        return self.nll_loss(F.log_softmax(inputs, dim = 1).type(Tensor), targets.type(torch.LongTensor))

if __name__=='__main__':

    net_path = 'ckpt/fcndensenet121_Potsdam_base_dsm_1/model_190_best.pth'
    # net_path = 'model_600.pth'
    net = FCNDenseNet121(args['input_channels'], num_classes=args['num_classes'], pretrained=False, skip=True, hidden_classes=hidden).to(args['device'])

    net.load_state_dict(torch.load(net_path, map_location=args['device']))
    net.eval()
    criterion = CrossEntropyLoss2d(weight=weights, size_average=False, ignore_index=args['num_classes']).to(args['device'])
    epoch = 600
    test_set = list_dataset.ListDataset(dataset_name, 'Test', (args['h_size'], args['w_size']), 'statistical', hidden, overlap=False, use_dsm=True)
    test_loader = DataLoader(test_set, batch_size=None, num_workers=args['num_workers'], shuffle=False)
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
    class_path = 'model_classifier_284_best.pth'
    classifier = classifierPlain(in_channels, (args['h_size'], args['w_size'])).to(args['device'])
    classifier.load_state_dict(torch.load(class_path, map_location=args['device']))
    
    # summary(classifier, input_size = (4, 324, 224, 224))
    criterion = CrossEntropyLoss2d()
    c_optimizer = torch.optim.Adam([param for name, param in classifier.named_parameters() if name[-4:] == 'bias'], lr=lr_c, weight_decay=wd_c, betas=(moment_c, 0.99))
    c_scheduler = torch.optim.lr_scheduler.StepLR(c_optimizer, epoch_c//3, 0.2)
    curr_epoch = 1
    # exit()
    all_pass_loss = test(test_loader, net, classifier, c_optimizer, criterion, 0, num_known_classes, num_unknown_classes, hidden, args, save_images=False, save_model=False)
    loss_test = all_pass_loss/len(test_loader)
    print('Test Dice Loss:', loss_test)

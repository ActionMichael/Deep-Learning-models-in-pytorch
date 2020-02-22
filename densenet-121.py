# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:24:25 2020

@author: User
"""

import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True



##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"
GRAYSCALE = True


##########################
### MNIST Dataset
##########################

train_indices = torch.arange(0, 59000)
valid_indices = torch.arange(59000, 60000)

resize_transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])


train_and_valid = datasets.MNIST(root='data', 
                                 train=True, 
                                 transform=resize_transform,
                                 download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=resize_transform,
                              download=True)

train_dataset = Subset(train_and_valid, train_indices)
valid_dataset = Subset(train_and_valid, valid_indices)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                          num_workers=0,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE,
                          num_workers=0,
                          shuffle=False)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE,
                         num_workers=0,
                         shuffle=False)

device = torch.device(DEVICE)
torch.manual_seed(0)

for epoch in range(2):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(device)
        y = y.to(device)
        break

# Check that shuffling works properly
# i.e., label indices should be in random order.
# Also, the label order should be different in the second
# epoch.
for images, labels in train_loader:  
    pass
print(labels[:10])

for images, labels in train_loader:  
    pass
print(labels[:10])

# Check that validation set and test sets are diverse
# i.e., that they contain all classes
for images, labels in valid_loader:  
    pass
print(labels[:10])

for images, labels in test_loader:  
    pass
print(labels[:10])



##########################
### MODEL
##########################

# The following code cell that implements the DenseNet-121 architecture 
# is a derivative of the code provided at 
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

#拼接之前dense_layer層特徵，且返回一層dense_layer
def _bn_function_factory(norm , relu , conv):
    #(*input)不懂
    def bn_function(*input):
        #將兩個張量（tensor）拼接在一起
        #dim表示以哪個维度連接，dim=0為横向連接；dim=1為縱向連接
        concate_features = torch.cat(input , 1)
        #頸縮張量BN->ReLU->1x1Conv
        bottleneck_output = conv(relu(norm(concate_features)))
        return bottleneck_output
    
    return bn_function

#卷積block:BN->ReLU->1x1Conv->BN->ReLU->3x3Conv
#DenseBlock中的內部結構
class _DenseLayer(nn.Module):
    """Basic unit of DenseBlock (using bottleneck layer) """
    #初始"memory_efficient=False"不懂
    def __init__(self , num_input_features , growth_rate , bn_size , drop_rate , memory_efficient=False):
        #num_input_features:輸入特徵圖個數
        #growth_rate:增長速率，第二個卷積層輸出特徵圖
        #growth_rate * bn_size:第一個卷積層輸出特徵圖
        #drop_rate:dropout丟掉率

        #繼承初始化之上述
        super(_DenseLayer , self).__init__()
        #本體~~~
        self.add_module('norm1' , nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1' , nn.ReLU(inplace=True)),
        self.add_module('conv1' , nn.Conv2d(num_input_features , bn_size*growth_rate , 
                                            kernel_size=1 , strid=1 , bias=False)),
        self.add_module('norm2' , nn.BatchNorm2d(bn_size*growth_rate)),
        self.add_module('relu2' , nn.ReLU(inplace=True)),
        ##padding=1所以特徵圖大小一致
        self.add_module('conv2' , nn.Conv2d(bn_size*growth_rate , growth_rate , 
                                            kernel_size=3 , stride=1 , padding=1 , bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient
    
    #法一
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
    
    #法二，法一與法二差別在哪??
    #def forward(self, *prev_features):
    #    bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
    #    #if....節省顯示卡內存
    #    if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
    #        bottleneck_output = cp.checkpoint(bn_function, *prev_features)
    #    else:
    #        bottleneck_output = bn_function(*prev_features)
    #    new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    #    if self.drop_rate > 0:
    #        new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
    #    return new_features

    
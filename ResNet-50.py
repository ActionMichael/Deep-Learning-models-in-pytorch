# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:42:45 2020

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
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

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

##########################
### MODEL
##########################

#定義ResNet基本模組單元-3x3捲積層(kernel_size=3,打包)
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes , out_planes , kernel_size=3 , stride=stride , padding=1 , bias=False)

#定義Bottleneck瓶頸(打包，適用於50、101、152層的resnet){又有別稱residual block(殘差模塊)}
class Bottleneck(nn.Module):
    #這裡是三個卷積，分別是1x1,3x3,1x1,(在rasnet50以上有用到Bottleneck，該expansion=4)
    #分別用來壓縮維度，卷積處理，恢復維度，inplanes是輸入的通道數，
    #planes是輸出的通道數，expansion是對輸出通道數的倍乘，
    #在bottleneck中expansion是4，然而bottleneck就是不尋常，它的任務就是要對通道數進行壓縮，再放大，
    #於是，planes不再代表輸出的通道數，而是block內部壓縮後的通道數，
    #輸出通道數變為planes*expansion。 接著就是網絡主體了。
    expansion = 4
    
    def __init__(self , inplanes , planes , stride=1 , downsample=None):
        super(Bottleneck , self).__init__()
        self.conv1 = nn.Conv2d(inplanes , planes , kernel_size=1 , bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes , planes , kernel_size=3 , stride=stride , padding=1 , bias=False)
        self.bn2 = nn.BatchNorm2d(planes)#inplace=True是否將計算得到的值直接覆蓋之前的值，節省顯卡內存空間
        self.conv3 = nn.Conv2d(planes , planes*4 , kernel_size=1 , bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    #開始定義向前傳播函數
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        #把shortcut那的channel的维度统一
        #蠻關鍵的一環，有關channel(64,128,256,512)轉換時downsample/shortcut/dim-change就非0
        if self.downsample is not None:
            residual = self.downsample(x)
        #就是將殘差和原始的輸入相加，最後再用relu激活函數。很直觀。
        out += residual
        out = self.relu(out)

        return out
        
#定義ResNet50結構    
class ResNet(nn.Module):
    
    def __init__(self , block , layers , num_classes , grayscale):
        self.inplanes = 64
        #如果輸入圖為灰階或彩色的處理
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet , self).__init__()
        #開始編撰ResNet架構，前四行結構為ResNet的前四層
        #resnet共有七個階段，其中第一階段為一個7x7的卷積處理，stride為2，然後經過池化處理，
        #此時特征圖的尺寸已成為輸入的1/4(圖像轉換)
        self.conv1 = nn.Conv2d(in_dim , 64 , kernel_size=7 , stride=2 , padding=3 , bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)
        #接下來是四個階段，也就是代碼中的layer1,layer2,layer3,layer4
        #這裏用make_layer函數產生四個layer，需要用戶輸入每個layer的block數目（即layers列表)以及采用的block類型（基礎版還是bottleneck版）
        #_make_layer()函數用來產生4個layer，可以根據輸入的layers列表來創建網絡
        #這裡的layers[0]=2，然後我們進入到_make_layer函數，
        #由於stride=1或當前的輸入channel和上一個塊的輸出channel一樣，因而可以直接相加
        self.layer1 = self._make_layer(block , 64 , layers[0])
        #此時outplanes=128而self.inplanes=64*4為上basic_block的輸出channel，
        #此時channel不一致，需要對輸出的x擴維後才能相加，downsample實現的就是該功能
        self.layer2 = self._make_layer(block , 128 , layers[1] , stride=2)
        #此時outplanes=256而self.inplanes=128*4為，此時也需要擴維後才能相加，downsample實現的就是該功能
        self.layer3 = self._make_layer(block , 256 , layers[2] , stride=2)
        #此時outplanes=512而self.inplanes=256*4為，此時也需要擴維後才能相加，downsample實現的就是該功能
        self.layer4 = self._make_layer(block , 512 , layers[3] , stride=2)
        #接下來第六階段為avg-pooling
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #最後一層fully connect(全連接)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        #初始化權值参數初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self , block , planes , blocks , stride=1):
        #downsample 主要用來處理H(x)=F(x)+x中F(x)和x之channel维度不匹配問题
        downsample = None
        #self.inplanes為上個box_block的输出channel,planes為當前box_block的输入channel
        if stride!=1 or self.inplanes != planes*block.expansion:
            #https://www.youtube.com/watch?v=lK5rm2_OPGo
            #channel(64,128,256,512)轉換時downsample/shortcut/dim-change
            #捲積1x1"kernel_size=1"NIN(Network in Network)的概念用來change-channel加上BN
            downsample = nn.Sequential(nn.Conv2d(self.inplanes , planes*block.expansion
                                                 , kernel_size=1 , stride=stride , bias=False) , 
                                       nn.BatchNorm2d(planes*block.expansion))
        #開一個儲存容器
        layers=[]
        #只在這裡傳遞了stride=2的参數，因而一個box_block中的圖片大小只在第一次除以2
        #該部分是將每個blocks的第一個residual結構疊加保存在layers列表中。
        layers.append(block(self.inplanes , planes , stride , downsample))
        self.inplanes = planes*block.expansion
        #該部分是將每個block的剩下residual 結構保存在layers列表中，這樣就完成了一個blocks的構造。
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    #正向傳遞
    def forward(self , x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        # view函数將張量out變形成一维的向量形式，總特徵數並不改變，為接下来的全連接(full connection)作準備，更新x
        x = x.view(x.size(0), -1)
        # 執行全連接(full connection)
        logits = self.fc(x)
        #後啟用softmax為activation function(激勵函數)
        probas = F.softmax(logits, dim=1)
        return logits, probas

#
def resnet50(num_classes):
    """Constructs a ResNet-18 model."""
    #[3, 4, 23, 3]為101層；[3, 8, 36, 3]為152層~~~
    model = ResNet(block=Bottleneck, 
                   layers=[3, 4, 6, 3],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model

torch.manual_seed(RANDOM_SEED)

model = resnet50(NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, NUM_EPOCHS, 
              compute_accuracy(model, train_loader, device=DEVICE)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))
    
        
        
        
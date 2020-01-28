# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:03:42 2019

@author: User
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Local Response Normalization (LRN)區域性響應值歸一化層
# 這個網路貌似後續被其它正則化手段代替，如dropout、batch normalization等。
# 目前該網路基本上很少使用了，這為了原生的AlexNet而實現
#https://kknews.cc/zh-tw/other/4k8gg4x.html
#https://www.cnblogs.com/bonelee/p/8268459.html
#ai(x,y)输出结果的結構是一個四维组[batch,height,width,channel]
#a(指的是ai(x,y)),n/2,k,α,β分别表示函数中的input,depth_radius,bias,alpha,beta
class LRN(nn.Module):
    def __init__(self , local_size=1 , alpha=1.0 , beta=0.75 , ACROSS_CHANNELS=False):
        #繼承
        super(LRN , self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size , 1 , 1),#0.2.0_4會報錯，需要在最新的分支上AvgPool3d才有padding參數
                                        stride=1,
                                        padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
    
    def forward(self , x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)#x.pow(2)為x^2,unsqueeze(1)在第二個維度上添加一個一維矩陣
            div = self.average(div).squeeze(1)#squeeze(1)在第二個維度上刪除一個一維矩陣(非一維矩陣無法刪除)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)#这里的1.0即为bias
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

#進入AlexNet神經網路核心
class AlexNet(nn.Module):
    def __init__(self , num_classes=1000):#imagenet共1000個分類
        #繼承
        super().__init__()
        #nn.Sequential快速搭建神經網路的容器語法
        #第一層 Convolutional Layer: 96 kernels of size 11×11×3(stride: 4, pad: 0)
        #       3×3 Overlapping Max Pooling (stride: 2)
        #       Local Response Normalization
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3 , out_channels=96 , kernel_size=11 , stride=4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3 , stride=2),
                nn.LRN(local_size=5 , alpha=1e-4 , beta=0.75 , ACROSS_CHANNELS=True))
        #第二層 Convolutional Layer: 256 kernels of size 5×5×48(stride: 1, pad: 2)
        #       3×3 Overlapping Max Pooling (stride: 2)
        #       Local Response Normalization
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=96 , out_channels=256 , kernel_size=5 , groups=2 , padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3 , stride=2),
                nn.LRN(local_size=5 , alpha=1e-4 , beta=0.75 , ACROSS_CHANNELS=True))
        #第三層 Convolutional Layer: 384 kernels of size 3×3×256(stride: 1, pad: 1)
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=256 , out_channels=384 , kernel_size=3 , padding=1),
                nn.ReLU(inplace=True))
        #第四層 Convolutional Layer: 384 kernels of size 3×3×192(stride: 1, pad: 1)
        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=384 , out_channels=384 , kernel_size=3 , padding=1),
                nn.ReLU(inplace=True))
        #第五層 Convolutional Layer: 256 kernels of size 3×3×192(stride: 1, pad: 1)
        #       3×3 Overlapping Max Pooling (stride: 2)
        self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=384 , out_channels=256 , kernel_size=3 , padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3 , stride=2))
        #第六層需要針對上一層改變view 
        #Fully Connected (Dense) Layer of 
        self.layer6 = nn.Sequential(
                nn.Linear(in_features=6*6*256 , out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout())
        #第七層Fully Connected (Dense) Layer of 
        self.layer7 = nn.Sequential(
                nn.Linear(in_features=4096 , out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout())
         #第八層Fully Connected (Dense) Layer of
        self.layer8 = nn.Linear(in_features=4096, out_features=num_classes)
        
    #定義正向傳遞    
    def forward(self, x):
        #捲積層
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        #將所有參數拉成一直線(為第六層全連接準備)
        x = x.view(-1, 6*6*256)
        #全連接
        x = self.layer8(self.layer7(self.layer6(x)))
        return x
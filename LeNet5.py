# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:03:19 2019

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

class LeNet5(nn.Module):
    ###開始定義神經網路架構###
    #注意：特殊方法“ __init__ ”前後分別有兩個下劃線!!!
    #注意到__init__方法的第一個參數永遠是self，表示創建的實例本身，
    #因此在__init__方法內部，就可以把各種屬性綁定到self，因為self就指向創建的實例本身。
    #有了__init__方法，在創建實例的時候，就不能傳入空的參數了，
    #必須傳入與__init__方法匹配的參數，但self不需要傳，Python解釋器自己會把實例變量傳進去：
    #和普通的函數相比，在類中定義的函數只有一點不同，就是第一個參數永遠是實例變量self，並且調用時，不用傳遞該參數。
    #除此之外，類的方法和普通函數沒有什麼區別，所以，你仍然可以用默認參數、可變參數、關鍵字參數和命名關鍵字參數。
    def __init__(self):
        #這是對繼承自父類的屬性進行初始化
        super(LeNet5 , self).__init__()
        # 1 input image channel(1因為MINST是灰階，有人用3(RGB)),6 output channels,5x5 square convolution
        # "2d"是指2維的input圖
        self.conv1 = nn.Conv2d(1,6,5)
        # 6 input image channel, 16 output channels, 5x5 square convolution
        # "2d"是指2維的input圖
        self.conv2 = nn.Conv2d(6,16,5)
        # full connect layer 1, (5*5 square 16 channels input),(y = Wx + b)120 dot output
        self.fc1 = nn.Linear(16*5*5,120)
        # full connect layer 2,(y = Wx + b) 120 dot input 84 dot output
        self.fc2 = nn.Linear(120,84)
        # full connect layer 3,(y = Wx + b) 84 dot input 10 dot output
        self.fc3 = nn.Linear(84,10)
        
    ###開始定義向前傳播函數###
    def forward(self , x):
        # 過convolutional 1(卷積層1)後啟用ReLu為activation function(激勵函數)
        x = F.relu(self.conv1(x))
        # 將上述之out(過第一次激勵後)帶入一個2*2的max pooling(池化層)，更新x，"2d"是指2維的input圖
        x = F.max_pool2d(x, 2)
        # 過convolutional 2(卷積層2)後啟用ReLu為activation function(激勵函數)，更新x
        x = F.relu(self.conv2(x))
        # 將上述之out(過第二次激勵後)帶入一個2*2的max pooling(池化層)，更新x，"2d"是指2維的input圖
        x = F.max_pool2d(x, 2)
        # view函数將張量out變形成一维的向量形式，總特徵數並不改變，為接下来的全連接(full connection)作準備，更新x
        x = x.view(-1, self.num_flat_features(x))
        # 執行全連接1(full connection 1)後啟用ReLu為activation function(激勵函數)，更新x
        x = F.relu(self.fc1(x))
        # 執行全連接2(full connection 2)後啟用ReLu為activation function(激勵函數)，更新x
        x = F.relu(self.fc2(x))
        # 執行全連接3(full connection 3)，更新x
        x = self.fc3(x)
        return x
    
    ###使用num_flat_features函数计算張量x的總特徵量（把每個數字都看出是一個特徵，即特徵總量)
    ###比如x是4*2*2的張量，那它的特徵總量就是16### 
    def num_flat_features(self, x):
        # 為什麼要使用[1:],是因為pytorch只接受批量输入，若一次性输入多圖片那输入數據張量的维度自然上升到了4维。
        #【1:】使我们把注意力放在後3维上面
        #轉換（降低）資料維度，進入全連線層
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 


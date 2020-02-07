# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:51:09 2019

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


##########################
### SETTINGS
##########################

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 128

# Architecture
num_features = 784
num_classes = 10


##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.CIFAR10(root='data', 
                                 train=True, 
                                 transform=transforms.ToTensor(),
                                 download=True)

test_dataset = datasets.CIFAR10(root='data', 
                                train=False, 
                                transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break




#神經網路
class VGG16(nn.Module):
    
    def __init__(self , num_features , num_classes):
        super(VGG16 , self).__init__()
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        # padding是(1(64-1)- 64 + 3)/2 = 1
        # 第一區塊：2個搭配relu捲積層，1層最大池化層maxpooling
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=3 , out_channels=64 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64 , out_channels=64 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)))
        # 第二區塊：2個搭配relu捲積層，1層最大池化層maxpooling
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128 , out_channels=128 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)))
        # 第三區塊：3個搭配relu捲積層，1層最大池化層maxpooling
        self.block_3 = nn.Sequential(
                nn.Conv2d(in_channels=128 , out_channels=256 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256 , out_channels=256 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256 , out_channels=256 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)))                             
        # 第四區塊：3個搭配relu捲積層，1層最大池化層maxpooling
        self.block_4 = nn.Sequential(
                nn.Conv2d(in_channels=256 , out_channels=512 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)))
        # 第五區塊：3個搭配relu捲積層，1層最大池化層maxpooling
        self.block_5 = nn.Sequential(
                nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=(3,3) , stride=(1,1) , padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)))
        # 第六區塊：3個全連接full connect:fc1(4096),fc2(4096),fc3(1000)組建出分類器
        self.classifier = nn.Sequential(
                nn.Linear(512*4*4 , 4069),
                nn.ReLU(),
                nn.Linear(4069 , 4069),
                nn.ReLU(),
                nn.Linear(4069 , num_classes))
        
        # 第七區塊：初始化
        # We usually use for m in self.modules() loop to initialize weights and biases
        # https://blog.csdn.net/ys1305/article/details/94332007
        # https://github.com/pytorch/pytorch/issues/19348
        for m in self.modules():
            if isinstance(m , torch.nn.Conv2d) or isinstance(m , torch.nn.Linear):
                #kaiming_uniform均匀初始化(針對ReLu,#3)可以使非對稱的ReLu收斂更快(fast.ai)，初始化方法很多種
                #1 https://blog.csdn.net/ys1305/article/details/94332007
                #2 https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
                #3 https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                #不是太清楚，針對上述沒定義到的偏差值給初始zero_()
                if m.bias is not None:
                    m.bias.detach().zero_()
                    
        ###第七區塊：初始化(法二)
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, np.sqrt(2. / n))
        #         m.weight.detach().normal_(0, 0.05)
        #         if m.bias is not None:
        #             m.bias.detach().zero_()
        #     elif isinstance(m, torch.nn.Linear):
        #         m.weight.detach().normal_(0, 0.05)
        #         m.bias.detach().detach().zero_()
    
    #定義正向傳遞    
    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)

        return logits, probas



torch.manual_seed(random_seed)
model = VGG16(num_features=num_features,
              num_classes=num_classes)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss
    
    

start_time = time.time()
for epoch in range(num_epochs):
    
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
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%% |  Loss: %.3f' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader),
              compute_epoch_loss(model, train_loader)))


    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
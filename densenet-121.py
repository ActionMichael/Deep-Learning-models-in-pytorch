# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:24:25 2020

@author: User
"""
"""
有如下優點：
（1）由於存在很多跳連，減輕了空梯度問題，加強了梯度和信息流動，更容易訓練。
（2）加強了特徵的重用。
（3）DenseNet層的filter數量比較少，使得層比較狹窄。每層只生成了很小的特徵圖，成為共同知識中的一員。這樣網絡的參數更少，且更有效率。
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
        #本體~~~我們使用bottleneck結構
        self.add_module('norm1' , nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1' , nn.ReLU(inplace=True)),
        self.add_module('conv1' , nn.Conv2d(num_input_features , bn_size*growth_rate , 
                                            kernel_size=1 , stride=1 , bias=False)),
        self.add_module('norm2' , nn.BatchNorm2d(bn_size*growth_rate)),
        self.add_module('relu2' , nn.ReLU(inplace=True)),
        ##padding=1所以特徵圖大小一致
        self.add_module('conv2' , nn.Conv2d(bn_size*growth_rate , growth_rate , 
                                            kernel_size=3 , stride=1 , padding=1 , bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient
        
    #法一、正向傳遞，但是(*prev_features)不懂
    def forward(self, *prev_features):
        #(bn=BatchNorm2d)呼叫"_bn_function_factory"函式來實現"conv(relu(norm(concate_features)))"
        bn_function = _bn_function_factory(self.norm1 , self.relu1 , self.conv1)
        #節省顯示卡記憶體(看不懂~~~)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function , *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        #正向傳遞第二部分，因為上行"bottleneck_output"乘載了bn_function了所以沿用，套入第"2"部分
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        #導入drop-out，使梯度不會over爆炸及稀釋消失
        if self.drop_rate > 0:
            new_features = F.dropout(new_features , p=self.drop_rate , training=self.training)
        return new_features
    
    #法二，看起來相對法一清楚明瞭，但運作起來出現"NotImplementedError"
    #def forward(self, x):
    #    new_features = super(_DenseLayer, self).forward(x)
    #    if self.drop_rate > 0:
    #        new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
    #    return torch.cat([x, new_features], 1)


#法一、實現DenseBlock模塊，內部是密集連接方式(輸入特徵數線性增長)：
#num_input_features:輸入特徵圖個數
#growth_rate:增長速率，第二個卷積層輸出特徵圖
"""
Dense Block有L層dense layer組成
layer 0:輸入（56 * 56 * 64）->輸出（56 * 56 * 32）
layer 1:輸入（56 * 56 (32 * 1))->輸出（56 * 56 * 32）
layer 2:輸入（56 * 56 (32 * 2))->輸出（56 * 56 * 32）
…
layer L:輸入（56 * 56 * (32 * L))->輸出（56 * 56 * 32）

注意，L層dense layer的輸出都是不變的，而每層的輸入channel數是增加的，
因為如上所述，每層的輸入是前面所有層的拼接。
在一個DenseBlock裡面，每個非線性變換H輸出的channels數為恆定的Growth_rate，
那麼第i層的輸入的channels數便是num_input_features + i*Growth_rate, num_input_features為Input
的channels數，比如，假設我們把Growth_rate設為4，
上圖中H1的輸入的size為8 * 32 * 32，輸出為4 * 32 * 32， 則H2的輸入的size為12 * 32 * 32，
輸出還是4 * 32 * 32，H3、H4以此類推，在實驗中，用較小的Growth_rate就能實現較好的效果。
"""
class _DenseBlock(nn.Module):
    "num_layers:每個block内dense layer層數"
    def __init__(self , num_layers , num_input_features , bn_size , growth_rate , drop_rate , memory_efficient=False):
        super(_DenseBlock , self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1) , layer)

    def forward(self , init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features , 1)

#法二、
#class _DenseBlock(nn.Sequential):
#    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
#"num_layers:每個block内dense layer層數"
#        super(_DenseBlock, self).__init__()
#        for i in range(num_layers):
#            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
#            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_featuremaps (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_featuremaps=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False,
                 grayscale=False):

        super(DenseNet121, self).__init__()

        # First convolution
        if grayscale:
            in_channels=1
        else:
            in_channels=3
        
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=in_channels, out_channels=num_init_featuremaps,
                                kernel_size=7, stride=2,
                                padding=3, bias=False)), # bias is redundant when using batchnorm
            ('norm0', nn.BatchNorm2d(num_features=num_init_featuremaps)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_featuremaps
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

torch.manual_seed(RANDOM_SEED)

model = DenseNet121(num_classes=NUM_CLASSES, grayscale=GRAYSCALE)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def compute_acc(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    model.eval()
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        assert predicted_labels.size() == targets.size()
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

start_time = time.time()

cost_list = []
train_acc_list, valid_acc_list = [], []


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
        
        #################################################
        ### CODE ONLY FOR LOGGING BEYOND THIS POINT
        ################################################
        cost_list.append(cost.item())
        if not batch_idx % 150:
            print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                   f'Batch {batch_idx:03d}/{len(train_loader):03d} |' 
                   f' Cost: {cost:.4f}')

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        
        train_acc = compute_acc(model, train_loader, device=DEVICE)
        valid_acc = compute_acc(model, valid_loader, device=DEVICE)
        
        print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d}\n'
              f'Train ACC: {train_acc:.2f} | Validation ACC: {valid_acc:.2f}')
        
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')
  
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

plt.plot(cost_list, label='Minibatch cost')
plt.plot(np.convolve(cost_list, 
                     np.ones(200,)/200, mode='valid'), 
         label='Running average')

plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.legend()
plt.show()

plt.plot(np.arange(1, NUM_EPOCHS+1), train_acc_list, label='Training')
plt.plot(np.arange(1, NUM_EPOCHS+1), valid_acc_list, label='Validation')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


with torch.set_grad_enabled(False):
    test_acc = compute_acc(model=model,
                           data_loader=test_loader,
                           device=DEVICE)
    
    valid_acc = compute_acc(model=model,
                            data_loader=valid_loader,
                            device=DEVICE)
    

print(f'Validation ACC: {valid_acc:.2f}%')
print(f'Test ACC: {test_acc:.2f}%')    
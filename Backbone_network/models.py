import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

class CCLoss(nn.Module):
  def __init__(self, r1, thred):
    super(CCLoss, self).__init__()
    self.mse = nn.MSELoss(reduction='mean')
    self.bce = nn.BCELoss(reduction='mean')
    self.r1 = r1
    self.thred = torch.tensor([0.3]).to('cuda:1')
    # self.thred = torch.tensor(thred).to('cuda:1')

  def forward(self, x1, y):
    # print(y1.device)
    mseloss = self.mse(x1, y)
    # print("size of y: ", y.size())
    label = (y>=self.thred).float()*1
    # print("size of label: ", label.size())
    bceloss = self.bce(x1, label)
    # y = y.view(-1,1)
    # pcc = nn.functional.cosine_similarity(x - x.mean(dim=0, keepdim=True), y - y.mean(dim=0, keepdim=True), dim=0, eps=1e-6)
    # loss = (1 - self.r) * mseloss + self.r * (1 - pcc)**2
    loss1 = (1-self.r1) * mseloss + self.r1 * bceloss
    # print('MSE: ', mseloss)
    # print('PCC: ', torch.abs(pearson_correlation))
    return loss1

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.activation = nn.ELU()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 125), stride=(1,1), padding=(0,62), bias=False),
            nn.BatchNorm2d(16)
        )
        self. depthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(30, 1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d((1,4), stride=(1,4), padding=0),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 31), stride=(1,1), padding = (0,15), bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d((1,187), stride=(1,1), padding=0),
            # nn.AvgPool2d((1,8), stride=(1,8), padding=0),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(736, 2, bias=True),
            # nn.Sigmoid()
        )

        self.regressor = predictor(32)

    def forward(self, x):
        # print("input size: ", x.size())
        x = self.conv_block1(x)
        # print("size after block1: ", x.size())
        x = self.depthwiseConv(x)
        # print("size after depthwise: ", x.size()) 
        x = self.separableConv(x)
        # print(f'shape before flatten: {x.size()}')
        x = x.view(-1, 32)
        #print(f'reshape: {x.size()}')
        out = self.regressor(x)

        return x, out

# SCCNet archetecture
class SCCNet(nn.Module):
  def __init__(self):
    super(SCCNet, self).__init__()
    # temporal and spatial filter
    self.Conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=22, kernel_size=(30, 1)),
        # nn.BatchNorm2d(22)
    )
    self.Conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(22, 25), padding=(0,12)),
        # nn.BatchNorm2d(20)
    )

    self.AveragePooling1 = nn.AvgPool2d((1,750))
    # self.AveragePooling1 = nn.AvgPool2d((1,125), stride = (1,125)) # to 120
    # stride is set as 25 (0.1 sec correspond to time domain)
    # kernel size 125 mean 0.5 sec
    self.regressor = predictor(20) 
    self.sigmoid = nn.Sigmoid()

  def square(self, x):
    return torch.pow(x,2)

  def log_activation(self, x):
    return torch.log(x)

  def forward(self, x):

    x = self.Conv_block1(x)
    x = x.permute(0,2,1,3)

    x = self.Conv_block2(x)
    x = self.square(x)
    x = x.permute(0,2,1,3)

    x = self.AveragePooling1(x)
    latent = self.log_activation(x)
    x = torch.flatten(latent,1)
    
    x = self.regressor(x)
    output = self.sigmoid(x)
    return latent, output

class predictor(nn.Module):
    def __init__(self, fc2):
        super(predictor, self).__init__()

        self.regressor = nn.Sequential(
            # nn.Linear(fc1, fc2),
            # nn.ReLU(),
            nn.Linear(fc2, 1, bias = True)
            # nn.Sigmoid()
        )
    def forward(self, x):
        return self.regressor(x)

class self_attention(nn.Module):
    def __init__(self, hidden_size, head):
        super(self_attention,self).__init__()
        self.hidden_size = hidden_size
        self.head = head
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.head, batch_first = True)
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        q = self.query(x.permute(0,1,3,2)).squeeze(1)
        k = self.key(x.permute(0,1,3,2)).squeeze(1)
        v = self.value(x.permute(0,1,3,2)).squeeze(1)
        # print('shape q: ' ,q.size())
        output, score = self.multihead_attn(q,k,v)

        return output, score


class SCC_multi_window(nn.Module):
    def __init__(self):
        super(SCC_multi_window, self).__init__()

        self.sm_num = 10
        self.feat_extractor = SCCNet()
        self.GAP = nn.AvgPool2d((1,self.sm_num))
        # self.attention = self_attention(20, 2)
        self.regressor = predictor(20)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        latent, _ = self.feat_extractor(x[:,0,:,:].unsqueeze(1))
        for j in range(1, self.sm_num):
            t, _ = self.feat_extractor(x[:,j,:,:].unsqueeze(1))
            latent = torch.cat((latent, t), 3)

        # print('size after concate: ', x.size())

        # x, score = self.attention(x)
        latent = self.GAP(latent)

        x = torch.flatten(latent, 1)
        output = self.regressor(x)
        output = self.sigmoid(output)
        # print('output size', output.size())

        return latent, output

class Siamese_SCC(nn.Module):
    def __init__(self, num_smooth=10):
        super(Siamese_SCC, self).__init__()

        self.sm_num = num_smooth
        self.feat_extractor = SCCNet()

        self.GAP = nn.AvgPool2d((1,self.sm_num))

        self.regressor = predictor(20)
        self.delta_regressor = nn.Sequential(
            nn.Linear(40, 1, bias = True)
        )
        ## Activation
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        

    def forward(self, x):

        latent = []
        for j in range(self.sm_num, x.size()[1]):
            t, _ = self.feat_extractor(x[:,j,:,:].unsqueeze(1))
            latent.append(t)
            
        latent = torch.cat(latent, 3)
        latent = self.GAP(latent)

        x_di = torch.flatten(latent, 1)
        x_di = self.regressor(x_di)
        output_di = self.sigmoid(x_di)
        
        ### Parallel sub-network
        with torch.no_grad():
            b_latent = []
            for i in range(0,self.sm_num):
                b, _ = self.feat_extractor(x[:,i,:,:].unsqueeze(1))
                b_latent.append(b)

            b_latent = torch.cat(b_latent,3)
            b_latent = self.GAP(b_latent)
            
        x_delta = torch.cat((b_latent, latent), 2)
        x_delta = torch.flatten(x_delta, 1)
        output_delta = self.delta_regressor(x_delta)
        output_delta = self.tanh(output_delta)
        # print('output size', output.size())

        return x_delta, output_di, output_delta
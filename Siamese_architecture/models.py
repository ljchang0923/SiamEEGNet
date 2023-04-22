import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

class CCLoss(nn.Module):
    def __init__(self, r1, r2, thred):
        super(CCLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')
        self.r1 = r1
        self.r2 = r2
        self.thred = thred

    def forward(self, x1, x2, y):
        y1 = y[:,0].view(-1,1)
        y2 = y[:,1].view(-1,1)
        # print(y1.device)
        mseloss = self.mse(x1, y1)
        # print("size of y: ", y.size())
        label = (y1>=self.thred).float()*1
        # print("size of label: ", label.size())
        bceloss = self.bce(x1, label)
        # y = y.view(-1,1)
        # pcc = nn.functional.cosine_similarity(x - x.mean(dim=0, keepdim=True), y - y.mean(dim=0, keepdim=True), dim=0, eps=1e-6)
        # loss = (1 - self.r) * mseloss + self.r * (1 - pcc)**2
        loss1 = (1-self.r1) * mseloss + self.r1 * bceloss
        loss2 = self.mse(x2, y1-y2)
        # print('MSE: ', mseloss)
        # print('PCC: ', torch.abs(pearson_correlation))
        return (1-self.r2) * loss1 + (self.r2) * loss2

class EEGNet(nn.Module):
    def __init__(self, cfg):
        super(EEGNet, self).__init__()

        self.EEG_ch = cfg["EEG_ch"]
        self.F1 = 16
        self.F2 = 32

        self.activation = nn.ELU()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 125), stride=(1,1), padding=(0,62), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=self.F1, out_channels=self.F2, kernel_size=(self.EEG_ch, 1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(self.F2),
            self.activation,
            nn.AvgPool2d((1,4), stride=(1,4), padding=0),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=(1, 31), stride=(1,1), padding = (0,15), bias=False),
            nn.BatchNorm2d(self.F2),
            self.activation,
            nn.AvgPool2d((1,187), stride=(1,1), padding=0),
            # nn.AvgPool2d((1,8), stride=(1,8), padding=0),
            nn.Dropout(0.25)
        )

        self.regressor = predictor(32)

    def forward(self, x):

        x = self.conv_block1(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        latent = x.view(-1, 32)
        out = self.regressor(latent)

        return x, out

class ShallowConvNet(nn.Module):
    def __init__(self, cfg):
        super(ShallowConvNet, self).__init__()

        self.EEG_ch = cfg["EEG_ch"]
        self.F1 = 40
        self.F2 = 40

        self.conv1 = nn.Conv2d(1, self.F1, (1, 25), bias=False)
        self.conv2 = nn.Conv2d(self.F1, self.F2, (self.EEG_ch, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(self.F2)

        self.AvgPool1 = nn.AvgPool2d((1, 726), stride=(1, 1))
    
        self.Drop1 = nn.Dropout(0.25)
        self.regressor = predictor(40)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Bn1(x)
        x = x ** 2
        x = self.AvgPool1(x)
        log_power = torch.log(x)
    
        x = self.Drop1(log_power)
        x = torch.flatten(x, 1)
        output = self.regressor(x)

        return log_power ,output

class SCCNet(nn.Module):
    def __init__(self, cfg):
        super(SCCNet, self).__init__()

        # structure parameters
        self.num_ch = cfg['EEG_ch']
        self.fs = 250
        self.F1 = 22
        self.F2 = 20
        self.t1 = self.fs // 10
        self.t2 = self.fs * 3

        # temporal and spatial filter
        self.Conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(self.num_ch, 1)),
            nn.BatchNorm2d(self.F1)
        )
        self.Conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F2, kernel_size=(self.F1, self.t1), padding=(0, self.t1//2)),
            nn.BatchNorm2d(self.F2)
        )

        self.AveragePooling1 = nn.AvgPool2d((1, self.t2))
        # self.AveragePooling1 = nn.AvgPool2d((1,125), stride = (1,25))
        # stride is set as 25 (0.1 sec correspond to time domain)
        # kernel size 125 mean 0.5 sec
        self.dropout = nn.Dropout(0.5)
        self.regressor = predictor(self.F2)
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
        x = self.dropout(x)

        # x = x.permute(0,2,1,3)
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


class SCC_multi_window(nn.Module):
    def __init__(self, num_smooth=10):
        super(SCC_multi_window, self).__init__()

        self.feat_extractor = SCCNet()

        self.dim_latent = self.feat_extractor.F2
        self.sm_num = num_smooth

        self.GAP = nn.AvgPool2d((1,self.sm_num))
        self.regressor = predictor(self.dim_latent)
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

def selector(model):
    if model == "SCCNet":
        return SCCNet
    elif model == "EEGNet":
        return EEGNet
    elif model == "ShallowNet":
        return ShallowConvNet
    else:
        raise ValueError("Undefined model")

class Siamese_CNN(nn.Module):
    def __init__(self, cfg):
        super(Siamese_CNN, self).__init__()

        self.sm_num = cfg['num_window']

        eegmodel = selector(cfg["backbone"])      
        self.feat_extractor = eegmodel(cfg)
        # self.feat_extractor = SCCNet(cfg)
        self.dim_latent = self.feat_extractor.F2

        self.GAP = nn.AvgPool2d((1,self.sm_num))

        self.regressor = predictor(self.dim_latent) ## SCCNet: 20 EEGNet: 32 shallowConvNet: 40
        self.delta_regressor = nn.Sequential(
            nn.Linear(self.dim_latent*2, 1, bias = True)
        )
        
        ## Activation
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        ### Sub-network 1
        latent = []
        for j in range(self.sm_num, x.size()[1]):
            t, _ = self.feat_extractor(x[:,j,:,:].unsqueeze(1))
            latent.append(t)
        
        ### DI of the current input trial   
        latent = torch.cat(latent, 3)
        # print("latent size: ", latent.size())
        latent = self.GAP(latent)
        x_di = torch.flatten(latent, 1)
        x_di = self.regressor(x_di)
        output_di = self.sigmoid(x_di)
        
        ### Sub-network 2 (baseline)
        with torch.no_grad():
            b_latent = []
            for i in range(0,self.sm_num):
                b, _ = self.feat_extractor(x[:,i,:,:].unsqueeze(1))
                b_latent.append(b)

            b_latent = torch.cat(b_latent,3)
            b_latent = self.GAP(b_latent)
            
        ### Concatenate and Regression head
        x_delta = torch.cat((b_latent, latent), 2)
        # x_delta = torch.sub(latent, b_latent)
        x_delta = torch.flatten(x_delta, 1)
        output_delta = self.delta_regressor(x_delta)
        output_delta = self.tanh(output_delta)

        return x_delta, output_di, output_delta
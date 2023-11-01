import torch
import torch.nn as nn
from torchsummary import summary

from model.backbone import backbone_selector

class Multi_window_CNN(nn.Module):    
    def __init__(self, EEG_ch=30, fs=250, num_window=10, backbone='SCCNet'):
        super(Multi_window_CNN, self).__init__()

        eegmodel = backbone_selector(backbone)      
        self.feat_extractor = eegmodel(EEG_ch=EEG_ch, fs=fs)
        # summary(self.feat_extractor, input_size=(1,30,750), device='cpu')
        # input('break')
        self.dim_latent = self.feat_extractor.DL

        self.GAP = nn.AvgPool2d((1, num_window))
        self.regressor = nn.Linear(self.dim_latent, 1, bias = True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        intermediate_latent = {}
        latent = [] 
        for j in range(0, x.size()[1]):
            h, _ = self.feat_extractor(x[:,j,:,:].unsqueeze(1))
            latent.append(h)

        latent = torch.cat(latent, 3)
        smoothed_latent = self.GAP(latent)

        intermediate_latent['single'] = latent
        intermediate_latent['smoothed'] = smoothed_latent

        x = torch.flatten(smoothed_latent, 1)
        output = self.regressor(x)
        output = self.sigmoid(output)
        # print('output size', output.size())

        return intermediate_latent, output

class SiamEEGNet(nn.Module):
    def __init__(self, EEG_ch=30, fs=250, num_window=10, backbone='SCCNet'):
        super(SiamEEGNet, self).__init__()

        self.num_win = num_window
        # self.dim_inter = 10
        self.backbone = Multi_window_CNN(EEG_ch=EEG_ch, fs=fs, num_window=num_window, backbone=backbone)
        self.dim_latent = self.backbone.feat_extractor.DL
        self.GAP = nn.AvgPool2d((1, num_window))

        ## SCCNet: 20 EEGNet: 32 shallowConvNet: 40
        # self.intermediate = nn.Linear(self.dim_latent*2, self.dim_inter)
        self.delta_regressor = nn.Linear(self.dim_latent*2, 1, bias = True)
        
        ## Activation
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        intermediate_latent = {}
        ### Sub-network 1
        intermediate_latent, output_di = self.backbone(x[:, self.num_win:x.size()[1], :, :])
        
        ### Sub-network 2 (for baseline)
        with torch.no_grad():
            b_intermediate_latent, _ = self.backbone(x[:, :self.num_win, :, :])
            
        ### Concatenate and Regression head
        concat_latent = torch.cat((b_intermediate_latent["smoothed"], intermediate_latent["smoothed"]), 2)
        concat_latent = torch.flatten(concat_latent, 1)
        intermediate_latent['concat'] = concat_latent

        delta_DI = self.delta_regressor(concat_latent)
        delta_DI = self.tanh(delta_DI)

        return intermediate_latent, delta_DI
import torch
import torch.nn as nn
from backbone import SCCNet, EEGNet, ShallowConvNet, EEGTCNet, MBEEGSE

def selector(model):
    if model == "SCCNet":
        return SCCNet
    elif model == "EEGNet":
        return EEGNet
    elif model == "ShallowNet":
        return ShallowConvNet
    elif model == "EEGTCNet":
        return EEGTCNet
    elif model == "MBEEGSE":
        return MBEEGSE
    else:
        raise ValueError("Undefined model")

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

class Multi_window_input(nn.Module):    
    def __init__(self, cfg):
        super(Multi_window_input, self).__init__()

        eegmodel = selector(cfg["backbone"])      
        self.feat_extractor = eegmodel(cfg)
        
        self.dim_latent = self.feat_extractor.FN
        self.sm_num = cfg["num_window"]

        self.GAP = nn.AvgPool2d((1,self.sm_num))
        self.regressor = predictor(self.dim_latent)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        latent = [] 
        for j in range(0, self.sm_num):
            t, _ = self.feat_extractor(x[:,j,:,:].unsqueeze(1))
            latent.append(t)

        # print('size after concate: ', x.size())
        latent = torch.cat(latent, 3)
        latent = self.GAP(latent)

        x = torch.flatten(latent, 1)
        output = self.regressor(x)
        output = self.sigmoid(output)
        # print('output size', output.size())

        return latent, output


class Siamese_CNN(nn.Module):
    def __init__(self, cfg):
        super(Siamese_CNN, self).__init__()

        self.sm_num = cfg['num_window']

       
        self.backbone = Multi_window_input(cfg)
        self.dim_latent = self.backbone.feat_extractor.FN
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
        # latent = []
        # for j in range(self.sm_num, x.size()[1]):
        #     t, _ = self.feat_extractor(x[:,j,:,:].unsqueeze(1))
        #     latent.append(t)
        
        # ### DI of the current input trial   
        # latent = torch.cat(latent, 3)
        # # print("latent size: ", latent.size())
        # latent = self.GAP(latent)
        # x_di = torch.flatten(latent, 1)
        # x_di = self.regressor(x_di)
        # output_di = self.sigmoid(x_di)
        latent, output_di = self.backbone(x[:, self.sm_num:x.size()[1], :, :])
        
        ### Sub-network 2 (baseline)
        with torch.no_grad():
            # b_latent = []
            # for i in range(0,self.sm_num):
            #     b, _ = self.feat_extractor(x[:,i,:,:].unsqueeze(1))
            #     b_latent.append(b)

            # b_latent = torch.cat(b_latent,3)
            # b_latent = self.GAP(b_latent)
            b_latent, _ = self.backbone(x[:, :self.sm_num, :, :])
            
        ### Concatenate and Regression head
        x_delta = torch.cat((b_latent, latent), 2)
        # x_delta = torch.sub(latent, b_latent)
        x_delta = torch.flatten(x_delta, 1)
        output_delta = self.delta_regressor(x_delta)
        output_delta = self.tanh(output_delta)

        return x_delta, output_di, output_delta
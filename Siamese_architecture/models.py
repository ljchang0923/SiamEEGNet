import torch
import torch.nn as nn
from torchsummary import summary
from backbone import SCCNet, EEGNet, ShallowConvNet, EEGTCNet, MBEEGSE, InterpretableCNN, ESTCNN

def backbone_selector(model):
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
    elif model == "InterpretableCNN":
        return InterpretableCNN
    elif model == "ESTCNN":
        return ESTCNN
    else:
        raise ValueError("Undefined model")

class Multi_window_input(nn.Module):    
    def __init__(self, num_window=10, **kwargs):
        super(Multi_window_input, self).__init__()

        eegmodel = backbone_selector(kwargs["backbone"])      
        self.feat_extractor = eegmodel(**kwargs)
        # summary(self.feat_extractor, input_size=(1,30,750), device='cpu')
        # input('break')
        self.dim_latent = self.feat_extractor.FN

        self.GAP = nn.AvgPool2d((1, num_window))
        self.regressor = nn.Linear(self.dim_latent, 1, bias = True)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        latent = [] 
        for j in range(0, x.size()[1]):
            t, _ = self.feat_extractor(x[:,j,:,:].unsqueeze(1))
            latent.append(t)

        latent = torch.cat(latent, 3)
        latent = self.GAP(latent)

        x = torch.flatten(latent, 1)
        output = self.regressor(x)
        output = self.sigmoid(output)
        # print('output size', output.size())

        return latent, output


class Siamese_CNN(nn.Module):
    def __init__(self, num_window=10, **kwargs):
        super(Siamese_CNN, self).__init__()

        self.num_win = num_window
        # self.dim_inter = 10
        self.backbone = Multi_window_input(num_window, **kwargs)
        self.dim_latent = self.backbone.feat_extractor.FN
        self.GAP = nn.AvgPool2d((1, num_window))

        ## SCCNet: 20 EEGNet: 32 shallowConvNet: 40
        # self.intermediate = nn.Linear(self.dim_latent*2, self.dim_inter)
        self.delta_regressor = nn.Linear(self.dim_latent*2, 1, bias = True)
        
        ## Activation
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        ### Sub-network 1
        latent, output_di = self.backbone(x[:, self.num_win:x.size()[1], :, :])
        
        ### Sub-network 2 (for baseline)
        with torch.no_grad():
            b_latent, _ = self.backbone(x[:, :self.num_win, :, :])
            
        ### Concatenate and Regression head
        x_delta = torch.cat((b_latent, latent), 2)

        x_delta = torch.flatten(x_delta, 1)
        # inter_latent = self.intermediate(x_delta)
        output_delta = self.delta_regressor(x_delta)
        output_delta = self.tanh(output_delta)

        return (output_di, x_delta), output_delta
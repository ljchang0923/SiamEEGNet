import torch
import torch.nn as nn
import numpy as np
import os

class InterpretableCNN(torch.nn.Module):  
    
    """
    The codes implement the CNN model proposed in the paper "EEG-based Cross-Subject Driver Drowsiness Recognition
    with an Interpretable Convolutional Neural Network".(doi: 10.1109/TNNLS.2022.3147208)
    
    The network is designed to classify multi-channel EEG signals for the purposed of driver drowsiness recognition.
    
    Parameters:
        
    classes       : number of classes to classify, the default number is 2 corresponding to the 'alert' and 'drowsy' labels.
    sampleChannel : channel number of the input signals.
    sampleLength  : the length of the EEG signals. The default value is 384, which is 3s signal with sampling rate of 128Hz.
    N1            : number of nodes in the first pointwise layer.
    d             : number of kernels for each new signal in the second depthwise layer.      
    kernelLength  : length of convolutional kernel in second depthwise layer.
   
    if you have any problems with the code, please contact Dr. Cui Jian at cuij0006@ntu.edu.sg
    """    
    
    def __init__(self, classes=1, EEG_ch=30, fs=250 ,N1=16, d=2, kernelLength=125, **kwargs):
        super(InterpretableCNN, self).__init__()
        sampleLength = fs * 3
        self.pointwise = torch.nn.Conv2d(1, N1, (EEG_ch,1))
        self.depthwise = torch.nn.Conv2d(N1, d*N1, (1,kernelLength), groups=N1) 
        self.activ=torch.nn.ReLU()       
        self.batchnorm = torch.nn.BatchNorm2d(d*N1, track_running_stats=False)       
        self.GAP=torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))         
        self.fc = torch.nn.Linear(d*N1, classes) 
        self.FN = d*N1       

    def forward(self, inputdata):
        intermediate = self.pointwise(inputdata)        
        intermediate = self.depthwise(intermediate) 
        intermediate = self.activ(intermediate) 
        intermediate = self.batchnorm(intermediate)          
        latent = self.GAP(intermediate)
        output = latent.view(latent.size()[0], -1) 
        output = self.fc(output)    
        
        return latent, output

class ESTCNN(nn.Module):
    def __init__(self, n_classes=1, EEG_ch=30, fs=250, batch_norm_alpha=0.1, **kwargs):
        super(ESTCNN, self).__init__()
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        input_time = fs*3
        n_ch1, n_ch2, n_ch3 = 16, 32, 64
        self.FN = 50

        self.convnet = nn.Sequential(
            nn.Conv2d(1, n_ch1, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch1, n_ch1, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch1, n_ch1, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch2, n_ch2, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch2, n_ch2, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch3, n_ch3, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch3, n_ch3, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 7)),
        )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, EEG_ch, input_time))

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.spatial_fusion = nn.Sequential(nn.Linear(self.n_outputs, 50),
                                            nn.ReLU()
                                            )

        """ Classifier """
        self.clf = nn.Sequential(nn.Linear(50, self.n_classes),
                                 nn.Sigmoid()
                                 )
    def forward(self, x):
        intermediate = self.convnet(x)
        intermediate = intermediate.view(intermediate.size()[0], -1)
        latent = self.spatial_fusion(intermediate)
        output = self.clf(latent)
        
        return latent.unsqueeze(2).unsqueeze(3), output


class EEGNet(nn.Module):
    def __init__(self, EEG_ch=30, **kwargs):
        super(EEGNet, self).__init__()

        self.F1 = 16
        self.FN = 32

        self.activation = nn.ELU()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 125), stride=(1,1), padding=(0,62), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=self.F1, out_channels=self.FN, kernel_size=(EEG_ch, 1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(self.FN),
            self.activation,
            nn.AvgPool2d((1,4), stride=(1,4), padding=0),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=self.FN, out_channels=self.FN, kernel_size=(1, 31), stride=(1,1), padding = (0,15), bias=False),
            nn.BatchNorm2d(self.FN),
            self.activation,
            nn.AvgPool2d((1,187), stride=(1,1), padding=0),
            # nn.AvgPool2d((1,8), stride=(1,8), padding=0),
            nn.Dropout(0.25)
        )

        self.regressor = nn.Linear(self.FN, 1, bias=True)

    def forward(self, x):

        x = self.conv_block1(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        latent = x.view(-1, 32)
        out = self.regressor(latent)

        return x, out

class ShallowConvNet(nn.Module):
    def __init__(self, EEG_ch=30, **kwargs):
        super(ShallowConvNet, self).__init__()

        self.F1 = 40
        self.FN = 40

        self.conv1 = nn.Conv2d(1, self.F1, (1, 25), bias=False)
        self.conv2 = nn.Conv2d(self.F1, self.FN, (EEG_ch, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(self.FN)

        self.AvgPool1 = nn.AvgPool2d((1, 726), stride=(1, 1))
    
        self.Drop1 = nn.Dropout(0.25)
        self.regressor = nn.Linear(self.FN, 1, bias=True)

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
    def __init__(self, EEG_ch=30, fs=250, **kwargs):
        super(SCCNet, self).__init__()

        # structure parameters
        self.F1 = 22
        self.FN = 20
        self.t1 = fs // 10
        self.t2 = fs * 3

        # temporal and spatial filter
        self.Conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(EEG_ch, 1)),
            # nn.BatchNorm2d(self.F1)
        )
        self.Conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.FN, kernel_size=(self.F1, self.t1), padding=(0, self.t1//2)),
            # nn.BatchNorm2d(self.FN)
        )

        self.AveragePooling1 = nn.AvgPool2d((1, self.t2))
        # self.AveragePooling1 = nn.AvgPool2d((1,125), stride = (1,25))
        # stride is set as 25 (0.1 sec correspond to time domain)
        # kernel size 125 mean 0.5 sec
        self.dropout = nn.Dropout(0.5)
        self.regressor = nn.Linear(self.FN, 1, bias=True)
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

        x = x.permute(0,2,1,3)
        x = self.AveragePooling1(x)
        latent = self.log_activation(x)
   
        x= torch.flatten(latent, 1)
        x = self.regressor(x)
        output = self.sigmoid(x)

        return latent, output

### EEGTCNet ###
class CausalConv1d(nn.Module):
    """
    A causal 1D convolution. (to force no information flow from the future to the past)
    """
    def __init__(self, kernel_size, in_channels, out_channels, dilation):
        super(CausalConv1d, self).__init__()
        
        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.dilation = dilation
        
        # modules:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=1,
                                      padding=(kernel_size-1)*dilation,
                                      dilation=dilation)

    def forward(self, seq):
        """
        Note that Conv1d expects (batch, in_channels, in_length).
        We assume that seq ~ (len(seq), batch, in_channels), so we'll reshape it first.
        """       
        # seq = torch.squeeze(seq)
        seq_ = seq.squeeze(2)
        conv1d_out = self.conv1d(seq_)
        
        # remove k-1 values from the end: (remove the information from future)
        conv1d_out = conv1d_out.permute(2, 1, 0)
        conv1d_out = conv1d_out[0:-(self.kernel_size-1)*self.dilation].permute(2,1,0)
        
        conv1d_out = torch.unsqueeze(conv1d_out, 2)
        return conv1d_out

class TCNBlock(nn.Module):
    """
    Second block of the proposed model.
    """
    def __init__(self, F2=16, dilate_rate=1, KT=4, FT=12, pt=0.3):
        super(TCNBlock, self).__init__()

        self.F2 = F2  # number of pointwise filters
        self.dilate_rate = dilate_rate

        self.KT = KT  # kernal size of the first conv
        self.FT = FT  # number of convolution filters in TCN block
        self.pt = pt  # dropout rate

        self.dilated_causal_conv = nn.Sequential(
            CausalConv1d(self.KT, self.F2, self.FT, dilation=1),
            CausalConv1d(self.KT, self.FT, self.FT, dilation=2),
            CausalConv1d(self.KT, self.FT, self.FT, dilation=4),
            CausalConv1d(self.KT, self.FT, self.FT, dilation=8),
            nn.BatchNorm2d(self.FT),
            nn.ELU(),
            nn.Dropout(self.pt),

            CausalConv1d(self.KT, self.FT, self.FT, dilation=1),
            CausalConv1d(self.KT, self.FT, self.FT, dilation=2),
            CausalConv1d(self.KT, self.FT, self.FT, dilation=4),
            CausalConv1d(self.KT, self.FT, self.FT, dilation=8),
            nn.BatchNorm2d(self.FT),
            nn.ELU(),
            nn.Dropout(self.pt)
        )

        self.conv1d = nn.Conv1d(self.F2, self.FT, 1, padding='same')

    def forward(self, x):

        y = self.dilated_causal_conv(x)
            
        if self.F2 != self.FT: # not the end of the block2 (still having other residual blocks to go) 
            # x = torch.squeeze(x)
            x = x.squeeze(2)
            conv = self.conv1d(x)
            conv = torch.unsqueeze(conv, 2)
            add = y + conv     # => concate with feature after processing with conv1d
        else:                  # at the end of block2           
            add = y + x        # concat with previous feature map


        return add

class EEGBlock4EEGTCN(nn.Module):
    """
    First block of the proposed model. (temporal conv. > depth-wise conv. > separable conv.)
    """
    def __init__(self, EEG_ch=30, F1 = 8, F2 = 16, D = 2, KE = 32, pe = 0.3):
        super(EEGBlock4EEGTCN, self).__init__()

        self.F1 = F1  # number of temporal filters
        self.F2 = F2  # number of pointwise filters
        self.D = D    # depth multiplier

        self.KE = KE  # kernal size of the first conv
        self.pe = pe  # dropout rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.KE), padding='same', bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.depwise = nn.Conv2d(self.F1, self.D*self.F1, (EEG_ch, 1), padding='valid', groups=self.F1, bias=False)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.pe)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.pe)
        )

    def forward(self, x):
        x1 = self.conv1(x)     # temporal conv.
        x1 = self.depwise(x1)  # depthwise conv.
        x2 = self.conv2(x1)    # separable conv. (go to block3 directly)
        x3 = self.conv3(x2)    # separable conv. (need to go through block2)
        return x3

class EEGTCNet(nn.Module):
    def __init__(self, EEG_ch=30, **kwargs):
        super().__init__()
        self.F1 = 8   # number of temporal filters
        self.F2 = 16  # number of pointwise filters
        self.FN = 12  # number of convolution filters in TCN block
        self.D = 2    # depth multiplier
        
        # block1
        self.block1 = EEGBlock4EEGTCN(EEG_ch, pe=0.3) 
        modules = []
        for i in range(self.D):
            F2 = self.F2 if i == 0 else self.FN
            modules.append(TCNBlock(F2=F2, dilate_rate=2**i, pt=0.3))
        
        self.block2 = nn.Sequential(*modules)
        self.AvgPooling = nn.AvgPool2d((1, 23))
        self.flat = nn.Flatten()
        # block3
        self.classifier = nn.Linear(self.FN, 1, bias=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) # (B, 12, 1, 23)
        latent = self.AvgPooling(x) 
        x = self.flat(latent)
        output = self.classifier(x)
        return latent, output
### EEGTCNet ###

### MBEEGSE ###
class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EEGNet_Block(nn.Module):
    def __init__(self, EEG_ch, F1=8, F2=16, D=1, ks=2, dropout=0):
        super(EEGNet_Block, self).__init__()

        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_size = ks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_size),
                      stride=(1, 1), padding=(0, self.kernel_size//2), bias=False),
            nn.BatchNorm2d(self.F1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (EEG_ch, 1),
                      stride=(1, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, kernel_size=(1, 16),
                      padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2, eps=1e-5, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8), padding=0),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class MBEEGSE(nn.Module):

    def __init__(self, EEG_ch=30, **kwargs):
        super(MBEEGSE, self).__init__()

        self.FN = 12

        self.b1_F1 = 4  # Number of temporal filters
        self.b1_F2 = 16
        self.b1_D = 1
        self.b1_ks = 16
        self.b1_dropout = 0

        self.b2_F1 = 8  # Number of temporal filters
        self.b2_F2 = 16
        self.b2_D = 1
        self.b2_ks = 32
        self.b2_dropout = 0.1

        self.b3_F1 = 16  # Number of temporal filters
        self.b3_F2 = 16
        self.b3_D = 1
        self.b3_ks = 64
        self.b3_dropout = 0.2

        # Reduction ratio
        reduction1 = 4
        reduction2 = 4
        reduction3 = 2

        self.eegnet1 = EEGNet_Block(
            EEG_ch, self.b1_F1, self.b1_F2, self.b1_D, self.b1_ks, self.b1_dropout)
        self.eegnet2 = EEGNet_Block(
            EEG_ch, self.b2_F1, self.b2_F2, self.b2_D, self.b2_ks, self.b2_dropout)
        self.eegnet3 = EEGNet_Block(
            EEG_ch, self.b3_F1, self.b3_F2, self.b3_D, self.b3_ks, self.b3_dropout)

        self.se1 = SE_Block(self.b1_F2, reduction1)
        self.se2 = SE_Block(self.b2_F2, reduction2)
        self.se3 = SE_Block(self.b3_F2, reduction3)

        self.flat1 = nn.Flatten()
        self.flat2 = nn.Flatten()
        self.flat3 = nn.Flatten()

        self.b1_classifier = nn.Linear(368, 4, bias=True)
        self.b2_classifier = nn.Linear(368, 4, bias=True)
        self.b3_classifier = nn.Linear(368, 4, bias=True)

        self.classifier = nn.Linear(3*4, 1, bias=True)

    def forward(self, x):
        x1 = self.eegnet1(x)
        x1 = self.se1(x1)

        x2 = self.eegnet2(x)
        x2 = self.se2(x2)

        x3 = self.eegnet3(x)
        x3 = self.se3(x3)

        x1 = self.flat1(x1)
        x2 = self.flat1(x2)
        x3 = self.flat1(x3)

        x1 = self.b1_classifier(x1)
        x2 = self.b2_classifier(x2)
        x3 = self.b3_classifier(x3)
        latent = torch.cat((x1, x2, x3), dim=1)

        output = self.classifier(latent)

        return latent.unsqueeze(2).unsqueeze(3), output
### MBEEGSE ###

### FBCNet ###

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm, padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, 
                EEG_ch=30, 
                nClass = 1, 
                nBands = 8, 
                m = 8, 
                strideFactor= 1, 
                doWeightNorm = True,  
                fs=250, 
                *args, **kwargs):
        super(FBCNet, self).__init__()
        # self.filtBank = filtBank
        self.fs = fs
        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor
        
        # create all the parrallel SCBc
        self.scb = self.SCB(m, EEG_ch, self.nBands, doWeightNorm = doWeightNorm)
        
        # Formulate the temporal agreegator
        self.temporalLayer = LogVarLayer(dim = 3)
        # self.temporalLayer = current_module.__dict__[temporalLayer](dim = 3)
        self.FN = m * nBands

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):
        # x.unsqueeze_(1)
        x = torch.squeeze(x.permute((0,4,2,3,1)), dim = 4)
        # print(x.shape, '1')
        x = self.scb(x)
        # print("After SCB: ", x.size())
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        # print("After reshape: ", x.size())
        latent = self.temporalLayer(x)
        x = torch.flatten(latent, start_dim= 1)
        x = self.lastLayer(x)
        return latent, x
### FBCNet ###
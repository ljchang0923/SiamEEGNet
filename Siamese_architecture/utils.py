import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import scipy
from math import sqrt,isnan
import json
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_json(filepath):
    if filepath is None:
        return None
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content 


def create_multi_window_input(filename, low_bound, cfg):
    sm_num = cfg['num_window'] - 1
    filepath = cfg['data_dir'] + filename

    if cfg['EEG_ch'] == 8:
        ch_num = [0, 1, 4, 7, 11, 24, 27, 29]  ## default 8-ch: [0, 1, 4, 7, 11, 24, 27, 29]
    elif cfg['EEG_ch'] == 4:
        ch_num = [0, 1, 27, 29] 
    elif cfg['EEG_ch'] == 2:
        ch_num = [27, 29] 
    elif cfg['EEG_ch'] == 30:
        ch_num = np.arange(0,30)
    else:
        raise ValueError('invalid EEG channel number')

    session = scipy.io.loadmat(filepath)
    session = session['session'][0,0]
    data = np.transpose(session['eeg_trial'],(2,0,1)) ## Size of raw data: (trials, ch, time)

    ## revise data to extract the channels we wanted
    data = data[:, ch_num, :]
    truth = session['DI'][0][sm_num:]
    onset_time = session['onset_time'][0][sm_num:]
    
    # create multi-window input
    multi_win = []
    session_boundary = []
    low = low_bound

    for i in range(sm_num, data.shape[0]):
        multi_win.append(data[i-sm_num:i+1, :, :])
        session_boundary.append([low, low+data.shape[0] - (sm_num+1)])
    
    low += data.shape[0] - (sm_num+1)  + 1
    return multi_win, truth, session_boundary, onset_time, low

class grad_accumulator:
    def __init__(self, thres_alert, thres_drowsy, cfg):

        self.num_win = cfg["num_window"]
        self.thres_alert = thres_alert
        self.thres_drowsy = thres_drowsy
        self.grad_acc = {}
        self.grad_acc["drowsy"] = torch.zeros((self.num_win, cfg["EEG_ch"], 750))
        self.grad_acc["all"] = torch.zeros((self.num_win, cfg["EEG_ch"], 750))
        self.grad_acc["alert"] = torch.zeros((self.num_win, cfg["EEG_ch"], 750))

    def update(self, grad, y_train):
        self.grad_acc["all"] += torch.sum(grad[:, self.num_win:, :].cpu(), 0)

        drowsy_grad = grad[(y_train[:, 0] >= self.thres_drowsy).view(-1)]
        drowsy_grad = drowsy_grad[:, self.num_win:, :, :]
        self.grad_acc["drowsy"] += torch.sum(drowsy_grad.cpu(), 0)
        
        alert_grad = grad[(y_train[:, 0] <= self.thres_alert).view(-1)]
        alert_grad = alert_grad[:, self.num_win:, :, :]
        self.grad_acc["alert"] += torch.sum(alert_grad.cpu(), 0)



"""# Training setup"""

def train_model(model, train_dl, test_dl, device, cfg, thres_alert, thres_drowsy):
    optimizer = getattr(optim, cfg['optimizer'])(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = nn.MSELoss(reduction='mean')

    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    record = {'train loss': [], 'val loss':[]}
    total_loss = 0
    mini = 1e8
    best_rmse = 0
    best_cc = 0

    grad = grad_accumulator(thres_alert, thres_drowsy, cfg)
    
    for epoch in range(cfg['epoch']):
        model.train()
        print(f"[{epoch+1}/{cfg['epoch']}]")
        with tqdm(train_dl, unit="batch") as tepoch:
            for b, (x_train, y_train) in enumerate(tepoch):
                
                optimizer.zero_grad()
                    
                x_train, y_train = torch.flatten(x_train, 0, 1).to(device), torch.flatten(y_train, 0,1).to(device)
                x_train.requires_grad = True
                _, di, delta_di = model(x_train)

                truth = y_train[:, 0] - y_train[:, 1]
                loss = criterion(delta_di, truth)
                total_loss += loss.detach().cpu().item()

                loss.backward(retain_graph=True)

                grad.update(x_train.grad, y_train)

                optimizer.step()
                
                tepoch.set_postfix(loss = total_loss/(b+1))
                tepoch.update(1)

            val_loss, rmse, cc = val_model(model, test_dl, device)
            record["train loss"].append(total_loss/len(train_dl.dataset))
            record["val loss"].append(val_loss)

            print(f"val loss-> {val_loss} rmse -> {rmse} cc -> {cc}")
            matrice = 0.5* rmse + 0.5*(1-cc)
            if(matrice < mini and not isnan(cc)):
                mini = matrice
                best_rmse = rmse
                best_cc = cc
                model_save_path = f'{cfg["model_dir"]}{cfg["ts_sub"]}_model.pt'
                torch.save(model.state_dict(), model_save_path)
            
            total_loss = 0

    torch.cuda.empty_cache()
    return record, grad.grad_acc

def val_model(model, test_dl, device):
    criterion_mse = nn.MSELoss(reduction='mean')
    total_loss = 0
    output = []

    with torch.no_grad():
        model.eval()
        for x_test, y_test in test_dl:

            x_test, y_test = torch.flatten(x_test, 0, 1).to(device), torch.flatten(y_test, 0, 1).to(device)
            _, di, delta_di = model(x_test)
            
            mse = criterion_mse(delta_di, y_test[:,0]-y_test[:,1])
            rmse = sqrt(mse)
            cc = np.corrcoef((y_test[:,0]-y_test[:,1]).cpu().detach().numpy().reshape(-1), delta_di.cpu().detach().numpy().reshape(-1))

    return mse, rmse, cc[0,1] 

def test_model(model, test_dl, device):
    model.eval()
    output = []
    latent_feat = []

    with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test, y_test = torch.flatten(x_test, 0,1).to(device), torch.flatten(y_test, 0,1).to(device)
            b_latent, latent, delta_di = model(x_test)
            output.append(delta_di)

    return output

def plot_result(output, test_truth, time_point, cfg, idx=None):
    
    plt.figure(figsize=(10, 5))
    plt.rc('font', size=12)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', labelsize=12)
    hfont = {'fontname':'Helvetica'}
    plt.plot(time_point/60, np.abs(output), 'r', linewidth=0.8, alpha=0.8)
    plt.plot(time_point/60, test_truth.reshape(-1), 'k', linewidth=0.8)
    plt.xlabel('Time(min)', **hfont)
    plt.ylabel('Delta DI', **hfont)
    plt.legend(['Prediction', 'True delta DI'])

    if(idx == None):
        plt.savefig(cfg['fig_dir'] + f'{cfg["ts_sub"]}.png')
    else:
        plt.savefig(cfg['fig_dir'] + f'{cfg["ts_sub"]}-{idx+1}.png')
    plt.clf()

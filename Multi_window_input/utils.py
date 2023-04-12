import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from math import sqrt
import json

BATCH_SIZE = 100

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

def train_model(model, train_dl, test_dl, device, cfg):
    optimizer = getattr(optim, cfg['optimizer'])(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion_mse = nn.MSELoss(reduction='mean')
    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    record = {'train loss': [], 'val loss':[]}
    total_loss = 0
    mini = 1e8
    best_rmse = 0
    best_cc = 0
    
    for epoch in range(config['epoch']):
        model.train()
        for x_train, y_train in train_dl:
        
            optimizer.zero_grad()

            x_train, y_train = x_train.to(device), y_train.to(device)
            _, di = model(x_train)
            loss = criterion(di, y_train)
            total_loss += loss.detach().cpu().item()

            loss.backward()
            optimizer.step()
            #lr_scheduler.step()

        val_loss, rmse, cc = val_model(model, test_dl, device)
        record["train loss"].append(total_loss/len(train_dl))
        record["val loss"].append(val_loss)

        print(f"{epoch+1} epoch: train loss-> {total_loss/len(train_dl)}")
        print(f"val loss-> {val_loss} rmse -> {rmse} cc -> {cc}")
        matrice = 0.5 * rmse + 0.5*(1-cc)
        if(matrice < mini):
            mini = matrice
            best_rmse = rmse
            best_cc = cc
            model_save_path = f'{cfg["model_dir"]}{cfg["ts_filename"]}_model.pt'
            torch.save(model.state_dict(), model_save_path)
        
        total_loss = 0
    torch.cuda.empty_cache()
  
    return record

def val_model(model, test_dl, device):
    criterion_mse = nn.MSELoss(reduction='mean')
    total_loss = 0
    output = []

    with torch.no_grad():
        model.eval()
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device)
            _, di = model(x_test)
            loss = criterion(di, y_test)
            mse = criterion_mse(di, y_test)
            rmse = sqrt(mse)
            cc = np.corrcoef(y_test.cpu().detach().numpy().reshape(-1), di.cpu().detach().numpy().reshape(-1))

    return mse, rmse, cc[0,1] 

def test_model(model, test_dl, device):
    model.eval()
    output = []

    with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device)
            _, di = model(x_test)
            output.append(di)
            
    return output

def individual_dataloader(train_data, train_truth, device, mode):

    x_train = torch.Tensor(train_data).to(device)
    y_train = torch.Tensor(train_truth).to(device).view(-1,1)
    train_dataset = Dataloader(x_train, y_train)
    if mode == 'train':
        dl = DataLoader(
            dataset = train_dataset,
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 4,
            pin_memory = True
        )
    elif mode == 'test':
        dl = DataLoader(
            dataset = train_dataset,
            batch_size = len(train_dataset),
            shuffle = False,
            num_workers = 0
        )
    return dl


def single_trial_input(data_path, filename):
    filepath = data_path + filename
    session = scipy.io.loadmat(filepath)
        
    session = session['session'][0,0]
    onset_time = session['onset_time'][0]
    data = np.transpose(session['eeg_trial'],(2,0,1))
    truth = session['DI'][0]

    return data, truth, onset_time

def create_multi_window_input(filename, low_bound, args):
    sm_num = args.num_smooth - 1
    filepath = args.data_dir + filename
    session = scipy.io.loadmat(filepath)
    session = session['session'][0,0]
    data = np.transpose(session['eeg_trial'],(2,0,1))
    truth = session['DI'][0][sm_num:]
    onset_time = session['onset_time'][0][sm_num:]
    
    multi_win = []
    session_boundary = []
    low = low_bound
    sm_num = args.num_smooth - 1
    for i in range(sm_num, data.shape[0]):
        multi_win.append(data[i-sm_num:i+1, :, :])
        session_boundary.append([low, low+data.shape[0]-10])
    
    low = low + data.shape[0] -10 + 1
    return multi_win, truth, session_boundary, onset_time, low


def plot_result(output, test_truth, time_point, cfg, idx):
    
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

    fig_dir = f'fig_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + f'{cfg["ts_sub"]}_{idx}.png')
    plt.clf()


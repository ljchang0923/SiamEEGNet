import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import random

BATCH_SIZE = 100

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def get_dataloader(data, truth, smooth, mode, device):

    if smooth:
        x = torch.Tensor(data).to(device)
    else:
        x = torch.Tensor(data).unsqueeze(1)

    y = torch.Tensor(truth).view(-1,1)
  
    # y_train = torch.Tensor(truth).to(device)
    # y_test = torch.Tensor(test_truth).to(device)

    print(x.size())
    print(y.size())

    # 存成tensordataset
    dataset = TensorDataset(x, y)

    # 包成dataloader
    if mode == 'train':
        dl = DataLoader(
            dataset = dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 4,
            pin_memory = True
        )
    elif mode == 'test':
        dl = DataLoader(
            dataset = dataset,
            batch_size = len(dataset),
            shuffle = False,
            num_workers = 4,
            pin_memory = True
        )

    return dl

def plot_result(output, test_truth, time_point, info):
    plt.figure(figsize=(10, 4))
    plt.rc('font', size=12)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', labelsize=12)
    plt.plot(time_point/60, output, 'r', linewidth=0.8, alpha=0.8)
    plt.plot(time_point/60, test_truth.reshape(-1), 'k', linewidth=0.8)
    plt.xlabel('Time(min)')
    plt.ylabel('Drownsiness Index(DI)')
    plt.legend(['Prediction', 'True delta DI'])
    plt.savefig(f'{info["save_path"]}/{info["filename"]}.png')
    plt.clf()

class Dataloader(Dataset):
    def __init__(self, data, truth):
        self.data = data
        self.truth = truth

    def __len__(self):
        return len(self.data)		

    def __getitem__(self, index):
        x = self.data[index,:,:,:]
        y = self.truth[index]

        return x, y

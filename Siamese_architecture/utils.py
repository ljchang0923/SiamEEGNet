import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy import io
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


def create_multi_window_input(file_path, num_window=10, EEG_ch=30):
    start_idx = num_window - 1

    if EEG_ch == 8:
        ch_num = [0, 1, 4, 7, 11, 24, 27, 29]  ## default 8-ch: [0, 1, 4, 7, 11, 24, 27, 29]
    elif EEG_ch == 4:
        ch_num = [0, 1, 27, 29] 
    elif EEG_ch == 2:
        ch_num = [27, 29] 
    elif EEG_ch == 30:
        ch_num = np.arange(0,30)
    else:
        raise ValueError('invalid EEG channel number')

    session = io.loadmat(file_path)
    session = session['session'][0,0]
    data = np.transpose(session['eeg_trial'],(2,0,1)) ## Size of raw data: (trials, ch, time)

    ## revise data to extract the channels we wanted
    data = data[:, ch_num, :]
    truth = session['DI'][0][start_idx:]
    onset_time = session['onset_time'][0][start_idx:]
    
    # create multi-window input
    multi_win = []
    for i in range(start_idx, data.shape[0]):
        multi_win.append(data[i-start_idx:i+1, :, :])
    
    return multi_win, truth, onset_time



def plot_result(output, test_truth, time_point, fig_dir, cfg, idx=None):
    
    plt.figure(figsize=(10, 5))
    plt.rc('font', size=12)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', labelsize=12)
    hfont = {'fontname':'Helvetica'}
    plt.plot(time_point/60, output, 'r', linewidth=0.8, alpha=0.8)
    plt.plot(time_point/60, test_truth.reshape(-1), 'k', linewidth=0.8)
    plt.xlabel('Time(min)', **hfont)
    plt.ylabel('Delta DI', **hfont)
    plt.legend(['Prediction', 'True delta DI'])
    plt.xlim([0, time_point[-1]/60])
    plt.ylim([0, 1])
    plt.tight_layout()

    if(idx == None):
        plt.savefig(fig_dir + f'{cfg["ts_sub"]}.png')
    else:
        plt.savefig(fig_dir + f'{cfg["ts_sub"]}-{idx+1}.png')
    plt.clf()

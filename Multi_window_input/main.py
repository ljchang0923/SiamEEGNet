"""# Import libary and set random seed and gup """
import time
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import os
import argparse

from DataLoader import dataloader
from utils import read_json, plot_result, create_multi_window_input, train_model, test_model
from models import Multi_window_input

def train_within_subject(cfg, save_path):
    filelist = sorted(os.listdir(save_path['data_dir']))

    # Load all data
    sub_list = {}
    data = {}
    truth = {}
    onset_time = {}
    print("Loading data...")
    for filename in filelist:
        if not filename.endswith('.mat'):
            continue

        file_path = save_path['data_dir'] + filename
        single_train_data, single_train_truth, single_onset_time = create_multi_window_input(file_path, cfg['num_window'], cfg['EEG_ch'])
        if(filename[:3] not in sub_list):
            sub_list[filename[:3]] = []
            data[filename[:3]] = {}
            truth[filename[:3]] = {}
            onset_time[filename[:3]] = {}

        sub_list[filename[:3]].append(filename[:-4])
        data[filename[:3]][filename[:-4]] = single_train_data
        truth[filename[:3]][filename[:-4]] = single_train_truth
        onset_time[filename[:3]][filename[:-4]] = single_onset_time

    for sub, sess in sub_list.items():
        for train_sess in sess:

            cfg["ts_sub"] = train_sess
            train_data = np.array(data[sub][train_sess], dtype=np.float32)
            train_truth = truth[sub][train_sess].astype('float32')
            tr_session_bound = np.tile([0, train_data.shape[0] - 1], (train_data.shape[0], 1))

            thres_drowsy = np.quantile(train_truth, 0.85)
            thres_alert = np.quantile(train_truth, 0.15)
            thres = {
                    'drowsy': thres_drowsy,
                    'alert': thres_alert
                    }

            x_train, x_val, y_train, y_val = train_test_split(train_data, train_truth, test_size=0.3, shuffle=True)

            train_dl = dataloader(x_train, y_train, 0, 'train', cfg)
            val_dl = dataloader(x_val, y_val, 0, 'test', cfg)

            print("Train size: ", x_train.shape)
            print("Val size: ", x_val.shape)
            print("Training: ", train_sess)
            model = Multi_window_input(**cfg).to(cfg['device'])
            loss_record = train_model(model, train_dl, val_dl, cfg, save_path['model_dir']) 

            del train_data, train_truth, tr_session_bound
            del train_dl, val_dl
            del model

def test_within_subject(cfg, save_path):
    filelist = sorted(os.listdir(save_path['data_dir']))

    for filename in filelist:
        if not filename.endswith('.mat'):
            continue

        file_path = save_path['data_dir'] + filename
        test_data, test_truth, onset_time = create_multi_window_input(file_path, cfg['num_window'], cfg['EEG_ch'])

        baseline_idx = 0    
        cfg["ts_sub"] = filename[:-4]
        test_data = np.array(test_data, dtype=np.float32)
        test_truth = test_truth.astype('float32')
        ts_session_bound = np.tile([0, test_data.shape[0] - 1], (test_data.shape[0], 1))

        print("data size: ", test_data.shape)
        print("truth shape: ", test_truth.shape)

        test_dl = dataloader(test_data, test_truth, ts_session_bound, 'test', cfg)

        print("testing session: ", filename[:-4])
        pred_set = []
        for name in filelist:
            if name.endswith('.mat') and name != filename and name[:4] == filename[:4]:
                model_path = name[:-4] +'_model.pt'
                model = Multi_window_input(**cfg).to(cfg['device'])
                model.load_state_dict(torch.load(save_path['model_dir'] + model_path))
                print(model_path)
                pred = test_model(model, test_dl, cfg['device'])
                pred_set.append(pred[0])
      
        pred_set = torch.cat(pred_set, 1)
        print(pred_set.size())
        pred = torch.mean(pred_set, 1)
        output = [tensor.detach().cpu().item() for tensor in pred]

        rmse = sqrt(mean_squared_error((test_truth-test_truth[baseline_idx]).reshape(-1), output))
        cc = np.corrcoef((test_truth-test_truth[baseline_idx]).reshape(-1), output)
        print('test on model ' + model_path)
        print('RMSE: ', rmse, ' CC:', cc[0,1])
        
        plot_result(output, (test_truth-test_truth[baseline_idx]).reshape(-1), onset_time, save_path['fig_dir'], cfg)
        with open(save_path['log_file'], 'a') as f:
            f.writelines('%s\t%.3f\t%.3f\n'%(filename[:-4], rmse, cc[0,1]))

def model_selection():
    device = get_device()
    set_seed(34)
    data_path = '/mnt/left/jason/dataset/individual/'
    filelist = os.listdir(data_path)
    filelist = sorted(filelist)
    for filename in filelist:
        if filename.endswith('.mat'):
            filepath = data_path + filename
            session = scipy.io.loadmat(filepath)
    
        session = session['session'][0,0]
        test_data = np.transpose(session['eeg_trial'],(2,0,1))
        test_truth = session['DI'][0][9]
        baseline = torch.Tensor(test_data[:10])
        baseline = torch.cat([baseline, baseline],0).unsqueeze(0).to(device)
        print("baseline shape: ", baseline.size())
        for name in filelist:
            if name.endswith('.mat') and name!=filename:
                model_path = name[:-4] +'_model.pt'
                model = SCC_delta().to(device)
                model.load_state_dict(torch.load(f"indv_model_v2/{model_path}"))
                bs_pred, delta_pred = model(baseline)
                with open(f"{session[0][0]}_select.txt", 'a') as f:
                    f.writelines(f'{model_path[:-9]}\t{abs(bs_pred.detach().cpu().item()-test_truth)}\t{abs(delta_pred.detach().cpu().item())}\n')

def train_cross_subject(cfg, save_path):
    filelist = sorted(os.listdir(save_path['data_dir']))

    # Load all data
    sub_list = []
    data = {}
    truth = {}
    onset_time = {}
    print("Loading data...")
    for filename in filelist:
        if not filename.endswith('.mat'):
            continue
        
        file_path = save_path['data_dir'] + filename
        single_train_data, single_train_truth, single_onset_time = create_multi_window_input(file_path, cfg['num_window'], cfg['EEG_ch'])
            
        if(filename[:3] not in sub_list):
            sub_list.append(filename[:3])
            data[filename[:3]] = []
            truth[filename[:3]] = []
            onset_time[filename[:3]] = []

        data[filename[:3]].append(single_train_data)
        truth[filename[:3]].append(single_train_truth)
        onset_time[filename[:3]].append(single_onset_time)

    # train the model for all subject iteratively
    for ts_sub_idx in range(len(sub_list)):
        # testing data 
        cfg['ts_sub'] = sub_list[ts_sub_idx]
    
        # validation data
        if ts_sub_idx == len(sub_list) - 1:
            val_sub_idx = 2
        else:
            val_sub_idx = ts_sub_idx + 1
        
        val_data = np.array(data[sub_list[val_sub_idx]][0], dtype=np.float32)
        val_truth = truth[sub_list[val_sub_idx]][0].astype('float32')

        # training data
        train_data = []
        train_truth = []
        for tr_sub_idx in range(len(sub_list)):
            if tr_sub_idx == ts_sub_idx or tr_sub_idx == val_sub_idx:
                continue

            for idx in range(len(data[sub_list[tr_sub_idx]])):
                train_data = train_data + data[sub_list[tr_sub_idx]][idx]
                train_truth.append(truth[sub_list[tr_sub_idx]][idx].astype('float32'))

        train_data = np.array(train_data, dtype=np.float32) # (#total training trial, #window, #channel, #timepoint)
        train_truth = np.concatenate(train_truth, 0) # (#total training trial, )

        # wrap up to the Dataloader
        train_dl = dataloader(train_data, train_truth, 'train', cfg)
        val_dl = dataloader(val_data, val_truth, 'test', cfg)

        ''' Model setup and training '''
        model = Multi_window_input(**cfg).to(cfg['device'])
        print ('validate on: ', sub_list[val_sub_idx])
        print('test on: ', cfg['ts_sub'])
        print('Start training...')

        _ = train_model(model, train_dl, val_dl, cfg, save_path['model_dir'])

        ''' Testing '''
        for idx in range(len(data[sub_list[ts_sub_idx]])):
            test_data = np.array(data[sub_list[ts_sub_idx]][idx], dtype=np.float32) # (#testing trial, #window, #channel, #timepoint)
            test_truth = truth[sub_list[ts_sub_idx]][idx].astype('float32') # (#testing trial, )
            ts_onset_time = onset_time[sub_list[ts_sub_idx]][idx]

            test_dl = dataloader(test_data, test_truth, 'test', cfg)
            pred = test_model(model, test_dl, device)
            output = [tensor.detach().cpu().item() for tensor in pred[0]]
            
            rmse = sqrt(mean_squared_error(test_truth.reshape(-1), output))
            cc = np.corrcoef(test_truth.reshape(-1), output)
        
            plot_result(output, test_truth.reshape(-1), ts_onset_time, save_path['fig_dir'], cfg, idx)
            with open(save_path['log_file'], 'a') as f:
                f.writelines('%s\t%.3f\t%.3f\n'%(f"{cfg['ts_sub']}-{idx+1}", rmse, cc[0,1]))

        del train_data, train_truth, test_data, test_truth,  val_data, val_truth
        del train_dl, test_dl, val_dl

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=str, help="path to configuration file", default='config/config.json')
    parser.add_argument("--scenario", type=str, help="within_subject or cross_subject", default="cross_subject")
    parser.add_argument("--device", type=str, default = 'cuda:0')
    args = parser.parse_args()

    cfg = read_json(args.config_path)
    cfg['device'] = args.device
    cfg['scenario'] = args.scenario

    data_dir = '/home/cecnl/ljchang/CECNL/sustained-attention/selected_data/'
    model_dir = f'/home/cecnl/ljchang/CECNL/sustained-attention/model/siamese{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'
    log_file = f'log/multi_window_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["scenario"]}_{cfg["EEG_ch"]}ch'
    fig_dir = f'fig/multi_window_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    save_path ={
                "data_dir": data_dir,
                "model_dir": model_dir,
                "fig_dir": fig_dir,
                "log_file": log_file
                }

    print(f'Backbone: {cfg["backbone"]}')

    if args.scenario == 'cross_subject':
        for i in range(5):
            train_cross_subject(cfg, save_path)
    elif args.scenario == 'within_subject':
        for i in range(5):
            train_within_subject(cfg, save_path)
            test_within_subject(cfg, save_path)
    else:
        raise ValueError('Invalid scenario')

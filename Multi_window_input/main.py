"""# Import libary and set random seed and gup """
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import os
import argparse
import csv

from DataLoader import dataloader
from utils import read_json, plot_result, create_multi_window_input, train_model, test_model
from models import Multi_window_input

REPEAT = 3

def train_within_subject(cfg, save_path, *dataset):

    data, test_truth, onset_time = dataset

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

            train_dl = dataloader(x_train, y_train, 'train', cfg)
            val_dl = dataloader(x_val, y_val, 'test', cfg)

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
    record = []
    for filename in filelist:
        if not filename.endswith('.mat'):
            continue

        file_path = save_path['data_dir'] + filename
        test_data, test_truth, onset_time = create_multi_window_input(file_path,    cfg['num_window'], cfg['EEG_ch'])

        baseline_idx = 0    
        cfg["ts_sub"] = filename[:-4]
        test_data = np.array(test_data, dtype=np.float32)
        test_truth = test_truth.astype('float32')

        print("data size: ", test_data.shape)
        print("truth shape: ", test_truth.shape)

        test_dl = dataloader(test_data, test_truth, 'test', cfg)

        print("testing session: ", filename[:-4])
        pred_set = []
        for name in filelist:
            if name.endswith('.mat') and name != filename and name[:3] == filename[:3]:
                model_path = name[:-4] +'_model.pt'
                print("inference model: ", model_path)
                model = Multi_window_input(**cfg).to(cfg['device'])
                model.load_state_dict(torch.load(save_path['model_dir'] + model_path))
                pred = test_model(model, test_dl, cfg['device'])
                pred_set.append(pred)
      
        pred_set = torch.cat(pred_set, 1)
        print(pred_set.size())
        pred = torch.mean(pred_set, 1)
        output = [tensor.detach().cpu().item() for tensor in pred]

        rmse = sqrt(mean_squared_error((test_truth-test_truth[baseline_idx]).reshape(-1), output))
        cc = np.corrcoef((test_truth-test_truth[baseline_idx]).reshape(-1), output)
        print('test on model ' + model_path)
        print('RMSE: ', rmse, ' CC:', cc[0,1])
        
        plot_result(output, (test_truth-test_truth[baseline_idx]).reshape(-1), onset_time, save_path['fig_dir'], cfg)
        record.append([rmse, cc[0,1]])
        
        # with open(f'decoding_result/{cfg["ts_sub"]}.npy', 'wb') as f:
        #     np.save(f, np.array(output))
    return np.array(record)

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

def train_cross_subject(cfg, save_path, *dataset):
    data, truth, onset_time = dataset

    # train the model for all subject iteratively
    record = []
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
            pred = test_model(model, test_dl, cfg['device'])
            output = [tensor.detach().cpu().item() for tensor in pred]
            
            rmse = sqrt(mean_squared_error(test_truth.reshape(-1), output))
            cc = np.corrcoef(test_truth.reshape(-1), output)
            record.append([rmse, cc[0,1]])
            plot_result(output, test_truth.reshape(-1), ts_onset_time, save_path['fig_dir'], cfg, idx)

        del train_data, train_truth, test_data, test_truth,  val_data, val_truth
        del train_dl, test_dl, val_dl

    return np.array(record)

def test_cross_subject(cfg, save_path):

    filelist = sorted(os.listdir(save_path['data_dir']))

    # Load all data
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

    record = []
    for ts_sub_idx in range(len(sub_list)):
        for idx in range(len(data[sub_list[ts_sub_idx]])):

            test_data = np.array(data[sub_list[ts_sub_idx]][idx], dtype=np.float32) # (#testing trial, #window, #channel, #timepoint)
            test_truth = truth[sub_list[ts_sub_idx]][idx].astype('float32') # (#testing trial, )
            ts_session_bound = np.tile([0, test_data.shape[0]-1], (1, test_data.shape[0]))
            ts_onset_time = onset_time[sub_list[ts_sub_idx]][idx]
            test_dl = dataloader(test_data, test_truth, 'test', cfg)

            model = Multi_window_input(**cfg).to(cfg['device'])
            model_path = ts_sub_idx +'_model.pt'
            model.load_state_dict(torch.load(save_path['model_dir'] + model_path))
            _, pred = test_model(model, test_dl, cfg['device'])
            output = [tensor.detach().cpu().item() for tensor in pred]
            
            rmse = sqrt(mean_squared_error(test_truth.reshape(-1), output))
            cc = np.corrcoef(test_truth.reshape(-1), output)
            record.append([rmse, cc[0,1]])
        
            plot_result(output, test_truth.reshape(-1), ts_onset_time, cfg, idx)

    return np.array(record)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=str, help="path to configuration file", default='config/config.json')
    parser.add_argument("--scenario", type=str, help="within_subject or cross_subject", default="cross_subject")
    parser.add_argument("--device", type=str, default = 'cuda:0')
    parser.add_argument("--mode", type=str, default='train')
    args = parser.parse_args()

    cfg = read_json(args.config_path)
    cfg['device'] = args.device
    cfg['scenario'] = args.scenario

    save_path = {
        'data_dir':'/home/cecnl/ljchang/CECNL/sustained-attention/selected_data/',
        'model_dir':f'/home/cecnl/ljchang/CECNL/sustained-attention/model/multi_window_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/',
        'log_file':f'log/Baseline_method/{cfg["scenario"]}/multi_window_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["EEG_ch"]}ch.csv',
        'fig_dir':f'fig/multi_window_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'
    }
    
    if not os.path.exists(save_path['fig_dir']):
        os.makedirs(save_path['fig_dir'])

    if not os.path.exists(save_path['model_dir']):
        os.makedirs(save_path['model_dir'])

    # Load data
    print("Load data...")
    filelist = sorted(os.listdir(save_path['data_dir']))
    sub_list = []
    data, truth, onset_time = {}, {}, {}
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

    total_record = []
    if args.mode == 'train':
        for i in range(REPEAT):

            print('Repeatition: {}'.format(i+1))
            print(f'Backbone: {cfg["backbone"]}')

            if args.scenario == 'cross_subject':
                record = train_cross_subject(cfg, save_path, data, truth, onset_time)
                record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
                print('Repeatation {} {}'.format(i+1, record))
                total_record.append(record)

            elif args.scenario == 'within_subject':
                train_within_subject(cfg , save_path, data, truth, onset_time)
                record = test_within_subject(cfg, save_path)
                record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
                total_record.append(record)

            else:
                raise ValueError('Invalid scenario')

    elif args.mode == 'inference':
        if args.scenario == 'within_subject':
            trecord = est_within_subject(cfg, save_path)
            record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
            total_record.append(record)

        elif args.scenario == 'cross_subject':
            record = test_cross_subject(cfg, save_path)
            record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
            total_record.append(record)


    total_record = np.concatenate(total_record, axis=1)
    np.savetxt(save_path['log_file'], total_record, delimiter='\t', fmt='%.3f')

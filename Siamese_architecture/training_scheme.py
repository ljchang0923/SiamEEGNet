import time
import scipy.io
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import pickle

from utils import create_multi_window_input, plot_result
from train_test import train_model, test_model
from models import Siamese_CNN
from DataLoader import dataloader


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

    drowsy_grad_dict = {}
    alert_grad_dict = {}
    all_grad_dict = {}
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

            train_dl = dataloader(train_data, train_truth, tr_session_bound, 'train', cfg)
            val_dl = dataloader(train_data, train_truth, tr_session_bound, 'test', cfg)

            print("Training session: ", train_sess)
            model = Siamese_CNN(**cfg).to(cfg['device'])
            loss_record, grad_acc = train_model(model, train_dl, val_dl, thres, cfg, save_path['model_dir']) 

            all_grad_dict[cfg['ts_sub']] = grad_acc["all"]
            alert_grad_dict[cfg['ts_sub']] = grad_acc["alert"]
            drowsy_grad_dict[cfg['ts_sub']] = grad_acc["drowsy"]

            del train_data, train_truth, tr_session_bound
            del train_dl, val_dl
            del model

    if cfg['saliency_map']:
        with open(f'gradient/drowsy_grad_{cfg["backbone"]}_{cfg["scenario"]}.pkl', 'wb') as f:
            pickle.dump(drowsy_grad_dict, f)

        with open(f'gradient/alert_grad_{cfg["backbone"]}_{cfg["scenario"]}.pkl', 'wb') as f:
            pickle.dump(alert_grad_dict, f)
            
        with open(f'gradient/all_grad_{cfg["backbone"]}_{cfg["scenario"]}.pkl', 'wb') as f:
            pickle.dump(all_grad_dict, f)


def test_within_subject(cfg, save_path):
    filelist = sorted(os.listdir(save_path['data_dir']))

    record = []
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
                model = Siamese_CNN(**cfg).to(cfg['device'])
                model.load_state_dict(torch.load(save_path['model_dir'] + model_path))
                print(model_path)
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

        with open(f'decoding_result/{cfg["ts_sub"]}.npy', 'wb') as f:
            np.save(f, np.array(output))

    return record

def model_fusion(cfg):

    device = cfg["device"]
    sm_num = cfg["num_smooth"] - 1
    ch_num = [0, 1, 27, 29]
    
    filelist = sorted(os.listdir(args.data_dir))
    for filename in filelist:
        if filename.endswith('.mat'):
            test_data, test_truth, ts_session_bound, onset_time, _ = create_multi_window_input(filename, 0, cfg)
            cfg['ts_filename'] = filename[:-4]
        else:
            continue

        test_data = np.array(test_data, dtype=np.float32)
        test_dl = dataloader(test_data, test_truth, ts_session_bound, 'test', cfg)

        pred_set = []
        idx = 0
        print('testing model')
        for name in filelist:
        # for name in filelist:
            if name.endswith('.mat') and name[:4] != filename[:4]:
                model = Siamese_SCC(cfg).to(device)
                model_save_path = f'{cfg["model_dir"]}{cfg["ts_filename"]}_model.pt'
                model.load_state_dict(torch.load(model_save_path))
                print(cfg["ts_filename"])

                pred = test_model(model, test_dl, device)
                pred_set.append(pred[0])
                # pred_set.append(pred[0]*simi_score[idx])
                # sum_ += sorted_metrics[name]
                idx += 1
                del model

        pred_set = torch.cat(pred_set, 1)
        pred = torch.mean(pred_set, 1)
        output = [tensor.detach().cpu().item() for tensor in pred]
        # pred = torch.div(torch.sum(pred_set, 1), sum_)
        # output = [tensor.detach().cpu().item() for tensor in pred_set[0]]

        rmse = sqrt(mean_squared_error((test_truth-test_truth[0]).reshape(-1), output))
        cc = np.corrcoef((test_truth-test_truth[0]).reshape(-1), output)


        plot_result(output, (test_truth-test_truth[0]).reshape(-1), onset_time, cfg)
        with open(cfg["log_file"], 'a') as f:
            f.writelines('%s\t%.3f\t%.3f\n'%(filename[:-4], rmse, cc[0,1]))
            # with open("model_selection/" + filename[:-4], 'a') as f:
            #     f.writelines('%s\t%.3f\t%.3f\t%f\n'%(name[:-4], rmse, cc[0,1], abs(bs_delta)))
    
def train_cross_subject(cfg, save_path):

    filelist = sorted(os.listdir(save_path['data_dir']))

    # Load all data
    sub_list = []
    data = {}
    truth = {}
    onset_time = {}
    low_bound = 0

    # dictionary to store gradient
    drowsy_grad_dict = {}
    alert_grad_dict = {}
    all_grad_dict = {}

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
        val_session_bound = np.tile([0, val_data.shape[0]-1], (val_data.shape[0], 1))

        # training data
        train_data = []
        train_truth = []
        tr_session_bound = []
        low = 0
        for tr_sub_idx in range(len(sub_list)):
            if tr_sub_idx == ts_sub_idx or tr_sub_idx == val_sub_idx:
                continue
            for sess_idx in range(len(data[sub_list[tr_sub_idx]])):
                train_data = train_data + data[sub_list[tr_sub_idx]][sess_idx]
                train_truth.append(truth[sub_list[tr_sub_idx]][sess_idx].astype('float32'))
                data_len = len(data[sub_list[tr_sub_idx]][sess_idx])
                tr_session_bound.append(np.tile([low, low + data_len - 1], (data_len, 1)))
                low += data_len

        train_data = np.array(train_data, dtype=np.float32) # (#total training trial, #window, #channel, #timepoint)
        train_truth = np.concatenate(train_truth, 0) # (#total training trial, )
        tr_session_bound = np.concatenate(tr_session_bound, 0) # (#total training trial, 2)

        thres_drowsy = np.quantile(train_truth, 0.85)
        thres_alert = np.quantile(train_truth, 0.15)
        thres = {
                'drowsy': thres_drowsy,
                'alert': thres_alert
                }

        # wrap up to the Dataloader
        train_dl = dataloader(train_data, train_truth, tr_session_bound, 'train', cfg)
        val_dl = dataloader(val_data, val_truth, val_session_bound, 'test', cfg)

        ''' Model setup and training '''
        model = Siamese_CNN(**cfg).to(cfg['device'])
        print ('Validate on: ', sub_list[val_sub_idx])
        print('Test on: ', cfg['ts_sub'])
        print('Start training...')

        _, grad_acc = train_model(model, train_dl, val_dl, thres, cfg, save_path['model_dir'])

        all_grad_dict[cfg['ts_sub']] = grad_acc["all"]
        alert_grad_dict[cfg['ts_sub']] = grad_acc["alert"]
        drowsy_grad_dict[cfg['ts_sub']] = grad_acc["drowsy"]

        ''' Testing '''
        for idx in range(len(data[sub_list[ts_sub_idx]])):
            ### get testing data from testing subject
            test_data = np.array(data[sub_list[ts_sub_idx]][idx], dtype=np.float32) # (#testing trial, #window, #channel, #timepoint)
            test_truth = truth[sub_list[ts_sub_idx]][idx].astype('float32') # (#testing trial, )
            ts_session_bound = np.tile([0, test_data.shape[0]-1], (test_data.shape[0], 1))
            ts_onset_time = onset_time[sub_list[ts_sub_idx]][idx]

            ### Inference
            test_dl = dataloader(test_data, test_truth, ts_session_bound, 'test', cfg)
            pred = test_model(model, test_dl, cfg['device'])
            output = [tensor.detach().cpu().item() for tensor in pred]
            
            rmse = sqrt(mean_squared_error(test_truth.reshape(-1), output))
            cc = np.corrcoef(test_truth.reshape(-1), output)
        
            plot_result(output, test_truth.reshape(-1), ts_onset_time, save_path['fig_dir'], cfg, idx)
            
            record.append([rmse, cc[0, 1]])

        del train_data, train_truth, test_data, test_truth,  val_data, val_truth
        del train_dl, test_dl, val_dl

    if cfg['saliency_map']:
        with open(f'gradient/drowsy_grad_{cfg["backbone"]}_{cfg["scenario"]}.pkl', 'wb') as f:
            pickle.dump(drowsy_grad_dict, f)

        with open(f'gradient/alert_grad_{cfg["backbone"]}_{cfg["scenario"]}.pkl', 'wb') as f:
            pickle.dump(alert_grad_dict, f)
            
        with open(f'gradient/all_grad_{cfg["backbone"]}_{cfg["scenario"]}.pkl', 'wb') as f:
            pickle.dump(all_grad_dict, f)

    return record


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
            test_dl = dataloader(test_data, test_truth, ts_session_bound, 'test', cfg)

            model = Siamese_CNN(**cfg).to(cfg['device'])
            model_path = ts_sub_idx + '_model.pt'
            model.load_state_dict(torch.load(save_path['model_dir'] + model_path))
            pred = test_model(model, test_dl, cfg['device'])
            output = [tensor.detach().cpu().item() for tensor in pred]
            
            rmse = sqrt(mean_squared_error(test_truth.reshape(-1), output))
            cc = np.corrcoef(test_truth.reshape(-1), output)
        
            plot_result(output, test_truth.reshape(-1), ts_onset_time, cfg, idx)
            record.append([rmse, cc[0,1]])

    return record
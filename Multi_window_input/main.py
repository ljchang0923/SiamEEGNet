"""# Import libary and set random seed and gup """
import time
import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
from pathlib import Path
import argparse

from DataLoader import dataloader
from utils import read_json, plot_result, individual_dataloader, create_multi_window_input, train_model, test_model
from models import SCCNet, SCC_multi_window, CCLoss, Siamese_SCC


def individual(ratio):
    device = get_device()
    # set_seed(34)
    data_path = '/mnt/left/jason/dataset/selected_data/' # individual SAD_dataset_smooth_correct
    filelist = os.listdir(data_path)
    filelist = sorted(filelist)
    for filename in filelist:
        print("training: ", filename[:-4])
        if filename.endswith('.mat'):
            filepath = data_path + filename
            session = scipy.io.loadmat(filepath)
        
        wrapp_train = []
        tr_session = []
        low = 0
        sm_num = 10-1

        session = session['session'][0,0]
        train_data = np.transpose(session['eeg_trial'],(2,0,1))
        train_truth = session['DI'][0][sm_num:]
        threshold = np.median(train_truth)
        print("%s: %f"%(filename[:-4], threshold))

        for i in range(sm_num,train_data.shape[0]):
            wrapp_train.append(train_data[i-sm_num:i+1,:,:])
            tr_session.append([low, low+train_data.shape[0]-10])
        train_data = np.array(wrapp_train)
        print("data size: ", train_data.shape)
        print("truth shape: ", train_truth.shape)

        train_dl = individual_dataloader(train_data, train_truth, device, "train")
        test_dl = individual_dataloader(train_data, train_truth, device, "test")
        model = SCC_smoothing().to(device)
        print(session[0][0])
        config = {
            'epoch': 30,
            'optimizer': 'Adam',
            'lr': 0.001,
            'CE_weight': ratio,
            'CE_threshold': threshold,
            'save_path': '/mnt/left/jason/model/multi_window_SCC/'+ session[0][0] +'_model.pt',
            "filename":session[0][0]
        }

        loss_record = train_model(model, train_dl, test_dl, device, config)

        del train_data, train_truth
        del train_dl


def individual_test(ratio):
  device = get_device()
  # set_seed(34)
  data_path = '/mnt/left/jason/dataset/selected_data/'
  filelist = os.listdir(data_path)
  filelist = sorted(filelist)
  for filename in filelist:
      if filename.endswith('.mat'):
          filepath = data_path + filename
          session = scipy.io.loadmat(filepath)
      
      wrapp_test = []
      ts_session = []
      low = 0
      sm_num = 10-1

      session = session['session'][0,0]
      test_data = np.transpose(session['eeg_trial'],(2,0,1))
      test_truth = session['DI'][0][sm_num:]
      onset_time = session['onset_time'][0][sm_num:]
      baseline_idx = session['baseline_idx'][0,0]
      for i in range(sm_num,test_data.shape[0]):
          wrapp_test.append(test_data[i-sm_num:i+1,:,:])
          ts_session.append([low, low+test_data.shape[0]-10])
      test_data = np.array(wrapp_test)
      print("data size: ", test_data.shape)
      print("truth shape: ", test_truth.shape)

      test_dl = individual_dataloader(test_data, test_truth, device, 'test')
      smooth = False
      _, test_dl_scc = get_dataloader(np.transpose(session['eeg_trial'],(2,0,1))[sm_num:], session['DI'][0][sm_num:], np.transpose(session['eeg_trial'],(2,0,1))[sm_num:], session['DI'][0][sm_num:], smooth, device)

      print("testing session: ", filename[:-4])

      pred_set = []
      pred_set_scc = []
      for name in filelist:
        if name.endswith('.mat') and name != filename and name[:4] == filename[:4]:
          model_path = name[:-4] +'_model.pt'
          model = SCC_multi_window().to(device)
          model.load_state_dict(torch.load(f"/mnt/left/jason/model/multi_window_SCC/{model_path}"))
          model_scc = SCCNet().to(device)
          model_scc.load_state_dict(torch.load(f"/mnt/left/jason/model/SCCRegression/{model_path}"))

          print(model_path)
          pred = test_model(model, test_dl, device)
          pred_scc = test_model(model_scc, test_dl_scc, device)
          pred_set.append(pred[0])
          pred_set_scc.append(pred_scc[0])
      
      pred_set = torch.cat(pred_set, 1)
      pred_set_scc = torch.cat(pred_set_scc, 1)
      print(pred_set.size())
      pred = torch.mean(pred_set, 1)
      pred_scc = torch.mean(pred_set_scc, 1)
      output = [tensor.detach().cpu().item() for tensor in pred]
      output_scc = [tensor.detach().cpu().item() for tensor in pred_scc]

      info = {
        "method": "SCC_delta",
        "filename": filename[:-4],
        "cc_weight": model_path
      }

      

      # plot_result(output, test_truth, onset_time, info)
      plt.figure(figsize=(10,4.5))
      plt.rc('font', size=12)
      plt.rc('legend', fontsize=11)
      plt.rc('axes', labelsize=12)
      hfont = {'fontname':'Helvetica'}
      plt.plot(onset_time/60, test_truth.reshape(-1), 'k', linewidth=0.8)
      plt.plot(onset_time/60, output_scc, 'g', linewidth=0.8, alpha=0.5)
      plt.plot(onset_time/60, output, 'b', linewidth=0.8, alpha=0.7)
      plt.legend(['True DI', 'SCCNet', "SCC w/ multi-window"])

      if (filename[:-4] == "s44_070325n"):
          plt.plot(onset_time/60, output_siamese, 'r', linewidth=0.8, alpha=0.8)
          plt.legend(['True DI', 'SCCNet', "SCC w/ multi-window", "Siamese SCCNet"])

      plt.xlabel('Time(min)', **hfont)
      plt.ylabel('DI', **hfont)
      
      # plt.savefig('log/dataset2/SCC_smoothing_test_session'+filename+'.png')
      plt.savefig(f'fig_scc_and_multi/{info["filename"]}.png')
      plt.clf()

      rmse = sqrt(mean_squared_error(test_truth.reshape(-1), output))
      cc = np.corrcoef(test_truth.reshape(-1), output)
      print('test on model ' + model_path)
      print('RMSE: ', rmse, ' CC:', cc[0,1])
      
      # with open(f"multi_window_SCC_log.txt", 'a') as f:
      #   f.writelines('%s\t%.3f\t%.3f\n'%(filename[:-4], rmse, cc[0,1]))

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


def train_cross_subject(cfg):
    device = cfg['device']

    data_path = cfg['data_dir'] # individual SAD_dataset_smooth_correct
    filelist = os.listdir(data_path)
    filelist = sorted(filelist)

    # Load all data
    sub_list = []
    data = {}
    truth = {}
    onset_time = {}
    print("Loading data...")
    for filename in filelist:
        single_train_data, single_train_truth, _, single_onset_time, _ = create_multi_window_input(filename, 0, cfg)
        if(filename[:3] not in sub_list):
            sub_list.append(filename[:3])
            data[filename[:3]] = []
            truth[filename[:3]] = []
            onset_time[filename[:3]] = []
        data[filename[:3]].append(single_train_data)
        truth[filename[:3]].append(single_train_truth)
        onset_time[filename[:3]].append(single_onset_time)

    # train the model for all subject iteratively
    for t in range(1):
        print(f"Repeatition {t + 1}")
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
                if tr_sub_idx != ts_sub_idx and tr_sub_idx != val_sub_idx:
                    for idx in range(len(data[sub_list[tr_sub_idx]])):
                        train_data = train_data + data[sub_list[tr_sub_idx]][idx]
                        train_truth.append(truth[sub_list[tr_sub_idx]][idx].astype('float32'))

            train_data = np.array(train_data, dtype=np.float32) # (#total training trial, #window, #channel, #timepoint)
            train_truth = np.concatenate(train_truth, 0) # (#total training trial, )
    
            # wrap up to the Dataloader
            train_dl = dataloader(train_data, train_truth, 'train', cfg)
            val_dl = dataloader(val_data, val_truth, 'test', cfg)

            ''' Model setup and training '''
            model = SCC_multi_window(cfg).to(device)
            print ('validate on: ', sub_list[val_sub_idx])
            print('test on: ', cfg['ts_sub'])
            print('Start training...')

            _ = train_model(model, train_dl, val_dl, device, cfg)

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
            
                plot_result(output, test_truth.reshape(-1), ts_onset_time, cfg, idx)
                with open(cfg['log_file'], 'a') as f:
                    f.writelines('%s\t%.3f\t%.3f\n'%(f"{cfg['ts_sub']}-{idx+1}", rmse, cc[0,1]))

            del train_data, train_truth, test_data, test_truth,  val_data, val_truth
            del train_dl, test_dl, val_dl

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=str, help="path to configuration file", default='config/config.json')
    parser.add_argument("--scenario", type=str, help="within_subject or cross_subject", default="cross_subject_DF")
    parser.add_argument("--device", type=str, default = 'cuda:0')
    args = parser.parse_args()

    cfg = read_json(args.config_path)
    cfg['device'] = args.device
    cfg['scenario'] = args.scenario
    cfg['log_file'] = f'log/multi_window_{cfg["backbone"]}_{cfg["scenario"]}_{cfg["EEG_ch"]}ch'
    cfg['model_dir'] = f'{cfg["model_dir"]}multi_window_{cfg["backbone"]}_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'
    if not os.path.exists(cfg['model_dir']):
        os.makedirs(cfg['model_dir'])

    if args.scenario == 'cross_subject_DF':
        train_cross_subject(cfg)
    elif args.scenario == "cross_subject_MF":
        train_individual(cfg)
        model_fusion(cfg)
    elif args.scenario == 'individual':
        train_individual(cfg)
        # test_within_subject(cfg)
    else:
        raise ValueError('Invalid scenario')

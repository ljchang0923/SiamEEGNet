"""# Import libary and set random seed and gup """
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import argparse

from utils import set_seed, get_dataloader, plot_result, single_trial_input
from models import SCCNet, SCC_multi_window, CCLoss, EEGNet


"""# Training setup"""

def train_model(model, train_dl, test_dl, device, config):
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['lr'], weight_decay=0.0001)
    criterion = nn.MSELoss(reduction='mean')
    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    record = {'train loss': [], 'val loss':[]}
    total_loss = 0
    mini = 1e8
    best_rmse = 0
    best_cc = 0
    print('\n training start')
    
    for epoch in range(config['epoch']):
        model.train()
        print("model mode: ", model.training)
        for x_train, y_train in train_dl:
            
            optimizer.zero_grad()

            x_train, y_train = x_train.to(device), y_train.to(device)
            _, pred = model(x_train)
            loss = criterion(pred, y_train)
            total_loss += loss.detach().cpu().item()

            loss.backward()
            optimizer.step()
        #lr_scheduler.step()

        val_loss, rmse, cc = val_model(model, test_dl, device, config)
        record["train loss"].append(total_loss/len(train_dl))
        record["val loss"].append(val_loss)

        print(f"{epoch+1} epoch: train loss-> {total_loss/len(train_dl)}")
        print(f"val loss-> {val_loss} rmse -> {rmse} cc -> {cc}")
        matrice = 0.5 * rmse + 0.5*(1-cc)
        if(matrice < mini):
            mini = matrice
            best_rmse = rmse
            best_cc = cc
            torch.save(model.state_dict(), config['save_path'])
        
        total_loss = 0
        
    # print(f"best RMSE ->{best_rmse} CC->{best_cc}")
    # torch.cuda.empty_cache()
    
    return record

def val_model(model, test_dl, device, config):
    criterion = nn.MSELoss(reduction='mean')
    total_loss = 0
    output = []

    with torch.no_grad():
        model.eval()
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device)
            _, pred = model(x_test)
            mse = criterion(pred, y_test)
            rmse = sqrt(mse)
            cc = np.corrcoef(y_test.cpu().detach().numpy().reshape(-1), pred.cpu().detach().numpy().reshape(-1))

    return mse/len(test_dl), rmse, cc[0,1] 

def test_model(model, test_dl, device):
    criterion = nn.MSELoss(reduction='mean')
    model.eval()
    total_loss = 0
    output = []

    with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device)
            _, pred = model(x_test)
            # loss = criterion(pred, y_test)
            # total_loss += loss.detach().cpu().item()
            output.append(pred)
        # print(f'testing loss: {total_loss}')
    return output

def train_individual(args):

    device = args.device
    data_path = args.data_dir # individual SAD_dataset_smooth_correct

    filelist = sorted(os.listdir(data_path))
    for filename in filelist:
        train_data, train_truth, onset_time = single_trial_input(data_path, filename)
        train_dl, test_dl = get_dataloader(train_data, train_truth, 'train', false, device)

        model = EEGNet().to(device)
        print(session[0][0])
        config = {
            'epoch': 300,
            'optimizer': 'Adam',
            'lr': 0.001,
            'save_path': '/mnt/left/jason/model/EEGNet/' + session[0][0] + '_model.pt',
            'filename': session[0][0]
        }

        loss_record = train_model(model, train_dl, test_dl, device, config)

    del train_data, train_truth
    del train_dl, test_dl

def test_within_sub(args):
    device = get_device()
    data_path = args.data_dir # individual SAD_dataset_smooth_correct

    filelist = sorted(os.listdir(data_path))
    for filename in filelist:
        print("training: ", filename[:-4])
        test_data, test_truth, onset_time = single_trial_input(data_path, filename)
        test_dl = get_dataloader(test_data, test_truth, 'test', False, device)

        pred_set = []
        for name in filelist:
            if name.endswith('.mat') and name != filename and name[:4] == filename[:4]:
                model_path = name[:-4] + '_model.pt'
                model = EEGNet().to(device)
                model.load_state_dict(torch.load(f"{args.model_dir + model_path}"))
                print(model_path)
                pred = test_model(model, test_dl, device)
                pred_set.append(pred[0])
            
        pred_set = torch.cat(pred_set, 1)
        pred = torch.mean(pred_set, 1)
        output = [tensor.detach().cpu().item() for tensor in pred]

        rmse = sqrt(mean_squared_error((test_truth).reshape(-1), output))
        cc = np.corrcoef((test_truth).reshape(-1), output)
        print('test on model ' + model_path)
        print('RMSE: ', rmse, ' CC:', cc[0,1])

        if (args.record):
            info = {
                "save_path" : args.fig_dir,
                "filename" : filename_ts[:-4]}
            plot_result(output, test_truth.reshape(-1), onset_time, info)
            with open(args.log, 'a') as f:
                f.writelines('%s\t%.3f\t%.3f\n'%(filename_ts[:-4], rmse, cc[0,1]))

def train_cross_subject(args):
    device = args.device
    # set_seed(34)
    data_path = args.data_dir # individual SAD_dataset_smooth_correct
    filelist = os.listdir(data_path)
    filelist = sorted(filelist)

    for sub_idx, filename_ts in enumerate(filelist):
        ''' Load test data '''
        test_sub = filename_ts[:3]
        test_data, test_truth, onset_time = single_trial_input(data_path, filename_ts)
        
        ''' select validation data 
            # select the subject next to the test data
            # when the tes subject is the last subject in the list, then choose the first instead 
        '''
        val_idx = sub_idx + 1
        while(1):
            if(val_idx >= len(filelist)):
                val_idx = 2
            elif filelist[val_idx][:3] == test_sub:
                val_idx += 1
            else:
                break
        val_data, val_truth, _ = single_trial_input(data_path, filelist[val_idx])
        
        ''' Load training dataset'''
        train_data = []
        train_truth = []
        tr_session_bound = []
        low = 0
        for filename_tr in filelist:
            if filename_tr.endswith('.mat') and filename_tr[:3] != test_sub and filename_tr != filelist[val_idx]:
                print("training: ", filename_tr[:-4])
                data, truth, _ = single_trial_input(data_path, filename_tr)
                train_data.append(data)
                train_truth.append(truth)
    
        
        train_data = np.concatenate(train_data, 0) ## train_data is a list with length of # total trials. Inside, each element is np array with size of 10*30*750
        train_truth = np.concatenate(train_truth, 0) ## size of train true: (#total_trials, )
        
        
        ''' wrap data to the dataloarder'''
        train_dl = get_dataloader(train_data, train_truth, False, 'train', device)
        val_dl = get_dataloader(val_data, val_truth, False, 'test', device)
        test_dl = get_dataloader(test_data, test_truth, False, 'test', device)
        
        ''' model setup and training '''
        model = EEGNet().to(device)
        print ('validate on: ', filelist[val_idx][:-4])
        print('test on: ', filename_ts[:-4])
        threshold = 0.3
        config = {
                'epoch': args.epoch,
                'optimizer': args.optimizer,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'CE_threshold': threshold,
                'save_path': args.model_dir + filename_ts[:-4] +'_model.pt',
                "filename": filename_ts[:-4]
        }
        _ = train_model(model, train_dl, val_dl, device, config)

        ''' Testing '''
        model_name = filename_ts[:-4] +'_model.pt'
        model = EEGNet().to(device)
        model.load_state_dict(torch.load(args.model_dir + model_name))
        pred = test_model(model, test_dl, device)
        output = [tensor.detach().cpu().item() for tensor in pred[0]]
        
        rmse = sqrt(mean_squared_error((test_truth-test_truth[0]).reshape(-1), output))
        cc = np.corrcoef((test_truth-test_truth[0]).reshape(-1), output)

        if (args.record):
            info = {
                "save_path" : args.fig_dir,
                "filename" : filename_ts[:-4]}
            plot_result(output, test_truth.reshape(-1), onset_time, info)
            with open(args.log, 'a') as f:
                f.writelines('%s\t%.3f\t%.3f\n'%(filename_ts[:-4], rmse, cc[0,1]))

        del train_data, train_truth, test_data, test_truth, val_data, val_truth
        del train_dl, test_dl, val_dl
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default = 'cuda:1')
    parser.add_argument("--epoch", default = 50)
    parser.add_argument("--optimizer", default = "Adam")
    parser.add_argument("--lr", default= 0.001)
    parser.add_argument("--weight_decay", default = 0.0001)
    parser.add_argument("--model_dir", default = f"/mnt/left/jason/model/EEGNet_cross_subject/") # /model_test -> for CEweight test
    parser.add_argument("--data_dir", default = "/mnt/left/jason/dataset/selected_data2/")
    parser.add_argument("--fig_dir", default = f"fig_EEGNet_cross_subject")
    parser.add_argument("--record", default = True)
    parser.add_argument("--log", default = f"EEGNet_cross_subject.txt")
    parser.add_argument("--num_repeat", default = 5)
    args = parser.parse_args()

    for i in range(args.num_repeat):
        start = time.time()
        # train_individual()
        # test_within_sub()
        train_cross_subject(args)
        end = time.time()
        print(f'time cost: {end - start} s')

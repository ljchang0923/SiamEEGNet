import os
import argparse
import numpy as np
import csv
import pickle

from utils.functions import get_dataset, evaluate
from utils.getDataLoader import get_dataloader, get_dataloader_4_cross_sub
from utils.train_test import train_model, test_model
from model.SiamEEGNet import SiamEEGNet, Multi_window_CNN

def get_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    ## about model setting
    parser.add_argument("--backbone", type=str, help="choose EEG decoding model", default="EEGNet")
    parser.add_argument("--method", type=str, help="method to use (siamese or multi-window)", default='siamese')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--weight_decay", type=int, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="Adam")
    
    # about EEG data
    parser.add_argument("--EEG_ch", type=int, default=30)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--num_window", type=int, default=10)
    parser.add_argument("--pairing", type=int, default=1)
    
    # about experiment
    parser.add_argument("--device", type=str, default = 'cuda:0')
    parser.add_argument("--repeat", type=int, default = 1)
    parser.add_argument("--save_grad", type=bool, default=False)
    
    args = vars(parser.parse_args())

    return args

def main(args):

    # change to the path you would like to use #
    save_path = {
        'data_dir':'/home/cecnl/ljchang/CECNL/sustained-attention/selected_data/',
        'model_dir':f'/home/cecnl/ljchang/CECNL/sustained-attention/model/test/{args["method"]}{args["backbone"]}_{args["num_window"]}window_{args["pairing"]}pair_cross_subject_{args["EEG_ch"]}ch/',
        'log_file':f'log/test/{args["method"]}_{args["backbone"]}_{args["num_window"]}window_{args["pairing"]}pair_cross_subject_{args["EEG_ch"]}ch.csv',
        'fig_dir':f'fig/test/{args["method"]}_{args["backbone"]}_{args["num_window"]}window_{args["pairing"]}pair_cross_subject_{args["EEG_ch"]}ch/'
    }

    if not os.path.exists(save_path['fig_dir']):
        os.makedirs(save_path['fig_dir'])

    if not os.path.exists(save_path['model_dir']):
        os.makedirs(save_path['model_dir'])

    with open(save_path['log_file'], 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in args.items():
            writer.writerow(row)

    # Load dataset
    print("Backbone: ", args["backbone"])
    '''
    sub_list:   store all subject IDs in the dataset
    data:       store the Processed EEG data using dict (key: subject ID, value: a list of EEG sessions)
    truth:      store ground truth delta DI using dict (key: subject ID, value: a list of ground truth)
    onset time: store the time stamps corresponding to each trial
    ** Use subject ID in the sub_list to access session data in each dict **
    '''
    print("Loading dataset...")
    sub_list, data, truth, _ = get_dataset(save_path['data_dir'], args["EEG_ch"], args["num_window"])

    total_record = []
    all_grad_dict = {}
    for i in range(args["repeat"]):
        print("Repeatition: {}".format(i+1))
        record = []
        for ts_sub_idx in range(len(sub_list)):
            
            ts_sub = sub_list[ts_sub_idx]

            # Obtain train and val data loader based on given testing subject ID #
            train_dl, val_dl = get_dataloader_4_cross_sub(sub_list, data, truth, ts_sub_idx, args)

            # Model setup and training #
            if args["method"] == 'siamese':
                model = SiamEEGNet(
                EEG_ch=args["EEG_ch"],
                    fs=args["fs"],
                    num_window=args["num_window"],
                    backbone=args["backbone"],
                )
            else:
                model = Multi_window_CNN(
                EEG_ch=args["EEG_ch"],
                    fs=args["fs"],
                    num_window=args["num_window"],
                    backbone=args["backbone"],
                )

            print('Test on: ', ts_sub)
            print('Start training...')
            model = model.to(args["device"])
            _, grad_acc = train_model(
                model, 
                train_dl, 
                val_dl, 
                save_path = save_path['model_dir'], 
                model_name = ts_sub,
                **args
            ) 

            all_grad_dict[ts_sub] = grad_acc["all"]

            # Test all sessions of testing subject #
            for sess in range(len(data[ts_sub])):

                # Get testing data from testing subject #
                ts_data = np.array(data[ts_sub][sess], dtype=np.float32) # (#testing trial, #window, #channel, #timepoint)
                ts_truth = truth[ts_sub][sess].astype('float32') # (#testing trial, )
                ts_session_bound = np.tile([0, ts_data.shape[0] - 1], (ts_data.shape[0], 1)) 
                test_dl = get_dataloader(ts_data, ts_truth, ts_session_bound, 'test', 'static', **args)

                # Inference #
                _, pred = test_model(model, test_dl, args["method"], args["device"])
                output = [tensor.detach().cpu().item() for tensor in pred]
                
                rmse,cc = evaluate(output, ts_truth, args["method"])
                record.append([rmse, cc])
                print('RMSE: {} CC: {}'.format(rmse, cc))
            
            del ts_data, ts_truth
            del train_dl, test_dl, val_dl
        
        record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
        total_record.append(record)

    total_record = np.concatenate(total_record, axis=1)

    with open(save_path['log_file'], 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in total_record:
            writer.writerow(row)

    print(total_record)

    if args["save_grad"]:
        with open(f'gradient/all_grad_{args["backbone"]}_cross_subject.pkl', 'wb') as f:
            pickle.dump(all_grad_dict, f)

if __name__ == "__main__":
    
    args = get_arg_parser()
    main(args)
    
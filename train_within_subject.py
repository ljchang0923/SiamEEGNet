import os
import argparse
import torch
import numpy as np
import csv
import pickle
from sklearn.model_selection import train_test_split

from utils.functions import get_dataset, evaluate
from utils.getDataLoader import get_dataloader
from utils.train_test import train_model, test_model
from model.SiamEEGNet import SiamEEGNet, Multi_window_CNN

def get_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    # model param
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
    parser.add_argument("--training_method", type=str, default="dynamic")
    parser.add_argument("--repeat", type=int, default = 1)
    parser.add_argument("--save_grad", type=bool, default=False)
    
    args = vars(parser.parse_args())

    return args

def main(args):

    # change to the path you would like to use
    save_path = {
        'data_dir':'/home/cecnl/ljchang/CECNL/sustained-attention/selected_data/',
        'model_dir':f'/home/cecnl/ljchang/CECNL/sustained-attention/model/test/{args["method"]}{args["backbone"]}_{args["num_window"]}window_{args["pairing"]}pair_within_subject_{args["EEG_ch"]}ch_baseline/',
        'log_file':f'log/test/{args["method"]}_{args["backbone"]}_{args["num_window"]}window_{args["pairing"]}pair_within_subject_{args["EEG_ch"]}ch_baseline.csv',
        'fig_dir':f'fig/test/{args["method"]}_{args["backbone"]}_{args["num_window"]}window_{args["pairing"]}pair_within_subject_{args["EEG_ch"]}ch/'
    }

    if not os.path.exists(save_path['fig_dir']):
        os.makedirs(save_path['fig_dir'])

    if not os.path.exists(save_path['model_dir']):
        os.makedirs(save_path['model_dir'])

    with open(save_path['log_file'], 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for arg in args.items():
            writer.writerow(arg)

    
    print("Backbone: ", args["backbone"])
    '''
    sub_list:   store all subject IDs in the dataset
    data:       store the processed EEG data using dict (key: subject ID, value: a list of EEG sessions)
    truth:      store ground truth delta DI using dict (key: subject ID, value: a list of ground truth)
    onset time: store the time stamps corresponding to each trial
    ** Use subject ID to access the data in each dict **
    '''
    print("Loading dataset...")
    # Load dataset w/ multi-window processing and return all subject IDs, EEG data, and ground truth#
    sub_list, data, truth, _ = get_dataset(save_path['data_dir'], args["EEG_ch"], args["num_window"])

    total_record = []
    all_grad_dict = {}
    for i in range(args["repeat"]):
        print("Repeatition: {}".format(i+1))
        record = []
        # Iterate all sessions and train individual model for each session #
        for sub in sub_list:
            for sess in range(len(data[sub])):
                # Obtain the training data from 1 session
                tr_sub = f"{sub}_{sess+1}"
                tr_data = np.array(data[sub][sess], dtype=np.float32)
                tr_truth = truth[sub][sess].astype('float32')
                tr_session_bound = np.tile([0, tr_data.shape[0] - 1], (tr_data.shape[0], 1))

                # Obtain dataloader according to different method we chose #
                # Siamese -> need customized pair dataset
                # Normal model -> just using nomal dataset
                if args["method"] == 'siamese':
                    train_dl = get_dataloader(tr_data, tr_truth, tr_session_bound, 'train', args["training_method"], **args)
                    val_dl = get_dataloader(tr_data, tr_truth, tr_session_bound, 'test', 'static', **args) # use static baseline pairing as validation
                    model = SiamEEGNet(
                        EEG_ch=args["EEG_ch"],
                        fs=args["fs"],
                        num_window=args["num_window"],
                        backbone=args["backbone"],
                    )
                else:
                    x_train, x_val, y_train, y_val = train_test_split(tr_data, tr_truth, test_size=0.3, shuffle=True) # we need to split train/val set manually in the normal models
                    train_dl = get_dataloader(x_train, y_train, 'train', args["training_method"], **args)
                    val_dl = get_dataloader(x_val, y_val, 'test', 'static', **args)
                    model = Multi_window_CNN(
                        EEG_ch=args["EEG_ch"],
                        fs=args["fs"],
                        num_window=args["num_window"],
                        backbone=args["backbone"],
                    )

                # Set up model and train the individual model for 1 session#    
                print(f'Training session: {tr_sub}')
                model = model.to(args["device"])
                _, grad_acc = train_model(
                    model, 
                    train_dl, 
                    val_dl, 
                    save_path = save_path['model_dir'], 
                    model_name = tr_sub,
                    **args
                ) 
                
                # store the accumulated input gradient
                all_grad_dict[tr_sub] = grad_acc["all"]

            # After we obtain all individual models from 1 subject, we can conduct within-subject test for sessions of the current subject
            # Test within subject #
            model_list = sorted(os.listdir(save_path['model_dir'])) # Obtain current trained models
            for sess in range(len(data[sub])):
                # Obtain test data
                ts_sub = f"{sub}_{sess+1}"
                ts_data = np.array(data[sub][sess], dtype=np.float32)
                ts_truth = truth[sub][sess].astype('float32')
                tr_session_bound = np.tile([0, ts_data.shape[0] - 1], (ts_data.shape[0], 1))
                print("Data size: {} Label size: {}".format(ts_data.shape, ts_truth.shape))

                # obtain testing dataloader, and we use static baseline inference
                test_dl = get_dataloader(ts_data, ts_truth, tr_session_bound, 'test', 'static', **args)

                # Use model fusion to get average prediction result #
                # averaging the predictions from the models trained by other sessions# 
                print("Testing session: {}".format(ts_sub))
                pred_pool = []
                for model_path in model_list:
                    if model_path[:3] != sub or model_path[:5] == ts_sub:
                        continue

                    model.load_state_dict(torch.load(save_path['model_dir'] + model_path))
                    _, pred = test_model(model, test_dl, args['method'], args['device'])
                    pred_pool.append(pred)

                pred_pool = torch.cat(pred_pool, 1)
                pred = torch.mean(pred_pool, 1)
                output = [tensor.detach().cpu().item() for tensor in pred]

                rmse, cc = evaluate(output, ts_truth, args["method"])
                record.append([rmse, cc])
                print('RMSE: {} CC: {}'.format(rmse, cc))
                
        record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
        total_record.append(record)

    total_record = np.concatenate(total_record, axis=1)

    with open(save_path['log_file'], 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in total_record:
            writer.writerow(row)

    print(total_record)

    # save gradient if needed
    if args["save_grad"]:
        with open(f'gradient/all_grad_{args["backbone"]}_within_subject.pkl', 'wb') as f:
            pickle.dump(all_grad_dict, f)

if __name__ == "__main__":
    
    args = get_arg_parser()
    main(args)
    
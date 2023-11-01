import os
import argparse
import numpy as np
import csv
import torch

from utils.functions import get_dataset, evaluate
from utils.getDataLoader import get_dataloader
from utils.train_test import test_model
from model.SiamEEGNet import SiamEEGNet, Multi_window_CNN

def get_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    ## about model setting
    parser.add_argument("--backbone", type=str, help="choose EEG decoding model", default="EEGNet")
    parser.add_argument("--method", type=str, help="method to use (siamese or multi-window)", default='siamese')
    parser.add_argument("--model_dir", type=str, help="enter the directory path to trained models")
    
    # about EEG data
    parser.add_argument("--EEG_ch", type=int, default=30)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--num_window", type=int, default=10)
    
    # about experiment
    parser.add_argument("--scenario", type=str, default='cross_subject')
    parser.add_argument("--device", type=str, default = 'cuda:0')
    
    args = vars(parser.parse_args())

    return args

def main(args):

    # change to the path you would like to use
    path = {
        'data_dir':'/home/cecnl/ljchang/CECNL/sustained-attention/selected_data/',
        'log_file':f'log/test/{args["method"]}_{args["backbone"]}_{args["num_window"]}window_1pair_{args["scenario"]}_{args["EEG_ch"]}ch.csv',
    }
    path['model_dir'] = args["model_dir"]

    if not os.path.exists(path['model_dir']):
        raise ValueError("No such model directory. Please check model directory.")
    else:
        model_list = sorted(os.listdir(path['model_dir']))

    with open(path['log_file'], 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in args.items():
            writer.writerow(row)

    # Load dataset
    '''
    sub_list:   store all subject IDs in the dataset
    data:       store the Processed EEG data using dict (key: subject ID, value: a list of EEG sessions)
    truth:      store ground truth delta DI using dict (key: subject ID, value: a list of ground truth)
    onset time: store the time stamps corresponding to each trial
    ** Use subject ID in the sub_list to access session data in each dict **
    '''
    print("Loading dataset...")
    sub_list, data, truth, _ = get_dataset(path['data_dir'], args["EEG_ch"], args["num_window"])

    ''' Load model architecture '''
    print("Backbone: ", args["backbone"])
    if args["method"] == "siamese":
        model = SiamEEGNet(
        EEG_ch=args["EEG_ch"],
            fs=args["fs"],
            num_window=args["num_window"],
            backbone=args["backbone"],
        )
    elif args["method"] == "multi_window":
        model = Multi_window_CNN(
        EEG_ch=args["EEG_ch"],
            fs=args["fs"],
            num_window=args["num_window"],
            backbone=args["backbone"],
        )
    else:
        raise ValueError("Invalid method. Please choose either siamese or multi_window for the method.")

    model = model.to(args["device"])

    record = []
    for ts_sub_idx in range(len(sub_list)):
        
        ts_sub = sub_list[ts_sub_idx]
        print('Test on: ', ts_sub)

        for sess in range(len(data[ts_sub])):

            ### Get testing data from testing subject
            ts_data = np.array(data[ts_sub][sess], dtype=np.float32) # (#testing trial, #window, #channel, #timepoint)
            ts_truth = truth[ts_sub][sess].astype('float32') # (#testing trial, )
            ts_session_bound = np.tile([0, ts_data.shape[0]-1], (ts_data.shape[0], 1)) 
            test_dl = get_dataloader(ts_data, ts_truth, ts_session_bound, 'test', **args)
            print("Data size: {} Label size: {}".format(ts_data.shape, ts_truth.shape))

            ### Inference
            if args["scenario"] == 'cross_subject':
                model_path = ts_sub + '_model.pt'
                model.load_state_dict(torch.load(path['model_dir'] + model_path))
                model = model.to(args["device"])
                _, pred = test_model(model, test_dl, args["method"], args["device"])

            elif args["scenario"] == 'within_subject':
                pred_pool = []
                for model_path in model_list:
                    if model_path[:3] != ts_sub or model_path[:5] == f"{ts_sub}_{sess+1}":
                        continue

                    model.load_state_dict(torch.load(path['model_dir'] + model_path))
                    _, pred = test_model(model, test_dl, args['method'], args['device'])
                    pred_pool.append(pred)

                pred_pool = torch.cat(pred_pool, 1)
                pred = torch.mean(pred_pool, 1)

            else:
                raise ValueError("Invalid scenario. Please choose either cross_subject or within_subject for scenario.")
            
            output = [tensor.detach().cpu().item() for tensor in pred]
            rmse, cc = evaluate(output, ts_truth, args["method"])
            record.append([rmse, cc])
            print('RMSE: {} CC: {}'.format(rmse, cc))

            output_path = f'decoding_result/{args["method"]}{args["backbone"]}_{args["num_window"]}win'
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            with open(f'{output_path}/{ts_sub}-{sess+1}.npy', 'wb') as f:
                np.save(f, output)
        
        del ts_data, ts_truth, test_dl
    
    record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
    print(record)

    with open(path['log_file'], 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in record:
            writer.writerow(row)

if __name__ == "__main__":
    
    args = get_arg_parser()
    main(args)
    
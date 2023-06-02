import os
from pathlib import Path
import argparse
from utils import read_json, create_multi_window_input
from training_scheme import train_within_subject, train_cross_subject
from training_scheme import test_within_subject, test_cross_subject
import numpy as np
import csv

    
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str, help="train or inference", default="train")
    parser.add_argument("--config_path", type=str, help="path to configuration file", default='config/config.json')
    parser.add_argument("--scenario", type=str, help="within_subject or cross_subject", default="cross_subject")
    parser.add_argument("--device", type=str, default = 'cuda:0')
    parser.add_argument("--repeat", type=int, default = 1)
    args = parser.parse_args()

    cfg = read_json(args.config_path)
    cfg['device'] = args.device
    cfg['scenario'] = args.scenario
    cfg['saliency_map'] = False

    save_path = {
        'data_dir':'/home/cecnl/ljchang/CECNL/sustained-attention/selected_data/',
        'model_dir':f'/home/cecnl/ljchang/CECNL/sustained-attention/model/ablation/siamese{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch_baseline/',
        'log_file':f'log/ablation/siamese_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch_baseline.csv',
        'fig_dir':f'fig/ablation/fig_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch_baseline/'
    }

    if not os.path.exists(save_path['fig_dir']):
        os.makedirs(save_path['fig_dir'])

    if not os.path.exists(save_path['model_dir']):
        os.makedirs(save_path['model_dir'])

    with open(save_path['log_file'], 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in cfg.items():
            writer.writerow(row)

    # Load all data
    print("Loading data...")
    sub_list = []
    data, truth, onset_time = {}, {}, {}

    filelist = sorted(os.listdir(save_path['data_dir']))
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
    if args.mode == "train":
        for i in range(args.repeat):

            print("Repeatition: {}".format(i+1))
            print("Backbone: ", cfg["backbone"])

            if args.scenario == 'cross_subject':
                record = train_cross_subject(cfg, save_path, sub_list, data, truth, onset_time)
                record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
                total_record.append(record)

            elif args.scenario == 'within_subject':
                train_within_subject(cfg, save_path, sub_list, data, truth, onset_time)
                record = test_within_subject(cfg, save_path, sub_list, data, truth, onset_time)
                record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
                total_record.append(record)

            else:
                raise ValueError('Invalid scenario')

    elif args.mode == "inference":
        if args.scenario == 'within_subject':
            record = test_within_subject(cfg, save_path, sub_list, data, truth, onset_time)
            record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
            total_record.append(record)

        elif args.scenario == 'cross_subject':
            record = test_cross_subject(cfg, save_path, sub_list, data, truth, onset_time)
            record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
            total_record.append(record)
        else:
            raise ValueError('Invalid scenario')

    else:
        raise ValueError('Invalid mode')

    total_record = np.concatenate(total_record, axis=1)

    with open(save_path['log_file'], 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in total_record:
            writer.writerow(row)

    print(total_record)
    # np.savetxt(save_path['log_file'], total_record, delimiter='\t', fmt='%.3f')

    

if __name__ == "__main__":
    main()
    
    

    
    
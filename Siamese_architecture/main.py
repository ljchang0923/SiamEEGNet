import os
from pathlib import Path
import argparse
from utils import read_json
from training_scheme import train_within_subject, train_cross_subject
from training_scheme import test_within_subject, test_cross_subject

REPEAT = 1
    
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str, help="train or inference", default="train")
    parser.add_argument("--config_path", type=str, help="path to configuration file", default='config/config.json')
    parser.add_argument("--scenario", type=str, help="within_subject or cross_subject", default="cross_subject")
    parser.add_argument("--device", type=str, default = 'cuda:0')
    args = parser.parse_args()

    cfg = read_json(args.config_path)
    cfg['device'] = args.device
    cfg['scenario'] = args.scenario
    cfg['saliency_map'] = False

    save_path = {
        'data_dir':'/home/cecnl/ljchang/CECNL/sustained-attention/selected_data/',
        'model_dir':f'/home/cecnl/ljchang/CECNL/sustained-attention/model/siamese{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/',
        'log_file':f'log/siamese_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch.csv',
        'fig_dir':f'fig/fig_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'
    }

    if not os.path.exists(save_path['fig_dir']):
        os.makedirs(save_path['fig_dir'])

    if not os.path.exists(save_path['model_dir']):
        os.makedirs(save_path['model_dir'])


    print("Backbone: ", cfg["backbone"])
    total_record = []
    if args.mode == "train":
        for i in range(REPEAT):
            if args.scenario == 'cross_subject':
                record = train_cross_subject(cfg, save_path)
                record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
                total_record.append(record)
            elif args.scenario == 'within_subject':
                train_within_subject(cfg, save_path)
                record = test_within_subject(cfg, save_path)
                record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
                total_record.append(record)
            else:
                raise ValueError('Invalid scenario')

    elif args.mode == "inference":
        if args.scenario == 'within_subject':
            record = test_within_subject(cfg, save_path)
            record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
            total_record.append(record)
        elif args.scenario == 'cross_subject':
            record = test_cross_subject(cfg)
            record = np.concatenate((record, np.mean(record, axis=0).reshape(1, 2)), axis=0)
            total_record.append(record)
        else:
            raise ValueError('Invalid scenario')

    else:
        raise ValueError('Invalid mode')

    total_record = np.concatenate(total_record, axis=1)
    np.savetxt(save_path['log_file'], total_record, delimiter='\t', fmt='%.3f')


if __name__ == "__main__":
    main()
    
    

    
    
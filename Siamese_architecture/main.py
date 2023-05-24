import os
from pathlib import Path
import argparse
from utils import read_json
from training_scheme import train_within_subject, train_cross_subject
from training_scheme import test_within_subject, test_cross_subject, model_fusion

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
    cfg['saliency_map'] = True
    
    data_dir = '/home/cecnl/ljchang/CECNL/sustained-attention/selected_data/'
    model_dir = f'/home/cecnl/ljchang/CECNL/sustained-attention/model/siamese{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'
    log_file = f'log/siamese_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch'
    fig_dir = f'fig/fig_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    save_path ={
                "data_dir": data_dir,
                "model_dir": model_dir,
                "fig_dir": fig_dir,
                "log_file": log_file
                }

    print("Backbone: ", cfg["backbone"])
    if args.mode == "train":
        if args.scenario == 'cross_subject':
            for i in range(REPEAT):
                train_cross_subject(cfg, save_path)
        elif args.scenario == 'within_subject':
            for i in range(REPEAT):
                train_within_subject(cfg, save_path)
                test_within_subject(cfg, save_path)
        else:
            raise ValueError('Invalid scenario')

    elif args.mode == "inference":
        if args.scenario == 'within_subject':
            test_within_subject(cfg, save_path)
        elif args.scenario == 'cross_subject':
            test_cross_subject(cfg)
        else:
            raise ValueError('Invalid scenario')

    else:
        raise ValueError('Invalid mode')


if __name__ == "__main__":
    main()
    
    

    
    
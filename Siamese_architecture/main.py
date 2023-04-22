import os
from pathlib import Path
import argparse
from utils import read_json
from training_scheme import train_individual, train_cross_subject
from training_scheme import test_within_subject, test_cross_subject, model_fusion

    
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
    cfg['log_file'] = f'log/siamese_{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch'
    cfg['model_dir'] = f'{cfg["model_dir"]}siamese{cfg["backbone"]}_{cfg["num_window"]}window_{cfg["pairing"]}pair_{cfg["scenario"]}_{cfg["EEG_ch"]}ch/'
    if not os.path.exists(cfg['model_dir']):
        os.makedirs(cfg['model_dir'])

    if args.mode == "train":
        if args.scenario == 'cross_subject':
            train_cross_subject(cfg)
        # elif args.scenario == "cross_subject_MF":
        #     train_individual(cfg)
        #     model_fusion(cfg)
        elif args.scenario == 'within_subject':
            for i in range(4):
                train_individual(cfg)
                test_within_subject(cfg)
        else:
            raise ValueError('Invalid scenario')

    elif args.mode == "inference":
        if args.scenario == 'within_subject':
            test_within_subject(cfg)
        elif args.scenario == 'cross_subject':
            test_cross_subject(cfg)
        else:
            raise ValueError('Invalid scenario')

    else:
        raise ValueError('Invalid mode')


if __name__ == "__main__":
    main()
    
    

    
    
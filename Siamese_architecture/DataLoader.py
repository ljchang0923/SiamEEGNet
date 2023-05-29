import numpy as np
import torch
from torch import from_numpy as np2TT
from torch.utils.data import TensorDataset, DataLoader, Dataset
import random


def dataloader(data, truth, session, mode, cfg):
    
    data = np2TT(data)
    truth = np2TT(truth)

    print("x train:　", type(data), data.size())
    train_dataset = Pair_Dataloader(data, truth, session, mode, cfg['pairing'])
    
    if mode == 'train' or mode == 'baseline':
        dl = DataLoader(
            dataset = train_dataset,
            batch_size = cfg['batch_size'],
            shuffle = cfg["shuffle"] == "True",
            num_workers = 4,
            pin_memory=True,
        )
    elif mode == 'test':
        dl = DataLoader(
            dataset = train_dataset,
            batch_size = len(train_dataset),
            shuffle = False,
            num_workers = 4,
            pin_memory=True
        )
    
    return dl

class Pair_Dataloader(Dataset):
    def __init__(self, data, truth, session, mode, pairing=1):
        self.data = data
        self.truth = truth
        self.session = session
        self.mode = mode
        self.pairing = pairing

    def __len__(self):
        return len(self.data)		

    def __getitem__(self, index):
        x = self.data[index]
        y = self.truth[index].view(1, 1)

        if self.mode=='train':
            population = range(self.session[index][0], self.session[index][1])
            sample_idx = random.sample(population, self.pairing)
        elif self.mode =='test' or self.mode == 'baseline':
            sample_idx = [self.session[index][0]]
        else:
            raise ValueError('Invalid mode')

        pair = []
        label = []
        for i in range(len(sample_idx)):
            sample_data = self.data[sample_idx[i]]
            sample_truth = self.truth[sample_idx[i]].view(1,1)

            pair.append(torch.cat((sample_data, x), 0))
            label.append(torch.cat((y, sample_truth), 0))

        x_pair = torch.stack(pair)
        label_pair = torch.stack(label)

        return x_pair, label_pair


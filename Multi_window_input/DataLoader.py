import numpy as np
import torch
from torch import from_numpy as np2TT
from torch.utils.data import TensorDataset, DataLoader, Dataset
import random


def dataloader(data, truth, session, mode, cfg):

    x = np2TT(data)
    y = np2TT(truth).unsqueeze(1)
    dataset = Dataloader(x, y)

    if mode == 'train':
        dl = DataLoader(
            dataset = dataset,
            batch_size = cfg['batch_size'],
            shuffle = cfg["shuffle"] == "True",
            num_workers = 4,
            pin_memory=True
        )
    elif mode == 'test':
        dl = DataLoader(
            dataset = dataset,
            batch_size = len(x),
            shuffle = False,
            num_workers = 4,
            pin_memory=True
        )
    
    return dl

class Dataloader(Dataset):
    def __init__(self, data, truth):
        self.data = data
        self.truth = truth

    def __len__(self):
        return len(self.data)		

    def __getitem__(self, index):
        x = self.data[index]
        y = self.truth[index]

        return x, y


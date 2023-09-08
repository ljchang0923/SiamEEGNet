import numpy as np
import torch
from torch import from_numpy as np2TT
from torch.utils.data import DataLoader, Dataset
import random

def get_dataloader_4_cross_sub(sub_list, data, truth, ts_sub_idx, args):
    
    # use the subject right after the testing subject as validation subject
    if ts_sub_idx == len(sub_list)-1:
        val_sub_idx = 2
    else:
        val_sub_idx = ts_sub_idx + 1
    
    val_data = np.array(data[sub_list[val_sub_idx]][0], dtype=np.float32)
    val_truth = truth[sub_list[val_sub_idx]][0].astype('float32')
    val_session_bound = np.tile([0, val_data.shape[0]-1], (val_data.shape[0], 1))

    # Gathering cross-subject training data
    train_data = []
    train_truth = []
    tr_session_bound = []
    low = 0
    for tr_sub_idx in range(len(sub_list)):
        if tr_sub_idx == ts_sub_idx or tr_sub_idx == val_sub_idx:
            continue
        for sess_idx in range(len(data[sub_list[tr_sub_idx]])):
            train_data = train_data + data[sub_list[tr_sub_idx]][sess_idx]
            train_truth.append(truth[sub_list[tr_sub_idx]][sess_idx].astype('float32'))
            data_len = len(data[sub_list[tr_sub_idx]][sess_idx])
            tr_session_bound.append(np.tile([low, low + data_len - 1], (data_len, 1)))
            low += data_len

    train_data = np.array(train_data, dtype=np.float32) # (#total training trial, #window, #channel, #timepoint)
    train_truth = np.concatenate(train_truth, 0) # (#total training trial, )
    tr_session_bound = np.concatenate(tr_session_bound, 0) # (#total training trial, 2)

    # Wrap up as dataloader
    train_dl = get_dataloader(train_data, train_truth, tr_session_bound, 'train', **args)
    val_dl = get_dataloader(val_data, val_truth, val_session_bound, 'test', **args)

    del train_data, train_truth, val_data, val_truth, tr_session_bound, val_session_bound

    return train_dl, val_dl


def get_dataloader(data, truth, sess_bound=None, mode = 'train', batch_size=20, shuffle=True, pairing=1, method='siamese', **kwargs):
    
    data = np2TT(data)
    truth = np2TT(truth)

    print("x train:　", type(data), data.size())
    if method == 'siamese':
        dataset = pair_Dataset(data, truth, sess_bound, mode, pairing)
    else:
        dataset = normal_dataset(data, truth)
    
    if mode == 'train' or mode == 'baseline':
        dl = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = 4,
            pin_memory=True,
        )
    elif mode == 'test':
        dl = DataLoader(
            dataset = dataset,
            batch_size = len(dataset),
            shuffle = False,
            num_workers = 4,
            pin_memory=True
        )
    
    return dl

class pair_Dataset(Dataset):
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

class normal_dataset(Dataset):
    def __init__(self, data, truth):
        self.data = data
        self.truth = truth

    def __len__(self):
        return len(self.data)       

    def __getitem__(self, index):
        x = self.data[index]
        y = self.truth[index].view(-1)

        return x, y


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import random
import itertools

continous_features = range(0,13)
#categorical_features = range(13,39)
categorical_features = range(13,65)

class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient 
    dataloader tool provided by PyTorch.
    """ 
    def __init__(self, root, train=True, train_file="train.txt", test_file="test.txt"):
        """
        Initialize file path and train/test mode.

        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        super(CriteoDataset, self).__init__()
        self.continous_features = continous_features
        self.categorical_features = categorical_features
        self.root = root
        self.train = train

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, train_file), header=None)
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, test_file), header=None)
            self.test_data = data.iloc[:, :].values
        del data
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xi = torch.cat([torch.zeros(len(self.continous_features),dtype=torch.int64), torch.tensor(dataI[self.categorical_features], dtype=torch.int64)], dim=0).unsqueeze(-1)
            Xv = torch.cat([torch.tensor(dataI[self.continous_features],dtype=torch.float32), torch.ones(len(self.categorical_features),dtype=torch.float32)], dim=0).type(torch.float32)
            return Xi, Xv, targetI
        else:
            dataI = self.test_data[idx, :]
            Xi = torch.cat([torch.zeros(len(self.continous_features),dtype=torch.int64), torch.tensor(dataI[self.categorical_features], dtype=torch.int64)], dim=0).unsqueeze(-1)
            Xv = torch.cat([torch.tensor(dataI[self.continous_features],dtype=torch.float32), torch.ones(len(self.categorical_features),dtype=torch.float32)], dim=0).type(torch.float32)
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)

class OneLabelDataset(Dataset):
    def __init__(self, data, target_label, max_cached = 20000000):
        super(OneLabelDataset, self).__init__()
        self.target_label = target_label
        self.continous_features = continous_features
        self.categorical_features = categorical_features
        self.data = data
        self.f = open(self.data['file_path'], 'rb')
        self.collect()
        self.cached = {}
        self.max_cached = max_cached

    def close(self):
        self.f.close()

    def collect(self):
        self.train_data = []
        for i, (offset, label) in enumerate(zip(self.data['offset'], self.data['label'])):
            if label == self.target_label:
                self.train_data.append( offset )
        self.train_data = torch.tensor(self.train_data, dtype=torch.int64)

    def load_data(self, idx):
        keys = self.cached.keys()
        if not idx in self.cached:
            self.f.seek( self.train_data[idx] )
            line = self.f.readline().decode('utf-8').rstrip().split(',')
            feat = [float(f) for f in line]
            data = torch.tensor(feat, dtype=torch.float32)
            self.cached[idx] = data
            if len(self.cached) > self.max_cached:
                del_idx = random.randint(0,len(keys)-1)
                key_val = next(itertools.islice(iter(keys),del_idx,None))
                if key_val != idx:
                    del self.cached[key_val]

    def to_instance(self, idx):
        self.load_data(idx)
        data = self.cached[idx]
        return data, torch.tensor(self.target_label, dtype=torch.int64)

    def __getitem__(self, idx):
        dataI, targetI = self.to_instance(idx)
        Xi = torch.cat([torch.zeros(len(self.continous_features),dtype=torch.int64), torch.tensor(dataI[self.categorical_features], dtype=torch.int64)], dim=0).unsqueeze(-1)
        Xv = torch.cat([torch.tensor(dataI[self.continous_features],dtype=torch.float32), torch.ones(len(self.categorical_features),dtype=torch.float32)], dim=0)
        return Xi, Xv, targetI

    def __len__(self):
        return len(self.train_data)

def build_offset_with_label(large_file_path):
    offset_dict = []
    label_dict = []
    with open(large_file_path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            offset_dict.append(offset)
            line = line.decode('utf-8').rstrip('\n').split(',')
            label_dict.append(int(line[-1]))
    return {'offset': offset_dict, 'label': label_dict, 'file_path': large_file_path}

def test_offset(large_file_path, offset):
    with open(large_file_path, "rb") as f:
        for i in range(0,100,10):
            f.seek( offset[i] )
            line = f.readline()
            print("{} - {}".format(i, line))
        
def get_split_dataset(root, train_file):
    large_file_path = os.path.join(root, train_file)
    data = build_offset_with_label(large_file_path)
    pos_dataset = OneLabelDataset(data, 1)
    neg_dataset = OneLabelDataset(data, 0)
    return pos_dataset, neg_dataset

class MultipleIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = []
        for i, loader_iter in enumerate(self.loader_iters):
            try:
                bx = loader_iter.next()
                batches.append(bx)
            except StopIteration:
                if i == self.my_loader.pivot_loader:
                    raise StopIteration()
                # once try again
                self.loader_iters[i] = iter(self.my_loader.loaders[i])
                bx = self.loader_iters[i].next()
                batches.append(bx)
        return self.my_loader.combine_batch(batches)

# Python 2 compatibility
    next = __next__

    def __len__(self):
        return len(self.my_loader)

    
class MultipleLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time 
    taking a batch from each of them and then combining these several batches 
    into one. This class mimics the `for batch in loader:` interface of 
    pytorch `DataLoader`.
    Args: 
        loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders, pivot_loader):
        self.loaders = loaders
        self.pivot_loader = pivot_loader

    def __iter__(self):
        return MultipleIter(self)

    def __len__(self):
        return len(self.loaders[self.pivot_loader])

    def shuffle_batch(self, batch):
        indices = torch.randperm(len(batch[0]))
        batch = [col[indices] for col in batch]
        return batch

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        result = [[] for _ in range(len(batches[0]))]
        for xi in range(len(batches)):
            ns = batches[xi]
            for ci in range(len(ns)):
                result[ci].append( ns[ci] )
        merged = self.shuffle_batch([torch.cat(rs,dim=0) for rs in result])
        return merged

import numpy as np
import math
import torch
import torch.optim as optim
import radam
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import random
import os

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset, get_split_dataset, MultipleLoader

def generate_sample_idx(dataset, train_split_ratio=0.8, debug=False):
    total_set = len(dataset)
    samples_idx = np.arange(0,total_set)
    np.random.shuffle(samples_idx)
    num_train = math.floor(len(samples_idx)*train_split_ratio)
    num_valid = len(samples_idx) - num_train
    train_idx = samples_idx[:num_train]
    valid_idx = samples_idx[num_train:]
    return train_idx, valid_idx

def split_train_and_valid(pos_dataset, neg_dataset, debug=False):
    if debug:
        samples_idx = np.arange(0,256)
        pos_idx = samples_idx
        neg_idx = samples_idx
        return [pos_idx, neg_idx], [pos_idx, neg_idx]
    else:
        pos_train_idx, pos_valid_idx = generate_sample_idx(pos_dataset)
        neg_train_idx, neg_valid_idx = generate_sample_idx(neg_dataset)
        return [pos_train_idx, neg_train_idx], [pos_valid_idx, neg_valid_idx]

seed = 20170705
batch_size = 256
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#load_model = "./chkp.20190919/model.pth"
load_model = None
train_file = "train_large.txt"
feature_sizes_file = "feature_sizes_large.txt"
debug = False
#`train_file = "train.txt"
#`feature_sizes_file = "feature_sizes.txt"
#`debug = True

# load data
pos_dataset, neg_dataset = get_split_dataset('./data', train_file)
datasets = [pos_dataset, neg_dataset]

# split trani and valid set
train_idx, valid_idx = split_train_and_valid(pos_dataset, neg_dataset, debug)
loader_train = MultipleLoader([DataLoader(ds, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(s_idx)) for ds, s_idx in zip(datasets, train_idx)], pivot_loader=0)

loader_val = MultipleLoader([DataLoader(ds, batch_size=500, sampler=sampler.SubsetRandomSampler(s_idx)) for ds, s_idx in zip(datasets, valid_idx)], pivot_loader=0)

# loader
feature_sizes = np.loadtxt('./data/{}'.format(feature_sizes_file), delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

model = DeepFM(feature_sizes, use_cuda=True, overfitting=debug)
if not load_model is None and os.path.exists(load_model):
    model_state = torch.load( load_model )
    model.load_state_dict( model_state['model_state_dict'] )
    del model_state
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
optimizer = radam.RAdam(model.parameters(), lr=1e-3, weight_decay=0.0)
model.fit(loader_train, loader_val, optimizer, epochs=1000, verbose=True, print_every=1000, checkpoint_dir="./chkp")

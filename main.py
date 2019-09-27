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
from data.dataset import CriteoDataset

def split_train_and_valid(train_data, debug=False):
    if debug:
        samples_idx = np.arange(0,128)
        train_idx = samples_idx
        valid_idx = samples_idx
        return train_idx, valid_idx
    else:
        total_set = len(train_data)
        samples_idx = np.arange(0,total_set)
        np.random.shuffle(samples_idx)
        num_train = math.floor(len(samples_idx)*0.8)
        num_valid = len(samples_idx) - num_train
        train_idx = samples_idx[:num_train]
        valid_idx = samples_idx[num_train:]
        return train_idx, valid_idx

seed = 20170705
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#load_model = "./chkp.20190919/model.pth"
load_model = None
train_file = "train_large.txt"
feature_sizes_file = "feature_sizes_large.txt"
debug = False
#train_file = "train.txt"
#feature_sizes_file = "feature_sizes.txt"
#debug = True

# load data
train_data = CriteoDataset('./data', train=True, train_file=train_file)

# split trani and valid set
train_idx, valid_idx = split_train_and_valid(train_data, debug)

# loader
loader_train = DataLoader(train_data, batch_size=256, sampler=sampler.SubsetRandomSampler(train_idx), num_workers=0)
loader_val = DataLoader(train_data, batch_size=1000, sampler=sampler.SubsetRandomSampler(valid_idx), num_workers=0)

feature_sizes = np.loadtxt('./data/{}'.format(feature_sizes_file), delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

model = DeepFM(feature_sizes, use_cuda=True, overfitting=debug)
if not load_model is None and os.path.exists(load_model):
    model_state = torch.load( load_model )
    model.load_state_dict( model_state['model_state_dict'] )
    del model_state
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
optimizer = radam.RAdam(model.parameters(), lr=1e-2, weight_decay=0.0)
model.fit(loader_train, loader_val, optimizer, epochs=1000, verbose=True, print_every=1000, checkpoint_dir="./chkp")

import torch
import numpy as np
import os
import gzip
import shutil

from torch.utils.data import DataLoader
from torch.utils.data import sampler
from model.DeepFM import DeepFM
from data.dataset import CriteoDataset
from tqdm import tqdm

batch_size = 128
test_file = "test_large.txt"
feature_sizes_file = "feature_sizes_large.txt"
#checkpoint_dir = "./chkp/model.pth"
checkpoint_dir = "./chkp/best.pth"
#test_file = "test.txt"
#feature_sizes_file = "feature_sizes_large.txt"

#test_file = "test.txt"
#feature_sizes_file = "feature_sizes.txt"

test_data = CriteoDataset('./data', train=False, test_file = test_file)
print("dataset_size: {}".format(len(test_data)))
loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, sampler=sampler.SequentialSampler(test_data))

feature_sizes = np.loadtxt('./data/{}'.format(feature_sizes_file), delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
model = DeepFM(feature_sizes, use_cuda=True)


model_state = torch.load( checkpoint_dir )

model.load_state_dict( model_state['model_state_dict'] )
model.eval()


print("loaded with loss={}, best_score={}, epoch={}".format(model_state['loss'], model_state['best_score'], model_state['epoch']))
print("num_batches:", len(loader_test))
with torch.no_grad():
    b0 = 60000000
    i = 0
    predict_file = "predicted_test.csv"
    with open(predict_file, "w") as f:
        f.write("Id,Predicted\n")
        for _, (xi, xv) in tqdm(enumerate(loader_test), total=len(loader_test)):
            #y = (torch.sigmoid(model(xi,xv)) > 0.5).type(torch.int64)
            y = torch.softmax(model(xi,xv),dim=1)[:,1]
            y =  y.numpy().tolist()
            index = [j + i + b0 for j in range(len(y))]
            f.write('\n'.join(["{},{}".format(idx,v) for idx, v in zip(index,y)]))
            i += len(y)
            if len(y) < batch_size:
                print("last batch: {} / {} / {}".format(i, len(y), len(test_data)))
            f.write("\n")
    with open(predict_file, 'rb') as f_in:
        with gzip.open(predict_file + ".gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


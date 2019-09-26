# -*- coding: utf-8 -*-

"""
A pytorch implementation of DeepFM for rates prediction problem.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1, overfitting=False):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.overfitting = overfitting
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if dropout is not None:
            if not self.overfitting:
                scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v):
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

		# calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat).squeeze(dim=1)
        return output


class DeepFM(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this 
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, embedding_size=8,
                 hidden_dims=[200,200,200,200,200],
                 dropout=[0.5, 0.5, 0.5], 
                 
                 use_cuda=True, verbose=False, overfitting=False):
        """
        Initialize a new network

        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.dtype = torch.long
        self.output_dim = 4
        self.overfitting = overfitting
        #self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        """
            init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + self.hidden_dims
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'att_'+str(i), MultiHeadAttention(self.field_size, all_dims[0], overfitting=self.overfitting))
        self.avg_acc = None

        num_f1 = self.field_size
        num_f2 = self.embedding_size
        self.merge_linear = nn.ModuleList( [nn.Linear(num_f1,self.output_dim), nn.Linear(num_f2, self.output_dim), nn.Linear(self.field_size*self.embedding_size, self.output_dim)] )

        num_ffw = self.output_dim * 3
        self.ffw1 = nn.Linear(num_ffw, num_ffw)
        self.ffw_dn = nn.Dropout(0.5)
        self.fm_dense_linear = nn.Linear(num_ffw, 2)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.fm_dense_linear.weight, 1.)

    def forward(self, Xi, Xv):
        """
        Forward process of network. 

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """

        # average term
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
        #f1 = F.normalize(torch.cat(fm_first_order_emb_arr, 1), dim=1)
        f1 = torch.cat(fm_first_order_emb_arr, 1)

        # use 2xy = (x+y)^2 - (x^2 + y^2) reduce calculation
        xv = torch.stack([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)], dim=1)
        xv = F.normalize(xv,dim=1)
        s1 = torch.sum(xv,dim=1).pow(2.0)
        s2 = torch.sum(xv.pow(2.0), dim=1)
        f2 = 0.5 * (s1 - s2)
        self.xv = xv
        self.s1 = s1
        self.s2 = s2

        """
            deep part
        """
        deep_emb = torch.flatten(xv, start_dim=1)
        deep_out = deep_emb
        deep_k = deep_emb
        deep_q = deep_emb
        for i in range(1,len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'att_'+str(i))(deep_k, deep_q, deep_out)
        
        """
            sum
        """
        self.f1 = self.merge_linear[0](f1)
        self.f2 = self.merge_linear[1](f2)
        self.deep_out = self.merge_linear[2](deep_out.squeeze(1))
        #total_sum = torch.sum(f1, 1) + torch.sum(f2, 1) + torch.sum(deep_out, 1) + self.bias
        #self.final_out = torch.stack([torch.sum(f1, 1), torch.sum(f2, 1), torch.sum(deep_out, 1)],dim=1)
        #self.final_out = torch.stack([self.f1, self.f2, self.deep_out], dim=1)
        self.ffw_in = torch.cat([self.f1, self.f2, self.deep_out], dim=1)
        self.ffw_out = F.relu(self.ffw1(self.ffw_in))
        if not self.overfitting:
            self.ffw_out = self.ffw_dn(self.ffw_out)
        total_sum = self.fm_dense_linear(self.ffw_out)
        return total_sum


    def l2_reg(self):
        reg_loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        for varname, param in self.named_parameters():
            if 'bias' not in varname:
                if 'fm_dense_linear' in varname:
                    reg_loss = reg_loss + 1000.*torch.norm(param)
                else:
                    reg_loss = reg_loss + torch.norm(param)
        return reg_loss

    def l1_reg(self):
        reg_loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        reg_loss = reg_loss.to(device=self.device, dtype=torch.float32)
        for varname, param in self.named_parameters():
            if 'bias' not in varname:
                reg_loss = reg_loss + torch.sum(torch.abs(param))
                #if 'fm_dense_linear' not in varname:
                #    reg_loss = reg_loss + torch.sum(torch.abs(param))
        self.last_reg = torch.sum(self.fm_dense_linear.weight)
        return reg_loss

    def smooth_one_hot(self, true_labels, smoothing = 0.0):
        """
            if smoothing == 0, it's one-hot method
            if 0 < smoothing < 1, it's smooth method
        """
        assert 0 <= smoothing < 1
        classes = 2
        confidence = 1.0 - smoothing
        label_shape = torch.Size((true_labels.size(0), classes))
        with torch.no_grad():
            true_dist = torch.empty(size=label_shape, device=true_labels.device)
            true_dist.fill_(smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
        return true_dist

    def cross_entropy(self, _input, target, size_average=True):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean

        Examples::

            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)

            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        _input = F.log_softmax(_input,dim=1)
        if size_average:
            return torch.mean(torch.sum(-target * _input, dim=1))
        else:
            return torch.sum(torch.sum(-target * _input, dim=1))

    def fit(self, loader_train, loader_val, optimizer, epochs=1, verbose=False, print_every=100, checkpoint_dir="./chkp"):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        """
        """
            load input data
        """
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except FileExistsError:
            pass
        model = self.train().to(device=self.device)
        #criterion = torch.nn.BCEWithLogitsLoss()
        #criterion = nn.NLLLoss()
        #criterion = nn.MSELoss()
        self.iter_val = iter(loader_val)
        #l2_loss = self.l2_reg()
        max_avg_acc = 0
        save_checkpoint = False
        start_checkpoint = 50000

        total_loop = 0
        for epoch in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                total_loop += 1
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.int64)
                
                total = model(xi, xv)
                reg = self.l1_reg().to(device=self.device, dtype=torch.float32)
                smooth_label = self.smooth_one_hot(y, 0.0001)
                #total = torch.softmax(total, dim=1)
                #err = criterion(total, smooth_label) 
                err = self.cross_entropy(total, smooth_label)
                fm_dense_reg = torch.abs(3.0 - self.last_reg)
                #loss = err + 1e-8*reg + fm_dense_reg
                loss = err + 1e-8*reg
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if verbose and t % print_every == 0:
                    print('Epoch: %d, Iteration %d, max_avg_acc=%.4f, loss = %.4f,%.4f,%.4f, fm_reg=%.4f' % (epoch, t, max_avg_acc, loss.item(), reg.item(), err.item(), fm_dense_reg.item()))
                    try:
                        avg_acc = self.check_accuracy(self.iter_val, model)
                        if max_avg_acc < avg_acc:
                            save_checkpoint = True
                            max_avg_acc = avg_acc
                    except StopIteration:
                        self.iter_val = iter(loader_val)
                    model.train()

                    if total_loop > start_checkpoint and save_checkpoint:
                        self.save_model(epoch, loss.item(), checkpoint_dir, "best.pth")
                        save_checkpoint = False
                if t > start_checkpoint and t%100000 == 0:
                    self.save_model(epoch, loss.item(), checkpoint_dir, "model.pth")

                    
    def save_model(self, epoch, loss, checkpoint_dir, filename):
        checkpoint_file = os.path.join( checkpoint_dir, filename )
        torch.save( {'epoch': epoch,
            'loss': loss,
            'model_state_dict': self.state_dict()}, checkpoint_file )
    
    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            xi, xv, y = next(loader)
            xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
            xv = xv.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)
            total = model(xi, xv)
            preds = (torch.argmax(torch.softmax(total,dim=1),dim=1)).type(torch.float32)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            try:
                acc = float(num_correct) / num_samples
                if self.avg_acc is None:
                    self.avg_acc = acc
                else:
                    self.avg_acc = 0.9 * self.avg_acc + 0.1 * acc
                print('Got %d / %d correct (%.2f%%), avg_acc=%.2f%%' % (num_correct, num_samples, 100 * acc, 100 * self.avg_acc))
                return self.avg_acc
            except ZeroDivisionError as e:
                print(e)
                return None




                        

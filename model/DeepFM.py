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

    def attention(self, q, k, v, d_k):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if not self.overfitting:
            scores = self.dropout(scores)
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
        scores = self.attention(q, k, v, self.d_k)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
                / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        #norm = self.alpha * F.normalize(x,dim=-1) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout = 0.1, overfitting=False):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.d_model = d_model
        self.overfitting = overfitting
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x1 = self.linear_1(x)
        if x1.dim() == 3:
            x1 = F.relu(self.bn( x1.transpose(1,2) ).transpose(1,2))
        else:
            x1 = F.relu(self.bn( x1 ))
        if not self.overfitting:
            x1 = self.dropout(x1)
        x = self.linear_2(x1)
        return x

class ClassificationLayer(nn.Module):
    def __init__(self, d_model, output, d_ff=128, dropout = 0.1, overfitting=False):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.overfitting = overfitting
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_ff)
        self.linear_2 = nn.Linear(d_ff, output)

    def forward(self, x):
        x1 = self.linear_1(x)
        x1 = F.relu(self.bn( x1 ))
        if not self.overfitting:
            x1 = self.dropout( x1 )
        x = self.linear_2(x1)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1, overfitting=False):
        super().__init__()
        self.overfitting = overfitting
        self.heads = heads
        self.d_model = d_model
        self.norm1 = Norm(self.d_model)
        self.norm2 = Norm(self.d_model)
        self.ff = FeedForward(d_model, overfitting = overfitting)
        self.att = MultiHeadAttention(self.heads, self.d_model, dropout=dropout, overfitting=self.overfitting)
        self.dn1 = nn.Dropout(dropout)
        self.dn2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm1(x)
        att = self.att(x2,x2,x2)
        if not self.overfitting:
            att = self.dn1(att)
        x = x + att
        x2 = self.norm2(x)
        ff_out = self.ff(x2)
        if not self.overfitting:
            ff_out = self.dn2(ff_out)
        x = x + ff_out
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len].clone().detach().requires_grad_(True)
        return x

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
            setattr(self, "encoder_"+str(i), EncoderLayer(1, self.embedding_size, overfitting=self.overfitting))
        self.norm2 = Norm(self.embedding_size)
        self.avg_acc = None
        self.avg_loss = None

        num_f1 = self.field_size
        num_f2 = self.embedding_size
        self.merge_linear = nn.ModuleList( [nn.Linear(num_f1,self.output_dim), nn.Linear(num_f2, self.output_dim), nn.Linear(self.field_size*self.embedding_size, self.output_dim)] )

        num_ffw = self.output_dim * 3
        self.fm_dense_linear = ClassificationLayer(num_ffw, 2, dropout=0.1, overfitting=self.overfitting)
        self.pe = PositionalEncoder(self.embedding_size, max_seq_len=self.field_size)
        self.init_weight()

    def check_num(self, data):
        #if torch.sum( torch.isinf(data)).item() > 0 or torch.sum(torch.isnan(data)) > 0:
        if torch.sum( data > 1000.0).item() > 0 or torch.sum( data < -1000.0 ) > 0:
            breakpoint()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Embedding):
                nn.init.sparse_(m.weight, 0.01)
        #nn.init.constant_(self.fm_dense_linear.weight, 1.)

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
        f1 = torch.cat(fm_first_order_emb_arr, 1)

        # use 2xy = (x+y)^2 - (x^2 + y^2) reduce calculation
        xv = torch.stack([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)], dim=1)

        #self.check_num(xv)
        xv = F.normalize(xv,dim=-1)
        #self.check_num(xv)
        s1 = torch.sum(xv,dim=1).pow(2.0)
        s2 = torch.sum(xv.pow(2.0), dim=1)
        f2 = 0.5 * (s1 - s2)
        self.xv = xv
        self.s1 = s1
        self.s2 = s2

        """
            deep part
        """
        deep_out = self.pe(xv)
        for i in range(1,len(self.hidden_dims) + 1):
            deep_out = getattr(self, "encoder_" + str(i))(deep_out)
            #self.check_num(deep_out)
        deep_out = self.norm2(deep_out)
        #self.check_num(deep_out)
        deep_out = torch.flatten(deep_out, start_dim=1)

        self.f1 = self.merge_linear[0](f1)
        self.f2 = self.merge_linear[1](f2)
        self.deep_out = self.merge_linear[2](deep_out.squeeze(1))
        #total_sum = torch.sum(f1, 1) + torch.sum(f2, 1) + torch.sum(deep_out, 1) + self.bias
        #self.final_out = torch.stack([torch.sum(f1, 1), torch.sum(f2, 1), torch.sum(deep_out, 1)],dim=1)
        #self.final_out = torch.stack([self.f1, self.f2, self.deep_out], dim=1)
        self.ffw_in = torch.cat([self.f1, self.f2, self.deep_out], dim=1)
        #self.ffw_out = F.relu(self.ffw1(self.ffw_in))
        #if not self.overfitting:
        #    self.ffw_out = self.ffw_dn(self.ffw_out)
        #self.check_num(self.ffw_out)
        #total_sum = self.fm_dense_linear(self.ffw_out)
        total_sum = self.fm_dense_linear(self.ffw_in)
        return total_sum

    def l1_reg(self):
        reg_loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        reg_loss = reg_loss.to(device=self.device, dtype=torch.float32)
        for varname, param in self.named_parameters():
            if 'bias' not in varname:
                reg_loss = reg_loss + torch.sum(torch.abs(param))
            #if 'fm_second_order_embeddings' in varname:
            #    reg_loss = reg_loss + 100000000.0*torch.sum(torch.abs(param))
        #self.last_reg = torch.sum(self.fm_dense_linear.weight)
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
        #max_avg_score = 0
        max_avg_score = 99999999
        save_checkpoint = False
        start_checkpoint = 50000
        fm_dense_reg = torch.zeros(1)

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
                #fm_dense_reg = torch.abs(3.0 - self.last_reg)
                #loss = err + 1e-8*reg + fm_dense_reg
                loss = err + 1e-8*reg
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.000001)
                    optimizer.step()
                else:
                    print("BUG!!")
                    print('Epoch: %d, Iteration %d, max_avg_score=%.4f, avg(%.4f,%.4f), loss = %.4f,%.4f,%.4f, fm_reg=%.4f' % (epoch, t, max_avg_score, avg_acc, avg_loss, loss.item(), reg.item(), err.item(), fm_dense_reg.item()))
                    breakpoint()

                if verbose and t % print_every == 0:
                    try:
                        avg_acc, avg_loss = self.check_accuracy(self.iter_val, model)
                        #if max_avg_score < avg_loss:
                        if max_avg_score > avg_loss:
                            save_checkpoint = True
                            max_avg_score = avg_loss
                        print('Epoch: %d, Iteration %d, max_avg_score=%.4f, avg(%.4f,%.4f), loss = %.4f,%.4f,%.4f, fm_reg=%.4f' % (epoch, t, max_avg_score, avg_acc, avg_loss, loss.item(), reg.item(), err.item(), fm_dense_reg.item()))
                    except StopIteration:
                        self.iter_val = iter(loader_val)
                    model.train()

                    if total_loop > start_checkpoint and save_checkpoint:
                        self.save_model(epoch, loss.item(), checkpoint_dir, "best.pth")
                        save_checkpoint = False
                if total_loop > start_checkpoint and t%100000 == 0:
                    self.save_model(epoch, loss.item(), checkpoint_dir, "model.pth")

                    
    def save_model(self, epoch, loss, checkpoint_dir, filename):
        checkpoint_file = os.path.join( checkpoint_dir, filename )
        torch.save( {'epoch': epoch,
            'loss': loss,
            'model_state_dict': self.state_dict()}, checkpoint_file )
   
    def logloss(self, y, p, eps=1e-30):
        p1 = torch.log(p + eps)
        p2 = torch.log(1.0 - p + eps)
        L = -(y *p1  + (1.0 - y) *p2).mean()
        #s = (1. + torch.exp(p))
        #L = -(y*torch.log( torch.exp(p) /s ) + (1.-y) *torch.log(1./s)).mean()
        return L

    def check_accuracy(self, loader, model):
        print('Checking accuracy on validation set')
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            xi, xv, y = next(loader)
            xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
            xv = xv.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)
            total = model(xi, xv)
            cvr = torch.softmax(total,dim=1)
            prob = cvr[:,1]
            loss = self.logloss(y, prob)
            if torch.isnan(loss):
                print("Valid Bug!")
                breakpoint()
            loss = loss.item()
            preds = (torch.argmax(cvr,dim=1)).type(torch.float32)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            try:
                acc = float(num_correct) / num_samples
                if self.avg_acc is None:
                    self.avg_acc = acc
                    self.avg_loss = loss
                else:
                    self.avg_acc = 0.9 * self.avg_acc + 0.1 * acc
                    self.avg_loss = 0.9 * self.avg_loss + 0.1 * loss
                print("Got %d / %d correct (%.2f%%), avg_acc=%.2f%%, avg_loss=%.2f, loss=%.2f" % (num_correct, num_samples, 100 * acc, 100 * self.avg_acc, self.avg_loss, loss))
                return self.avg_acc, self.avg_loss
            except ZeroDivisionError as e:
                print(e)
                return None, None




                        

#coding:utf-8
import os
import torch
import pickle
import importlib
import torch.nn as nn
from torch import autograd

from ..models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)

#a network consists of model and loss
class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()
        #can be any model, such as squeeze, cornernet..
        self.model = model
        self.loss  = loss
    #implement the forward method
    def forward(self, xs, ys):
        preds = self.model(xs)
        loss,focal_loss,pull_loss,push_loss,off_loss  = self.loss(preds, ys)
        return loss,focal_loss,pull_loss,push_loss,off_loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    #overwirte forward
    def forward(self, xs, **kwargs):
        return self.module(xs, **kwargs)

#netowork->DummyModule->squeeze model
class NetworkFactory(object):
    def __init__(self, system_config, model, gpu=None):
        super(NetworkFactory, self).__init__()

        self.system_config = system_config

        self.gpu     = gpu
        self.model   = DummyModule(model)
        self.loss    = model.loss
        self.network = Network(self.model, self.loss)


        self.network = DataParallel(self.network, chunk_sizes=system_config.chunk_sizes)

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))

        if system_config.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_config.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_config.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def _t_cuda(self, xs):
        if type(xs) is list:
            return [x.cuda(self.gpu, non_blocking=True) for x in xs]
        return xs.cuda(self.gpu, non_blocking=True)

    def train(self, xs, ys):
        with autograd.detect_anomaly():
            xs = [self._t_cuda(x) for x in xs]
            ys = [self._t_cuda(y) for y in ys]
            
            #把loss关于weight的导数变成0，即把梯度置零
            self.optimizer.zero_grad()
            #执行model(x)的时候，底层自动调用forward计算结果
            loss,focal_loss,pull_loss,push_loss,off_loss = self.network(xs, ys)
            #计算loss
            loss = loss.mean()
            
            #added
            focal_loss = focal_loss.mean()
            pull_loss = pull_loss.mean()
            push_loss = push_loss.mean()
            off_loss = off_loss.mean()
        
            #反向传播求梯度
            loss.backward()
            #更新参数，即w* = w+alpha*grad
            self.optimizer.step()
    
            return loss,focal_loss,pull_loss,push_loss,off_loss

    def validate(self, xs, ys):
        with torch.no_grad():
            xs = [self._t_cuda(x) for x in xs]
            ys = [self._t_cuda(y) for y in ys]

#             loss = self.network(xs, ys)
            loss,focal_loss,pull_loss,push_loss,off_loss = self.network(xs, ys)
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [self._t_cuda(x) for x in xs]
            return self.model(xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration):
        cache_file = self.system_config.snapshot_file.format(iteration)
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def save_params(self, iteration):
        cache_file = self.system_config.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)

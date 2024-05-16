# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

import torch
import torch.nn as nn

class Regularization(object):
    def __init__(self, weight_decay = 0,alpha = 0):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.weight_decay_L2 = weight_decay/2 * (1-alpha)
        self.weight_decay_L1 = weight_decay * alpha

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss_l1,reg_loss_l2 = 0,0
        for name, w in model.named_parameters():
            if 'weight' in name:
                if self.weight_decay_L2 != 0:
                    reg_loss_l2 = reg_loss_l2 + torch.norm(w, p=2) 
                if self.weight_decay_L1 != 0:
                    reg_loss_l1 = reg_loss_l1 + torch.norm(w, p=1) 
        reg_loss = self.weight_decay_L2 * reg_loss_l2 + self.weight_decay_L1 * reg_loss_l1
        return reg_loss

class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg'] + random.uniform(-0.01,0.01)
        self.alpha = nn.Parameter(torch.tensor(config['alpha']))
        print("==========weight_decay : {:.3f}===========alpha : {:.3f}==================".format(self.L2_reg,self.alpha))
        self.reg = Regularization(weight_decay=self.L2_reg ,alpha=self.alpha)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss
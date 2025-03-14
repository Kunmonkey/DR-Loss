import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from .utils import weight_reduce_loss
from ..registry import LOSSES
from .cross_entropy_loss import cross_entropy, _expand_binary_labels, binary_cross_entropy, partial_cross_entropy
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.nn import Parameter
import math
import os
import sys
from collections import OrderedDict
from functools import partial
from .resample_loss import ResampleLoss



@LOSSES.register_module
class DRLoss(nn.Module):
    def __init__(self,
                gamma1=1,
                gamma2=1,
                use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 focal=dict(
                     focal=True,
                     balance_param=2.0,
                     gamma=2,
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 freq_file='./class_freq.pkl',
                 num_classes = 20,
                 class_split = './class_split.pkl'):
        super(DRLoss,self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        # self.db_loss = ResampleLoss(freq_file=freq_file,
        #     use_sigmoid=True,
        #     reweight_func='rebalance',
        #     focal=focal,
        #     logit_reg=logit_reg,
        #     map_param=map_param,
        #     loss_weight=loss_weight, num_classes = num_classes,
        #     class_split = class_split)
    def forward(self,cls_score,labels,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        cls_score0 = cls_score.clone()
        cls_score0 = (1 - 2 * labels) * cls_score0
        neg_score = cls_score0 - labels * 1e12
        pos_score = cls_score0 - (1 - labels) * 1e12

        ## positive scores and negative scores
        s_p0 = pos_score * self.gamma1
        s_n0 = self.gamma1 * neg_score

        ######### DR Loss
        loss_dr = (1 + torch.exp(torch.logsumexp(s_p0,dim=0)) * torch.exp(torch.logsumexp(s_n0,dim=0))  \
             + torch.exp(torch.logsumexp(neg_score * self.gamma2,dim=0))
             ).log()

        return loss_dr 


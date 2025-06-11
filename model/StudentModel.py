'''
Description: 
Author: Rui Dong
Date: 2024-04-11 13:44:37
LastEditors: Please set LastEditors
LastEditTime: 2024-06-03 15:45:10
'''

import numpy as np
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch.nn import LayerNorm as LN
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj


class SpecMLP(nn.Module):
    def __init__(self, args):
        super(SpecMLP, self).__init__()
        self.num_features   = args.num_features
        self.stu_hidden     = args.stu_hidden       # 64
        self.num_classes    = args.num_classes
        self.dropout        = args.dropout
        self.stu_layers     = args.stu_layers
        self.distill_type   = args.distill_type
        
        self.conv1          = nn.Linear(self.num_features, self.stu_hidden)
        self.BN1            = nn.BatchNorm1d(self.stu_hidden)
        self.layers         = nn.ModuleList()
        self.BNs            = nn.ModuleList()
        
        for i in range(self.stu_layers - 1):
            self.layers.append(nn.Linear(self.stu_hidden, self.stu_hidden))
            self.BNs.append(nn.BatchNorm1d(self.stu_hidden))
            
        self.fc             = nn.Linear(self.stu_hidden, self.num_classes)
        self.BN2            = nn.BatchNorm1d(self.num_classes)
        self.weights        = nn.Parameter(torch.Tensor(self.stu_layers))
        nn.init.uniform_(self.weights, 0, 1)

    def forward(self, data):
        """ TBD

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = data.x
        middle_feats = []
        
        #* self.weights => normalization
        x = self.conv1(x)
        x = self.BN1(x)
        middle_feats.append(x)
        
        for i in range(self.stu_layers - 1):
            x = self.layers[i](x)
            x = self.BNs[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            middle_feats.append(x)
        
        #* layer residual
        if self.distill_type == "SpecMLP":
            x *= self.weights[-1]
        for i in range(self.stu_layers - 1):
            x = torch.add(x, middle_feats[i] * self.weights[i])
            # x = torch.add(x, middle_feats[i])
        x /= (self.weights.sum())
        
        x = self.fc(x)
        middle_feats.append(x)
        return F.log_softmax(x, dim=1), middle_feats
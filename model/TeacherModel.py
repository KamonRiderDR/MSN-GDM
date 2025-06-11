'''
Description: 
Author: Rui Dong
Date: 2024-04-11 13:43:25
LastEditors: Please set LastEditors
LastEditTime: 2024-05-01 10:53:57
'''

import torch
import torch.nn as nn
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv, SAGEConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.special import comb

from .Bernpro import Bern_prop


class BernNet(torch.nn.Module):
    def __init__(self, args):
        super(BernNet, self).__init__()
        self.args = args
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)
        self.m = torch.nn.BatchNorm1d(args.num_classes)
        self.prop1 = Bern_prop(args.K)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        """BernNet forward process

        Args:
            data: 'PYG' data

        Returns:
            soft label, middle features
        """
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1), x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1), x

# TODO
class ChebNet(torch.nn.Module):
    def __init__(self, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(args.num_features, 32, K=2)
        self.conv2 = ChebConv(32, args.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), x




class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args           = args
        self.num_features   = args.num_features
        self.hidden         = args.hidden
        self.num_classes    = args.num_classes
        self.n_layers       = args.n_layers
        
        self.layers         = nn.ModuleList()
        self.dropout        = args.dropout
        self.BNs            = nn.ModuleList()
        
        self.conv1          = GCNConv(self.num_features, self.hidden)
        self.BN1            = nn.BatchNorm1d(self.hidden)

        for i in range(self.n_layers - 1):
            self.layers.append(GCNConv(self.hidden, self.hidden))
            self.BNs.append(nn.BatchNorm1d(self.hidden))
                    
        self.fc1             = nn.Linear(self.hidden, self.num_classes)
    
    # def fc_forward(self, x):
    #     x = F.relu(self.fc1(x), inplace=True)
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = F.relu(self.fc2(x), inplace=True)

    #     return x
    
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index if "ogbn" not in self.args.dataset else data.adj_t
        x = self.conv1(x, edge_index)
        x = self.BN1(x)
        x = F.relu(x, inplace=True)
        for i, layer in enumerate(self.layers):
            # x = F.relu(layer(x, edge_index), inplace=True)
            x = layer(x, edge_index)
            x = self.BNs[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=1), x


class GAT(torch.nn.Module):
    def __init__(self, args, mode='cat'):
        super(GAT, self).__init__()
        self.dropout = args.dropout
        self.hidden = args.hidden
        self.num_layers = args.n_layers

        # self.embedding = Linear(args.num_features, self.hidden)
        self.conv1 = GATConv(args.num_features, self.hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(self.hidden, self.hidden))

        self.BN = nn.BatchNorm1d(self.hidden)
        self.BNs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.BNs.append(nn.BatchNorm1d(self.hidden))

        self.lin1 = Linear(self.hidden, args.num_classes)
        # self.lin2 = Linear(self.hidden, args.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.BN(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.BNs[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.lin1(x), inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1), x

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.dropout = args.dropout
        self.hidden = args.hidden
        self.num_layers = args.n_layers

        self.conv1 = SAGEConv(args.num_features, self.hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.hidden, self.hidden))
        self.BN = nn.BatchNorm1d(self.hidden)
        self.BNs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.BNs.append(nn.BatchNorm1d(self.hidden))

        self.lin1 = Linear(self.hidden, args.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.BN(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.BNs[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = F.relu(self.lin1(x))
        out = F.dropout(out, p=self.dropout, training=self.training)

        return F.log_softmax(out, dim=1), out

    def __repr__(self):
        return self.__class__.__name__

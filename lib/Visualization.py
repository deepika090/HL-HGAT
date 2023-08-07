#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:24:56 2023

@author: jinghan
"""

import numpy as np
from torch.nn import Linear, Dropout
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch_geometric.nn as gnn
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import time
from lib.Hodge_Cheb_Conv import *
from lib.Hodge_Dataset import *
import torchvision as tv
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import unbatch
from torchmetrics import F1Score
from torchmetrics.classification import BinaryF1Score

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_networkx, remove_self_loops
from torch_geometric.data import Data
from matplotlib.pyplot import figure


class HL_HGCNN(torch.nn.Module):
    def __init__(self, channels=[2,2,2], filters=[64,128,256], mlp_channels=[], K=2, node_dim=2, 
                  edge_dim=1, num_classes=1, dropout_ratio=0.0, dropout_ratio_mlp=0.0, 
                  keig=20):
        super(HL_HGCNN, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim# + keig
        self.edge_dim = edge_dim# + keig
        self.initial_channel = self.filters[0]
        
        layers = [(HodgeLaguerreConv(self.node_dim, self.initial_channel, K=K),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
                  (nn.ReLU(), 'x_t -> x_t'),
                  (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                  (HodgeLaguerreConv(self.edge_dim, self.initial_channel, K=K),
                    'x_s, edge_index_s, edge_weight_s -> x_s'),
                  (gnn.BatchNorm(self.initial_channel), 'x_s -> x_s'),
                  (nn.ReLU(), 'x_s -> x_s'),
                  (Dropout(p=dropout_ratio), 'x_s -> x_s'),
                  (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
        fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)
        setattr(self, 'HL_init_conv', fc)
        gcn_insize = self.initial_channel
            
        for i, gcn_outsize in enumerate(self.filters):

            for j in range(self.channels[i]):
                # int term
                fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize)
                setattr(self, 'NEInt{}{}'.format(i,j), fc)
                # HL node edge filtering
                layers = [(HodgeLaguerreConv(gcn_outsize, gcn_outsize, K=K),
                            'x_t, edge_index_t, edge_weight_t -> x_t'),
                          (gnn.BatchNorm(gcn_outsize), 'x_t -> x_t'),
                          (nn.ReLU(), 'x_t -> x_t'),
                          (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                          (HodgeLaguerreConv(gcn_outsize, gcn_outsize, K=K),
                            'x_s, edge_index_s, edge_weight_s -> x_s'),
                          (gnn.BatchNorm(gcn_outsize), 'x_s -> x_s'),
                          (nn.ReLU(), 'x_s -> x_s'),
                          (Dropout(p=dropout_ratio), 'x_s -> x_s'),
                          (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
                fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)
                setattr(self, 'NEConv{}{}'.format(i,j), fc)
                gcn_insize = gcn_outsize + gcn_insize
        mlp_insize = gcn_outsize*2 #sum(Node_channels)+ sum(Edge_channels)#[-1]

        layers = [(HodgeLaguerreConv(mlp_insize, 1, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),]
        fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t', layers)
        setattr(self, 'out', fc)

    def forward(self, data, device='cuda:0'):
        # 1. Obtain node embeddings
        if isinstance(data.num_node1,int):
            data.num_node1, data.num_edge1 = [data.num_node1], [data.num_edge1]
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        sout, tout = [], []
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        x_s, edge_index_s, edge_weight_s = data.x_s, data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t
        par_1 = adj2par1(data.edge_index, x_t.shape[0], x_s.shape[0])
        D = degree(data.edge_index.view(-1),num_nodes=x_t.shape[0]) + 1e-6
            
        for i, _ in enumerate(self.channels):
            for j in range(self.channels[i]):
                sout.append(x_s)
                tout.append(x_t)
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
        # 2. Readout layer
        x_t2s = torch.sparse.mm(par_1.abs().transpose(0,1), x_t)/2
        x_s = torch.cat([x_s,x_t2s], dim=-1)

        return sout, tout, self.out(x_s, edge_index_s, edge_weight_s)
    
    
    
def pdata2data(data, data1, pred):
    ei1_key = {}
    for i in range(data.edge_index.shape[1]):
        imax = max(data.edge_index[0][i],data.edge_index[1][i])
        imin = min(data.edge_index[0][i],data.edge_index[1][i])
        ekey = imax + 1000*imin
        ei1_key[int(ekey)] = pred[i]
    y_pred = torch.zeros(data1.edge_index.shape[1])
    for i in range(data1.edge_index.shape[1]):
        imax = max(data1.edge_index[0][i],data1.edge_index[1][i])
        imin = min(data1.edge_index[0][i],data1.edge_index[1][i])
        ekey = int(imax + 1000*imin)
        if ekey in ei1_key:
            y_pred[i] = ei1_key[ekey]
        else:
            print('err')
    return y_pred


def test(loader):
    model.eval()
    total_loss = 0
    N = 0
    y_f1 = []
    for idx,data in enumerate(loader):
        data = data.to(device)
        y = data.y.view(-1,1)
        with torch.no_grad():
            _, _, out = model(data)
        loss = criterion(out, y)
        total_loss += loss.item() * data.num_graphs
        data1 = trainset[idx]
        y_pred = pdata2data(data.cpu(), data1, out.cpu())
        y_f1.append(f1(y_pred.view(-1),data1.y.view(-1)))
        N += data.num_graphs

    y_f1 = torch.tensor(y_f1)
    test_perf = y_f1#.mean()
    test_loss = total_loss/N
    return test_loss, test_perf


def normalize01(x):
    return (x - x.min())/(x.max()-x.min())
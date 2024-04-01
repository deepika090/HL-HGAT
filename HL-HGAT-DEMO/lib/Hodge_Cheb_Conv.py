#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:58:34 2022

@author: jinghan
"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.nn.pool import graclus, max_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, dense_to_sparse, degree
from typing import Callable, Optional, Tuple, Union
import torch_geometric.nn as gnn
from torch_scatter import scatter_add, scatter_max, scatter_add, scatter_mean
from torch_geometric.typing import SparseTensor
import torch_sparse

from lib.Hodge_Dataset import *
from torch_geometric.utils import unbatch_edge_index,softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.utils as ut
from scipy.sparse.linalg import eigsh

###############################################
########### Modularized HL-HGAT ###############
###############################################
class SAPool(torch.nn.Module):
    def __init__(self, d=64, dk=32):
        '''
        d: input feature dim
        dk: feature dim of key & query
        '''
        super().__init__()
        self.NEAtt = MSI(d=d, dk=dk, only_att=True, sigma=nn.Sigmoid())

    def forward(self, x_t0, x_s0, par_1, D, datas, pos_ts, pos_ss, k, device='cuda:0'):     
        att_t, att_s = self.NEAtt(x_t0, x_s0, par_1, D)
        x_t0 = x_t0 * att_t
        x_s0 = x_s0 * att_s
        pos_t, pos_s = pos_ts[k], pos_ss[k]
        x_t0 = x_t0[~torch.isinf(pos_t).view(-1)]
        pos_t = pos_t[~torch.isinf(pos_t).view(-1)]
        x_t0 = scatter_mean(x_t0,pos_t.to(torch.long),dim=0)
        x_s0 = x_s0[~torch.isinf(pos_s).view(-1)]
        pos_s = pos_s[~torch.isinf(pos_s).view(-1)]
        x_s0 = scatter_mean(x_s0,pos_s.to(torch.long),dim=0)
        edge_index_s, edge_weight_s = datas[k+1].edge_index_s.to(device), datas[k+1].edge_weight_s.to(device)
        edge_index_t, edge_weight_t = datas[k+1].edge_index_t.to(device), datas[k+1].edge_weight_t.to(device)
        k += 1
        par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
        D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
        return  x_t0, x_s0, par_1, D, k, edge_index_t, edge_weight_t, edge_index_s, edge_weight_s, att_t, att_s

class MSI(nn.Module):
    # node edge interaction module
    def __init__(self, d=64, dk=32, dv=64, dl=64, only_att=False, sigma=nn.Sigmoid(), l=0.9):
        # d: input feature dim
        # dk: feature dim of key & query
        # dv: feature dim of value
        # dl: feature dim of latent
        # only_att: if true, only output the attention value
        # sigma: activation function
        super().__init__()
        dl = dv
        self.sigma = sigma
        self.dk = dk
        self.only_att = only_att
        if only_att:
            self.WQ_Node = nn.Linear(d, dk)
            self.WK_Node = nn.Linear(d, dk)
            
            self.WQ_Edge = nn.Linear(d, dk)
            self.WK_Edge = nn.Linear(d, dk)
        else:
            self.WV_Node = nn.Sequential(
                nn.Linear(d*2, dl),
                nn.BatchNorm1d(dl),
                nn.ReLU(),
                nn.Linear(dl, dv),
                nn.BatchNorm1d(dv),
                nn.ReLU())
            self.WV_Edge = nn.Sequential(
                nn.Linear(d*2, dl),
                nn.BatchNorm1d(dl),
                nn.ReLU(),
                nn.Linear(dl, dv),
                nn.BatchNorm1d(dv),
                nn.ReLU())
        self.lambda_Node = l#self.sigma(nn.Parameter(torch.zeros(1)).to(device))
        self.lambda_Edge = l#self.sigma(nn.Parameter(torch.zeros(1)).to(device))
        
    def forward(self, x_t, x_s, par, D):    
        x_s2t = (1/D).view(-1,1)*torch.sparse.mm(par.abs(), x_s)
        x_t2s = torch.sparse.mm(par.abs().transpose(0,1), x_t)/2
        # print(x_t.shape, x_s.shape, x_s2t.shape, x_t2s.shape)
        if self.only_att:
            # print(x_t.shape, x_s.shape, x_s2t.shape, x_t2s.shape,)
            a_t = self.sigma(( (1-self.lambda_Node)*(self.WQ_Edge(x_s2t)*(self.WK_Node(x_t))).sum(dim=1, keepdim=True)
                              + self.lambda_Node*(self.WQ_Node(x_t)*self.WK_Node(x_t)).sum(dim=1, keepdim=True)
                              )/np.sqrt(self.dk))
            a_s = self.sigma(( (1-self.lambda_Edge)*(self.WQ_Node(x_t2s)*(self.WK_Edge(x_s))).sum(dim=1, keepdim=True)
                              + self.lambda_Edge*(self.WQ_Edge(x_s)*self.WK_Edge(x_s)).sum(dim=1, keepdim=True)
                              )/np.sqrt(self.dk))    
            return a_t, a_s
        else:
            x_t1 = self.WV_Node(torch.cat([x_s2t, x_t], dim=-1))
            x_s1 = self.WV_Edge(torch.cat([x_t2s, x_s], dim=-1))
            return x_t1, x_s1

class HL_filter(torch.nn.Module):
    def __init__(self, channels=2, filters=32, K=4, node_dim=64, 
                edge_dim=64, dropout_ratio=0.0, leaky_slope=0.1, if_dense=True):
        '''
        HL-filtering layer
        channels: number of HL-filtering layer
        filters: number of filters in each layer
        K: polynomial order
        node_dim: input node dimension
        edge_dim: input edge dimension
        '''
        self.channels = channels
        self.filters = filters
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        gcn_outsize = self.filters
        t_insize = self.node_dim
        s_insize = self.edge_dim
        self.if_dense = if_dense
        super().__init__()

        for j in range(self.channels):
            if self.if_dense:
                fc = MSI(d=t_insize, dv=gcn_outsize)
                setattr(self, 'MSI{}'.format(j), fc)
                layers = [(HodgeLaguerreFastConv(gcn_outsize, gcn_outsize, K=K),
                            'x_t, adj_t -> x_t'),
                            (gnn.BatchNorm(gcn_outsize), 'x_t -> x_t'),
                            (nn.LeakyReLU(negative_slope=leaky_slope), 'x_t -> x_t'),
                            (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                            (HodgeLaguerreFastConv(gcn_outsize, gcn_outsize, K=K),
                            'x_s, adj_s -> x_s'),
                            (gnn.BatchNorm(gcn_outsize), 'x_s -> x_s'),
                            (nn.LeakyReLU(negative_slope=leaky_slope), 'x_s -> x_s'),
                            (Dropout(p=dropout_ratio), 'x_s -> x_s'),
                            (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
                fc = gnn.Sequential('x_t, adj_t, x_s, adj_s', layers)
                setattr(self, 'NEConv{}'.format(j), fc)
                t_insize = t_insize + gcn_outsize
                s_insize = s_insize + gcn_outsize
            else:
                layers = [(HodgeLaguerreFastConv(t_insize, gcn_outsize, K=K),
                            'x_t, adj_t -> x_t'),
                            (gnn.BatchNorm(gcn_outsize), 'x_t -> x_t'),
                            (nn.LeakyReLU(negative_slope=leaky_slope), 'x_t -> x_t'),
                            (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                            (HodgeLaguerreFastConv(s_insize, gcn_outsize, K=K),
                            'x_s, adj_s -> x_s'),
                            (gnn.BatchNorm(gcn_outsize), 'x_s -> x_s'),
                            (nn.LeakyReLU(negative_slope=leaky_slope), 'x_s -> x_s'),
                            (Dropout(p=dropout_ratio), 'x_s -> x_s'),
                            (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
                fc = gnn.Sequential('x_t, adj_t, x_s, adj_s', layers)
                setattr(self, 'NEConv{}'.format(j), fc)
                t_insize = gcn_outsize
                s_insize = gcn_outsize

    def forward(self, x_t0, edge_index_t, edge_weight_t, x_s0, edge_index_s, edge_weight_s, par_1=None, D=None):
        adj_t = SparseTensor(row=edge_index_t[0], col=edge_index_t[1], value=edge_weight_t).t()
        adj_s = SparseTensor(row=edge_index_s[0], col=edge_index_s[1], value=edge_weight_s).t()
        for j in range(self.channels):
            if self.if_dense:
                fc = getattr(self, 'MSI{}'.format(j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}'.format(j))
                x_t, x_s = fc(x_t, adj_t, x_s, adj_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
            else:
                fc = getattr(self, 'NEConv{}'.format(j))
                x_t, x_s = fc(x_t0, adj_t, x_s0, adj_s)
                x_t0,x_s0 = x_t,x_s

        return x_t0, x_s0
    
# class HL_filter(torch.nn.Module):
#     def __init__(self, channels=2, filters=32, K=4, node_dim=64, 
#                 edge_dim=64, dropout_ratio=0.0, leaky_slope=0.1, if_dense=True):
#         '''
#         HL-filtering layer
#         channels: number of HL-filtering layer
#         filters: number of filters in each layer
#         K: polynomial order
#         node_dim: input node dimension
#         edge_dim: input edge dimension
#         '''
#         self.channels = channels
#         self.filters = filters
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim
#         gcn_outsize = self.filters
#         t_insize = self.node_dim
#         s_insize = self.edge_dim
#         self.if_dense = if_dense
#         super().__init__()

#         for j in range(self.channels):
#             layers = [(HodgeLaguerreConv(t_insize, gcn_outsize, K=K),
#                         'x_t, edge_index_t, edge_weight_t -> x_t'),
#                         (gnn.BatchNorm(gcn_outsize), 'x_t -> x_t'),
#                         (nn.LeakyReLU(negative_slope=leaky_slope), 'x_t -> x_t'),
#                         (Dropout(p=dropout_ratio), 'x_t -> x_t'),
#                         (HodgeLaguerreConv(s_insize, gcn_outsize, K=K),
#                         'x_s, edge_index_s, edge_weight_s -> x_s'),
#                         (gnn.BatchNorm(gcn_outsize), 'x_s -> x_s'),
#                         (nn.LeakyReLU(negative_slope=leaky_slope), 'x_s -> x_s'),
#                         (Dropout(p=dropout_ratio), 'x_s -> x_s'),
#                         (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
#             fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)
#             setattr(self, 'NEConv{}'.format(j), fc)
#             if self.if_dense:
#                 t_insize = t_insize + gcn_outsize
#                 s_insize = s_insize + gcn_outsize
#             else:
#                 t_insize = gcn_outsize
#                 s_insize = gcn_outsize

#     def forward(self, x_t0, edge_index_t, edge_weight_t, x_s0, edge_index_s, edge_weight_s):
#         for j in range(self.channels):
#             fc = getattr(self, 'NEConv{}'.format(j))
#             x_t, x_s = fc(x_t0, edge_index_t, edge_weight_t, x_s0, edge_index_s, edge_weight_s)
#             if self.if_dense:
#                 x_t0 = torch.cat([x_t0, x_t], dim=-1)
#                 x_s0 = torch.cat([x_s0, x_s], dim=-1)
#             else:
#                 x_t0,x_s0 = x_t,x_s

#         return x_t0, x_s0

class HL_HGAT_attpool(torch.nn.Module):
    def __init__(self, channels=[2,2,2], filters=[32,64,128], mlp_channels=[], K=4, node_dim=64, 
                  edge_dim=1, num_classes=1, dropout_ratio=0.0, dropout_ratio_mlp=0.0, pool_num=2,
                  keig=0,num_nodepedge=2089):
        super(HL_HGAT_attpool, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
        self.initial_channel = self.filters[0]
        self.pool_loc = [i for i in range(pool_num)]
        self.node_embedding = Inception1D(if_readout=True)
        self.keig = keig
        self.num_nodepedge = num_nodepedge
        leaky_slope = 0.1
        
        layers = [(HodgeLaguerreFastConv(self.node_dim, self.initial_channel, K=K),
                    'x_t, adj_t -> x_t'),
                  (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
                  (nn.LeakyReLU(negative_slope=leaky_slope), 'x_t -> x_t'),
                  (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                  (HodgeLaguerreFastConv(self.edge_dim, self.initial_channel, K=K),
                   'x_s, adj_s -> x_s'),
                  (gnn.BatchNorm(self.initial_channel), 'x_s -> x_s'),
                  (nn.LeakyReLU(negative_slope=leaky_slope), 'x_s -> x_s'),
                  (Dropout(p=dropout_ratio), 'x_s -> x_s'),
                  (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
        fc = gnn.Sequential('x_t, adj_t, x_s, adj_s', layers)
        setattr(self, 'HL_init_conv', fc)
        gcn_insize = self.initial_channel
            
        for i, gcn_outsize in enumerate(self.filters):

            for j in range(self.channels[i]):
                # int term
                fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize)
                setattr(self, 'NEInt{}{}'.format(i,j), fc)
                # HL node edge filtering
                layers = [(HodgeLaguerreFastConv(gcn_outsize, gcn_outsize, K=K),
                            'x_t, adj_t -> x_t'),
                          (gnn.BatchNorm(gcn_outsize), 'x_t -> x_t'),
                          (nn.LeakyReLU(negative_slope=leaky_slope), 'x_t -> x_t'),
                          (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                          (HodgeLaguerreFastConv(gcn_outsize, gcn_outsize, K=K),
                           'x_s, adj_s -> x_s'),
                          (gnn.BatchNorm(gcn_outsize), 'x_s -> x_s'),
                          (nn.LeakyReLU(negative_slope=leaky_slope), 'x_s -> x_s'),
                          (Dropout(p=dropout_ratio), 'x_s -> x_s'),
                          (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
                fc = gnn.Sequential('x_t, adj_t, x_s, adj_s', layers)
                setattr(self, 'NEConv{}{}'.format(i,j), fc)
                gcn_insize = gcn_insize + gcn_outsize
                final_out = gcn_outsize
            
            if i in self.pool_loc:# < len(self.filters)-1:
                # ATT
                fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize, only_att=True, sigma=nn.Sigmoid())
                setattr(self, 'NEAtt{}'.format(i), fc)
        
        layers = [(HodgeLaguerreFastConv(final_out, 1, K=1),
                    'x_t, adj_t -> x_t'),
                  (HodgeLaguerreFastConv(final_out, 1, K=1),
                    'x_s, adj_s -> x_s'),
                  (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
        self.readout = gnn.Sequential('x_t, adj_t, x_s, adj_s', layers)  
        
        mlp_insize = self.num_nodepedge #self.filters[-1] * 2 
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, datas, device='cuda:0'):
        data = datas[0].to(device)
        # 1. node & edge postion
        pos_ts, pos_ss = [], []
        for p in range(len(self.pool_loc)):
            n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_node1)], dim=-1)
            n_batch = n_batch.to(device)
            s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_edge1)], dim=-1)
            s_batch = s_batch.to(device)
            n_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_node1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            s_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_edge1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            pos_ts.append((datas[p].pos_t.to(device).view(-1) + n_ahead[n_batch]).view(-1,1))
            pos_ss.append((datas[p].pos_s.to(device).view(-1) + s_ahead[s_batch]).view(-1,1))
            
        x_s, edge_index_s, edge_weight_s = data.x_s, data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
        # 2. Obtain node & edge embeddings
        x_t = self.node_embedding(x_t)
        adj_t = SparseTensor(row=edge_index_t[0], col=edge_index_t[1], value=edge_weight_t).t()
        adj_s = SparseTensor(row=edge_index_s[0], col=edge_index_s[1], value=edge_weight_s).t()
        x_t, x_s = self.HL_init_conv(x_t, adj_t, x_s, adj_s)
        x_s0, x_t0 = x_s, x_t
        k = 0
        par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
        D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
        for i, _ in enumerate(self.channels):
            for j in range(self.channels[i]):
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, adj_t, x_s, adj_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
            # structural pooling        
            if i in self.pool_loc:
                # compute attention
                fc = getattr(self, 'NEAtt%d' % i)
                att_t, att_s = fc(x_t0, x_s0, par_1, D)
                if k == 0:
                    node_att, edge_att = att_t.view(data.num_graphs,-1), att_s.view(data.num_graphs,-1)
                x_t0 = x_t0 * att_t
                x_s0 = x_s0 * att_s
                
                # signal pooling
                pos_t, pos_s = pos_ts[k], pos_ss[k]
                x_t0 = x_t0[~torch.isinf(pos_t).view(-1)]
                pos_t = pos_t[~torch.isinf(pos_t).view(-1)]
                x_t0 = scatter_mean(x_t0,pos_t.to(torch.long),dim=0)
                x_s0 = x_s0[~torch.isinf(pos_s).view(-1)]
                pos_s = pos_s[~torch.isinf(pos_s).view(-1)]
                x_s0 = scatter_mean(x_s0,pos_s.to(torch.long),dim=0)
                edge_index_s, edge_weight_s = datas[k+1].edge_index_s.to(device), datas[k+1].edge_weight_s.to(device)
                edge_index_t, edge_weight_t = datas[k+1].edge_index_t.to(device), datas[k+1].edge_weight_t.to(device)
                adj_t = SparseTensor(row=edge_index_t[0], col=edge_index_t[1], value=edge_weight_t).t()
                adj_s = SparseTensor(row=edge_index_s[0], col=edge_index_s[1], value=edge_weight_s).t()
                k += 1
                par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
                D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
             
        # 2. Readout layer
        x_t, x_s = self.readout(x_t, adj_t, x_s, adj_s)
        x = torch.cat([x_s.view(data.num_graphs,-1), x_t.view(data.num_graphs,-1)], dim=-1)
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        return self.out(x), x, node_att, edge_att
    
###############################################################################
def unbatch_edge_attr(edge_index: Tensor, edge_attr: Tensor, batch: Tensor):
    deg = ut.degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = ut.degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1), edge_attr.split(sizes, dim=0)


class NodeEdgeInt(nn.Module):
    # node edge interaction module
    def __init__(self, d=64, dk=32, dv=64, dl=64, only_att=False, sigma=nn.Sigmoid(), l=0.9):
        # d: input feature dim
        # dk: feature dim of key & query
        # dv: feature dim of value
        # dl: feature dim of latent
        # only_att: if true, only output the attention value
        # sigma: activation function
        super().__init__()
        dl = dv
        self.sigma = sigma
        self.dk = dk
        self.only_att = only_att
        if only_att:
            self.WQ_Node = nn.Linear(d, dk)
            self.WK_Node = nn.Linear(d, dk)
            
            self.WQ_Edge = nn.Linear(d, dk)
            self.WK_Edge = nn.Linear(d, dk)
        else:
            self.WV_Node = nn.Sequential(
                nn.Linear(d*2, dl),
                nn.BatchNorm1d(dl),
                nn.ReLU(),
                nn.Linear(dl, dv),
                nn.BatchNorm1d(dv),
                nn.ReLU())
            self.WV_Edge = nn.Sequential(
                nn.Linear(d*2, dl),
                nn.BatchNorm1d(dl),
                nn.ReLU(),
                nn.Linear(dl, dv),
                nn.BatchNorm1d(dv),
                nn.ReLU())
        self.lambda_Node = l#self.sigma(nn.Parameter(torch.zeros(1)).to(device))
        self.lambda_Edge = l#self.sigma(nn.Parameter(torch.zeros(1)).to(device))
        
    def forward(self, x_t, x_s, par, D):    
        x_s2t = (1/D).view(-1,1)*torch.sparse.mm(par.abs(), x_s)
        x_t2s = torch.sparse.mm(par.abs().transpose(0,1), x_t)/2
        # print(x_t.shape, x_s.shape, x_s2t.shape, x_t2s.shape)
        if self.only_att:
            # print(x_t.shape, x_s.shape, x_s2t.shape, x_t2s.shape,)
            a_t = self.sigma(( (1-self.lambda_Node)*(self.WQ_Edge(x_s2t)*(self.WK_Node(x_t))).sum(dim=1, keepdim=True)
                              + self.lambda_Node*(self.WQ_Node(x_t)*self.WK_Node(x_t)).sum(dim=1, keepdim=True)
                              )/np.sqrt(self.dk))
            a_s = self.sigma(( (1-self.lambda_Edge)*(self.WQ_Node(x_t2s)*(self.WK_Edge(x_s))).sum(dim=1, keepdim=True)
                              + self.lambda_Edge*(self.WQ_Edge(x_s)*self.WK_Edge(x_s)).sum(dim=1, keepdim=True)
                              )/np.sqrt(self.dk))    
            return a_t, a_s
        else:
            x_t1 = self.WV_Node(torch.cat([x_s2t, x_t], dim=-1))
            x_s1 = self.WV_Edge(torch.cat([x_t2s, x_s], dim=-1))
            return x_t1, x_s1
        

 
###############################################################################
############################# Convolution #####################################
###############################################################################

class Inception1D(nn.Module):
    def __init__(self, in_channels=64, num_channels=8, maxpool=3, if_dim_reduction=False, 
                 leaky_slope=0.1, if_readout=False):
        # inception module for fmri time course data
        super(Inception1D, self).__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.if_dim_reduction = if_dim_reduction
        self.if_readout = if_readout
        self.embedding = nn.Conv1d(1, in_channels, 5, padding=2)

        self.channel1_1 = nn.Conv1d(in_channels, int(in_channels/4), 1, padding=0)
        self.channel2_1 = nn.Conv1d(in_channels, int(in_channels/2), 3, padding=1)
        self.channel3_1 = nn.Conv1d(in_channels, int(in_channels/4), 5, padding=2)
        self.pool_1 = nn.MaxPool1d(maxpool, stride=int(maxpool-1), padding=int((maxpool-1)/2))
        
        self.channel1_2 = nn.Conv1d(in_channels, num_channels, 1, padding=0)
        self.channel2_2 = nn.Conv1d(in_channels, num_channels*2, 3, padding=1)
        self.channel3_2 = nn.Conv1d(in_channels, num_channels, 5, padding=2)
        self.leakyReLU = nn.LeakyReLU(leaky_slope)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.bn2 = nn.BatchNorm1d(self.num_channels*4)
    
    def forward(self, x):
        # Temporal Feature x: N*T
        # print(x.shape)
        x = torch.unsqueeze(x,1) # change dim to N*C*T
        x = self.embedding(x)
        x1 = self.channel1_1(x)
        x2 = self.channel2_1(x)
        x3 = self.channel3_1(x)
        x = self.pool_1(self.leakyReLU(self.bn1(torch.cat((x1,x2,x3), dim=1))))
        x1 = self.channel1_2(x)
        x2 = self.channel2_2(x)
        x3 = self.channel3_2(x)
        x = self.leakyReLU(self.bn2(torch.cat((x1,x2,x3), dim=1)))
        
        if self.if_readout:
            temp = x.max(dim=-1)
            return torch.cat([temp[0], x.mean(dim=-1)],dim=-1)
        else:
            return torch.transpose(x,1,2)
    
    
###############################################################################
class HodgeLaguerreFastConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                  bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)])
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x, adj_t):
        """"""
        # x: N*T*C
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        xshape = x.shape
        k = 1

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            x = x.view(xshape[0],-1)
            Tx_1 = x - self.message_and_aggregate(adj_t=adj_t, x=x)
            if len(xshape)>=3:
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            out = out + self.lins[1](Tx_1)
        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0],-1)
            Tx_2 = self.message_and_aggregate(adj_t=adj_t, x=x)
            if len(xshape)>=3:
                Tx_2 = Tx_2.view(inshape[0],inshape[1],-1)
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_0) / (k+1)
            k += 1
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')
    
###############################################################################
class HodgeChebConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None, batch: OptTensor = None):
        """"""
        # x: N*T*C
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        xshape = x.shape

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            # print(x.shape,xshape[0])
            # x = x.view(xshape[0],-1)
            
            if len(xshape)==3:
                x = torch.transpose(x,1,2) # change dim to [N,C,T]
                x = x.view(xshape[0],-1)
                Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
                Tx_1 = Tx_1.view(xshape[0],xshape[2],-1) #[N,C,T]
                Tx_1 = torch.transpose(Tx_1,1,2)
            else:
                Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            
            if len(xshape)>=3:
                Tx_1 = torch.transpose(Tx_1,1,2) # change dim to [N,C,T]
                Tx_1 = Tx_1.view(xshape[0],-1)
                Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
                Tx_2 = Tx_2.view(xshape[0],xshape[2],-1) #[N,C,T]
                Tx_2 = torch.transpose(Tx_2,1,2) #[N,T,C]
                Tx_1 = Tx_1.view(xshape[0],xshape[2],-1) #[N,C,T]
                Tx_1 = torch.transpose(Tx_1,1,2)
            else:
                Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')
    
###############################################################################

class HodgeLaguerreConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                  bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                    weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None, batch: OptTensor = None):
        """"""
        # x: N*T*C
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        xshape = x.shape
        k = 1

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            x = x.view(xshape[0],-1)
            Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0],-1)
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_2 = Tx_2.view(inshape[0],inshape[1],-1)
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_0) / (k+1)
            k += 1
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')



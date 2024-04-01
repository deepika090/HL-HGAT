import argparse
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
# import torchvision as tv
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import unbatch
from torchmetrics import F1Score
from torchmetrics.classification import BinaryF1Score
from torch_scatter import scatter_max, scatter_mean, scatter_add


###############################################################################
#################################### ABCD #####################################
###############################################################################
class HL_HGCNN_ABCD_dense_int3_attpool(torch.nn.Module):
    def __init__(self, channels=[2,2,2], filters=[64,128,256], mlp_channels=[], K=2, node_dim=64, 
                  edge_dim=1, num_classes=1, dropout_ratio=0.0, dropout_ratio_mlp=0.0, pool_loc=[0],
                  keig=10,num_nodepedge=2089):
        super(HL_HGCNN_ABCD_dense_int3_attpool, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
        self.initial_channel = self.filters[0]
        self.pool_loc = pool_loc
        self.node_embedding = Inception1D(if_readout=True)
        self.keig = keig
        self.num_nodepedge = num_nodepedge#global754860+137#70+2126
        leaky_slope = 0.1
        
        layers = [(HodgeLaguerreConv(self.node_dim, self.initial_channel, K=K),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
                  (nn.LeakyReLU(negative_slope=leaky_slope), 'x_t -> x_t'),
                  (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                  (HodgeLaguerreConv(self.edge_dim, self.initial_channel, K=K),
                   'x_s, edge_index_s, edge_weight_s -> x_s'),
                  (gnn.BatchNorm(self.initial_channel), 'x_s -> x_s'),
                  (nn.LeakyReLU(negative_slope=leaky_slope), 'x_s -> x_s'),
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
                          (nn.LeakyReLU(negative_slope=leaky_slope), 'x_t -> x_t'),
                          (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                          (HodgeLaguerreConv(gcn_outsize, gcn_outsize, K=K),
                           'x_s, edge_index_s, edge_weight_s -> x_s'),
                          (gnn.BatchNorm(gcn_outsize), 'x_s -> x_s'),
                          (nn.LeakyReLU(negative_slope=leaky_slope), 'x_s -> x_s'),
                          (Dropout(p=dropout_ratio), 'x_s -> x_s'),
                          (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
                fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)
                setattr(self, 'NEConv{}{}'.format(i,j), fc)
                gcn_insize = gcn_insize + gcn_outsize
                final_out = gcn_outsize
            
            if i in self.pool_loc:# < len(self.filters)-1:
                # ATT
                fc = NodeEdgeInt(d=gcn_outsize, dv = gcn_outsize, only_att=True, sigma=nn.Sigmoid())
                setattr(self, 'NEAtt{}'.format(i), fc)
        
        layers = [(HodgeLaguerreConv(final_out, 1, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (HodgeLaguerreConv(final_out, 1, K=1),
                    'x_s, edge_index_s, edge_weight_s -> x_s'),
                  (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
        self.readout = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)  
        
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


    def forward(self, datas, device='cuda:0', if_final_layer=False):
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
            pos_ts.append((datas[p].x_t[:,0].to(device) + n_ahead[n_batch]).view(-1,1))
            pos_ss.append((datas[p].x_s[:,0].to(device) + s_ahead[s_batch]).view(-1,1))
            
        x_s, edge_index_s, edge_weight_s = data.x_s[:,1:], data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t[:,1:], data.edge_index_t, data.edge_weight_t
        # 2. Obtain node & edge embeddings
        x_t = self.node_embedding(x_t)
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t
        k = 0
        par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
        D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
        for i, _ in enumerate(self.channels):
            for j in range(self.channels[i]):
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
            # structural pooling        
            if i in self.pool_loc:
                fc = getattr(self, 'NEAtt%d' % i)
                att_t, att_s = fc(x_t, x_s, par_1, D)
                x_t0 = x_t0 * att_t
                x_s0 = x_s0 * att_s
                pos_t, pos_s = pos_ts[k], pos_ss[k]
                x_t0 = scatter_mean(x_t0,pos_t.to(torch.long),dim=0)
                x_s0 = x_s0[~torch.isinf(pos_s).view(-1)]
                pos_s = pos_s[~torch.isinf(pos_s).view(-1)]
                x_s0 = scatter_mean(x_s0,pos_s.to(torch.long),dim=0)
                edge_index_s, edge_weight_s = datas[k+1].edge_index_s.to(device), datas[k+1].edge_weight_s.to(device)
                edge_index_t, edge_weight_t = datas[k+1].edge_index_t.to(device), datas[k+1].edge_weight_t.to(device)
                k += 1
                par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
                D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
             
        # 2. Readout layer
        x_t, x_s = self.readout(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x = torch.cat([x_s.view(data.num_graphs,-1), x_t.view(data.num_graphs,-1)], dim=-1)
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        if if_final_layer:
            return x, self.out(x)
        else:
            return self.out(x)
    
###############################################################################
##################### pepfunc #########################
###############################################################################
class HL_HGCNN_pepfunc_dense_int3_attpool(torch.nn.Module):
    def __init__(self, channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, node_dim=9, 
                  edge_dim=3, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.0, pool_loc=0,
                  keig=20):
        super(HL_HGCNN_pepfunc_dense_int3_attpool, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
        self.initial_channel = self.filters[0]
        self.pool_loc = pool_loc
        
        layers = [(HodgeLaguerreConv(self.node_dim, self.initial_channel, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
                  (nn.ReLU(), 'x_t -> x_t'),
                  (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                  (HodgeLaguerreConv(self.edge_dim, self.initial_channel, K=1),
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
                gcn_insize = gcn_insize + gcn_outsize
            
            if i == self.pool_loc:
                # ATT
                fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize, only_att=True)
                setattr(self, 'NEAtt{}'.format(i), fc)
        
        mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, datas, device='cuda:0'):
        data = datas[0].to(device)
        # 1. Obtain node embeddings
        pos_ts, pos_ss = [], []
        for p in range(0,1):
            n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_node1)], dim=-1)
            n_batch = n_batch.to(device)
            s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_edge1)], dim=-1)
            s_batch = s_batch.to(device)
            n_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_node1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            s_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_edge1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            pos_ts.append((datas[p].x_t[:,0].to(device) + n_ahead[n_batch]).view(-1,1))
            pos_ss.append((datas[p].x_s[:,0].to(device) + s_ahead[s_batch]).view(-1,1))
            
        x_s, edge_index_s, edge_weight_s = data.x_s[:,1:], data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t[:,1:], data.edge_index_t, data.edge_weight_t
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t
        k = 0
        par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
        D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6

        for i, _ in enumerate(self.channels):
            for j in range(self.channels[i]):
                # print(x_t.shape, x_s.shape, x_t0.shape, x_s0.shape)
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
            # structural pooling        
            if i == self.pool_loc:
                fc = getattr(self, 'NEAtt%d' % i)
                att_t, att_s = fc(x_t0, x_s0, par_1, D)
                x_t0 = x_t0 * att_t
                x_s0 = x_s0 * att_s
                pos_t, pos_s = pos_ts[k], pos_ss[k]
                x_t0 = scatter_mean(x_t0,pos_t.to(torch.long),dim=0)
                x_s0 = x_s0[~torch.isinf(pos_s).view(-1)]
                pos_s = pos_s[~torch.isinf(pos_s).view(-1)]
                x_s0 = scatter_mean(x_s0,pos_s.to(torch.long),dim=0)
                edge_index_s, edge_weight_s = datas[k+1].edge_index_s.to(device), datas[k+1].edge_weight_s.to(device)
                edge_index_t, edge_weight_t = datas[k+1].edge_index_t.to(device), datas[k+1].edge_weight_t.to(device)
                k=1
                par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
                D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
             
        # 2. Readout layer
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[min(i,1)].num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[min(i,1)].num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        return self.out(x) 

###############################################################################
class HL_HGCNN_pepfunc_dense_int3_pyr(torch.nn.Module):
    def __init__(self, channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, node_dim=9, 
                  edge_dim=3, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.0, 
                  keig=20):
        super(HL_HGCNN_pepfunc_dense_int3_pyr, self).__init__()
        self.channels = channels
        self.filters = filters
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
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
            
        
        mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, data, device='cuda:0', if_final_layer=False):

        # 1. Obtain node embeddings
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        
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
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
        # 2. Readout layer
        x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        if if_final_layer:
            return x, self.out(x)
        else:
            return self.out(x)
    
###############################################################################
##################### zinc #########################
##############################################################################
class HL_HGCNN_zinc_dense_int3_attpool(torch.nn.Module):
    def __init__(self, channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, node_dim=21, 
                  edge_dim=3, num_classes=1, dropout_ratio=0.0, dropout_ratio_mlp=0.0, pool_loc=0,
                  keig=7):
        super(HL_HGCNN_zinc_dense_int3_attpool, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
        self.initial_channel = self.filters[0]
        self.pool_loc = pool_loc
        
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
                gcn_insize = gcn_insize + gcn_outsize
            
            if i == self.pool_loc:
                # ATT
                fc = NodeEdgeInt(d=gcn_outsize, dv = gcn_outsize, only_att=True, sigma=nn.ReLU())
                setattr(self, 'NEAtt{}'.format(i), fc)
        
        mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, datas, device='cuda:0'):
        data = datas[0].to(device)
        # 1. Obtain node embeddings
        pos_ts, pos_ss = [], []
        for p in range(0,1):#len(self.channels)-1):
            n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_node1)], dim=-1)
            n_batch = n_batch.to(device)
            s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_edge1)], dim=-1)
            s_batch = s_batch.to(device)
            n_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_node1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            s_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_edge1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            pos_ts.append((datas[p].x_t[:,0].to(device) + n_ahead[n_batch].to(device)).view(-1,1))
            pos_ss.append((datas[p].x_s[:,0].to(device) + s_ahead[s_batch].to(device)).view(-1,1))
            
        x_s, edge_index_s, edge_weight_s = data.x_s[:,1:], data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t[:,1:], data.edge_index_t, data.edge_weight_t
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t
        k = 0
        
        for i, _ in enumerate(self.channels):
            par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
            D = degree(datas[k].edge_index.view(-1).to(device))
            
            for j in range(self.channels[i]):
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
            # structural pooling        
            if i == self.pool_loc:
                fc = getattr(self, 'NEAtt%d' % i)
                att_t, att_s = fc(x_t, x_s, par_1, D)
                x_t = x_t * att_t
                x_s = x_s * att_s
                pos_t, pos_s = pos_ts[k], pos_ss[k]
                x_t0 = scatter_mean(x_t0,pos_t.to(torch.long),dim=0)
                x_s0 = x_s0[~torch.isinf(pos_s).view(-1)]
                pos_s = pos_s[~torch.isinf(pos_s).view(-1)]
                x_s0 = scatter_mean(x_s0,pos_s.to(torch.long),dim=0)
                edge_index_s, edge_weight_s = datas[i+1].edge_index_s.to(device), datas[i+1].edge_weight_s.to(device)
                edge_index_t, edge_weight_t = datas[i+1].edge_index_t.to(device), datas[i+1].edge_weight_t.to(device)
                k=1
                
        # 2. Readout layer
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[min(i,1)].num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[min(i,1)].num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        return self.out(x) 

###############################################################################
class HL_HGCNN_zinc_dense_int3_pyr(torch.nn.Module):
    def __init__(self, channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, node_dim=21, 
                  edge_dim=3, num_classes=1, dropout_ratio=0.0, dropout_ratio_mlp=0.0, 
                  keig=7):
        super(HL_HGCNN_zinc_dense_int3_pyr, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
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
            
        
        mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, data, device='cuda:0', if_final_layer=False):

        # 1. Obtain node embeddings
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        x_s, edge_index_s, edge_weight_s = data.x_s, data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t

        for i, _ in enumerate(self.channels):
            par_1 = adj2par1(data.edge_index, x_t.shape[0], x_s.shape[0])
            D = degree(data.edge_index.view(-1))
            
            for j in range(self.channels[i]):
                # print(x_t.shape, x_s.shape, x_t0.shape, x_s0.shape)
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
        # 2. Readout layer
        x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)
            
        if if_final_layer:
            return x, self.out(x)
        else:
            return self.out(x)
    
###############################################################################
class HL_HGCNN_zinc_dense_poolint3_pyr(torch.nn.Module):
    def __init__(self, channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, node_dim=21, 
                  edge_dim=3, num_classes=1, dropout_ratio=0.0, dropout_ratio_mlp=0.0, 
                  keig=7):
        super(HL_HGCNN_zinc_dense_poolint3_pyr, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
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
                # HL node edge filtering
                layers = [(HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                            'x_t, edge_index_t, edge_weight_t -> x_t'),
                          (gnn.BatchNorm(gcn_outsize), 'x_t -> x_t'),
                          (nn.ReLU(), 'x_t -> x_t'),
                          (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                          (HodgeLaguerreConv(gcn_insize, gcn_outsize, K=K),
                            'x_s, edge_index_s, edge_weight_s -> x_s'),
                          (gnn.BatchNorm(gcn_outsize), 'x_s -> x_s'),
                          (nn.ReLU(), 'x_s -> x_s'),
                          (Dropout(p=dropout_ratio), 'x_s -> x_s'),
                          (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
                fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)
                setattr(self, 'NEConv{}{}'.format(i,j), fc)
                gcn_insize = gcn_outsize + gcn_insize
            # int term
            fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize)
            setattr(self, 'NEInt{}'.format(i), fc)
            gcn_insize = gcn_outsize + gcn_insize
        
        mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, data, device='cuda:0'):

        # 1. Obtain node embeddings
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        x_s, edge_index_s, edge_weight_s = data.x_s, data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t

        for i, _ in enumerate(self.channels):
            par_1 = adj2par1(data.edge_index, x_t.shape[0], x_s.shape[0])
            D = degree(data.edge_index.view(-1))
            
            for j in range(self.channels[i]):
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, edge_index_t, edge_weight_t, x_s0, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
            fc = getattr(self, 'NEInt{}'.format(i))
            x_t, x_s = fc(x_t0, x_s0, par_1, D)
            x_t0 = torch.cat([x_t0, x_t], dim=-1)
            x_s0 = torch.cat([x_s0, x_s], dim=-1)
        # 2. Readout layer
        x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        return self.out(x)



###############################################################################
##################### TSP #########################
##############################################################################
class HL_HGCNN_TSP_dense_int3_pyr(torch.nn.Module):
    def __init__(self, channels=[2,2,2], filters=[64,128,256], mlp_channels=[], K=2, node_dim=2, 
                  edge_dim=1, num_classes=1, dropout_ratio=0.0, dropout_ratio_mlp=0.0, 
                  keig=20):
        super(HL_HGCNN_TSP_dense_int3_pyr, self).__init__()
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
        if len(self.mlp_channels)==1:
            layers = [(HodgeLaguerreConv(mlp_insize, self.mlp_channels[0], K=1),
                        'x_t, edge_index_t, edge_weight_t -> x_t'),
                      (gnn.BatchNorm(self.mlp_channels[0]), 'x_t -> x_t'),
                      (nn.ReLU(), 'x_t -> x_t'),
                      (Dropout(p=dropout_ratio), 'x_t -> x_t'),]
            fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t', layers)
            setattr(self, 'mlp', fc)
            mlp_insize = self.mlp_channels[0]
        layers = [(HodgeLaguerreConv(mlp_insize, num_classes, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),]
        fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t', layers)
        setattr(self, 'out', fc)

    def forward(self, data, device='cuda:0'):

        # 1. Obtain node embeddings
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        x_s, edge_index_s, edge_weight_s = data.x_s[:,:1], data.edge_index_s, data.edge_weight_s
        edge_mask = data.x_s[:,1:]
        x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t
        par_1 = adj2par1(data.edge_index, x_t.shape[0], x_s.shape[0])
        D = degree(data.edge_index.view(-1),num_nodes=x_t.shape[0]) + 1e-6
            
        for i, _ in enumerate(self.channels):
            for j in range(self.channels[i]):
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
        # 2. Readout layer
        x_t2s = torch.sparse.mm(par_1.transpose(0,1), x_t).abs()/2
        x_s = torch.cat([x_s,x_t2s], dim=-1)
        if len(self.mlp_channels)==1:
            x_s = self.mlp(x_s, edge_index_s, edge_weight_s)
        return self.out(x_s, edge_index_s, edge_weight_s)*edge_mask, s_batch
    
    
###############################################################################
#######################  Superpixel dataset  #########################
###############################################################################
class HL_HGCNN_CIFAR10SP_dense_int3_pyr(torch.nn.Module):
    def __init__(self, channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, node_dim=5, 
                  edge_dim=4, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.0, l=0.9,
                  keig=10):
        super(HL_HGCNN_CIFAR10SP_dense_int3_pyr, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
        self.initial_channel = self.filters[0]
        
        layers = [(HodgeLaguerreConv(self.node_dim, self.initial_channel, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
                  (nn.ReLU(), 'x_t -> x_t'),
                  (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                  (HodgeLaguerreConv(self.edge_dim, self.initial_channel, K=1),
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
                fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize, l=l)
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
            
        
        mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, data, device='cuda:0', if_final_layer=False):

        # 1. Obtain node embeddings
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        
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
                # print(x_t.shape, x_s.shape, x_t0.shape, x_s0.shape)
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
        # 2. Readout layer
        x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        return self.out(x)
    
###############################################################################
class HL_HGAT_CIFAR10SP(torch.nn.Module):
    def __init__(self, channels=[2,2,2], filters=[64,128,256], mlp_channels=[], K=2, node_dim=5, 
                  edge_dim=4, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.0, pool_loc=0,
                  keig=10):
        """
        For visualization
        """
        super(HL_HGAT_CIFAR10SP, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
        self.initial_channel = self.filters[0]
        self.pool_loc = pool_loc
        # self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        self.relu = nn.ReLU()
        
        layers = [(HodgeLaguerreConv(self.node_dim, self.initial_channel, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
                  (nn.ReLU(), 'x_t -> x_t'),
                  (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                  (HodgeLaguerreConv(self.edge_dim, self.initial_channel, K=1),
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
                gcn_insize = gcn_insize + gcn_outsize
            
            if i == self.pool_loc:# < len(self.filters)-1:
                # ATT
                fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize, only_att=True)
                setattr(self, 'NEAtt{}'.format(i), fc)
        
        mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, datas, device='cuda:0'):
        # time3 = time.time() 
        data = datas[0].to(device)
        # 1. Obtain node embeddings
        pos_ts, pos_ss = [], []
        for p in range(0,1):#len(self.channels)-1):
            n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_node1)], dim=-1)
            n_batch = n_batch.to(device)
            s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_edge1)], dim=-1)
            s_batch = s_batch.to(device)
            n_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_node1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            s_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_edge1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            pos_ts.append((datas[p].x_t[:,0].to(device) + n_ahead[n_batch]).view(-1,1))
            pos_ss.append((datas[p].x_s[:,0].to(device) + s_ahead[s_batch]).view(-1,1))
            
        x_s, edge_index_s, edge_weight_s = data.x_s[:,1:], data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t[:,1:], data.edge_index_t, data.edge_weight_t
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t
        k = 0
        par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
        D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
        # time2 = time.time() 
        for i, _ in enumerate(self.channels):
            par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
            D = degree(datas[k].edge_index.view(-1).to(device))
            
            if i > self.pool_loc:
                for j in range(self.channels[i]):
                    fc = getattr(self, 'NEInt{}{}'.format(i,j))
                    x_t, x_s = fc(x_t0, x_s0, par_1, D)
                    fc = getattr(self, 'NEConv{}{}'.format(i,j))
                    x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                    x_t0 = torch.cat([x_t0, x_t], dim=-1)
                    x_s0 = torch.cat([x_s0, x_s], dim=-1)
                    fc = getattr(self, 'NEInt{}{}'.format(i,j))
                    x_t1, x_s1 = fc(x_t01, x_s01, par_1, D)
                    fc = getattr(self, 'NEConv{}{}'.format(i,j))
                    x_t1, x_s1 = fc(x_t1, edge_index_t, edge_weight_t, x_s1, edge_index_s, edge_weight_s)
                    x_t01 = torch.cat([x_t01, x_t1], dim=-1)
                    x_s01 = torch.cat([x_s01, x_s1], dim=-1)
            else:
                for j in range(self.channels[i]):
                    fc = getattr(self, 'NEInt{}{}'.format(i,j))
                    x_t, x_s = fc(x_t0, x_s0, par_1, D)
                    fc = getattr(self, 'NEConv{}{}'.format(i,j))
                    x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                    x_t0 = torch.cat([x_t0, x_t], dim=-1)
                    x_s0 = torch.cat([x_s0, x_s], dim=-1)
                    
            # structural pooling        
            if i == self.pool_loc:
                fc = getattr(self, 'NEAtt%d' % i)
                att_t, att_s = fc(x_t0, x_s0, par_1, D)
                return att_t, att_s
            
            
class HL_HGCNN_CIFAR10SP_dense_int3_attpool(torch.nn.Module):
    def __init__(self, channels=[2,2,2], filters=[64,128,256], mlp_channels=[], K=2, node_dim=5, l=0.5, 
                  edge_dim=4, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.0, pool_loc=0,
                  keig=10):
        super(HL_HGCNN_CIFAR10SP_dense_int3_attpool, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
        self.initial_channel = self.filters[0]
        self.pool_loc = pool_loc
        
        layers = [(HodgeLaguerreConv(self.node_dim, self.initial_channel, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
                  (nn.ReLU(), 'x_t -> x_t'),
                  (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                  (HodgeLaguerreConv(self.edge_dim, self.initial_channel, K=1),
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
                gcn_insize = gcn_insize + gcn_outsize
            
            if i == self.pool_loc:
                # ATT
                fc = NodeEdgeInt(d=gcn_outsize, dv = gcn_outsize, only_att=True, sigma=nn.ReLU(), l=l)
                setattr(self, 'NEAtt{}'.format(i), fc)
        
        mlp_insize = self.filters[-1] * 2 
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.ReLU(),
                nn.Dropout(dropout_ratio_mlp),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = Linear(mlp_insize, num_classes)


    def forward(self, datas, device='cuda:0', if_final_layer=False, if_att=False):
        data = datas[0].to(device)
        # 1. Obtain node & edge embeddings
        pos_ts, pos_ss = [], []
        for p in range(0,1):
            n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_node1)], dim=-1)
            n_batch = n_batch.to(device)
            s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_edge1)], dim=-1)
            s_batch = s_batch.to(device)
            n_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_node1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            s_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_edge1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
            pos_ts.append((datas[p].x_t[:,0].to(device) + n_ahead[n_batch]).view(-1,1))
            pos_ss.append((datas[p].x_s[:,0].to(device) + s_ahead[s_batch]).view(-1,1))
            
        x_s, edge_index_s, edge_weight_s = data.x_s[:,1:], data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t[:,1:], data.edge_index_t, data.edge_weight_t
        
        x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        x_s0, x_t0 = x_s, x_t
        k = 0
        par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
        D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
        for i, _ in enumerate(self.channels):
            for j in range(self.channels[i]):
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
            # structural pooling        
            if i == self.pool_loc:
                fc = getattr(self, 'NEAtt%d' % i)
                att_t, att_s = fc(x_t, x_s, par_1, D)
                att_t = att_t / att_t.max()
                att_s = att_s / att_s.max()
                x_t = x_t * att_t
                x_s = x_s * att_s
                pos_t, pos_s = pos_ts[k], pos_ss[k]
                x_t0 = scatter_mean(x_t0,pos_t.to(torch.long),dim=0)
                x_s0 = x_s0[~torch.isinf(pos_s).view(-1)]
                pos_s = pos_s[~torch.isinf(pos_s).view(-1)]
                x_s0 = scatter_mean(x_s0,pos_s.to(torch.long),dim=0)
                edge_index_s, edge_weight_s = datas[k+1].edge_index_s.to(device), datas[k+1].edge_weight_s.to(device)
                edge_index_t, edge_weight_t = datas[k+1].edge_index_t.to(device), datas[k+1].edge_weight_t.to(device)
                k=1
                par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
                D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
             
        # 2. Readout layer
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[min(i,1)].num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[min(i,1)].num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)

        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)
        if if_final_layer:
            return x, self.out(x)
        if if_att:
            return self.out(x), att_t, att_s
        return self.out(x) 
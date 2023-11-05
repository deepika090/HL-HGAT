#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:04:33 2023

@author: jinghan
"""

#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch_geometric.utils as gutils
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_geometric.data import Data
from lib.Hodge_Cheb_Conv import *
from lib.Hodge_ST_Model import *
from lib.Hodge_Dataset import *
from scipy.io import savemat
import scipy.io as io
import numpy as np

import mat73
import os

parser = argparse.ArgumentParser()
parser.add_argument('--c1', type=int, default=2, help='layer num in block1')
parser.add_argument('--c2', type=int, default=2, help='layer num in block2')
parser.add_argument('--c3', type=int, default=0, help='layer num in block3')
parser.add_argument('--filters', type=int, default=32, help='filter num in each channel')
parser.add_argument('--mlp_channels', type=int, default=2, help='num of fully connected layers')
parser.add_argument('--dropout_ratio', type=float, default=0.25, help='dropout_ratio')
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--l2', type=float, default=1e-3, help='weight decay ratio')
parser.add_argument('--K', type=int, default=4, help='polynomial order')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# parser.add_argument('--gpu', type=int, default=0, help='gpu index')
# parser.add_argument('--root', type=str, default='/home/jinghan/Documents/MATLAB/Hodge_Laplacian/data/ABCD'
                    # , help='data root')
parser.add_argument('--fold', type=int, default=-1, help='which fold')
parser.add_argument('--run', type=int, default=1, help='which run (split)')
parser.add_argument('--finetune', type=int, default=0, help='if finetune')
parser.add_argument('--test', type=int, default=0, help='if test')
parser.add_argument('--pool_loc', type=int, default=0, help='Pooling loc')
parser.add_argument('--normmode', type=int, default=0, help='normalization mode')
parser.add_argument('--threshmode', type=int, default=0, help='thresholding mode')
parser.add_argument('--k_ratio', type=float, default=0.25, help='top k% values when thresholding')
parser.add_argument('--seed', type=int, default=10086, help='random seed')
args = parser.parse_args()

def loadmat(path):
    try:
        return io.loadmat(path)
    except:
        return mat73.loadmat(path)


def time_norm(Time_series, trainidx=None, mode='group_roi'):
    # Mode:
    #   group_roi: compute mean and std over all subjects and timepoints per roi
    #   subject_roi: compute mean and std over all timepoints per roi per subject
    #   subject_all: compute mean and std over all timepoints and rois per subject
    
    if mode == 'group_roi':
        if trainidx is None:
            x_t = torch.cat([x for x in Time_series],dim=-1)
        else:
            x_t = torch.cat([x for x in Time_series[trainidx]],dim=-1)
        x_t_mean = x_t.mean(dim=-1)
        x_t_std = x_t.std(dim=-1)
        X_T = (Time_series.transpose(dim0=1, dim1=0)-x_t_mean.view(-1,1,1)) / x_t_std.view(-1,1,1)
        X_T = X_T.transpose(dim0=1, dim1=0)
    elif mode == 'subject_all':
        X_T = Time_series.clone()
        for i,x_t in enumerate(Time_series):
            X_T[i] = (x_t - x_t.mean()) / x_t.std() 
    elif mode == 'subject_roi':
        X_T = Time_series.clone()
        for i,x_t in enumerate(Time_series):
            X_T[i] = (x_t-x_t.mean(dim=-1).view(-1,1))/ x_t.std(dim=-1).view(-1,1)
    else:
        raise('wrong mode')
    return X_T

class ABCD_MLGC(Dataset):
    def __init__(self, root, X_t, X_s, Y, datas, pool_num=1):
        # data aug
        self.root = root
        self.X_t = X_t
        self.X_s = X_s
        self.Y = Y
        self.pool_num = pool_num
        self.node_dim = X_t.shape[-1]
        self.edge_dim = 1
        self.size = X_t.shape[0]
        self.datas = datas
        # data_zip = torch.load(osp.join(name, 'ABCD_GRAPH_STRUCTURE.pt'))
        # self.data = data_zip['graph']
        # data_zip = torch.load(osp.join(name, 'ABCD_SUBGRAPH_STRUCTURE0.pt'))
        # self.datas = data_zip['graph']
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['ABCD_MLGC_'+str(fileidx)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return self.size

    def get(self,idx):
        x_s = self.X_s[idx]
        x_s = x_s[self.datas[0].edge_index[0],self.datas[0].edge_index[1]].view(-1,1)
        x_t = self.X_t[idx]
        ei1, ew1 = self.datas[0].edge_index_s, self.datas[0].edge_weight_s
        ei0, ew0 = self.datas[0].edge_index_t, self.datas[0].edge_weight_t
        ei = self.datas[0].edge_index
        data = PairData(x_s=x_s,edge_index_s=ei1,edge_weight_s=ew1,x_t=x_t,
                        edge_index_t=ei0,edge_weight_t=ew0,edge_index=ei,y=self.Y[idx])
        data.num_node1=self.datas[0].num_node1
        data.num_nodes=self.datas[0].num_nodes
        data.num_edge1=self.datas[0].num_edge1
        data.pos_s = self.datas[0].pos_s.view(-1,1)
        data.pos_t = self.datas[0].pos_t.view(-1,1)
        datas = [data]
        for i in range(self.pool_num):
            datas.append(self.datas[i+1])
        return datas
        
    def process(self):        
        return None



###############################################################################
##################### MODEL #########################
###############################################################################
class HL_HGCNN_ABCD_dense_int3_attpool(torch.nn.Module):
    def __init__(self, channels=[2,2,2], filters=[32,64,128], mlp_channels=[], K=4, node_dim=64, init_time_conv=64,
                  time_pool_step=5, edge_dim=1, num_classes=1, dropout_ratio=0.0, pool_loc=[0], leaky_slope = 0.1,
                  keig=0, dk=64, num_nodepedge=None):
        self.channels = channels
        self.filters = filters#[]
        self.mlp_channels = mlp_channels
        self.node_dim = node_dim + keig
        self.edge_dim = edge_dim + keig
        self.initial_channel = self.filters[0]
        self.pool_loc = pool_loc
        self.keig = keig # number of eigenvalue
        self.num_nodepedge = num_nodepedge
        super().__init__()
        
        ## Temporal convolution of fMRI time-series
        self.node_embedding = Inception1D(in_channels=init_time_conv, num_channels=int(node_dim/4), 
                                          maxpool=time_pool_step, if_readout=True)
        ## Initial HL-filter
        self.HL_init_conv = HL_filter(channels=1, filters=self.initial_channel, K=K, node_dim=self.node_dim,
                                      edge_dim=self.edge_dim, dropout_ratio=dropout_ratio, 
                                      leaky_slope=leaky_slope,if_dense=False)
        gcn_insize = self.initial_channel
        
        ## multiple blocks
        for i, gcn_outsize in enumerate(self.filters):
            if self.channels[i] == 0:
                continue
            fc = HL_filter(channels=self.channels[i], filters=gcn_outsize, K=K, node_dim=gcn_insize, 
                           edge_dim=gcn_insize, dropout_ratio=dropout_ratio, leaky_slope=leaky_slope)
            setattr(self, 'HLconv{}'.format(i), fc)
            gcn_insize = gcn_insize + self.channels[i]*gcn_outsize
            
            fc = MSI(d=gcn_insize, dv = gcn_outsize)
            setattr(self, 'MSI{}'.format(i), fc)
            gcn_insize = gcn_insize + gcn_outsize
            
            if i in self.pool_loc:
                fc = SAPool(d=gcn_insize, dk=dk)
                setattr(self, 'SAP{}'.format(i), fc)
        
        ## Readout
        layers = [(HodgeLaguerreConv(gcn_insize, 1, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (HodgeLaguerreConv(gcn_insize, 1, K=1),
                    'x_s, edge_index_s, edge_weight_s -> x_s'),
                  (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
        self.readout = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)  
        
        ## output layer
        mlp_insize = self.num_nodepedge 
        for i, mlp_outsize in enumerate(mlp_channels):
            fc = nn.Sequential(
                Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.LeakyReLU(negative_slope=leaky_slope),
                nn.Dropout(dropout_ratio),
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
            pos_ts.append((datas[p].pos_t.to(device).view(-1) + n_ahead[n_batch]).view(-1,1))
            pos_ss.append((datas[p].pos_s.to(device).view(-1) + s_ahead[s_batch]).view(-1,1))
            
        x_s, edge_index_s, edge_weight_s = data.x_s, data.edge_index_s, data.edge_weight_s
        x_t, edge_index_t, edge_weight_t = data.x_t, data.edge_index_t, data.edge_weight_t
        # 2. Obtain node & edge embeddings
        x_t = self.node_embedding(x_t)
        
        x_t0, x_s0 = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
        k = 0
        par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
        D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
        for i, channel in enumerate(self.channels):
            if channel == 0:
                continue
            fc = getattr(self, 'HLconv{}'.format(i))
            # print(x_t0.shape,x_s0.shape, edge_index_t.shape)
            x_t0, x_s0 = fc(x_t0, edge_index_t, edge_weight_t, x_s0, edge_index_s, edge_weight_s,par_1,D)
            fc = getattr(self, 'MSI{}'.format(i))
            x_t, x_s = fc(x_t0, x_s0, par_1, D)
            x_t0 = torch.cat([x_t0, x_t], dim=-1)
            x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
            # structural pooling        
            if i in self.pool_loc:
                fc = getattr(self, 'SAP{}'.format(i))
                x_t0, x_s0, par_1, D, k, edge_index_t, edge_weight_t, edge_index_s, edge_weight_s, att_t, att_s = fc(x_t0, x_s0, par_1, D, datas, pos_ts, pos_ss, k)
             
        # 2. Readout layer
        x_t, x_s = self.readout(x_t0, edge_index_t, edge_weight_t, x_s0, edge_index_s, edge_weight_s)
        x = torch.cat([x_s.view(data.num_graphs,-1), x_t.view(data.num_graphs,-1)], dim=-1)
        # 3. Apply a final classifier
        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x = fc(x)

        if if_final_layer:
            return x, self.out(x)
        else:
            return self.out(x)
        
        
def train(loader):
    model.train()
    total_loss = 0
    y_pred, y = [], []
    for data in loader: 
        out = model(data)  
        loss = criterion1( out, data[0].y.view(-1,1))  # Compute the loss.
        total_loss += loss*data[0].num_graphs
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        y_pred.append(out)
        y.append(data[0].y.view(-1,1))
       
    y_pred, y = torch.cat(y_pred, dim=0), torch.cat(y, dim=0)
    y_pred = (y_pred - y_pred.mean()) / y_pred.std()
    y = (y - y.mean()) / y.std() 
    return torch.sqrt(total_loss/len(loader.dataset)) * 7.3, torch.mean(y_pred*y)

def test(loader):
     model.eval()
     y_pred, y = [], []
     total_loss, rmse = 0, 0
     
     for idx, data in enumerate(loader):
         with torch.no_grad():
             out = model(data)  
         loss = criterion1(out, data[0].y.view(-1,1))*7.3   # Compute the loss.
         rmse += criterion(out, data[0].y.view(-1,1))*data[0].num_graphs
         total_loss += loss * data[0].num_graphs
         y_pred.append(out)
         y.append(data[0].y.view(-1,1))
       
     y_pred, y = torch.cat(y_pred, dim=0), torch.cat(y, dim=0)
     y_pred = (y_pred - y_pred.mean()) / y_pred.std()
     y = (y - y.mean()) / y.std() 
     return torch.mean(y_pred*y), total_loss/len(loader.dataset), torch.sqrt(rmse/len(loader.dataset))*7.3

def visualize(loader, model, device='cuda:0'):
    model.eval()
    y_pred = []
    ys = []
    xs = []
    for idx, data in enumerate(loader):  # Iterate in batches over the training/test dataset. 
        ys.append(data.y.view(-1,1).cpu())
        with torch.no_grad():
            data = data.to(device)
            x, y = model(data, if_final=True) 
            xs.append(x.detach().cpu())
            y_pred.append(y.detach().cpu())
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    return xs, ys, y_pred

if __name__ == '__main__':
    # test degree based spatial pooling
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    start = time.time()
    keig = 0
    filename = './weights/ABCD/'
    if not osp.exists(filename):
        os.makedirs(filename)
    txtfile = './records/ABCD/'
    if not osp.exists(txtfile):
        os.makedirs(txtfile)
        
    if args.fold == -1:
        folds = [0,1,2,3,4]
    else:
        folds = [args.fold]
    for fold in folds:
        print('Fold {} begin'.format(fold))
        torch.manual_seed(args.seed)
        
        #### load data
        batch_size = args.batch_size
        num_workers = 4
        raw_path = '/home/jinghan/Documents/MATLAB/Hodge_Laplacian/data/ABCD/' 
        data_file = 'FC_Shen'
        info_file_name = 'ABCD7693_info.mat' 
        split_file_name = 'TrainValidTest_Idx_r1.mat'
        
        temp = loadmat(os.path.join(raw_path, info_file_name))
        IQ = torch.tensor(temp['IQ'])
        ID = temp['SubjectID']
        
        temp = loadmat(os.path.join(raw_path, split_file_name))
        TrainIdx = temp['TrainSet']
        ValidIdx = temp['ValidSet']
        TestIdx = temp['TestSet']
        
        usable_idx = []
        FC = torch.zeros(len(ID),268,268)
        Time_series = torch.zeros(len(ID),268,375)
        raw_path = '/home/jinghan/Documents/MATLAB/SF_Coupling/data/' 
        for id_idx, id in enumerate(ID):
            try:
                FC[id_idx] = torch.tensor(loadmat(os.path.join(raw_path,'FC_Shen',id[0]+'.mat'))['R_cat'])
                Time_series[id_idx] = torch.tensor(loadmat(os.path.join(raw_path,'FC_Shen',id[0]+'.mat'))['fmri'])[:,:375]
                if torch.count_nonzero(torch.isnan(torch.corrcoef(Time_series[id_idx])))>0:
                    continue
                usable_idx.append(id_idx)
            except:
                continue
        
        FC_rest = []
        TS_rest = []
        IQ_rest = []
        id_part = [id[0]+'.mat' for id in ID]
        id_all = os.listdir(os.path.join(raw_path,'FC_Shen'))
        
        for id in id_all:
            if id not in id_part:
                temp_ts = torch.tensor(loadmat(os.path.join(raw_path,'FC_Shen',id))['fmri'])[:,:375]
                temp_fc = torch.tensor(loadmat(os.path.join(raw_path,'FC_Shen',id))['R_cat'])
                if torch.count_nonzero(torch.isnan(torch.corrcoef(temp_ts)))>0:
                    continue
                try: 
                    if torch.isnan(temp_fc):
                        continue
                except:
                    x = None
                    
                IQ_rest.append(torch.tensor(loadmat(os.path.join(raw_path,'FC_Shen',id))['iq']))
                FC_rest.append(temp_fc)
                TS_rest.append(temp_ts)
        
        FC_rest = torch.cat(FC_rest,dim=0).view(-1,268,268)
        TS_rest = torch.cat(TS_rest,dim=0).view(-1,268,375)
        IQ_rest = torch.cat(IQ_rest,dim=0).view(-1,1)
        
        mask = np.isin(TrainIdx[fold][0]-1, np.array(usable_idx))
        trainidx = TrainIdx[fold][0][mask]-1
        trainidx = np.concatenate((trainidx,np.arange(FC_rest.shape[0])+FC.shape[0]), axis=0)
        mask = np.isin(ValidIdx[fold][0]-1, np.array(usable_idx))
        valididx = ValidIdx[fold][0][mask]-1
        mask = np.isin(TestIdx[fold][0]-1, np.array(usable_idx))
        testidx = TestIdx[fold][0][mask]-1
        trainidx = np.concatenate((trainidx,valididx), axis=0)
        
        IQ = torch.cat([IQ.view(-1,1),IQ_rest],dim=0)
        FC = torch.cat([FC,FC_rest],dim=0)
        Time_series = torch.cat([Time_series,TS_rest],dim=0)
        
        # ### Construct graph skeleton (group-level) by thresholding
        if args.threshmode == 1:
            # select top k percent absolute average values
            mean_FC = FC.abs().mean(dim=0)
            mean_FC = mean_FC.triu(1)
            # args.k_ratio = 0.25
            v,i = mean_FC[mean_FC>0].topk(k=int(134*267*args.k_ratio))
            mask = mean_FC>v[-1]
            mask = mask.to(torch.long)
            
        elif args.threshmode == 2:
            # select bottom k percent Consistency
            mean_FC = FC.abs().mean(dim=0)
            std_FC = FC.abs().std(dim=0)
            mean_FC = std_FC / mean_FC
            mean_FC = mean_FC.triu(1)
            # args.k_ratio = 0.3
            v,i = mean_FC[mean_FC>0].topk(k=int(134*267*args.k_ratio),largest=False)
            mask = mean_FC<v[-1]
            mask = mask.to(torch.long)
            
        else:
            # select top k percent absolute average values per roi
            mean_FC = FC.abs().mean(dim=0)
            mask = torch.zeros_like(mean_FC)
            # args.k_ratio = 0.1
            for i in range(mean_FC.shape[0]):
                v,i = mean_FC[i].topk(k=int(268*args.k_ratio))
                temp = mean_FC[i]>v[-1]
                mask[i] = temp.to(torch.float)
            mask = mask + mask.T
            mask[mask == 2] = 1     
            mask = mask.triu(1)
        
        fc = mean_FC * mask
        skeleton = fc.to_sparse()
        if gutils.contains_isolated_nodes(skeleton.indices(), num_nodes=268):
            print('contain isolated nodes')
        
        # build graph
        par1 = adj2par1(skeleton.indices(), fc.shape[0], skeleton.indices().shape[-1]).to_dense()
        
        L0 = torch.matmul(par1, par1.T)
        lambda0, _ = torch.linalg.eigh(L0)
        L0 = 2*L0 / lambda0.max()
        ei0, ew0 = dense_to_sparse(L0)
        maxeig = lambda0.max()
        L1 = 2*torch.matmul(par1.T, par1)/maxeig
        ei1, ew1 = dense_to_sparse(L1)
        
        # precompute pooling
        pool_num = 1
        data = PairData(x_s=torch.ones(L1.shape[0],1), edge_index_s=ei1, 
                        edge_weight_s=ew1,x_t=torch.ones(L0.shape[0],1), 
                        edge_index_t=ei0, edge_weight_t=ew0,
                        edge_index=skeleton.indices())
        data.num_node1=L0.shape[0]
        data.num_nodes=L0.shape[0]
        data.num_edge1=L1.shape[0]
        datas = [data]
        
        for i in range(pool_num):
            data, c_node, c_edge = MLGC(data)
            datas[i].pos_s = c_edge
            datas[i].pos_t = c_node 
            datas.append(data)

        num_nodepedge = int(c_node.max() + c_edge[~torch.isinf(c_edge)].max())+2
        if gutils.contains_isolated_nodes(datas[-1].edge_index_s, num_nodes=datas[-1].num_edge1):
            print('contain isolated nodes')
        
        if args.normmode == 0:
            normmode = 'group_roi'
        elif args.normmode == 1:
            normmode = 'subject_roi'
        elif args.normmode == 2:
            normmode = 'subject_all'
        else:
            normmode = 'NA'
        X_T = time_norm(Time_series=Time_series, trainidx=trainidx, mode=normmode)
        Y = (IQ - IQ[trainidx].mean())/IQ[trainidx].std()

        print(FC[trainidx].shape)
        trainset = ABCD_MLGC('ABCD', X_T[trainidx], FC[trainidx], Y[trainidx], datas, pool_num=1)
        validset = ABCD_MLGC('ABCD', X_T[valididx], FC[valididx], Y[valididx], datas, pool_num=1)
        testset = ABCD_MLGC('ABCD', X_T[testidx], FC[testidx], Y[testidx], datas, pool_num=1)
        
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
        
        #### define and load model
        # data = torch.load('BRAIN_DEMO.pt')
        # datas = data['graph']
        # num_nodepedge = int((datas[1].x_t.shape[0]+datas[1].x_s.shape[0])/4)
        
        mlp_channels = [] if args.mlp_channels==0 else [256]*args.mlp_channels
        mlp_num = 0 if args.mlp_channels==0 else 1
        model = HL_HGCNN_ABCD_dense_int3_attpool(channels=[args.c1,args.c2,args.c3], filters=[args.filters,args.filters*2,args.filters*4],
                                                  pool_loc=[0], mlp_channels=mlp_channels, K=args.K, dropout_ratio=args.dropout_ratio, 
                                                  num_nodepedge=num_nodepedge).to(device)#.to(torch.float)
        temp = str(args.c1) +str(args.c2)+str(args.c3) +'conv'+str(args.filters)
        save_name = 'HGCNN_dense_int3'+'_spool_ABCD_'+temp+'_k'+str(args.K)+'_threshmode'+str(args.threshmode)+'_normmode'+str(args.normmode)+'_kratio'+str(args.k_ratio*100)+'_seed'+str(args.seed)+'_mlp'+str(mlp_num)+'_FOLD{}'.format(fold)
        
        # model(datas)
        
        save_path = filename + save_name + '.pt'
        txt_path = txtfile + save_name + '.txt'
        load_path = filename + save_name + '.pt'
        if args.finetune==1:
            model.load_state_dict(torch.load(save_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, 
                                                                patience=5, factor=0.5, min_lr=1e-6)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = torch.nn.MSELoss()
        criterion1 = torch.nn.L1Loss()

        
        best_corr, best_loss, best_rmse = test(test_loader)
        print('==================================================================================')
        print(f'Test Loss: {best_loss:.4f}, Test Corr: {best_corr:.4f}, Test RMSE: {best_rmse:.4f}')
        print('==================================================================================')
        
        
        if args.test == 0:
            for epoch in range(1, 20):
                total_loss, train_corr = train(train_loader)
                valid_corr, valid_loss, valid_rmse = test(valid_loader)
                scheduler.step(valid_loss)
                if optimizer.param_groups[-1]['lr']<5e-6:
                    break    
                
                elapsed = (time.time()-start) / 60
                with open(txt_path, "a") as f:
                    f.write(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Train Corr: {train_corr:.4f}, Valid Loss: {valid_loss:.4f}, Valid Corr: {valid_corr:.4f}, Valid RMSE: {valid_rmse:.4f}')
                print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Train Corr: {train_corr:.4f}, Valid Loss: {valid_loss:.4f}, Valid Corr: {valid_corr:.4f}, Valid RMSE: {valid_rmse:.4f}')
                if valid_loss<best_loss:
                    best_loss = valid_loss
                    torch.save(model.state_dict(), save_path)
                    print('Model saved! \n')  
                    best_corr1, best_loss1, best_rmse1 = test(test_loader)
                    print('==================================================================================')
                    print(f'Test Loss: {best_loss1:.4f}, Test Corr: {best_corr1:.4f}, Test RMSE: {best_rmse1:.4f}')
                    print('==================================================================================')
        # else:
            # testset = Subset(dataset, randseed_all['testset'][fold][0].squeeze()-1)
            # test_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
            # model.load_state_dict(torch.load(load_path))
            # best_corr, best_loss, best_rmse = test(test_loader)
            # print('==================================================================================')
            # print(f'Test Loss: {best_loss:.4f}, Test Corr: {best_corr:.4f}, Test RMSE: {best_rmse:.4f}')
            # print('==================================================================================')
            
            # mat_path = './Visualization/HGAT_pyr_all_mlp1_fold{}.mat'.format(fold)
            # xs, ys, y_pred = visualize(test_loader, model)
            # data_zip = {'x':xs.numpy(), 'y':ys.numpy(), 'y_pred':y_pred.numpy()}
            # savemat(mat_path, data_zip) 

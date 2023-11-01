#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:10:46 2023

@author: jinghan
"""


###############################################################################
##################### network in network #########################
###############################################################################

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
from lib.Hodge_ST_Model import *
import torchvision as tv
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import unbatch
from torchmetrics import F1Score
from torchmetrics.classification import BinaryF1Score
# int before each conv
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument('--c1', type=int, default=4, help='layer num in each channel')
parser.add_argument('--c2', type=int, default=4, help='layer num in each channel')
parser.add_argument('--c3', type=int, default=4, help='layer num in each channel')
parser.add_argument('--filters', type=int, default=32, help='filter num in the first channel')
parser.add_argument('--mlp_channels', type=int, default=256, help='mlp_channels')
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--aug_prob', type=int, default=75, help='augmentation probability%')
parser.add_argument('--l2', type=float, default=1e-3, help='dropout_ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.25, help='dropout_ratio')
parser.add_argument('--K', type=int, default=4, help='polynomial order') 
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--fold', type=int, default=-1, help='The second pooling loc')
parser.add_argument('--test', type=int, default=0, help='The second pooling loc')
args = parser.parse_args()

###############################################################################
##################### attention interaction #########################
###############################################################################
from torch_geometric.utils import subgraph

class FocalLoss(nn.Module): 
    def __init__(self, alpha=0.25, gamma=2, weight=None): 
        super(FocalLoss, self).__init__() 
        self.alpha = alpha 
        self.gamma = gamma 
        self.weight = weight 
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight) 
        
    def forward(self, preds, labels): 
        logpt = -self.bce_fn(preds, labels) 
        pt = torch.exp(logpt) 
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt 
        return loss*1e4
    
def dropout_node(edge_index: Tensor, 
                  edge_attr: Tensor, 
                  y_loc: Tensor,
                  p: float = 0.0,
                  num_nodes: Optional[int] = None,
                  training: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                          f'(got {p}')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_attr, edge_mask, node_mask
    y_loc = y_loc.to(torch.bool)
    prob = torch.rand(num_nodes, device=edge_index.device)
    p = p + np.random.rand(1)[0]/2
    node_mask = prob > p
    node_mask = torch.logical_or(node_mask, y_loc)
    edge_index, edge_attr, edge_mask = subgraph(node_mask, edge_index,edge_attr,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_attr, edge_mask, node_mask

class TSP_EigPE(Dataset):
    def __init__(self, root, dataset, keig=8, num_pool=2, if_aug=False, aug_prob=0.75):
        # data aug
        self.root = root
        self.dataset = dataset
        self.keig = keig
        self.num_pool = num_pool
        self.if_aug = if_aug
        self.node_dim = 2
        self.edge_dim = 1
        self.aug_prob = aug_prob
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['TSP_alleig_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return len(self.dataset)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'TSP_alleig_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
        data.x_t = data.x_t[:,:2]
        data.x_s = data.x_s[:,:1]
        seed = torch.initial_seed()
        torch.seed()
        if self.if_aug:
            if (torch.rand(1)>self.aug_prob)[0]:
                edge_index, edge_attr,edge_mask, node_mask = dropout_node(edge_index=data.edge_index_s,
                                                        edge_attr=data.edge_weight_s, y_loc=data.y, training=False)
            else:                
                edge_index, edge_attr,edge_mask, node_mask = dropout_node(edge_index=data.edge_index_s,
                                                        edge_attr=data.edge_weight_s, y_loc=data.y, training=True)
        else:
            edge_index, edge_attr,edge_mask, node_mask = dropout_node(edge_index=data.edge_index_s,
                                                    edge_attr=data.edge_weight_s, y_loc=data.y, training=False)
        torch.manual_seed(seed)
        data.edge_index_s, data.edge_weight_s = edge_index, edge_attr
        data.x_s = torch.cat([data.x_s, node_mask.to(torch.float).view(-1,1)], dim=-1)
        return data
        
    def process(self):
        i=0
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        for data in loader:
            ea = torch.cat([data.edge_attr.view(-1,1), data.y.view(-1,1)], dim=-1)
            edge_index,edge_attr = to_undirected(data.edge_index, ea,reduce='min')
            idx = edge_index[0]<edge_index[1]
            edge_index,edge_attr = edge_index[:,idx], edge_attr[idx]
            y = edge_attr[:,1]
            edge_attr = edge_attr[:,0]
            par1 = adj2par1(edge_index, data.pos.shape[0], edge_index.shape[1]).to_dense()
            L0 = torch.matmul(par1, par1.T)
            lambda0, _ = torch.linalg.eigh(L0)
            maxeig = lambda0.max()
            L0 = 2*torch.matmul(par1, par1.T)/maxeig
            L1 = 2*torch.matmul(par1.T, par1)/maxeig
            node_pe = eig_pe(L0, k=100)
            edge_pe = eig_pe(L1, k=100)
            x_t = data.pos
            x_s = edge_attr.view(-1,1)
            x_s = torch.cat([x_s.to(torch.float),edge_pe], dim=-1)
            x_t = torch.cat([x_t.to(torch.float),node_pe], dim=-1)
            data = PairData(x_s=x_s, edge_index_s=None, edge_weight_s=None,
                              x_t=x_t, edge_index_t=None, edge_weight_t=None,
                              y = y)
            edge_index_t, edge_weight_t = dense_to_sparse(L0)
            edge_index_s, edge_weight_s = dense_to_sparse(L1)
            data.edge_index_t, data.edge_weight_t = edge_index_t, edge_weight_t
            data.edge_index_s, data.edge_weight_s = edge_index_s, edge_weight_s
            data.num_node1 = data.x_t.shape[0]
            data.num_edge1 = data.x_s.shape[0]
            data.num_nodes = data.x_t.shape[0]
            data.edge_index=edge_index
            data_zip = {'graph':data, 'maxeig':lambda0.max(), 'par1':par1}
            torch.save(data_zip, osp.join(self.processed_dir, 'TSP_alleig_'+str(i+1)+'.pt'))
            i += 1
            
###############################################################################
##################### Only Edge Convolution & Pooling #########################
###############################################################################
            
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

    def forward(self, data, if_final=False):

        # 1. Obtain node embeddings
        n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_node1)], dim=-1)
        n_batch = n_batch.to(device)
        
        s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(data.num_edge1)], dim=-1)
        s_batch = s_batch.to(device)
        # x_s, edge_index_s, edge_weight_s = data.x_s[:,:1], data.edge_index_s, data.edge_weight_s
        x_s, edge_index_s, edge_weight_s = data.x_s[:,:1], data.edge_index_s, data.edge_weight_s
        edge_mask = data.x_s[:,1:]
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
        x_t2s = torch.sparse.mm(par_1.transpose(0,1), x_t).abs()/2
        x_s = torch.cat([x_s,x_t2s], dim=-1)
        if len(self.mlp_channels)==1:
            x_s = self.mlp(x_s, edge_index_s, edge_weight_s)
        if if_final:
            return self.out(x_s, edge_index_s, edge_weight_s), s_batch, x_s
        else:
            return self.out(x_s, edge_index_s, edge_weight_s)*edge_mask, s_batch


###############################################################################
################################# Train & Test ################################
###############################################################################
def visualize_TSP(loader, model, device='cuda:0'):
    model.eval()
    outs = []
    y_pred = []
    ys = []
     
    for idx, data in enumerate(loader):  # Iterate in batches over the training/test dataset. 
        if data is not list:
            data = data.to(device)
            ys.append(data.y.cpu())
        else:
            ys.append(data[0].y.cpu())
        with torch.no_grad():
            out, s_batch, x = model(data, if_final=True)
            # outs.append(x.detach().cpu())
            y_pred.append(out.detach().cpu())
    # outs = torch.cat(outs, dim=0)
    ys = torch.cat(ys, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    return outs, ys, y_pred


def train(train_loader):
    model.train()
    total_loss = 0
    N = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y = data.y.view(-1,1)
        out, s_batch = model(data)
        loss = criterion(out, y)# + criterion1(out, y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()

    train_loss = total_loss / N
    return train_loss


def test(loader):
    model.eval()
    total_loss = 0
    N = 0
    y_f1 = []
    for data in loader:
        data = data.to(device)
        y = data.y.view(-1,1)
        with torch.no_grad():
            out, s_batch = model(data)
        # print(out.shape,y.shape)
        loss = criterion(out, y)# + criterion1(out, y)
        total_loss += loss.item() * data.num_graphs
        out = unbatch(out, s_batch)
        y = unbatch(y, s_batch)
        for y_pred, y_true in zip(out,y):
            y_f1.append(f1(y_pred.view(-1), y_true.view(-1)))

        N += data.num_graphs

    y_f1 = torch.tensor(y_f1)
    test_perf = y_f1.mean()
    test_loss = total_loss/N
    return test_loss, test_perf


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    start = time.time()
    name = 'TSP'

    keig = 10
    f1 = BinaryF1Score().to(device)
    if args.fold == -1:
        folds = [0,1,2,3,4]
    else:
        folds = [args.fold]
    for fold in folds:
        torch.manual_seed(fold)
        print('Fold {} begin'.format(fold))
        # build file path
        temp = str(args.c1) +str(args.c2)+str(args.c3) +'conv'
        if args.mlp_channels == 0:
            model = HL_HGCNN(channels=[args.c1,args.c2,args.c3], filters=[args.filters,args.filters*2,args.filters*4], mlp_channels=[], 
                              K=args.K, dropout_ratio=args.dropout_ratio, dropout_ratio_mlp=0.0, keig=keig).to(device) 
            save_name = 'HGCNN_dense_int3_eigpe'+str(keig)+'_pyr_focal_aug'+str(args.aug_prob)+'_TSP_'+temp+'_k'+str(args.K)+'_mlp0_FOLD{}'.format(fold)
        else:            
            model = HL_HGCNN(channels=[args.c1,args.c2,args.c3], filters=[args.filters,args.filters*2,args.filters*4], mlp_channels=[args.mlp_channels], 
                              K=args.K, dropout_ratio=args.dropout_ratio, dropout_ratio_mlp=0.0, keig=keig).to(device) 
            save_name = 'HGCNN_dense_int3_eigpe'+str(keig)+'_pyr_focal_aug'+str(args.aug_prob)+'_TSP_'+temp+'_k'+str(args.K)+'_mlp1_FOLD{}'.format(fold)
        save_path = './weights/' + save_name + '.pt'
        txt_path = './records/' + save_name + '.txt'
        # load_path = './selected_model_para/' + save_name + '.pt'
        load_path = './weights/' + save_name + '.pt'
        mat_path = './Visualization/' + save_name + '_all.mat'
        
        if args.test == 1:
            model.load_state_dict(torch.load(load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, 
                                                                patience=5, factor=0.5, min_lr=1e-6)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = FocalLoss()
        # criterion1 = SoftDiceLoss()
        batch_size = args.batch_size
        
        trainset = GNNBenchmarkDataset(root=name, name=name, split='train')
        valset = GNNBenchmarkDataset(root=name, name=name, split='val')
        testset = GNNBenchmarkDataset(root=name, name=name, split='test')
        trainset = TSP_EigPE(root=osp.join(name,'train_data'), dataset=trainset, keig=keig+1, if_aug=True, aug_prob=args.aug_prob/100)
        valset = TSP_EigPE(root=osp.join(name,'val_data'), dataset=valset, keig=keig+1)
        testset = TSP_EigPE(root=osp.join(name,'test_data'), dataset=testset, keig=keig+1)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valset, batch_size=batch_size, num_workers=4)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
        
        best_loss, best_acc = test(test_loader)
        print('==================================================================================')
        print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
        print('==================================================================================')
        if args.test == 0:
            for epoch in range(1, 400):
                total_loss = train(train_loader)
                valid_loss, valid_acc = test(valid_loader)
                scheduler.step(total_loss)
                if optimizer.param_groups[-1]['lr']<1e-5:
                    break
        
                elapsed = (time.time()-start) / 60
                with open(txt_path, "a") as f:
                    f.write(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}\n')
                print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
                if valid_acc>0.75 and valid_acc>best_acc:
                    best_acc = valid_acc
                    torch.save(model.state_dict(), save_path)
                    print('Model saved! \n')   
                    best_loss1, best_acc1 = test(test_loader)
                    print('==================================================================================')
                    print(f'Test Loss: {best_loss1:.4f}, Test Acc: {best_acc1:.4f}')
                    print('==================================================================================')
                    
            model.load_state_dict(torch.load(save_path))
            best_loss, best_acc = test(test_loader)
            print('==================================================================================')
            print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
            print('==================================================================================')
        # else:
        #     best_loss, best_acc = test(test_loader)
        #     print('==================================================================================')
        #     print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
        #     print('==================================================================================')
        #     outs, ys, y_pred = visualize_TSP(test_loader, model)
        #     data_zip = {'y':ys.numpy(), 'y_pred':y_pred.numpy()}
        #     # data_zip = {'x':outs[:20000,:].numpy(), 'y':ys[:20000].numpy(), 'y_pred':y_pred[:20000].numpy()}
        #     savemat(mat_path, data_zip) 

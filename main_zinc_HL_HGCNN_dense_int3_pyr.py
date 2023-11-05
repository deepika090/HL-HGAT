#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:43:44 2023

@author: jinghan
"""

import argparse
import numpy as np
from torch.nn import Linear, Dropout
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch_geometric.nn as gnn
import torch.nn as nn

from torch_geometric.loader import DataLoader
import time
from lib.Hodge_Cheb_Conv import *
from lib.Hodge_Dataset import *
from lib.Hodge_ST_Model import *
from lib.Loss_function import *

parser = argparse.ArgumentParser()
parser.add_argument('--c1', type=int, default=2, help='layer num in each channel')
parser.add_argument('--c2', type=int, default=3, help='layer num in each channel')
parser.add_argument('--c3', type=int, default=3, help='layer num in each channel')
parser.add_argument('--filters', type=list, default=64, help='filter num in the first channel')
parser.add_argument('--mlp_channels', type=int, default=2, help='mlp_channels')
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--l2', type=float, default=1e-3, help='weight decay')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout_ratio')
parser.add_argument('--K', type=int, default=6, help='polynomial order')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--fold', type=int, default=-1, help='fold index')
parser.add_argument('--test', type=int, default=0, help='whether testing')
args = parser.parse_args()

###############################################################################
################################# Train & Test ################################
###############################################################################
class HL_HGCNN_zinc_dense_int3_pyr(torch.nn.Module):
    def __init__(self, channels=[2,2,2,2], filters=[64,128,256,512], mlp_channels=[], K=2, node_dim=21, 
                  edge_dim=3, num_classes=1, dropout_ratio=0.0, dropout_ratio_mlp=0.0, 
                  keig=7):
        super(HL_HGCNN_zinc_dense_int3_pyr, self).__init__()
        self.channels = channels
        self.filters = filters#[]
        self.initial_channel = self.filters[0]
        self.mlp_channels = mlp_channels
        self.node_embedding = nn.Embedding(28,self.initial_channel-keig)
        self.edge_embedding = nn.Embedding(4,self.initial_channel-keig)
        self.node_dim = self.initial_channel#node_dim + keig
        self.edge_dim = self.initial_channel#edge_dim + keig

        
        layers = [(HodgeLaguerreConv(self.node_dim, self.initial_channel, K=1),
                    'x_t, edge_index_t, edge_weight_t -> x_t'),
                  (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
                  (nn.ReLU(), 'x_t -> x_t'),
                  # (Dropout(p=dropout_ratio), 'x_t -> x_t'),
                  (HodgeLaguerreConv(self.edge_dim, self.initial_channel, K=1),
                    'x_s, edge_index_s, edge_weight_s -> x_s'),
                  (gnn.BatchNorm(self.initial_channel), 'x_s -> x_s'),
                  (nn.ReLU(), 'x_s -> x_s'),
                  # (Dropout(p=dropout_ratio), 'x_s -> x_s'),
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
        x_t = torch.cat([self.node_embedding(x_t[:,:1].to(torch.long)).squeeze(),x_t[:,1:]], dim=-1)
        x_s = torch.cat([self.node_embedding(x_s[:,:1].to(torch.long)).squeeze(),x_s[:,1:]], dim=-1)
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
        
def train(loader):
    model.train()
    total_loss = 0
    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data)
        loss = criterion(out.view(-1,1), data.y.view(-1,1))#.view(-1,1))  # Compute the loss.
        total_loss += loss*data.num_graphs
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return total_loss/len(loader.dataset)


def test(loader):
      model.eval()
      total_loss, acc = 0, 0
     
      for data in loader:  # Iterate in batches over the training/test dataset. 
          data = data.to(device)
          with torch.no_grad():
            out = model(data) 
          loss = criterion(out.view(-1,1), data.y.view(-1,1))#.view(-1,1))  # Compute the loss.
          acc += torch.abs(out.view(-1)-data.y.view(-1)).sum()
          total_loss += loss * data.num_graphs

      return total_loss/len(loader.dataset), acc/len(loader.dataset)*2.0109


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    start = time.time()
    keig = 15
    name = 'ZINC'
    if not os.path.exists(name):
        os.makedirs(name)
    for fold in range(0,5):
        print('Fold {} begin'.format(fold))
        torch.manual_seed(fold)
        temp = str(args.c1) +str(args.c2)+str(args.c3) +'conv'
        mlp_channels = [] if args.mlp_channels==0 else [256]*args.mlp_channels
        mlp_num = args.mlp_channels
        
        model = HL_HGCNN_zinc_dense_int3_pyr(channels=[args.c1,args.c2,args.c3], filters=[args.filters,args.filters*2,args.filters*4], mlp_channels=mlp_channels, 
                          K=args.K, dropout_ratio=args.dropout_ratio, dropout_ratio_mlp=0.0, keig=keig).to(device) 
        save_name = 'HGCNN_dense_int3_eigpe_pyr_ZINC_'+temp+'_k'+str(args.K)+'_batch'+str(args.batch_size)+'_mlp'+str(mlp_num)+'_FOLD{}'.format(fold)

        save_path = './weights/' + save_name + '.pt'
        txt_path = './records/' + save_name + '.txt'
        load_path = './weights/' + save_name + '.pt'
        mat_path = './Visualization/' + save_name + '.mat'
        
        if args.test == 1:
            model.load_state_dict(torch.load(load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, 
                                                                patience=10, factor=0.5, min_lr=1e-6)

        criterion = torch.nn.L1Loss()
        batch_size = args.batch_size
        
        cifar_train = ZINC(root=name,subset=True,split='train')
        cifar_val = ZINC(root=name,subset=True,split='val')
        cifar_test = ZINC(root=name,subset=True,split='test')
        trainset = ZINC_HG_BM_par1_EigPE(root=osp.join('ZINC','train_data'), dataset=cifar_train, keig=keig+1)
        validset = ZINC_HG_BM_par1_EigPE(root=osp.join('ZINC','val_data'), dataset=cifar_val, keig=keig+1)
        testset = ZINC_HG_BM_par1_EigPE(root=osp.join('ZINC','test_data'), dataset=cifar_test, keig=keig+1)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=4)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
        
        best_loss, best_acc = test(test_loader)
        print('==================================================================================')
        print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
        print('==================================================================================')
        if args.test == 0:
            for epoch in range(1, 600):
                total_loss = train(train_loader)
                valid_loss, valid_acc = test(valid_loader)
                scheduler.step(valid_loss)
                # if optimizer.param_groups[-1]['lr']<1e-5:
                #     break
        
                elapsed = (time.time()-start) / 60
                print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
                if valid_acc<0.4 and valid_acc<best_acc:
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
        #     outs, ys, y_pred = visualize(test_loader, model)
        #     outs1, ys1, y_pred1 = visualize(valid_loader, model)
        #     outs = torch.cat([outs,outs1], dim=0)
        #     ys = torch.cat([ys,ys1],dim=0)
        #     y_pred = torch.cat([y_pred,y_pred1],dim=0)
        #     data_zip = {'x':outs.numpy(), 'y':ys.numpy(), 'y_pred':y_pred.numpy()}
        #     savemat(mat_path, data_zip)   



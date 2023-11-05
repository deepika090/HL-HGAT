#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 19:26:52 2023

@author: jinghan
"""

import argparse
import torch
from torch_geometric.loader import DataLoader
import time
from lib.Hodge_Cheb_Conv import *
from lib.Hodge_Dataset import *
from lib.Hodge_ST_Model import *
from lib.Loss_function import *

parser = argparse.ArgumentParser()
parser.add_argument('--c1', type=int, default=2, help='layer num in each channel')
parser.add_argument('--c2', type=int, default=2, help='layer num in each channel')
parser.add_argument('--c3', type=int, default=2, help='layer num in each channel')
parser.add_argument('--filters', type=list, default=64, help='filter num in the first channel')
parser.add_argument('--mlp_channels', type=int, default=256, help='mlp_channels')
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--l2', type=float, default=1e-3, help='weight decay')
parser.add_argument('--dropout_ratio', type=float, default=0.25, help='dropout_ratio')
parser.add_argument('--K', type=int, default=6, help='polynomial order')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--fold', type=int, default=-1, help='fold index')
parser.add_argument('--test', type=int, default=0, help='whether testing')
args = parser.parse_args()
###############################################################################
################################# Train & Test ################################
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
            fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize, only_att=True, l=0.5)
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


    def forward(self, datas, device='cuda:0', if_att=False, if_final_layer=False):
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
                fc = getattr(self, 'NEInt{}{}'.format(i,j))
                x_t, x_s = fc(x_t0, x_s0, par_1, D)
                fc = getattr(self, 'NEConv{}{}'.format(i,j))
                x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
                x_t0 = torch.cat([x_t0, x_t], dim=-1)
                x_s0 = torch.cat([x_s0, x_s], dim=-1)
            fc = getattr(self, 'NEAtt%d' % i)
            att_t, att_s = fc(x_t0, x_s0, par_1, D)
            x_t0 = x_t0 * att_t
            x_s0 = x_s0 * att_s
                    
            # structural pooling        
            if i == self.pool_loc:
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

        if if_att:
            return self.out(x), att_t, att_s
        elif if_final_layer:
            return x, self.out(x)
        else:
            return self.out(x)


def train(train_loader):
    model.train()
    total_loss = 0
    N = 0
    y_preds, y_trues = [], []
    time1=0
    for data in train_loader:
        optimizer.zero_grad()
        mask = ~torch.isnan(data[0].y)
        y = data[0].y.to(torch.float)
        time3 = time.time()
        out, att_t, att_s = model(data, if_att=True)
        time2 = time.time()
        y_preds.append(out)
        y_trues.append(y)
        loss = criterion(out[mask], y[mask].to(device))# + 1e-5*att_t.abs().mean() + 1e-5*att_s.abs().mean()
        # print(criterion(out[mask], y[mask].to(device)), att_t.abs().mean(), att_s.abs().mean())
        loss.backward()
        total_loss += loss.item() * data[0].num_graphs
        N += data[0].num_graphs
        optimizer.step()

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)

    train_perf = eval_ap(y_true=y_trues.to(device), y_pred=y_preds)
    train_loss = total_loss / N
    return train_loss, train_perf


def test(loader):
    model.eval()
    total_loss = 0
    N = 0
    y_preds, y_trues = [], []
    attts, attss = 0, 0
    for data in loader:
        mask = ~torch.isnan(data[0].y)
        y = data[0].y.to(torch.float)
        with torch.no_grad():
            out, att_t, att_s = model(data, if_att=True)
        attts += torch.count_nonzero(att_t<0.5)
        attss += torch.count_nonzero(att_s<0.5)
        y_preds.append(out)
        y_trues.append(y)
        loss = criterion(out[mask], y[mask].to(device))
        total_loss += loss.item() * data[0].num_graphs
        N += data[0].num_graphs

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    print('att_t:{} att_s:{}'.format(attts/N, attss/N))
    test_perf = eval_ap(y_true=y_trues.to(device), y_pred=y_preds)
    test_loss = total_loss/N
    return test_loss, test_perf


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    start = time.time()
    name = 'Peptides-func'
    keig = 10
    if args.fold == -1:
        folds = [0,1,2,3]
    else:
        folds = [args.fold]
        
    for fold in folds:
        # build file path
        temp = str(args.c1) +str(args.c2)+str(args.c3) +'conv'
        torch.manual_seed(fold*300)
        print('Fold {} begin'.format(fold))

        if args.mlp_channels == 0:
            model = HL_HGCNN_pepfunc_dense_int3_attpool(channels=[args.c1,args.c2,args.c3], filters=[args.filters,args.filters*2,args.filters*4], mlp_channels=[], 
                              pool_loc=1, K=args.K, dropout_ratio=args.dropout_ratio, dropout_ratio_mlp=0.0, keig=keig).to(device) 
            save_name = 'HGCNN_dense_int3_eigpe'+str(keig)+'_attpool_attout_focal'+'_pepfunc_'+temp+'_k'+str(args.K)+'_mlp0_FOLD{}'.format(fold)
        else:            
            model = HL_HGCNN_pepfunc_dense_int3_attpool(channels=[args.c1,args.c2,args.c3], filters=[args.filters,args.filters*2,args.filters*4], mlp_channels=[args.mlp_channels], 
                              pool_loc=1, K=args.K, dropout_ratio=args.dropout_ratio, dropout_ratio_mlp=0.0, keig=keig).to(device) 
            save_name = 'HGCNN_dense_int3_eigpe'+str(keig)+'_attpool_attout_focal'+'_pepfunc_'+temp+'_k'+str(args.K)+'_mlp1_FOLD{}'.format(fold)

        save_path = './Ablation_weights/' + save_name + '.pt'
        txt_path = './records/' + save_name + '.txt'
        load_path = './Ablation_weights/' + save_name + '.pt'
        mat_path = './Visualization/' + save_name + '.mat'
        if args.test == 1:
            model.load_state_dict(torch.load(save_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, 
                                                               patience=10, factor=0.5, min_lr=1e-6, mode='max')
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = FocalLoss()#torch.nn.BCEWithLogitsLoss()
        
        batch_size = args.batch_size
        
        cifar_train = LRGBDataset(root=name,name=name,split='train')
        trainset = Peptides_Func_EigPE_MLGC(root=osp.join(name,'train_data'), dataset=cifar_train, keig=keig+1)
        cifar_train = LRGBDataset(root=name,name=name,split='val')
        validset = Peptides_Func_EigPE_MLGC(root=osp.join(name,'val_data'), dataset=cifar_train, keig=keig+1)
        cifar_train = LRGBDataset(root=name,name=name,split='test')
        testset = Peptides_Func_EigPE_MLGC(root=osp.join(name,'test_data'), dataset=cifar_train, keig=keig+1)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=2)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=2)
        
        best_loss, best_acc = test(test_loader)
        print('==================================================================================')
        print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
        print('==================================================================================')
        if args.test == 0:
            for epoch in range(1, 600):
                total_loss,_ = train(train_loader)
                valid_loss, valid_acc = test(valid_loader)
                scheduler.step(valid_acc)
                if optimizer.param_groups[-1]['lr']<1e-5:
                    break
        
                elapsed = (time.time()-start) / 60
                print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
                if valid_acc>0.5 and valid_acc>best_acc:
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
        

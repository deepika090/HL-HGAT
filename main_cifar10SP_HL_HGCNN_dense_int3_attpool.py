
##################################################################### cpu aug
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:26:04 2023

@author: jinghan
"""

import argparse
from torch.nn import Linear, Dropout
import torch
import torch_geometric.nn as gnn
import torch.nn as nn
from torch_geometric.loader import DataLoader
import time
from lib.Hodge_Cheb_Conv import *
from lib.Hodge_Dataset import *
from lib.Hodge_ST_Model import *
from lib.Loss_function import *

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=-1, help='The second pooling loc')
parser.add_argument('--c1', type=int, default=2, help='layer num in each channel')
parser.add_argument('--c2', type=int, default=2, help='layer num in each channel')
parser.add_argument('--c3', type=int, default=2, help='layer num in each channel')
parser.add_argument('--filters', type=list, default=64, help='filter num in the first channel')
parser.add_argument('--mlp_channels', type=int, default=1, help='mlp_channels')
parser.add_argument('--dropout_ratio', type=float, default=0.25, help='dropout_ratio')
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--l2', type=float, default=1e-3, help='weight decay')
parser.add_argument('--lt', type=float, default=1e-3, help='weight decay')
parser.add_argument('--ls', type=float, default=1e-3, help='weight decay')
parser.add_argument('--K', type=int, default=4, help='polynomial order')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--cpu', type=int, default=4, help='num of cpu')
parser.add_argument('--test', type=int, default=0, help='whether testing')
parser.add_argument('--finetune', type=int, default=0, help='whether finetuning')
args = parser.parse_args()

###############################################################################
################################# Train & Test ################################
###############################################################################


class CIFAR10SP_EigPE_MLGC(Dataset):
    def __init__(self, root, dataset, keig=10, num_pool=1, if_aug=False):
        # data aug
        self.root = root
        self.dataset = dataset
        self.if_aug = if_aug
        self.keig = keig
        self.num_pool = num_pool
        self.node_dim = 5
        self.edge_dim = 4
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['CIFAR10SP_alleig_MLGC1_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return len(self.dataset)
#         return len(self.processed_file_names)

    def get(self,idx):
        seed = torch.initial_seed()
        torch.seed()
        data = self.dataset[idx]
        edge_index,edge_attr = to_undirected(data.edge_index, data.edge_attr,reduce='min')
        idx = edge_index[0]<edge_index[1]
        edge_index,edge_attr = edge_index[:,idx], edge_attr[idx]
        if self.if_aug and (torch.rand(1)>0.75)[0]:             
            edge_index, edge_mask = dropout_edge(edge_index=edge_index, training=True)
        else:
            edge_index, edge_mask = dropout_edge(edge_index=edge_index, training=False)
        edge_attr = edge_attr[edge_mask]
        
        par1 = adj2par1(edge_index, data.x.shape[0], edge_mask.shape[0]).to_dense()
        L0 = torch.matmul(par1, par1.T)
        lambda0, _ = torch.linalg.eigh(L0)
        maxeig = lambda0.max()
        L0 = 2*torch.matmul(par1, par1.T)/maxeig
        L1 = 2*torch.matmul(par1.T, par1)/maxeig
        node_pe = eig_pe(L0, k=10)
        edge_pe = torch.abs(node_pe[edge_index[0]]+node_pe[edge_index[1]])
        x_s = edge_attr.view(-1,1)
        x_t = data.x
        x_s = torch.cat([x_s.to(torch.float),torch.abs(x_t[edge_index[0]]
                                                        -x_t[edge_index[1]]),edge_pe], dim=-1)
        x_t = torch.cat([x_t,data.pos,node_pe], dim=-1)
        data = PairData(x_s=x_s, edge_index_s=None, edge_weight_s=None,
                          x_t=x_t, edge_index_t=None, edge_weight_t=None,
                          y = data.y)
        edge_index_t, edge_weight_t = dense_to_sparse(L0)
        edge_index_s, edge_weight_s = dense_to_sparse(L1)
        data.edge_index_t, data.edge_weight_t = edge_index_t, edge_weight_t
        data.edge_index_s, data.edge_weight_s = edge_index_s, edge_weight_s
        data.num_node1 = data.x_t.shape[0]
        data.num_edge1 = data.x_s.shape[0]
        data.num_nodes = data.x_t.shape[0]
        data.edge_index=edge_index
        datas = [data]
        for i in range(self.num_pool):
            # temp, c_node, c_edge = MLGC_weighted(datas[i])
            temp, c_node, c_edge = MLGC(datas[i])
            datas[i].x_t = torch.cat([c_node, datas[i].x_t], dim=-1)
            datas[i].x_s = torch.cat([c_edge, datas[i].x_s], dim=-1)
            datas.append(temp)
        # sign flip
        node_dim, edge_dim = self.node_dim+self.keig-1, self.edge_dim+self.keig-1
        sign = torch.cat([torch.ones(self.node_dim+1),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
        if datas[0].x_t.shape[1]<node_dim+1:
            datas[0].x_t = torch.cat([datas[0].x_t,torch.zeros(datas[0].x_t.shape[0], node_dim+1-datas[0].x_t.shape[1])], dim=-1) * sign
        else:
            datas[0].x_t = datas[0].x_t[:,:self.node_dim+self.keig] * sign
            
        sign = torch.cat([torch.ones(self.edge_dim+1),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
        if datas[0].x_s.shape[1]<edge_dim+1:
            datas[0].x_s = torch.cat([datas[0].x_s,torch.zeros(datas[0].x_s.shape[0], edge_dim+1-datas[0].x_s.shape[1])], dim=-1) * sign
        else:
            datas[0].x_s = datas[0].x_s[:,:self.edge_dim+self.keig] * sign
        torch.manual_seed(seed)
        return datas
        
    def process(self):
        return None
    

def train(loader):
    model.train()
    total_loss = 0
    for data in loader:  # Iterate in batches over the training dataset.
        y = data[0].y.to(device)
        out, att_t, att_s = model(data, if_att=True)
        loss = criterion(out, y)# + args.lt*att_t.square().mean() + args.ls*att_s.square().mean() #.view(-1,1))  # Compute the loss.
        
        total_loss += loss*data[0].num_graphs
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return total_loss/len(loader.dataset)


def test(loader):
    model.eval()
    total_loss, acc = 0, 0
    attts, attss = 0, 0
    for data in loader:  # Iterate in batches over the training/test dataset. 
        y = data[0].y.to(device)
        with torch.no_grad():
            out, att_t, att_s = model(data, if_att=True)
        loss = criterion(out, y) + att_t.abs().mean() + att_s.abs().mean()#.view(-1,1))  # Compute the loss.
        attts += torch.count_nonzero(att_t<0.5)
        attss += torch.count_nonzero(att_s<0.5)
        acc += torch.count_nonzero(torch.argmax(out,dim=1) == y)
        total_loss += loss * data[0].num_graphs
        
    print('att_t:{} att_s:{}'.format(attts/len(loader.dataset), attss/len(loader.dataset)))
    return total_loss/len(loader.dataset), acc/len(loader.dataset)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    start = time.time()
    keig = 10
    name = 'CIFAR10_bm'
    if not os.path.exists(name):
        os.makedirs(name)
    if args.fold == -1:
        folds = [0,1,2,3,4]
    else:
        folds = [args.fold]
    for fold in folds:
        torch.manual_seed(fold*360)
        print('Fold {} begin'.format(fold))
        temp = str(args.c1) +str(args.c2)+str(args.c3) +'conv'
        mlp_channels = [] if args.mlp_channels==0 else [256]*args.mlp_channels
        mlp_num = mlp_channels
        
        model = HL_HGCNN_CIFAR10SP_dense_int3_attpool(channels=[args.c1,args.c2,args.c3], filters=[args.filters,args.filters*2,args.filters*4],mlp_channels=[256], 
                          K=args.K, dropout_ratio=args.dropout_ratio, dropout_ratio_mlp=0.0, keig=keig, pool_loc=1, l=0.5).to(device)  # 4conv
        save_name = 'HGCNN_dense_int3_eigpe'+str(keig)+'_attpool_cifar10SP_'+temp+'_k'+str(args.K)+'_batch'+str(args.batch_size)+'_mlp'+str(mlp_num)+'_FOLD{}'.format(fold)
            
        save_path = './weights/' + save_name + '.pt'
        txt_path = './records/' + save_name + '.txt'
        # load_path = './selected_model_para/' + save_name + '.pt'
        load_path = save_path
        mat_path = './Visualization/GPS.mat'
        
        if args.test == 1:
            model.load_state_dict(torch.load(load_path))
        elif args.finetune == 1:
            model.load_state_dict(torch.load(save_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, 
                                                               patience=10, factor=0.5, min_lr=1e-6, mode='max')
        criterion = torch.nn.CrossEntropyLoss()
        batch_size = args.batch_size
        dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='train')
        trainset = CIFAR10SP_EigPE_MLGC(root=osp.join('CIFAR10_bm','train'),dataset=dataset, keig=keig+1, if_aug=True)
        
        dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='val')
        validset = CIFAR10SP_EigPE_MLGC(root=osp.join('CIFAR10_bm','val'), dataset=dataset, keig=keig+1)
        
        dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='test')
        testset = CIFAR10SP_EigPE_MLGC(root=osp.join('CIFAR10_bm','test'), dataset=dataset, keig=keig+1)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.cpu)
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
                scheduler.step(valid_acc)
                if optimizer.param_groups[-1]['lr']<1e-5:
                    break
        
                elapsed = (time.time()-start) / 60
                print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
                if valid_acc>0.6 and valid_acc>best_acc:
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
        else:
            outs, ys, y_pred = visualize(test_loader, model)
            outs1, ys1, y_pred1 = visualize(valid_loader, model)
            outs = torch.cat([outs,outs1], dim=0)
            ys = torch.cat([ys,ys1],dim=0)
            y_pred = torch.cat([y_pred,y_pred1],dim=0)
            data_zip = {'x':outs.numpy(), 'y':ys.numpy(), 'y_pred':y_pred.numpy()}
            savemat(mat_path, data_zip) 

#################################################################### no aug

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sun May  7 19:26:04 2023

# @author: jinghan
# """


# import argparse
# import numpy as np
# from torch.nn import Linear, Dropout
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool, global_max_pool
# import torch_geometric.nn as gnn
# import torch.nn as nn
# from torch.utils.data import Subset
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# import time
# from lib.Hodge_Cheb_Conv import *
# from lib.Hodge_Dataset import *
# from lib.Hodge_ST_Model import *
# import torchvision as tv
# from scipy.sparse.linalg import eigsh
# from torch_scatter import scatter_max, scatter_mean

# # int before each conv
# parser = argparse.ArgumentParser()
# parser.add_argument('--c1', type=int, default=2, help='layer num in each channel')
# parser.add_argument('--c2', type=int, default=2, help='layer num in each channel')
# parser.add_argument('--c3', type=int, default=2, help='layer num in each channel')
# parser.add_argument('--filters', type=list, default=[64,128,256], help='filter num in each channel')
# parser.add_argument('--mlp_channels', type=list, default=256, help='mlp_channels')
# parser.add_argument('--dropout_ratio', type=float, default=0.25, help='dropout_ratio')
# parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
# parser.add_argument('--K', type=int, default=6, help='polynomial order')
# parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
# parser.add_argument('--gpu', type=int, default=0, help='gpu index')
# args = parser.parse_args()


# ###############################################################################
# ##################### Only Edge Convolution & Pooling #########################
# ###############################################################################
# def MLGC(data, keig=1):
#     '''
#     multi-level graph coarsening (MLGC)
#     input: 
#        data: input graph
#        keig: dim of position encoding
#     output:
#        data: output graph
#        c_node: node assignment matrix
#        c_edge: edge assignment matrix
#     '''
#     edge_index, edge_weight = data.edge_index, torch.exp(-data.x_s[:,0]**2)
#     edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='mean')
#     c_node = graclus_cluster(edge_index[0], edge_index[1], edge_weight, data.num_node1)
#     c_unique = torch.unique(c_node)
#     d = {int(j):i for i,j in enumerate(c_unique)}
#     ei1, idx = [[],[]], 0
#     ei1_key = {}
#     c_edge = torch.zeros(data.x_s.shape[0])
#     c_node = [d[int(c)] for c in c_node]
#     for i,_ in enumerate(data.edge_index[0]):
#         if c_node[data.edge_index[0][i]] == c_node[data.edge_index[1][i]]:
#             c_edge[i] = float("inf")
#         else:
#             imax = max(c_node[data.edge_index[0][i]],(c_node[data.edge_index[1][i]]))
#             imin = min(c_node[data.edge_index[0][i]],(c_node[data.edge_index[1][i]]))
#             ekey = imax + 0.0001*imin
#             if ekey in ei1_key:
#                 c_edge[i] = ei1_key[ekey]
#             else:
#                 ei1_key[ekey] = idx
#                 ei1[0].append(imin)
#                 ei1[1].append(imax)
#                 idx += 1
#                 c_edge[i] = idx - 1
#     ei1 = torch.tensor(ei1)
#     try:
#         par1 = adj2par1(ei1, c_unique.shape[0], ei1.shape[1]).to_dense()
#     except:
#         print(ei1, c_unique, c_node, data)
#     L0 = torch.matmul(par1, par1.T)
#     lambda0, _ = torch.linalg.eigh(L0)
#     maxeig = lambda0.max()
#     L0 = 2*torch.matmul(par1, par1.T)/maxeig
#     L1 = 2*torch.matmul(par1.T, par1)/maxeig
#     node_pe = torch.ones(c_unique.shape[0],1) #eig_pe(L0.numpy(), k=keig)
#     edge_pe = torch.ones(ei1.shape[1],1) #eig_pe(L1.numpy(), k=keig)
#     eit, ewt = dense_to_sparse(L0)
#     eis, ews = dense_to_sparse(L1)
#     graph = PairData(x_s=edge_pe, edge_index_s=eis, edge_weight_s=ews,
#                      x_t=node_pe, edge_index_t=eit, edge_weight_t=ewt,)
#     graph.edge_index = ei1
#     graph.num_node1 = c_unique.shape[0]
#     graph.num_edge1 = ei1.shape[1]
#     graph.num_nodes = c_unique.shape[0]
#     return graph, torch.tensor(c_node).view(-1,1), c_edge.view(-1,1)

# def eig_pe(L, k=9):
#     eig_vals, eig_vecs = eigh(L)
#     eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()[::-1]])
#     pe = torch.from_numpy(eig_vecs[:, -1-k:-1])
#     return pe

# class CIFAR10SP_EigPE_MLGC(Dataset):
#     def __init__(self, root, dataset, keig=10, num_pool=1, if_aug=False):
#         # data aug
#         self.root = root
#         self.dataset = dataset
#         self.if_aug = if_aug
#         self.keig = keig
#         self.num_pool = num_pool
#         self.node_dim = 5
#         self.edge_dim = 4
#         super().__init__(root)
  
#     @property
#     def processed_file_names(self):
#         return ['CIFAR10SP_alleig_MLGC1_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

#     def len(self):
#         return len(self.dataset)
# #         return len(self.processed_file_names)

#     def get(self,idx):
#         data_zip = torch.load(osp.join(self.processed_dir, 'CIFAR10SP_alleig_MLGC1_'+str(idx+1)+'.pt'))
#         datas = data_zip['graph']
#         node_dim, edge_dim = self.node_dim+self.keig-1, self.edge_dim+self.keig-1
#         sign = torch.cat([torch.ones(self.node_dim+1),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
#         if datas[0].x_t.shape[1]<node_dim+1:
#             datas[0].x_t = torch.cat([datas[0].x_t,torch.zeros(datas[0].x_t.shape[0], node_dim+1-datas[0].x_t.shape[1])], dim=-1) * sign
#         else:
#             datas[0].x_t = datas[0].x_t[:,:self.node_dim+self.keig] * sign
            
#         sign = torch.cat([torch.ones(self.edge_dim+1),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
#         if datas[0].x_s.shape[1]<edge_dim+1:
#             datas[0].x_s = torch.cat([datas[0].x_s,torch.zeros(datas[0].x_s.shape[0], edge_dim+1-datas[0].x_s.shape[1])], dim=-1) * sign
#         else:
#             datas[0].x_s = datas[0].x_s[:,:self.edge_dim+self.keig] * sign
#         return datas
        
#     def process(self):
#         loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
#         for idx, _ in enumerate(loader):
#             data_zip = torch.load(osp.join(self.processed_dir, 'CIFAR10SP_alleig_'+str(idx+1)+'.pt'))
#             data = data_zip['graph']
#             datas = [data]
#             for i in range(self.num_pool):
#                 temp, c_node, c_edge = MLGC(datas[i])
#                 datas[i].x_t = torch.cat([c_node, datas[i].x_t], dim=-1)
#                 datas[i].x_s = torch.cat([c_edge, datas[i].x_s], dim=-1)
#                 datas.append(temp)
#             data_zip = {'graph':datas}
#             torch.save(data_zip, osp.join(self.processed_dir, 'CIFAR10SP_alleig_MLGC1_'+str(idx+1)+'.pt'))
#     #     data_zip = torch.load(osp.join(self.processed_dir, 'CIFAR10SP_alleig_'+str(idx+1)+'.pt'))
#     #     data = data_zip['graph']
#     #     node_dim, edge_dim = self.node_dim+self.keig-1, self.edge_dim+self.keig-1
#     #     # print(datas[0].x_t.shape, datas[0].x_s.shape)
#     #     datas = [data]
#     #     for i in range(self.num_pool):
#     #         temp, c_node, c_edge = MLGC(datas[i])
#     #         datas[i].x_t = torch.cat([c_node, datas[i].x_t], dim=-1)
#     #         datas[i].x_s = torch.cat([c_edge, datas[i].x_s], dim=-1)
#     #         datas.append(temp)
#     #     # print(datas)
#     #     sign = torch.cat([torch.ones(self.node_dim+1),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
#     #     if datas[0].x_t.shape[1]<node_dim+1:
#     #         datas[0].x_t = torch.cat([datas[0].x_t,torch.zeros(datas[0].x_t.shape[0], node_dim+1-datas[0].x_t.shape[1])], dim=-1) * sign
#     #     else:
#     #         datas[0].x_t = datas[0].x_t[:,:self.node_dim+self.keig] * sign
            
#     #     sign = torch.cat([torch.ones(self.edge_dim+1),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
#     #     if datas[0].x_s.shape[1]<edge_dim+1:
#     #         datas[0].x_s = torch.cat([datas[0].x_s,torch.zeros(datas[0].x_s.shape[0], edge_dim+1-datas[0].x_s.shape[1])], dim=-1) * sign
#     #     else:
#     #         datas[0].x_s = datas[0].x_s[:,:self.edge_dim+self.keig] * sign
#     #     return datas
        
#     # def process(self):
#     #     return None

# ###############################################################################
# ##################### Only Edge Convolution & Pooling #########################
# ###############################################################################
# class HL_HGCNN_CIFAR10SP_dense_int3_attpool(torch.nn.Module):
#     def __init__(self, channels=[2,2,2], filters=[64,128,256], mlp_channels=[], K=2, node_dim=5, 
#                   edge_dim=4, num_classes=10, dropout_ratio=0.0, dropout_ratio_mlp=0.0, pool_loc=0,
#                   keig=10):
#         super(HL_HGCNN_CIFAR10SP_dense_int3_attpool, self).__init__()
#         self.channels = channels
#         self.filters = filters#[]
#         self.mlp_channels = mlp_channels
#         self.node_dim = node_dim + keig
#         self.edge_dim = edge_dim + keig
#         self.initial_channel = self.filters[0]
#         self.pool_loc = pool_loc
#         # self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
#         # self.relu = nn.ReLU()
        
#         layers = [(HodgeLaguerreConv(self.node_dim, self.initial_channel, K=1),
#                     'x_t, edge_index_t, edge_weight_t -> x_t'),
#                   (gnn.BatchNorm(self.initial_channel), 'x_t -> x_t'),
#                   (nn.ReLU(), 'x_t -> x_t'),
#                   (Dropout(p=dropout_ratio), 'x_t -> x_t'),
#                   (HodgeLaguerreConv(self.edge_dim, self.initial_channel, K=1),
#                     'x_s, edge_index_s, edge_weight_s -> x_s'),
#                   (gnn.BatchNorm(self.initial_channel), 'x_s -> x_s'),
#                   (nn.ReLU(), 'x_s -> x_s'),
#                   (Dropout(p=dropout_ratio), 'x_s -> x_s'),
#                   (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
#         fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)
#         setattr(self, 'HL_init_conv', fc)
#         gcn_insize = self.initial_channel
            
#         for i, gcn_outsize in enumerate(self.filters):
#             for j in range(self.channels[i]):
#                 # int term
#                 fc = NodeEdgeInt(d=gcn_insize, dv = gcn_outsize)
#                 setattr(self, 'NEInt{}{}'.format(i,j), fc)
#                 # HL node edge filtering
#                 layers = [(HodgeLaguerreConv(gcn_outsize, gcn_outsize, K=K),
#                             'x_t, edge_index_t, edge_weight_t -> x_t'),
#                           (gnn.BatchNorm(gcn_outsize), 'x_t -> x_t'),
#                           (nn.ReLU(), 'x_t -> x_t'),
#                           (Dropout(p=dropout_ratio), 'x_t -> x_t'),
#                           (HodgeLaguerreConv(gcn_outsize, gcn_outsize, K=K),
#                             'x_s, edge_index_s, edge_weight_s -> x_s'),
#                           (gnn.BatchNorm(gcn_outsize), 'x_s -> x_s'),
#                           (nn.ReLU(), 'x_s -> x_s'),
#                           (Dropout(p=dropout_ratio), 'x_s -> x_s'),
#                           (lambda x1, x2: [x1,x2],'x_t, x_s -> x'),]
#                 fc = gnn.Sequential('x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s', layers)
#                 setattr(self, 'NEConv{}{}'.format(i,j), fc)
#                 gcn_insize = gcn_insize + gcn_outsize
            
#             if i == self.pool_loc:# < len(self.filters)-1:
#                 # ATT
#                 fc = NodeEdgeInt(d=gcn_outsize, dv = gcn_outsize, only_att=True, sigma=nn.ReLU())
#                 setattr(self, 'NEAtt{}'.format(i), fc)
        
#         mlp_insize = self.filters[-1] * 2 #sum(Node_channels)+ sum(Edge_channels)#[-1]
#         for i, mlp_outsize in enumerate(mlp_channels):
#             fc = nn.Sequential(
#                 Linear(mlp_insize, mlp_outsize),
#                 nn.BatchNorm1d(mlp_outsize),
#                 nn.ReLU(),
#                 nn.Dropout(dropout_ratio_mlp),
#                 )
#             setattr(self, 'mlp%d' % i, fc)
#             mlp_insize = mlp_outsize

#         self.out = Linear(mlp_insize, num_classes)


#     def forward(self, datas, device='cuda:0'):
#         # time3 = time.time() 
#         data = datas[0].to(device)
#         # 1. Obtain node embeddings
#         pos_ts, pos_ss = [], []
#         for p in range(0,1):#len(self.channels)-1):
#             n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_node1)], dim=-1)
#             n_batch = n_batch.to(device)
#             s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[p].num_edge1)], dim=-1)
#             s_batch = s_batch.to(device)
#             n_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_node1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
#             s_ahead = torch.cumsum(torch.cat([torch.zeros(1),datas[p+1].num_edge1],dim=-1).to(device), dim=0, dtype=torch.long)[:-1]
#             # n_ahead = torch.tensor([sum(datas[p+1].num_node1[:j]) for j in range(datas[p].num_node1.shape[0])])
#             # s_ahead = torch.tensor([sum(datas[p+1].num_edge1[:j]) for j in range(datas[p].num_edge1.shape[0])])
#             pos_ts.append((datas[p].x_t[:,0].to(device) + n_ahead[n_batch]).view(-1,1))
#             pos_ss.append((datas[p].x_s[:,0].to(device) + s_ahead[s_batch]).view(-1,1))
            
#         x_s, edge_index_s, edge_weight_s = data.x_s[:,1:], data.edge_index_s, data.edge_weight_s
#         x_t, edge_index_t, edge_weight_t = data.x_t[:,1:], data.edge_index_t, data.edge_weight_t
        
#         x_t, x_s = self.HL_init_conv(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
#         x_s0, x_t0 = x_s, x_t
#         k = 0
#         par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
#         D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
#         # time2 = time.time() 
#         for i, _ in enumerate(self.channels):
#             for j in range(self.channels[i]):
#                 # print(x_t.shape, x_s.shape, x_t0.shape, x_s0.shape)
#                 fc = getattr(self, 'NEInt{}{}'.format(i,j))
#                 x_t, x_s = fc(x_t0, x_s0, par_1, D)
#                 fc = getattr(self, 'NEConv{}{}'.format(i,j))
#                 x_t, x_s = fc(x_t, edge_index_t, edge_weight_t, x_s, edge_index_s, edge_weight_s)
#                 x_t0 = torch.cat([x_t0, x_t], dim=-1)
#                 x_s0 = torch.cat([x_s0, x_s], dim=-1)
                
#             # structural pooling        
#             if i == self.pool_loc:
#                 fc = getattr(self, 'NEAtt%d' % i)
#                 att_t, att_s = fc(x_t, x_s, par_1, D)
#                 att_t = att_t / att_t.max()
#                 att_s = att_s / att_s.max()
#                 x_t = x_t * att_t
#                 x_s = x_s * att_s
#                 pos_t, pos_s = pos_ts[k], pos_ss[k]
#                 x_t0 = scatter_mean(x_t0,pos_t.to(torch.long),dim=0)
#                 x_s0 = x_s0[~torch.isinf(pos_s).view(-1)]
#                 pos_s = pos_s[~torch.isinf(pos_s).view(-1)]
#                 x_s0 = scatter_mean(x_s0,pos_s.to(torch.long),dim=0)
#                 edge_index_s, edge_weight_s = datas[k+1].edge_index_s.to(device), datas[k+1].edge_weight_s.to(device)
#                 edge_index_t, edge_weight_t = datas[k+1].edge_index_t.to(device), datas[k+1].edge_weight_t.to(device)
#                 k=1
#                 par_1 = adj2par1(datas[k].edge_index.to(device), x_t0.shape[0], x_s0.shape[0])
#                 D = degree(datas[k].edge_index.view(-1).to(device),num_nodes=x_t0.shape[0]) + 1e-6
             
#         # 2. Readout layer
#         n_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[min(i,1)].num_node1)], dim=-1)
#         n_batch = n_batch.to(device)
#         s_batch = torch.cat( [torch.tensor([i]*nn) for i,nn in enumerate(datas[min(i,1)].num_edge1)], dim=-1)
#         s_batch = s_batch.to(device)
#         x = torch.cat( (global_mean_pool(x_s, s_batch),global_mean_pool(x_t, n_batch)), -1)
        
#         # 3. Apply a final classifier
#         for i, _ in enumerate(self.mlp_channels):
#             fc = getattr(self, 'mlp%d' % i)
#             x = fc(x)

#         return self.out(x) 


# ###############################################################################
# ################################# Train & Test ################################
# ###############################################################################

# def train(loader):
#     model.train()
#     total_loss = 0
#     for data in loader:  # Iterate in batches over the training dataset.
#         y = data[0].y.to(device)
#         out = model(data)
#         loss = criterion(out, y)#.view(-1,1))  # Compute the loss.
        
#         total_loss += loss*data[0].num_graphs
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         optimizer.zero_grad()  # Clear gradients.
#     return total_loss/len(loader.dataset)


# def test(loader):
#       model.eval()
#       total_loss, acc = 0, 0
     
#       for data in loader:  # Iterate in batches over the training/test dataset. 
#           y = data[0].y.to(device)
#           with torch.no_grad():
#             out = model(data) 
#           loss = criterion(out, y)#.view(-1,1))  # Compute the loss.
#           acc += torch.count_nonzero(torch.argmax(out,dim=1) == y)
#           total_loss += loss * data[0].num_graphs

#       return total_loss/len(loader.dataset), acc/len(loader.dataset)


# if __name__ == '__main__':
#     # test degree based spatial pooling
#     if torch.cuda.is_available():
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#     print(device)
#     start = time.time()
#     keig = 10
#     for fold in range(5):
#         torch.manual_seed(fold)
#         print('Fold {} begin'.format(fold))
#         temp = str(args.c1) +str(args.c2)+str(args.c3) +'conv'
#         model = HL_HGCNN_CIFAR10SP_dense_int3_attpool(channels=[args.c1,args.c2,args.c3], filters=args.filters,mlp_channels=[args.mlp_channels], 
#                           K=args.K, dropout_ratio=args.dropout_ratio, dropout_ratio_mlp=0.0, keig=keig).to(device)  # 4conv
#         save_name = 'HGCNN_dense_int3_eigpe'+str(keig)+'_attpool_cifar10SP_'+temp+'_k'+str(args.K)+'_mlp1_FOLD{}'.format(fold)
#         save_path = './weights/' + save_name + '.pt'
#         txt_path = './records/' + save_name + '.txt'
        
#         # model.load_state_dict(torch.load(save_path))
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, 
#                                                                 patience=10, factor=0.5, min_lr=1e-6)
#         # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
#         criterion = torch.nn.CrossEntropyLoss()
#         batch_size = args.batch_size
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='train')
#         trainset = CIFAR10SP_EigPE_MLGC(root=osp.join('CIFAR10_bm','train'),dataset=dataset, keig=keig+1)
        
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='val')
#         validset = CIFAR10SP_EigPE_MLGC(root=osp.join('CIFAR10_bm','val'), dataset=dataset, keig=keig+1)
        
#         dataset = GNNBenchmarkDataset(root='CIFAR10_bm',name='CIFAR10',split='test')
#         testset = CIFAR10SP_EigPE_MLGC(root=osp.join('CIFAR10_bm','test'), dataset=dataset, keig=keig+1)

#         train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
#         valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=4)
#         test_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
        
#         best_loss, best_acc = test(test_loader)
#         print('==================================================================================')
#         print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
#         print('==================================================================================')
#         for epoch in range(1, 600):
#             total_loss = train(train_loader)
                
#             # train_corr, _, _ = test(train_loader)
#             valid_loss, valid_acc = test(valid_loader)
#             scheduler.step(total_loss)
#             if optimizer.param_groups[-1]['lr']<1e-5:
#                 break
    
#             elapsed = (time.time()-start) / 60
#             print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
#             if valid_acc>0.6 and valid_acc>best_acc:
#                 best_acc = valid_acc
#                 torch.save(model.state_dict(), save_path)
#                 print('Model saved! \n')   
#                 best_loss1, best_acc1 = test(test_loader)
#                 print('==================================================================================')
#                 print(f'Test Loss: {best_loss1:.4f}, Test Acc: {best_acc1:.4f}')
#                 print('==================================================================================')
                
#         model.load_state_dict(torch.load(save_path))
#         best_loss, best_acc = test(test_loader)
#         print('==================================================================================')
#         print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
#         print('==================================================================================')
        
#     # export CUDA_VISIBLE_DEVICES=3


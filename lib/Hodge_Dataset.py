#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:07:12 2022

@author: jinghan
"""

import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Dataset, download_url, Data, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_undirected, dense_to_sparse, coalesce,to_scipy_sparse_matrix,subgraph
from scipy.io import loadmat
import torch
import torch.utils.data as tud
import numpy as np
from lib.Hodge_Cheb_Conv import *
from torch_geometric.datasets import GNNBenchmarkDataset, ZINC
from torch_geometric.loader import DataLoader
# from timm.data.mixup import Mixup
from torch_cluster import graclus_cluster
from scipy.linalg import eigh
from lib.LRGBDataset import *
from sklearn.metrics import average_precision_score
from scipy.io import savemat

class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None,
                edge_weight_s=None, edge_weight_t=None, edge_index=None, y=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_weight_s = edge_weight_s
        self.edge_weight_t = edge_weight_t
        self.edge_index = edge_index
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index':
            return self.x_t.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

        
def visualize(loader, model, device='cuda:0'):
    model.eval()
    outs = []
    y_pred = []
    ys = []
     
    for idx, data in enumerate(loader):  # Iterate in batches over the training/test dataset. 
        if not isinstance(data, list):
            data = data.to(device)
            ys.append(data.y.cpu())
        else:
            ys.append(data[0].y.cpu())
        with torch.no_grad():
            out, y = model(data, if_final_layer=True)
            outs.append(out.detach().cpu())
            y_pred.append(y.detach().cpu())
    outs = torch.cat(outs, dim=0)
    ys = torch.cat(ys, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    return outs, ys, y_pred

        
def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)


def eig_pe(L, k=9):
    """
    Parameters
    ----------
    L : Laplacian matrix.
    k : number of eigenvectors. The default is 9.

    Returns:
    -------
    pe : Laplacian position encoding.

    """
    eig_vals, eig_vecs = eigh(L)
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:k])
    return pe


def dropout_edge(edge_index, p: float = 0.5,
                  force_undirected: bool = False,
                  training: bool = True):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                          f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


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


def adj2par1(edge_index, num_node, num_edge):
    """
    Compute the 1-st boundary operator based on the undirected adjacency.
    
    Parameters
    ----------
    edge_index : undirected adjacency.
    num_node : number of nodes.
    num_edge : number of edges (undirected).

    Returns
    -------
    par1_sparse : boundary operator (sparse matrix).

    """
    col_idx = torch.cat([torch.arange(edge_index.shape[1]),torch.arange(edge_index.shape[1])]
                        ,dim=-1).to(edge_index.device)
    row_idx = torch.cat([edge_index[0],edge_index[1]], dim=-1).to(edge_index.device)
    val = torch.cat([edge_index[0].new_full(edge_index[0].shape,-1),
                     edge_index[0].new_full(edge_index[0].shape,1)],dim=-1).to(torch.float)
    par1_sparse = torch.sparse.FloatTensor(torch.cat([row_idx, col_idx], dim=-1).view(2,-1),
                                           val,torch.Size([num_node, num_edge]))
    return par1_sparse


def par2adj(par1):
    """
    Compute the undirected adjacency based on the 1-st boundary operator

    Parameters
    ----------
    par1 : 1-st boundary operator (dense matrix).

    Returns
    -------
    edge_index: undirected adjacency.

    """
    a = par1.to_sparse()
    _, perm = a.indices()[1].sort(dim=-1, descending=False)
    return a.indices()[0][perm].view(-1,2).T


def post2poss(pos_t, edge_index, edge_index1):
    """
    Compute the edge assignment matrix based on the node assignment matrix.
    
    Parameters
    ----------
    pos_t : node clusters.
    edge_index : boundary operator before pooling.
    edge_index1 : boundary operator after pooling.

    Returns
    -------
    pos_s : edge clusters.

    """
    pos_s = torch.zeros(edge_index.shape[1],1)
    idx = pos_t[edge_index[0]]==pos_t[edge_index[1]] # edges within node clusters.
    pos_s[idx] = float('inf') # these edges should be deleted after pooling
    for i in range(edge_index.shape[1]):
        if pos_t[edge_index[0][i]] == pos_t[edge_index[1][i]]:
            pos_s[i] = float('inf')
        else:
            temp1 = min(pos_t[edge_index[0][i]],pos_t[edge_index[1][i]]) == edge_index1[0]
            temp2 = max(pos_t[edge_index[0][i]],pos_t[edge_index[1][i]]) == edge_index1[1]
            temp = torch.logical_and(temp1,temp2)
            pos_s[i] = torch.arange(temp.shape[0])[temp]
    return pos_s


def MLGC(data, keig=1):
    '''
    multi-level graph coarsening (MLGC)
    input: 
       data: input graph
       keig: dim of position encoding
    output:
       data: output graph
       c_node: node assignment matrix
       c_edge: edge assignment matrix
    '''
    c_node = graclus_cluster(data.edge_index_t[0], data.edge_index_t[1],
                    torch.ones_like(data.edge_index_t[1]), data.num_node1)
    c_unique = torch.unique(c_node)
    d = {int(j):i for i,j in enumerate(c_unique)}
    ei1, idx = [[],[]], 0
    ei1_key = {}
    c_edge = torch.zeros(data.x_s.shape[0])
    c_node = [d[int(c)] for c in c_node]
    for i,_ in enumerate(data.edge_index[0]):
        if c_node[data.edge_index[0][i]] == c_node[data.edge_index[1][i]]:
            c_edge[i] = float("inf")
        else:
            imax = max(c_node[data.edge_index[0][i]],(c_node[data.edge_index[1][i]]))
            imin = min(c_node[data.edge_index[0][i]],(c_node[data.edge_index[1][i]]))
            ekey = imax + 0.0001*imin
            if ekey in ei1_key:
                c_edge[i] = ei1_key[ekey]
            else:
                ei1_key[ekey] = idx
                ei1[0].append(imin)
                ei1[1].append(imax)
                idx += 1
                c_edge[i] = idx - 1
    ei1 = torch.tensor(ei1)
    try:
        par1 = adj2par1(ei1, c_unique.shape[0], ei1.shape[1]).to_dense()
    except:
        print(ei1, c_unique, c_node, data)
    L0 = torch.matmul(par1, par1.T)
    lambda0, _ = torch.linalg.eigh(L0)
    maxeig = lambda0.max()
    L0 = 2*torch.matmul(par1, par1.T)/maxeig
    L1 = 2*torch.matmul(par1.T, par1)/maxeig
    node_pe = torch.ones(c_unique.shape[0],1) #eig_pe(L0.numpy(), k=keig)
    edge_pe = torch.ones(ei1.shape[1],1) #eig_pe(L1.numpy(), k=keig)
    eit, ewt = dense_to_sparse(L0)
    eis, ews = dense_to_sparse(L1)
    graph = PairData(x_s=edge_pe, edge_index_s=eis, edge_weight_s=ews,
                     x_t=node_pe, edge_index_t=eit, edge_weight_t=ewt,)
    graph.edge_index = ei1
    graph.num_node1 = c_unique.shape[0]
    graph.num_edge1 = ei1.shape[1]
    graph.num_nodes = c_unique.shape[0]
    return graph, torch.tensor(c_node).view(-1,1), c_edge.view(-1,1)

##############################################################################
def MLGC_weighted(data, keig=1):
    '''
    multi-level graph coarsening (MLGC)
    input: 
        data: input graph
        keig: dim of position encoding
    output:
        data: output graph
        c_node: node assignment matrix
        c_edge: edge assignment matrix
    '''
    edge_index, edge_weight = data.edge_index, torch.exp(-data.x_s[:,0]**2)
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='mean')
    c_node = graclus_cluster(edge_index[0], edge_index[1], edge_weight, data.num_node1)
    c_unique = torch.unique(c_node)
    d = {int(j):i for i,j in enumerate(c_unique)}
    ei1, idx = [[],[]], 0
    ei1_key = {}
    c_edge = torch.zeros(data.x_s.shape[0])
    c_node = [d[int(c)] for c in c_node]
    for i,_ in enumerate(data.edge_index[0]):
        if c_node[data.edge_index[0][i]] == c_node[data.edge_index[1][i]]:
            c_edge[i] = float("inf")
        else:
            imax = max(c_node[data.edge_index[0][i]],(c_node[data.edge_index[1][i]]))
            imin = min(c_node[data.edge_index[0][i]],(c_node[data.edge_index[1][i]]))
            ekey = imax + 0.0001*imin
            if ekey in ei1_key:
                c_edge[i] = ei1_key[ekey]
            else:
                ei1_key[ekey] = idx
                ei1[0].append(imin)
                ei1[1].append(imax)
                idx += 1
                c_edge[i] = idx - 1
    ei1 = torch.tensor(ei1)
    try:
        par1 = adj2par1(ei1, c_unique.shape[0], ei1.shape[1]).to_dense()
    except:
        print(ei1, c_unique, c_node, data)
    L0 = torch.matmul(par1, par1.T)
    lambda0, _ = torch.linalg.eigh(L0)
    maxeig = lambda0.max()
    L0 = 2*torch.matmul(par1, par1.T)/maxeig
    L1 = 2*torch.matmul(par1.T, par1)/maxeig
    node_pe = torch.ones(c_unique.shape[0],1) #eig_pe(L0.numpy(), k=keig)
    edge_pe = torch.ones(ei1.shape[1],1) #eig_pe(L1.numpy(), k=keig)
    eit, ewt = dense_to_sparse(L0)
    eis, ews = dense_to_sparse(L1)
    graph = PairData(x_s=edge_pe, edge_index_s=eis, edge_weight_s=ews,
                      x_t=node_pe, edge_index_t=eit, edge_weight_t=ewt,)
    graph.edge_index = ei1
    graph.num_node1 = c_unique.shape[0]
    graph.num_edge1 = ei1.shape[1]
    graph.num_nodes = c_unique.shape[0]
    return graph, torch.tensor(c_node).view(-1,1), c_edge.view(-1,1)


# def MLGC_weighted(data, keig=1):
#     '''
#     multi-level graph coarsening (MLGC)
#     input: 
#         data: input graph
#         keig: dim of position encoding
#     output:
#         data: output graph
#         c_node: node assignment matrix
#         c_edge: edge assignment matrix
#     '''
#     c_node = graclus_cluster(data.edge_index_t[0], data.edge_index_t[1],
#                     torch.ones_like(data.edge_index_t[1]), data.num_node1)
#     c_unique = torch.unique(c_node)
#     d = {int(j):i for i,j in enumerate(c_unique)}
#     c_edge = torch.zeros(data.x_s.shape[0])
#     c_node = torch.tensor([d[int(c)] for c in c_node])
#     par = adj2par1(data.edge_index, data.x_t.shape[0], data.edge_index.shape[1]).to_dense()
#     par1 = scatter_add(par, c_node, dim=0)
#     Ck = torch.matmul(par1.abs().t(), par1.abs())
#     mask = torch.ones(Ck.shape[0])
#     for i in range(Ck.shape[0]):
#         if Ck[i,i]!=2 or torch.count_nonzero(Ck[:i,i]==2)>0:
#             mask[i] = 0
#     par1 = par1.t()[mask.to(torch.bool)].t()
#     c_edge = torch.matmul(par1.abs().t(), scatter_add(par, c_node, dim=0).abs())
#     c_edge = c_edge==2
#     temp = torch.arange(par1.shape[1]).to(torch.float)+1
#     c_edge = torch.matmul(c_edge.t().to(torch.float), temp.view(-1,1))
#     c_edge[c_edge == 0] = float('inf')
#     c_edge = c_edge - 1
    
#     L0 = torch.matmul(par1, par1.T)
#     lambda0, _ = torch.linalg.eigh(L0)
#     maxeig = lambda0.max()
#     L0 = 2*torch.matmul(par1, par1.T)/maxeig
#     L1 = 2*torch.matmul(par1.T, par1)/maxeig
#     node_pe = torch.ones(par1.shape[0],1) #eig_pe(L0.numpy(), k=keig)
#     edge_pe = torch.ones(par1.shape[1],1) #eig_pe(L1.numpy(), k=keig)
#     eit, ewt = dense_to_sparse(L0)
#     eis, ews = dense_to_sparse(L1)
#     graph = PairData(x_s=edge_pe, edge_index_s=eis, edge_weight_s=ews,
#                       x_t=node_pe, edge_index_t=eit, edge_weight_t=ewt,)
#     graph.edge_index = par2adj(par1)
#     graph.num_node1 = par1.shape[0]
#     graph.num_edge1 = par1.shape[1]
#     graph.num_nodes = par1.shape[0]
#     return graph, c_node.view(-1,1), c_edge.view(-1,1)

###############################################################################
#########################  ZINC  ###########################
###############################################################################
class ZINC_HG_BM_par1_EigPE(Dataset):
    def __init__(self, root, dataset, keig=8, num_pool=2, if_aug=False):
        # data aug
        self.root = root
        self.dataset = dataset
        self.keig = keig
        self.num_pool = num_pool
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['ZINC_BM_alleig_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return len(self.dataset)
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'ZINC_BM_alleig_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
        node_dim, edge_dim = 21+self.keig-1, 3+self.keig-1
        sign = torch.cat([torch.ones(21),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
        if data.x_t.shape[1]<node_dim:
            data.x_t = torch.cat([data.x_t,torch.zeros(data.x_t.shape[0], node_dim-data.x_t.shape[1])], dim=-1)
        else:
            data.x_t = data.x_t[:,:node_dim] * sign
        sign = torch.cat([torch.ones(3),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
        
        if data.x_s.shape[1]<edge_dim:
            data.x_s = torch.cat([data.x_s,torch.zeros(data.x_s.shape[0], edge_dim-data.x_s.shape[1])], dim=-1)
        else:
            data.x_s = data.x_s[:,:edge_dim] * sign
        return data
        
    def process(self):
        i=0
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        for data in loader:

            edge_index,edge_attr = to_undirected(data.edge_index, data.edge_attr,reduce='min')
            idx = edge_index[0]<edge_index[1]

            edge_index,edge_attr = edge_index[:,idx], edge_attr[idx]
            par1 = adj2par1(edge_index, data.x.shape[0], edge_index.shape[1]).to_dense()
            L0 = torch.matmul(par1, par1.T)
            lambda0, _ = torch.linalg.eigh(L0)
            maxeig = lambda0.max()
            L0 = 2*torch.matmul(par1, par1.T)/maxeig
            L1 = 2*torch.matmul(par1.T, par1)/maxeig
            node_pe = eig_pe(L0, k=100)
            edge_pe = eig_pe(L1, k=100)
            x_s = F.one_hot(edge_attr-1,num_classes=3) # the min value of edge_attr=1, one_hot start from value 0
            x_t = F.one_hot(data.x.squeeze(-1),num_classes=21)
            x_s = torch.cat([x_s.to(torch.float),edge_pe], dim=-1)
            x_t = torch.cat([x_t.to(torch.float),node_pe], dim=-1)
            data.y = (data.y - 0.0153)/2.0109
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
            data_zip = {'graph':data, 'maxeig':lambda0.max(), 'par1':par1}
            torch.save(data_zip, osp.join(self.processed_dir, 'ZINC_BM_alleig_'+str(i+1)+'.pt'))
            i += 1

###############################################################################
class ZINC_HG_BM_par1_MLGC(Dataset):
    def __init__(self, root, dataset, keig=8, num_pool=1, if_aug=False):
        # data aug
        self.root = root
        self.dataset = dataset
        self.if_aug = if_aug
        self.keig = keig
        self.node_dim = 21
        self.edge_dim = 3
        self.num_pool = num_pool
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['ZINC_BM_MLGC_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return len(self.dataset)
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'ZINC_BM_MLGC_'+str(idx+1)+'.pt'))
        datas = data_zip['graph']
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
        return datas
        
    def process(self):
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        for idx, _ in enumerate(loader):
            data_zip = torch.load(osp.join(self.processed_dir, 'ZINC_BM_alleig_'+str(idx+1)+'.pt'))
            data = data_zip['graph']
            datas = [data]
            for i in range(self.num_pool):
                temp, c_node, c_edge = MLGC(datas[i])
                datas[i].x_t = torch.cat([c_node, datas[i].x_t], dim=-1)
                datas[i].x_s = torch.cat([c_edge, datas[i].x_s], dim=-1)
                datas.append(temp)
            data_zip = {'graph':datas}
            torch.save(data_zip, osp.join(self.processed_dir, 'ZINC_BM_MLGC_'+str(idx+1)+'.pt'))
            
            
###############################################################################
#########################  Peptides  ###########################
###############################################################################
class Peptides_Func_EigPE(Dataset):
    def __init__(self, root, dataset, keig=8, num_pool=2, if_aug=False):
        # data aug
        self.root = root
        self.dataset = dataset
        self.keig = keig
        self.num_pool = num_pool
        self.node_dim = 9
        self.edge_dim = 3
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['Peptides_Func_alleig_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return len(self.dataset)
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'Peptides_Func_alleig_'+str(idx+1)+'.pt'))
        data = data_zip['graph']
        node_dim, edge_dim = self.node_dim+self.keig-1, self.edge_dim+self.keig-1
        sign = torch.cat([torch.ones(self.node_dim),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
        if data.x_t.shape[1]<node_dim:
            data.x_t = torch.cat([data.x_t,torch.zeros(data.x_t.shape[0], node_dim-data.x_t.shape[1])], dim=-1)
        else:
            data.x_t = data.x_t[:,:node_dim] * sign
        sign = torch.cat([torch.ones(self.edge_dim),-1 + 2 * torch.randint(0, 2, (self.keig-1, ))])
        
        if data.x_s.shape[1]<edge_dim:
            data.x_s = torch.cat([data.x_s,torch.zeros(data.x_s.shape[0], edge_dim-data.x_s.shape[1])], dim=-1)
        else:
            data.x_s = data.x_s[:,:edge_dim] * sign
        return data
        
    def process(self):
        i=0
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        for data in loader:

            edge_index,edge_attr = to_undirected(data.edge_index, data.edge_attr,reduce='min')
            idx = edge_index[0]<edge_index[1]

            edge_index,edge_attr = edge_index[:,idx], edge_attr[idx]
            par1 = adj2par1(edge_index, data.x.shape[0], edge_index.shape[1]).to_dense()
            L0 = torch.matmul(par1, par1.T)
            lambda0, _ = torch.linalg.eigh(L0)
            maxeig = lambda0.max()
            L0 = 2*torch.matmul(par1, par1.T)/maxeig
            L1 = 2*torch.matmul(par1.T, par1)/maxeig
            node_pe = eig_pe(L0, k=100)
            edge_pe = eig_pe(L1, k=100)
            x_s = edge_attr
            x_t = data.x
            x_s = torch.cat([x_s.to(torch.float),edge_pe], dim=-1)
            x_t = torch.cat([x_t.to(torch.float),node_pe], dim=-1)
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
            data_zip = {'graph':data, 'maxeig':lambda0.max(), 'par1':par1}
            torch.save(data_zip, osp.join(self.processed_dir, 'Peptides_Func_alleig_'+str(i+1)+'.pt'))
            i += 1
 
###############################################################################
class Peptides_Func_EigPE_MLGC(Dataset):
    def __init__(self, root, dataset, keig=8, num_pool=1, if_aug=False):
        # data aug
        self.root = root
        self.dataset = dataset
        self.if_aug = if_aug
        self.keig = keig
        self.num_pool = num_pool
        self.node_dim = 9
        self.edge_dim = 3
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['Peptides_Func_alleig_MLGC_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return len(self.dataset)
#         return len(self.processed_file_names)

    def get(self,idx):
        data_zip = torch.load(osp.join(self.processed_dir, 'Peptides_Func_alleig_MLGC_'+str(idx+1)+'.pt'))
        datas = data_zip['graph']
        node_dim, edge_dim = self.node_dim+self.keig-1, self.edge_dim+self.keig-1
        # print(datas[0].x_t.shape, datas[0].x_s.shape)
        for i in range(self.num_pool):
            temp, c_node, c_edge = MLGC(datas[i])
            datas[i].x_t = torch.cat([c_node, datas[i].x_t[:,1:]], dim=-1)
            datas[i].x_s = torch.cat([c_edge, datas[i].x_s[:,1:]], dim=-1)
            datas[i+1] = temp
        
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
        return datas
        
    def process(self):
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
        for idx, _ in enumerate(loader):
            data_zip = torch.load(osp.join(self.processed_dir, 'Peptides_Func_alleig_'+str(idx+1)+'.pt'))
            data = data_zip['graph']
            datas = [data]
            for i in range(self.num_pool):
                temp, c_node, c_edge = MLGC(datas[i])
                datas[i].x_t = torch.cat([c_node, datas[i].x_t], dim=-1)
                datas[i].x_s = torch.cat([c_edge, datas[i].x_s], dim=-1)
                datas.append(temp)
            data_zip = {'graph':datas}
            torch.save(data_zip, osp.join(self.processed_dir, 'Peptides_Func_alleig_MLGC_'+str(idx+1)+'.pt'))
            
            
###############################################################################
#######################  Travelling salesman problem  #########################
###############################################################################
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
#######################  Superpixel dataset  #########################
###############################################################################
class CIFAR10SP_EigPE(Dataset):
    def __init__(self, root, dataset, keig=10, if_aug=False):
        # data aug
        self.root = root
        self.dataset = dataset
        self.keig = keig
        self.node_dim = 5
        self.edge_dim = 4
        self.if_aug = if_aug
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['CIFAR10SP_alleig_'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return len(self.dataset)

    def get(self,idx):
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
        node_pe = eig_pe(L0, k=self.keig)
        edge_pe = torch.abs(node_pe[edge_index[0]]-node_pe[edge_index[1]])
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
        return data
        
    def process(self):
        return None
            
            
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

    def get(self,idx):
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
        node_pe = eig_pe(L0, k=self.keig)
        edge_pe = torch.abs(node_pe[edge_index[0]]-node_pe[edge_index[1]])
        x_s = edge_attr.view(-1,1)
        x_t = data.x
        x_s = torch.cat([x_s.to(torch.float),torch.abs(x_t[edge_index[0]]-x_t[edge_index[1]]),edge_pe], dim=-1)
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
            temp, c_node, c_edge = MLGC_weighted(datas[i])
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
        return datas
        
    def process(self):
        return None
    
###############################################################################
#######################  ABCD dataset  #########################
###############################################################################
class ABCD_EigPE_MLGC(Dataset):
    def __init__(self, root, X_t, X_s, Y, pool_num=2, keig=10):
        # data aug
        self.root = root
        self.X_t = X_t
        self.X_s = X_s
        self.Y = Y
        self.keig = keig
        self.pool_num = pool_num
        self.node_dim = X_t.shape[-1]
        self.edge_dim = 1
        self.size = X_s.shape[0]
        data_zip = torch.load(osp.join(root, 'ABCD_GRAPH_STRUCTURE.pt'))
        self.data = data_zip['graph']
        data_zip = torch.load(osp.join(root, 'ABCD_SUBGRAPH_STRUCTURE0.pt'))
        self.datas = data_zip['graph']
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['hi']#['ABCD'+str(fileidx+1)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return self.size
#         return len(self.processed_file_names)

    def get(self,idx):
        # data = self.data
        datas = self.datas
        
        datas[0].x_s = torch.cat([datas[0].x_s[:,:1],self.X_s[idx].view(-1,1)], dim=-1).to(torch.float)
        datas[0].x_t = torch.cat([datas[0].x_t[:,:1],self.X_t[idx]], dim=-1).to(torch.float)
        datas[0].y = self.Y[idx].to(torch.float)
        
        return datas
        
    def process(self):
        return None
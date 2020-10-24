import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from graph_dataset import Graph_Dataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Softmax
from torch_geometric.nn import radius, TAGConv, global_max_pool as gmp, fps
from train import Net

if __name__ == '__main__':
    dataset_name = 'ModelNet10'
    path = osp.join('dataset', dataset_name)
    train_dataset = Graph_Dataset(path, '10', True)
    test_dataset = Graph_Dataset(path, '10', False)
    print(len(train_dataset))
    print(len(test_dataset))
    print('Dataset loaded.')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True,
                                  num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True,
                                 num_workers=2)

    device = torch.device('cuda')

    model = Net()

    model.to(device)

    it = iter(train_loader)
    data = next(it)
    data = data.to(device)

    r, pos, batch = data.r, data.pos, data.batch
    #idx = fps(pos, batch, ratio=0.5)
    #print(idx[:10])
    #print(idx[4096-10:])
    
    bz = batch[-1]+1
    rep_index = np.array(list(range(128)))
    idx = torch.tensor([rep_index+512*i for i in range(bz)])
    idx = idx.view(-1,1)
    idx = torch.squeeze(idx)
    idx = idx.long()
    idx = idx.to(device)
    r_limit = r[0]*1.5

    #row, col = radius(pos, pos[idx], r_limit, batch, batch[idx], max_num_neighbors=64)
    #edge_index = torch.stack([col, row], dim=0)# (col, row), or (col row)
    #edge_attr = torch.ones((edge_index.shape[1],1))
    y = model(data,idx)
    print(y)
    
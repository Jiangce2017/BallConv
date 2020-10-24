import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from graph_dataset import Graph_Dataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Softmax
from torch_geometric.nn import radius, TAGConv, global_max_pool as gmp, knn
from ballconvnet import BallConv
from point_cloud_models import DynamicEdge

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = TAGConv(4, 128, 3)
        #self.conv2 = TAGConv(128, 128, 3)
        self.conv1 = BallConv(MLP([2*3, 128, 128, 128]), 'max')
        self.conv2 = BallConv(MLP([2*128, 256]), 'max')        
        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            #BN(512),
        )     
        self.lin2 = torch.nn.Sequential(
            BN(128),
            torch.nn.Linear(128,128),
            torch.nn.ReLU(),
            BN(128),
            torch.nn.Dropout(0.5)
            )        
        self.lin3 = torch.nn.Sequential(
            torch.nn.Linear(128,128),
            torch.nn.ReLU(),
            BN(128),
            torch.nn.Dropout(0.5),
        )        
        self.output = torch.nn.Sequential(
            torch.nn.Linear(128, 10)
        )        
        
        #self.condense = torch.nn.Sequential(
            #torch.nn.Linear(512,512),
            #torch.nn.ReLU(),
            #BN(64),
            #torch.nn.Dropout(0.5),
            #torch.nn.Linear(64,1),
            #torch.nn.ReLU(),
        #)
        
    def forward(self, data, idx):
        r, pos, batch = data.r, data.pos, data.batch
        #x, edge_index, batch, edge_attr = data.x.float(), data.edge_index, data.batch, data.edge_attr.float()
        #r_limit = r[0]*0.5
        #row, col = radius(pos, pos[idx], r_limit, batch, batch[idx], max_num_neighbors=64)
        row, col = knn(pos,pos, 32,batch, batch)
        #row, col = radius(pos, pos, r_limit, batch, batch, max_num_neighbors=32)
        edge_index = torch.stack([col, row], dim=0).to(device)# (col, row), or (col row)
        #edge_attr = torch.ones((edge_index.shape[1],1)).to(device)
        x1 = F.relu(self.conv1(pos, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x = self.lin1(x2)
        
        
        
        # x = x.view(-1,512,128)
        # x = torch.transpose(x,1,2)
        # #x = x[:,:,:64]
        # x = x.reshape(-1,512)
        # x = self.condense(x)
        # x = x.view(-1,128,512)
        # x = torch.transpose(x,1,2)
        # x = x.reshape(-1,128)
        
        x = gmp(x, batch)
        x = self.lin2(x)       
        
        x = self.lin3(x)

        x = self.output(x)        
        return F.log_softmax(x, dim=-1)

def train():
    model.train()
    train_metrics = {"loss": [], "acc": []}
    for batch_i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        predictions = model(data)
        loss = F.nll_loss(predictions, data.y)
        loss.backward()
        optimizer.step()
        acc = 100 * (predictions.detach().argmax(1) == data.y).cpu().numpy().mean()
        train_metrics["loss"].append(loss.item())
        train_metrics["acc"].append(acc)
    return np.mean(train_metrics["acc"]), np.mean(train_metrics["loss"])
        
def test():
    model.eval()
    test_metrics = {"acc": []}
    correct = 0
    for batch_i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            predictions = model(data)
        acc = 100 * (predictions.detach().argmax(1) == data.y).cpu().numpy().mean()
        test_metrics["acc"].append(acc)
    return np.mean(test_metrics["acc"])
    

if __name__ == '__main__':
    sphere_num = 1024
    dataset_name = 'ModelNet10_256'
    path = osp.join('dataset', dataset_name)
    train_dataset = Graph_Dataset(path, '10', True)
    test_dataset = Graph_Dataset(path, '10', False)

    print(len(train_dataset))
    print(len(test_dataset))
    print('Dataset loaded.')
    
    bz = 8
    train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True, drop_last=True,
                                  num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=True, drop_last=True,
                                 num_workers=2)

    

    model = DynamicEdge(10)
    model_name = 'Net'
    device = torch.device('cuda:0')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    #model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    exp_name = dataset_name+model_name
    num_epochs = 400
    print(exp_name)
    result_path = osp.join('.', 'results')

    best_acc = 0

    for epoch in range(num_epochs): 
        train_acc, train_loss = train()
        test_acc= test()
        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }
        if is_best:
            torch.save(state, '%s/%s_checkpoint.pth' % (result_path, exp_name))

        print(exp_name)
        log = 'Epoch: {:03d}, Train_Loss: {:.4f}, Train_Acc: {:.4f}, Test_Acc: {:.4f}'
        print(log.format(epoch, train_loss, train_acc, test_acc))


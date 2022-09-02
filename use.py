#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sys
root = str(sys.argv[1])   #path to the test_samples directory
features_dir= str(sys.argv[2]) #path to the test samples bps_features file *.h5
PDB_codes=str(sys.argv[3])   #path to a dataframe.csv containing column 'PDB' : PDB codes of the test samples


# In[1]:


import torch
import torch_geometric
import deepdish as dd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import copy
from os.path import join
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from rdkit import RDConfig
from rdkit.Chem.rdmolfiles import MolFromMolFile
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import random
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, Dropout
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, Set2Set, GCNConv, global_add_pool, global_mean_pool,GATConv,GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset #easily fits into cpu memory


# In[2]:


def add_edges_list(root, pdb_code):
    ligand_filename = pdb_code + ".sdf"   #note: you need to delete hydrogen atoms from the ligand file
#     print(join(root, code, ligand_filename))
    m = MolFromMolFile(join(root, pdb_code, ligand_filename))
    atoms1 = [b.GetBeginAtomIdx() for b in m.GetBonds()]
    atoms2 = [b.GetEndAtomIdx() for b in m.GetBonds()]    
    # Edge attributes: distance; SINGLE; DOUBLE; TRIPLE; AROMATIC.
    edge_weights= []
    coords = m.GetConformers()[0].GetPositions()  # Get a const reference to the vector of atom positions
    for b in m.GetBonds():
        if str(b.GetBondType()) == "SINGLE":
            edge_weights.append(1)
        elif str(b.GetBondType()) == "DOUBLE":
            edge_weights.append(2)
        elif str(b.GetBondType()) == "TRIPLE":
            edge_weights.append(3)
        else:
            edge_weights.append(4)
    edge_features = np.array(edge_weights) 
    # since the torch-geometric graphs are directed add reverse direction of edges
    return np.array([atoms1 + atoms2, atoms2 + atoms1]), np.concatenate((edge_features, edge_features), 0)


# In[3]:


class PDBbindDataset(InMemoryDataset):
    def __init__(self, root, node_features,transform=None, pre_transform=None):
        self.root = root
        self.node_features = node_features
        self.activity_csv = activity_csv
        super(PDBbindDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):

        self.node_data = dd.io.load(join(self.root, self.node_features))
#         print(self.node_data.keys())
        # load csv with activity data and simlarity scores
        self.activity = pd.read_csv(join(self.root, self.activity_csv))
        # create lists of edges and edge descriptors 
        self.edge_indexes = {key: add_edges_list(self.root, key)[0] for key in self.activity.PDB }
        self.edge_data = {key: add_edges_list(self.root, key)[1] for key in self.activity.PDB }
        
        # Read data into huge `Data` list.
        data_list = [Data(x = torch.FloatTensor(self.node_data[key]),
                          edge_index = torch.LongTensor(self.edge_indexes[key]),
                          edge_attr = torch.FloatTensor(self.edge_data[key]),
                          ) for key in self.activity.PDB ]
      
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# In[10]:


test_samples = PDBbindDataset(root, 
                         ,features_dir,
                         PDB_codes)
test_loader = DataLoader(test_samples, batch_size=1, shuffle=False)


# In[7]:


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #GCN-representation
        self.conv1 = GCNConv(373, 256, cached=False )
        self.bn01 = BatchNorm1d(256)
        self.conv2 = GCNConv(256, 128, cached=False )
        self.bn02 = BatchNorm1d(128)
        self.conv3 = GCNConv(128, 128, cached=False)
        self.bn03 = BatchNorm1d(128)
        #GAT-representation
        self.gat1 = GATConv(373, 256,heads=3)
        self.bn11 = BatchNorm1d(256*3)
        self.gat2 = GATConv(256*3, 128,heads=3)
        self.bn12 = BatchNorm1d(128*3)
        self.gat3 = GATConv(128*3, 128,heads=3)
        self.bn13 = BatchNorm1d(128*3)
        #GIN-representation
        fc_gin1=Sequential(Linear(373, 256), ReLU(), Linear(256, 256))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(256)
        fc_gin2=Sequential(Linear(256, 128), ReLU(), Linear(128, 128))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(128)
        fc_gin3=Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(64)
        #Fully connected layers for concatinating outputs
        self.fc1=Linear(128*4 + 64, 256)
        self.fc2=Linear(256, 64)
        self.fc3=Linear(64, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        y=x
        z=x
        #GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_add_pool(x, data.batch)
        #GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_add_pool(y, data.batch)
        #GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_add_pool(z, data.batch)
        #Concatinating_representations
        cr=torch.cat((x,y,z),1)
        cr = F.relu(self.fc1(cr))
        cr = F.dropout(cr, p=0.2, training=self.training)
        cr = F.relu(self.fc2(cr))
        cr = F.dropout(cr, p=0.2, training=self.training)
        cr = self.fc3(cr)
        cr = F.relu(cr).view(-1)
        return cr  


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@torch.no_grad()
def test_predictions(model, loader):
    model.eval()
    pred = []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
    return pred, true


# In[13]:


model=Net()
model.load_state_dict(torch.load('path_to_the_downloaded_model_weights.pt'))
preds = test_predictions(model,test_loader)
print(preds)


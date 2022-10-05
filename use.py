#!/usr/bin/env python
# coding: utf-8

# Make sure all the following libraries are installed:
import sys
import deepdish as dd
from os.path import join
from rdkit.Chem.rdmolfiles import MolFromMolFile
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.nn import  GCNConv, global_add_pool,GATConv,GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

root = str(sys.argv[1])        #path to the test_samples directory
features_dir= str(sys.argv[2]) #path to the test samples bps_features file *.h5 
PDB_codes=str(sys.argv[3])     #path to a dataframe.csv containing column 'PDB' : PDB codes of the test samples,
model_path=str(sys.argv[4])    #path_to_the_downloaded_model_weights *.pt

def add_edges_list(root, pdb_code):
    '''
    This function builds edge index list and edge attributes given the directory of ligand.sdf 
    Input: 
    - root: path to the directory containing ligand samples
    - pdb_code: A list of PDB codes of complexes that contain these lignads
    Output:
    - edge list: numpy array of shape (2, num_edges) 
    '''
    ligand_filename = pdb_code + ".sdf"   #note: you need to delete hydrogen atoms from the ligand file
    m = MolFromMolFile(join(root, pdb_code, ligand_filename))
    atoms1 = [b.GetBeginAtomIdx() for b in m.GetBonds()]
    atoms2 = [b.GetEndAtomIdx() for b in m.GetBonds()]    
    # since the torch-geometric graphs are directed add reverse direction of edges
    edge_list = np.array([atoms1 + atoms2, atoms2 + atoms1])
    return edge_list

class PDBbindDataset(InMemoryDataset):
    '''
    Dataset class that will transform test samples to a test dataset containing all samples 
    '''
    def __init__(self, root, node_features,activity):
        '''
        root: path to data
        node_features: path to *.h5 file that contains the descriptors computed in the previous step 
        avtivity: Path to *.csv file that contains the PDB codes of test complexes
        
        '''
        self.root = root
        self.node_features = node_features
        self.activity_csv = activity_csv
        super(PDBbindDataset, self).__init__(root)
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



class MegaDTA(torch.nn.Module):
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
        self.dropout1=Dropout(p=0.2,)
        self.fc2=Linear(256, 64)
        self.dropout2=Dropout(p=0.2,)
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
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        cr = F.relu(cr).view(-1)
        return cr  

test_samples = PDBbindDataset(root, 
                         ,features_dir,
                         PDB_codes)
test_loader = DataLoader(test_samples, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def test_predictions(model, loader):
    '''
    This function returns the predicted affinity of the test samples contained in the test data loader;
    Input:
    - model: MegaDTA model
    -loader: Test loader
    '''
    model.eval()
    pred = []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
    return pred

model= MegaDTA()
model.load_state_dict(torch.load(model_path))
preds = test_predictions(model,test_loader)
print(preds)

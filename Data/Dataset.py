import numpy as np
from os.path import join
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from rdkit import RDConfig
from rdkit.Chem.rdmolfiles import MolFromMolFile
from Edge_builder import *

class PDBbindDataset(Dataset):
'''
The Dataset class. Arguments:
1. root: path to root directory 
2. node_features: name of precomputed BPS features saved in dd format for compression
3. activity_csv: name of csv file containing the target variable for refined set

returns data list: - First list contains the graph with node features and edges features
'''
class PDBbindDataset(InMemoryDataset):
    def __init__(self, root, node_features,activity_csv,transform=None, pre_transform=None):
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
        # load csv with activity data and simlarity scores
        # create lists of edges and edge descriptors 
        self.edge_indexes = {key: add_edges_list(self.root, key)[0] for key in self.activity.PDB }
        self.edge_data = {key: add_edges_list(self.root, key)[1] for key in self.activity.PDB }
        
        # Read data into huge `Data` list.
        data_list = [Data(x = torch.FloatTensor(self.node_data[key]),
                          edge_index = torch.LongTensor(self.edge_indexes[key]),
                          edge_attr = torch.FloatTensor(self.edge_data[key]),
                          y = torch.FloatTensor([self.activity[self.activity.PDB == key].pk.iloc[0]])) for key in self.activity.PDB ]
      
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

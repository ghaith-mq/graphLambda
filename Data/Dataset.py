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
4. sequence_csv: path to the csv file containg the amino acid (AA) sequences in the binding site for each sample

returns two data lists: - First list contains the graph with node features and edges features
			 - Second list contains the Amino acid sequence embedding to accompany each graph 
'''
	def __init__(self, root, node_features,activity_csv,sequence_csv):
		self.root = root
		self.node_features = node_features
		self.activity_csv = activity_csv
		self.node_data = dd.io.load(join(self.root, self.node_features))
		# load csv with activity data and simlarity scores
		self.activity = pd.read_csv(join(self.root, self.activity_csv))
		# create lists of edges and edge descriptors 
		self.edge_indexes = {key: add_edges_list(self.root, key)[0] for key in self.activity.PDB }
		self.edge_data = {key: add_edges_list(self.root, key)[1] for key in self.activity.PDB }

		self.seq=pd.read_csv(sequence_csv)
		dic={ 'C': 1 , 'D': 2,'S': 3,'Q': 4,'K': 5,'I': 6,'P': 7,'T': 8,'F': 9,'N': 10,
		      'G': 11,'H': 12,'L': 13,'R': 14,'W': 15,'A': 16,'V': 17,'E': 18,'Y': 19,'M': 20}
		embedding=dict()
		AA_embedding=torch.nn.Embedding(20,8)
		AA_encoded=[]
        	for seq in self.seq['seq']:
			s_enc=[0]*20
			for i in range(len(seq)):
				s_enc[dic[seq[i]]-1]+=1
		AA_encoded.append(AA_embedding(torch.tensor(s_enc,dtype=torch.int)))
		for i,key in enumerate(self.seq.PDB):
			embedding[key]=AA_encoded[i]
        	# Read graph and node features data into huge `Data` list.
		data_list = [Data(x = torch.FloatTensor(self.node_data[key]),
				  edge_index = torch.LongTensor(self.edge_indexes[key]),
				  edge_attr = torch.FloatTensor(self.edge_data[key]),
				  y = torch.FloatTensor([self.activity[self.activity.PDB == key].pk.iloc[0]])) for key in self.activity.PDB ]
	# data_list2 : Contains graph_level features : For each graph AA sequence embedding      
        data_list2=[embedding[key].t() for key in self.activity.PDB ]
        self.data1=data_list
        self.data2=data_list2
	def __getitem__ (self,index):
		index=int(index)
		data=self.data1[index]
		seq_embed=self.data2[index]
		return data,seq_embed,index
	def __len__ (self,):
		return len(self.data1)

import numpy as np
from os.path import join
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from rdkit import RDConfig
from rdkit.Chem.rdmolfiles import MolFromMolFile

def add_edges_list(root, key):     
''' 
This function builds the edges between the nodes in the graph. 
Input : 1. Path to ligands directory
	2. PDB key for the lignad: *.sdf

'''
	ligand_filename = code + ".sdf"
	m = MolFromMolFile(join(root, code, ligand_filename))
	atoms1 = [b.GetBeginAtomIdx() for b in m.GetBonds()]
	atoms2 = [b.GetEndAtomIdx() for b in m.GetBonds()]    
	# Edge attributes: SINGLE; DOUBLE; TRIPLE; AROMATIC.
	edge_weights= []
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

#Compute ligand-ligand fingerprints similarity using RDKit
from rdkit import Chem
import os
from rdkit import DataStructs


dirs=os.listdir()
dirs.sort()
i=0
res=[]
keys=[]
ligands=[]
for dir_ in dirs:
	files=os.listdir(dir_)
	for file_ in files:
		if (file_[-5:]=="h.sdf"):
		ligand1=Chem.MolFromMolFile(dir_ + "/" + file_,removeHs=False)
		if(ligand1 is None):
			print("error in: ",dir_)
		fps1 =Chem.RDKFingerprint(ligand1)
		for dir2_ in dirs:
				files2=os.listdir(dir2_)
				for file2_ in files2:
					if (file2_[-5:]=="h.sdf"):
						ligand2=(Chem.MolFromMolFile(dir2_ + "/" + file2_,removeHs=False))
						keys.append((dir_,dir2_))
						fps2=Chem.RDKFingerprint(ligand2)
					      	res.append(DataStructs.FingerprintSimilarity(fps1,fps2))
						print(i)
						i=i+1
print(keys)
print(res)

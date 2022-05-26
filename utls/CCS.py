#Compute protein-lignad interaction in complex1 with protein-ligand interaction in complex2 using Open Drug Discovery Toolkit (ODDT)

import os
import oddt.fingerprints
from rdkit import Chem
import time
from multiprocessing import Pool
import pandas as pd
import numpy as np

lst=[]
for dir1 in os.listdir():
	if(len(dir1)==4):
		for dir2 in os.listdir():
			if(len(dir2)==4):
				lst.append((dir1,dir2))

def compute(dirs):
	protein1= next(oddt.toolkit.readfile('pdb', dirs[0] + '/' + dirs[0] + '_pocket.pdb'))
	protein2= next(oddt.toolkit.readfile('pdb', dirs[1] + '/' + dirs[1] + '_pocket.pdb'))
	if((protein1 is None) or (protein2 is None)):
		print(dirs[0], end='')
		print(',',end='')
		print(dirs[1], end='')
		print(',',end='')
		print("Nan")
	else:
		protein1.protein = True
		protein2.protein = True
		ligand1 = next(oddt.toolkit.readfile('sdf', dirs[0] + '/' + dirs[0] + '_h.sdf'))
		ligand2 = next(oddt.toolkit.readfile('sdf', dirs[1] + '/' + dirs[1] + '_h.sdf'))
		fps1=oddt.fingerprints.SimpleInteractionFingerprint(ligand1, protein1)
		fps2=oddt.fingerprints.SimpleInteractionFingerprint(ligand2, protein2)
		sim=oddt.fingerprints.tanimoto(fps1, fps2)
		print(dirs[0], end='')
		print(',',end='')
		print(dirs[1], end='')
		print(',',end='')
		print(sim)
		
def main(): 
	p = Pool(8) 
	codes = lst
	p.map(compute, codes)	
	print("computation is finished")	
	
if __name__ == "__main__":   
    main()			

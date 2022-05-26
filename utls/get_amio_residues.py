#Extract amino-acid residues that exist in the binding site


import os
from Bio.PDB import PDBParser, PDBIO


d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
dirs=os.listdir()
dirs.sort()
aminos=['CYS','ASP','SER','GLN','LYS','ILE','PRO','THR','PHE','ASN','GLY','HIS','LEU','ARG','TRP',
		'ALA','VAL','GLU','TYR','MET']
i=0
for dir_ in dirs:
	if (len(dir_)==4):
		files=os.listdir(dir_)
		for file_ in files:
			if (file_[-10:]=="pocket.pdb"):
				parser=PDBParser(QUIET=True)
				pdb = parser.get_structure(file_[:4], dir_+ "/"+ file_)
				seq=[]
				for model in pdb:
					for chain in model:
						for residue in chain:
							if (residue.resname not in aminos):
								continue
							seq.append(d3to1[residue.resname])
				seq=''.join(seq)
				with open(dir_ + "/" + dir_ +'_pocket_seq.txt', 'w') as f:
					f.write("%s\n" % seq)
		i+=1
		print(dir_)
		print(i)
		#				print('>some_header\n',''.join(seq))	 to seperate chains






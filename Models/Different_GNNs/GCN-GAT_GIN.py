import os
from rdkit import Chem

noisy= ["remove_Hsdf.py","remove_water.py" , "index" , "readme", "script.py" , "rdkit_codepdb2xyz.py","rdkit_code.py" ,"remove_H.py","copy.py"]
i=0
j=0
files=os.listdir()
files.sort()
for file_ in files:
	if(file_ in noisy):
		continue
	if (file_[13:19]=="ligand"):
		if(len(file_)==24):
			command= """ sed '/H$/d' """ + file_ + " > " + "L" + file_[19:20] + "_h.pdb" 
		if(len(file_)==25):
			command= """ sed '/H$/d' """ + file_ + " > " + "L" + file_[19:21] + "_h.pdb" 
	if (file_[13:21]=="receptor"):
		if(len(file_)==26):
			command= """ sed '/H$/d' """ + file_ + " > " + "R" + file_[21:22] + "_h.pdb" 
		if(len(file_)==27):
			command= """ sed '/H$/d' """ + file_ + " > " + "R" + file_[21:23] + "_h.pdb" 
	os.system(command)
	




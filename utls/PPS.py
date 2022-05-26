#Compute protein-protein structural similarity using TM-align
from rdkit import Chem
import os
from multiprocessing import Pool
i=0
uk=0
visited= []
for dir1 in os.listdir():
	if(len(dir1)==4):
		for file_ in os.listdir(dir1):
			if(file_[-4:]==".txt") :
				visited.append(dir1)

lst=[]
for dir1 in os.listdir():
	if(len(dir1)==4):
		if(dir1 in visited):
			continue
		for dir2 in os.listdir():
			if(len(dir2)==4):
				lst.append((dir1,dir2))
		
print(len(lst))

def compute(dirs):
	print(dirs)
	command = "~/Downloads/TMalign ./" + dirs[0] + "/" + dirs[0] +"_protein_nowater.pdb" + " ./" + dirs[1] + "/" + dirs[1] +"_protein_nowater.pdb" + " -outfmt 2 -ter 0 -split 0 -fast  >> " + dirs[0] + "/" + dirs[0] + ".txt"
	os.system(command)

def main(): 

	p = Pool(8) 
	codes = lst
	p.map(compute, codes)	
	print("computation is finished")	
	
if __name__ == "__main__":   
    main()				

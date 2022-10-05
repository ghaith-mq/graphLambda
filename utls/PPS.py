#Compute protein-protein structural similarity using TM-align (TM-align installation is required)
from rdkit import Chem
import os
from multiprocessing import Pool
lst = [] # A list that contains all possible pairwise combinations of PBD file names

def compute(dirs):
	command = "path_to_where_TM_align_is /TMalign " + dirs[0] + ".pdb " +  dirs[1]  +".pdb" + " -outfmt 2 -ter 0 -split 0 -fast  >> " + dirs[0] + "/" + dirs[0] + ".txt"
	os.system(command)

def main(): 

	p = Pool(8) 
	codes = lst
	p.map(compute, codes)	
	print("computation is finished")	
	
if __name__ == "__main__":   
    main()				

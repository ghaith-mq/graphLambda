import os
dirs=os.listdir()
dirs.sort()

for dir_ in dirs:
	files=os.listdir(dir_)
	for file_ in files:
		if (file_[-5:]=="n.pdb"):
			command= "grep -vwE" + " \" HOH \" " + dir_ + "/" + file_ + " > " + dir_ + "/" + dir_ + "_nowater.pdb" 
			os.system(command)




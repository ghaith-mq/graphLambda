import random
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def educated_split(dist_thre,similarity_score_path=None):
'''
This function produces train/validation sets for K-folds (K training sets, K validation sets)  for a certain similarity type with a certain thresold.
Input params: 1- Dist_thre: Distance metric threshold used in the clustering algorithm.
              2- similarity_score_path : Path to the csv_file containing the pair-wise similarities in the data. The similarity score column should be titled 'similarity'.
              
Output:
This function returns train/validation sets resulting from the clustering and randomly distribution of the clusters
'''
	train_datasets=[]
	val_datasets=[]
	data_size=5316   #number of samples in the PDBbind refined set
	folds =5
	if similarity_score_path == None:
		print("Please provide a valid file path with csv file format containing similarity scores")
		return
	else:
		df= pd.read_csv(similarity_score_path)
		similarity_matrix = np.array(df['similarity'])
		similarity_matrix=np.reshape(similarity_matrix,(data_size,data_size))
		cl = AgglomerativeClustering(n_clusters=None,compute_full_tree=True,linkage='average',\
		                                distance_threshold=dist_thre,affinity='precomputed' ).fit(1-similarity_matrix)

		cl_labels=np.array(cl.labels_)
		dic=dict()       #This dictionary would store each cluster's samples indices
		for i in range(len(np.unique(cl_labels))):
			dic[i]=np.where(cl_labels==i)
		########## divide training set into 5 parts for later perfomeing cross validation
		visited_clusters=[]
		train_datasets=[]
		val_datasets=[]
		cl_used_in_previous_folds=[]
		for i in range(folds):     #Generating train/val sets for each fold and appending them to train_datasets, val_datasets lists
			print('fold' ,i)
			visited_clusters=[]
			size=0
			train_ids=[]
			train_keys=[]
			val_ids=[]
			val_keys=[]
			if i ==folds-1:
				for id_ in np.unique(cl_labels):
					if id_ not in cl_used_in_previous_folds:
						val_ids.append(dic[id_][0])
					else:
						train_ids.append(dic[id_][0])
			else:
				while(size<(data_size/folds)):
					x = random.choice(np.unique(cl_labels))
					if((x not in visited_clusters) and (x not in cl_used_in_previous_folds)):
						visited_clusters.append(x)
						cl_used_in_previous_folds.append(x)
						val_ids.append(dic[x][0])
						size+=len(dic[x][0])
				for id_ in np.unique(cl_labels):
					if id_ not in visited_clusters:
						train_ids.append(dic[id_][0])
			train_ids=[item for lis in train_ids for item in lis]
			val_ids=[item for lis in val_ids for item in lis]
			ntrain_ids=[]
			nval_ids=[]
			for i in range(len(train_ids)):
				train_keys.append(df.loc[train_ids[i]].PDB2)
			for i in range(len(val_ids)):
				val_keys.append(df.loc[val_ids[i]].PDB2)
			for key in tqdm(train_keys):
				ntrain_ids.append((df_ind.index[df_ind.PDB==key]).tolist())
			for key in tqdm(val_keys):        
				nval_ids.append((df_ind.index[df_ind.PDB==key]).tolist())
			train_ids = [item for sublist in ntrain_ids for item in sublist]
			val_ids = [item for sublist2 in nval_ids for item in sublist2]
			#split dataset to train and validation sets according to similarity score
			train_datasets.append([dataset[id_] for id_ in train_ids])
			val_datasets.append([dataset[id_]for id_ in val_ids])
	return train_datasets,val_datasets

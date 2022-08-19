#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
This script provides a direct example how to train the graph model.
'''


# In[5]:


import deepdish as dd
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import copy
from os.path import join
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from rdkit import RDConfig
from rdkit.Chem.rdmolfiles import MolFromMolFile
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import random
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, Dropout
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, Set2Set, GCNConv, global_add_pool, global_mean_pool,GATConv,GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset #easily fits into cpu memory


# In[8]:


def add_edges_list(root, code):
    ligand_filename = code + "_h.sdf"
    m = MolFromMolFile(join(root, code, ligand_filename))
    atoms1 = [b.GetBeginAtomIdx() for b in m.GetBonds()]
    atoms2 = [b.GetEndAtomIdx() for b in m.GetBonds()]    
    # Edge attributes: distance; SINGLE; DOUBLE; TRIPLE; AROMATIC.
    edge_weights= []
    coords = m.GetConformers()[0].GetPositions()  # Get a const reference to the vector of atom positions
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


# In[9]:


class PDBbindDataset(InMemoryDataset):
    def __init__(self, root, node_features,activity_csv,transform=None, pre_transform=None):
        self.root = root
        self.node_features = node_features
        self.activity_csv = activity_csv
        super(PDBbindDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):

        self.node_data = dd.io.load(join(self.root, self.node_features))
        # load csv with activity data and simlarity scores
        # create lists of edges and edge descriptors 
        self.edge_indexes = {key: add_edges_list(self.root, key)[0] for key in self.activity.PDB }
        self.edge_data = {key: add_edges_list(self.root, key)[1] for key in self.activity.PDB }
        
        # Read data into huge `Data` list.
        data_list = [Data(x = torch.FloatTensor(self.node_data[key]),
                          edge_index = torch.LongTensor(self.edge_indexes[key]),
                          edge_attr = torch.FloatTensor(self.edge_data[key]),
                          y = torch.FloatTensor([self.activity[self.activity.PDB == key].pk.iloc[0]])) for key in self.activity.PDB ]
      
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# In[13]:


from numpy import random

def educated_split(dist_thre,score_file_path=None,dbscan=False):
    train_datasets=[]
    val_datasets=[]
    data_size=5314
    if score_file_path != None:
        df= pd.read_csv(score_file_path)
        if score_file_path=='TMscore.csv' :
            similarity_matrix=np.array(df['TM'])
        elif score_file_path =='nligands_fps.csv':
            similarity_matrix=np.array(df['similarity'])
        elif score_file_path == 'ninteractions_fps.csv':
            similarity_matrix = np.array(df['similarity'])
        similarity_matrix=np.reshape(similarity_matrix,(5314,5314))
        if not dbscan:
            cl = AgglomerativeClustering(n_clusters=None,compute_full_tree=True,linkage='average',                                        distance_threshold=dist_thre,affinity='precomputed' ).fit(1-similarity_matrix)
        else:
            cl = DBSCAN(eps=dist_thre,min_samples=2 ,metric='precomputed').fit(1-similarity_matrix)

        cl_labels=np.array(cl.labels_)
        dic=dict()
        for i in range(len(np.unique(cl_labels))):
            dic[i]=np.where(cl_labels==i)
        visited_clusters=[]
        ########## divide training set into 5 parts for later perfomeing cross validation
        train_datasets=[]
        val_datasets=[]
        cl_used_in_previous_folds=[]
        for i in range(5):
            print('fold' ,i)
            visited_clusters=[]
            size=0
            train_ids=[]
            train_keys=[]
            val_ids=[]
            val_keys=[]
            if i ==4:
                for id_ in np.unique(cl_labels):
                    if id_ not in cl_used_in_previous_folds:
                        val_ids.append(dic[id_][0])
                    else:
                        train_ids.append(dic[id_][0])
            else:
                while(size<(data_size/5)):
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
            #split dataset to train and validation sets according to TM-score
            train_datasets.append(dataset[train_ids])
            val_datasets.append(dataset[val_ids])
    else:
        train_datasets=[]
        val_datasets=[]
        sample_used_in_previous_folds=[]
        for i in range(5):
            print('fold' ,i)
            visited_sample=[]
            size=0
            train_ids=[]
            val_ids=[]
            ids=list(range(data_size))
            if i ==4:
                for id_ in list(range(data_size)):
                    if(id_ not in sample_used_in_previous_folds):
                        val_ids.append(id_)
                for id_ in list(range(data_size)):
                    if id_ not in val_ids:
                        train_ids.append(id_)
            else:
                while(size<(data_size/5)):
                    x = random.choice(ids)
                    if((x not in visited_sample) and (x not in sample_used_in_previous_folds)):
                        visited_sample.append(x)
                        sample_used_in_previous_folds.append(x)
                        val_ids.append(x)
                        size+=1
                for id_ in list(range(data_size)):
                    if id_ not in val_ids:
                        train_ids.append(id_)
            train_datasets.append(dataset[train_ids])
            val_datasets.append(dataset[val_ids])
    return train_datasets,val_datasets


# In[14]:


# train_sets,val_sets=educated_split(0.5,'TMscore.csv')


# In[15]:


torch_geometric.seed_everything(20)
#create train and test set instances
dataset = PDBbindDataset("refined-set1", 
                         "refined_set.h5",
                         "index/refined_data2020.csv")

df_ind=pd.read_csv("refined-set1/index/nrefined_data2020.csv")
# df_ind.reset_index(drop=True, inplace=True)


# In[10]:


coreset = PDBbindDataset("coreset2016", 
                         "coreset.h5",
                         "coreset2016.csv")
test_loader = DataLoader(coreset, batch_size=64, shuffle=False)


# In[6]:


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #GCN-representation
        self.conv1 = GCNConv(373, 256, cached=False )
        self.bn01 = BatchNorm1d(256)
        self.conv2 = GCNConv(256, 128, cached=False )
        self.bn02 = BatchNorm1d(128)
        self.conv3 = GCNConv(128, 128, cached=False)
        self.bn03 = BatchNorm1d(128)
        #GAT-representation
        self.gat1 = GATConv(373, 256,heads=3)
        self.bn11 = BatchNorm1d(256*3)
        self.gat2 = GATConv(256*3, 128,heads=3)
        self.bn12 = BatchNorm1d(128*3)
        self.gat3 = GATConv(128*3, 128,heads=3)
        self.bn13 = BatchNorm1d(128*3)
        #GIN-representation
        fc_gin1=Sequential(Linear(373, 256), ReLU(), Linear(256, 256))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(256)
        fc_gin2=Sequential(Linear(256, 128), ReLU(), Linear(128, 128))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(128)
        fc_gin3=Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(64)
        #Fully connected layers for concatinating outputs
        self.fc1=Linear(128*4 + 64, 256)
        self.fc2=Linear(256, 64)
        self.fc3=Linear(64, 1)
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        y=x
        z=x
        #GCN-representation
        x = F.relu(self.conv1(x, edge_index,edge_weight))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index,edge_weight))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index,edge_weight))
        x = self.bn03(x)
        x = global_add_pool(x, data.batch)
        #GAT-representation
        y = F.relu(self.gat1(y, edge_index,edge_weight))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index,edge_weight))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index,edge_weight))
        y = self.bn13(y)
        y = global_add_pool(y, data.batch)
        #GIN-representation
        z = F.relu(self.gin1(z, edge_index,edge_weight))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index,edge_weight))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index,edge_weight))
        z = self.bn23(z)
        z = global_add_pool(z, data.batch)
        #Concatinating_representations
        cr=torch.cat((x,y,z),1)
        cr = F.relu(self.fc1(cr))
        cr = F.dropout(cr, p=0.2, training=self.training)
        cr = F.relu(self.fc2(cr))
        cr = F.dropout(cr, p=0.2, training=self.training)
        cr = self.fc3(cr)
        cr = F.relu(cr).view(-1)
        return cr  


# In[11]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(model, train_loader,epoch,device,optimizer,scheduler):
    model.train()
    loss_all = 0
    error = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)  #MSE loss
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        error += (model(data) - data.y).abs().sum().item()  # MAE
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
    return loss_all / len(train_loader.dataset), error / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader,device):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) - data.y).abs().sum().item()  # MAE
    return error / len(loader.dataset)


@torch.no_grad()
def test_predictions(model, loader):
    model.eval()
    pred = []
    true = []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
        true += data.y.detach().cpu().numpy().tolist()
    return pred, true


# In[33]:


def kfold(train_sets,val_sets=None,test_set=None):

    ####################################################
    best_val_error_f=[]
    best_model_f=[]
    all_train_errors=[]
    all_validation_errors=[]
    all_test_errors=[]
    exp_ind=[]
    for fold in range(5):
        print(f'FOLD {fold}')
        print('--------------------------------')
#         train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#         val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = DataLoader(
                         train_sets[fold], 
                          batch_size=64, shuffle = True,drop_last=True)

        val_loader = DataLoader(
                          val_sets[fold],
                          batch_size=64, shuffle = True , drop_last = True)
        best_val_error = None
        best_model = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_errors, valid_errors,test_errors = [], [],[]
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.95, patience=10,
                                                           min_lr=0.00001)
        for epoch in range(1, 501):
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss, train_error = train(model, train_loader,epoch,device,optimizer,scheduler)
            val_error = test(model, val_loader,device)
            train_errors.append(train_error)
            valid_errors.append(val_error)
            test_er=None
            if(test_set != None):
                test_er = test(model, test_loader,device)
                test_errors.append(test_er)
            scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                best_val_error = val_error
                best_model = copy.deepcopy(model)

            print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}'
                  .format(epoch, lr, loss, val_error))
        print('leng of test errors = ', len(test_errors))
        all_train_errors.append(train_errors)
        all_validation_errors.append(valid_errors)
        all_test_errors.append(test_errors)
        best_model_f.append(best_model)
        best_val_error_f.append(best_val_error)
    return all_train_errors,all_validation_errors,all_test_errors,best_val_error_f,best_model_f


# In[34]:


expirements=['random_split_unweighted_graph','educated_TM_unweighted_graph', 'educated_ligand2ligand_similarity_unweighted_graph','educated_interactions_similarity_unweighted_graph']


# In[35]:


dic=dict()
for thre in [0.5]:
    for exp in expirements:
        l=[]
#         exp+=str(thre)
        print(exp)
        if 'random_split' in exp:
            train_sets,val_sets=educated_split(0.5)
            res01,res02,res03,res04,res5=kfold(train_sets,val_sets)
            l.append(res01)
            l.append(res02)
            l.append(res03)
            l.append(res04)
            l.append(res5)
            dic[exp]=l
        elif 'TM' in exp:
            test_ids=[]
            train_sets,val_sets=educated_split(thre,'TMscore.csv')
            res11,res12,res13,res14,res5=kfold(train_sets,val_sets)
            l.append(res11)
            l.append(res12)
            l.append(res13)
            l.append(res14)
            l.append(res5)
            dic[exp]=l
        elif 'ligand' in exp:
            train_sets,val_sets=educated_split(thre,'nligands_fps.csv')
            res11,res12,res13,res14,res5=kfold(train_sets,val_sets)
            l.append(res11)
            l.append(res12)
            l.append(res13)
            l.append(res14)
            l.append(res5)
            dic[exp]=l
        elif 'interaction' in exp:
            train_sets,val_sets=educated_split(thre,'ninteractions_fps.csv')
            res11,res12,res13,res14,res5=kfold(train_sets,val_sets)
            l.append(res11)
            l.append(res12)
            l.append(res13)
            l.append(res14)
            l.append(res5)
            dic[exp]=l


# In[48]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(12, 8), dpi=80)

train_line, = plt.plot(list(range(1, 501)), dic[expirements[0]][0][0], label = 'train MAE F1')
test_line, = plt.plot(list(range(1, 501)), dic[expirements[0]][1][0], label = 'Validation MAE F1')

train_line1, = plt.plot(list(range(1, 501)), dic[expirements[0]][0][1],  label = 'train MAE F2')
test_line1, = plt.plot(list(range(1, 501)), dic[expirements[0]][1][1], label = 'Validation MAE F2')

train_line2, = plt.plot(list(range(1, 501)), dic[expirements[0]][0][2], label = 'train MAE F3')
test_line2, = plt.plot(list(range(1, 501)), dic[expirements[0]][1][2],  label = 'Validation MAE F3')

train_line3, = plt.plot(list(range(1, 501)), dic[expirements[0]][0][3],  label = 'train MAE F4')
test_line3, = plt.plot(list(range(1, 501)), dic[expirements[0]][1][3],  label = 'Validation MAE F4')

train_line4, = plt.plot(list(range(1, 501)), dic[expirements[0]][0][4],  label = 'train MAE F5' )
test_line4, = plt.plot(list(range(1, 501)), dic[expirements[0]][1][4],  label = 'Validation MAE F5')

plt.axhline(y=1.0, color='b', linestyle='dashed')
plt.legend()
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.show()


# ## TM Split Results:

# In[36]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(12, 8), dpi=80)
train_line, = plt.plot(list(range(1, 501)), dic[expirements[1]][0][0], label = 'train MAE F1')
test_line, = plt.plot(list(range(1, 501)), dic[expirements[1]][1][0], label = 'Validation MAE F1')


train_line1, = plt.plot(list(range(1, 501)), dic[expirements[1]][0][1],  label = 'train MAE F2')
test_line1, = plt.plot(list(range(1, 501)), dic[expirements[1]][1][1], label = 'Validation MAE F2')


train_line2, = plt.plot(list(range(1, 501)), dic[expirements[1]][0][2], label = 'train MAE F3')
test_line2, = plt.plot(list(range(1, 501)), dic[expirements[1]][1][2],  label = 'Validation MAE F3')


train_line3, = plt.plot(list(range(1, 501)), dic[expirements[1]][0][3],  label = 'train MAE F4')
test_line3, = plt.plot(list(range(1, 501)), dic[expirements[1]][1][3],  label = 'Validation MAE F4')


train_line4, = plt.plot(list(range(1, 501)), dic[expirements[1]][0][4],  label = 'train MAE F5' )
test_line4, = plt.plot(list(range(1, 501)), dic[expirements[1]][1][4],  label = 'Validation MAE F5')
plt.axhline(y=1.0, color='b', linestyle='dashed')
plt.legend()
plt.ylabel('MAE Error')
plt.xlabel('Epoch')
plt.show()


# ## Lig2Lig Splitting :

# In[40]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(12, 8), dpi=80)

train_line, = plt.plot(list(range(1, 501)), dic[expirements[2]][0][0], label = 'train MAE F1')
test_line, = plt.plot(list(range(1, 501)), dic[expirements[2]][1][0], label = 'Validation MAE F1')

train_line1, = plt.plot(list(range(1, 501)), dic[expirements[2]][0][1],  label = 'train MAE F2')
test_line1, = plt.plot(list(range(1, 501)), dic[expirements[2]][1][1], label = 'Validation MAE F2')

train_line2, = plt.plot(list(range(1, 501)), dic[expirements[2]][0][2], label = 'train MAE F3')
test_line2, = plt.plot(list(range(1, 501)), dic[expirements[2]][1][2],  label = 'Validation MAE F3')

train_line3, = plt.plot(list(range(1, 501)), dic[expirements[2]][0][3],  label = 'train MAE F4')
test_line3, = plt.plot(list(range(1, 501)), dic[expirements[2]][1][3],  label = 'Validation MAE F4')


train_line4, = plt.plot(list(range(1, 501)), dic[expirements[2]][0][4],  label = 'train MAE F5' )
test_line4, = plt.plot(list(range(1, 501)), dic[expirements[2]][1][4],  label = 'Validation MAE F5')

plt.axhline(y=1.0, color='b', linestyle='dashed')
plt.legend()
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.show()


# ## Interaction Fingerprint Split

# In[44]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(12, 8), dpi=80)

train_line, = plt.plot(list(range(1, 501)), dic[expirements[3]][0][0], label = 'train MAE F1')
test_line, = plt.plot(list(range(1, 501)), dic[expirements[3]][1][0], label = 'Validation MAE F1')

train_line1, = plt.plot(list(range(1, 501)), dic[expirements[3]][0][1],  label = 'train MAE F2')
test_line1, = plt.plot(list(range(1, 501)), dic[expirements[3]][1][1], label = 'Validation MAE F2')


train_line2, = plt.plot(list(range(1, 501)), dic[expirements[3]][0][2], label = 'train MAE F3')
test_line2, = plt.plot(list(range(1, 501)), dic[expirements[3]][1][2],  label = 'Validation MAE F3')


train_line3, = plt.plot(list(range(1, 501)), dic[expirements[3]][0][3],  label = 'train MAE F4')
test_line3, = plt.plot(list(range(1, 501)), dic[expirements[3]][1][3],  label = 'Validation MAE F4')


train_line4, = plt.plot(list(range(1, 501)), dic[expirements[3]][0][4],  label = 'train MAE F5' )
test_line4, = plt.plot(list(range(1, 501)), dic[expirements[3]][1][4],  label = 'Validation MAE F5')

plt.axhline(y=1.0, color='b', linestyle='dashed')
plt.legend()
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.show()


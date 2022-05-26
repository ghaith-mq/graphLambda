import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.nn import  GCNConv, global_add_pool,GATConv,GINConv
from torch.nn import Sequential, Linear, ReLU

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#GCN-network
		self.conv1 = GCNConv(373, 256, cached=False )
		self.bn01 = BatchNorm1d(256)
		self.conv2 = GCNConv(256, 128, cached=False )
		self.bn02 = BatchNorm1d(128)
		self.conv3 = GCNConv(128, 128, cached=False)
		self.bn03 = BatchNorm1d(128)
		#GAT-network
		self.gat1 = GATConv(373, 256,heads=3)
		self.bn11 = BatchNorm1d(256*3)
		self.gat2 = GATConv(256*3, 128,heads=3)
		self.bn12 = BatchNorm1d(128*3)
		self.gat3 = GATConv(128*3, 128,heads=3)
		self.bn13 = BatchNorm1d(128*3)
		#GIN-network
		fc_gin1=Sequential(Linear(373, 256), ReLU(), Linear(256, 256))
		self.gin1 = GINConv(fc_gin1)
		self.bn21 = BatchNorm1d(256)
		fc_gin2=Sequential(Linear(256, 128), ReLU(), Linear(128, 128))
		self.gin2 = GINConv(fc_gin2)
		self.bn22 = BatchNorm1d(128)
		fc_gin3=Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
		self.gin3 = GINConv(fc_gin3)
		self.bn23 = BatchNorm1d(64)
		#1D-CNN
		self.con1= nn.Conv1d(3,16,3)
		self.ba1= BatchNorm1d(16)
		self.con2= nn.Conv1d(16,32,3)
		self.ba2= BatchNorm1d(32)
		self.dropout3=Dropout(p=0.2)
		self.con3= nn.Conv1d(32,8,3)
		self.ba3= BatchNorm1d(8)
		self.flatten=nn.Flatten(start_dim=1)
		#Fully connected layers for concatinating outputs
		self.fc1=Linear(128*4 + 64 + 120, 256)
		self.dropout1=Dropout(p=0.2)
		self.fc2=Linear(256, 64)
		self.dropout2=Dropout(p=0.2)
		self.fc3=Linear(64, 1)
	def forward(self, data):
		x, edge_index = data[0].x, data[0].edge_index,
		y=x
		z=x
		s=data[1]
		#GCN-representation
		x = F.relu(self.conv1(x, edge_index))
		x = self.bn01(x)
		x = F.relu(self.conv2(x, edge_index))
		x = self.bn02(x)
		x = F.relu(self.conv3(x, edge_index))
		x = self.bn03(x)
		x = global_add_pool(x, data[0].batch)
		#GAT-representation
		y = F.relu(self.gat1(y, edge_index))
		y = self.bn11(y)
		y = F.relu(self.gat2(y, edge_index))
		y = self.bn12(y)
		y = F.relu(self.gat3(y, edge_index))
		y = self.bn13(y)
		y = global_add_pool(y, data[0].batch)
		#GIN-representation
		z = F.relu(self.gin1(z, edge_index))
		z = self.bn21(z)
		z = F.relu(self.gin2(z, edge_index))
		z = self.bn22(z)
		z = F.relu(self.gin3(z, edge_index))
		z = self.bn23(z)
		z = global_add_pool(z, data[0].batch)
		#Sequence_representation
		s= F.relu(self.con1(s))
		s=self.ba1(s)
		s= F.relu(self.con2(s))
		s=self.ba2(s)
		s=self.dropout3(s)
		s= F.relu(self.con3(s))
		s=self.ba3(s)
		s=self.flatten(s)
		#Concatinating_representations
		cr=torch.cat((x,y,z,s),1)
		cr = F.relu(self.fc1(cr))
		cr = self.dropout1(cr)
		cr = F.relu(self.fc2(cr))
		cr = self.dropout2(cr)
		cr = self.fc3(cr)
		cr = F.relu(cr).view(-1)
		return cr  

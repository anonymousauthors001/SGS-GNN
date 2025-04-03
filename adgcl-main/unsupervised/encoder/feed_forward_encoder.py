import torch
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn



class FeedForwardNetwork(torch.nn.Module):
	def __init__(self, in_features, out_features, num_fc_layers, dropout):
		super(FeedForwardNetwork, self).__init__()

		self.num_fc_layers = num_fc_layers
		self.fcs = torch.nn.ModuleList()	

		for i in range(num_fc_layers):
			if i == num_fc_layers -1:
				fc = Linear(in_features, out_features)
			else:
				fc = Sequential(Linear(in_features, in_features), ReLU(), Dropout(p=dropout))
			self.fcs.append(fc)

	def forward(self, x):
		for i in range(self.num_fc_layers):
			x = self.fcs[i](x)
		return x
	


class EncoderFeedForward(torch.nn.Module):
	def __init__(self,  num_features, dim, num_gc_layers, num_fc_layers, out_features, dropout):
		super(EncoderFeedForward, self).__init__()

		# self.encoder = GCNConv(num_features, dim, num_gc_layers)
		# # input_size_to_feed_forward = dim*num_gc_layers
		# # self.feed_forward = FeedForwardNetwork(input_size_to_feed_forward, out_features, num_fc_layers, dropout)
		# self.feed_forward = nn.Linear(dim,out_features)
		
		self.out_graph_dim = out_features
		self.out_node_dim = out_features

		self.gcn1 = GCNConv(num_features, dim)
		self.dropout = nn.Dropout(dropout)
		self.gcn2 = GCNConv(dim, out_features)


	def forward(self, batch, x, edge_index, edge_weight=None):

		# x = self.encoder(batch, x, edge_index, edge_weight)
		# x = self.feed_forward(x)
		# #x = F.log_softmax(x, dim=-1)
		x = F.relu(self.gcn1(x, edge_index, edge_weight))
		x = self.dropout(x)
		out = self.gcn2(x, edge_index, edge_weight)

		return out, out
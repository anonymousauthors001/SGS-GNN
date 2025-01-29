import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, ChebConv, GAT, GIN

# Define the MLP for edge probability with dropout
class EdgeProbMLP(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout_prob=0.2):
        super(EdgeProbMLP, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.fcdim = nn.Linear(in_channels, hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index, random_sampled_edge_index=None):

        x = self.dropout(F.relu(self.fcdim(node_features[edge_index[0]])))
        y = self.dropout(F.relu(self.fcdim(node_features[edge_index[1]])))

        edge_features = torch.cat([x*y,x-y], dim=1) #x*y|x-y
        x = F.relu(self.fc1(edge_features))
        # x = x + self.fc_residual(edge_features) #  # Add residual connection
        x = self.dropout(x)
        prob = torch.sigmoid(self.fc2(x))
        # prob.requires_grad_(True)
        return prob
    
class EdgeProbSAGE(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout_prob=0.2):
        super(EdgeProbSAGE, self).__init__()
        self.gcn1 = SAGEConv(in_channels, hidden_dim) #sample 1 hop neighborhood        
        #self.gcn2 = SAGEConv(hidden_dim, hidden_dim) #sample 2 hop neighborhood
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.fc_residual = nn.Linear(2 * hidden_dim, hidden_dim)  # For residual connection
        #self.fc1 = nn.Linear(2 * in_channels, hidden_dim) #if straightforward link
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index, random_sampled_edge_index=None):

        if random_sampled_edge_index is not None:
            out = self.dropout(F.relu(self.gcn1(node_features, random_sampled_edge_index)))        
            # out = F.relu(self.gcn2(out, random_sampled_edge_index))
        else:
            out = self.dropout(F.relu(self.gcn1(node_features, edge_index)))        
            # out = F.relu(self.gcn2(out, edge_index))

        x = out[edge_index[0]]
        y = out[edge_index[1]]
    
        edge_features = torch.cat([x*y,x-y], dim=1) #x*y|x-y
        x = F.relu(self.fc1(edge_features))
        # x = x + self.fc_residual(edge_features) #  # Add residual connection
        x = self.dropout(x)
        prob = torch.sigmoid(self.fc2(x))
        # prob.requires_grad_(True)
        return prob

class EdgeProbGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout_prob=0.2):
        super(EdgeProbGCN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim) #sample 1 hop neighborhood        
        self.gcn2 = GCNConv(hidden_dim, hidden_dim) #sample 2 hop neighborhood
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.fc_residual = nn.Linear(2 * hidden_dim, hidden_dim)  # For residual connection
        #self.fc1 = nn.Linear(2 * in_channels, hidden_dim) #if straightforward link
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index, random_sampled_edge_index=None):

        if random_sampled_edge_index is not None:
            out = self.dropout(F.relu(self.gcn1(node_features, random_sampled_edge_index)))        
            out = F.relu(self.gcn2(out, random_sampled_edge_index))
        else:
            out = self.dropout(F.relu(self.gcn1(node_features, edge_index)))        
            out = F.relu(self.gcn2(out, edge_index))
        
        x = out[edge_index[0]]
        y = out[edge_index[1]]
        
        edge_features = torch.cat([x*y,x-y], dim=1) #x*y|x-y
        x = F.relu(self.fc1(edge_features))
        # x = x + self.fc_residual(edge_features) #  # Add residual connection
        x = self.dropout(x)
        prob = torch.sigmoid(self.fc2(x))
        # prob.requires_grad_(True)
        return prob

def get_edge_mlp(in_channels, hidden_dim, dropout_prob,edge_mlp_type='MLP'):
    edge_prob_mlp = None
    if edge_mlp_type == 'MLP':
        edge_prob_mlp = EdgeProbMLP(in_channels, hidden_dim, dropout_prob)
    elif edge_mlp_type == 'GSAGE':
        edge_prob_mlp = EdgeProbSAGE(in_channels, hidden_dim, dropout_prob)            
    elif edge_mlp_type == 'GCN':
        edge_prob_mlp = EdgeProbGCN(in_channels, hidden_dim, dropout_prob)
    else:
        raise NotImplementedError
    return edge_prob_mlp

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout_prob=0.3, edge_mlp_type='MLP'):
        super(GNNModel, self).__init__()
        self.edge_prob_mlp = get_edge_mlp(in_channels, hidden_dim, dropout_prob,edge_mlp_type)        
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.gcn2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data, edge_index, edge_weight=None):
        x = F.relu(self.gcn1(data.x, edge_index, edge_weight))
        x = self.dropout(x)
        out = self.gcn2(x, edge_index, edge_weight)
        return out


class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout_prob=0.3,edge_mlp_type='MLP'):
        super(GINModel, self).__init__()
        self.edge_prob_mlp = get_edge_mlp(in_channels, hidden_dim, dropout_prob,edge_mlp_type)      
        self.dropout_prob = dropout_prob
        self.GIN = GIN(in_channels = in_channels,
                        hidden_channels = hidden_dim, 
                        num_layers = 2, out_channels = num_classes,
                        dropout = dropout_prob, 
                        act = 'relu')
                    
        
    def forward(self, data, edge_index, edge_weight=None):
        x = self.GIN(data.x, edge_index, edge_weight=edge_weight)
        return x


class GATModel(torch.nn.Module):        
    def __init__(self, in_channels, hidden_dim, num_classes, dropout_prob=0.3, heads=8, edge_mlp_type='MLP'):
        super(GATModel, self).__init__()
        self.edge_prob_mlp = get_edge_mlp(in_channels, hidden_dim, dropout_prob,edge_mlp_type)      
        self.dropout_prob = dropout_prob
        
        self.GAT =  GAT(in_channels = in_channels,
                        hidden_channels = hidden_dim, 
                        num_layers = 2, out_channels = num_classes,
                        dropout = dropout_prob, 
                        act = 'relu')
        
    def forward(self, data, edge_index, edge_weight=None):
        x = self.GAT(data.x, edge_index, edge_weight=edge_weight)        
        return x


class ChebModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout_prob=0.3, edge_mlp_type='MLP'):
        super(ChebModel, self).__init__()
        self.edge_prob_mlp = get_edge_mlp(in_channels, hidden_dim, dropout_prob,edge_mlp_type)      
        self.dropout_prob = dropout_prob
        
        self.gcn1 = ChebConv(in_channels, hidden_dim, K=1, normalization='sym')
        self.dropout = nn.Dropout(dropout_prob)
        self.gcn2 = ChebConv(hidden_dim, num_classes, K=1, normalization='sym')

    def forward(self, data, edge_index, edge_weight=None):
        x = F.relu(self.gcn1(data.x, edge_index, edge_weight))
        x = self.dropout(x)
        out = self.gcn2(x, edge_index, edge_weight)
        return out
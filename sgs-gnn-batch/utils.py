import torch 
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
import torch.nn.functional as F
import networkx as nx
import matplotlib.colors as mcolors
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from sampling import * 

def fix_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_probs(edge_probs,sampling_probs):
    ep = edge_probs.detach().cpu().numpy()
    sp = sampling_probs.detach().cpu().numpy()
    
    plt.figure(figsize=(20, 6))    
    plt.subplot(2, 1, 1)

    #plt.plot(ep)
    plt.scatter(range(len(ep)), ep, s=2)
    plt.title('Curve of Edge Probs')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    #plt.plot(sp)
    plt.scatter(range(len(sp)), sp, s=2)
    plt.title('Curve of Samples Probs')
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    
def plot_hist(edge_probs,sampling_probs, ep_select, sp_select):
    ep = edge_probs.detach().cpu().numpy()
    sp = sampling_probs.detach().cpu().numpy()
    ep_select = ep_select.detach().cpu().numpy()
    sp_select = sp_select.detach().cpu().numpy()
    
    plt.figure(figsize=(20, 15))    
    plt.subplot(4, 1, 1)

    #plt.plot(ep)
    plt.hist(ep, bins= 30, edgecolor='black')
    plt.title('Histogram of Edge Probs')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    #plt.plot(sp)
    plt.hist(sp, bins= 30, edgecolor='black')
    plt.title('histogram of Samples Probs')
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    #plt.plot(sp)
    plt.hist(ep_select, bins= 30, edgecolor='black')
    plt.title('histogram of selected Edge Probs')
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    #plt.plot(sp)
    plt.hist(sp_select, bins= 30, edgecolor='black')
    plt.title('histogram of selected sampled Probs')
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.grid(True)

    
    
    plt.tight_layout()
    plt.show()
    
# F1 score calculation helper
def calculate_f1(logits, labels, mask):
    preds = logits[mask].argmax(dim=1)
    f1 = f1_score(labels[mask].cpu(), preds.cpu(), average='micro')
    return f1

# def compute_node_similarity(node_embeddings):
#     """
#     Compute pairwise node similarity using cosine similarity.
#     Args:
#         node_embeddings (torch.Tensor): Node embeddings, shape [num_nodes, embedding_dim].
#     Returns:
#         torch.Tensor: Pairwise similarity matrix, shape [num_nodes, num_nodes].
#     """
#     print(node_embeddings.shape)
#     similarity_matrix = F.cosine_similarity(
#         node_embeddings.unsqueeze(1), node_embeddings.unsqueeze(0), dim=-1
#     )
#     return similarity_matrix

def consistency_loss(edge_probs, edge_indices, node_embeddings):
    """
    Consistency loss to align edge probabilities with global node similarity.
    Args:
        edge_probs (torch.Tensor): Predicted edge probabilities, shape [num_edges].
        edge_indices (torch.Tensor): Indices of edges, shape [2, num_edges].
        node_embeddings (torch.Tensor): Node embeddings, shape [num_nodes, embedding_dim].
    Returns:
        torch.Tensor: Consistency loss.
    """
    # print(edge_indices.shape)
    # Compute similarity matrix
    # similarity_matrix = compute_node_similarity(node_embeddings)
    
    # Extract similarities for given edges
    src, dst = edge_indices  # Split edge indices into source and destination
    # edge_similarities = similarity_matrix[src, dst]
    src_embeddings = node_embeddings[src]
    dst_embeddings = node_embeddings[dst]
    # Compute cosine similarity for edge pairs
    edge_similarities = F.cosine_similarity(src_embeddings, dst_embeddings, dim=-1)
    
    # Consistency loss as MSE between edge probabilities and node similarities
    loss = F.mse_loss(edge_probs[src], edge_similarities)
    return loss

def plot_full_graph_with_train_nodes(DATASET_NAME, data, train_mask, ax=None):
    """
    Plots the full graph with training nodes highlighted and colored by their labels.
    Args:
        data: Data object containing edge_index, node labels, and other features.
        train_mask: Boolean tensor indicating training nodes.
        ax: Matplotlib axis on which to plot (for side-by-side plotting).
    """
    # Convert PyG data to NetworkX graph for visualization
    G = to_networkx(data)
    
    # Generate color map for each label
    unique_labels = torch.unique(data.y[train_mask]).tolist()
    color_map = {label: color for label, color in zip(unique_labels, mcolors.TABLEAU_COLORS)}

    # Set node colors based on whether they are a training node and their label
    node_colors = [
        color_map[data.y[i].item()] if train_mask[i].item() else 'gray' for i in range(data.num_nodes)
    ]
    
    pos = None
    
    if DATASET_NAME == 'moon':
        pos = {i: (data.x[i][0].item(), data.x[i][1].item()) for i in range(len(data.x))}
    elif pos is None:
        pos = nx.spring_layout(G, seed=42)

    # Plot the full graph with training nodes highlighted by label color
    nx.draw(G, pos, node_color=node_colors, ax=ax, with_labels=False, node_size=100, edge_color="lightblue")
    
    # Add label above each node showing the label value
    for node, p in pos.items():
        label = data.y[node].item()
        ax.text(p[0], p[1] + 0.05, str(label), fontsize=8, ha='center', va='center')
    
    ax.set_title("Full Graph with Training Nodes Highlighted by Label")
    ax.axis("off")
    
    return pos


def plot_sampled_subgraph(data, sampled_edge_index, train_mask, ax=None, pos=None):
    """
    Plots the sampled subgraph based on the edge probability model with training nodes highlighted by label.
    Args:
        data: Data object containing node information.
        sampled_edge_index: Edge index of the sampled subgraph.
        train_mask: Boolean tensor indicating training nodes.
        ax: Matplotlib axis on which to plot (for side-by-side plotting).
    """
    # Convert sampled edges to NetworkX graph
    G_sampled = to_networkx(Data(x = data.x, y = data.y, edge_index = sampled_edge_index))
    
    # Use same color map as the full graph
    unique_labels = torch.unique(data.y[train_mask]).tolist()
    color_map = {label: color for label, color in zip(unique_labels, mcolors.TABLEAU_COLORS)}

    # Use same layout as the full graph for consistent positioning
    if pos is None:
        pos = nx.spring_layout(G_sampled, seed=42)
    
    # Set node colors based on training status and label for nodes in the sampled graph
    node_colors = [
        color_map[data.y[i].item()] if train_mask[i].item() and i in G_sampled.nodes else 'gray'
        for i in G_sampled.nodes
    ]

    # Plot the sampled subgraph with node labels
    nx.draw(G_sampled, pos, node_color=node_colors, ax=ax, with_labels=False, node_size=100, edge_color="orange")
    
    # Add label above each node showing the label value
    for node, p in pos.items():
        label = data.y[node].item()
        ax.text(p[0], p[1] + 0.05, str(label), fontsize=8, ha='center', va='center')
    
    ax.set_title("Sampled Subgraph (Edge Probability Model) with Training Nodes Highlighted by Label")
    ax.axis("off")

def count_edges_with_different_labels(data, sampled_edge_index):
    """
    Counts edges with different labels at endpoints in both the original and sampled graphs.
    
    Args:
        data: Data object containing the original graph information.
        sampled_edge_index: Edge index of the sampled subgraph.
    
    Returns:
        A dictionary with counts and percentages of edges with different labels.
    """
    # Original graph edges
    original_edges = data.edge_index
    num_original_edges = original_edges.size(1)
    
    # Count edges with different labels in the original graph
    different_label_edges_original = 0
    for i in range(num_original_edges):
        node1, node2 = original_edges[:, i]
        if data.y[node1].item() != data.y[node2].item():
            different_label_edges_original += 1
    
    # Sampled graph edges
    sampled_edges = sampled_edge_index
    num_sampled_edges = sampled_edges.size(1)

    # Count edges with different labels in the sampled graph
    different_label_edges_sampled = 0
    for i in range(num_sampled_edges):
        node1, node2 = sampled_edges[:, i]
        if data.y[node1].item() != data.y[node2].item():
            different_label_edges_sampled += 1

    # Calculate percentages
    percentage_original = (different_label_edges_original / num_original_edges) * 100 if num_original_edges > 0 else 0
    percentage_sampled = (different_label_edges_sampled / num_sampled_edges) * 100 if num_sampled_edges > 0 else 0

    # Prepare the result
    result = {
        "original_graph": {
            "different_label_edges": different_label_edges_original,
            "total_edges": num_original_edges,
            "percentage": percentage_original
        },
        "sampled_graph": {
            "different_label_edges": different_label_edges_sampled,
            "total_edges": num_sampled_edges,
            "percentage": percentage_sampled
        }
    }

    return result

def visualize_graphs_side_by_side(data, train_mask, sampled_edge_index, show=True):
    """
    Visualizes the full graph with training nodes and the sampled subgraph side by side, with training nodes colored by label.
    Args:
        data: Data object containing the full graph information.
        train_mask: Boolean tensor indicating training nodes.
        sampled_edge_index: Edge index of the sampled subgraph.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot full graph with training nodes highlighted by label
    pos1 = plot_full_graph_with_train_nodes(data, train_mask, ax=ax1)

    # Plot sampled subgraph with training nodes highlighted by label, using the same layout
    plot_sampled_subgraph(data, sampled_edge_index, train_mask, ax=ax2, pos=pos1)
    
    if show==False:
        plt.close()
    else:
        plt.show()
    
    return fig

# Example Usage:
# Assuming `train_mask` is a boolean mask tensor, and `sampled_edge_index` is the edge index after sampling
# visualize_graphs_side_by_side(data, train_mask=data.train_mask, sampled_edge_index=sampled_edge_index)
def visualize(use_metis, sample_size, data, model, istest=True, show=True):
    if use_metis == False:
        with torch.no_grad():
            edge_probs = model.edge_prob_mlp(data.x, data.edge_index).squeeze()
            sampled_edge_indices, sampled_edge_weight = gumbel_softmax_sampling(data,
                edge_probs,
                data.edge_index,
                q=sample_size, istest=istest)

            sampled_edge_index = data.edge_index[:, sampled_edge_indices]
            fig = None
            #fig = visualize_graphs_side_by_side(data, train_mask=data.train_mask, sampled_edge_index=sampled_edge_index, show=show)

            # Example Usage
            # Assuming `data` is your original graph and `sampled_edge_index` is your sampled edge index
            result = count_edges_with_different_labels(data, sampled_edge_index)

            print("\% of heterophily:")

            # Print the results
            print(f"Original graph: {result['original_graph']['different_label_edges']} edges with different labels "
                  f"out of {result['original_graph']['total_edges']} total edges "
                  f"({result['original_graph']['percentage']:.2f}%).")

            print(f"Sampled graph: {result['sampled_graph']['different_label_edges']} edges with different labels "
                  f"out of {result['sampled_graph']['total_edges']} total edges "
                  f"({result['sampled_graph']['percentage']:.2f}%).")
            
            return fig
            
# fig = visualize(False) #Sampled from the best
# fig = visualize(True)  #Best Sparsifier

def plot_learning_curves(run, train_f1_scores,val_f1_scores,test_f1_scores, save = True):
    # Plot the F1 scores of last run
    plt.figure(figsize=(10, 6))
    plt.plot(train_f1_scores, label='Train F1 Score')
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.plot(test_f1_scores, label='Test F1 Score')
    # plt.plot(loss_scores, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores over Epochs')
    plt.legend()
    # plt.show()
    plt.savefig(str(run)+'.png')
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import ft2font

font_path = "KDDFigures/Times New Roman.ttf"

# 1) Register the .ttf with Matplotlib
fm.fontManager.addfont(font_path)

# 2) Use the *exact* internal family name from the file
family_name = ft2font.FT2Font(font_path).family_name

# 3) Make it the default for all figures
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [family_name]

# # quick sanity plot
# plt.figure()
# plt.title("Times New Roman title")
# plt.xlabel("x label")
# plt.ylabel("y label")
# plt.text(0.5, 0.5, f"Family: {family_name}", ha="center")
# plt.show()



from .utils import select_two_classes


def count_edges_with_different_labels(data, edge_index):
    original_edges = data.edge_index
    num_original_edges = original_edges.size(1)

    different_label_edges_original = 0
    for i in range(num_original_edges):
        node1, node2 = original_edges[:, i]
        if data.y[node1].item() != data.y[node2].item():
            different_label_edges_original += 1

    sampled_edges = edge_index
    num_sampled_edges = sampled_edges.size(1)

    different_label_edges_sampled = 0
    for i in range(num_sampled_edges):
        node1, node2 = sampled_edges[:, i]
        if data.y[node1].item() != data.y[node2].item():
            different_label_edges_sampled += 1

    percentage_original = (
        (different_label_edges_original / num_original_edges) * 100
        if num_original_edges > 0
        else 0
    )
    percentage_sampled = (
        (different_label_edges_sampled / num_sampled_edges) * 100
        if num_sampled_edges > 0
        else 0
    )

    return {
        "original_graph": {
            "different_label_edges": different_label_edges_original,
            "total_edges": num_original_edges,
            "percentage": percentage_original,
        },
        "sampled_graph": {
            "different_label_edges": different_label_edges_sampled,
            "total_edges": num_sampled_edges,
            "percentage": percentage_sampled,
        },
    }


def compute_graph_layout(data, graph=None, dataset_name: Optional[str] = None, seed: int = 42):
    if dataset_name == "moon":
        nodes = graph.nodes if graph is not None else range(data.num_nodes)
        return {i: (float(data.x[i][0]), float(data.x[i][1])) for i in nodes}

    if graph is None:
        graph = to_networkx(data)

    return nx.spring_layout(graph, seed=seed)


def plot_graph_two_class(
    data,
    edge_index=None,
    ax=None,
    pos=None,
    class_pair: Optional[Iterable[int]] = None,
    dataset_name: Optional[str] = None,
    node_size: int = 50,
    highlight_colors: Tuple[str, str] = ("blue", "red"),
    edge_color: str = "lightgray",
    other_color: str = "lightgray",
    edge_alpha: float = 0.35,
    title: Optional[str] = None,
    filter_to_classes: bool = False,
    with_labels: bool = False,
    seed: int = 42,
):
    class_pair = select_two_classes(data.y, class_pair)
    graph_data = data if edge_index is None else Data(x=data.x, y=data.y, edge_index=edge_index)
    graph = to_networkx(graph_data)

    if filter_to_classes:
        nodes = [i for i in graph.nodes if int(data.y[i].item()) in class_pair]
        graph = graph.subgraph(nodes).copy()

    if pos is None:
        pos = compute_graph_layout(data, graph=graph, dataset_name=dataset_name, seed=seed)

    node_colors = []
    for node in graph.nodes:
        label = int(data.y[node].item())
        if label == class_pair[0]:
            node_colors.append(highlight_colors[0])
        elif label == class_pair[1]:
            node_colors.append(highlight_colors[1])
        else:
            node_colors.append(other_color)

    if ax is None:
        _, ax = plt.subplots()

    edge_colors = []
    for u, v in graph.edges:
        u_label = int(data.y[u].item())
        v_label = int(data.y[v].item())
        if u_label in class_pair and v_label in class_pair:
            if u_label == v_label == class_pair[0]:
                edge_colors.append("blue")
            elif u_label == v_label == class_pair[1]:
                edge_colors.append("red")
            else:
                edge_colors.append("gray")
        else:
            edge_colors.append(edge_color)

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=edge_colors,
        alpha=edge_alpha,
        width=0.5,
    )
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_size, linewidths=0)

    if with_labels:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)

    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")

    return pos


def visualize_graphs_side_by_side(
    data,
    sampled_edge_index,
    class_pair: Optional[Iterable[int]] = None,
    dataset_name: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 7),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    pos = plot_graph_two_class(
        data,
        ax=ax1,
        class_pair=class_pair,
        dataset_name=dataset_name,
        title="Original dense graph",
    )
    plot_graph_two_class(
        data,
        edge_index=sampled_edge_index,
        ax=ax2,
        pos=pos,
        class_pair=class_pair,
        dataset_name=dataset_name,
        title="Learned sparse subgraph",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

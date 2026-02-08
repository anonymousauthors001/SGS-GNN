from __future__ import annotations

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from .utils import select_two_classes


def extract_embeddings(
    model,
    data,
    edge_index=None,
    edge_weight=None,
    device=None,
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    data = data.to(device)
    if edge_index is None:
        edge_index = data.edge_index
    else:
        edge_index = edge_index.to(device)

    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        if hasattr(model, "get_embeddings"):
            embeddings = model.get_embeddings(data, edge_index, edge_weight=edge_weight)
        else:
            embeddings = torch.relu(model.gcn1(data.x, edge_index, edge_weight))

    return embeddings.detach().cpu()


def reduce_embeddings_2d(embeddings, method: str = "pca", random_state: int = 42):
    if torch.is_tensor(embeddings):
        emb = embeddings.detach().cpu().numpy()
    else:
        emb = np.asarray(embeddings)

    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(emb)
    if method in {"tsne", "t-sne"}:
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        ).fit_transform(emb)

    raise ValueError("Unknown reduction method. Use 'pca' or 'tsne'.")


def plot_embeddings_2d(
    embeddings_2d,
    labels,
    class_pair: Optional[Iterable[int]] = None,
    ax=None,
    highlight_colors: Tuple[str, str] = ("blue", "red"),
    other_color: str = "lightgray",
    title: Optional[str] = None,
    s: int = 50,
    alpha: float = 0.8,
):
    class_pair = select_two_classes(labels, class_pair)
    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    colors = []
    for label in labels_np:
        label = int(label)
        if label == class_pair[0]:
            colors.append(highlight_colors[0])
        elif label == class_pair[1]:
            colors.append(highlight_colors[1])
        else:
            colors.append(other_color)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))


    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        s=s,
        alpha=alpha,
        edgecolors="none",
    )
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

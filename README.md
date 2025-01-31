# SGS-GNN: A Supervised Graph Sparsifier for Graph Neural Networks

SGS-GNN is a novel supervised graph sparsification algorithm that learns the sampling probability distribution of edges and samples sparse subgraphs of a user-specified size to reduce the memory required by GNNs for inference tasks on large graphs.

# Installation:

These are the necessary packages for installation from scratch and other related packages.

```
Python version: 3.11
Pytorch version: 2.0.1
Cuda: 11.7
Cudnn: 8.6
Pytorch-Geometric: 2.3.1
```

For direct installation, Conda packages are in `environment.yml`, and PIP packages are in `requirements.txt` and can be imported as,

```
conda env create -f environment.yml
pip install -r requirements.txt
```

# Example usage

In the `sgs-gnn-batch/Scripts` folder, there are bash-scripts that show how our methods can be run. Below are some of the usage examples.

`python main.py --dataset SmallCora`: Will run dataset `SmallCora` with default settings.

For a more specific combination,

`python main.py --dataset SmallCora --mode 'learned' --runs 5 --epochs 200 --save_csv True --sample_perc 0.2 --edge_mlp_type GCN --GNN GCN --nhid 128 --sparse_edge_mlp True --conditional True --reg1 True --reg2 True`


Interpretation:

- `mode` `learned` is for SGS-GNN. Options are `full, edge, random, learned` for fixed distribution samplers, complete graphs version, or our learned sampler.
- `runs` How many times we want to execute the progam
- `epochs` refers to maximum number of epochs
- `sample_perc` sparsity control 0.2 refers to 20% sparsity
- `edge_mlp_type` GCN means for EdgeMLP probabilty encoding what type neural network we want to use, choices are `MLP`, `GSAGE`, and `GCN`
- `GNN` what would be downstream GNN for node classification. Choices are, `GCN`, `GAT`, `GIN`, `Cheb`
- `nhid` Number of hidden neurons
- `sparse_edge_mlp` This is EdgeMLP; during the encoding step, do you want to use a sparse graph or not? This is primarily for large-scale graphs.
- `conditional` Perform conditional updates of EdgeMLP
- `reg1` Regularizer 1 is for the assortative loss $L_{assor}$
- `reg2` Regularizer 2 is for the consistency loss $L_{cons}$

Some other parameters and settings can be found at `sgs-gnn-batch/parser.py`

# Codes of Related Methods:

- NeuralSparse: https://github.com/flyingdoog/PTDNet
- SparseGAT: https://github.com/Yangyeeee/SGAT
- Mixture of graphs (MOG): https://github.com/yanweiyue/MoG
- GraphSAINT: https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/loader/graph_saint.html
- ClusterGCN: https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/loader/cluster.html
- DropEdge: https://github.com/DropEdge/DropEdge



#!/bin/bash

echo "Current date and time: $(date)"

# # for mode in "full" "random" "edge"  "learned"; do
# for mode in "learned"; do    
#     #for dataset in "Cornell" "Texas" "Wisconsin" "reed98" "amherst41" "penn94" "Roman-empire" "cornell5" "Squirrel" "johnshopkins55" "Actor" "Minesweeper" "Questions" "Chameleon" "Tolokers" "Amazon-ratings" "Cora" "DBLP" "Computers" "PubMed" "Cora_ML" "SmallCora" "CS" "Photo" "Physics" "CiteSeer" "wiki"; do
#     for dataset in "Cornell" "Texas" "Wisconsin"; do
#         python main.py --dataset $dataset --mode learned --runs 5 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GCN --log False --sparse_edge_mlp False --conditional True --reg1 True --reg2 True --sample_perc 0.2
#         python main.py --dataset $dataset --mode learned --runs 5 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GAT --log False --sparse_edge_mlp False --conditional True --reg1 True --reg2 True --sample_perc 0.2
#     done
    
# done 

# for mode in "full" "random" "edge"  "learned"; do
for mode in "learned"; do    
    #for dataset in "Cornell" "Texas" "Wisconsin" "reed98" "amherst41" "penn94" "Roman-empire" "cornell5" "Squirrel" "johnshopkins55" "Actor" "Minesweeper" "Questions" "Chameleon" "Tolokers" "Amazon-ratings" "Cora" "DBLP" "Computers" "PubMed" "Cora_ML" "SmallCora" "CS" "Photo" "Physics" "CiteSeer" "wiki"; do
    for dataset in "SmallCora" "Cornell" "Texas" "Wisconsin"; do
        python main.py --dataset $dataset --mode learned --runs 3 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GCN --log False --sparse_edge_mlp False --conditional True --reg1 True --reg2 True --sample_perc 0.2
        python main.py --dataset $dataset --mode learned --runs 3 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GAT --log False --sparse_edge_mlp False --conditional False --reg1 True --reg2 True --sample_perc 0.2
    done
    
done 

# dataset="SmallCora"
# edgeGNN="GSAGE"

# python main.py --dataset $dataset --mode learned --runs 3 --epochs 250 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional True --reg1 True --reg2 True --nhid 64 --eval True --convergence 0.001
# python main.py --dataset $dataset --mode learned --runs 3 --epochs 250 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional False --reg1 True --reg2 True --nhid 64 --eval True --convergence 0.001


# python main.py --dataset Cornell --mode learned --runs 5 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GCN --log False --sparse_edge_mlp False --conditional False --reg1 True --reg2 True --sample_perc 0.2
# python main.py --dataset Cornell --mode learned --runs 5 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GCN --log False --sparse_edge_mlp False --conditional True --reg1 True --reg2 True --sample_perc 0.2
#!/bin/bash

# echo ---------------Convergence----------

# for mode in "edge"; do
#     echo ------$mode-----------
#     #for dataset in "Cornell" "Texas" "Wisconsin" "reed98" "amherst41" "penn94" "Roman-empire" "cornell5" "Squirrel" "johnshopkins55" "Actor" "Minesweeper" "Questions" "Chameleon" "Tolokers" "Amazon-ratings" "Cora" "DBLP" "Computers" "PubMed" "Cora_ML" "SmallCora" "CS" "Photo" "Physics" "CiteSeer" "wiki"; do
#     for dataset in "SmallCora" "Cora" "CiteSeer" "johnshopkins55" "Squirrel" "Roman-empire"; do
#         echo ---------$dataset----------        
#         python main.py --dataset $dataset --mode $mode --runs 10 --epochs 500 --save_csv True --sample_perc 0.2 --convergence 0.001 --ER True
#         #python main.py --dataset SmallCora --mode learned --runs 1 --epochs 500 --save_csv True --sample_perc 0.2 --convergence 0.001 --log True
#         #python main.py --dataset SmallCora --mode random --runs 1 --epochs 500 --save_csv True --sample_perc 0.2 --convergence 0.001 --log True
#     done
# done

# echo ---bothregulazier-----
# for edgeGNN in "GSAGE"; do
#     echo -----------$edgeGNN---------
#     #for dataset in "cornell5" "Tolokers" "genius" "pokec" "arxiv-year" "snap-patents" "Reddit"; do
#     for dataset in "SmallCora" "Cora_ML" "CiteSeer" "reed98" "Roman-empire" "amherst41"; do
#         echo ----------Conditional------------
#         python main.py --dataset $dataset --mode learned --runs 5 --epochs 500 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional True --reg1 True --reg2 True --nhid 64 --eval True --convergence 0.001
#         echo -----------WithoutConditional--------
#         python main.py --dataset $dataset --mode learned --runs 5 --epochs 500 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional False --reg1 True --reg2 True --nhid 64 --eval True --convergence 0.001
#     done
# done

dataset="SmallCora"
edgeGNN="GSAGE"

python main.py --dataset $dataset --mode learned --runs 5 --epochs 500 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional True --reg1 True --reg2 True --nhid 64 --eval True --convergence 0.001
python main.py --dataset $dataset --mode learned --runs 5 --epochs 500 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional False --reg1 True --reg2 True --nhid 64 --eval True --convergence 0.001
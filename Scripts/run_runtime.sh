#!/bin/bash

echo -------Perepochruntime------

PIPELINE="${PIPELINE:-hybrid}"

for mode in "learned"; do
    echo $mode
    #for dataset in "Cornell" "Texas" "Wisconsin" "reed98" "amherst41" "penn94" "Roman-empire" "cornell5" "Squirrel" "johnshopkins55" "Actor" "Minesweeper" "Questions" "Chameleon" "Tolokers" "Amazon-ratings" "Cora" "DBLP" "Computers" "PubMed" "Cora_ML" "SmallCora" "CS" "Photo" "Physics" "CiteSeer" "wiki"; do
    #for dataset in "CS"	"Questions"	"Amazon-ratings"	"johnshopkins55"	"amherst41"; do               
        # python main.py --dataset $dataset --mode $mode --runs 10 --epochs 250 --save_csv True --sample_perc 0.2 --GNN GCN --nhid 128 --pipeline "${PIPELINE}"
        #python main.py --dataset SmallCora --mode edge --runs 1 --epochs 200 --save_csv True --sample_perc 0.2 --GNN GCN --pipeline "${PIPELINE}"
    # done

#     echo "Current date and time: $(date)"
#     echo ---regulazier1only-----
#     for edgeGNN in "GSAGE"; do
#         echo -----------$edgeGNN---------
#         #for dataset in "cornell5" "Tolokers" "genius" "pokec" "arxiv-year" "snap-patents" "Reddit"; do
#         for dataset in "cornell5" "Tolokers" "genius" "pokec" "arxiv-year" "snap-patents" "Reddit"; do
#             echo ----------Conditional------------
#             python main.py --dataset $dataset --mode learned --runs 3 --epochs 100 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional True --reg1 True --reg2 False --nhid 64 --eval False --pipeline "${PIPELINE}"
#             echo -----------WithoutConditional--------
#             python main.py --dataset $dataset --mode learned --runs 3 --epochs 100 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional False --reg1 True --reg2 False --nhid 64 --eval False --pipeline "${PIPELINE}"
#         done
#     done
#     echo ---bothregulazier-----
#     for edgeGNN in "GSAGE"; do
#         echo -----------$edgeGNN---------
#         for dataset in "cornell5" "Tolokers" "genius" "pokec" "arxiv-year" "snap-patents" "Reddit"; do
#             echo ----------Conditional------------
#             python main.py --dataset $dataset --mode learned --runs 3 --epochs 100 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional True --reg1 True --reg2 True --nhid 64 --eval False --pipeline "${PIPELINE}"
#             echo -----------WithoutConditional--------
#             python main.py --dataset $dataset --mode learned --runs 3 --epochs 100 --save_csv True --edge_mlp_type $edgeGNN --GCN GCN --log False --sparse_edge_mlp True --conditional False --reg1 True --reg2 True --nhid 64 --eval False --pipeline "${PIPELINE}"
#         done
#     done
# done

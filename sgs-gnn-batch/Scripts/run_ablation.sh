#!/bin/bash

# python main.py --dataset SmallCora --mode learned --runs 5 --epochs 250 --save_csv True --edge_mlp_type GCN --GNN GAT --log True --sparse_edge_mlp True --conditional True

echo "Current date and time: $(date)"

# for mode in "learned"; do
#     echo $mode
#     for dataset in "SmallCora" "Cora" "johnshopkins55"; do
#     #for dataset in "SmallCora" "Cora" "CiteSeer" "johnshopkins55" "Squirrel" "Roman-empire"; do
#         echo ---------$dataset--------------
#         for GNN in "GCN" "GAT"; do
#             echo ------GNN----$GNN------
#             for edgeGNN in "MLP" "GCN" "GSAGE"; do
#                 echo ----EdgeGNN---$edgeGNN------
#                 python main.py --dataset $dataset --mode learned --runs 5 --epochs 500 --save_csv True --edge_mlp_type $edgeGNN --GNN $GNN --log False --sparse_edge_mlp True --conditional True --reg1 True --reg2 False
#                 python main.py --dataset $dataset --mode learned --runs 5 --epochs 500 --save_csv True --edge_mlp_type $edgeGNN --GNN $GNN --log False --sparse_edge_mlp True --conditional True --reg1 True --reg2 True                
#             done
#         done
#         echo ---------end--------------
#     done 
# done 



# for mode in "learned"; do
#     echo $mode
#     for dataset in "Cora" "johnshopkins55"; do
#     #for dataset in "SmallCora" "Cora" "CiteSeer" "johnshopkins55" "Squirrel" "Roman-empire"; do
#         echo ---------$dataset--------------
#         for GNN in "GCN"; do
#             echo ------GNN----$GNN------
#             for edgeGNN in "MLP"; do
#                 echo ----EdgeGNN---$edgeGNN------
#                 python main.py --dataset $dataset --mode learned --runs 5 --epochs 250 --save_csv True --edge_mlp_type $edgeGNN --GNN $GNN --log False --sparse_edge_mlp True --conditional False --reg1 False --reg2 False
#                 python main.py --dataset $dataset --mode learned --runs 5 --epochs 250 --save_csv True --edge_mlp_type $edgeGNN --GNN $GNN --log False --sparse_edge_mlp True --conditional False --reg1 True --reg2 False
#             done
#         done
#         echo ---------end--------------
#     done 
# done 

# for dataset in "SmallCora"; do
#     echo ---------$dataset--------------    
#     for cons in 0.1 0.3 0.5 0.7 0.9; do
#         for bias in 0.1 0.3 0.5 0.7 0.9; do
#             # cons=0.1
#             # bias=0.1
#             python main.py --dataset $dataset --mode learned --runs 2 --epochs 200 --save_csv True --edge_mlp_type GCN --GNN GCN --log False --sparse_edge_mlp True --conditional False --reg1 True --reg2 True --consist_reg_coef $cons --degree_bias_coef $bias
#         done
#     done        
#     echo ---------end--------------
# done 



for mode in "learned"; do
    echo $mode
    for dataset in "SmallCora"; do
    #for dataset in "SmallCora" "Cora" "CiteSeer" "johnshopkins55" "Squirrel" "Roman-empire"; do
        echo ---------$dataset--------------
        for GNN in "GCN"; do
            echo ------GNN----$GNN------
            for edgeGNN in "GCN"; do
                echo ----EdgeGNN---$edgeGNN------
                python main.py --dataset $dataset --mode learned --runs 1 --epochs 1 --save_csv True --edge_mlp_type $edgeGNN --GNN $GNN --log False --sparse_edge_mlp True --conditional False --reg1 True --reg2 True
                
            done
        done
        echo ---------end--------------
    done 
done 
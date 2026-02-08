#!/bin/bash

#python main.py --dataset SmallCora --mode learned --runs 1 --epochs 100 --save_csv True --sample_perc 0.2 --syn True --degree 100 --hn 0.1 --train 0.2 --log True

echo --------------experimentruning--------

PIPELINE="${PIPELINE:-hybrid}"

for mode in "learned"; do
    echo $mode
    for dataset in "SmallCora"; do
        echo $dataset
        #for hn in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        for hn in 0.6 0.7 0.8 0.9; do
            echo ------------------------homophily:$hn------------------------
            for percent in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do        
                echo $percent            
                python main.py --dataset $dataset --mode learned --runs 10 --epochs 250 --save_csv True --sample_perc $percent --syn True --degree 25 --hn $hn --train 0.1 --pipeline "${PIPELINE}"
                #python main.py --dataset SmallCora --mode learned --runs 5 --epochs 200 --save_csv True --sample_perc 0.2 --syn True --degree 25 --hn 0.1 --train 0.1
            done
            echo ------------------------end------------------------
        done
    done
done 

#!/bin/bash

echo "Current date and time: $(date)"

PIPELINE="${PIPELINE:-hybrid}"

for mode in "learned"; do
    echo $mode
    #for dataset in "SmallCora" "Cora_ML" "Cora" "CiteSeer" "Cornell" "Texas" "Wisconsin"; do
    for dataset in "reed98" "Roman-empire" "amherst41"; do
        echo $dataset
        for percent in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
            echo $percent
            python main.py --dataset $dataset --mode learned --runs 3 --epochs 250 --save_csv True --sample_perc $percent --pipeline "${PIPELINE}"
            #python main.py --dataset Cornell --mode learned --runs 1 --epochs 200 --save_csv True --sample_perc 0.2
        done
    done
done

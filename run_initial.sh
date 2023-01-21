#!/bin/bash
for i in "adult_old" "adult_new" # "public" "taiwan"
do
    for j in "lr" "rf" "svm" "mlp"  # algs
    do
        echo "====================== $i $j ======================"
        python run_initial.py --data=$i --algo=$j --trials=6 --savedir="refactor_results"
    done
done
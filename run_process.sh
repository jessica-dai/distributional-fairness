#!/bin/bash
for i in "adult_old" "adult_new" "public" "taiwan"
do
    for j in "lr" "rf" "svm" "mlp"  # algs
    do
        echo "====================== $i $j ======================"
        python -W ignore run_process.py --data=$i --algo=$j --trials=10 
    done
done
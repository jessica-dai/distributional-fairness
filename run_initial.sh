#!/bin/bash
for i in "taiwan" # "adult_old" "adult_new" "public" "taiwan"
do
    for j in "lr" "rf" "svm" "mlp"  # algs
    do
        echo "====================== $i $j ======================"
        python run_initial.py --data=$i --algo=$j --trials=10 
    done
done
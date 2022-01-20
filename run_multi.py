import numpy as np
import pandas as pd

import argparse

from src.bcmap import geometric_adjustment
from src.lexi import lexicographicOptimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default="1", help="number of trials to run")
    parser.add_argument("--n", default="8000", help="total number of samples used")
    args = parser.parse_args()

    print("All using FICO dataset, downloaded from `responsibly`")
    allprobs = pd.read_csv('data/fico_probs.csv')
    bins = np.arange(0, 100.5, 0.5) # specific bins for FICO

    train_adjusts = pd.DataFrame(columns=['trial', 'adjust', 'score', 'label', 'group', 'repaired_score'])
    test_adjusts = pd.DataFrame(columns=['trial', 'adjust', 'score', 'label', 'group', 'repaired_score', 'tpr_lexi', 'tpr_maxmin', 'fpr_lexi', 'fpr_maxmin'])

    train_lambdas = pd.DataFrame(columns=['trial', 'metric', 'soln', 'asian', 'black', 'hispanic', 'white'])
    # metric: TPR, FPR
    # soln: maxmin and lexi
    train_losses = pd.DataFrame(columns=['trial', 'metric', 'soln', 'asian', 'black', 'hispanic', 'white'])
    # metric: TPR, FPR, PR
    # soln: unconst, maxmin, lexi, fullrep (unconst and fullrep only for PR)

    for i in range(int(args.trials)):
        print()
        print("====== Trial", i + 1, "of", args.trials, "======")
        probs = allprobs.sample(frac=int(args.n)/len(allprobs), random_state=i)

        print("**Splitting dataset and preprocessing**")
        split_ind = int(len(probs)*0.5)
        if i == 0:
            print(split_ind, "examples in train and test")

        print("**Calculating adjustment**")
        adjusted = geometric_adjustment(train_df=probs.iloc[:split_ind], # labelled
                                        test_df=probs, # will be adjusted
                                        sens_col="group", 
                                        score_col="score",
                                        solver="bregman",
                                        bins=bins)

        adjusted['trial'] = i

        adjusted_train = adjusted.iloc[:split_ind]
        adjusted_unlab = adjusted.iloc[split_ind:]

        print("**Finding lambdas - TPR**")
        data_pos_label = adjusted_train[adjusted_train["label"] == 1].copy()
        tpr_lambdas, group_map, tpr_group_losses = lexicographicOptimizer(
            df=data_pos_label, attr_col="group", score_col="score", shift_col="adjust"
        )
        assert(group_map == {'asian': 0, 'black': 1, 'hispanic': 2, 'white': 3})
        
        adjusted_unlab["tpr_lexi"]= adjusted_unlab.apply(lambda x: x['score'] + x['adjust']*tpr_lambdas[-1,group_map[x['group']]], axis=1)
        adjusted_unlab["tpr_maxmin"] =  adjusted_unlab.apply(lambda x: x['score'] + x['adjust']*tpr_lambdas[0, group_map[x['group']]], axis=1)

        print("**Finding lambdas -- FPR**")
        data_neg_label = adjusted_train[adjusted_train["label"] == 0].copy()
        fpr_lambdas, group_map, fpr_group_losses = lexicographicOptimizer(
            df=data_neg_label, attr_col="group", score_col="score", shift_col="adjust"
        )
        assert(group_map == {'asian': 0, 'black': 1, 'hispanic': 2, 'white': 3})
        
        adjusted_unlab["fpr_lexi"]= adjusted_unlab.apply(lambda x: x['score'] + x['adjust']*fpr_lambdas[-1,group_map[x['group']]], axis=1)
        adjusted_unlab["fpr_maxmin"] =  adjusted_unlab.apply(lambda x: x['score'] + x['adjust']*fpr_lambdas[0, group_map[x['group']]], axis=1)

        train_adjusts = train_adjusts.append(adjusted_train, ignore_index=True)
        test_adjusts = test_adjusts.append(adjusted_unlab, ignore_index=True)
    
        print("** Saving lambdas and losses **") # these are all train lambdas and train losses
        curr_lambdas = pd.DataFrame(columns = ['metric', 'soln', 'asian', 'black', 'hispanic', 'white'])
        curr_lambdas = curr_lambdas.append({
            'metric': 'tpr', 'soln': 'lexi', 'asian': tpr_lambdas[-1, 0], 'black': tpr_lambdas[-1, 1], 'hispanic': tpr_lambdas[-1, 2], 'white': tpr_lambdas[-1, 3]
        }, ignore_index=True)
        curr_lambdas = curr_lambdas.append({
            'metric': 'tpr', 'soln': 'maxmin', 'asian': tpr_lambdas[0, 0], 'black': tpr_lambdas[0, 1], 'hispanic': tpr_lambdas[0, 2], 'white': tpr_lambdas[0, 3]
        }, ignore_index=True)
        curr_lambdas = curr_lambdas.append({
            'metric': 'fpr', 'soln': 'lexi', 'asian': fpr_lambdas[-1, 0], 'black': fpr_lambdas[-1, 1], 'hispanic': fpr_lambdas[-1, 2], 'white': fpr_lambdas[-1, 3]
        }, ignore_index=True)
        curr_lambdas = curr_lambdas.append({
            'metric': 'fpr', 'soln': 'maxmin', 'asian': fpr_lambdas[0, 0], 'black': fpr_lambdas[0, 1], 'hispanic': fpr_lambdas[0, 2], 'white': fpr_lambdas[0, 3]
        }, ignore_index=True) 

        curr_lambdas['trial'] = i
        train_lambdas = train_lambdas.append(curr_lambdas, ignore_index=True)

        curr_losses = pd.DataFrame(columns = ['metric', 'soln', 'asian', 'black', 'hispanic', 'white'])
        # tpr
        curr_losses = curr_losses.append({
            'metric': 'tpr', 'soln': 'lexi', 'asian': tpr_group_losses[-1, 0], 'black': tpr_group_losses[-1, 1], 'hispanic': tpr_group_losses[-1, 2], 'white': tpr_group_losses[-1, 3]
        }, ignore_index=True)
        curr_losses = curr_losses.append({
            'metric': 'tpr', 'soln': 'unconst', 'asian': tpr_group_losses[0, 0], 'black': tpr_group_losses[0, 1], 'hispanic': tpr_group_losses[0, 2], 'white': tpr_group_losses[0, 3]
        }, ignore_index=True)
        curr_losses = curr_losses.append({
            'metric': 'tpr', 'soln': 'maxmin', 'asian': tpr_group_losses[2, 0], 'black': tpr_group_losses[2, 1], 'hispanic': tpr_group_losses[2, 2], 'white': tpr_group_losses[2, 3]
        }, ignore_index=True)
        curr_losses = curr_losses.append({
            'metric': 'tpr', 'soln': 'fullrep', 'asian': tpr_group_losses[1, 0], 'black': tpr_group_losses[1, 1], 'hispanic': tpr_group_losses[1, 2], 'white': tpr_group_losses[1, 3]
        }, ignore_index=True)

        # fpr 
        curr_losses = curr_losses.append({
            'metric': 'fpr', 'soln': 'lexi', 'asian': fpr_group_losses[-1, 0], 'black': fpr_group_losses[-1, 1], 'hispanic': fpr_group_losses[-1, 2], 'white': fpr_group_losses[-1, 3]
        }, ignore_index=True)
        curr_losses = curr_losses.append({
            'metric': 'fpr', 'soln': 'unconst', 'asian': fpr_group_losses[0, 0], 'black': fpr_group_losses[0, 1], 'hispanic': fpr_group_losses[0, 2], 'white': fpr_group_losses[0, 3]
        }, ignore_index=True) 
        curr_losses = curr_losses.append({
            'metric': 'fpr', 'soln': 'maxmin', 'asian': fpr_group_losses[2, 0], 'black': fpr_group_losses[2, 1], 'hispanic': fpr_group_losses[2, 2], 'white': fpr_group_losses[2, 3]
        }, ignore_index=True) 
        curr_losses = curr_losses.append({
            'metric': 'fpr', 'soln': 'fullrep', 'asian': fpr_group_losses[1, 0], 'black': fpr_group_losses[1, 1], 'hispanic': fpr_group_losses[1, 2], 'white': fpr_group_losses[1, 3]
        }, ignore_index=True) 

        curr_losses['trial'] = i
        train_losses = train_losses.append(curr_losses, ignore_index=True)

    train_adjusts.to_csv('results_fico/fico__adjust_train.csv', index=False)
    test_adjusts.to_csv('results_fico/fico__adjust_test.csv', index=False)
    # test points modified with train lambdas

    # all train lambdas and train losses
    train_lambdas.to_csv('results_fico/fico__lambdas.csv', index=False)
    train_losses.to_csv('results_fico/fico__losses.csv', index=False)





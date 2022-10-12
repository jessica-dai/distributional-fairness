import argparse
import numpy as np
import pandas as pd

from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset

import sys
sys.path.append('..')

from datasets import get_data

def gen_hardt_preds(seed=0, dataset='adult_old', verbose=False):

    # get probabilities, format
    allscores = get_data(dataset, seed=seed, verb=verbose) # adult_old, adult_new, public, taiwan
    ds_orig = BinaryLabelDataset(df=allscores, label_names=['label'], protected_attribute_names=['group'])
    ds_pred = ds_orig.copy()
    ds_pred.scores = allscores['score'].ravel().reshape(-1, 1)
    ds_pred.labels = (allscores['score'] > 0.5).astype(int).values.reshape(-1, 1) # not true labels, but the predicted decisions

    # split into val and test (each should be ~1/3 of original dataset)
    splitind = int(len(ds_orig.labels.reshape(-1))*0.5)
    val_orig, test_orig = ds_orig.split([splitind], shuffle=False)
    val_pred, test_pred = ds_pred.split([splitind], shuffle=False)

    test_scores = test_pred.convert_to_dataframe()[0] # dataframe with score group label

    # check results for all available constraints
    privileged_groups = [{'group': 1}]
    unprivileged_groups = [{'group': 0}]
    prb = EqOddsPostprocessing(unprivileged_groups, privileged_groups, seed=seed)
    prb.fit(val_orig, val_pred) # fit on val data
    ret = prb.predict(test_pred) # predict using test data 
    test_scores['preds'] = ret.labels # these labels are the ones changed by hardt
    
    return test_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="adult_old") # adult_old, adult_new, public, taiwan
    parser.add_argument("--trials", default="1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(args)

    scores_all_trials = pd.DataFrame(columns=[
        'trial', 'score', 'group', 'label'
    ])

    for i in range(int(args.trials)):
        if args.verbose:
            print("========= running trial", i + 1, "of", args.trials, "=========")
        scores = gen_hardt_preds(seed=i, dataset=args.dataset, verbose=args.verbose)

        scores['trial'] = i

        scores_all_trials = scores_all_trials.append(scores, ignore_index=True)

    scores_all_trials.to_csv(args.dataset + '_results/hardt__preds.csv', index=False)
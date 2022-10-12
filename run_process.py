from unittest import result
import numpy as np
import pandas as pd

import argparse

from eval_helpers import get_eval_single 
from src.exact_solver import get_tpr, get_fpr, get_eo_1, get_eo_2

def _get_lambdas(traindf):

    lambdas = {
        'orig': 0,
        'full': 1,
        'tpr': get_tpr(traindf, 'score', 'group', 'adjust'),
        'fpr': get_fpr(traindf, 'score', 'group', 'adjust'), 
        'eo_1': get_eo_1(traindf, 'score', 'group', 'adjust'),
        'eo_2': get_eo_2(traindf, 'score', 'group', 'adjust')
    }

    return lambdas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="adult_old")
    parser.add_argument("--algo", default="rf")
    parser.add_argument("--trials", default=1)
    args = parser.parse_args()

    train_df = pd.read_csv('results/' + args.data + '_' + args.algo + '__adjust_train.csv')
    test_df = pd.read_csv('results/' + args.data + '_' + args.algo + '__adjust_test.csv')

    resultdf = pd.DataFrame(columns = ["thresholds",
                    "tpr_A",
                    "tpr_B",
                    "selection_A",
                    "selection_B",
                    "fpr_A",
                    "fpr_B",
                    "trial",
                    "lambda"
            ])

    lambdadf = pd.DataFrame(columns=["trial", "tpr", "fpr", "eo_1", "eo_2"])

    for i in range(int(args.trials)):

        currtrain = train_df.loc[train_df.trial == i]
        currtest = test_df.loc[test_df.trial == i]

        # get and save lambdas per trial using training data
        trial_lambdas = _get_lambdas(currtrain)
        currlambdas = pd.DataFrame({k : [trial_lambdas[k]] for k in ['tpr', 'fpr', 'eo_1', 'eo_2']})
        currlambdas['trial'] = [i]
        lambdadf = pd.concat([lambdadf, currlambdas], ignore_index=True)



        # apply lambdas to test data
        for lmbd in trial_lambdas: 
            curreval = get_eval_single(currtest.label, currtest.score + trial_lambdas[lmbd]*currtest.adjust, currtest.group)
            currres = curreval[["thresholds",
                    "tpr_A",
                    "tpr_B",
                    "selection_A",
                    "selection_B",
                    "fpr_A",
                    "fpr_B",
                    "acc_overall"]]
            currres['eqodds_A'] = 1 - currres.tpr_A + currres.fpr_A
            currres['eqodds_B'] = 1 - currres.tpr_B + currres.fpr_B
            currres['lambda'] = lmbd 
            currres['trial'] = i 

            resultdf = pd.concat([resultdf, currres], ignore_index=True)

    lambdadf.to_csv('results/' + args.data + '_' + args.algo + '__lambdas.csv', index=False)
    resultdf.to_csv('results/' + args.data + '_' + args.algo + '__evalthresholds.csv', index=False)
            
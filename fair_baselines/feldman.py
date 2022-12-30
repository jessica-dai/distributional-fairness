import argparse
from random import random
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from aif360.algorithms.preprocessing import DisparateImpactRemover

import sys
sys.path.append('..')

from datasets import gen_adult_probs, gen_new_adult, _get_model
from eval_helpers import get_eval_single

"""
Feldman CAN handle non-binary sensitive attributes. 
https://github.com/algofairness/BlackBoxAuditing

Run `python feldman.py` for baseline experiments. 
"""

def gen_feldman_probs(seed=0, dataset='adult_old', algo = 'lr', repairlevels=np.linspace(0.1,1,10), verbose=False):

    if dataset == 'adult_old':
        ad = gen_adult_probs(seed=seed, interv='pre/in')
    elif dataset == 'adult_new':
        ad = gen_new_adult(seed=seed, verb=verbose, task='income', interv=True)
    # elif dataset == 'public':
    #     ad = gen_new_adult(seed=seed, verb=verbose, task='public', interv=True)
    # elif dataset == 'taiwan':
    #     ad = gen_taiwan_credit(seed=seed, verb=verbose, interv=True)

    train, val, test = ad.split(3, shuffle=False) # train for training classifier, val for picking repair level, test for getting results

    all_repairlevels = pd.DataFrame(columns=['repairlevel', 'avg_prd', 'avg_tprd', 'avg_eod', 'avg_acc'])

    if verbose:
        print("Checking all repair levels")

    for repairlevel in repairlevels:

        if verbose:
            print("Checking repair level", repairlevel)

        di = DisparateImpactRemover(repair_level=repairlevel)
        
        train_repd = di.fit_transform(train)
        val_repd = di.fit_transform(val)

        x_tr = train_repd.features
        x_val = val_repd.features

        # x_tr = np.delete(train_repd.features, sensind, axis=1)
        # x_val = np.delete(val_repd.features, sensind, axis=1)
        y_tr = train_repd.labels.ravel()
        y_val = val_repd.labels.ravel()

        classifier = _get_model(algo, seed=seed).fit(x_tr, y_tr)

        probs = classifier.predict_proba(x_val)[:,1]
        results = get_eval_single(y_val,probs,val.protected_attributes.reshape(-1))

        # get results for threshold = 0.5
        ind05 = results.loc[results ['thresholds'] == 0.5] 
        prd05 = ind05['positivity_rate_differences']
        acc05 = ind05['acc_overall']
        tprd05 = ind05['tpr_differences']
        fprd05 = ind05['fpr_differences']

        resultdict = {
            'repairlevel': repairlevel,
            'avg_prd': np.mean(np.abs(results['positivity_rate_differences'])),
            'avg_tprd': np.mean(np.abs(results['tpr_differences'])),
            'avg_eod': np.mean(results['eqodds_differences']),
            'avg_acc': np.mean(results['acc_overall']), 
            '05_prd': prd05.values[0],
            '05_tprd': tprd05.values[0],
            '05_eod': (np.abs(tprd05) + np.abs(fprd05)).values[0],
            '05_acc': acc05.values[0]
        }

        all_repairlevels = pd.concat([all_repairlevels, pd.DataFrame({k: [resultdict[k]] for k in resultdict})], ignore_index=True)

    if verbose:
        print("Checked all repair levels, saving best results")

    # then find the best repair level for each and return the probabilities
    best_repairlevels = {}
    metr_probs = {}
    for metr in ['pr', 'tpr', 'eo', 'acc']:
        if metr == 'acc':
            best_repairlevels[metr] = all_repairlevels['repairlevel'][all_repairlevels['avg_' + metr].argmin()]

        else:
            best_repairlevels[metr] = all_repairlevels['repairlevel'][all_repairlevels['avg_' + metr +'d'].argmin()]

        # use that repairlevel for unseen test data
        repairer = DisparateImpactRemover(repair_level=best_repairlevels[metr])
        train_repd = repairer.fit_transform(train)
        test_repd = repairer.fit_transform(test)

        x_tr = train_repd.features
        x_te = test_repd.features

        # x_tr = np.delete(train_repd.features, sensind, axis=1)
        # x_te = np.delete(test_repd.features, sensind, axis=1)
        y_tr = train_repd.labels.ravel()
        y_te = test_repd.labels.ravel()
        classifier = _get_model(algo, seed=seed).fit(x_tr, y_tr)

        probs = classifier.predict_proba(x_te)[:,1]
        metr_probs[metr] = probs
        
    repaired_scores = pd.DataFrame({
        'scores_pr': metr_probs['pr'], 
        'scores_tpr': metr_probs['tpr'], 
        'scores_eo': metr_probs['eo'], 
        'scores_acc': metr_probs['acc'], 
        'label': y_te, 
        'group': test_repd.protected_attributes.reshape(-1)
    })

    return repaired_scores, best_repairlevels, all_repairlevels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="adult_old") # adult_old, adult_new, public, taiwan
    parser.add_argument("--algo", default="lr") # for body of the paper, adult_old lr and adult_new svm
    parser.add_argument("--trials", default="1", help="number of trials to run")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(args)
    scores_all_trials = pd.DataFrame(columns=[
        'trial', 'scores_pr', 'scores_tpr', 'scores_eo', 'scores_acc', 'labels', 'groups'
    ])
    bestlevels_all_trials = pd.DataFrame(columns=[
        'trial', 'pr', 'tpr', 'eo', 'acc'
    ])
    overlevels_all_trials = pd.DataFrame(columns=[
        'trial', 'repairlevel', 'avg_prd', 'avg_tprd', 'avg_eod', 'avg_acc', '05_acc', '05_eod', '05_prd', '05_tprd'
    ])

    for i in range(int(args.trials)):
        if args.verbose:
            print("========= running trial", i + 1, "of", args.trials, "=========")
        # 10.22 added dataset argument
        scores, bestlevels, all_levels = gen_feldman_probs(seed=i, dataset=args.dataset, algo=args.algo, repairlevels=np.linspace(0,1,21), verbose=args.verbose)

        scores['trial'] = i
        bestlevels['trial'] = i
        all_levels['trial'] = i

        scores_all_trials = scores_all_trials.append(scores, ignore_index=True)
        bestlevels_all_trials = bestlevels_all_trials.append(bestlevels, ignore_index=True)
        overlevels_all_trials = overlevels_all_trials.append(all_levels, ignore_index=True)

    scores_all_trials.to_csv(args.dataset + '_results/feldman_' + args.algo + '__scores.csv', index=False)
    bestlevels_all_trials.to_csv(args.dataset + '_results/feldman_' + args.algo + '__bestlevels.csv', index=False)
    overlevels_all_trials.to_csv(args.dataset + '_results/feldman_' + args.algo + '__overlevels.csv', index=False)
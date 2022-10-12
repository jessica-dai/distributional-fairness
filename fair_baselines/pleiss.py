import argparse
import numpy as np
import pandas as pd

from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.metrics import utils
from aif360.datasets import BinaryLabelDataset

import sys
sys.path.append('..')

from datasets import get_data

class ProbsCalibratedEqOddsPostprocessing(CalibratedEqOddsPostprocessing):
    
    def test_print(self):
        print(self.cost_constraint)
    
    def predict_proba(self, dataset):
        if self.seed is not None:
            np.random.seed(self.seed)

        cond_vec_priv = utils.compute_boolean_conditioning_vector(
            dataset.protected_attributes,
            dataset.protected_attribute_names,
            self.privileged_groups)
        cond_vec_unpriv = utils.compute_boolean_conditioning_vector(
            dataset.protected_attributes,
            dataset.protected_attribute_names,
            self.unprivileged_groups)

        unpriv_indices = (np.random.random(sum(cond_vec_unpriv))
                       <= self.unpriv_mix_rate)
        unpriv_new_pred = dataset.scores[cond_vec_unpriv].copy()
        unpriv_new_pred[unpriv_indices] = self.base_rate_unpriv

        priv_indices = (np.random.random(sum(cond_vec_priv))
                     <= self.priv_mix_rate)
        priv_new_pred = dataset.scores[cond_vec_priv].copy()
        priv_new_pred[priv_indices] = self.base_rate_priv

        dataset_new = dataset.copy(deepcopy=True)

        dataset_new.scores_new = np.zeros_like(dataset.scores, dtype=np.float64)
        dataset_new.scores_new[cond_vec_priv] = priv_new_pred
        dataset_new.scores_new[cond_vec_unpriv] = unpriv_new_pred

        # # Create labels from scores using a default threshold
        # dataset_new.labels = np.where(dataset_new.scores >= threshold,
        #                               dataset_new.favorable_label,
        #                               dataset_new.unfavorable_label)
        return dataset_new

def gen_pleiss_probs(seed=0, pleiss_thresh=0.5, dataset='adult', verbose=False):

    # get probabilities, format
    allscores = get_data(dataset, seed=seed, verb=verbose) # adult_old, adult_new, public, taiwan
    ds_orig = BinaryLabelDataset(df=allscores, label_names=['label'], protected_attribute_names=['group'])
    ds_pred = ds_orig.copy()
    ds_pred.scores = allscores['score'].ravel().reshape(-1, 1)
    ds_pred.labels = (allscores['score'] > pleiss_thresh).astype(int).values.reshape(-1, 1) # not true labels, but the predicted decisions

    # split into val and test (each should be ~1/3 of original dataset)
    splitind = int(len(ds_orig.labels.reshape(-1))*0.5)
    val_orig, test_orig = ds_orig.split([splitind], shuffle=False)
    val_pred, test_pred = ds_pred.split([splitind], shuffle=False)

    test_scores = test_pred.convert_to_dataframe()[0] # dataframe with score group label

    # check results for all available constraints
    privileged_groups = [{'group': 1}]
    unprivileged_groups = [{'group': 0}]
    for constraint in ['fpr', 'fnr', 'weighted']:
        prb = ProbsCalibratedEqOddsPostprocessing(unprivileged_groups, privileged_groups, cost_constraint=constraint)
        prb.fit(val_orig, val_pred) # fit on val data
        ret = prb.predict_proba(test_pred) # predict using test data 
        test_scores[constraint] = ret.scores_new
    
    method_to_name_map = {
        'fpr': 'scores_fpr',
        'fnr': 'scores_tpr',
        'weighted': 'scores_eo'
    }

    test_scores.rename(columns=method_to_name_map, inplace=True)

    return test_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="adult_old") # adult_old, adult_new, public, taiwan
    parser.add_argument("--trials", default="1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(args)

    scores_all_trials = pd.DataFrame(columns=[
        'trial', 'scores_fpr', 'scores_tpr', 'scores_eo'
    ])

    for i in range(int(args.trials)):
        if args.verbose:
            print("========= running trial", i + 1, "of", args.trials, "=========")
        scores = gen_pleiss_probs(seed=i, dataset=args.dataset, verbose=args.verbose)

        scores['trial'] = i

        scores_all_trials = scores_all_trials.append(scores, ignore_index=True)

    scores_all_trials.to_csv(args.dataset + '_results/pleiss__scores.csv', index=False)
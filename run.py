import numpy as np
import pandas as pd

import argparse

from pandas.core.indexes import base

from data import get_data
from eval import get_eval_single, auc_result_summary
from regression_wasserstein.regression_postprocess import f_postprocess_vectorized_adjustments

# OVER THRESHOLDS
eval_columns = [
                "positivity_rate_differences",
                "tpr_differences",
                "fpr_differences",
                "eqodds_differences",
                "tpr_A",
                "tpr_B",
                "selection_A",
                "selection_B", 
                "fpr_A",
                "fpr_B",
                "acc_A",
                "acc_B",
                "acc_overall"]


def print_summary(data):
    print("P(Y=1 | G): ", np.mean(data.loc[data.group == 0].score), np.mean(data.loc[data.group == 1].score))
    print("percent group 0: ", np.mean(data.group == 0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",default="adult_old",help="which dataset to use")
    parser.add_argument("--trials", default="1", help="number of trials to run")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--lambdas", action="store_true")
    args = parser.parse_args()

    print(args)

    adjusts = pd.DataFrame(columns=['trial', 'adjust', 'score', 'label', 'group'])

    if args.baselines:
        baseline_evals = pd.DataFrame(columns=eval_columns + ["thresholds", "trial"])
        fulladj_evals = pd.DataFrame(columns=eval_columns + ["thresholds", "trial"])
    if args.lambdas:
        over_weights = pd.DataFrame(columns=eval_columns + ["adjust_weight", "trial"])

    for i in range(int(args.trials)):
        print()
        print("====== Trial", i + 1, "of", args.trials, "======")

        print("**Loading %s dataset**" % args.data)
        if i == 0: # only print information if it's the first run
            probs = get_data(args.data, seed=0, verb=True)
            print_summary(probs)
        else:
            probs = get_data(args.data, seed=i, verb=False)

        print("**Splitting dataset and postprocessing**")
        split_ind = int(len(probs)/2)
        if i == 0:
            print(split_ind, "examples in train and test")
        train = probs.iloc[:split_ind]
        test = probs.iloc[split_ind:]
        curr_trial = np.array([i]*len(test))
        adjust = f_postprocess_vectorized_adjustments(train.score.values.copy(), test.score.values.copy(), train.group.values.copy(), test.group.values.copy())

        test['trial'] = curr_trial 
        test['adjust'] = adjust
        adjusts = adjusts.append(test, ignore_index=True)

        if args.baselines:
            print("**Getting baseline and full evals**")
            curr_trial = np.array([i]*101)
            baseline = get_eval_single(test.label.values, test.score.values, test.group.values)
            baseline["trial"] = curr_trial
            full = get_eval_single(test.label.values, test.score.values + adjust, test.group.values)
            full["trial"] = curr_trial

            baseline_evals = baseline_evals.append(baseline, ignore_index=True)
            fulladj_evals = fulladj_evals.append(full, ignore_index=True)

        if args.lambdas:
            print("**Varying values of lambda**")
            curr_trial = np.array([i]*101)
            weights = np.linspace(0,1,101)
            weightaucs = pd.DataFrame(columns= eval_columns + ['adjust_weight'])
            for weight in weights:
                adjust_weighted = np.multiply(weight, adjust)
                scores_weighted = test.copy()
                scores_weighted['score'] = test.score.values + adjust_weighted

                res = get_eval_single(scores_weighted.label, scores_weighted.score, scores_weighted.group)
                aucs = auc_result_summary(res)
                aucs['adjust_weight'] = weight 
            
                weightaucs = weightaucs.append(aucs, ignore_index=True)
            weightaucs["trial"] = curr_trial
            over_weights = over_weights.append(weightaucs, ignore_index=True)
    
    if args.baselines:
        baseline_evals.to_csv("results/"+ args.data +"__baseline.csv", index=False)
        fulladj_evals.to_csv("results/"+ args.data +"__fulladj.csv", index=False)
    if args.lambdas:
        over_weights.to_csv("results/"+ args.data +"__overweights.csv", index=False)

    adjusts.to_csv('results/' + args.data + "__adjustments.csv", index=False)
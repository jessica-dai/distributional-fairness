from tokenize import group

import datasets
import numpy as np
from .bcmap import *

from scipy.optimize import minimize_scalar


import datasets
import numpy as np
from src.bcmap import *

from scipy.optimize import minimize_scalar

def _eoRates(scores,bins, n=100):
    return _positiveRates(scores,bins, n) + _negativeRates(scores, bins, n)

def _positiveRates(scores, bins, n=100):
    cdf = empiricalCDF(scores,bins)
    return np.array([(cdf >= tau).astype(np.int).mean() for tau in np.linspace(0,1,n+1)])

def _negativeRates(scores, bins, n=100):
    cdf = empiricalCDF(scores,bins)
    return np.array([(cdf <= tau).astype(np.int).mean() for tau in np.linspace(0,1,n+1)])

def _rate_diff(rates_1, rates_2):
    return np.abs(rates_1 - rates_2).mean()

def _uniform_diff(group_a, group_b, score_col, shift_col, bins, sign):
    return (lambda x: _uniform_diff_help(x, group_a, group_b, score_col, shift_col, bins, sign))

def _uniform_diff_help(lam, group_a, group_b, score_col, shift_col, bins, sign):
    a_scores = group_a[score_col].to_numpy()
    a_shift = group_a[shift_col].to_numpy()

    b_scores = group_b[score_col].to_numpy()
    b_shift = group_b[shift_col].to_numpy()

    a_repair = a_scores + np.multiply(a_shift, lam)
    b_repair = b_scores + np.multiply(b_shift, lam)

    if sign.lower() == "positive":
        a_pr_rates = _positiveRates(a_repair, bins=bins)
        b_pr_rates = _positiveRates(b_repair, bins=bins)

        assert len(a_pr_rates) == len(b_pr_rates), "sanity check to make sure there are an equal number of rates"
        return _rate_diff(a_pr_rates, b_pr_rates)
    elif sign.lower() == "negative":
        a_nr_rates = _positiveRates(a_repair, bins=bins)
        b_nr_rates = _positiveRates(b_repair, bins=bins)

        assert len(a_nr_rates) == len(b_nr_rates), "sanity check to make sure there are an equal number of rates"
        return _rate_diff(a_nr_rates, b_nr_rates)

    elif sign.lower() == "eo_v2":
        a_eo_rates = _eoRates(a_repair, bins=bins)
        b_eo_rates = _eoRates(b_repair, bins=bins)
        # a_pr = _positiveRates(a_repair, bins=bins)
        # b_pr = _positiveRates(b_repair, bins=bins)

        # a_nr = _negativeRates(a_repair, bins=bins)
        # b_nr = _negativeRates(b_repair, bins=bins)

        assert len(a_eo_rates) == len(b_eo_rates), "sanity check to make sure there are an equal number of rates"
        return _rate_diff(a_eo_rates, b_eo_rates)
    else:
        return None


def _objective(df, score_col, group_col, shift_col, bins, metrics, weights):
        metric_fns = []
        for m in metrics:
            groups = [g[1] for g in df.groupby(group_col)]
            assert len(groups) == 2, "can only support binary protected groups"

            group_a, group_b = groups

            if m.lower() == "pr":
                diff_fn = _uniform_diff(
                    group_a=group_a,
                    group_b=group_b,
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="positive")
                metric_fns.append(diff_fn)
            if m.lower() == "nr":
                diff_fn = _uniform_diff(
                    group_a=group_a,
                    group_b=group_b,
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="negative")
                metric_fns.append(diff_fn)
            elif m.lower() == "tpr":
                diff_fn = _uniform_diff(
                    group_a=group_a[group_a["label"]==1],
                    group_b=group_b[group_b["label"]==1],
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="positive")
                metric_fns.append(diff_fn)
            elif m.lower() == "fpr":
                diff_fn = _uniform_diff(
                    group_a=group_a[group_a["label"]==0],
                    group_b=group_b[group_b["label"]==0],
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="positive")
                metric_fns.append(diff_fn)
            elif m.lower() == "tnr":
                diff_fn = _uniform_diff(
                    group_a=group_a[group_a["label"]==1],
                    group_b=group_b[group_b["label"]==1],
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="negative")
                metric_fns.append(diff_fn)
            elif m.lower() == "fnr":
                diff_fn = _uniform_diff(
                    group_a=group_a[group_a["label"]==0],
                    group_b=group_b[group_b["label"]==0],
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="negative")
                metric_fns.append(diff_fn)
            
            elif m.lower() == "eo_2":
                """
                not used last time
                """
                diff_fn = _uniform_diff(
                    group_a=group_a[group_a["label"]==1],
                    group_b=group_b[group_b["label"]==1],
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="eo_v2")
                metric_fns.append(diff_fn)

            elif m.lower() == "eo_1":

                diff_fn = _uniform_diff(
                    group_a=group_a[group_a["label"]==1],
                    group_b=group_b[group_b["label"]==1],
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="positive")
                metric_fns.append(diff_fn)
                diff_fn = _uniform_diff(
                    group_a=group_a[group_a["label"]==0],
                    group_b=group_b[group_b["label"]==0],
                    score_col=score_col,
                    shift_col=shift_col,
                    bins=bins,
                    sign="positive")
                metric_fns.append(diff_fn)

        return (lambda x: np.array([metric_fns[i](x)*weights[i] for i in range(len(metric_fns))]).sum())
        

def _test():
    df = datasets.gen_adult_probs().sample(frac=1)
    n = len(df)
    alpha = .2
    split_ix = int(n * alpha)

    df_train = df.iloc[:split_ix]
    df_test = df.iloc[split_ix:]

    bins = np.linspace(0, 1, 101)
    repaired, _ = geometric_adjustment(train_df=df_train,
                                        test_df=df_test,
                                        sens_col="group",
                                        score_col="score",
                                        solver="exact_LP",
                                        bins=bins,
                                        return_barycenter=True)
    #TPR
    obj_fun_tpr = objective(df=repaired,
                        score_col="score",
                        group_col="group",
                        shift_col="shift",
                        bins=bins,
                        metrics=["TPR"],
                        weights=[1])



    #FPR+FPR
    obj_fun_eq_odd = objective(df=repaired,
                        score_col="score",
                        group_col="group",
                        shift_col="shift",
                        bins=bins,
                        metrics=["TPR", "FPR"],
                        weights=[.5, .5])

    #FPR
    obj_fun_fpr = objective(df=repaired,
                               score_col="score",
                               group_col="group",
                               shift_col="shift",
                               bins=bins,
                               metrics=["FPR"],
                               weights=[1])

    #PR
    obj_fun_eq_pr = objective(df=repaired,
                               score_col="score",
                               group_col="group",
                               shift_col="shift",
                               bins=bins,
                               metrics=["PR"],
                               weights=[1])



    res = minimize_scalar(obj_fun_eq_pr, bounds=(0, 1.01), method='Golden')
    print(res.x)

    res = minimize_scalar(obj_fun_tpr, bounds=(0, 1.01), method='Golden')
    print(res.x)

    res = minimize_scalar(obj_fun_fpr, bounds=(0, 1.01), method='Golden')
    print(res.x)

    res = minimize_scalar(obj_fun_eq_odd, bounds=(0, 1.01), method='Golden')
    print(res.x)


def get_fpr(repaired_df, score_col, group_col, shift_col):

    obj_fun_fpr = _objective(df=repaired_df,
                               score_col=score_col,
                               group_col=group_col,
                               shift_col=shift_col,
                               bins = np.linspace(0, 1, 101),
                               metrics=["FPR"],
                               weights=[1])

    return round(minimize_scalar(obj_fun_fpr, bounds=(0, 1.01), method='Golden').x, 2)

def get_tpr(repaired_df, score_col, group_col, shift_col):

    obj_fun_tpr = _objective(df=repaired_df,
                               score_col=score_col,
                               group_col=group_col,
                               shift_col=shift_col,
                               bins = np.linspace(0, 1, 101),
                               metrics=["TPR"],
                               weights=[1])

    return round(minimize_scalar(obj_fun_tpr, bounds=(0, 1.01), method='Golden').x, 2)

def get_eo_1(repaired_df, score_col, group_col, shift_col):

    obj_fun_eo = _objective(df=repaired_df,
                            score_col=score_col,
                            group_col=group_col,
                            shift_col=shift_col,
                            bins=np.linspace(0,1,101),
                            metrics=["eo_1"],
                            weights=[0.5, 0.5]
                            )

    return round(minimize_scalar(obj_fun_eo, bounds=(0,1.01), method='Golden').x, 2)

def get_eo_2(repaired_df, score_col, group_col, shift_col):
    obj_fun_eo = _objective(df=repaired_df,
                            score_col=score_col,
                            group_col=group_col,
                            shift_col=shift_col,
                            bins=np.linspace(0,1,101),
                            metrics=["eo_2"],
                            weights=[1]
                            )

    return round(minimize_scalar(obj_fun_eo, bounds=(0,1.01), method='Golden').x, 2)
    
# main()
import numpy as np
import pandas as pd

from fairlearn.metrics import selection_rate, true_positive_rate, false_positive_rate


def get_auc(x_int, y):
    """
    x_int : interval
    """
    
    heights = np.sum(y) + 0.5*(y[0] + y[-1])

    return heights*x_int

def auc_result_summary(resultdf):
    aucs = {k: get_auc(0.01, resultdf[k].values) for k in resultdf}
    aucs['abs_pos'] = get_auc(0.01, np.abs(resultdf['positivity_rate_differences'].values))
    aucs['abs_tpr'] = get_auc(0.01, np.abs(resultdf['tpr_differences'].values))
    aucs['abs_fpr'] = get_auc(0.01, np.abs(resultdf['fpr_differences'].values))
    return aucs

def get_eval_single(labels, scores, groups):
    """
    For a single trial
    """
    tpr_A = []
    tpr_B = []
    selection_A = []
    selection_B = []
    fpr_A = []
    fpr_B = []
    accuracies_overall = []
    accuracy_A = []
    accuracy_B = []
    thresholds = np.linspace(0,1,101)
    labels = labels.astype(int)


    for threshold in thresholds:
      preds = scores > threshold

      selection_A.append(
          selection_rate(labels[groups == 1], preds[groups == 1])
      )
      selection_B.append(
          selection_rate(labels[groups == 0], preds[groups == 0])
      )

      tpr_A.append(
          true_positive_rate(labels[groups == 1], preds[groups == 1])
      )
      tpr_B.append(
          true_positive_rate(labels[groups == 0], preds[groups == 0])
      )
      fpr_A.append(
          false_positive_rate(labels[groups == 1], preds[groups == 1])
      )
      fpr_B.append(
          false_positive_rate(labels[groups == 0], preds[groups == 0])
      )

      accuracies_overall.append(
          np.mean(labels == preds)
      )
      accuracy_A.append(
          np.mean(labels[groups == 1] == preds[groups == 1])
      )
      accuracy_B.append(
          np.mean(labels[groups == 0] == preds[groups == 0])
      )
    
    selection_A = np.array(selection_A)
    selection_B = np.array(selection_B)
    tpr_A = np.array(tpr_A)
    tpr_B = np.array(tpr_B)
    fpr_A = np.array(fpr_A)
    fpr_b = np.array(fpr_B)
    return pd.DataFrame({"thresholds":thresholds,
                         "positivity_rate_differences": selection_A - selection_B,
                         "tpr_differences": tpr_A - tpr_B,
                         "fpr_differences": fpr_A - fpr_B,
                         "eqodds_differences": np.abs(tpr_A - tpr_B) + np.abs(fpr_A - fpr_B),
                         "tpr_A": tpr_A,
                         "tpr_B": tpr_B,
                         "selection_A": selection_A,
                         "selection_B": selection_B, 
                         "fpr_A": fpr_A,
                         "fpr_B": fpr_B,
                         "acc_A": accuracy_A,
                         "acc_B": accuracy_B,
                         "acc_overall": accuracies_overall
                        })
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse 

from eval_helpers import get_abs_ci

metric_to_lambda = {
    'selection' : ['full', 'orig'], 
    'tpr' : ['tpr', 'orig'],
    'fpr' : ['fpr', 'orig'],
    # 'eqodds': ['eo_1', 'eo_2', 'orig']
    'eqodds': ['eo_1', 'orig']
}
metric_to_title = {
    'selection': 'Demographic Parity',
    'tpr': 'Equal Opportunity', 
    'fpr': 'FPR',
    'eqodds': 'Equalized Odds', 
    'orig': 'Unrepaired',
    'eo_1': 'Equalized Odds', 
    'full': 'Full Repair'
}
metric_to_gamma = {
  'selection': '$\gamma$ = PR',
  'tpr': '$\gamma$ = TPR', 
  'eqodds': '$\gamma$ = FNR + FPR',
  'fpr': 'FPR'
}
datasets = {
    'adult_old': "Income (S)", 
    'adult_new': "Income (R)", 
    'taiwan': "Taiwan Credit", 
    'public': "Public Coverage"
}
algos = {
    'lr': "LR",
    'svm': 'SVM', 
    'rf': 'RF', 
    'mlp': 'MLP'
}

# resultdf is the output of `get_eval`

def _plot_result(resultdf, 
                metrics=['selection_A', 'selection_B'],
                filters={}, # e.g. {'lambda': 'eo_1'}
                legend_map={}, # e.g. {'selection_A': 'Positive Rate Group A'}
                ylabel="",
                title=None, size=20, filename=None):

  if "trial" not in resultdf.columns:
    print("Expected multiple trials! Plotting for the single trial:")

  plt.figure(figsize=(10, 5))
  for flt in filters:
    filtdf = resultdf.loc[resultdf[flt] == filters[flt]]

  colors = ["cornflowerblue", "firebrick"]
  counter = 0
  for metric in metrics:
    if len(legend_map.keys()) < len(metrics):
      legend_map[metric] = None 
    ycol = np.abs(filtdf[metric])
    color = colors[counter] if counter < 2 else None
    ax = sns.lineplot(x=filtdf.thresholds, y=ycol, label=legend_map[metric], color=color)
    counter +=1 

  if legend_map[metrics[0]]: # hacky
    plt.legend(fontsize=int(size*0.8))
  plt.ylim((0,1))
  plt.xticks(fontsize=size*0.7)
  plt.yticks(fontsize=size*0.7)
  ax.set_xlabel(r'Threshold $\tau$', fontsize=int(size*0.9))
  ax.set_ylabel(ylabel, fontsize=int(size*0.9))
  ax.spines.top.set_visible(False)
  ax.spines.right.set_visible(False)
  
  if title:
    plt.title(title, size=size)  

  if filename:
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, format='pdf')

  plt.close()
  ax.clear()

def _plot_ROC_curve(resultdf,
                    filters={},
                    ab_labels =[], # this is so sus i'm sorry
                    title=None, size=20, filename=None
                  ):
  """
  this very much violates DRY. i'm sorry
  """

  plt.figure(figsize=(10,5))
  for flt in filters:
    filtdf = resultdf.loc[resultdf[flt] == filters[flt]]

  # todo fix captions
  ax = sns.lineplot(x=filtdf.fpr_A, y=filtdf.tpr_A, label=ab_labels[0], color="cornflowerblue")
  ax = sns.lineplot(x=filtdf.fpr_B, y=filtdf.tpr_B, label=ab_labels[1], color="firebrick")

  plt.legend(fontsize=int(size*0.8))
  plt.xlim((0,1))
  plt.ylim((0,1))
  plt.xticks(fontsize=size*0.7)
  plt.yticks(fontsize=size*0.7)
  ax.set_xlabel(r'FPR', fontsize=int(size*0.9))
  ax.set_ylabel(r'TPR', fontsize=int(size*0.9))
  ax.spines.top.set_visible(False)
  ax.spines.right.set_visible(False)
  
  if title:
    plt.title(title, size=size)  

  if filename:
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, format='pdf')

  plt.close()
  ax.clear()

###### OLD BELOW 
def plot_result_summary(resultdf, filename=None): 
  plt.figure(figsize=(25, 10))
  plt.subplot(2,3,1)
  plot_ROC_curve(resultdf)
  plt.subplot(2,3,2)
  _plot_result(resultdf)
  plt.subplot(2,3,3)
  _plot_result(resultdf, ['acc_A', 'acc_B', 'acc_overall'])
  plt.subplot(2,3,4)
  _plot_result(resultdf, ['selection_A', 'selection_B'])
  plt.subplot(2,3,5)
  _plot_result(resultdf, ['tpr_A', 'tpr_B'])
  plt.subplot(2,3,6)
  _plot_result(resultdf, ['fpr_A', 'fpr_B'])

  if filename:
    plt.savefig(filename + '.png')

def lambda_aucs_abs(wts, filename=None): # FIG 2 
  sns.lineplot(data=wts, x="adjust_weight", y=[0]*1010, label="zero area between curves", linestyle='dashed', color='darkgray')
  sns.lineplot(data=wts, x="adjust_weight", y="abs_pos", label="area between selection rate curves", color='indianred')
  sns.lineplot(data=wts, x="adjust_weight", y="abs_tpr", label="area between TPR curves", color='mediumseagreen')
  ax = sns.lineplot(data=wts, x="adjust_weight", y="abs_fpr", label="area between FPR curves", color='slateblue')

  plt.rcParams["axes.labelsize"] = 13
  plt.rcParams["axes.titlesize"] = 17
  ax.set(xlabel="adjustment weight $\\lambda$")
  ax.set(ylabel="Area between curves (absolute)")
  ax.set(title="Absolute area between curves over $\\lambda$")

  if filename:
    plt.savefig(filename + '_auc_abs.png')

def lambda_aucs_signed(wts, filename=None): # FIG 2 
  sns.lineplot(data=wts, x="adjust_weight", y=[0]*1010, label="zero area between curves", linestyle='dashed', color='darkgray')
  sns.lineplot(data=wts, x="adjust_weight", y='positivity_rate_differences', label="area between selection rate curves", color='indianred')
  sns.lineplot(data=wts, x="adjust_weight", y="tpr_differences", label="area between TPR curves", color='mediumseagreen')
  ax = sns.lineplot(data=wts, x="adjust_weight", y="fpr_differences", label="area between FPR curves", color='slateblue')

  plt.rcParams["axes.labelsize"] = 13
  plt.rcParams["axes.titlesize"] = 17
  ax.set(xlabel="adjustment weight $\\lambda$")
  ax.set(ylabel="Area between curves (signed)")
  ax.set(title="Signed area between curves over $\\lambda$")

  if filename:
    plt.savefig(filename + '_auc_signed.png')

def lambda_acc(wts, filename=None):
  sns.lineplot(data=wts, x="adjust_weight", y="acc_overall", label="overall")
  sns.lineplot(data=wts, x="adjust_weight", y="acc_A", label="A")
  ax = sns.lineplot(data=wts, x="adjust_weight", y="acc_B", label="B")

  plt.rcParams["axes.labelsize"] = 13
  plt.rcParams["axes.titlesize"] = 17
  ax.set(xlabel="adjustment weight $\\lambda$")
  ax.set(ylabel="Accuracy")
  ax.set(title="Average overall accuracy over $\\lambda$")

  plt.savefig(filename + '_acc.png')

def lambda_auc_full(wts, title=None, hide_key=True):
  if not hide_key:
    sns.lineplot(data=wts, x="adjust_weight", y=[0]*1010, label="zero area between curves", linestyle='dashed', color='darkgray')
    sns.lineplot(data=wts, x="adjust_weight", y="abs_pos", label="area between selection rate curves", color='indianred')
    sns.lineplot(data=wts, x="adjust_weight", y="abs_tpr", label="area between TPR curves", color='mediumseagreen')
    sns.lineplot(x=wts.adjust_weight, y=wts.abs_tpr + wts.abs_fpr, label="area between TPR & FPR curves", color='cadetblue')
    ax = sns.lineplot(data=wts, x="adjust_weight", y="abs_fpr", label="area between FPR curves", color='slateblue')

  else: # this is horrible but i gave up on fighting matplotlib
    sns.lineplot(data=wts, x="adjust_weight", y=[0]*1010, linestyle='dashed', color='darkgray')
    sns.lineplot(data=wts, x="adjust_weight", y="abs_pos", color='indianred')
    sns.lineplot(data=wts, x="adjust_weight", y="abs_tpr", color='mediumseagreen')
    sns.lineplot(x=wts.adjust_weight, y=wts.abs_tpr + wts.abs_fpr, color='cadetblue')
    ax = sns.lineplot(data=wts, x="adjust_weight", y="abs_fpr", color='slateblue')


  tpr_ci = get_abs_ci(wts, 'abs_tpr')
  tpr = np.mean([wts.iloc[101*i + int(np.mean(tpr_ci)*100)]['abs_tpr'] for i in range(10)])
  plt.plot(np.mean(tpr_ci), tpr, 'd', color='seagreen')
  sns.lineplot(x=tpr_ci, y=tpr, color='seagreen', linestyle='dashed')

  fpr_ci = get_abs_ci(wts, 'abs_fpr')
  fpr = np.mean([wts.iloc[101*i + int(np.mean(fpr_ci)*100)]['abs_fpr'] for i in range(10)])
  plt.plot(np.mean(fpr_ci), fpr, 'd', color='darkslateblue')
  sns.lineplot(x=fpr_ci, y=fpr, color='darkslateblue', linestyle='dashed')

  eo_ci = get_abs_ci(wts, ['abs_fpr', 'abs_tpr'])
  eo = np.mean([wts.iloc[101*i + int(np.mean(eo_ci)*100)]['abs_fpr'] for i in range(10)])
  eo += np.mean([wts.iloc[101*i + int(np.mean(eo_ci)*100)]['abs_tpr'] for i in range(10)])
  plt.plot(np.mean(eo_ci), eo, 'd', color='darkcyan')
  sns.lineplot(x=eo_ci, y=eo, color='darkcyan', linestyle='dashed')

  plt.rcParams["axes.labelsize"] = 13
  plt.rcParams["axes.titlesize"] = 17
  ax.set(xlabel="adjustment weight $\\lambda$")
  ax.set(ylabel="Area between curves (absolute)")
  if title is None:
    ax.set(title="Absolute area between curves over $\\lambda$")
  else:
    ax.set(title=title)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--savedir", default="results", help="where to save results")
  args = parser.parse_args()

  for dataset in datasets:
      print(" ========= ", dataset, " ========")
      for alg in algos: 
          print("   -- ", alg)
          lambdadf = pd.read_csv(args.savedir + '/' + dataset + '_' + alg + '__lambdas.csv')
          
          # plot the paired curves for regular metrics
          for metric in metric_to_lambda:
              for correction in metric_to_lambda[metric]:
                  filename = args.savedir + "_plots/" + dataset + "/" + dataset + '_' + alg + '_' + metric + "_lmbd=" + correction
                  resultdf = pd.read_csv(args.savedir + '/' + dataset + "_" + alg + "__evalthresholds.csv")

                  repaired = 'Repaired' if correction != 'orig' else 'Unrepaired'
                  plot_title = metric_to_title[metric] + ", " + repaired

                  metrics = [metric + '_A', metric + '_B']

                  # handle legends: remove for all except pos rates uncorrected, and change to actual demographic
                  if (correction == 'orig') and (metric == 'selection'):
                    if dataset == 'adult_old':
                      legend_map = { metric + '_A': 'Male', 
                                    metric + '_B': 'Female'}
                    elif dataset == 'adult_new':
                      legend_map = { metric + '_A': 'White', 
                                    metric + '_B': 'Non-White'}
                    elif dataset == 'public':
                      legend_map = { metric + '_A': 'White', 
                                    metric + '_B': 'Non-White'}  
                    elif dataset == 'taiwan':
                      legend_map = { metric + '_A': 'Higher Education', 
                                    metric + '_B': 'Lower Education'}                      
                    else:
                      legend_map = { metric + '_A': 'Group A', 
                                  metric + '_B': 'Group B'}
                  else:
                    legend_map = {}

                  _plot_result(resultdf, 
                              metrics=metrics, 
                              filters={'lambda': correction}, 
                              title=plot_title, 
                              size=35,
                              legend_map=legend_map,
                              ylabel=metric_to_gamma[metric],
                              filename=filename + ".pdf")
          
                  # plot ROC
                  if dataset == 'adult_old':
                    ab_labels = ['Male', 'Female']
                  elif dataset == 'adult_new':
                    ab_labels = ['White', 'Non-White']
                  roc_plot_title = 'ROC per Group, ' + repaired
                  roc_plot_filename = filename + "_ROC.pdf"
                  _plot_ROC_curve(resultdf, 
                                  filters={'lambda': correction},
                                  ab_labels = ab_labels,
                                  title=roc_plot_title,
                                  size=35,
                                  filename=roc_plot_filename
                                  )

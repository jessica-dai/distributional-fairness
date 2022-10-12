# distributional-fairness

## todos

currently we have two EO solvers that compute but give quite different answers. which one is right? 
here's what we'll do: keep both of them for now, then just do the eval script for each, and keep whichever one performs better. yikes science but whatever right?! 


## notes on what's in here: 

### run scripts
- `run.py`: runs exp for binary sens attrs for any binary dataset in `data`

### helper scripts 
- `datasets.py`: get predicted probabilities (for binary sens attr datasets)
- `eval_exp.py`: mainly `get_eval_single`, which gives results for all metrics over all thresholds at a single lambda. also has `get_prob_lambda`, which is probabilistic estimation of (binary) lambda. 
- `plot.py`: source code to generate plots

### algorithm source (`src` folder)
- `bcmap.py`: calculate adjustment to the barycenter (multi-group)
- `bin_postprocess.py`: calculate adjustment to the barycenter (binary-group)
- `exact_solver.py`: exact calculation for lambda (binary-group). only used in lambda plotting notebook

### plotting notebooks
- `plotting_single_adjust.ipynb`: plot overall results for binary-group scenario (fig 2)

### multi stuff
- `run_multi.py`: runs fico exp and finds lambdas. (fico features are hardcoded)
- `plotting_nb.ipynb`: plot overall results for multi-group scenario (nb version of fig 1) (not used as of may '22)
- `lexi.py`: lexi & maxmin implementation to optimize lambdas (multi-group)


### old plotting notebooks
- `plotting_lambdas.ipynb`: illustrate existence of a minimum over lambdas and lambda comparisons (not used as of may '22)
- `plotting_single.ipynb`: plot metrics over thresholds for a single lambda (good for preliminaries/gut checking otherwise not used in paper)






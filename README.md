# distributional-fairness

Modification for new branch: 

Instead of just sampling `ADULT_SIZE` from the full dataset randomly (with seed=trial), instead we divide the full dataset into 6, use the 6th to calculate lambdas, then apply them to trials 1-5

For old branch: 

To reproduce figures and experiments:
1. run `./run_initial.sh` (or for a single dataset/algorithm, `python run_initial.py` with requisite cli args). This calculates the barycenter adjustments and saves the resulting `.csv`s.
2. run `./run_process.sh` (or for a single dataset/algorithm, `python run_process.py` with requisite cli args). This calculates $\lambda$s and evaluates distributional parity for each, and saves the resulting `.csv`s.
3. run `plot.py` to generate plots illustrating distributional parity (figs 2 and 3). 

To reproduce baselines: 
1. in `fair_baselines` folder, run (e.g.) `feldman.py` to generate predicted probabilities with the corresponding fair baseline (+ any hp tuning) applied.
2. generate plots using `baselines_overthresholds.ipynb`. 

### helper scripts and files 
- `datasets.py`: get predicted probabilities (for binary sens attr datasets) of a baseline algorithm
- `eval_helpers.py`: mainly `get_eval_single`, which gives results for all metrics over all thresholds at a single lambda. also has `get_prob_lambda`, which is probabilistic estimation of (binary) lambda. 

### algorithm source (`src` folder)
- `bcmap.py`: calculate adjustment to the barycenter (multi-group)
- `bin_postprocess.py`: calculate adjustment to the barycenter (binary-group)
- `exact_solver.py`: exact calculation for lambda (binary-group). only used in lambda plotting notebook



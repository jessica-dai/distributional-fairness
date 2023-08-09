# distributional-fairness

To reproduce figures and experiments:
1. run `./run_initial.sh` (or for a single dataset/algorithm, `python run_initial.py` with requisite cli args). This calculates the barycenter adjustments and saves the resulting `.csv`s.
2. run `./run_process.sh` (or for a single dataset/algorithm, `python run_process.py` with requisite cli args). This calculates $\lambda$s and evaluates distributional parity for each, and saves the resulting `.csv`s.
3. run `plot.py` to generate plots illustrating distributional parity (figs 2 and 3). 

To reproduce baselines: 
1. in `fair_baselines` folder, run (e.g.) `feldman.py` to generate predicted probabilities with the corresponding fair baseline (+ any hp tuning) applied.
2. generate plots using `baselines_overthresholds.ipynb`. 

### helper scripts and files 
- `datasets.py`: get predicted probabilities (for binary sens attr datasets) of a baseline algorithm. 
  - To add a dataset, follow template in `get_new_adult()` to implement something similar (eg `get_<DATASET_NAME>()`). 
  - This method should have a flag for `interv`, which will check if we are running a baseline fairness algorithm or just ours. 
  - If `interv` is not `None`, then you should construct a `BinaryLabelDataset` (following AIF360 convention) and return it (see current L259). 
  - The rest should be pretty similar to what's already in `get_new_adult()`, just make sure splits and sensitive features are set properly. 
- `eval_helpers.py`: mainly `get_eval_single`, which gives results for all metrics over all thresholds at a single lambda. also has `get_prob_lambda`, which is probabilistic estimation of (binary) lambda. 

### algorithm source (`src` folder)
- `bcmap.py`: calculate adjustment to the barycenter (multi-group)
- `bin_postprocess.py`: calculate adjustment to the barycenter (binary-group)
- `exact_solver.py`: exact calculation for lambda (binary-group). only used in lambda plotting notebook



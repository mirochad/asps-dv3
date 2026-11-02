# DV3: exact and approximate Data Shapley

First milestone: exact Shapley for 12 training points with logistic regression.
This is a baseline, not the complete assignment benchmark. Exact Shapley and
TMC convergence are implemented. Stratified estimation and noise experiments
are next.

## Run

```bash
cd asps-dv3
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python exact_shapley.py --seed 0 --output results/exact_seed0
```

The starter dataset is scikit-learn's built-in Breast Cancer dataset (no download).
Splits are stratified: 60% candidate training pool, 20% validation, 20% test;
12 stratified points are selected from the training pool. Saved indices make
the split reproducible. Scaling is fitted once on these 12 input vectors and
held fixed across coalitions, defining a fixed-representation valuation game.
The test set is used only for a final accuracy measurement.

Utility is validation accuracy. All 4,096 coalition utilities are recorded.
The empty coalition has utility zero; a one-class coalition uses a constant
classifier. Other coalitions train logistic regression with fixed settings.
These conventions are part of the game and must stay identical across future
estimators. Coalition evaluations and actual model fits are counted separately.

Each run writes point values, coalition utilities, split indices and metrics
under the chosen output directory. Valuation wall time includes data loading,
preprocessing, coalition evaluation and exact aggregation, excluding the final
test model and file writing. Peak RSS is the Linux process high-water mark,
including imports, rather than incremental method-only memory.

## TMC-Shapley convergence

With the virtual environment activated (Python 3.12 on Linux/WSL):

```bash
python benchmark_tmc.py
```

This recomputes exact Shapley and runs 10, 25, 50, 100 and 200 permutations
with estimator seeds 0, 1, 2. The data split, representation and logistic model
seed stay fixed (`--split-seed 0`) across every method and estimator seed.
Both exact and TMC use `LogisticGame` from `valuation_game.py`.

Every permutation adds points in random order and averages their marginal
validation-accuracy contributions. By default the remaining marginals are set
to zero once three consecutive prefix utilities are within absolute accuracy
0.01 of the full-training utility. This is a configurable heuristic and can
introduce bias; accuracy is not monotone in coalition size. Use
`--tolerance 0` to disable truncation and run ordinary permutation Monte Carlo
as a control. The empty/full reference evaluations are included in cost counts.
No coalition cache or exact-utility lookup is used by TMC: its costs represent
actual retraining. Checkpoints are cumulative within one run, not independent
restarts. Wall time excludes checkpoint reporting and data preparation; exact
time includes data preparation and aggregation as documented above.

Outputs in `results/tmc/`:

- `exact/`: freshly computed exact reference and saved split indices.
- `config.json`: data and estimator seeds, budgets and truncation parameters.
- `convergence.csv`: RMSE, Spearman against exact, time, utility calls, actual
  fits, truncation count and process peak RSS at each checkpoint/seed.
- `values.csv`: every point's estimate and exact value at each checkpoint.
- `stability.csv`: pairwise Spearman between estimator seeds at each budget.
- `convergence.png`: error/rank versus budget and error versus wall time.
- `value_convergence.png`: estimated values versus budget for estimator seed 0.

Peak RSS in this combined runner is the process lifetime high-water mark,
including the exact reference and prior runs, not isolated per-method memory.
It must not be used as a fair method-memory comparison until subprocess-based
measurement is added. No noisy-label F1 is reported yet: these are clean-data
convergence experiments. Sampling error need not decrease at every checkpoint.

For an untruncated control with the same budgets:

```bash
python benchmark_tmc.py --tolerance 0 --output results/mc_control
```

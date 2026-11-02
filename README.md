# DV3: exact and approximate Data Shapley

First milestone: exact Shapley for 12 training points with logistic regression.
This is a baseline, not the complete assignment benchmark. TMC, stratified
estimation, noise experiments and convergence plots are next.

## Run

```bash
cd /workspace/asps-dv3
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

"""Small, reproducible exact Data Shapley baseline; no dataset download needed."""
import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def shapley_from_utilities(utilities, n):
    """Each mask encodes a coalition; use the standard Shapley weights."""
    values = np.zeros(n)
    for i in range(n):
        for mask in range(1 << n):
            if not mask & (1 << i):
                weight = 1 / (n * math.comb(n - 1, mask.bit_count()))
                values[i] += weight * (utilities[mask | (1 << i)] - utilities[mask])
    return values


def run(seed=0, output="results/exact_seed0"):
    started = time.perf_counter()
    data = load_breast_cancer()
    ids = np.arange(len(data.target))
    pool, test = train_test_split(ids, test_size=0.2, stratify=data.target, random_state=seed)
    pool, validation = train_test_split(pool, test_size=0.25, stratify=data.target[pool], random_state=seed)
    train, _ = train_test_split(pool, train_size=12, stratify=data.target[pool], random_state=seed)
    # Fit preprocessing once on the selected training inputs, never validation/test.
    # This defines a fixed representation shared by every coalition.
    scaler = StandardScaler().fit(data.data[train])
    x_train = scaler.transform(data.data[train])
    x_validation = scaler.transform(data.data[validation])
    y_train, y_validation = data.target[train], data.target[validation]
    utilities = np.zeros(1 << len(train))
    fits = 0
    for mask in range(1, len(utilities)):
        subset = [i for i in range(len(train)) if mask & (1 << i)]
        labels = y_train[subset]
        if len(np.unique(labels)) == 1:
            # A constant classifier is well-defined when logistic regression is not.
            utilities[mask] = np.mean(y_validation == labels[0])
        else:
            model = LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)
            model.fit(x_train[subset], labels)
            utilities[mask] = accuracy_score(y_validation, model.predict(x_validation))
            fits += 1
    # Empty coalition utility is explicitly zero; utility is validation accuracy.
    values = shapley_from_utilities(utilities, len(train))
    elapsed = time.perf_counter() - started
    final = LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed).fit(x_train, y_train)
    test_accuracy = accuracy_score(data.target[test], final.predict(scaler.transform(data.data[test])))
    metrics = dict(dataset="sklearn_breast_cancer", seed=seed, n_train=12,
                   n_validation=len(validation), n_test=len(test), coalition_evaluations=len(utilities),
                   valuation_model_fits=fits, final_test_model_fits=1, wall_seconds=elapsed,
                   peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   empty_utility=float(utilities[0]), full_utility=float(utilities[-1]),
                   efficiency_residual=float(values.sum() - (utilities[-1] - utilities[0])),
                   test_accuracy=float(test_accuracy))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(point=np.arange(12), dataset_index=train, label=y_train, value=values)).to_csv(destination / "values.csv", index=False)
    np.save(destination / "coalition_utilities.npy", utilities)
    (destination / "splits.json").write_text(json.dumps(dict(train=train.tolist(), validation=validation.tolist(), test=test.tolist()), indent=2))
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/exact_seed0")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.output), indent=2))

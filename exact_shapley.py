"""Small, reproducible exact Data Shapley baseline; no dataset download needed."""
import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from valuation_game import LogisticGame


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
    game = LogisticGame(seed)
    train, validation, test = game.train, game.validation, game.test
    y_train = game.y
    utilities = np.array([game(mask) for mask in range(1 << game.n)])
    fits = game.fits
    # Empty coalition utility is explicitly zero; utility is validation accuracy.
    values = shapley_from_utilities(utilities, len(train))
    elapsed = time.perf_counter() - started
    final = game.model().fit(game.x, y_train)
    test_accuracy = float(np.mean(game.data.target[test] == final.predict(game.scaler.transform(game.data.data[test]))))
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
